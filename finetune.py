import argparse
import json
import os
import time

import clip
import numpy as np
import torch
from tqdm import tqdm

from timm.data.transforms_factory import transforms_imagenet_train
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from datasets.imagenet import ImageNet98p, ImageNet
from utils import ModelWrapper, maybe_dictionarize_batch, cosine_lr
from zeroshot import zeroshot_classifier
from openai_imagenet_template import openai_imagenet_template


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('~/data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--model-location",
        type=str,
        default=os.path.expanduser('~/ssd/checkpoints/soups'),
        help="Where to download the models.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--custom-template", action="store_true", default=False,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--warmup-length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--model",
        default='ViT-B/32',
        help='Model to use -- you can try another like ViT-L/14'
    )
    parser.add_argument(
        "--name",
        default='finetune_cp',
        help='Filename for the checkpoints.'
    )
    parser.add_argument(
        "--model-id",
        type=int,
        default=0,
        help="Model ID (used for bookkeeping in sweeps).",
    )
    parser.add_argument(
        "--trial-id",
        type=int,
        default=0,
        help="Trial ID (used for bookkeeping in sweeps).",
    )
    parser.add_argument(
        "--timm-aug", action="store_true", default=False,
    )

    parser.add_argument(
        "--train-augmentation",
        choices=['clip', 'timm', 'minimal', 'randaug'],
        default='clip',
        help="Override for training augmentations."
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor for cross-entropy loss."
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.0,
        help="Mixup alpha parameter; 0 disables mixup."
    )
    parser.add_argument(
        "--randaug-m",
        type=float,
        default=15,
        help="RandAugment magnitude (used when --train-augmentation=randaug)."
    )
    parser.add_argument(
        "--randaug-n",
        type=int,
        default=2,
        help="RandAugment number of ops (used when --train-augmentation=randaug)."
    )

    parser.add_argument("--bootstrap", action="store_true", default=False,
                    help="Use bootstrap sampling (WITHOUT replacement) for the training loader.")
    parser.add_argument("--bootstrap-seed", type=int, default=0,
                    help="Seed that defines the bootstrap sample for this run.")
    parser.add_argument("--bootstrap-size-ratio", type=float, default=0.7,
                    help="Draws per epoch as a fraction of the 98%% train set size; 1.0 = classic bootstrap.")

    return parser.parse_args()


def get_state_dict(model):
    """Return a state dict regardless of DataParallel wrapping."""
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


if __name__ == '__main__':
    # Help avoid CUDA fragmentation-induced OOMs
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    args = parse_arguments()
    DEVICE = 'cuda'

    # --- Paths for checkpoints/metrics ---
    os.makedirs(args.model_location, exist_ok=True)
    run_stub = f"m{args.model_id}_t{args.trial_id}"
    best_path = os.path.join(args.model_location, "best.pt")
    last_path = os.path.join(args.model_location, "last.pt")
    metrics_path = os.path.join(args.model_location, "metrics.json")
    best_top1 = float('-inf')
    best_epoch = -1


    if args.custom_template:
        template = [lambda x : f"a photo of a {x}."]
    else:
        template = openai_imagenet_template

    base_model, preprocess = clip.load(args.model, 'cuda', jit=False)
    # 98p is the 98% of ImageNet train set that we train on -- the other 2% is hodl-out val.
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std = (0.26862954, 0.26130258, 0.27577711)

    if args.timm_aug or args.train_augmentation == 'timm':
        train_preprocess = transforms_imagenet_train(
            img_size=base_model.visual.input_resolution,
            mean=clip_mean,
            std=clip_std
        )
    elif args.train_augmentation == 'minimal':
        train_preprocess = transforms.Compose([
            transforms.RandomResizedCrop(
                base_model.visual.input_resolution,
                scale=(0.9, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=clip_mean, std=clip_std),
        ])
    elif args.train_augmentation == 'randaug':
        randaug_m = max(0, int(round(args.randaug_m)))
        randaug_n = max(1, int(round(args.randaug_n)))
        train_preprocess = transforms.Compose([
            transforms.RandomResizedCrop(
                base_model.visual.input_resolution,
                scale=(0.9, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=randaug_n, magnitude=randaug_m),
            transforms.ToTensor(),
            transforms.Normalize(mean=clip_mean, std=clip_std),
        ])
    else:
        train_preprocess = preprocess
    train_dset = ImageNet98p(train_preprocess, location=args.data_location, batch_size=args.batch_size, num_workers=args.workers)

    # ===== BEGIN: Bootstrap loader (WITH replacement) over the 98% split =====
    from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler, Subset
    import torch, math

    use_pin_memory = torch.cuda.is_available()
    if args.bootstrap:
        base_ds = train_dset.train_dataset   # the 98% ImageNet training subset
        n_total = len(base_ds)               # how many items in that 98% pool

        # Classic bootstrap: draw n_total samples WITH replacement each epoch.
        m = max(1, int(round(args.bootstrap_size_ratio * n_total)))

        # Reproducible sampler (change seed to change the sample)
        g = torch.Generator()
        g.manual_seed(args.bootstrap_seed)

        subset_indices = torch.randperm(n_total, generator=g)[:m].tolist()
        subset_ds = Subset(base_ds, subset_indices)

        bootstrap_sampler = RandomSampler(
            data_source=subset_ds,
            replacement=True,
            num_samples=n_total,
            generator=g
        )

        # New loader using the sampler (IMPORTANT: do not pass shuffle=True with a sampler)
        train_dset.train_loader = DataLoader(
            subset_ds,
            batch_size=args.batch_size,
            num_workers=args.workers,
            sampler=bootstrap_sampler,
            pin_memory=use_pin_memory,
        )


        print(f">> BOOTSTRAP ACTIVE: 70% subset ({m}/{n_total}) WITH replacement up to {n_total} draws")
    # ===== END: Bootstrap loader =====





    test_dset = ImageNet(preprocess, location=args.data_location, batch_size=args.batch_size, num_workers=args.workers)
    clf = zeroshot_classifier(base_model, train_dset.classnames, template, DEVICE)
    NUM_CLASSES = len(train_dset.classnames)
    feature_dim = base_model.visual.output_dim

    model = ModelWrapper(base_model, feature_dim, NUM_CLASSES, normalize=True, initial_weights=clf)
    for p in model.parameters():
        p.data = p.data.float()

    model = model.cuda()
    devices = [x for x in range(torch.cuda.device_count())]
    if len(devices) > 1:
        # Only use DataParallel when >1 GPU to avoid extra VRAM overhead on a single card
        model = torch.nn.DataParallel(model, device_ids=devices)

    model_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(model_parameters, lr=args.lr, weight_decay=args.wd)

    num_batches = len(train_dset.train_loader)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    def mixup_data(inputs, targets, alpha):
        if alpha <= 0.0:
            return inputs, targets, targets, 1.0
        lam = np.random.beta(alpha, alpha)
        batch_size = inputs.size(0)
        index = torch.randperm(batch_size, device=inputs.device)
        mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
        target_a, target_b = targets, targets[index]
        return mixed_inputs, target_a, target_b, lam

    #model_path = os.path.join(args.model_location, f'{args.name}_0.pt')
    #print('Saving model to', model_path)
    #torch.save(model.module.state_dict(), model_path)

    for epoch in range(args.epochs):
        # Train
        model.train()
        end = time.time()
        for i, batch in enumerate(train_dset.train_loader):
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()
            batch = maybe_dictionarize_batch(batch)
            inputs, labels = batch['images'].to(DEVICE), batch['labels'].to(DEVICE)
            data_time = time.time() - end

            if args.mixup_alpha > 0:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, args.mixup_alpha)
                logits = model(inputs)
                loss = lam * loss_fn(logits, targets_a) + (1 - lam) * loss_fn(logits, targets_b)
            else:
                logits = model(inputs)
                loss = loss_fn(logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            batch_time = time.time() - end
            end = time.time()

            if i % 20 == 0:
                percent_complete = 100.0 * i / len(train_dset.train_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(train_dset.train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )

        # #Evaluate
        test_loader = test_dset.test_loader
        model.eval()
        with torch.no_grad():
            print('*'*80)
            print('Starting eval')
            correct, count = 0.0, 0.0
            pbar = tqdm(test_loader)
            for batch in pbar:
                batch = maybe_dictionarize_batch(batch)
                inputs, labels = batch['images'].to(DEVICE), batch['labels'].to(DEVICE)

                logits = model(inputs)

                loss = loss_fn(logits, labels)

                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
                count += len(logits)
                pbar.set_description(
                    f"Val loss: {loss.item():.4f}   Acc: {100*correct/count:.2f}")
            top1 = correct / count



        epoch_idx = epoch + 1
        print(f'Val acc at epoch {epoch_idx}: {100*top1:.2f}')

        if top1 > best_top1:
            best_top1 = top1
            best_epoch = epoch_idx
            torch.save(get_state_dict(model), best_path)
            print(f"[INFO] New best top-1={100*best_top1:.2f} at epoch {best_epoch} -> {best_path}")

    # --- Save final checkpoint + metrics per trial ---
    torch.save(get_state_dict(model), last_path)
    print(f"[INFO] Saved FINAL checkpoint -> {last_path}")

    metrics = {
        "model_id": args.model_id,
        "trial_id": args.trial_id,
        "run_stub": run_stub,
        "epochs": args.epochs,
        "best_top1": best_top1,
        "best_epoch": best_epoch,
        "final_top1": top1,
        "hyperparams": {
            "lr": args.lr,
            "wd": args.wd,
            "mixup_alpha": args.mixup_alpha,
            "label_smoothing": args.label_smoothing,
            "train_augmentation": args.train_augmentation,
            "randaug_m": args.randaug_m,
            "randaug_n": args.randaug_n,
            "bootstrap": args.bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_size_ratio": args.bootstrap_size_ratio,
        },
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Wrote metrics -> {metrics_path}")



    #    if args.save_last_only and epoch_idx != args.epochs:
            # Skip saving intermediate epochs
    #        continue

    #    last_path = os.path.join(args.model_location, 'last.pt' if args.save_last_only else f'{args.name}_{epoch_idx}.pt')
    #    torch.save(model.module.state_dict(), last_path)
    #    print(f'[INFO] Saved checkpoint -> {last_path}')



       # print(f'Val acc at epoch {epoch}: {100*top1:.2f}')

       # model_path = os.path.join(args.model_location, f'{args.name}_{epoch + 1}.pt')
       # print('Saving model to', model_path)
       # torch.save(model.module.state_dict(), model_path)
