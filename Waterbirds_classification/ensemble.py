"""Main file for model distillation experiments."""

# Ignores nuisance warnings. Must be called first.
from distutils.util import strtobool 
from milkshake.datamodules.disagreement import Disagreement 
from milkshake.datamodules.Idxloader import IndexedDataModule
from milkshake.imports import valid_debiase_mechanisms, valid_models_and_datamodules
from milkshake.utils import compute_accuracy, ignore_warnings
from utils import save_results
import copy

ignore_warnings()

# Imports Python packages.
from configargparse import Parser

# Imports PyTorch packages.
from pytorch_lightning import Trainer
import torch
from torch import nn
import torch.nn.functional as F
import os
import os.path as osp
import pickle
import numpy as np
from torch.autograd import Variable

# Imports milkshake packages.
from milkshake.args import add_input_args
from milkshake.main import main
from milkshake.models.cnn import CNN
from milkshake.models.resnet import ResNet
from milkshake.models.bert import BERT
from milkshake.models.convnextv2 import ConvNeXtV2
from milkshake.models.vit import ViT
from milkshake.utils import get_weights
from milkshake.datamodules.celeba import CelebA
from milkshake.datamodules.civilcomments import CivilComments
from milkshake.datamodules.multinli import MultiNLI
from milkshake.datamodules.waterbirds import Waterbirds


class BaseNet(nn.Module):
    """Class for a CNN which learns via distillation from the logits of a teacher ResNet."""

    def __init__(self, args):
        super().__init__(args)

        copy_args = copy.deepcopy(args)

        def load_models(ensemble_data):
            # Loads teacher ResNet and freezes parameters.
            teachers = []

            for t in ensemble_data:
                individual_model = t["individual_model"] 
                if individual_model == "resnet": 
                    teacher = ResNet(args)
                elif individual_model == "bert": 
                    teacher = BERT(args)
                elif individual_model == "convnextv2": 
                    teacher = ConvNeXtV2(args)
                elif individual_model == "vit": 
                    teacher = ViT(args)

                state_dict = torch.load(t["weights"], map_location="cpu", weights_only=False)["state_dict"]
                teacher.load_state_dict(state_dict, strict=False)
                teacher.to("cuda").eval()
                for p in teacher.parameters():
                    p.requires_grad = False
                teachers.append(teacher)
            return teachers

        self.ensemble = load_models(args.ensemble_data)
        self.biased_teachers = None
        args = copy_args
        self.alpha = args.alpha
        # args.resnet_version = resnet_version
        # args.bert_version = "distill" 

    def load_msg(self):
        return (
            super().load_msg()[:-1]
            + f" with distillation from {self.hparams.ensemble_size} {self.hparams.individual_model} teachers."
        )
 
    def step_and_log_metrics(self, batch, idx, dataloader_idx, stage):
        """Performs a step, then computes and logs metrics.

        Args:
            batch: A tuple containing the inputs and targets as torch.Tensor.
            idx: The index of the given batch.
            dataloader_idx: The index of the current dataloader.
            stage: "train", "val", or "test".

        Returns:
            A dictionary containing the loss, prediction probabilities, targets, and metrics.
        """

        if stage == "train":
            result = self.step_2(batch, idx)
        else:
            return super().step_and_log_metrics(batch, idx, dataloader_idx, stage)

        accs = compute_accuracy(
            result["probs"],
            result["targets"],
            self.hparams.num_classes,
            self.hparams.num_groups,
        )

        self.milkshake_logger.add_metrics_to_result(result, accs, dataloader_idx)

        self.log_metrics(result, stage, dataloader_idx)

        return result

    def step_2(self, batch, idx):
        """Performs a single step of prediction and loss calculation.

        For distillation, we compute the MSE loss between the student and teacher logits.

        Args:
            batch: A tuple containing the inputs and targets as torch.Tensor.
            idx: The index of the given batch.

        Returns:
            A dictionary containing the loss, prediction probabilities, and targets.
        """

        inputs, orig_targets = batch
        # idx = idx.cpu().detach()
        # Removes extra targets (e.g., group index used for metrics).
        groups = None
        if orig_targets[0].ndim > 0:
            targets = orig_targets[:, 0]
            groups = orig_targets[:, 1]
        else:
            targets = orig_targets

        student_logits = self(inputs)

        student_loss = F.cross_entropy(student_logits, targets, reduction="none")
        probs = F.softmax(student_logits, dim=1)
        weight = None

        with torch.no_grad():
            teachers_output_logits = [
                teacher.to(inputs.device)(inputs) for teacher in self.ensemble
            ]
            # teachers_output_logits = torch.stack(teachers_output_logits)

        if self.hparams.ensemble_method == "OURS":
            with torch.no_grad():
                biased_teacher_logits = [
                    teacher.to(inputs.device)(inputs)
                    for teacher in self.biased_teachers
                ]
                biased_teacher_logits = torch.stack(biased_teacher_logits).mean(0)

            teacher_loss, weight = self.kl_fair_loss(
                student_logits, teachers_output_logits, biased_teacher_logits, targets
            )

            # print(">>>>>", weight.shape)
            # Computes CE loss between student and teachers' logits.
            # self.sample_weight.update(weight.cpu().detach(), idx ,targets)

            # weight = self.sample_weight.parameter[idx].to("cuda")

            # label_cpu = targets.cpu()
            # for c in range(self.hparams.num_classes):
            #    class_index = np.where(label_cpu == c)[0]
            #    max_loss = self.sample_weight.max_loss(c)
            #    weight[class_index] /= max_loss

            student_loss = student_loss * weight
        else:
            raise ValueError(
                f"Ensemble method {self.hparams.ensemble_method} is not implemented"
            )

        loss = self.alpha * teacher_loss + (1 - self.alpha) * student_loss.mean()

        return {
            "loss": loss,
            "probs": probs,
            "targets": orig_targets,
            "weights": weight,
        }
 

class ModelSoup(BaseNet):
    def __init__(self, args):
        super().__init__(args)
        # self.ensemble = self.ensemble

        # Average the weights of all teacher models into a single model
        with torch.no_grad():
            # Initialize the student model with the first teacher's architecture
            self.model = copy.deepcopy(self.ensemble[0])

            # Get state dictionaries for all teachers
            state_dicts = [model.state_dict() for model in self.ensemble]

            # Average the weights across all teachers
            for key in self.model.state_dict():
                # Stack the same parameter from all teachers
                stacked_params = torch.stack(
                    [
                        dict[key].to(self.model.device)
                        for dict in state_dicts
                    ]
                )
                # Convert to float before computing mean if it's a Long tensor
                if stacked_params.dtype == torch.long:
                    stacked_params = (
                        stacked_params.float().mean(dim=0).to(dtype=torch.long)
                    )
                else:
                    stacked_params = stacked_params.mean(dim=0)
                # Compute the average
                self.model.state_dict()[key].copy_(stacked_params)

    def load_msg(self):
        return super().load_msg()[:-1] + f" Load Model Weight-space-averaging (Model Soup))"


class ENS(BaseNet):
    def __init__(self, args):
        super().__init__(args)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Performs a single prediction step.

        Args:
            batch: A tuple containing the inputs and targets as torch.Tensor.
            idx: The index of the given batch.
            dataloader_idx: The index of the current dataloader.
        """
        inputs, orig_targets = batch
        teacher_probs = [F.softmax(teacher(inputs), dim=1) for teacher in self.ensemble]

        teacher_probs = [torch.argmax(probs, dim=1) for probs in teacher_probs]

        majority_vote, _ = torch.mode(teacher_probs, dim=0)
        return majority_vote

    def step_2(self, batch, idx):
        """Performs a single step of prediction and loss calculation.

        For distillation, we compute the MSE loss between the student and teacher logits.

        Args:
            batch: A tuple containing the inputs and targets as torch.Tensor.
            idx: The index of the given batch.

        Returns:
            A dictionary containing the loss, prediction probabilities, and targets.
        """

        inputs, orig_targets = batch
        # idx = idx.cpu().detach()
        # Removes extra targets (e.g., group index used for metrics).
        groups = None
        if orig_targets[0].ndim > 0:
            targets = orig_targets[:, 0]
            groups = orig_targets[:, 1]
        else:
            targets = orig_targets

        # student_logits = self(inputs)

        # student_loss = F.cross_entropy(student_logits, targets, reduction="none")
        # probs = F.softmax(student_logits, dim=1)

        teacher_logits = [teacher(inputs) for teacher in self.ensemble]
        teacher_logits = torch.stack(teacher_logits)
        teacher_probs = F.softmax(teacher_logits, dim=2).mean(0)

        # majority_votes, _ = torch.mode(teacher_probs, dim=0)

        # teacher_probs = F.softmax(self.ensemble[0](inputs), dim=1)
        loss = 0  # self.alpha * teacher_loss + (1 - self.alpha) * student_loss.mean()
        return {
            "loss": loss,
            "probs": teacher_probs,
            "targets": orig_targets,
            "weights": None,
        }

    def step_and_log_metrics(self, batch, idx, dataloader_idx, stage):
        """Performs a step, then computes and logs metrics.

        Args:
            batch: A tuple containing the inputs and targets as torch.Tensor.
            idx: The index of the given batch.
            dataloader_idx: The index of the current dataloader.
            stage: "train", "val", or "test".

        Returns:
            A dictionary containing the loss, prediction probabilities, targets, and metrics.
        """

        result = self.step_2(batch, idx)
        accs = compute_accuracy(
            result["probs"],
            result["targets"],
            self.hparams.num_classes,
            self.hparams.num_groups,
        )

        self.milkshake_logger.add_metrics_to_result(result, accs, dataloader_idx)

        self.log_metrics(result, stage, dataloader_idx)

        return result


class StudentResnet(BaseNet, ResNet):
    def __init__(self, args):
        super().__init__(args)

  
class StudentENS(ENS, ResNet):
    def __init__(self, args):
        super().__init__(args)


class StudentENSViT(ENS, ViT):
    def __init__(self, args):
        super().__init__(args)

class StudentModelSoup(ModelSoup, ResNet):
    def __init__(self, args):
        super().__init__(args)

class StudentModelSoupConvNeXtV2(ModelSoup, ConvNeXtV2):
    def __init__(self, args):
        super().__init__(args)

class StudentModelSoupViT(ModelSoup, ViT):
    def __init__(self, args):
        super().__init__(args)

class StudentModelSoupBERT(ModelSoup, BERT):
    def __init__(self, args):
        super().__init__(args)


class StudentENSConvNeXtV2(ENS, ConvNeXtV2):
    def __init__(self, args):
        super().__init__(args)


class StudentBERT(BaseNet, BERT):
    def __init__(self, args):
        super().__init__(args)


class StudentENSBERT(ENS, BERT):
    def __init__(self, args):
        super().__init__(args)


def pct(x):
    return round(x, 3) * 100


class WaterbirdsDisagreement(Waterbirds, Disagreement):
    """DataModule for the WaterbirdsDisagreement dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)


class CelebADisagreement(CelebA, Disagreement):
    """DataModule for the CelebADisagreeement dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)


class CivilCommentsDisagreement(CivilComments, Disagreement):
    """DataModule for the CivilCommentsDisagreement dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)


class MultiNLIDisagreement(MultiNLI, Disagreement):
    """DataModule for the MultiNLIDisagreement dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)


def set_training_parameters(args):
    if args.datamodule == "waterbirds":
        args.datamodule_class = Waterbirds
        args.num_classes = 2
        # args.training_size = 200000
    elif args.datamodule == "celeba":
        args.datamodule_class = CelebA
        args.num_classes = 2
        # args.training_size = 200000
    elif args.datamodule == "civilcomments":
        args.datamodule_class = CivilComments
        args.num_classes = 2
    elif args.datamodule == "multinli":
        args.datamodule_class = MultiNLI
        args.num_classes = 3
    else:
        raise ValueError(f"DataModule {args.datamodule} not supported.")

    args.save_dir = "training_logs"


def print_metrics(metrics):
    val_metrics, test_metrics = metrics

    val_group_accs = [acc for name, acc in val_metrics[0].items() if "group" in name]
    val_avg_acc = val_metrics[0]["val_acc"]
    val_avg_acc = round(val_avg_acc * 100, 1)
    val_worst_group_acc = min(val_group_accs)
    val_worst_group_acc = round(val_worst_group_acc * 100, 1)
    print(f"Val Average Acc: {val_avg_acc}")
    print(f"Val Worst Group Acc: {val_worst_group_acc}")

    if test_metrics:
        test_group_accs = [
            acc for name, acc in test_metrics[0].items() if "group" in name
        ]
        test_avg_acc = test_metrics[0]["test_acc"]
        test_avg_acc = round(test_avg_acc * 100, 1)
        test_worst_group_acc = min(test_group_accs)
        test_worst_group_acc = round(test_worst_group_acc * 100, 1)
        print(f"Test Average Acc: {test_avg_acc}")
        print(f"Test Worst Group Acc: {test_worst_group_acc}")

    print()


def get_teachers_models_uniform(args, size, debiased=False):
    file = f"{args.log_path}/teacher_erm.pkl"
    if osp.isfile(file):
        with open(file, "rb") as f:
            erm = pickle.load(f)
    else:
        raise FileNotFoundError("Pretrained teachers models not founds")

    teachers = []
    seeds = range(1, 51)
    model_name = get_individual_model_name(args)
    teacher_type = "DEBIASED" if debiased else "BIASED"
    print(f"Loading {size} {teacher_type} {model_name} Teachers")
    for seed in seeds:
        t = erm[args.datamodule][seed][model_name][args.cb_erm_teachers][debiased][args.bootstrap]
        if t["version"] != -1:
            try:
                weights = get_weights(args, t["version"], idx=-1)
                #weights = get_teacher_weights(args, t["version"], debiased=debiased)
                teachers.append(
                    {"individual_model": args.individual_model, "weights": weights}
                )
                if len(teachers) > size:
                    print(f"Got all {teacher_type} teachers")
                    break
            except Exception as e:
                print(f"Exception missed + {t['version']} + {e}")
        else:
            print(f"Skept teacher {model_name}_{seed} version {t['version']} not found")
    np.random.shuffle(teachers)

    return teachers[0:size]



def _get_val_worst_from_entry(t):
    """Extract worst-group val accuracy from a teacher_erm entry."""
    if not t["metrics"]:
        return None
    val_metrics, _ = t["metrics"]
    val_group_accs = [
        acc for name, acc in val_metrics[0].items()
        if "group" in name
    ]
    if not val_group_accs:
        return None
    return float(min(val_group_accs))

def get_teachers_models_ordered(args, size, debiased=False):
    """
    Greedy-style ORDERED selection:
    sort all available teachers by val worst-group acc (descending),
    then take the top-K and do uniform soup over them.
    """
    file = f"{args.log_path}/teacher_erm.pkl"
    if osp.isfile(file):
        with open(file, "rb") as f:
            erm = pickle.load(f)
    else:
        raise FileNotFoundError("Pretrained teachers models not found")

    candidates = []
    seeds = range(1, 51)
    model_name = get_individual_model_name(args)
    teacher_type = "DEBIASED" if debiased else "BIASED"
    print(f"Loading {size} {teacher_type} {model_name} Teachers (ORDERED)")

    for seed in seeds:
        t = erm[args.datamodule][seed][model_name][args.cb_erm_teachers][debiased][args.bootstrap]
        if t["version"] == -1 or not t["metrics"]:
            print(f"Skip teacher {model_name}_{seed} version {t['version']}")
            continue

        val_worst = _get_val_worst_from_entry(t)
        if val_worst is None:
            print(f"Could not compute val_worst for seed={seed}, skip.")
            continue

        candidates.append((seed, t["version"], val_worst))

    # sort by worst-group val accuracy (higher is better)
    candidates.sort(key=lambda x: x[2], reverse=True)

    teachers = []
    for seed, version, val_worst in candidates[:size]:
        try:
            weights = get_weights(args, version, idx=-1)
            teachers.append(
                {"individual_model": args.individual_model, "weights": weights}
            )
            print(f" Using seed={seed}, version={version}, val_worst={val_worst:.4f}")
        except Exception as e:
            print(f"Failed to load seed={seed}, version={version}: {e}")

    assert len(teachers) > 0, f"teachers must not be empty {len(teachers)}"
    return teachers





def get_teacher_weights(args, version, debiased=False):
    teacher_model_name = get_individual_model_name(args)
    """Returns weights path given model version and checkpoint index.
    
    Args:
        ckpt_path: The root directory for checkpoints.
        version: The model's PyTorch Lightning version.
        best: Whether to return the best model checkpoint.
        idx: The model's checkpoint index (-1 selects the latest checkpoint).
    
    Returns:
        The filepath of the desired model weights.
    """

    # Selects the right naming convention for PL versions based on
    # whether the version input is an int or a string.
    
    if debiased:
        ckpt_path = f"{args.log_path}/training_logs/debiased_teachers/checkpoints/{teacher_model_name}/{args.datamodule}" 
    else:
        ckpt_path = f"{args.log_path}/training_logs/biased_teachers/checkpoints/{teacher_model_name}/{args.datamodule}" 
    
    ckpt_path = osp.join(ckpt_path, f"version_{version}/checkpoint.cktp")

    return osp.join(os.getcwd(), ckpt_path) 
 

def get_model_name(args):
    model_name = args.model
    if args.model == "bert":
        model_name += f"_{args.bert_version}"
    elif args.model == "resnet":
        model_name += f"_{args.resnet_version}"
    elif args.model == "convnextv2":
        model_name += f"_{args.convnextv2_version}"
    elif args.model == "vit":
        model_name += f"_{args.vit_version}"
    return model_name


def get_individual_model_name(args):
    model_name = args.individual_model
    model_version = {}
    if args.individual_model == "bert":
        ver = getattr(args, "individual_bert_version", "large")
        model_name += f"_{ver}"
        model_version["bert_version"] = ver
    elif args.individual_model == "resnet":
        model_name += f"_{args.individual_resnet_version}"
        model_version["resnet_version"] = args.individual_resnet_version
    elif args.individual_model == "convnextv2":
        model_name += f"_base"
        model_version["convnextv2_version"] = "base"
    elif args.individual_model == "vit":
        model_name += f"_large"
        model_version["vit_version"] = "large"
    return model_name


def experiment(args):
    # cifar10 = CIFAR10(args)
    # models, datamodules = valid_models_and_datamodules()
    # Trains a teacher ResNet.
    # Adds experiment-specific parameters to args.
    set_training_parameters(args)


    # Select the appropriate model class based on ensemble method and model type
    if args.ensemble_method == "ModelSoup":

        model_mapping = {
            "bert": StudentModelSoupBERT,
            "resnet": StudentModelSoup,
            "convnextv2": StudentModelSoupConvNeXtV2,
            "vit": StudentModelSoupViT,
            "default": StudentModelSoup,
        }
        model_class = model_mapping.get(args.model, model_mapping["default"])
        args.eval_only = True 
    elif args.ensemble_method == "ENS":
        # args.individual_model = args.model
        model_mapping = {
            "bert": StudentRandomBERT,
            "resnet": StudentENS,
            "default": StudentENS,
            "convnextv2": StudentENSConvNeXtV2,
            "vit": StudentENSViT,
        }
        model_class = model_mapping.get(
            args.model, model_mapping["default"]
        )  # StudentENSBERT if args.model == "bert" else StudentENS
        args.eval_only = True

    else:
        raise NotImplementedError

    individual_model = get_individual_model_name(args)
    student_model = get_model_name(args)

    datasets_models = {
        "waterbirds": ["resnet", "vit", "convnextv2"],
        "celeba": ["resnet", "vit", "convnextv2"],
        "civilcomments": ["bert"],
    }

    # Choose selection strategy
    if args.soup_selection == "uniform":
        select_teachers = get_teachers_models_uniform
    else:  # "ordered"
        select_teachers = get_teachers_models_ordered

    if args.individual_model == "mix":
        t_model = args.individual_model
        ensemble_data = []
        mix_models = datasets_models[args.datamodule]
        per_model_size = round(args.ensemble_size / len(mix_models))
        for model_class_arch in mix_models:
            args.individual_model = model_class_arch
            ensemble_data += select_teachers(
                args, size=per_model_size, debiased=not args.erm_teachers
            )
        args.individual_model = t_model
        args.ensemble_data = ensemble_data

    elif args.finetuned_teachers_size is not None:
        # Mix debiased and biased teachers
        ensemble_data = []
        ensemble_data = select_teachers(
            args, size=args.ensemble_size, debiased=False
        )
        ensemble_data += select_teachers(
            args,
            size=args.ensemble_size - args.finetuned_teachers_size,
            debiased=True,
        )
        args.ensemble_data = ensemble_data

    else:
        # Pure biased or pure debiased teachers
        args.ensemble_data = select_teachers(
            args, size=args.ensemble_size, debiased=not args.erm_teachers
        )

    assert (
        len(args.ensemble_data) > 0
    ), f"teachers must not be empty {len(args.ensemble_data)}"
    ## Get biased teachers only for orchestrating the distillation process


    print(len(args.ensemble_data))

    # Trains a BaseNet with distillation from the teachers ensemble.
    _, val_metrics, test_metrics = main(args, model_class, args.datamodule_class)

    metrics = [val_metrics, test_metrics]

    print_metrics(metrics) 
    
    result_dir = ""
    if args.exp_name is not None:
        result_dir = f"Ablation/{args.exp_name}/"

    if args.erm_teachers:
        result_dir += f"StudentERMteacher/"
    else:
        result_dir += f"Student/"
        result_dir += f"{args.debiased_teachers_method}/"

    ensemble_method = f"ensemble_{args.ensemble_method}"

    result_dir += (
        f"{args.datamodule}/"
        f"{get_model_name(args)}/"
        f"t_size_{args.ensemble_size}/"
        f"{ensemble_method}"
        f"seed_{args.seed}"
    )

    if args.bootstrap:
        result_dir += "/bootstrap"
    else:
        result_dir += "/nonbootstrap"

    save_results(metrics, seed=args.seed, result_dir=result_dir)




if __name__ == "__main__":
    parser = Parser(
        args_for_setting_config_path=["-c", "--cfg", "--config"],
        config_arg_is_required=True,
    )

    debiased_mechanisms = ["group-balanced retraining"]

    # valid_debiase_mechanisms()

    parser = add_input_args(parser)
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(strategy="ddp_find_unused_parameters_false")
    parser.set_defaults(resnet_version=18)

    parser.add("--temperature", default=4, type=float)
    parser.add("--individual_model", default="resnet", type=str)
    parser.add("--teacher_version", default=50)
    parser.add("--exp_name", default=None, type=str)
    parser.add("--individual_resnet_version", default=50, type=int)
    parser.add(
        "--individual_bert_version",
        default="base",
        choices=["tiny","mini","small","medium","base","large"],
        help="Version of BERT used by the *teacher* models in the ensemble",
    )
    parser.add("--beta", default=0.02, type=float)
    parser.add("--alpha", default=0.5, type=float)
    parser.add("--lamda", default=0.8, type=float)
    parser.add(
        "--finetuned_teachers_size",
        default=None,
        type=int,
        help="The number of finetuned teachers (debiased) in the ensemble",
    )
    parser.add("--ensemble_size", default=5, type=int)

    parser.add(
        "--erm_teachers",
        default=False,
        type=lambda x: bool(strtobool(x)),
        help="Whether to use teacher trained with ERM only, i.e., biased teachers",
    )

    parser.add(
        "--cb_erm_teachers",
        default=False,
        type=lambda x: bool(strtobool(x)),
        help="Whether to use class-balanced pretrained teachers",
    )

    parser.add(
        "--no_distillation",
        default=False,
        type=lambda x: bool(strtobool(x)),
        help="Whether to use train the student model without distillation",
    )

    parser.add(
        "--debiased_teachers_method",
        default=debiased_mechanisms[0],
        type=str,
        choices=debiased_mechanisms,
    )

    parser.add(
        "--ensemble_method",
        default="ModelSoup",
        type=str,
        choices=[
            "ModelSoup",
            "RANDOM",
            "ENS",
        ],
    )

    parser.add(
        "--norm_grads",
        default=True,
        type=lambda x: bool(strtobool(x)),
        help="Whether to normalize the gradients before computing the dot products.",
    )

    ## CNN params

    parser.add(
        "--cnn_batchnorm",
        default=True,
        type=lambda x: bool(strtobool(x)),
        help="Whether to use batch normalization in the CNN.",
    )
    parser.add(
        "--cnn_initial_width",
        default=32,
        type=int,
        help="The number of filters in the first layer of the CNN.",
    )
    parser.add(
        "--cnn_kernel_size",
        default=3,
        type=int,
        help="The size of the kernel in the CNN.",
    )
    parser.add(
        "--bootstrap",
        default=False,
        type=lambda x: bool(strtobool(x)),
        help="Use bootstrap-trained teachers if True, else non-bootstrap.",
    )
    
    parser.add("--bootstrap_frac", default=0.8, type=float)
    parser.add("--bootstrap_n_sets", default=5, type=int)
    parser.add("--bootstrap_with_replacement", default=False, type=lambda x: bool(strtobool(x)))

    parser.add(
        "--cnn_num_layers", default=5, type=int, help="The number of layers in the CNN."
    )
    parser.add(
        "--cnn_padding",
        default=0,
        type=int,
        help="The amount of input padding in the CNN.",
    )
    parser.add("--cnn_epochs", default=50, type=int)
    parser.add("--cnn_lr", default=0.1, type=float)

    parser.add("--log_path", default="/project/aip-ebrahimi/shared/model-soup-project", type=str)
    parser.add(
    "--soup_selection",
    default="uniform",
    type=str,
    choices=["uniform", "ordered"],
    help="How to pick teachers for ModelSoup: "
         "'uniform' = random subset, 'ordered' = top-K by val worst-group acc.",
    )


    
 
    args = parser.parse_args()
    torch.autograd.set_detect_anomaly(True)
    experiment(args)
