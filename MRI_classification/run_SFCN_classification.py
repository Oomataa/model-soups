# =====================================================
# FILE: run_SFCN_classification_fast.py
# Trains CNN for AD vs CN classification from 2D NIfTI images
# Based on PMC7872776 architecture: 5 conv layers + maxpool
# Bootstrap + Early stopping (patience=5) + hyperparam search
# =====================================================

import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset, RandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nibabel as nib

# ---------------- USER CONFIG ----------------
CSV_2D_PATH = '/home/sara.early/MIPLAB/ENEL645/AD_classification/ADNI_selected_2D.csv'

BASE_OUTPUT_DIR = "/home/sara.early/MIPLAB/ENEL645/AD_classification/Results"
RESULTS_ORIG_DIR = os.path.join(BASE_OUTPUT_DIR, "Results_orig_0.3_new")
RESULTS_BOOT_DIR = os.path.join(BASE_OUTPUT_DIR, "Results_bootstrap")

SEED = 42
EPOCHS = 150  # Paper used 150 epochs
BATCH_SIZE = 64  # Paper used batch size 64
NUM_WORKERS = 0

EARLY_STOPPING_PATIENCE = 10  # Increased patience for 150 epochs
BOOTSTRAP = False 

LR_GRID = [1e-3, 1e-4]
WD_GRID = [0, 1e-4, 1e-3]
NUM_MODELS = 30
SEED_BASE = SEED


# ----------------- MODEL (Based on PMC7872776) -----------------------
class AD_CNN(torch.nn.Module):
    """
    5-layer CNN architecture based on PMC7872776
    Each conv layer followed by ReLU and MaxPool
    """
    def __init__(self, num_classes=2):
        super().__init__()
        
        # 5 Convolutional blocks (as per paper)
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.drop_conv = torch.nn.Dropout2d(0.3)
        self.pool1 = torch.nn.MaxPool2d(2)
        
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = torch.nn.ReLU()
        self.drop_conv = torch.nn.Dropout2d(0.3)
        self.pool2 = torch.nn.MaxPool2d(2)
        
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = torch.nn.ReLU()
        self.drop_conv = torch.nn.Dropout2d(0.3)
        self.pool3 = torch.nn.MaxPool2d(2)
        
        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu4 = torch.nn.ReLU()
        self.drop_conv = torch.nn.Dropout2d(0.3)
        self.pool4 = torch.nn.MaxPool2d(2)
        
        self.conv5 = torch.nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu5 = torch.nn.ReLU()
        self.drop_conv = torch.nn.Dropout2d(0.3)
        self.pool5 = torch.nn.MaxPool2d(2)
        
        # Global average pooling
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(512, 256)
        self.relu_fc = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(256, num_classes)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # Conv blocks
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.pool4(self.relu4(self.conv4(x)))
        x = self.pool5(self.relu5(self.conv5(x)))
        
        # Global pooling and classifier
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.relu_fc(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# ----------------- DATASET -------------------
class NiftiDataset(Dataset):
    """Loads 2D NIfTI images from file paths with labels."""
    def __init__(self, filepaths, labels):
        self.paths = filepaths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = nib.load(self.paths[idx]).get_fdata()
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        label = int(self.labels[idx])
        return {'image': img_tensor, 'label': label}


# ----------------- DATA SPLITS ------------------
def get_datasets(csv_path, seed):
    df = pd.read_csv(csv_path)
    
    old_prefix = "/home/saraearly/OneDrive/Documents/UCalgary/MIPLAB/"
    new_prefix = "/home/sara.early/MIPLAB/main/"
    df['filepath'] = df['filepath'].str.replace(old_prefix, new_prefix, regex=False)
    
    if 'disease' not in df.columns:
        raise ValueError("CSV must contain a 'disease' column with values 'CN' and 'AD'.")
    df['label'] = df['disease'].map({'CN': 0, 'AD': 1})

    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=seed, stratify=df['label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=seed, stratify=temp_df['label']
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ----------------- BUILD LOADERS ------------------
def make_loader_from_df(df, batch_size, shuffle=False):
    dataset = NiftiDataset(df['filepath'].tolist(), df['label'].tolist())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        pin_memory=True, num_workers=NUM_WORKERS,
                        persistent_workers=(NUM_WORKERS>0))
    return loader, dataset


def prepare_train_loader_fast(train_df, batch_size, bootstrap, ratio, seed):
    dataset = NiftiDataset(train_df['filepath'].tolist(), train_df['label'].tolist())
    n = len(dataset)

    if not bootstrap:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            pin_memory=True, num_workers=NUM_WORKERS,
                            persistent_workers=(NUM_WORKERS>0))
        return loader, n

    subset_size = max(1, int(n * ratio))
    rng = np.random.RandomState(seed)
    subset_indices = rng.choice(n, size=subset_size, replace=False).tolist()

    sampler = RandomSampler(data_source=Subset(dataset, subset_indices),
                            replacement=True, num_samples=n)
    loader = DataLoader(Subset(dataset, subset_indices), batch_size=batch_size, sampler=sampler,
                        pin_memory=True, num_workers=NUM_WORKERS,
                        persistent_workers=(NUM_WORKERS>0))
    return loader, n


# ----------------- TRAIN + EVAL ------------------
def train_and_eval(model, train_loader, val_loader, device, lr, wd, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_preds, train_gts = [], []

        for batch in train_loader:
            imgs = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_gts.extend(labels.cpu().numpy())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device)
                labels = batch['label'].to(device)
                outputs = model(imgs)
                val_losses.append(criterion(outputs, labels).item())

        avg_val_loss = float(np.mean(val_losses))

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if epoch >= 50 and patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"[EARLY STOP] at epoch {epoch+1}")
                break
    return model

# ----------------- EVALUATE ------------------
def evaluate_on_test(model, test_loader, device):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch['image'].to(device)
            labels = batch['label'].to(device)
            outputs = model(imgs)
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            gts.extend(labels.cpu().numpy())
    return accuracy_score(gts, preds) if gts else None


# ----------------- MAIN ------------------
def main():
    os.makedirs(RESULTS_ORIG_DIR, exist_ok=True)
    os.makedirs(RESULTS_BOOT_DIR, exist_ok=True)
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    # 1) Split datasets
    train_df, val_df, test_df = get_datasets(CSV_2D_PATH, SEED_BASE)
    print(f"[INFO] Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Save holdout CSV
    test_df.to_csv(os.path.join(BASE_OUTPUT_DIR, "holdout_test_fast.csv"), index=False)

    # 2) Build val/test loaders once
    val_loader, _ = make_loader_from_df(val_df, BATCH_SIZE)
    test_loader, _ = make_loader_from_df(test_df, BATCH_SIZE)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")
    
    results_summary = []
    
    for run_idx in range(NUM_MODELS):
        print(f"\n{'='*60}")
        print(f"RUN {run_idx+1}/{NUM_MODELS}")
        print(f"{'='*60}")
        
        seed = SEED_BASE + run_idx
        torch.manual_seed(seed)
        np.random.seed(seed)

        best_val_acc = 0.0
        best_model_state = None
        best_lr, best_wd = None, None

        for lr in LR_GRID:
            for wd in WD_GRID:
                print(f"\n[Hyperparameter] Testing LR={lr}, WD={wd}")
                train_loader, _ = prepare_train_loader_fast(
                    train_df, BATCH_SIZE, BOOTSTRAP, 0.7, seed
                )

                model = AD_CNN().to(device)

                # Train model -- early stopping on VAL LOSS
                trained_model = train_and_eval(
                    model, train_loader, val_loader, device, lr, wd, EPOCHS
                )

                # --- Validation-based model selection ---
                val_acc = evaluate_on_test(trained_model, val_loader, device)
                print(f"[VAL EVAL] Val Acc={val_acc:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = trained_model.state_dict()
                    best_lr, best_wd = lr, wd
        # Save the best model for this run
        folder = RESULTS_BOOT_DIR if BOOTSTRAP else RESULTS_ORIG_DIR
        fname = f"model_run{run_idx}_best_test_lr{best_lr}_wd{best_wd}_seed{seed}.pt"
        torch.save(best_model_state, os.path.join(folder, fname))
    
        # Save summary info
        results_summary.append({
            'run': run_idx,
            'seed': seed,
            'lr': best_lr,
            'wd': best_wd,
            'best_val_acc': float(best_val_acc),
            'bootstrap': BOOTSTRAP
        })
    
    # Save summary of all runs
    with open(os.path.join(BASE_OUTPUT_DIR, 'results_runs_boot_test.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n[MAIN] Done. All runs saved with best test accuracy.")


if __name__ == '__main__':
    main()