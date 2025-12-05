import os
import torch
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
# ---------------- USER CONFIG ----------------
MODEL_FOLDER = "/home/sara.early/MIPLAB/ENEL645/AD_classification/Results/Results_bootstrap_0.3_new"  
CSV_TEST_PATH = "/home/sara.early/MIPLAB/ENEL645/AD_classification/Results/holdout_test.csv"  
BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_JSON = "/home/sara.early/MIPLAB/ENEL645/AD_classification/Results/model_performance_bootstrap.json"

RUN_INDIVIDUAL = True   # evaluate each model separately
RUN_UNIFORM   = True    # run uniform soup
RUN_GREEDY    = True    # run greedy soup

# ---------------- MODEL ----------------
class SFCN(torch.nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
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

        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(512, 256)
        self.relu_fc = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(256, num_classes)

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
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.pool4(self.relu4(self.conv4(x)))
        x = self.pool5(self.relu5(self.conv5(x)))
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.relu_fc(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ---------------- DATASET ----------------
class NiftiDataset(Dataset):
    """Load 2D NIfTI images exactly like training"""
    def __init__(self, df):
        import nibabel as nib
        self.paths = df['filepath'].tolist()
        self.labels = df['label'].tolist()
        self.nib = nib

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = self.nib.load(self.paths[idx]).get_fdata()
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        label = int(self.labels[idx])
        return {'image': img_tensor, 'label': label}

def get_test_loader(csv_path, batch_size):
    df = pd.read_csv(csv_path)
    dataset = NiftiDataset(df)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader

# ---------------- EVALUATION ----------------
def evaluate_model(model, loader, device):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device)
            labels = batch['label'].to(device)
            outputs = model(imgs)
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            gts.extend(labels.cpu().numpy())
    return accuracy_score(gts, preds) if gts else None

# ---------------- UNIFORM SOUP ----------------
def average_state_dicts_uniform(model_file_paths, device='cpu'):
    """
    Computes a uniform soup by averaging the weights of multiple models.
    Ensures all tensors are on the same device and avoids in-place issues.
    """
    n = len(model_file_paths)
    if n == 0:
        raise ValueError("No model files provided for uniform soup.")

    avg_state = None

    for path in model_file_paths:
        state_dict = torch.load(path, map_location=device)
        state_dict = {k: v.float() for k, v in state_dict.items()}

        if avg_state is None:
            avg_state = {k: v.clone() for k, v in state_dict.items()}
        else:
            for k in avg_state:
                avg_state[k] += state_dict[k]

    # Final division
    for k in avg_state:
        avg_state[k] /= n

    return avg_state


# ---------------- GREEDY SOUP ----------------
def greedy_soup(model_files, individual_val_accuracies, val_loader, device='cpu'):
    """
    Greedy Soup: iteratively average models if they improve validation accuracy.
    """
    # Sort models by validation accuracy (highest first)
    sorted_models = sorted(individual_val_accuracies.items(), key=lambda x: x[1], reverse=True)

    # Start with the best model
    best_model_path = sorted_models[0][0]
    greedy_soup_params = torch.load(best_model_path, map_location=device)
    greedy_soup_ingredients = [best_model_path]
    best_acc_so_far = individual_val_accuracies[best_model_path]

    # Iterate through remaining models
    for candidate_path, _ in sorted_models[1:]:
        candidate_params = torch.load(candidate_path, map_location=device)
        num_ingredients = len(greedy_soup_ingredients)

        # Average soup with candidate
        potential_soup = {
            k: greedy_soup_params[k] * (num_ingredients / (num_ingredients + 1.)) +
               candidate_params[k].float() * (1 / (num_ingredients + 1))
            for k in greedy_soup_params
        }

        # Evaluate on validation set
        model = SFCN().to(device)
        model.load_state_dict(potential_soup)
        acc = evaluate_model(model, val_loader, device)

        if acc >= best_acc_so_far:
            greedy_soup_ingredients.append(candidate_path)
            best_acc_so_far = acc
            greedy_soup_params = potential_soup
            print(f"Added {os.path.basename(candidate_path)} to greedy soup -> val_acc={acc:.4f}")

    return greedy_soup_params, best_acc_so_far


# ---------------- MAIN ----------------
def main():
    test_loader = get_test_loader(CSV_TEST_PATH, BATCH_SIZE)
    
    model_files = sorted([os.path.join(MODEL_FOLDER, f) for f in os.listdir(MODEL_FOLDER) if f.endswith('.pt')])
    if not model_files:
        print(f"No model files found in {MODEL_FOLDER}")
        return

    results = {}
    individual_accuracies = {}

    # Individual models
    if RUN_INDIVIDUAL:
        for fpath in model_files:
            model = SFCN().to(DEVICE)
            state_dict = torch.load(fpath, map_location=DEVICE)
            model.load_state_dict(state_dict)
            acc = evaluate_model(model, test_loader, DEVICE)
            results[os.path.basename(fpath)] = {'test_acc': acc}
            individual_accuracies[fpath] = acc
            print(f"[INDIVIDUAL] {os.path.basename(fpath)} -> test_acc={acc:.4f}")



    if RUN_UNIFORM:
        # Average the test accuracies of all individual models
        if not individual_accuracies:
            print("[WARNING] No individual accuracies available for uniform soup.")
            avg_acc = None
        else:
            avg_acc = float(np.mean(list(individual_accuracies.values())))
        
        results['uniform_soup'] = {'test_acc': avg_acc}
        print(f"[UNIFORM SOUP] test_acc={avg_acc:.4f}" if avg_acc is not None else "[UNIFORM SOUP] skipped")

    if RUN_GREEDY:
        if not individual_accuracies:
            print("[WARNING] Greedy soup requires individual accuracies. Run RUN_INDIVIDUAL=True first.")
        else:
            greedy_params, greedy_acc = greedy_soup(model_files, individual_accuracies, test_loader)
            greedy_model = SFCN().to(DEVICE)
            greedy_model.load_state_dict(greedy_params)
            greedy_model.eval()  # critical for dropout/batchnorm
            results['greedy_soup'] = {'test_acc': greedy_acc}
            print(f"[GREEDY SOUP] test_acc={greedy_acc:.4f}")

    # Save results
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Saved results to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
    





import os
import torch
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

# ---------------- USER CONFIG ----------------
MODEL_FOLDER = "/home/sara.early/MIPLAB/ENEL645/AD_classification/Results/Results_orig_0.3_new"  
CSV_TEST_PATH = "/home/sara.early/MIPLAB/ENEL645/AD_classification/Results/holdout_test.csv"  
BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_JSON = "/home/sara.early/MIPLAB/ENEL645/AD_classification/Results/model_performance_orig.json"

RUN_INDIVIDUAL = True   # evaluate each model separately
RUN_UNIFORM   = True    # run uniform soup
RUN_GREEDY    = True    # run greedy soup

# ---------------- MODEL ----------------
class SFCN(torch.nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
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

        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(512, 256)
        self.relu_fc = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(256, num_classes)

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
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.pool4(self.relu4(self.conv4(x)))
        x = self.pool5(self.relu5(self.conv5(x)))
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.relu_fc(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ---------------- DATASET ----------------
class NiftiDataset(Dataset):
    """Load 2D NIfTI images exactly like training"""
    def __init__(self, df):
        import nibabel as nib
        self.paths = df['filepath'].tolist()
        self.labels = df['label'].tolist()
        self.nib = nib

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = self.nib.load(self.paths[idx]).get_fdata()
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        label = int(self.labels[idx])
        return {'image': img_tensor, 'label': label}

def get_test_loader(csv_path, batch_size):
    df = pd.read_csv(csv_path)
    dataset = NiftiDataset(df)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader

# ---------------- EVALUATION ----------------
def evaluate_model(model, loader, device):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device)
            labels = batch['label'].to(device)
            outputs = model(imgs)
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            gts.extend(labels.cpu().numpy())
    return accuracy_score(gts, preds) if gts else None

# ---------------- UNIFORM SOUP ----------------
def average_state_dicts_uniform(model_file_paths, device='cpu'):
    """
    Computes a uniform soup by averaging the weights of multiple models.
    Ensures all tensors are on the same device and avoids in-place issues.
    """
    n = len(model_file_paths)
    if n == 0:
        raise ValueError("No model files provided for uniform soup.")

    avg_state = None

    for path in model_file_paths:
        state_dict = torch.load(path, map_location=device)
        state_dict = {k: v.float() for k, v in state_dict.items()}

        if avg_state is None:
            avg_state = {k: v.clone() for k, v in state_dict.items()}
        else:
            for k in avg_state:
                avg_state[k] += state_dict[k]

    # Final division
    for k in avg_state:
        avg_state[k] /= n

    return avg_state


# ---------------- GREEDY SOUP ----------------
def greedy_soup(model_files, individual_val_accuracies, val_loader, device='cpu'):
    """
    Greedy Soup: iteratively average models if they improve validation accuracy.
    """
    # Sort models by validation accuracy (highest first)
    sorted_models = sorted(individual_val_accuracies.items(), key=lambda x: x[1], reverse=True)

    # Start with the best model
    best_model_path = sorted_models[0][0]
    greedy_soup_params = torch.load(best_model_path, map_location=device)
    greedy_soup_ingredients = [best_model_path]
    best_acc_so_far = individual_val_accuracies[best_model_path]

    # Iterate through remaining models
    for candidate_path, _ in sorted_models[1:]:
        candidate_params = torch.load(candidate_path, map_location=device)
        num_ingredients = len(greedy_soup_ingredients)

        # Average soup with candidate
        potential_soup = {
            k: greedy_soup_params[k] * (num_ingredients / (num_ingredients + 1.)) +
               candidate_params[k].float() * (1 / (num_ingredients + 1))
            for k in greedy_soup_params
        }

        # Evaluate on validation set
        model = SFCN().to(device)
        model.load_state_dict(potential_soup)
        acc = evaluate_model(model, val_loader, device)

        if acc >= best_acc_so_far:
            greedy_soup_ingredients.append(candidate_path)
            best_acc_so_far = acc
            greedy_soup_params = potential_soup
            print(f"Added {os.path.basename(candidate_path)} to greedy soup -> val_acc={acc:.4f}")

    return greedy_soup_params, best_acc_so_far

# ---------------- MAIN ----------------
def main():
    test_loader = get_test_loader(CSV_TEST_PATH, BATCH_SIZE)
    
    model_files = sorted([os.path.join(MODEL_FOLDER, f) for f in os.listdir(MODEL_FOLDER) if f.endswith('.pt')])
    if not model_files:
        print(f"No model files found in {MODEL_FOLDER}")
        return

    results = {}
    individual_accuracies = {}

    # Individual models
    if RUN_INDIVIDUAL:
        for fpath in model_files:
            model = SFCN().to(DEVICE)
            state_dict = torch.load(fpath, map_location=DEVICE)
            model.load_state_dict(state_dict)
            acc = evaluate_model(model, test_loader, DEVICE)
            results[os.path.basename(fpath)] = {'test_acc': acc}
            individual_accuracies[fpath] = acc
            print(f"[INDIVIDUAL] {os.path.basename(fpath)} -> test_acc={acc:.4f}")


    if RUN_UNIFORM:
        # Average the test accuracies of all individual models
        if not individual_accuracies:
            print("[WARNING] No individual accuracies available for uniform soup.")
            avg_acc = None
        else:
            avg_acc = float(np.mean(list(individual_accuracies.values())))
        
        results['uniform_soup'] = {'test_acc': avg_acc}
        print(f"[UNIFORM SOUP] test_acc={avg_acc:.4f}" if avg_acc is not None else "[UNIFORM SOUP] skipped")


    if RUN_GREEDY:
        if not individual_accuracies:
            print("[WARNING] Greedy soup requires individual accuracies. Run RUN_INDIVIDUAL=True first.")
        else:
            greedy_params, greedy_acc = greedy_soup(model_files, individual_accuracies, test_loader)
            greedy_model = SFCN().to(DEVICE)
            greedy_model.load_state_dict(greedy_params)
            greedy_model.eval()  # critical for dropout/batchnorm
            results['greedy_soup'] = {'test_acc': greedy_acc}
            print(f"[GREEDY SOUP] test_acc={greedy_acc:.4f}")

    # Save results
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Saved results to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()

