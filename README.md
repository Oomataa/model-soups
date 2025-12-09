# Bootstrapped Model Soups for Fairness

##  Project Overview
This project investigates **Model Soups** (Wortsman et al., 2022) as a strategy to improve fairness and robustness under **spurious correlations**.

We extend the original approach by incorporating **bootstrapping** during teacher training and evaluating whether model diversity leads to improved **worst-group accuracy**.

Experiments focus on:
- **Waterbirds** (spurious background correlations)
- **ImageNet + OOD variants** (distribution shift)
- A **medical imaging task** (Alzheimer’s Disease vs Cognitively Normal classification)

---

##  Key Goals
- Reproduce **Uniform Model Soup** results from the original paper (no bootstrapping).
- Train **bootstrapped teacher models** to introduce dataset-level diversity.
- Compare:
  - **Uniform Soup (non-bootstrap vs bootstrap)**
  - **Greedy/Ordered Soup (non-bootstrap vs bootstrap)**
- Analyze both:
  - **Average accuracy**
  - **Worst-group / worst-subgroup accuracy** as a fairness-oriented metric.

---

##  Dataset: Waterbirds
- Binary classification: *Waterbird vs Landbird*
- Environment attribute: *Water vs Land background*
- **4 groups** → measures reliance on background:

| Group | Species   | Background | Frequency |
|-------|-----------|------------|-----------|
| 0     | Waterbird | Water      | Majority  |
| 1     | Waterbird | Land       | Minority  |
| 2     | Landbird  | Land       | Majority  |
| 3     | Landbird  | Water      | Minority  |

Performance is evaluated using:
- **Average accuracy**
- **Worst-group accuracy** → fairness indicator

---

##  ImageNet & Medical Imaging Experiments

### ImageNet + OOD Benchmarks
Beyond Waterbirds, we also run experiments on **ImageNet** and several **out-of-distribution (OOD) variants**, such as:
- ImageNet-R (renditions)
- ImageNet-Sketch
- ObjectNet
- Other ImageNet-based OOD splits

For these experiments, we:
- Train multiple **teacher models** (e.g., ResNet / ViT) on ImageNet.
- Build **Model Soups** (uniform and ordered/“greedy”) over these teachers.
- Compare:
  - Single models vs soups
  - Bootstrap vs non-bootstrap training
  - Uniform vs ordered (greedy) selection strategies

The goal is to test whether **bootstrapping + soups** improves **OOD robustness** in a large-scale setting.

### Medical Imaging: AD vs CN Classification
We also apply model soups to a **binary medical imaging task**:
- Task: distinguish **Alzheimer’s Disease (AD)** vs **Cognitively Normal (CN)** patients.
- Setting: multiple independently trained teacher models (with and without bootstrapping).
- Aggregation: compare **uniform** soups vs **ordered/greedy** soups.

In this domain:
- Greedy soups are especially useful, as they selectively include only models that **improve validation performance**, which helps avoid overfitting to spurious or noisy patterns.
- We observe that greedy soups can **outperform both individual models and uniform soups**, especially when signal is subtle and data is limited.


---

##  Teacher Training (Waterbirds)
We trained 30 ResNet-50 teacher models under two settings:

| Setting | Description |
|--------|-------------|
| **Non-Bootstrap ERM** | Standard training on full dataset |
| **Bootstrap ERM** | Each model trains on a **70% resampled dataset** to increase diversity |

Hyperparameter: **Learning Rate = 3e-3**  
(Additional HP search planned / used to refine teacher selection per seed.)

---

## Model Soup Methods

### 1️ Uniform Model Soup
- *Simple average* of model weights  
- All teachers contribute equally
- Sensitive to low-performing or overfitted teachers

### 2️ Greedy (Ordered) Model Soup
- Sort teachers by validation accuracy
- Add models one by one in that order
- Only keep a model in the soup if it **improves validation performance**


---

## Results Summary on Waterbirds (5 runs — mean ± std)

| Method                     | Test Avg Acc      | Test Worst-Group Acc |
|---------------------------|-------------------|----------------------|
| Uniform Soup (non-bootstrap)   | ~89.6% ± 0.1 | ~43.0% ± 0.8 |
| Uniform Soup (bootstrap)       | ~87.9% ± 0.1 | **49.6% ± 0.4** |
| Greedy Soup (non-bootstrap)    | ~89.4% ± 0.0 | ~43.2% ± 0.0 |
| Greedy Soup (bootstrap)        | ~86.7% ± 0.0 | **50.9% ± 0.0** |

###  Key Finding (Waterbirds)
Bootstrapping **improves worst-group generalization**, showing increased robustness to spurious cues, even when average accuracy slightly drops. Greedy/ordered soups benefit most from having diverse, strong teachers, while uniform soups are more sensitive to the inclusion of weaker models.

---

##  Additional Experiments: ImageNet & OOD Benchmarks

Beyond Waterbirds, we also study model soups on large-scale **ImageNet** classification and multiple **distribution-shifted test sets**.

### Datasets
- **In-distribution**: ImageNet-1k (ILSVRC 2012)
- **OOD datasets** (same label space, different distributions):
  - ImageNet-R
  - ImageNet-A
  - ImageNet-Sketch
  - ImageNet-V2
  - ObjectNet

### Setup (High-Level)
- Train multiple teacher models (e.g., ResNet / ViT backbones) on ImageNet under:
  - **Non-bootstrapped ERM**
  - **Bootstrapped ERM** at different bootstrap fractions (e.g., 40%, 70%, 90%)
- For each setting:
  - Evaluate **single models** and **uniform soups**
  - Analyze performance on both **ImageNet** and its **OOD variants**

---

##  Medical Imaging: AD vs CN Classification

We additionally test model soups on a **downstream medical imaging task**:

### Task
- Binary classification: **Alzheimer’s Disease (AD)** vs **Cognitively Normal (CN)**.
- Input: Brain MRI scans (preprocessed into 2D slices or 3D volumes, depending on the setup).
- Goal: Assess whether findings from Waterbirds and ImageNet **transfer to a clinically relevant setting**.

### Approach
- Start from ImageNet-pretrained backbones (e.g., ResNet-50).
- Fine-tune multiple teacher models on the AD vs CN task.
- Build:
  - **Uniform soups** over all fine-tuned teachers.
  - **Greedy/ordered soups** that prioritize models that best generalize on a validation split (e.g., using accuracy or AUROC).

### High-Level Outcome
- **Greedy soups consistently outperform uniform soups** and individual models on the AD vs CN task.
- The selective inclusion of only helpful teachers is particularly beneficial when:
  - The dataset is small,
  - The signal is subtle,
  - And overfitting is common.
- This suggests that **bootstrapped and greedy model soups are promising for medical imaging applications**, where reliability and generalization are critical.

---

##  Reproducing Experiments

### 1️ Train Waterbirds Teachers
```bash
for s in {1..30}; do
python3 finetune.py -c cfgs/waterbirds.yaml \
  --seed $s \
  --bootstrap True  # or False
done

---


## Evaluate Model Soup (Uniform / Ordered)
for s in {1..5}; do
python3 ensemble.py -c cfgs/waterbirds.yaml \
  --seed $s \
  --ensemble_size 30 \
  --bootstrap True \
  --soup_selection ordered   # "uniform" for baseline
done

##2️ ImageNet + OOD (Cluster / SLURM Jobs)
# From the project root
sbatch slurm/eval_models.slurm

##We use run_bootstrap_array.slurm to build model soups with and without bootstrapping:
# From the project root
sbatch slurm/run_bootstrap_array.slurm

##To call ensembles
python3 ensemble.py \
  -c cfgs/imagenet.yaml \
  --seed $SEED \
  --ensemble_size 30 \
  --bootstrap $BOOTSTRAP \
  --soup_selection $SOUP_SELECTION



