# Anonymous Supplementary Code

This repository is an anonymized code snapshot accompanying a double-blind submission.

It contains code for:
- fine-tuning CLIP-based image classifiers
- constructing uniform and greedy model soups
- running bootstrap-resampled fine-tuning for soup diversity
- evaluating robustness under natural distribution shift and subgroup shift

## Repository Layout

- `main.py`: evaluation, soup construction, plotting entrypoint
- `finetune.py`: fine-tuning entrypoint
- `datasets/`: dataset loaders and metadata
- `imagenetv2_pytorch/`: ImageNet-V2 helper loader
- `aggregate_seed_results.py`: repeated-run result aggregation
- `run_soups.slurm`, `run_bootstrap_array.slurm`, `eval_models.slurm`: generic cluster templates

Only the core files needed for the main training and evaluation pipeline are included in this snapshot.

## Environment

Create the conda environment with:

```bash
conda env create -f environment.yml
conda activate model_soups
```

See [environment.md](environment.md) for package notes.

## Datasets

Dataset preparation notes are in [datasets.md](datasets.md).

Most scripts expect a directory passed through `--data-location`. Adjust this to match your local or cluster storage layout.

## Minimal Usage

Evaluate individual checkpoints:

```bash
python main.py --eval-individual-models --data-location <data_dir> --model-location <models_dir>
```

Build a uniform soup:

```bash
python main.py --uniform-soup --data-location <data_dir> --model-location <models_dir>
```

Build a greedy soup:

```bash
python main.py --greedy-soup --data-location <data_dir> --model-location <models_dir>
```

Run bootstrap fine-tuning:

```bash
python finetune.py --help
```

Aggregate repeated-run summaries:

```bash
python aggregate_seed_results.py --help
```

## Notes

- This snapshot is intentionally anonymized for review.
- Cluster scripts are templates and require local path/account adjustments.
- Large model checkpoints, datasets, and generated result files are intentionally omitted from this repository snapshot.
