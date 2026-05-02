## Environment Notes

Recommended setup:

```bash
conda env create -f environment.yml
conda activate model_soups
```

If you prefer a manual setup, install:

```bash
conda create --name model_soups python=3.11
conda activate model_soups
conda install --yes -c pytorch -c nvidia pytorch torchvision pytorch-cuda=12.1
pip install ftfy regex tqdm wget requests matplotlib pandas
```

If your environment does not provide CLIP already, install the implementation expected by the codebase separately.
