# Net-Wise Attention for Residual-Path Aggregation

This repository contains the seed-42 reproducibility subset for the experiments reported in **Net-Wise Attention for Residual-Path Aggregation in Residual Networks**.

Released under the permissive [MIT License](LICENSE).

## Repository layout

- `exp01_baselines`: CIFAR-100 ResNet-18 and ResNet-34 references.
- `exp02_full_netattn`: Full NetAttn with learned or uniform path coefficients.
- `exp03_topk`: inference-only Top-7 ranking controls from the trained Full NetAttn checkpoint.
- `exp04_path`: predefined Path9/Path37 training and Top-1/Top-4 inference.
- `models`, `datasets`, `utils`: the transitive Python dependency closure of the retained experiments.
- `paper/NetAttn`: MinerU Markdown conversion and extracted figures.

The Full NetAttn internal path index is retained for checkpoint compatibility,
while all displayed path bits are ordered from shallow to deep. See
[`paper/NetAttn/ERRATUM.md`](paper/NetAttn/ERRATUM.md) for the correction to the
original path-label and analytical-cost interpretation.

Each retained training run contains only `best_model.pth` and the available training/attention CSV files. `checkpoint_latest.pth`, TensorBoard events, caches, exploratory architectures, ResAttn, trained-TopK variants, and experiments outside the paper are excluded.

Model weights are tracked with [Git LFS](https://git-lfs.com/). Install Git
LFS before cloning if you need the checkpoints.

## Paper-aligned scope

The release retains the paper's ResNet-18/34 baselines, Full NetAttn Uniform, Full NetAttn, Full NetAttn Top-7 controls, Path9/Top-1, and Path37/Top-4. Top-K is evaluated by pruning the trained Full NetAttn checkpoint without coefficient renormalization, matching the paper methodology.

## Environment and data

The direct dependencies are pinned for Python 3.10 and the CUDA 12.8 PyTorch build:

```bash
python -m pip install -r requirement.txt
```

CIFAR-100 is downloaded automatically to `data/cifar-100` in the repository.
Call `get_dataloaders(..., data_root="/custom/path")` to use another location.
The historical `is_server` argument remains accepted but no longer changes the
path.

Run the public path-order regression tests with:

```bash
python -m unittest tests.test_full_netattn_path_order -v
```
