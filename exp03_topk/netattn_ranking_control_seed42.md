# Seed-42 NetAttn Ranking Control

## Protocol

- Checkpoint: `exp31-ConfigNetAttn\runs\ResNet-18-CIFAR100-NetAttn-nozero-sum=1_2026-04-15_21-40-57\best_model.pth`
- Checkpoint SHA-256: `dbf305a2d2f0c3c3bb3a2aca22f9ebc045b9bb28815fc7bb2f5f3940c518fb20`
- Dataset: CIFAR-100 test split (10,000 images), using the same normalization as the source notebook and no test-time augmentation.
- Environment: `G:\Tools\miniconda\envs\snn\python.exe`; PyTorch `2.9.1+cu128`; device `NVIDIA GeForce RTX 5080`.
- Path order: the eight bits are ordered from shallow to deep as `L1.B0, L1.B1, L2.B0, L2.B1, L3.B0, L3.B1, L4.B0, L4.B1`; `11111100` therefore skips the last two residual transformations. The stable internal index retains its historical encoding for checkpoint compatibility.
- Full NetAttn uses the checkpoint softmax coefficients, whose sum is one. Matching the source notebook, Top-7 is ranked and weighted by `attention_final.csv` and retains those seven learned coefficients without renormalization.
- Equal mass-matched controls assign each selected path `M7/7 = 0.098733632`, where `M7 = 0.691135423`.
- Random control: Each Random-7 set is the Top-7 index set transformed by one fixed random 8-bit XOR mask. This randomizes path identities while preserving the exact prefix-tree structure and MAC proxy. The fixed RNG seed is 42007.
- The active-state ratio is the sum of unique active states after the eight residual blocks divided by the eight states of ResNet-18. The relative MAC proxy additionally weights parent states by block convolution cost. Neither quantity is a wall-clock measurement. All random sets have exactly the same prefix profile and both cost proxies as Top-7.
- All coefficient settings are evaluated from the same terminal path features in each batch. This avoids redundant full-family backbone passes and does not change the resulting weighted features or logits.

## Results

| Setting | Paths | Coeff. sum | Active states | MAC proxy | Accuracy (%) |
|---|---:|---:|---:|---:|---:|
| Full NetAttn | 256 | 1.000000 | 63.750x | 32.125x | 81.600 |
| NetAttn Top-7, learned | 7 | 0.691135 | 5.000x | 4.171x | 81.500 |
| Top-7, equal mass-matched | 7 | 0.691135 | 5.000x | 4.171x | 81.220 |
| Structure-matched Random-7, equal mass-matched | 7 | 0.691135 | 5.000x | 4.171x | 30.380 +/- 24.903 |
| Bottom-7, equal mass-matched | 7 | 0.691135 | 5.500x | 4.655x | 42.430 |

The random value is the mean and sample standard deviation over 20 structure-matched path sets. Its observed range is 4.390% to 80.380%.

## Selected Paths

### NetAttn Top-7 (rank order)

1. `11101111` (index 247, learned coefficient 0.140296891): L1.B0:Full | L1.B1:Full | L2.B0:Full | L2.B1:Skip | L3.B0:Full | L3.B1:Full | L4.B0:Full | L4.B1:Full
2. `10111111` (index 253, learned coefficient 0.128205165): L1.B0:Full | L1.B1:Skip | L2.B0:Full | L2.B1:Full | L3.B0:Full | L3.B1:Full | L4.B0:Full | L4.B1:Full
3. `01111111` (index 254, learned coefficient 0.119977012): L1.B0:Skip | L1.B1:Full | L2.B0:Full | L2.B1:Full | L3.B0:Full | L3.B1:Full | L4.B0:Full | L4.B1:Full
4. `11111111` (index 255, learned coefficient 0.112901852): L1.B0:Full | L1.B1:Full | L2.B0:Full | L2.B1:Full | L3.B0:Full | L3.B1:Full | L4.B0:Full | L4.B1:Full
5. `11111100` (index 63, learned coefficient 0.080659650): L1.B0:Full | L1.B1:Full | L2.B0:Full | L2.B1:Full | L3.B0:Full | L3.B1:Full | L4.B0:Skip | L4.B1:Skip
6. `10101111` (index 245, learned coefficient 0.068017691): L1.B0:Full | L1.B1:Skip | L2.B0:Full | L2.B1:Skip | L3.B0:Full | L3.B1:Full | L4.B0:Full | L4.B1:Full
7. `01101111` (index 246, learned coefficient 0.041077163): L1.B0:Skip | L1.B1:Full | L2.B0:Full | L2.B1:Skip | L3.B0:Full | L3.B1:Full | L4.B0:Full | L4.B1:Full

### NetAttn Bottom-7 (rank order)

1. `10110111` (index 237, learned coefficient 0.000017448): L1.B0:Full | L1.B1:Skip | L2.B0:Full | L2.B1:Full | L3.B0:Skip | L3.B1:Full | L4.B0:Full | L4.B1:Full
2. `10001111` (index 241, learned coefficient 0.000016752): L1.B0:Full | L1.B1:Skip | L2.B0:Skip | L2.B1:Skip | L3.B0:Full | L3.B1:Full | L4.B0:Full | L4.B1:Full
3. `11000111` (index 227, learned coefficient 0.000015388): L1.B0:Full | L1.B1:Full | L2.B0:Skip | L2.B1:Skip | L3.B0:Skip | L3.B1:Full | L4.B0:Full | L4.B1:Full
4. `11010111` (index 235, learned coefficient 0.000014906): L1.B0:Full | L1.B1:Full | L2.B0:Skip | L2.B1:Full | L3.B0:Skip | L3.B1:Full | L4.B0:Full | L4.B1:Full
5. `01100111` (index 230, learned coefficient 0.000013756): L1.B0:Skip | L1.B1:Full | L2.B0:Full | L2.B1:Skip | L3.B0:Skip | L3.B1:Full | L4.B0:Full | L4.B1:Full
6. `10010111` (index 233, learned coefficient 0.000010912): L1.B0:Full | L1.B1:Skip | L2.B0:Skip | L2.B1:Full | L3.B0:Skip | L3.B1:Full | L4.B0:Full | L4.B1:Full
7. `10100111` (index 229, learned coefficient 0.000010872): L1.B0:Full | L1.B1:Skip | L2.B0:Full | L2.B1:Skip | L3.B0:Skip | L3.B1:Full | L4.B0:Full | L4.B1:Full

The 20 Random-7 path sets, XOR masks, individual accuracies, and coefficient vectors are stored in the accompanying JSON and CSV.

## Validation and Limitations

- Maximum absolute difference between checkpoint-derived and CSV attention coefficients: `4.733e-09`.
- Full-checkpoint accuracy reproduced here: `81.600%`.
- This is a single-checkpoint control. The 20 random sets quantify subset-selection variability within seed 42, not training-seed variability.
- The structure-matched random sets are conditional controls, not uniform samples from all possible 7-of-256 subsets. The XOR construction deliberately holds prefix sharing and the cost proxy constant.
- Accuracy comparisons isolate path identity and coefficient assignment for this jointly trained mixture. They do not measure standalone subnetwork accuracy or prove globally optimal path selection.
- The reported cost is analytic. No wall-clock latency or reduced CUDA-memory measurement was performed in this control.
