# Baseline and XGBoost comparison results

Run date: July 14, 2026

Command:

```bash
sensorguard train \
  --data data/raw/ai4i2020.csv \
  --out outputs/baseline \
  --with-torch \
  --torch-epochs 40
```

## Data audit

- 10,000 rows;
- 339 failure rows, or 3.39%;
- six model features;
- no missing values in required columns;
- no duplicate UDI values;
- IDs and five failure-mode target columns excluded.

The deterministic stratified split contained 6,000 training rows, 2,000 validation rows, and 2,000 test rows. The test split contained 68 failure rows.

## Validation comparison

| Model | Average precision | F1 at selected validation threshold |
|---|---:|---:|
| Majority baseline | 0.034 | 0.066 |
| Logistic regression | 0.407 | 0.420 |
| Decision tree | 0.539 | 0.635 |
| Random forest | 0.692 | 0.656 |
| PyTorch MLP | 0.505 | 0.496 |

The random forest was selected because it had the highest validation average precision. Its validation-selected probability threshold was 0.39.

## Frozen XGBoost extension — August 4, 2026

The extension preserved the dataset, six leakage-safe features, random state
42, 60/20/20 stratified split, validation selection rule, and test policy. It
added one XGBoost 3.3.0 candidate using CPU histogram trees, 500 estimators,
learning rate 0.05, maximum depth 4, row/column subsampling 0.8, and a positive
class weight calculated only from the training split.

| Model | Validation average precision | Validation F1 |
|---|---:|---:|
| Random forest | 0.6917 | 0.6565 |
| XGBoost | **0.7659** | **0.7463** |

The frozen rule selected XGBoost and threshold 0.66. Its one held-out test
evaluation produced precision 0.7538, recall 0.7206, F1 0.7368, ROC-AUC 0.9770,
average precision 0.7617, and confusion matrix [[1916, 16], [19, 49]]. Relative
to the preserved random-forest result, test F1 improved about 0.036 and average
precision about 0.051. This does not remove the dataset's synthetic-data and
random-split limitations.

Evidence SHA-256:

- frozen protocol: `d02636b10babf386159e6de53626eadff56e4b1ba40392a6bd24057379cba340`;
- metrics: `6168f939237d1486bd9523fb4d6ae54674817e715caf7503ed22af075e15502e`;
- saved model bundle: `fc1662afbc8d96e9e3c8f7a8f0a8de96e3eea2fe053b107e8387f799cf269a28`.

## Held-out test result

The selected random forest produced:

- precision: 0.712;
- recall: 0.691;
- F1: 0.701;
- ROC-AUC: 0.974;
- average precision: 0.711;
- true negatives: 1,913;
- false positives: 19;
- false negatives: 21;
- true positives: 47.

The 40-epoch PyTorch comparison produced test average precision 0.529 and F1 0.516. It did not outperform the random forest. On this small structured dataset, the added neural-network complexity was not justified by the measured comparison.

## Validation permutation importance

Shuffling each feature on validation data produced these average-precision drops:

| Feature | Mean AP drop |
|---|---:|
| Torque | 0.424 |
| Rotational speed | 0.343 |
| Tool wear | 0.253 |
| Air temperature | 0.242 |
| Process temperature | 0.104 |
| Product type | 0.027 |

For this fitted pipeline and validation split, torque caused the largest loss in ranking performance when shuffled. This describes model reliance, not physical causation.

## Interpretation limits

These results apply to one deterministic random split of the synthetic UCI dataset. The random split may place nearby values from the generated sequence into different splits, and it does not test a future time period, new machine, or new factory. The scores must not be described as real-world predictive-maintenance performance.

## CUDA implementation status (August 4, 2026)

A Colab-ready CPU-versus-CUDA XGBoost benchmark is implemented in
`notebooks/sensorguard_cuda_colab.ipynb`. The workflow verifies the NVIDIA
runtime, reuses the frozen XGBoost configuration, fits preprocessing on training
data only, and evaluates parity on validation data only. It explicitly records
that zero official-test rows were evaluated.

No CUDA timing or metric is reported here yet. This MacBook cannot execute CUDA,
and an unexecuted notebook is not experimental evidence. The generated
`outputs/cuda-benchmark/report.json` should be reviewed after a successful Colab
T4 run before any GPU result is documented or used in application materials.
