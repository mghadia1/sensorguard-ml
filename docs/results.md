# Baseline results

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
