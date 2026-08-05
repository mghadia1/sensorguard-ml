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

## Verified CUDA comparison — August 4, 2026

A Colab Tesla T4 run used XGBoost 3.3.0 built with CUDA 12.9. The workflow
reused the frozen XGBoost configuration, fitted preprocessing on 6,000 training
rows, and compared predictions on 2,000 validation rows. It evaluated zero
official-test rows.

| Device | Median fit time (5 runs) | Validation AP | Validation ROC-AUC |
|---|---:|---:|---:|
| CPU | 0.8554 s | 0.7769 | 0.9660 |
| CUDA (Tesla T4) | **0.4180 s** | 0.7613 | 0.9670 |

**This run is underpowered and no longer verifies.** It is kept as history; the
n=15 protocol below supersedes it.

CUDA's median fit time was 2.05x faster than CPU's over five timed repeats per
device. An exact permutation test on the median difference across all 252 splits
gives **p = 0.0595**, so the difference is *not* established at this sample size.
The fastest CPU run (0.3681 s) beat the slowest CUDA run (0.7310 s), so the two
distributions overlap. This is not a power ceiling: the smallest p attainable
from a 5-vs-5 median statistic is 6/252 = 0.0238, so the design had room and five
repeats simply were not enough to separate a noisy shared Colab CPU from the GPU.

CPU fit times varied **4.29x** across repeats against CUDA's **1.77x**, so the
more robust finding is consistency rather than raw speed — and that one does not
depend on a median holding up.

The maximum absolute difference between CPU and CUDA validation probabilities was
0.2762. Read against the frozen 0.66 threshold, a row at 0.55 on CPU can be 0.83
on CUDA: opposite sides of the decision boundary. **These are two different
models, not one model on two devices** — XGBoost's CPU and CUDA `hist`
implementations sketch quantiles differently. The count of validation rows where
the two disagree at 0.66 is the number that matters, and this run did not record
the probabilities needed to compute it retrospectively; the new protocol emits it
as `disagreement_at_threshold`.

`sensorguard verify-evidence` now rejects this file:

```
Evidence rejected: underpowered benchmark: 5 timed runs per device, minimum is 10
```

### Pending: the n=15 rerun

The benchmark now takes 15 timed repeats per device after one discarded warm-up
fit, and computes the permutation p-value, standard deviations, spread ratios and
threshold disagreement inside the run. **That run has not been performed** — it
needs a Colab T4, which is Mayank's to execute. Until it exists there is no
n=15 claim to make, and the honest statement is the one above: not established at
n=5.

`TODO(phase-3)`: insert the n=15 measured claim here only after the downloaded
report passes `sensorguard verify-evidence`. The claim must state whether the
p-value was exact or a seeded Monte Carlo approximation; at 15-vs-15,
`C(30, 15)` exceeds the protocol's 200,000-split exact-enumeration limit.

The submitted values are preserved as normalized JSON in
`docs/evidence/cuda-colab-t4-report.json`. The original downloaded report's
SHA-256 is
`39916dd184af82243ba30528bc527fc229b2e59c3e7d61663dd98ece6fe69fc6`.
