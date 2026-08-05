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

## Corrected CUDA comparison — 15 repeats, warm-up excluded

A Colab Tesla T4 run used XGBoost 3.3.0 built with CUDA 12.9. The workflow
reused the frozen XGBoost configuration, fitted preprocessing on 6,000 training
rows, and compared predictions on 2,000 validation rows. It evaluated zero
official-test rows.

| Device | Discarded warm-up | Median fit time (15 runs) | Standard deviation | Spread |
|---|---:|---:|---:|---:|
| CPU | 1.3072 s | **0.2742 s** | 0.0928 s | 2.25x |
| CUDA (Tesla T4) | 0.7949 s | 0.4248 s | **0.0340 s** | **1.25x** |

With a discarded warm-up fit and 15 timed repeats per device, CPU was faster
than the Tesla T4: median 0.2742 s against 0.4248 s, a CPU-over-CUDA ratio of
0.6456, or **1.55x faster on CPU**. A seeded permutation test over 100,000
resamples gives **p = 0.0003 for CPU being faster**; the prespecified opposite
alternative, CUDA being faster, gives p = 0.9999. Thirteen of 15 CPU runs were
faster than every CUDA run.

The earlier CUDA headline was an artifact of warm-up contamination. In the
corrected run, the discarded CPU warm-up took 1.3072 s against a steady-state
median of 0.2742 s. The original five-repeat protocol did not separate warm-up
from timed fits and reported a CPU median of 0.8554 s. Once warm-up was measured
and excluded, the effect disappeared and reversed direction.

On 6,000 rows and 500 trees there is not enough work per boosting round to
amortise host-to-device transfer and kernel-launch overhead. This is a result for
this small tabular workload, not a universal CPU-versus-GPU conclusion.

**What survived the reversal:** CUDA remained the more consistent device—standard
deviation 0.0340 against 0.0928 and spread 1.25x against 2.25x. A finding that
holds through a reversal of the headline is the more trustworthy of the two.

### CPU and CUDA are two different models

XGBoost's `hist` implementations can sketch quantiles differently across device
and build contexts. At the frozen 0.66 threshold, the two Colab models disagree
on **8 of 2000 validation rows (0.4%)**, with a maximum absolute probability
difference of 0.2762. Run independently, the same validation F1 sweep selects
0.84 for the Colab CPU model and 0.76 for the CUDA model; a tuned threshold does
not transfer between fitted artifacts.

The validation agreement metrics are byte-identical to the August 4 Colab run:
CPU AP 0.7769458889549244 and ROC-AUC 0.9660211910851297; CUDA AP
0.7612584673736296 and ROC-AUC 0.9670411642918036. Reproducing those values with
`random_state=42` is a reproducibility signal, not a copy-paste error.

### Why 0.66, 0.84, and 0.76 are all recorded

The discrepancy is not caused by different objectives, splits, or sweep grids.
All use the same validation-only rule: maximize F1 over thresholds 0.01 through
0.99, then recall, then closeness to 0.5. The saved CPU model that produced the
official held-out result has validation AP 0.7659315934980411 and selects 0.66.
That frozen threshold governs its recorded test F1 of 0.7368.

The Colab benchmark trained separate Linux CPU and CUDA model instances rather
than loading the saved model. Their validation AP values are 0.7769458889549244
and 0.7612584673736296, proving that their fitted predictions differ from the
saved artifact even under the same configuration and random state. They select
0.84 and 0.76 respectively. The evidence supports platform/build-dependent
histogram training as the explanation, but it does not isolate the lower-level
cause further because model hashes and validation probabilities were not saved.
No threshold was changed and the official test split was not evaluated again.

### Superseded: the underpowered August 4 run

The retained five-repeat report produced the historical **2.05x CUDA speedup**
headline, but its timing sample included startup effects and its one-sided exact
permutation result was p = 0.0595. It is preserved unchanged apart from a
`superseded_by` pointer to the corrected n=15 evidence. The verifier returns
`superseded` for this file; an underpowered report without a verified replacement
remains a hard error.

Evidence files and downloaded-source SHA-256 values:

- current n=15 report: `docs/evidence/cuda-colab-t4-report-n15.json`,
  `d40b9b0e1550f361f7026b82acc3539e94680420bf9b5c251fd9ef5742399dcc`;
- superseded n=5 report: `docs/evidence/cuda-colab-t4-report.json`,
  `39916dd184af82243ba30528bc527fc229b2e59c3e7d61663dd98ece6fe69fc6`.
