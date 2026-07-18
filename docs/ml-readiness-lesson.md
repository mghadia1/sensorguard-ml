# SensorGuard ML-readiness lesson

Read this after running SensorGuard and before retaking `sensorguard learn`. The goal is to understand the decisions, not memorize sentences.

## 1. Why the failure-mode columns are excluded

The target is `Machine failure`. The columns `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` say whether five individual failure modes occurred. The dataset creates the final target from those failure modes.

If we give those columns to the model, we are giving it parts of the answer. This is **target leakage**. The resulting score could look excellent even though the model did not learn to predict failure from temperatures, speed, torque, wear, and product type.

Think of leakage as giving a student the answer key during an exam.

An acceptable answer in your own words should say that the columns directly reveal or construct the target, so using them would make evaluation misleading.

## 2. Precision and recall

SensorGuard's test confusion matrix was:

- 47 true positives: predicted failure and a failure was present;
- 19 false positives: predicted failure but no failure was present;
- 21 false negatives: predicted no failure but a failure was present;
- 1,913 true negatives: predicted no failure and no failure was present.

**Precision** asks: of everything predicted as a failure, how many were actually failures?

```text
precision = true positives / (true positives + false positives)
          = 47 / (47 + 19)
          = 0.712
```

**Recall** asks: of all actual failures, how many did the model find?

```text
recall = true positives / (true positives + false negatives)
       = 47 / (47 + 21)
       = 0.691
```

Precision is about the reliability of failure alerts. Recall is about how many real failures were caught.

## 3. Why validation chooses the threshold

SensorGuard creates three data groups:

- training data teaches the model;
- validation data compares models and chooses the probability threshold;
- test data measures the already-finished decision process.

The random forest outputs a probability. We selected threshold `0.39` using validation data. A probability at or above `0.39` becomes a predicted failure.

If we repeatedly change that threshold after looking at test results, we start fitting decisions to the test set. The test result would no longer be independent.

Think of validation as a practice exam used to adjust your study plan. The test set is the final exam. You cannot use final-exam answers to prepare and still call it a fair final exam.

## 4. Why the random forest beating the MLP matters

Validation average precision was:

- random forest: `0.692`;
- PyTorch MLP: `0.505`.

The random forest also had better held-out test F1: `0.701` versus `0.516`.

This teaches that a neural network is not automatically the best model. SensorGuard has a relatively small structured table with six features. Tree ensembles often work well on this type of data. We choose models using measured evidence, not because “deep learning” sounds more advanced.

The MLP is still useful because it teaches PyTorch. It simply was not the best model for this experiment.

## 5. PyTorch training steps

For each batch:

1. **Forward pass:** input features move through the network and produce logits or predictions.
2. **Loss calculation:** the predictions are compared with correct labels.
3. **Backward pass:** `loss.backward()` calculates how each parameter contributed to the loss. These values are gradients.
4. **Optimizer step:** `optimizer.step()` updates the weights using the gradients.
5. **Clear gradients:** gradients must be cleared before the next update because PyTorch accumulates them by default.

Short version:

```text
forward predicts -> loss measures error -> backward computes gradients -> optimizer updates weights
```

## 6. Why the result does not prove factory performance

Any one of these is a correct limitation:

- UCI describes AI4I as synthetic data, not measurements collected from a deployed factory;
- the split is random and does not test a future time period;
- it does not test a different machine or factory;
- only 339 of 10,000 rows are failures;
- model reliance does not prove that a feature causes failure;
- real sensors may have missing values, drift, calibration errors, or new operating conditions.

Therefore, the test result proves only that the saved pipeline performed this way on this held-out split of this dataset.

## Retake instructions

Do not copy the wording above. Close this file and explain the ideas as if speaking to a classmate.

```bash
cd "/Users/programming/Documents/auto job applier/projects/sensorguard-ml"
source .venv/bin/activate
sensorguard learn
```

Afterward, ask Codex to review `outputs/learning-check/answers.json` again.

