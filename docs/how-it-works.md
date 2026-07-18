# How SensorGuard ML works

## 1. The prediction problem

Each row describes one operating point with temperatures, rotational speed, torque, tool wear, and product type. The target is `Machine failure`, where `1` means a failure label and `0` means no failure label.

This is binary supervised classification because the training data supplies both features and known labels.

## 2. Why the split happens first

The data is divided into:

- 60% training data for fitting preprocessing and model parameters;
- 20% validation data for comparing models and choosing a probability threshold;
- 20% test data for one final evaluation.

If the test data influences feature scaling, model selection, or threshold selection, it stops being an independent test. The project uses stratification so each split has approximately the same small failure rate.

## 3. Leakage prevention

The five failure-mode columns directly contribute to the final machine-failure target. Supplying them as inputs would let the model reconstruct the label instead of learning from operating measurements. They are explicitly excluded, along with row and product identifiers.

## 4. Preprocessing

The numeric columns use median imputation followed by standardization. Product type uses most-frequent imputation followed by one-hot encoding. The current UCI file has no missing values, but the pipeline defines missing-value behavior so inference is predictable.

All preprocessing is inside the scikit-learn pipeline. Calling `fit` on training data learns medians, means, standard deviations, categories, and model parameters together. Validation and test data only call `transform` or `predict_proba`.

## 5. Models

- The majority baseline shows what happens without useful feature learning.
- Logistic regression is a linear probability model and provides a strong interpretable baseline.
- The decision tree learns nonlinear if/then regions.
- The random forest averages many trees to reduce a single tree's variance.
- The optional PyTorch MLP adds a learned nonlinear representation, but complexity is useful only if its measured comparison supports it.

## 6. Why accuracy is not enough

Failures are rare. A model that predicts "no failure" for every row can have high accuracy while finding zero failures.

The project therefore reports:

- precision: among predicted failures, how many were failures;
- recall: among actual failures, how many were found;
- F1: harmonic mean of precision and recall;
- ROC-AUC: ranking quality across thresholds;
- average precision: precision-recall ranking summary, especially useful for rare positives;
- confusion matrix: counts of true negatives, false positives, false negatives, and true positives.

## 7. Threshold selection

Models output probabilities, not final decisions. The default threshold 0.5 is not automatically best for an imbalanced problem. SensorGuard checks validation thresholds from 0.01 to 0.99, prefers the highest F1, then higher recall, then the value closest to 0.5.

That rule is part of the experiment, not a universal maintenance policy. A real organization would attach costs and safety consequences to false positives and false negatives.

## 8. Model selection and test evaluation

The classical model with the highest validation average precision wins, with validation F1 as the tie-breaker. Its already-selected threshold is then used once on the test split. The saved test predictions let another person reconstruct the confusion matrix.

Permutation importance is calculated on validation data. One feature is shuffled at a time, breaking its relationship with the labels, and the drop in average precision is measured. A larger drop means the trained pipeline relied more on that feature for this validation split. It does not prove that the feature causes failures.

## 9. What the PyTorch code teaches

The MLP has an input layer, one hidden ReLU layer, dropout, and one output logit. `BCEWithLogitsLoss` combines the sigmoid calculation with cross-entropy in a numerically stable form. A positive-class weight gives failures more influence during training.

Each batch follows five steps:

1. clear old gradients;
2. perform the forward pass;
3. calculate loss;
4. call `backward()` to calculate gradients;
5. call `step()` to update parameters.

## 10. What this project does not prove

The UCI data is synthetic. The split is random rather than chronological or site-based. The labels encode a simulated data-generating process. The result is evidence that the pipeline works on this dataset, not evidence that it predicts failures in a real factory.
