# ML-readiness checklist

Mayank should be able to answer these questions while pointing to code and saved results.

1. Which columns are features, and why are the failure-mode columns excluded?
2. What is data leakage?
3. What does the training split do that validation and test do not?
4. Why do we stratify this dataset?
5. Why can accuracy look good when recall is zero?
6. What is the difference between precision and recall?
7. Why is average precision useful for rare failures?
8. Why is the threshold selected on validation data?
9. What does standardization learn from the training data?
10. What is one difference between logistic regression and a decision tree?
11. What is overfitting, and how could you see it here?
12. What happens during the PyTorch forward pass, backward pass, and optimizer step?
13. Why is this project not evidence about a real factory?

## Required personal experiments

1. Run the baseline and record the selected model and test confusion matrix.
2. Remove `class_weight="balanced"` from logistic regression, rerun, and explain the recall change.
3. Change the decision-tree depth from 6 to 2 and then 15. Compare training intuition with validation evidence.
4. Choose thresholds 0.2, 0.5, and 0.8 for one model and write how precision and recall move.
5. Run the PyTorch comparison and explain whether its added complexity improved validation average precision.

