# SensorGuard ML

SensorGuard ML is a general machine-learning project for predicting machine-failure labels from equipment measurements. It is the ML-readiness project that comes before medical imaging or medical robotics.

The project uses the UCI AI4I 2020 Predictive Maintenance dataset. UCI describes it as a **synthetic** dataset reflecting industrial predictive-maintenance data. It contains 10,000 rows and is licensed under CC BY 4.0. It must not be described as data collected from deployed factory equipment.

Dataset source: <https://archive.ics.uci.edu/dataset/601/ai4i>

## What the pipeline does

1. Downloads the official archive and checks its SHA-256 hash.
2. Validates columns, categories, missing values, finite values, and unique row IDs.
3. Removes IDs and five failure-mode flags from the model features.
4. Makes deterministic 60/20/20 stratified train, validation, and test splits.
5. Fits preprocessing only on training data.
6. Compares a majority baseline, logistic regression, decision tree, and random forest.
7. Selects the model using validation average precision.
8. Selects a probability threshold using validation F1 and recall.
9. Evaluates the selected pipeline once on the held-out test split.
10. Calculates permutation importance on validation data, not test data.
11. Saves the model, metrics, predictions, comparison chart, and confusion matrix.
12. Optionally trains a small PyTorch MLP on the same split for comparison.

The verified baseline results are recorded in `docs/results.md`. Generated models and result files remain under ignored `outputs/` directories rather than being committed as source code.

## Why some columns are excluded

`UDI` and `Product ID` are identifiers rather than operating measurements. `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are individual failure-mode target columns that directly determine `Machine failure`. Using them would leak the answer into the features and produce a misleadingly easy result.

The actual features are product type, air temperature, process temperature, rotational speed, torque, and tool wear.

## Setup

```bash
cd projects/sensorguard-ml
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[dev,torch]'
```

## Download and audit the data

```bash
sensorguard download --destination data/raw
sensorguard audit --data data/raw/ai4i2020.csv
```

## Train and evaluate

```bash
sensorguard train \
  --data data/raw/ai4i2020.csv \
  --out outputs/baseline \
  --with-torch
```

Run the tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Complete the ML-readiness check

Read `docs/ml-readiness-lesson.md`. After running the project, record your explanations interactively:

```bash
sensorguard learn
```

The command saves `outputs/learning-check/answers.json`. Ask Codex to review that file. Medical ML work starts only after the answers show that the pipeline is understandable and interview-defensible.

## Batch inference

Provide a CSV containing the six feature columns, then run:

```bash
sensorguard predict \
  --model outputs/baseline/model.joblib \
  --input examples/prediction_rows.csv \
  --out outputs/predictions.csv
```

## Honest limitations

- The source data is synthetic and has a low failure rate.
- Random stratified splitting does not simulate future deployment or a different factory.
- A good held-out score does not prove causal understanding or safe maintenance decisions.
- Threshold selection reflects F1 and recall, not a verified business cost model.
- The model is a learning project and should not control real maintenance decisions.

Read `docs/how-it-works.md` before using this project on a resume.
