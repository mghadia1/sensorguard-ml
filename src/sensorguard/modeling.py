"""Classical model training, validation selection, and test evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .data import FEATURE_COLUMNS, TARGET_COLUMN, DatasetSplits, feature_target, split_summary
from .report import save_confusion_matrix, save_model_comparison


CATEGORICAL_FEATURES = ("Type",)
NUMERIC_FEATURES = tuple(column for column in FEATURE_COLUMNS if column != "Type")


@dataclass(frozen=True)
class CandidateResult:
    name: str
    pipeline: Pipeline
    threshold: float
    validation_metrics: dict[str, Any]


def make_preprocessor(*, dense: bool = False) -> ColumnTransformer:
    numeric = Pipeline(
        steps=(
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        )
    )
    categorical = Pipeline(
        steps=(
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=not dense),
            ),
        )
    )
    return ColumnTransformer(
        transformers=(
            ("numeric", numeric, list(NUMERIC_FEATURES)),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
        )
    )


def candidate_pipelines(*, random_state: int = 42) -> dict[str, Pipeline]:
    return {
        "majority_baseline": Pipeline(
            steps=(
                ("preprocess", make_preprocessor()),
                ("model", DummyClassifier(strategy="prior")),
            )
        ),
        "logistic_regression": Pipeline(
            steps=(
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            )
        ),
        "decision_tree": Pipeline(
            steps=(
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=6,
                        min_samples_leaf=8,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            )
        ),
        "random_forest": Pipeline(
            steps=(
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            )
        ),
    }


def binary_metrics(
    labels: np.ndarray | pd.Series,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    label_values = np.asarray(labels, dtype=np.int64)
    probability_values = np.asarray(probabilities, dtype=np.float64)
    if label_values.ndim != 1 or probability_values.shape != label_values.shape:
        raise ValueError("labels and probabilities must be matching vectors")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")
    predictions = (probability_values >= threshold).astype(np.int64)
    matrix = confusion_matrix(label_values, predictions, labels=(0, 1)).astype(int).tolist()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(label_values, predictions)),
        "precision": float(precision_score(label_values, predictions, zero_division=0)),
        "recall": float(recall_score(label_values, predictions, zero_division=0)),
        "f1": float(f1_score(label_values, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(label_values, probability_values)),
        "average_precision": float(average_precision_score(label_values, probability_values)),
        "confusion_matrix": matrix,
    }


def select_threshold(labels: np.ndarray | pd.Series, probabilities: np.ndarray) -> float:
    """Choose a validation threshold by F1, then recall, then closeness to 0.5."""

    best_threshold = 0.5
    best_key = (-1.0, -1.0, -1.0)
    for threshold in np.linspace(0.01, 0.99, 99):
        metrics = binary_metrics(labels, probabilities, threshold=float(threshold))
        key = (
            float(metrics["f1"]),
            float(metrics["recall"]),
            -abs(float(threshold) - 0.5),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def train_candidates(
    splits: DatasetSplits,
    *,
    random_state: int = 42,
) -> list[CandidateResult]:
    train_features, train_labels = feature_target(splits.train)
    validation_features, validation_labels = feature_target(splits.validation)
    results: list[CandidateResult] = []
    for name, pipeline in candidate_pipelines(random_state=random_state).items():
        pipeline.fit(train_features, train_labels)
        probabilities = pipeline.predict_proba(validation_features)[:, 1]
        threshold = select_threshold(validation_labels, probabilities)
        metrics = binary_metrics(validation_labels, probabilities, threshold=threshold)
        results.append(
            CandidateResult(
                name=name,
                pipeline=pipeline,
                threshold=threshold,
                validation_metrics=metrics,
            )
        )
    return results


def choose_candidate(results: list[CandidateResult]) -> CandidateResult:
    if not results:
        raise ValueError("at least one candidate result is required")
    return max(
        results,
        key=lambda result: (
            float(result.validation_metrics["average_precision"]),
            float(result.validation_metrics["f1"]),
        ),
    )


def train_evaluate_save(
    splits: DatasetSplits,
    output_dir: str | Path,
    *,
    random_state: int = 42,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidates = train_candidates(splits, random_state=random_state)
    selected = choose_candidate(candidates)
    validation_features, validation_labels = feature_target(splits.validation)
    test_features, test_labels = feature_target(splits.test)
    test_probabilities = selected.pipeline.predict_proba(test_features)[:, 1]
    test_metrics = binary_metrics(test_labels, test_probabilities, threshold=selected.threshold)

    bundle = {
        "pipeline": selected.pipeline,
        "threshold": selected.threshold,
        "feature_columns": list(FEATURE_COLUMNS),
        "model_name": selected.name,
        "random_state": random_state,
    }
    joblib.dump(bundle, output / "model.joblib")
    predictions = pd.DataFrame(
        {
            "UDI": splits.test["UDI"].to_numpy(),
            "actual_failure": test_labels.to_numpy(),
            "failure_probability": test_probabilities,
            "predicted_failure": (test_probabilities >= selected.threshold).astype(np.int64),
        }
    )
    predictions.to_csv(output / "test_predictions.csv", index=False)

    importance_result = permutation_importance(
        selected.pipeline,
        validation_features,
        validation_labels,
        scoring="average_precision",
        n_repeats=10,
        random_state=random_state,
        n_jobs=1,
    )
    importance_frame = pd.DataFrame(
        {
            "feature": list(FEATURE_COLUMNS),
            "mean_average_precision_drop": importance_result.importances_mean,
            "standard_deviation": importance_result.importances_std,
        }
    ).sort_values("mean_average_precision_drop", ascending=False, ignore_index=True)
    importance_frame.to_csv(output / "validation_permutation_importance.csv", index=False)

    report: dict[str, Any] = {
        "split_summary": split_summary(splits),
        "selection_rule": "highest validation average precision, then validation F1",
        "threshold_rule": "highest validation F1, then recall, then closeness to 0.5",
        "validation_candidates": {
            result.name: result.validation_metrics for result in candidates
        },
        "selected_model": selected.name,
        "selected_threshold": selected.threshold,
        "validation_permutation_importance": importance_frame.to_dict(orient="records"),
        "test_metrics": test_metrics,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    save_model_comparison(
        {
            result.name: float(result.validation_metrics["average_precision"])
            for result in candidates
        },
        output / "validation_model_comparison.svg",
        metric_name="average precision",
    )
    save_confusion_matrix(
        test_metrics["confusion_matrix"],
        output / "test_confusion_matrix.svg",
        title=f"Test confusion matrix - {selected.name}",
    )
    return report


def load_bundle(path: str | Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    required = {"pipeline", "threshold", "feature_columns", "model_name"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError("invalid SensorGuard model bundle")
    if tuple(bundle["feature_columns"]) != FEATURE_COLUMNS:
        raise ValueError("model bundle feature schema does not match this version")
    return bundle


def predict_rows(bundle: dict[str, Any], rows: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(FEATURE_COLUMNS) - set(rows.columns))
    if missing:
        raise ValueError(f"prediction rows are missing features: {missing}")
    features = rows.loc[:, list(FEATURE_COLUMNS)]
    probabilities = bundle["pipeline"].predict_proba(features)[:, 1]
    threshold = float(bundle["threshold"])
    return pd.DataFrame(
        {
            "failure_probability": probabilities,
            "predicted_failure": (probabilities >= threshold).astype(np.int64),
        }
    )
