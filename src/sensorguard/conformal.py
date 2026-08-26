"""Split Conformal Prediction for distribution-free uncertainty quantification."""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ConformalPredictionSet:
    prediction_sets: list[list[int]]
    coverage_level: float
    threshold: float


class ConformalClassifier:
    """Implements inductive Split Conformal Prediction for multi-class/binary classification.
    
    Guarantees 1 - alpha marginal coverage:
    P(Y_test in C(X_test)) >= 1 - alpha
    under the assumption of exchangeability.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha (significance level) must be in (0, 1)")
        self.alpha = alpha
        self.q_hat: float | None = None

    def fit_calibration(self, probas_calib: np.ndarray, y_calib: np.ndarray) -> float:
        """Compute conformity scores on calibration split.
        
        Score: 1 - proba(true_class)
        """
        n = len(y_calib)
        if n == 0:
            raise ValueError("Calibration set cannot be empty")

        # Non-conformity score: 1 - probability of true class
        true_class_probas = probas_calib[np.arange(n), y_calib]
        scores = 1.0 - true_class_probas

        # Conformal quantile: ceil((n + 1) * (1 - alpha)) / n
        p_val = math.ceil((n + 1) * (1.0 - self.alpha)) / n
        p_val = min(1.0, max(0.0, p_val))
        
        self.q_hat = float(np.quantile(scores, p_val, method="higher"))
        return self.q_hat

    def predict(self, probas: np.ndarray) -> ConformalPredictionSet:
        """Generate prediction sets where all classes with proba >= 1 - q_hat are included."""
        if self.q_hat is None:
            raise RuntimeError("ConformalClassifier must be calibrated before predict")

        threshold = 1.0 - self.q_hat
        prediction_sets: list[list[int]] = []

        for p_row in probas:
            included = [cls_idx for cls_idx, p in enumerate(p_row) if p >= threshold]
            if not included:
                # Fallback to argmax if empty set
                included = [int(np.argmax(p_row))]
            prediction_sets.append(included)

        return ConformalPredictionSet(
            prediction_sets=prediction_sets,
            coverage_level=1.0 - self.alpha,
            threshold=threshold,
        )

    def evaluate_coverage(self, prediction_sets: list[list[int]], y_true: np.ndarray) -> float:
        """Measure empirical test coverage."""
        covered = sum(1 for p_set, y in zip(prediction_sets, y_true) if y in p_set)
        return covered / len(y_true)
