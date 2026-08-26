"""Tests for Conformal Prediction and Drift Detection in SensorGuard ML."""

from __future__ import annotations

import numpy as np
import pytest

from sensorguard.conformal import ConformalClassifier
from sensorguard.drift import DriftDetector, compute_ks_two_sample, compute_psi


def test_conformal_classifier_guarantees_target_coverage():
    np.random.seed(42)
    n_calib = 500
    n_test = 500
    n_classes = 3

    # Generate synthetic calibrated probabilities and true labels
    y_calib = np.random.choice(n_classes, size=n_calib)
    probas_calib = np.random.dirichlet(np.ones(n_classes), size=n_calib)
    # Boost true class proba to simulate a trained model
    probas_calib[np.arange(n_calib), y_calib] += 0.8
    probas_calib = probas_calib / probas_calib.sum(axis=1, keepdims=True)

    y_test = np.random.choice(n_classes, size=n_test)
    probas_test = np.random.dirichlet(np.ones(n_classes), size=n_test)
    probas_test[np.arange(n_test), y_test] += 0.8
    probas_test = probas_test / probas_test.sum(axis=1, keepdims=True)

    clf = ConformalClassifier(alpha=0.10)  # 90% target coverage
    clf.fit_calibration(probas_calib, y_calib)

    pred_set = clf.predict(probas_test)
    coverage = clf.evaluate_coverage(pred_set.prediction_sets, y_test)

    # Coverage should be >= target coverage (90%) within empirical tolerance
    assert coverage >= 0.88
    assert pred_set.coverage_level == 0.90


def test_conformal_classifier_requires_calibration():
    clf = ConformalClassifier(alpha=0.05)
    with pytest.raises(RuntimeError, match="must be calibrated"):
        clf.predict(np.array([[0.8, 0.2]]))


def test_ks_two_sample_detects_distribution_shift():
    np.random.seed(42)
    ref = np.random.normal(loc=10.0, scale=2.0, size=1000)
    
    # Same distribution
    identical = np.random.normal(loc=10.0, scale=2.0, size=1000)
    stat, p_val = compute_ks_two_sample(ref, identical)
    assert p_val > 0.05
    assert stat < 0.10

    # Shifted distribution
    shifted = np.random.normal(loc=14.0, scale=2.0, size=1000)
    stat_shift, p_val_shift = compute_ks_two_sample(ref, shifted)
    assert p_val_shift < 0.001
    assert stat_shift > 0.50


def test_drift_detector_flags_dataset_drift():
    np.random.seed(42)
    ref_data = np.random.normal(loc=0.0, scale=1.0, size=(500, 4))
    
    # Drift 3 of 4 features
    cur_data = np.copy(ref_data)
    cur_data[:, 0] += 3.0
    cur_data[:, 1] += 3.0
    cur_data[:, 2] += 3.0

    detector = DriftDetector()
    report = detector.analyze_dataset_drift(ref_data, cur_data)

    assert report.dataset_drift_detected is True
    assert report.drifted_features_count == 3
