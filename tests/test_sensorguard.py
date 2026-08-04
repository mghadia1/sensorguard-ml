from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from sensorguard.data import (
    FAILURE_MODE_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    feature_target,
    split_dataset,
    validate_dataset,
)
from sensorguard.download import file_sha256
from sensorguard.gpu_benchmark import _nvidia_smi
from sensorguard.learning import QUESTIONS, save_answers
from sensorguard.modeling import (
    binary_metrics,
    candidate_pipelines,
    load_bundle,
    predict_rows,
    select_threshold,
    train_evaluate_save,
    xgboost_parameters,
)


def make_fixture(*, row_count: int = 500, seed: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    product_type = rng.choice(np.asarray(("L", "M", "H")), size=row_count, p=(0.5, 0.3, 0.2))
    air_temperature = rng.normal(300.0, 2.0, size=row_count)
    process_temperature = air_temperature + rng.normal(10.0, 0.8, size=row_count)
    speed = rng.normal(1500.0, 170.0, size=row_count)
    torque = rng.normal(40.0, 10.0, size=row_count)
    wear = rng.uniform(0.0, 240.0, size=row_count)
    risk = (
        0.11 * (torque - 40.0)
        + 0.012 * (wear - 120.0)
        - 0.005 * (speed - 1500.0)
        + rng.normal(0.0, 0.7, size=row_count)
    )
    threshold = float(np.quantile(risk, 0.86))
    failure = (risk >= threshold).astype(np.int64)
    frame = pd.DataFrame(
        {
            "UDI": np.arange(1, row_count + 1),
            "Product ID": [f"{kind}{index:05d}" for index, kind in enumerate(product_type)],
            "Type": product_type,
            "Air temperature [K]": air_temperature,
            "Process temperature [K]": process_temperature,
            "Rotational speed [rpm]": speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": wear,
            "Machine failure": failure,
            "TWF": np.zeros(row_count, dtype=np.int64),
            "HDF": np.zeros(row_count, dtype=np.int64),
            "PWF": np.zeros(row_count, dtype=np.int64),
            "OSF": np.zeros(row_count, dtype=np.int64),
            "RNF": np.zeros(row_count, dtype=np.int64),
        }
    )
    return frame


class DataTests(unittest.TestCase):
    def test_schema_audit_and_leakage_exclusions(self) -> None:
        frame = make_fixture()
        audit = validate_dataset(frame)
        features, labels = feature_target(frame)
        self.assertEqual(audit["rows"], len(frame))
        self.assertEqual(tuple(features.columns), FEATURE_COLUMNS)
        self.assertEqual(labels.name, TARGET_COLUMN)
        self.assertTrue(set(FAILURE_MODE_COLUMNS).isdisjoint(features.columns))

    def test_missing_required_column_is_rejected(self) -> None:
        frame = make_fixture().drop(columns=["Torque [Nm]"])
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            validate_dataset(frame)

    def test_split_is_reproducible_stratified_and_disjoint(self) -> None:
        frame = make_fixture()
        first = split_dataset(frame, random_state=12)
        second = split_dataset(frame, random_state=12)
        self.assertEqual(first.train["UDI"].tolist(), second.train["UDI"].tolist())
        self.assertEqual((len(first.train), len(first.validation), len(first.test)), (300, 100, 100))
        train_ids = set(first.train["UDI"])
        validation_ids = set(first.validation["UDI"])
        test_ids = set(first.test["UDI"])
        self.assertTrue(train_ids.isdisjoint(validation_ids))
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertTrue(validation_ids.isdisjoint(test_ids))
        overall_rate = frame[TARGET_COLUMN].mean()
        for split in (first.train, first.validation, first.test):
            self.assertAlmostEqual(split[TARGET_COLUMN].mean(), overall_rate, delta=0.02)


class MetricTests(unittest.TestCase):
    def test_binary_metrics_reports_expected_confusion_matrix(self) -> None:
        labels = np.asarray((0, 0, 1, 1))
        probabilities = np.asarray((0.1, 0.8, 0.4, 0.9))
        metrics = binary_metrics(labels, probabilities, threshold=0.5)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)

    def test_threshold_selection_is_not_fixed_at_half(self) -> None:
        labels = np.asarray((0, 0, 0, 1, 1))
        probabilities = np.asarray((0.05, 0.10, 0.20, 0.25, 0.30))
        threshold = select_threshold(labels, probabilities)
        self.assertLess(threshold, 0.5)
        self.assertGreaterEqual(threshold, 0.20)


class PipelineTests(unittest.TestCase):
    def test_cuda_benchmark_rejects_a_runtime_without_nvidia(self) -> None:
        with patch("sensorguard.gpu_benchmark.subprocess.run", side_effect=FileNotFoundError):
            with self.assertRaisesRegex(RuntimeError, "Colab"):
                _nvidia_smi()

    def test_frozen_xgboost_parameters_support_cpu_and_cuda(self) -> None:
        cpu = xgboost_parameters(device="cpu", random_state=17, scale_pos_weight=12.5)
        cuda = xgboost_parameters(device="cuda", random_state=17, scale_pos_weight=12.5)
        self.assertEqual(cpu["tree_method"], "hist")
        self.assertEqual(cpu["device"], "cpu")
        self.assertEqual(cuda["device"], "cuda")
        for key in cpu:
            if key != "device":
                self.assertEqual(cpu[key], cuda[key])
        with self.assertRaisesRegex(ValueError, "device"):
            xgboost_parameters(device="mps", random_state=17, scale_pos_weight=12.5)

    def test_xgboost_candidate_is_cpu_histogram_and_imbalance_aware(self) -> None:
        candidate = candidate_pipelines(
            random_state=17, scale_pos_weight=12.5
        )["xgboost"]
        parameters = candidate.named_steps["model"].get_params()
        self.assertEqual(parameters["tree_method"], "hist")
        self.assertEqual(parameters["device"], "cpu")
        self.assertEqual(parameters["scale_pos_weight"], 12.5)
        self.assertEqual(parameters["random_state"], 17)

    def test_training_saves_reusable_bundle_and_test_evidence(self) -> None:
        splits = split_dataset(make_fixture(row_count=700), random_state=5)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            report = train_evaluate_save(splits, output, random_state=5)
            bundle = load_bundle(output / "model.joblib")
            predictions = predict_rows(bundle, splits.test.head(4))
            saved = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, report)
            self.assertIn(report["selected_model"], report["validation_candidates"])
            self.assertIn("xgboost", report["validation_candidates"])
            self.assertGreater(report["test_metrics"]["average_precision"], 0.70)
            self.assertEqual(len(predictions), 4)
            self.assertTrue((output / "test_predictions.csv").exists())
            self.assertTrue((output / "validation_model_comparison.svg").exists())
            self.assertTrue((output / "test_confusion_matrix.svg").exists())
            importance = pd.read_csv(output / "validation_permutation_importance.csv")
            self.assertEqual(set(importance["feature"]), set(FEATURE_COLUMNS))
            self.assertEqual(len(report["validation_permutation_importance"]), len(FEATURE_COLUMNS))

    def test_file_hash_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sample.txt"
            path.write_text("sensor\n", encoding="utf-8")
            self.assertEqual(
                file_sha256(path),
                "16dab8f41e9b1f415f75cedcf0b91cc9a907a3a96c83fb279416e9114a082d29",
            )


class LearningCheckTests(unittest.TestCase):
    def test_learning_answers_are_saved_for_review(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "answers.json"
            payload = save_answers([f"answer {index}" for index in range(len(QUESTIONS))], output)
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved, payload)
            self.assertEqual(saved["status"], "awaiting_review")
            self.assertEqual(len(saved["responses"]), len(QUESTIONS))

    def test_empty_learning_answer_is_rejected(self) -> None:
        answers = ["answer"] * len(QUESTIONS)
        answers[2] = "  "
        with self.assertRaisesRegex(ValueError, "non-empty"):
            save_answers(answers, "unused.json")


if __name__ == "__main__":
    unittest.main()
