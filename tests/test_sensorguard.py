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
from sensorguard.evidence import verify_cuda_evidence
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
    def test_the_underpowered_august_evidence_is_now_rejected(self) -> None:
        """The five-repeat run is kept as history but no longer verifies.

        Its own numbers are unchanged and internally consistent; five repeats
        per device simply cannot separate a noisy shared CPU from the GPU.
        """
        with self.assertRaisesRegex(ValueError, "5 timed runs per device"):
            verify_cuda_evidence("docs/evidence/cuda-colab-t4-report.json")

    def test_cuda_evidence_rejects_tampered_speedup(self) -> None:
        payload = make_cuda_evidence()
        payload["timing_seconds"]["cpu_over_cuda_speedup"] = 99.0
        with self.assertRaisesRegex(ValueError, "speedup"):
            verify_written_evidence(payload)

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


# --------------------------------------------------------------------------
# CUDA benchmark statistics
# --------------------------------------------------------------------------

import statistics

from sensorguard.gpu_benchmark import (
    disagreement_at_threshold,
    permutation_p_value,
)
import sensorguard.gpu_benchmark as _gb
from sensorguard.evidence import MINIMUM_REPEATS_PER_DEVICE, verify_cuda_evidence


# The August 4, 2026 Colab T4 run. Pins the implementation to a published number.
AUGUST_CPU_RUNS = [0.368, 0.806, 0.855, 1.105, 1.580]
AUGUST_CUDA_RUNS = [0.413, 0.416, 0.418, 0.419, 0.731]


class PermutationTestTests(unittest.TestCase):
    def test_reproduces_the_published_august_p_value(self):
        """Known answer: the underpowered 5v5 run gives exactly 15/252."""
        p_value, exact, permutations = permutation_p_value(
            AUGUST_CPU_RUNS, AUGUST_CUDA_RUNS
        )
        self.assertTrue(exact)
        self.assertEqual(permutations, 252)
        self.assertAlmostEqual(p_value, 0.0595, places=4)
        self.assertEqual(round(p_value * 252), 15)

    def test_identical_inputs_give_no_evidence(self):
        """Identical samples cannot support 'CUDA is faster'."""
        p_value, _, _ = permutation_p_value([1.0, 2, 3, 4, 5], [1.0, 2, 3, 4, 5])
        self.assertEqual(p_value, 1.0)

    def test_perfectly_separated_inputs_hit_the_attainable_floor(self):
        """6/252, not 1/252 — the median statistic ties at the extreme."""
        p_value, _, permutations = permutation_p_value(
            [10.0, 11, 12, 13, 14], [1.0, 2, 3, 4, 5]
        )
        self.assertEqual(permutations, 252)
        self.assertAlmostEqual(p_value, 6 / 252, places=6)

    def test_exact_and_monte_carlo_paths_agree(self):
        cpu = [0.9, 1.0, 1.1, 1.2, 1.3]
        cuda = [0.4, 0.5, 0.6, 0.7, 0.8]
        exact_p, is_exact, _ = permutation_p_value(cpu, cuda)
        self.assertTrue(is_exact)

        original = _gb.EXACT_PERMUTATION_LIMIT
        _gb.EXACT_PERMUTATION_LIMIT = 1
        try:
            sampled_p, is_sampled_exact, permutations = permutation_p_value(cpu, cuda)
        finally:
            _gb.EXACT_PERMUTATION_LIMIT = original
        self.assertFalse(is_sampled_exact)
        self.assertEqual(permutations, _gb.MONTE_CARLO_PERMUTATIONS)
        self.assertLess(abs(exact_p - sampled_p), 0.01)

    def test_monte_carlo_is_reproducible_from_the_seed(self):
        original = _gb.EXACT_PERMUTATION_LIMIT
        _gb.EXACT_PERMUTATION_LIMIT = 1
        try:
            first, _, _ = permutation_p_value([3.0, 4, 5], [1.0, 2, 3], random_state=7)
            second, _, _ = permutation_p_value([3.0, 4, 5], [1.0, 2, 3], random_state=7)
        finally:
            _gb.EXACT_PERMUTATION_LIMIT = original
        self.assertEqual(first, second)

    def test_unknown_alternative_is_rejected(self):
        with self.assertRaises(ValueError):
            permutation_p_value([1.0], [1.0], alternative="two_sided")


class ThresholdDisagreementTests(unittest.TestCase):
    def test_identical_probabilities_never_disagree(self):
        probabilities = [0.1, 0.5, 0.66, 0.9]
        result = disagreement_at_threshold(probabilities, probabilities, 0.66)
        self.assertEqual(result["disagreeing_rows"], 0)
        self.assertEqual(result["disagreeing_fraction"], 0.0)

    def test_counts_rows_that_straddle_the_boundary(self):
        """A 0.28 gap matters only relative to the decision threshold."""
        cpu = [0.55, 0.10, 0.80]
        cuda = [0.83, 0.38, 0.90]  # only the first crosses 0.66
        result = disagreement_at_threshold(cpu, cuda, 0.66)
        self.assertEqual(result["disagreeing_rows"], 1)
        self.assertEqual(result["rows"], 3)
        self.assertAlmostEqual(result["disagreeing_fraction"], 1 / 3)


def make_cuda_evidence(
    *, repeats: int = 15, random_state: int = 42, include_source_hash: bool = True
) -> dict:
    """An internally consistent CUDA evidence payload with enough repeats to verify.

    Built from the raw runs so that every published statistic is derived, never
    typed — which is the property the verifier exists to check.
    """
    cpu_runs = [0.80 + 0.01 * index for index in range(repeats)]
    cuda_runs = [0.41 + 0.001 * index for index in range(repeats)]
    p_value, exact, permutations = permutation_p_value(
        cpu_runs, cuda_runs, random_state=random_state
    )
    cpu_median = float(statistics.median(cpu_runs))
    cuda_median = float(statistics.median(cuda_runs))
    agreement = {
        "cpu_average_precision": 0.7769,
        "cuda_average_precision": 0.7613,
        "cpu_roc_auc": 0.9772,
        "cuda_roc_auc": 0.9670,
        "maximum_absolute_probability_difference": 0.2762,
        "disagreement_at_threshold": {
            "threshold": 0.66,
            "rows": 2000,
            "disagreeing_rows": 11,
            "disagreeing_fraction": 11 / 2000,
        },
        "cpu_selected_threshold": 0.66,
        "cuda_selected_threshold": 0.66,
    }
    payload = {
        "status": "verified_cuda_run",
        "protocol": {
            "official_test_evaluated": False,
            "random_state": random_state,
            "repeats": repeats,
            "warmup_fits_discarded_per_device": 1,
            "frozen_decision_threshold": 0.66,
            "xgboost": {"device": "cuda"},
        },
        "environment": {
            "gpu": "Tesla T4",
            "xgboost": "3.3.0",
            "xgboost_build_info": {"USE_CUDA": True},
        },
        "rows": {"train": 6000, "validation": 2000, "test_evaluated": 0},
        "timing_seconds": {
            "cpu_fit_runs": cpu_runs,
            "cuda_fit_runs": cuda_runs,
            "cpu_fit_median": cpu_median,
            "cuda_fit_median": cuda_median,
            "cpu_over_cuda_speedup": cpu_median / cuda_median,
            "cpu_fit_stdev": statistics.stdev(cpu_runs),
            "cuda_fit_stdev": statistics.stdev(cuda_runs),
            "cpu_fit_spread_ratio": max(cpu_runs) / min(cpu_runs),
            "cuda_fit_spread_ratio": max(cuda_runs) / min(cuda_runs),
            "speedup_p_value": p_value,
            "speedup_test_exact": exact,
            "speedup_test_permutations": permutations,
            "warmup_seconds": {"cpu": 1.9, "cuda": 0.73},
        },
        "validation_agreement": agreement,
    }
    if include_source_hash:
        payload["source_file_sha256"] = "0" * 64
    return payload


def verify_written_evidence(payload: dict):
    with tempfile.TemporaryDirectory() as directory:
        report = Path(directory) / "report.json"
        report.write_text(json.dumps(payload), encoding="utf-8")
        return verify_cuda_evidence(report)


class CudaEvidenceVerifierTests(unittest.TestCase):
    def test_a_well_formed_fifteen_repeat_report_verifies(self):
        report = verify_written_evidence(make_cuda_evidence())
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["official_test_rows_evaluated"], 0)
        self.assertEqual(report["repeats"], 15)
        self.assertIn("speedup_p_value", report)

    def test_a_fresh_benchmark_report_without_archival_hash_verifies(self):
        report = verify_written_evidence(make_cuda_evidence(include_source_hash=False))
        self.assertEqual(report["status"], "verified")
        self.assertIsNone(report["source_file_sha256"])

    def test_a_hand_edited_p_value_is_rejected(self):
        """The p-value is recomputed from the raw runs, not trusted."""
        payload = make_cuda_evidence()
        payload["timing_seconds"]["speedup_p_value"] = 0.001
        with self.assertRaisesRegex(ValueError, "p-value"):
            verify_written_evidence(payload)

    def test_a_hand_edited_stdev_is_rejected(self):
        payload = make_cuda_evidence()
        payload["timing_seconds"]["cpu_fit_stdev"] = 0.0001
        with self.assertRaisesRegex(ValueError, "stdev"):
            verify_written_evidence(payload)

    def test_a_hand_edited_spread_ratio_is_rejected(self):
        payload = make_cuda_evidence()
        payload["timing_seconds"]["cuda_fit_spread_ratio"] = 1.0
        with self.assertRaisesRegex(ValueError, "spread ratio"):
            verify_written_evidence(payload)

    def test_fewer_than_ten_runs_per_device_is_rejected_by_count(self):
        payload = make_cuda_evidence(repeats=9)
        with self.assertRaisesRegex(ValueError, "9 timed runs per device"):
            verify_written_evidence(payload)
        self.assertEqual(MINIMUM_REPEATS_PER_DEVICE, 10)

    def test_a_warmup_fit_left_in_the_timed_runs_is_rejected(self):
        payload = make_cuda_evidence()
        timing = payload["timing_seconds"]
        timing["warmup_seconds"]["cuda"] = timing["cuda_fit_runs"][0]
        with self.assertRaisesRegex(ValueError, "warm-up"):
            verify_written_evidence(payload)

    def test_a_missing_disagreement_block_is_rejected(self):
        payload = make_cuda_evidence()
        del payload["validation_agreement"]["disagreement_at_threshold"]
        with self.assertRaisesRegex(ValueError, "disagreement_at_threshold"):
            verify_written_evidence(payload)

    def test_the_deprecated_parity_key_still_verifies(self):
        payload = make_cuda_evidence()
        payload["validation_parity"] = payload.pop("validation_agreement")
        self.assertEqual(verify_written_evidence(payload)["status"], "verified")
