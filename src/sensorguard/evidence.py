"""Integrity checks for the published SensorGuard CUDA benchmark evidence."""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

from .gpu_benchmark import permutation_p_value

#: A benchmark with fewer repeats than this cannot separate a noisy shared CPU
#: from a GPU. The August 2026 five-repeat run reached p = 0.0595 against a
#: floor of 0.0238 for that design — not a power ceiling, just too few repeats.
MINIMUM_REPEATS_PER_DEVICE = 10


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _finite_positive(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return number


def _unit_interval(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be between zero and one")
    return number


def verify_cuda_evidence(path: str | Path) -> dict[str, Any]:
    """Validate provenance, protocol guards, timing arithmetic, and metrics."""

    evidence_path = Path(path)
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    if payload.get("status") != "verified_cuda_run":
        raise ValueError("evidence status is not a verified CUDA run")
    source_hash = payload.get("source_file_sha256")
    if source_hash is not None and not SHA256_PATTERN.fullmatch(str(source_hash)):
        raise ValueError("source report SHA-256 is missing or malformed")

    protocol = payload["protocol"]
    rows = payload["rows"]
    environment = payload["environment"]
    if protocol.get("official_test_evaluated") is not False:
        raise ValueError("CUDA comparison must not evaluate the official test split")
    if int(rows.get("test_evaluated", -1)) != 0:
        raise ValueError("CUDA comparison reports official-test access")
    if int(rows.get("train", 0)) != 6000 or int(rows.get("validation", 0)) != 2000:
        raise ValueError("benchmark split sizes differ from the frozen protocol")
    if protocol["xgboost"].get("device") != "cuda":
        raise ValueError("published XGBoost device is not CUDA")
    if environment["xgboost_build_info"].get("USE_CUDA") is not True:
        raise ValueError("XGBoost build does not report CUDA support")

    repeats = int(protocol["repeats"])
    timing = payload["timing_seconds"]
    cpu_runs = [_finite_positive(value, "CPU fit time") for value in timing["cpu_fit_runs"]]
    cuda_runs = [
        _finite_positive(value, "CUDA fit time") for value in timing["cuda_fit_runs"]
    ]
    if len(cpu_runs) != len(cuda_runs):
        raise ValueError(
            f"device run counts differ: {len(cpu_runs)} CPU vs {len(cuda_runs)} CUDA"
        )
    if len(cpu_runs) < MINIMUM_REPEATS_PER_DEVICE:
        raise ValueError(
            f"underpowered benchmark: {len(cpu_runs)} timed runs per device, "
            f"minimum is {MINIMUM_REPEATS_PER_DEVICE}"
        )
    if repeats < 1 or len(cpu_runs) != repeats or len(cuda_runs) != repeats:
        raise ValueError(
            "timing-run counts differ from the declared repeat count: "
            f"declared {repeats}, found {len(cpu_runs)} CPU and "
            f"{len(cuda_runs)} CUDA"
        )
    if int(protocol.get("warmup_fits_discarded_per_device", 0)) != 1:
        raise ValueError("protocol must discard exactly one warm-up fit per device")

    cpu_median = float(statistics.median(cpu_runs))
    cuda_median = float(statistics.median(cuda_runs))
    if not math.isclose(cpu_median, float(timing["cpu_fit_median"]), rel_tol=1e-12):
        raise ValueError("published CPU median does not match the raw runs")
    if not math.isclose(cuda_median, float(timing["cuda_fit_median"]), rel_tol=1e-12):
        raise ValueError("published CUDA median does not match the raw runs")
    speedup = cpu_median / cuda_median
    if not math.isclose(speedup, float(timing["cpu_over_cuda_speedup"]), rel_tol=1e-12):
        raise ValueError("published CUDA speedup does not match the raw timings")

    # Recomputed, not trusted. A verifier that only rechecks the numbers it was
    # handed is a checksum; recomputing the p-value is what makes it evidence.
    for label, runs, published in (
        ("CPU", cpu_runs, timing["cpu_fit_stdev"]),
        ("CUDA", cuda_runs, timing["cuda_fit_stdev"]),
    ):
        if not math.isclose(statistics.stdev(runs), float(published), rel_tol=1e-9):
            raise ValueError(f"published {label} stdev does not match the raw runs")
    for label, runs, published in (
        ("CPU", cpu_runs, timing["cpu_fit_spread_ratio"]),
        ("CUDA", cuda_runs, timing["cuda_fit_spread_ratio"]),
    ):
        if not math.isclose(max(runs) / min(runs), float(published), rel_tol=1e-9):
            raise ValueError(f"published {label} spread ratio does not match the raw runs")

    recomputed_p, exact, permutations = permutation_p_value(
        cpu_runs, cuda_runs, random_state=int(protocol["random_state"])
    )
    if not math.isclose(recomputed_p, float(timing["speedup_p_value"]), rel_tol=1e-9):
        raise ValueError("published speedup p-value does not match the raw runs")
    if bool(timing["speedup_test_exact"]) is not exact:
        raise ValueError("published permutation test exactness does not match the raw runs")
    if int(timing["speedup_test_permutations"]) != permutations:
        raise ValueError("published permutation count does not match the raw runs")

    warmup = timing["warmup_seconds"]
    for device, runs in (("cpu", cpu_runs), ("cuda", cuda_runs)):
        value = _finite_positive(warmup[device], f"{device} warm-up time")
        if any(math.isclose(value, run, rel_tol=1e-12) for run in runs):
            raise ValueError(
                f"{device} warm-up fit appears in the timed runs; it must be discarded"
            )

    agreement = payload.get("validation_agreement", payload.get("validation_parity"))
    if agreement is None:
        raise ValueError("evidence has no validation_agreement block")
    for key in (
        "cpu_average_precision",
        "cuda_average_precision",
        "cpu_roc_auc",
        "cuda_roc_auc",
        "maximum_absolute_probability_difference",
    ):
        _unit_interval(agreement[key], key)

    disagreement = agreement.get("disagreement_at_threshold")
    if disagreement is None:
        raise ValueError("evidence does not report disagreement_at_threshold")
    disagreeing = int(disagreement["disagreeing_rows"])
    total_rows = int(disagreement["rows"])
    if total_rows != int(rows["validation"]):
        raise ValueError("threshold disagreement row count differs from the validation split")
    frozen_threshold = _unit_interval(
        protocol.get("frozen_decision_threshold"), "frozen decision threshold"
    )
    if not math.isclose(
        frozen_threshold, float(disagreement["threshold"]), rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("threshold disagreement was not measured at the frozen threshold")
    _unit_interval(agreement["cpu_selected_threshold"], "CPU selected threshold")
    _unit_interval(agreement["cuda_selected_threshold"], "CUDA selected threshold")
    if not 0 <= disagreeing <= total_rows:
        raise ValueError("threshold disagreement count is out of range")
    if not math.isclose(
        disagreeing / total_rows, float(disagreement["disagreeing_fraction"]), rel_tol=1e-9
    ):
        raise ValueError("published disagreement fraction does not match the counts")

    return {
        "status": "verified",
        "gpu": environment["gpu"],
        "xgboost": environment["xgboost"],
        "repeats": repeats,
        "cpu_fit_median_seconds": cpu_median,
        "cuda_fit_median_seconds": cuda_median,
        "cpu_over_cuda_speedup": speedup,
        "speedup_p_value": recomputed_p,
        "speedup_test_exact": exact,
        "speedup_test_permutations": permutations,
        "cpu_fit_spread_ratio": max(cpu_runs) / min(cpu_runs),
        "cuda_fit_spread_ratio": max(cuda_runs) / min(cuda_runs),
        "disagreeing_rows_at_frozen_threshold": disagreeing,
        "official_test_rows_evaluated": 0,
        "source_file_sha256": source_hash,
    }
