"""Integrity checks for the published SensorGuard CUDA benchmark evidence."""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


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
    source_hash = payload.get("source_file_sha256", "")
    if not SHA256_PATTERN.fullmatch(str(source_hash)):
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
    if repeats < 1 or len(cpu_runs) != repeats or len(cuda_runs) != repeats:
        raise ValueError("timing-run counts differ from the declared repeat count")
    cpu_median = float(statistics.median(cpu_runs))
    cuda_median = float(statistics.median(cuda_runs))
    if not math.isclose(cpu_median, float(timing["cpu_fit_median"]), rel_tol=1e-12):
        raise ValueError("published CPU median does not match the raw runs")
    if not math.isclose(cuda_median, float(timing["cuda_fit_median"]), rel_tol=1e-12):
        raise ValueError("published CUDA median does not match the raw runs")
    speedup = cpu_median / cuda_median
    if not math.isclose(speedup, float(timing["cpu_over_cuda_speedup"]), rel_tol=1e-12):
        raise ValueError("published CUDA speedup does not match the raw timings")

    parity = payload["validation_parity"]
    for key in (
        "cpu_average_precision",
        "cuda_average_precision",
        "cpu_roc_auc",
        "cuda_roc_auc",
        "maximum_absolute_probability_difference",
    ):
        _unit_interval(parity[key], key)

    return {
        "status": "verified",
        "gpu": environment["gpu"],
        "xgboost": environment["xgboost"],
        "repeats": repeats,
        "cpu_fit_median_seconds": cpu_median,
        "cuda_fit_median_seconds": cuda_median,
        "cpu_over_cuda_speedup": speedup,
        "official_test_rows_evaluated": 0,
        "source_file_sha256": source_hash,
    }
