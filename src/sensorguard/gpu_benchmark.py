"""Reproducible CPU-versus-CUDA XGBoost benchmark for a Colab GPU runtime."""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from .data import DatasetSplits, feature_target
from .modeling import make_preprocessor, xgboost_parameters


def _timed_fit(
    features: Any,
    labels: Any,
    *,
    device: str,
    random_state: int,
    scale_pos_weight: float,
    repeats: int,
) -> tuple[XGBClassifier, list[float]]:
    durations: list[float] = []
    model: XGBClassifier | None = None
    for _ in range(repeats):
        model = XGBClassifier(
            **xgboost_parameters(
                device=device,
                random_state=random_state,
                scale_pos_weight=scale_pos_weight,
            )
        )
        started = time.perf_counter()
        model.fit(features, labels)
        durations.append(time.perf_counter() - started)
    assert model is not None
    return model, durations


def _nvidia_smi() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(
            "No usable NVIDIA runtime was detected. In Colab choose Runtime > "
            "Change runtime type > T4 GPU, then run all cells again."
        ) from error
    value = result.stdout.strip()
    if not value:
        raise RuntimeError("nvidia-smi returned no GPU information")
    return value


def run_gpu_benchmark(
    splits: DatasetSplits,
    output_path: str | Path,
    *,
    random_state: int = 42,
    repeats: int = 3,
) -> dict[str, Any]:
    """Compare frozen CPU/CUDA XGBoost settings without touching the test split."""

    if repeats < 1:
        raise ValueError("repeats must be at least one")
    gpu = _nvidia_smi()
    train_features, train_labels = feature_target(splits.train)
    validation_features, validation_labels = feature_target(splits.validation)
    positive_count = int(np.asarray(train_labels).sum())
    if positive_count < 1:
        raise ValueError("training split must contain at least one positive example")
    scale_pos_weight = (len(train_labels) - positive_count) / positive_count

    preprocessing_started = time.perf_counter()
    preprocessor = make_preprocessor()
    transformed_train = preprocessor.fit_transform(train_features)
    transformed_validation = preprocessor.transform(validation_features)
    preprocessing_seconds = time.perf_counter() - preprocessing_started

    cpu_model, cpu_fit_seconds = _timed_fit(
        transformed_train,
        train_labels,
        device="cpu",
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
        repeats=repeats,
    )
    try:
        cuda_model, cuda_fit_seconds = _timed_fit(
            transformed_train,
            train_labels,
            device="cuda",
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
            repeats=repeats,
        )
    except xgb.core.XGBoostError as error:
        raise RuntimeError(
            "XGBoost could not train on CUDA. Confirm that Colab is using a GPU "
            "runtime and that the full xgboost package, not xgboost-cpu, is installed."
        ) from error

    cpu_probabilities = cpu_model.predict_proba(transformed_validation)[:, 1]
    cuda_probabilities = cuda_model.predict_proba(transformed_validation)[:, 1]
    cpu_median = float(np.median(cpu_fit_seconds))
    cuda_median = float(np.median(cuda_fit_seconds))
    report: dict[str, Any] = {
        "status": "verified_cuda_run",
        "protocol": {
            "purpose": "CPU versus CUDA implementation parity and timing",
            "dataset_partition": "train for fitting; validation for parity metrics",
            "official_test_evaluated": False,
            "preprocessing_fit_on": "training split only",
            "random_state": random_state,
            "repeats": repeats,
            "xgboost": xgboost_parameters(
                device="cuda",
                random_state=random_state,
                scale_pos_weight=scale_pos_weight,
            ),
        },
        "environment": {
            "gpu": gpu,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "xgboost": xgb.__version__,
            "xgboost_build_info": xgb.build_info(),
        },
        "rows": {
            "train": int(len(train_labels)),
            "validation": int(len(validation_labels)),
            "test_evaluated": 0,
        },
        "timing_seconds": {
            "shared_preprocessing": preprocessing_seconds,
            "cpu_fit_runs": cpu_fit_seconds,
            "cuda_fit_runs": cuda_fit_seconds,
            "cpu_fit_median": cpu_median,
            "cuda_fit_median": cuda_median,
            "cpu_over_cuda_speedup": cpu_median / cuda_median,
        },
        "validation_parity": {
            "cpu_average_precision": float(
                average_precision_score(validation_labels, cpu_probabilities)
            ),
            "cuda_average_precision": float(
                average_precision_score(validation_labels, cuda_probabilities)
            ),
            "cpu_roc_auc": float(roc_auc_score(validation_labels, cpu_probabilities)),
            "cuda_roc_auc": float(roc_auc_score(validation_labels, cuda_probabilities)),
            "maximum_absolute_probability_difference": float(
                np.max(np.abs(cpu_probabilities - cuda_probabilities))
            ),
        },
        "interpretation": (
            "A speedup above 1.0 means CUDA was faster. On this small tabular dataset, "
            "GPU startup and transfer overhead may make CUDA slower than CPU."
        ),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
