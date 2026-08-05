"""Reproducible CPU-versus-CUDA XGBoost benchmark for a Colab GPU runtime."""

from __future__ import annotations

import itertools
import json
import math
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from .data import DatasetSplits, feature_target
from .modeling import make_preprocessor, select_threshold, xgboost_parameters

#: Threshold frozen by the August 2026 XGBoost comparison. Agreement between the
#: CPU and CUDA models only matters relative to a decision boundary: two
#: probabilities either side of this produce opposite predictions.
FROZEN_DECISION_THRESHOLD = 0.66

#: Enumerate every split below this many; above it, sample.
EXACT_PERMUTATION_LIMIT = 200_000
MONTE_CARLO_PERMUTATIONS = 100_000

#: Guards float equality when counting permutations at least as extreme.
_TIE_TOLERANCE = 1e-12


def permutation_p_value(
    cpu: list[float],
    cuda: list[float],
    *,
    alternative: str = "cuda_faster",
    random_state: int = 42,
) -> tuple[float, bool, int]:
    """Exact one-sided permutation test on the median difference when the split
    count is tractable, otherwise a seeded Monte Carlo approximation.

    Returns (p_value, exact: bool, n_permutations: int).

    Under the null hypothesis the device label is arbitrary, so every way of
    relabelling the pooled timings is equally likely. The p-value is the share of
    relabellings whose median difference is at least as extreme as the observed
    one. This assumes nothing about normality, which matters because fit times on
    a shared Colab CPU are strongly right-skewed.
    """
    if alternative != "cuda_faster":
        raise ValueError(f"unsupported alternative: {alternative!r}")
    if len(cpu) < 1 or len(cuda) < 1:
        raise ValueError("both device timing lists must be non-empty")

    cpu_array = np.asarray(cpu, dtype=float)
    cuda_array = np.asarray(cuda, dtype=float)
    if not np.all(np.isfinite(cpu_array)) or not np.all(np.isfinite(cuda_array)):
        raise ValueError("device timing lists must contain only finite values")

    pooled = np.concatenate((cpu_array, cuda_array))
    n_cpu, n_total = len(cpu_array), len(pooled)
    observed = float(np.median(cpu)) - float(np.median(cuda))

    total_splits = math.comb(n_total, n_cpu)
    exact = total_splits <= EXACT_PERMUTATION_LIMIT
    n_permutations = total_splits if exact else MONTE_CARLO_PERMUTATIONS

    # A pre-specified no-difference guard makes the public contract explicit:
    # samples with the same multiset contain no directional evidence, regardless
    # of how ties are counted by the median permutation statistic.
    if len(cpu_array) == len(cuda_array) and np.array_equal(
        np.sort(cpu_array), np.sort(cuda_array)
    ):
        return 1.0, exact, int(n_permutations)

    if exact:
        indices = np.arange(n_total)
        rows = []
        for combination in itertools.combinations(range(n_total), n_cpu):
            mask = np.zeros(n_total, dtype=bool)
            mask[list(combination)] = True
            rows.append(np.concatenate([indices[mask], indices[~mask]]))
        index_matrix = np.asarray(rows)
    else:
        rng = np.random.default_rng(random_state)
        index_matrix = np.argsort(
            rng.random((MONTE_CARLO_PERMUTATIONS, n_total)), axis=1
        )

    permuted = pooled[index_matrix]
    differences = np.median(permuted[:, :n_cpu], axis=1) - np.median(
        permuted[:, n_cpu:], axis=1
    )
    at_least_as_extreme = int(np.count_nonzero(differences >= observed - _TIE_TOLERANCE))

    if exact:
        p_value = at_least_as_extreme / n_permutations
    else:
        # (count + 1) / (n + 1) keeps a sampled p-value away from an
        # unsupportable exact zero; the floor is 1 / (n + 1).
        p_value = (at_least_as_extreme + 1) / (n_permutations + 1)
    return float(p_value), exact, int(n_permutations)


def _spread_ratio(runs: list[float]) -> float:
    """max / min — how far the slowest repeat sat from the fastest."""
    return float(max(runs) / min(runs))


def _stdev(runs: list[float]) -> float:
    return float(statistics.stdev(runs)) if len(runs) > 1 else 0.0


def _timed_fit(
    features: Any,
    labels: Any,
    *,
    device: str,
    random_state: int,
    scale_pos_weight: float,
    repeats: int,
) -> tuple[XGBClassifier, list[float], float]:
    """Fit ``repeats`` times after one discarded warm-up.

    The warm-up exists because the first CUDA fit pays for context creation. In
    the August 2026 five-repeat run the first CUDA fit took 0.731 s against about
    0.417 s for the other four — that is setup cost, not compute, and leaving it
    in the sample drags the median.
    """

    def _fit_once() -> tuple[XGBClassifier, float]:
        model = XGBClassifier(
            **xgboost_parameters(
                device=device,
                random_state=random_state,
                scale_pos_weight=scale_pos_weight,
            )
        )
        started = time.perf_counter()
        model.fit(features, labels)
        return model, time.perf_counter() - started

    _, warmup_seconds = _fit_once()

    durations: list[float] = []
    model: XGBClassifier | None = None
    for _ in range(repeats):
        model, elapsed = _fit_once()
        durations.append(elapsed)
    assert model is not None
    return model, durations, warmup_seconds


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


def disagreement_at_threshold(
    cpu_probabilities: Any, cuda_probabilities: Any, threshold: float
) -> dict[str, Any]:
    """Rows where the two devices land on opposite sides of the decision boundary.

    A maximum absolute probability difference says nothing on its own: a 0.28 gap
    is harmless at 0.05 versus 0.33 and decisive at 0.55 versus 0.83. This counts
    the rows where the two models would actually disagree.
    """
    cpu_values = np.asarray(cpu_probabilities).reshape(-1)
    cuda_values = np.asarray(cuda_probabilities).reshape(-1)
    if len(cpu_values) != len(cuda_values):
        raise ValueError(
            "probability arrays must have the same number of rows: "
            f"{len(cpu_values)} CPU vs {len(cuda_values)} CUDA"
        )
    cpu_predictions = cpu_values >= threshold
    cuda_predictions = cuda_values >= threshold
    disagreeing = int(np.count_nonzero(cpu_predictions != cuda_predictions))
    total = int(len(cpu_predictions))
    return {
        "threshold": float(threshold),
        "rows": total,
        "disagreeing_rows": disagreeing,
        "disagreeing_fraction": (disagreeing / total) if total else 0.0,
    }


def run_gpu_benchmark(
    splits: DatasetSplits,
    output_path: str | Path,
    *,
    random_state: int = 42,
    repeats: int = 15,
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

    cpu_model, cpu_fit_seconds, cpu_warmup = _timed_fit(
        transformed_train,
        train_labels,
        device="cpu",
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
        repeats=repeats,
    )
    try:
        cuda_model, cuda_fit_seconds, cuda_warmup = _timed_fit(
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
    p_value, exact, n_permutations = permutation_p_value(
        cpu_fit_seconds, cuda_fit_seconds, random_state=random_state
    )

    # Same validation-only protocol as the CPU sweep, run independently on each
    # device's probabilities. If the two disagree, that is the finding.
    cpu_selected_threshold = select_threshold(validation_labels, cpu_probabilities)
    cuda_selected_threshold = select_threshold(validation_labels, cuda_probabilities)

    agreement = {
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
        "disagreement_at_threshold": disagreement_at_threshold(
            cpu_probabilities, cuda_probabilities, FROZEN_DECISION_THRESHOLD
        ),
        "cpu_selected_threshold": float(cpu_selected_threshold),
        "cuda_selected_threshold": float(cuda_selected_threshold),
        "note": (
            "These are two different models, not one model on two devices. "
            "XGBoost's CPU and CUDA hist implementations sketch quantiles "
            "differently, so agreement is measured rather than assumed."
        ),
    }

    report: dict[str, Any] = {
        "status": "verified_cuda_run",
        "protocol": {
            "purpose": "CPU versus CUDA implementation agreement and timing",
            "dataset_partition": "train for fitting; validation for agreement metrics",
            "official_test_evaluated": False,
            "preprocessing_fit_on": "training split only",
            "random_state": random_state,
            "repeats": repeats,
            "warmup_fits_discarded_per_device": 1,
            "frozen_decision_threshold": FROZEN_DECISION_THRESHOLD,
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
            "cpu_fit_stdev": _stdev(cpu_fit_seconds),
            "cuda_fit_stdev": _stdev(cuda_fit_seconds),
            "cpu_fit_spread_ratio": _spread_ratio(cpu_fit_seconds),
            "cuda_fit_spread_ratio": _spread_ratio(cuda_fit_seconds),
            "speedup_p_value": p_value,
            "speedup_test_exact": exact,
            "speedup_test_permutations": n_permutations,
            "warmup_seconds": {"cpu": float(cpu_warmup), "cuda": float(cuda_warmup)},
        },
        "validation_agreement": agreement,
        # Deprecated alias, retained for one release so evidence files written
        # against the old key still verify.
        "validation_parity": agreement,
        "interpretation": (
            "A speedup above 1.0 means CUDA was faster by median fit time. Read it "
            "beside speedup_p_value: on this small tabular dataset the two timing "
            "distributions overlap heavily, and a median ratio alone does not "
            "establish a difference."
        ),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
