"""Statistical drift detection for streaming telemetry."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FeatureDriftReport:
    feature_name: str
    ks_statistic: float
    p_value: float
    psi: float
    drift_detected: bool
    drift_level: str  # "none", "moderate", "severe"


@dataclass(frozen=True)
class DatasetDriftReport:
    n_features: int
    drifted_features_count: int
    dataset_drift_detected: bool
    feature_reports: list[FeatureDriftReport]


def compute_ks_two_sample(sample1: np.ndarray, sample2: np.ndarray) -> tuple[float, float]:
    """Compute 2-sample Kolmogorov-Smirnov test statistic and approximate p-value."""
    n1 = len(sample1)
    n2 = len(sample2)
    if n1 == 0 or n2 == 0:
        raise ValueError("Samples cannot be empty")

    data1 = np.sort(sample1)
    data2 = np.sort(sample2)

    data_all = np.concatenate([data1, data2])
    cdf1 = np.searchsorted(data1, data_all, side="right") / n1
    cdf2 = np.searchsorted(data2, data_all, side="right") / n2

    d_stat = float(np.max(np.abs(cdf1 - cdf2)))

    # Asymptotic approximation for p-value
    en = np.sqrt(n1 * n2 / (n1 + n2))
    lambda_val = (en + 0.12 + 0.11 / en) * d_stat
    
    # Kolmogorov distribution survival function approximation
    if lambda_val <= 0.0:
        p_val = 1.0
    elif lambda_val > 3.0:
        p_val = 0.0
    else:
        # Sum first few terms of Kolmogorov series
        terms = [2.0 * ((-1) ** (j - 1)) * np.exp(-2.0 * (j**2) * (lambda_val**2)) for j in range(1, 10)]
        p_val = float(np.clip(sum(terms), 0.0, 1.0))

    return d_stat, p_val


def compute_psi(reference: np.ndarray, current: np.ndarray, num_bins: int = 10) -> float:
    """Compute Population Stability Index (PSI) between reference and current feature distributions."""
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(reference, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    # Avoid zero divisions with Laplace smoothing
    ref_props = (ref_counts + 1e-4) / (len(reference) + 1e-4 * num_bins)
    cur_props = (cur_counts + 1e-4) / (len(current) + 1e-4 * num_bins)

    psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
    return float(max(0.0, psi))


class DriftDetector:
    """Monitors telemetry batches against reference training baseline."""

    def __init__(self, p_value_threshold: float = 0.05, psi_threshold: float = 0.20) -> None:
        self.p_value_threshold = p_value_threshold
        self.psi_threshold = psi_threshold

    def analyze_dataset_drift(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> DatasetDriftReport:
        n_features = reference_data.shape[1]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        reports: list[FeatureDriftReport] = []
        drifted_count = 0

        for col_idx in range(n_features):
            ref_col = reference_data[:, col_idx]
            cur_col = current_data[:, col_idx]

            ks_stat, p_val = compute_ks_two_sample(ref_col, cur_col)
            psi_val = compute_psi(ref_col, cur_col)

            drifted = p_val < self.p_value_threshold or psi_val > self.psi_threshold
            if psi_val > 0.25 or p_val < 0.001:
                level = "severe"
            elif drifted:
                level = "moderate"
            else:
                level = "none"

            if drifted:
                drifted_count += 1

            reports.append(
                FeatureDriftReport(
                    feature_name=feature_names[col_idx],
                    ks_statistic=round(ks_stat, 4),
                    p_value=round(p_val, 4),
                    psi=round(psi_val, 4),
                    drift_detected=drifted,
                    drift_level=level,
                )
            )

        # Dataset drifted if >= 30% of features drift
        dataset_drifted = (drifted_count / n_features) >= 0.30

        return DatasetDriftReport(
            n_features=n_features,
            drifted_features_count=drifted_count,
            dataset_drift_detected=dataset_drifted,
            feature_reports=reports,
        )
