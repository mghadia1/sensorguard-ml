"""Dataset validation and deterministic train/validation/test splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ID_COLUMNS = ("UDI", "Product ID")
FEATURE_COLUMNS = (
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
)
TARGET_COLUMN = "Machine failure"
FAILURE_MODE_COLUMNS = ("TWF", "HDF", "PWF", "OSF", "RNF")
EXPECTED_COLUMNS = ID_COLUMNS + FEATURE_COLUMNS + (TARGET_COLUMN,) + FAILURE_MODE_COLUMNS


@dataclass(frozen=True)
class DatasetSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def validate_dataset(frame: pd.DataFrame) -> dict[str, object]:
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"missing required columns: {missing_columns}")
    if len(frame) == 0:
        raise ValueError("dataset cannot be empty")
    if frame[list(EXPECTED_COLUMNS)].isnull().any().any():
        raise ValueError("required columns contain missing values")
    if set(frame[TARGET_COLUMN].unique()) - {0, 1}:
        raise ValueError("Machine failure must contain only 0 and 1")
    if set(frame["Type"].unique()) - {"L", "M", "H"}:
        raise ValueError("Type contains an unknown product category")
    numeric_columns = [column for column in FEATURE_COLUMNS if column != "Type"]
    numeric_values = frame[numeric_columns].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(numeric_values)):
        raise ValueError("numeric feature columns must be finite")
    if frame["UDI"].duplicated().any():
        raise ValueError("UDI must be unique")
    return {
        "rows": int(len(frame)),
        "feature_count": len(FEATURE_COLUMNS),
        "positive_rows": int(frame[TARGET_COLUMN].sum()),
        "positive_rate": float(frame[TARGET_COLUMN].mean()),
        "missing_values": 0,
        "duplicate_udi": 0,
        "excluded_id_columns": list(ID_COLUMNS),
        "excluded_failure_mode_columns": list(FAILURE_MODE_COLUMNS),
    }


def load_dataset(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    validate_dataset(frame)
    return frame


def split_dataset(
    frame: pd.DataFrame,
    *,
    random_state: int = 42,
) -> DatasetSplits:
    """Create 60/20/20 stratified splits before fitting preprocessing."""

    validate_dataset(frame)
    train, temporary = train_test_split(
        frame,
        test_size=0.40,
        random_state=random_state,
        stratify=frame[TARGET_COLUMN],
    )
    validation, test = train_test_split(
        temporary,
        test_size=0.50,
        random_state=random_state,
        stratify=temporary[TARGET_COLUMN],
    )
    return DatasetSplits(
        train=train.sort_values("UDI").reset_index(drop=True),
        validation=validation.sort_values("UDI").reset_index(drop=True),
        test=test.sort_values("UDI").reset_index(drop=True),
    )


def feature_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return frame.loc[:, list(FEATURE_COLUMNS)], frame.loc[:, TARGET_COLUMN].astype(np.int64)


def split_summary(splits: DatasetSplits) -> dict[str, dict[str, int | float]]:
    summary: dict[str, dict[str, int | float]] = {}
    for name, frame in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        summary[name] = {
            "rows": int(len(frame)),
            "positive_rows": int(frame[TARGET_COLUMN].sum()),
            "positive_rate": float(frame[TARGET_COLUMN].mean()),
        }
    return summary
