from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

ID_COL = "id"
TARGET = "is_canceled"
CATEGORICAL = ("meal_package", "room_category", "booking_channel")
MODE_FILLED = ("num_adults", "num_children", "needs_parking", "special_requests")
REGRESSION_FILLED = ("days_before_arrival", "nightly_rate_idr")
MISSING_INDICATORS = (*MODE_FILLED, *REGRESSION_FILLED, "meal_package", "room_category")
LOG_SCALED = (
    "days_before_arrival",
    "nightly_rate_idr",
    "prior_cancellations",
    "prior_completed_bookings",
)
NUMERIC = (
    "num_adults",
    "num_children",
    "weekend_nights",
    "weekday_nights",
    "days_before_arrival",
    "arrival_year",
    "arrival_month",
    "arrival_day",
    "prior_cancellations",
    "prior_completed_bookings",
    "nightly_rate_idr",
    "special_requests",
)
BINARY = ("needs_parking", "is_returning_guest")
UNKNOWN = "Unknown"
YEAR_REFERENCE = 2023.0


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={ID_COL: str})


def clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    columns = [c for c in df.columns if c != ID_COL]
    return df.drop_duplicates(subset=columns, keep="first").reset_index(drop=True)


def target_array(df: pd.DataFrame) -> IntArray:
    return np.asarray(df[TARGET].to_numpy(), dtype=np.int64)


def _column(df: pd.DataFrame, name: str) -> FloatArray:
    return np.asarray(df[name].to_numpy(dtype=np.float64), dtype=np.float64)


def _ridge_fit(design: FloatArray, y: FloatArray, ridge: float) -> FloatArray:
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y)


class Preprocessor:
    def __init__(self, ridge: float = 1.0) -> None:
        self.ridge = ridge
        self.modes_: dict[str, float] = {}
        self.categories_: dict[str, list[str]] = {}
        self.regressors_: dict[str, tuple[FloatArray, float, float]] = {}
        self.mean_: FloatArray = np.zeros(0)
        self.std_: FloatArray = np.ones(0)
        self.feature_names_: list[str] = []
        self.numeric_mask_: BoolArray = np.zeros(0, dtype=bool)

    def fit(self, df: pd.DataFrame) -> Self:
        self.modes_ = {c: float(df[c].mode().iloc[0]) for c in MODE_FILLED}
        self.categories_ = {
            c: [*sorted(str(v) for v in df[c].dropna().unique()), UNKNOWN]
            for c in CATEGORICAL
        }
        design = self._design(df)
        self.regressors_ = {}
        for name in REGRESSION_FILLED:
            seen = df[name].notna().to_numpy()
            target = np.log1p(_column(df, name)[seen])
            weights = _ridge_fit(design[seen], target, self.ridge)
            self.regressors_[name] = (weights, float(target.min()), float(target.max()))
        numeric = self._numeric_block(self._fill(df))
        std = numeric.std(axis=0)
        self.mean_ = numeric.mean(axis=0)
        self.std_ = np.where(std > 0.0, std, 1.0)
        self.feature_names_ = [
            *NUMERIC,
            *BINARY,
            *(f"{c}_missing" for c in MISSING_INDICATORS),
            *(f"{c}={v}" for c in CATEGORICAL for v in self.categories_[c]),
        ]
        self.numeric_mask_ = np.arange(len(self.feature_names_)) < len(NUMERIC)
        return self

    def transform(self, df: pd.DataFrame) -> FloatArray:
        filled = self._fill(df)
        numeric = (self._numeric_block(filled) - self.mean_) / self.std_
        columns = [_column(filled, c) for c in BINARY]
        columns += [_column(filled, f"{c}_missing") for c in MISSING_INDICATORS]
        columns += [
            (filled[c] == v).to_numpy(dtype=np.float64)
            for c in CATEGORICAL
            for v in self.categories_[c]
        ]
        return np.column_stack([numeric, *columns])

    def fit_transform(self, df: pd.DataFrame) -> FloatArray:
        return self.fit(df).transform(df)

    def _design(self, df: pd.DataFrame) -> FloatArray:
        angle = 2.0 * np.pi * _column(df, "arrival_month") / 12.0
        channels = [
            (df["booking_channel"] == name).to_numpy(dtype=np.float64)
            for name in self.categories_["booking_channel"][:-1]
        ]
        columns = [
            np.ones(len(df)),
            _column(df, "weekend_nights"),
            _column(df, "weekday_nights"),
            np.sin(angle),
            np.cos(angle),
            _column(df, "arrival_year") - YEAR_REFERENCE,
            _column(df, "is_returning_guest"),
            np.log1p(_column(df, "prior_cancellations")),
            np.log1p(_column(df, "prior_completed_bookings")),
            *channels,
        ]
        return np.column_stack(columns)

    def _fill(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for name in MISSING_INDICATORS:
            out[f"{name}_missing"] = out[name].isna().astype(float)
        for name, mode in self.modes_.items():
            out[name] = out[name].fillna(mode)
        design = self._design(df)
        for name, (weights, low, high) in self.regressors_.items():
            gaps = out[name].isna().to_numpy()
            if gaps.any():
                predicted = np.clip(design[gaps] @ weights, low, high)
                out.loc[gaps, name] = np.expm1(predicted)
        for name in CATEGORICAL:
            out[name] = out[name].fillna(UNKNOWN)
        return out

    @staticmethod
    def _numeric_block(filled: pd.DataFrame) -> FloatArray:
        columns = [
            np.log1p(np.maximum(_column(filled, c), 0.0))
            if c in LOG_SCALED
            else _column(filled, c)
            for c in NUMERIC
        ]
        return np.column_stack(columns)
