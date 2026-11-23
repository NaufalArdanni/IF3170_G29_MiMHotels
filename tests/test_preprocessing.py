import numpy as np
import pandas as pd

from ronaldohotel.data.preprocessing import (
    NUMERIC,
    Preprocessor,
    clean_training_data,
)

RANGES = {
    "num_adults": (1, 4),
    "num_children": (0, 3),
    "weekend_nights": (0, 3),
    "weekday_nights": (0, 6),
    "needs_parking": (0, 2),
    "days_before_arrival": (0, 300),
    "arrival_year": (2023, 2025),
    "arrival_month": (1, 13),
    "arrival_day": (1, 29),
    "is_returning_guest": (0, 2),
    "prior_cancellations": (0, 3),
    "prior_completed_bookings": (0, 5),
    "nightly_rate_idr": (0, 4_000_000),
    "special_requests": (0, 4),
}
WITH_GAPS = (
    "num_adults",
    "num_children",
    "needs_parking",
    "days_before_arrival",
    "nightly_rate_idr",
    "special_requests",
    "meal_package",
    "room_category",
)


def _frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict[str, object] = {"id": [f"b{i}" for i in range(n)]}
    for name, (low, high) in RANGES.items():
        data[name] = rng.integers(low, high, n).astype(float)
    data["meal_package"] = rng.choice(["Breakfast", "Room Only", "Half Board"], n)
    data["room_category"] = rng.choice(["Type A", "Type B", "Type G"], n)
    data["booking_channel"] = rng.choice(["Online Travel Agent", "Corporate"], n)
    data["is_canceled"] = rng.integers(0, 2, n)
    df = pd.DataFrame(data)
    for name in WITH_GAPS:
        df.loc[rng.random(n) < 0.1, name] = np.nan
    return df


def test_transform_has_no_missing_values() -> None:
    prep = Preprocessor()
    x = prep.fit_transform(_frame(300, seed=0))
    assert x.shape == (300, len(prep.feature_names_))
    assert not np.isnan(x).any()
    np.testing.assert_allclose(x[:, : len(NUMERIC)].mean(axis=0), 0.0, atol=1e-9)


def test_transform_never_changes_training_statistics() -> None:
    prep = Preprocessor().fit(_frame(300, seed=0))
    before = prep.mean_.copy()
    prep.transform(_frame(100, seed=1))
    np.testing.assert_array_equal(before, prep.mean_)


def test_unseen_category_becomes_all_zero_one_hot() -> None:
    prep = Preprocessor().fit(_frame(300, seed=0))
    other = _frame(50, seed=2)
    other.loc[0, "room_category"] = "Type Z"
    x = prep.transform(other)
    room_columns = [
        i for i, n in enumerate(prep.feature_names_) if n.startswith("room_category=")
    ]
    assert x[0, room_columns].sum() == 0.0


def test_clean_training_data_drops_duplicates_but_keeps_other_rows() -> None:
    df = _frame(50, seed=3)
    duplicated = pd.concat([df, df.iloc[[0]].assign(id="copy")], ignore_index=True)
    assert len(clean_training_data(duplicated)) == len(df)
