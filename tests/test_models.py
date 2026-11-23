from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from ronaldohotel.evaluation.metrics import (
    accuracy,
    best_threshold,
    macro_f1,
    stratified_kfold,
    stratified_split,
)
from ronaldohotel.scratch.logistic_regression import LogisticRegression, Penalty
from ronaldohotel.scratch.naive_bayes import NaiveBayes

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _toy(n: int = 600) -> tuple[FloatArray, IntArray]:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, 3))
    y = (x[:, 0] + 0.5 * x[:, 1] + 0.2 * rng.normal(size=n) > 0).astype(np.int64)
    return x, y


def test_macro_f1_known_value() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    y_pred = np.array([0, 1, 1, 1], dtype=np.int64)
    assert macro_f1(y_true, y_pred) == pytest.approx((2 / 3 + 4 / 5) / 2)


@pytest.mark.parametrize("penalty", ["none", "l1", "l2", "elasticnet"])
def test_logistic_regression_learns(penalty: Penalty) -> None:
    x, y = _toy()
    model = LogisticRegression(penalty, alpha=1e-3).fit(x, y)
    assert accuracy(y, model.predict(x)) > 0.9
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_l1_penalty_zeroes_irrelevant_weight() -> None:
    x, y = _toy()
    model = LogisticRegression("l1", alpha=0.05).fit(x, y)
    assert abs(model.coef_[2]) < 1e-6
    assert abs(model.coef_[0]) > 0.1


def test_naive_bayes_mixed_features() -> None:
    x, y = _toy()
    rng = np.random.default_rng(1)
    noisy = np.where(rng.random(len(y)) < 0.1, 1 - y, y)
    mixed = np.column_stack([x[:, :2], noisy]).astype(np.float64)
    mask = np.array([True, True, False])
    model = NaiveBayes().fit(mixed, y, mask)
    assert accuracy(y, model.predict(mixed)) > 0.85
    proba = model.predict_proba(mixed)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    x, y = _toy()
    model = LogisticRegression("elasticnet").fit(x, y)
    model.save(tmp_path / "model.pkl")
    loaded = LogisticRegression.load(tmp_path / "model.pkl")
    np.testing.assert_allclose(loaded.predict_proba(x), model.predict_proba(x))


def test_load_rejects_wrong_type(tmp_path: Path) -> None:
    x, y = _toy()
    LogisticRegression().fit(x, y).save(tmp_path / "model.pkl")
    with pytest.raises(TypeError):
        NaiveBayes.load(tmp_path / "model.pkl")


def test_stratified_split_and_kfold_cover_all_rows() -> None:
    _, y = _toy()
    train_idx, val_idx = stratified_split(y, 0.25, seed=3)
    assert len(np.intersect1d(train_idx, val_idx)) == 0
    assert len(train_idx) + len(val_idx) == len(y)
    assert y[val_idx].mean() == pytest.approx(y.mean(), abs=0.02)
    seen = np.concatenate([val for _, val in stratified_kfold(y, 5)])
    assert sorted(seen) == list(range(len(y)))


def test_best_threshold_is_inside_grid() -> None:
    x, y = _toy()
    proba = LogisticRegression().fit(x, y).predict_proba(x)
    threshold, score = best_threshold(y, proba)
    assert 0.2 <= threshold <= 0.8
    assert score > 0.9
