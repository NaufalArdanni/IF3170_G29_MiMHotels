from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from ronaldohotel.evaluation.metrics import accuracy
from ronaldohotel.scratch.decision_tree import DecisionTree
from ronaldohotel.scratch.knn import KNN

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _blobs(n: int = 200) -> tuple[FloatArray, IntArray]:
    rng = np.random.default_rng(0)
    y = (np.arange(n) % 2).astype(np.int64)
    x = rng.normal(size=(n, 3)) + 3.0 * y[:, None]
    return x, y


def test_knn_separates_blobs_and_memorizes_with_k1() -> None:
    x, y = _blobs()
    assert accuracy(y, KNN(k=7).fit(x, y).predict(x)) > 0.95
    assert accuracy(y, KNN(k=1).fit(x, y).predict(x)) == 1.0


def test_tree_learns_numeric_threshold_with_missing_values() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 2))
    y = (x[:, 0] > 0.3).astype(np.int64)
    x[rng.random(500) < 0.1, 0] = np.nan
    mask = np.zeros(2, dtype=bool)
    tree = DecisionTree(max_depth=4, min_samples_leaf=5).fit(x, y, mask)
    proba = tree.predict_proba(x)
    assert np.all((proba >= 0) & (proba <= 1))
    assert accuracy(y, tree.predict(x)) > 0.9


def test_tree_learns_categorical_split() -> None:
    rng = np.random.default_rng(2)
    codes = rng.integers(0, 4, 400).astype(np.float64)
    y = (codes == 2).astype(np.int64)
    x = np.column_stack([rng.normal(size=400), codes])
    tree = DecisionTree(max_depth=2, min_samples_leaf=5)
    tree.fit(x, y, np.array([False, True]))
    assert accuracy(y, tree.predict(x)) == 1.0
    assert tree.feature_[0] == 1


def test_tree_save_and_load(tmp_path: Path) -> None:
    x, y = _blobs()
    tree = DecisionTree(max_depth=3).fit(x, y, np.zeros(3, dtype=bool))
    tree.save(tmp_path / "tree.pkl")
    loaded = DecisionTree.load(tmp_path / "tree.pkl")
    np.testing.assert_array_equal(loaded.predict(x), tree.predict(x))


def test_load_rejects_wrong_type(tmp_path: Path) -> None:
    x, y = _blobs()
    KNN().fit(x, y).save(tmp_path / "knn.pkl")
    with pytest.raises(TypeError):
        DecisionTree.load(tmp_path / "knn.pkl")
