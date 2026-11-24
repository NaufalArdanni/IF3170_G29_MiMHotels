from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from ronaldohotel.evaluation.metrics import accuracy
from ronaldohotel.scratch.svm import SVM, Kernel

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _circles(n: int = 300) -> tuple[FloatArray, IntArray]:
    rng = np.random.default_rng(0)
    y = (np.arange(n) % 2).astype(np.int64)
    radius = np.where(y == 1, 3.0, 1.0) + 0.15 * rng.normal(size=n)
    angle = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([radius * np.cos(angle), radius * np.sin(angle)]), y


def test_linear_kernel_separates_blobs() -> None:
    rng = np.random.default_rng(1)
    y = (np.arange(200) % 2).astype(np.int64)
    x = rng.normal(size=(200, 2)) + 3.0 * y[:, None]
    model = SVM("linear", C=1.0, class_weight=None).fit(x, y)
    assert accuracy(y, model.predict(x)) > 0.95


@pytest.mark.parametrize("kernel", ["poly", "rbf"])
def test_nonlinear_kernels_solve_circles(kernel: Kernel) -> None:
    x, y = _circles()
    model = SVM(kernel, C=10.0, degree=2, class_weight=None).fit(x, y)
    assert accuracy(y, model.predict(x)) > 0.95


def test_linear_kernel_cannot_solve_circles() -> None:
    x, y = _circles()
    assert accuracy(y, SVM("linear").fit(x, y).predict(x)) < 0.75


def test_subsampling_and_save_load(tmp_path: Path) -> None:
    x, y = _circles(400)
    model = SVM("rbf", max_samples=150).fit(x, y)
    assert len(model.subsample_idx_) == 150
    model.save(tmp_path / "svm.pkl")
    loaded = SVM.load(tmp_path / "svm.pkl")
    np.testing.assert_allclose(loaded.decision_function(x), model.decision_function(x))
