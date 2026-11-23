from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


def confusion_matrix(y_true: IntArray, y_pred: IntArray) -> IntArray:
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


def f1_score(y_true: IntArray, y_pred: IntArray, positive: int = 1) -> float:
    tp = int(np.sum((y_true == positive) & (y_pred == positive)))
    fp = int(np.sum((y_true != positive) & (y_pred == positive)))
    fn = int(np.sum((y_true == positive) & (y_pred != positive)))
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def macro_f1(y_true: IntArray, y_pred: IntArray) -> float:
    return 0.5 * (f1_score(y_true, y_pred, 0) + f1_score(y_true, y_pred, 1))


def accuracy(y_true: IntArray, y_pred: IntArray) -> float:
    return float(np.mean(y_true == y_pred))


def stratified_split(
    y: IntArray, val_size: float = 0.2, seed: int = 0
) -> tuple[IntArray, IntArray]:
    rng = np.random.default_rng(seed)
    chosen = [
        rng.permutation(np.flatnonzero(y == label))[
            : round(int(np.sum(y == label)) * val_size)
        ]
        for label in np.unique(y)
    ]
    val_idx = np.sort(np.concatenate(chosen))
    train_idx = np.setdiff1d(np.arange(len(y)), val_idx)
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def stratified_kfold(
    y: IntArray, n_splits: int = 5, seed: int = 0
) -> Iterator[tuple[IntArray, IntArray]]:
    rng = np.random.default_rng(seed)
    folds = np.empty(len(y), dtype=np.int64)
    for label in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == label))
        folds[idx] = np.arange(len(idx)) % n_splits
    for fold in range(n_splits):
        yield np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)


def best_threshold(
    y_true: IntArray, proba: FloatArray, grid: FloatArray | None = None
) -> tuple[float, float]:
    candidates = np.linspace(0.2, 0.8, 61) if grid is None else grid
    scores = [macro_f1(y_true, (proba >= t).astype(np.int64)) for t in candidates]
    best = int(np.argmax(scores))
    return float(candidates[best]), float(scores[best])
