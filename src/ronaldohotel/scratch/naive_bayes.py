from typing import Self

import numpy as np
from numpy.typing import NDArray

from ronaldohotel.scratch.base import SavableModel

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


class NaiveBayes(SavableModel):
    def __init__(self, smoothing: float = 1.0, var_smoothing: float = 1e-2) -> None:
        self.smoothing = smoothing
        self.var_smoothing = var_smoothing
        self.numeric_mask_: BoolArray = np.zeros(0, dtype=bool)
        self.log_prior_: FloatArray = np.zeros(2)
        self.mean_: FloatArray = np.zeros((2, 0))
        self.var_: FloatArray = np.ones((2, 0))
        self.prob_: FloatArray = np.full((2, 0), 0.5)

    def fit(self, X: FloatArray, y: IntArray, numeric_mask: BoolArray) -> Self:
        self.numeric_mask_ = numeric_mask
        numeric = X[:, numeric_mask]
        binary = X[:, ~numeric_mask]
        counts = np.bincount(y, minlength=2).astype(np.float64)
        self.log_prior_ = np.log(counts / counts.sum())
        self.mean_ = np.vstack([numeric[y == c].mean(axis=0) for c in (0, 1)])
        variance = np.vstack([numeric[y == c].var(axis=0) for c in (0, 1)])
        self.var_ = variance + self.var_smoothing
        ones = np.vstack([binary[y == c].sum(axis=0) for c in (0, 1)])
        self.prob_ = (ones + self.smoothing) / (counts[:, None] + 2.0 * self.smoothing)
        return self

    def predict_log_joint(self, X: FloatArray) -> FloatArray:
        numeric = X[:, self.numeric_mask_]
        binary = X[:, ~self.numeric_mask_]
        joint = np.empty((len(X), 2))
        for c in (0, 1):
            gauss = -0.5 * (
                np.log(2.0 * np.pi * self.var_[c])
                + (numeric - self.mean_[c]) ** 2 / self.var_[c]
            ).sum(axis=1)
            bernoulli = binary @ np.log(self.prob_[c]) + (1.0 - binary) @ np.log(
                1.0 - self.prob_[c]
            )
            joint[:, c] = self.log_prior_[c] + gauss + bernoulli
        return joint

    def predict_proba(self, X: FloatArray) -> FloatArray:
        joint = self.predict_log_joint(X)
        weights = np.exp(joint - joint.max(axis=1, keepdims=True))
        return np.asarray(weights[:, 1] / weights.sum(axis=1), dtype=np.float64)

    def predict(self, X: FloatArray, threshold: float = 0.5) -> IntArray:
        return (self.predict_proba(X) >= threshold).astype(np.int64)
