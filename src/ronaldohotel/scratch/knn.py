from typing import Self

import numpy as np
from numpy.typing import NDArray

from ronaldohotel.scratch.base import SavableModel

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class KNN(SavableModel):
    def __init__(
        self, k: int = 25, weighted: bool = True, chunk_size: int = 512
    ) -> None:
        self.k = k
        self.weighted = weighted
        self.chunk_size = chunk_size
        self.X_: FloatArray = np.zeros((0, 0))
        self.y_: IntArray = np.zeros(0, dtype=np.int64)
        self.sq_norms_: FloatArray = np.zeros(0)

    def fit(self, X: FloatArray, y: IntArray) -> Self:
        self.X_ = np.asarray(X, dtype=np.float64)
        self.y_ = np.asarray(y, dtype=np.int64)
        self.sq_norms_ = (self.X_**2).sum(axis=1)
        return self

    def predict_proba(self, X: FloatArray) -> FloatArray:
        out = np.empty(len(X))
        for start in range(0, len(X), self.chunk_size):
            block = X[start : start + self.chunk_size]
            sq = (
                (block**2).sum(axis=1)[:, None]
                + self.sq_norms_
                - 2.0 * block @ self.X_.T
            )
            sq = np.maximum(sq, 0.0)
            nearest = np.argpartition(sq, self.k - 1, axis=1)[:, : self.k]
            dist = np.sqrt(np.take_along_axis(sq, nearest, axis=1))
            weight = 1.0 / (dist + 1e-9) if self.weighted else np.ones_like(dist)
            out[start : start + self.chunk_size] = (weight * self.y_[nearest]).sum(
                axis=1
            ) / weight.sum(axis=1)
        return out

    def predict(self, X: FloatArray, threshold: float = 0.5) -> IntArray:
        return (self.predict_proba(X) >= threshold).astype(np.int64)
