from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray

from ronaldohotel.scratch.base import SavableModel

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Penalty = Literal["none", "l1", "l2", "elasticnet"]


def sigmoid(z: FloatArray) -> FloatArray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


class LogisticRegression(SavableModel):
    def __init__(
        self,
        penalty: Penalty = "l2",
        alpha: float = 1e-3,
        l1_ratio: float = 0.5,
        learning_rate: float = 0.5,
        max_iter: int = 1000,
        tol: float = 1e-9,
        class_weight: Literal["balanced"] | None = None,
    ) -> None:
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.class_weight = class_weight
        self.coef_: FloatArray = np.zeros(0)
        self.intercept_: float = 0.0
        self.loss_history_: list[float] = []

    def _strengths(self) -> tuple[float, float]:
        if self.penalty == "l1":
            return self.alpha, 0.0
        if self.penalty == "l2":
            return 0.0, self.alpha
        if self.penalty == "elasticnet":
            return self.alpha * self.l1_ratio, self.alpha * (1.0 - self.l1_ratio)
        return 0.0, 0.0

    def _sample_weights(self, y: IntArray) -> FloatArray:
        if self.class_weight != "balanced":
            return np.ones(len(y))
        counts = np.bincount(y, minlength=2).astype(np.float64)
        per_class = len(y) / (2.0 * counts)
        return per_class[y]

    def fit(self, X: FloatArray, y: IntArray) -> Self:
        n_samples, n_features = X.shape
        l1, l2 = self._strengths()
        sample_weight = self._sample_weights(y)
        w = np.zeros(n_features)
        b = 0.0
        self.loss_history_ = []
        previous = np.inf
        for _ in range(self.max_iter):
            p = sigmoid(X @ w + b)
            log_loss = -(y * np.log(p + 1e-12) + (1 - y) * np.log(1.0 - p + 1e-12))
            loss = float(
                np.mean(sample_weight * log_loss)
                + l1 * np.abs(w).sum()
                + 0.5 * l2 * (w @ w)
            )
            self.loss_history_.append(loss)
            if abs(previous - loss) < self.tol:
                break
            previous = loss
            error = (p - y) * sample_weight
            w = w - self.learning_rate * (X.T @ error / n_samples + l2 * w)
            b -= self.learning_rate * float(error.mean())
            if l1 > 0.0:
                step = self.learning_rate * l1
                w = np.sign(w) * np.maximum(np.abs(w) - step, 0.0)
        self.coef_ = w
        self.intercept_ = b
        return self

    def predict_proba(self, X: FloatArray) -> FloatArray:
        return sigmoid(X @ self.coef_ + self.intercept_)

    def predict(self, X: FloatArray, threshold: float = 0.5) -> IntArray:
        return (self.predict_proba(X) >= threshold).astype(np.int64)
