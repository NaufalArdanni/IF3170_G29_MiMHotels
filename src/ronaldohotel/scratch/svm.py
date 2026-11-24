from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray

from ronaldohotel.scratch.base import SavableModel

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Kernel = Literal["linear", "poly", "rbf"]


class SVM(SavableModel):
    def __init__(
        self,
        kernel: Kernel = "rbf",
        C: float = 1.0,
        gamma: float | None = None,
        degree: int = 3,
        coef0: float = 1.0,
        tol: float = 1e-3,
        max_iter: int = 200_000,
        max_samples: int = 6000,
        class_weight: Literal["balanced"] | None = "balanced",
        seed: int = 0,
    ) -> None:
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.max_samples = max_samples
        self.class_weight = class_weight
        self.seed = seed
        self.gamma_ = 1.0
        self.subsample_idx_: IntArray = np.zeros(0, dtype=np.int64)
        self.support_vectors_: FloatArray = np.zeros((0, 0))
        self.dual_coef_: FloatArray = np.zeros(0)
        self.bias_ = 0.0
        self.n_iter_ = 0

    def _kernel(self, a: FloatArray, b: FloatArray) -> FloatArray:
        if self.kernel == "linear":
            return a @ b.T
        if self.kernel == "poly":
            return (self.gamma_ * (a @ b.T) + self.coef0) ** self.degree
        sq = (a**2).sum(axis=1)[:, None] + (b**2).sum(axis=1)[None, :] - 2.0 * a @ b.T
        return np.exp(-self.gamma_ * np.maximum(sq, 0.0))

    def _subsample(self, y: IntArray) -> IntArray:
        if len(y) <= self.max_samples:
            return np.arange(len(y), dtype=np.int64)
        rng = np.random.default_rng(self.seed)
        parts = [
            rng.choice(
                np.flatnonzero(y == c),
                size=round(self.max_samples * float(np.mean(y == c))),
                replace=False,
            )
            for c in (0, 1)
        ]
        return np.sort(np.concatenate(parts)).astype(np.int64)

    def fit(self, X: FloatArray, y: IntArray) -> Self:
        self.subsample_idx_ = self._subsample(y)
        xs = np.asarray(X[self.subsample_idx_], dtype=np.float64)
        labels = y[self.subsample_idx_]
        ys = np.where(labels == 1, 1.0, -1.0)
        m = len(ys)
        self.gamma_ = (
            self.gamma if self.gamma is not None else 1.0 / (xs.shape[1] * xs.var())
        )
        upper = np.full(m, self.C)
        if self.class_weight == "balanced":
            counts = np.bincount(labels, minlength=2).astype(np.float64)
            upper = self.C * (m / (2.0 * counts))[labels]
        kmat = self._kernel(xs, xs)
        diag = np.diag(kmat).copy()
        alpha = np.zeros(m)
        grad = -np.ones(m)
        eps = 1e-12
        g_max = g_min = 0.0
        for it in range(self.max_iter):
            score = -ys * grad
            up = ((ys > 0) & (alpha < upper - eps)) | ((ys < 0) & (alpha > eps))
            low = ((ys > 0) & (alpha > eps)) | ((ys < 0) & (alpha < upper - eps))
            s_up = np.where(up, score, -np.inf)
            i = int(np.argmax(s_up))
            g_max = float(s_up[i])
            g_min = float(np.where(low, score, np.inf).min())
            self.n_iter_ = it
            if g_max - g_min < self.tol:
                break
            gap = g_max - score
            curvature = np.maximum(diag[i] + diag - 2.0 * kmat[i], eps)
            j = int(
                np.argmin(np.where(low & (gap > 0), -(gap * gap) / curvature, np.inf))
            )
            quad = max(diag[i] + diag[j] - 2.0 * kmat[i, j], eps)
            old_i, old_j = alpha[i], alpha[j]
            c_i, c_j = upper[i], upper[j]
            if ys[i] != ys[j]:
                delta = (-grad[i] - grad[j]) / quad
                diff = old_i - old_j
                a_i, a_j = old_i + delta, old_j + delta
                if diff > 0:
                    if a_j < 0:
                        a_j, a_i = 0.0, diff
                elif a_i < 0:
                    a_i, a_j = 0.0, -diff
                if diff > c_i - c_j:
                    if a_i > c_i:
                        a_i, a_j = c_i, c_i - diff
                elif a_j > c_j:
                    a_j, a_i = c_j, c_j + diff
            else:
                delta = (grad[i] - grad[j]) / quad
                total = old_i + old_j
                a_i, a_j = old_i - delta, old_j + delta
                if total > c_i:
                    if a_i > c_i:
                        a_i, a_j = c_i, total - c_i
                elif a_j < 0:
                    a_j, a_i = 0.0, total
                if total > c_j:
                    if a_j > c_j:
                        a_j, a_i = c_j, total - c_j
                elif a_i < 0:
                    a_i, a_j = 0.0, total
            alpha[i], alpha[j] = a_i, a_j
            grad += ys * (
                ys[i] * kmat[i] * (a_i - old_i) + ys[j] * kmat[j] * (a_j - old_j)
            )
        score = -ys * grad
        free = (alpha > eps) & (alpha < upper - eps)
        self.bias_ = float(score[free].mean()) if free.any() else (g_max + g_min) / 2.0
        support = alpha > eps
        self.support_vectors_ = xs[support]
        self.dual_coef_ = alpha[support] * ys[support]
        return self

    def decision_function(self, X: FloatArray, chunk_size: int = 1000) -> FloatArray:
        out = np.empty(len(X))
        for start in range(0, len(X), chunk_size):
            block = X[start : start + chunk_size]
            k = self._kernel(block, self.support_vectors_)
            out[start : start + chunk_size] = k @ self.dual_coef_ + self.bias_
        return out

    def predict_proba(self, X: FloatArray) -> FloatArray:
        z = np.clip(self.decision_function(X), -35.0, 35.0)
        return np.asarray(1.0 / (1.0 + np.exp(-z)), dtype=np.float64)

    def predict(self, X: FloatArray, threshold: float = 0.5) -> IntArray:
        return (self.predict_proba(X) >= threshold).astype(np.int64)
