from dataclasses import dataclass
from typing import Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ronaldohotel.scratch.base import SavableModel

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass
class _Split:
    gain: float
    feature: int
    threshold: float
    missing_left: bool


def _gini(positive: ArrayLike, total: ArrayLike) -> FloatArray:
    share = np.asarray(positive, dtype=np.float64) / np.asarray(total, dtype=np.float64)
    return 2.0 * share * (1.0 - share)


class DecisionTree(SavableModel):
    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        min_gain: float = 1e-7,
        class_weight: Literal["balanced"] | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.class_weight = class_weight
        self.categorical_mask_: BoolArray = np.zeros(0, dtype=bool)
        self.feature_: IntArray = np.full(1, -1, dtype=np.int64)
        self.threshold_: FloatArray = np.zeros(1)
        self.left_: IntArray = np.zeros(1, dtype=np.int64)
        self.right_: IntArray = np.zeros(1, dtype=np.int64)
        self.missing_left_: BoolArray = np.zeros(1, dtype=bool)
        self.value_: FloatArray = np.zeros(1)

    @property
    def n_nodes_(self) -> int:
        return len(self.feature_)

    def _weights(self, y: IntArray) -> FloatArray:
        if self.class_weight != "balanced":
            return np.ones(len(y))
        counts = np.bincount(y, minlength=2).astype(np.float64)
        return (len(y) / (2.0 * counts))[y]

    def _find_split(
        self, X: FloatArray, y: IntArray, w: FloatArray, idx: IntArray
    ) -> _Split | None:
        best = _Split(self.min_gain, -1, 0.0, True)
        leaf = self.min_samples_leaf
        node_w = w[idx]
        node_pos = node_w * y[idx]
        total_w = node_w.sum()
        for j in range(X.shape[1]):
            col = X[idx, j]
            seen = ~np.isnan(col)
            n_seen = int(seen.sum())
            if n_seen < 2 * leaf:
                continue
            xs, ws, ps = col[seen], node_w[seen], node_pos[seen]
            w_seen, pos_seen = float(ws.sum()), float(ps.sum())
            base = float(_gini(pos_seen, w_seen))
            share = w_seen / total_w
            if self.categorical_mask_[j]:
                for code in np.unique(xs):
                    left = xs == code
                    n_left = int(left.sum())
                    if min(n_left, n_seen - n_left) < leaf:
                        continue
                    w_left, pos_left = float(ws[left].sum()), float(ps[left].sum())
                    w_right, pos_right = w_seen - w_left, pos_seen - pos_left
                    child = (
                        w_left * _gini(pos_left, w_left)
                        + w_right * _gini(pos_right, w_right)
                    ) / w_seen
                    gain = share * (base - float(child))
                    if gain > best.gain:
                        best = _Split(gain, j, float(code), n_left >= n_seen - n_left)
                continue
            order = np.argsort(xs, kind="stable")
            xo = xs[order]
            cum_w = np.cumsum(ws[order])[:-1]
            cum_pos = np.cumsum(ps[order])[:-1]
            sizes = np.arange(1, n_seen)
            valid = (xo[:-1] < xo[1:]) & (sizes >= leaf) & (n_seen - sizes >= leaf)
            if not valid.any():
                continue
            rest_w, rest_pos = w_seen - cum_w, pos_seen - cum_pos
            impurity = (
                cum_w * _gini(cum_pos, cum_w) + rest_w * _gini(rest_pos, rest_w)
            ) / w_seen
            gains = np.where(valid, share * (base - impurity), -np.inf)
            i = int(np.argmax(gains))
            if gains[i] > best.gain:
                cut = float((xo[i] + xo[i + 1]) / 2.0)
                best = _Split(float(gains[i]), j, cut, i + 1 >= n_seen - (i + 1))
        return best if best.feature >= 0 else None

    def fit(self, X: FloatArray, y: IntArray, categorical_mask: BoolArray) -> Self:
        self.categorical_mask_ = categorical_mask
        w = self._weights(y)
        feature, threshold, left, right = [-1], [0.0], [0], [0]
        missing_left, value = [True], [float((w * y).sum() / w.sum())]
        stack = [(0, np.arange(len(y), dtype=np.int64), 0)]
        while stack:
            node, idx, depth = stack.pop()
            splittable = (
                depth < self.max_depth
                and len(idx) >= self.min_samples_split
                and 0 < y[idx].sum() < len(idx)
            )
            split = self._find_split(X, y, w, idx) if splittable else None
            if split is None:
                continue
            col = X[idx, split.feature]
            is_cat = bool(categorical_mask[split.feature])
            cond = col == split.threshold if is_cat else col <= split.threshold
            go_left = np.where(np.isnan(col), split.missing_left, cond)
            feature[node], threshold[node] = split.feature, split.threshold
            missing_left[node] = split.missing_left
            for side, rows in ((left, idx[go_left]), (right, idx[~go_left])):
                feature.append(-1)
                threshold.append(0.0)
                left.append(0)
                right.append(0)
                missing_left.append(True)
                value.append(float((w[rows] * y[rows]).sum() / w[rows].sum()))
                side[node] = len(feature) - 1
                stack.append((len(feature) - 1, rows, depth + 1))
        self.feature_ = np.array(feature, dtype=np.int64)
        self.threshold_ = np.array(threshold)
        self.left_ = np.array(left, dtype=np.int64)
        self.right_ = np.array(right, dtype=np.int64)
        self.missing_left_ = np.array(missing_left, dtype=bool)
        self.value_ = np.array(value)
        return self

    def predict_proba(self, X: FloatArray) -> FloatArray:
        node = np.zeros(len(X), dtype=np.int64)
        rows = np.arange(len(X))
        while True:
            active = self.feature_[node] >= 0
            if not active.any():
                break
            r, n = rows[active], node[active]
            f = self.feature_[n]
            col = X[r, f]
            thr = self.threshold_[n]
            cond = np.where(self.categorical_mask_[f], col == thr, col <= thr)
            go_left = np.where(np.isnan(col), self.missing_left_[n], cond)
            node[r] = np.where(go_left, self.left_[n], self.right_[n])
        return self.value_[node]

    def predict(self, X: FloatArray, threshold: float = 0.5) -> IntArray:
        return (self.predict_proba(X) >= threshold).astype(np.int64)
