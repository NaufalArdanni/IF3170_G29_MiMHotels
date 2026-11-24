import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def knn(k: int = 31) -> KNeighborsClassifier:
    return KNeighborsClassifier(n_neighbors=k, weights="distance")


def decision_tree(
    max_depth: int = 14, min_samples_split: int = 40, min_samples_leaf: int = 20
) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=0,
    )


def logistic_regression(
    penalty: str, n_samples: int, alpha: float = 1e-4
) -> LogisticRegression:
    return LogisticRegression(
        C=1.0 / (n_samples * alpha),
        penalty=penalty,
        solver="saga",
        l1_ratio=0.5 if penalty == "elasticnet" else None,
        class_weight="balanced",
        max_iter=2000,
    )


def svm(kernel: str, C: float = 1.0, degree: int = 3) -> SVC:
    return SVC(
        kernel=kernel,
        C=C,
        degree=degree,
        gamma="scale",
        coef0=1.0,
        class_weight="balanced",
    )


class MixedNaiveBayes:
    def __init__(self, numeric_mask: NDArray[np.bool_]) -> None:
        self.numeric_mask = numeric_mask
        self.gaussian = GaussianNB(var_smoothing=1e-2)
        self.bernoulli = BernoulliNB(alpha=1.0, binarize=None)

    def fit(self, X: FloatArray, y: IntArray) -> "MixedNaiveBayes":
        self.gaussian.fit(X[:, self.numeric_mask], y)
        self.bernoulli.fit(X[:, ~self.numeric_mask], y)
        return self

    def predict_proba(self, X: FloatArray) -> FloatArray:
        joint = (
            self.gaussian.predict_joint_log_proba(X[:, self.numeric_mask])
            + self.bernoulli.predict_joint_log_proba(X[:, ~self.numeric_mask])
            - np.log(self.gaussian.class_prior_)
        )
        weights = np.exp(joint - joint.max(axis=1, keepdims=True))
        return np.asarray(weights[:, 1] / weights.sum(axis=1), dtype=np.float64)
