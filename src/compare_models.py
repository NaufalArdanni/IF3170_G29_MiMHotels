import time
import warnings
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ronaldohotel.baselines import sklearn_models as sk
from ronaldohotel.data.preprocessing import (
    Preprocessor,
    clean_training_data,
    load_csv,
    target_array,
)
from ronaldohotel.evaluation.metrics import (
    accuracy,
    best_threshold,
    stratified_split,
)
from ronaldohotel.scratch.decision_tree import DecisionTree
from ronaldohotel.scratch.knn import KNN
from ronaldohotel.scratch.logistic_regression import LogisticRegression, Penalty
from ronaldohotel.scratch.naive_bayes import NaiveBayes
from ronaldohotel.scratch.svm import SVM, Kernel

plt.switch_backend("Agg")
ROOT = Path(__file__).resolve().parents[1]
SEED = 42
FloatArray = NDArray[np.float64]
Run = Callable[[], FloatArray]
GRID = np.linspace(0.05, 0.95, 91)
SCRATCH_COLOR, SKLEARN_COLOR = "#1f77b4", "#ff7f0e"


def sigmoid(z: FloatArray) -> FloatArray:
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0))))


def main() -> None:
    train = clean_training_data(load_csv(ROOT / "data" / "train.csv"))
    y = target_array(train)
    fit_idx, val_idx = stratified_split(y, 0.2, SEED)
    prep = Preprocessor().fit(train.iloc[fit_idx])
    xf, xv = prep.transform(train.iloc[fit_idx]), prep.transform(train.iloc[val_idx])
    tf = prep.transform_tree(train.iloc[fit_idx])
    tv = prep.transform_tree(train.iloc[val_idx])
    yf, yv = y[fit_idx], y[val_idx]
    n = len(yf)

    def tree() -> FloatArray:
        model = DecisionTree(14, 40, 20, class_weight="balanced")
        return model.fit(tf, yf, prep.tree_categorical_mask_).predict_proba(tv)

    def logreg(penalty: Penalty) -> Run:
        def run() -> FloatArray:
            model = LogisticRegression(penalty, 1e-4, class_weight="balanced")
            return model.fit(xf, yf).predict_proba(xv)

        return run

    def logreg_sk(penalty: str) -> Run:
        def run() -> FloatArray:
            model = sk.logistic_regression(penalty, n).fit(xf, yf)
            return np.asarray(model.predict_proba(xv)[:, 1], dtype=np.float64)

        return run

    def svm_scratch(kernel: Kernel) -> Run:
        return lambda: SVM(kernel, C=1.0).fit(xf, yf).predict_proba(xv)

    def svm_sk(kernel: str) -> Run:
        def run() -> FloatArray:
            idx = SVM()._subsample(yf)
            model = sk.svm(kernel).fit(xf[idx], yf[idx])
            return sigmoid(np.asarray(model.decision_function(xv), dtype=np.float64))

        return run

    def proba_sk(
        model_fn: Callable[[], object], x_fit: FloatArray, x_val: FloatArray
    ) -> Run:
        def run() -> FloatArray:
            model = model_fn()
            model.fit(x_fit, yf)  # type: ignore[attr-defined]
            return np.asarray(model.predict_proba(x_val)[:, 1], dtype=np.float64)  # type: ignore[attr-defined]

        return run

    runs: list[tuple[str, str, Run, Run]] = [
        (
            "KNN",
            "k=31",
            lambda: KNN(31).fit(xf, yf).predict_proba(xv),
            proba_sk(lambda: sk.knn(31), xf, xv),
        ),
        ("Decision Tree", "CART (gini)", tree, proba_sk(sk.decision_tree, xf, xv)),
        ("Logistic Regression", "L1", logreg("l1"), logreg_sk("l1")),
        ("Logistic Regression", "L2", logreg("l2"), logreg_sk("l2")),
        (
            "Logistic Regression",
            "Elastic Net",
            logreg("elasticnet"),
            logreg_sk("elasticnet"),
        ),
        ("SVM", "Linear", svm_scratch("linear"), svm_sk("linear")),
        ("SVM", "Polynomial", svm_scratch("poly"), svm_sk("poly")),
        ("SVM", "Gaussian", svm_scratch("rbf"), svm_sk("rbf")),
        (
            "Naive Bayes",
            "Gaussian + Bernoulli",
            lambda: NaiveBayes().fit(xf, yf, prep.numeric_mask_).predict_proba(xv),
            lambda: (
                sk.MixedNaiveBayes(prep.numeric_mask_).fit(xf, yf).predict_proba(xv)
            ),
        ),
    ]
    rows = []
    for algorithm, variant, scratch, sklearn_run in runs:
        scores = {}
        for impl, run in (("scratch", scratch), ("scikit-learn", sklearn_run)):
            start = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                proba = run()
            seconds = time.perf_counter() - start
            threshold, f1 = best_threshold(yv, proba, GRID)
            labels = (proba >= threshold).astype(np.int64)
            scores[impl] = labels
            rows.append(
                {
                    "algorithm": algorithm,
                    "variant": variant,
                    "implementation": impl,
                    "macro_f1": round(f1, 4),
                    "threshold": round(threshold, 2),
                    "accuracy": round(accuracy(yv, labels), 4),
                    "seconds": round(seconds, 1),
                }
            )
        rows[-1]["label_agreement"] = round(
            accuracy(scores["scratch"], scores["scikit-learn"]), 4
        )
    table = pd.DataFrame(rows)
    out = ROOT / "output"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "results.csv", index=False)

    wide = table.pivot_table(
        index=["algorithm", "variant"],
        columns="implementation",
        values="macro_f1",
        sort=False,
    )
    labels_text = [f"{a}\n{v}" for a, v in wide.index]
    pos = np.arange(len(wide))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(pos - 0.2, wide["scratch"], 0.4, label="From scratch", color=SCRATCH_COLOR)
    ax.bar(
        pos + 0.2, wide["scikit-learn"], 0.4, label="scikit-learn", color=SKLEARN_COLOR
    )
    ax.set_xticks(pos, labels_text, fontsize=8)
    ax.set_ylabel("Macro F1 (validation)")
    ax.set_title("From-scratch vs scikit-learn")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=150)
    print(table.to_string(index=False))  # noqa: T201


if __name__ == "__main__":
    main()
