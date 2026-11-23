import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ronaldohotel.data.preprocessing import (
    Preprocessor,
    clean_training_data,
    load_csv,
    target_array,
)
from ronaldohotel.evaluation.metrics import best_threshold, stratified_split
from ronaldohotel.scratch.decision_tree import DecisionTree
from ronaldohotel.scratch.knn import KNN
from ronaldohotel.scratch.logistic_regression import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def proba_for(
    name: str, fit_df: pd.DataFrame, y_fit: NDArray[np.int64], eval_df: pd.DataFrame
) -> NDArray[np.float64]:
    prep = Preprocessor().fit(fit_df)
    if name == "tree":
        tree = DecisionTree(14, 40, 20, class_weight="balanced")
        tree.fit(prep.transform_tree(fit_df), y_fit, prep.tree_categorical_mask_)
        return tree.predict_proba(prep.transform_tree(eval_df))
    x_fit, x_eval = prep.transform(fit_df), prep.transform(eval_df)
    if name == "knn":
        return KNN(31).fit(x_fit, y_fit).predict_proba(x_eval)
    if name == "logreg":
        model = LogisticRegression("l2", 1e-4, class_weight="balanced")
        return model.fit(x_fit, y_fit).predict_proba(x_eval)
    raise SystemExit(f"unknown model {name!r}; use logreg, tree or knn")


def main(name: str) -> None:
    train = clean_training_data(load_csv(ROOT / "data" / "train.csv"))
    test = load_csv(ROOT / "data" / "test.csv")
    y = target_array(train)
    fit_idx, val_idx = stratified_split(y, 0.2, SEED)
    val_proba = proba_for(name, train.iloc[fit_idx], y[fit_idx], train.iloc[val_idx])
    grid = np.linspace(0.05, 0.95, 91)
    threshold, score = best_threshold(y[val_idx], val_proba, grid)
    print(f"{name}: validation macro-F1 {score:.4f} at threshold {threshold:.2f}")  # noqa: T201
    labels = (proba_for(name, train, y, test) >= threshold).astype(np.int64)
    out = ROOT / "output" / f"submission_{name}.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame({"id": test["id"], "is_canceled": labels}).to_csv(out, index=False)
    print(f"wrote {out.name}: {len(labels)} rows, {labels.mean():.3f} predicted 1")  # noqa: T201


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tree")
