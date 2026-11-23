from pathlib import Path

import numpy as np
import pandas as pd

from ronaldohotel.data.preprocessing import (
    Preprocessor,
    clean_training_data,
    load_csv,
    target_array,
)
from ronaldohotel.evaluation.metrics import (
    best_threshold,
    macro_f1,
    stratified_split,
)
from ronaldohotel.scratch.logistic_regression import LogisticRegression
from ronaldohotel.scratch.naive_bayes import NaiveBayes

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def main() -> None:
    train = clean_training_data(load_csv(ROOT / "data" / "train.csv"))
    test = load_csv(ROOT / "data" / "test.csv")
    y = target_array(train)
    fit_idx, val_idx = stratified_split(y, 0.2, SEED)
    prep = Preprocessor().fit(train.iloc[fit_idx])
    x_fit = prep.transform(train.iloc[fit_idx])
    x_val = prep.transform(train.iloc[val_idx])

    logreg = LogisticRegression("l2", alpha=1e-4, class_weight="balanced")
    logreg.fit(x_fit, y[fit_idx])
    proba = logreg.predict_proba(x_val)
    threshold, tuned = best_threshold(y[val_idx], proba)
    plain = macro_f1(y[val_idx], (proba >= 0.5).astype(np.int64))
    print(f"LR  macro-F1 @0.5 = {plain:.4f} | tuned {tuned:.4f} @ {threshold:.2f}")  # noqa: T201

    bayes = NaiveBayes().fit(x_fit, y[fit_idx], prep.numeric_mask_)
    nb_thr, nb_tuned = best_threshold(y[val_idx], bayes.predict_proba(x_val))
    print(f"NB  macro-F1 tuned {nb_tuned:.4f} @ {nb_thr:.2f}")  # noqa: T201

    prep_all = Preprocessor().fit(train)
    final = LogisticRegression("l2", alpha=1e-4, class_weight="balanced")
    final.fit(prep_all.transform(train), y)
    labels = final.predict(prep_all.transform(test), threshold)
    out = ROOT / "output" / "submission_logreg.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame({"id": test["id"], "is_canceled": labels}).to_csv(out, index=False)
    print(f"wrote {out.name}: {len(labels)} rows, {labels.mean():.3f} predicted 1")  # noqa: T201


if __name__ == "__main__":
    main()
