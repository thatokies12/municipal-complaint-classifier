"""Training and 5-fold cross-validation for all models.

Trains the naive baseline and the four substantive models for a chosen target,
runs stratified 5-fold cross-validation on the TRAINING set only (the test set
is never touched here), fits each model on the full training set, and saves the
fitted pipelines plus a CV summary table.

Outputs (per target, written under models/ and reports/tables/):
  * models/<target>/<name>.joblib          - fitted pipeline for each model
  * reports/tables/cv_<target>.csv          - mean +/- SD CV scores per model
"""
from __future__ import annotations

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate

from .models import ALL_MODELS, MODEL_LABELS, build_pipeline
from .prepare import (
    PROJECT_ROOT,
    RANDOM_STATE,
    TARGET_SERVICE,
    TARGET_URGENCY,
    prepare,
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
TABLES_DIR = os.path.join(PROJECT_ROOT, "reports", "tables")

CV_SCORING = {
    "accuracy": "accuracy",
    "macro_f1": "f1_macro",
    "weighted_f1": "f1_weighted",
    "macro_precision": "precision_macro",
    "macro_recall": "recall_macro",
}


def _dirs(target: str):
    md = os.path.join(MODELS_DIR, target)
    os.makedirs(md, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    return md


def cross_validate_model(name, X_train, y_train, cv):
    """Return a dict of mean/SD CV scores for one model pipeline."""
    pipe = build_pipeline(name)
    results = cross_validate(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring=CV_SCORING,
        return_train_score=True,
        n_jobs=1,  # Keras model is not fork-safe; keep single-process
        error_score="raise",
    )
    row = {"model": name, "label": MODEL_LABELS[name]}
    for metric in CV_SCORING:
        test = results[f"test_{metric}"]
        train = results[f"train_{metric}"]
        row[f"val_{metric}_mean"] = float(np.mean(test))
        row[f"val_{metric}_sd"] = float(np.std(test))
        row[f"train_{metric}_mean"] = float(np.mean(train))
    # Overfitting gap on the primary metric.
    row["overfit_gap_macro_f1"] = (
        row["train_macro_f1_mean"] - row["val_macro_f1_mean"]
    )
    return row


def train_target(target: str, do_cv: bool = True) -> pd.DataFrame:
    print(f"\n==================== TARGET: {target} ====================")
    data = prepare(target)
    model_dir = _dirs(target)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for name in ALL_MODELS:
        print(f"[train:{target}] {MODEL_LABELS[name]} ...")
        if do_cv:
            rows.append(cross_validate_model(name, data.X_train, data.y_train, cv))
        # Fit on full training set and persist.
        pipe = clone(build_pipeline(name))
        pipe.fit(data.X_train, data.y_train)
        joblib.dump(pipe, os.path.join(model_dir, f"{name}.joblib"))

    cv_df = pd.DataFrame(rows)
    if do_cv and not cv_df.empty:
        cv_path = os.path.join(TABLES_DIR, f"cv_{target}.csv")
        cv_df.to_csv(cv_path, index=False)
        print(f"[train:{target}] CV table -> {cv_path}")
        show = cv_df[["label", "val_macro_f1_mean", "val_macro_f1_sd",
                      "val_accuracy_mean", "overfit_gap_macro_f1"]]
        print(show.to_string(index=False))
    return cv_df


def main():
    ap = argparse.ArgumentParser(description="Train models for a target.")
    ap.add_argument("--target", choices=[TARGET_SERVICE, TARGET_URGENCY, "both"],
                    default="both")
    ap.add_argument("--no-cv", action="store_true", help="skip cross-validation")
    args = ap.parse_args()

    targets = ([TARGET_SERVICE, TARGET_URGENCY] if args.target == "both"
               else [args.target])
    for t in targets:
        train_target(t, do_cv=not args.no_cv)


if __name__ == "__main__":
    main()
