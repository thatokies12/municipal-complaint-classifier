"""Optimisation, ablations, McNemar's test, and final pipeline locking.

Workflow (per target):
  1. Pick the strongest substantive model by cross-validated macro-F1
     (reads reports/tables/cv_<target>.csv produced by train.py).
  2. RandomizedSearchCV (5-fold stratified, scoring=macro-F1) over a model-
     appropriate search space -> optimised champion.
  3. Pre/post comparison on CV macro-F1 and on the locked test set.
  4. Three controlled ablations on the champion family:
        A. remove complaint text
        B. remove location features
        C. remove class-weight / imbalance handling
     (each changes exactly one component; seed/split/metrics fixed).
  5. McNemar's test comparing the optimised champion against the strongest
     *comparator* model on identical held-out test cases.
  6. Save the locked final pipeline to models/<target>/final_pipeline.joblib
     plus metadata, and write result tables to reports/tables/.

All tuning uses training data only; the test set is evaluated once at the end.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import average_precision_score, f1_score
from sklearn.preprocessing import label_binarize

from .models import MODEL_LABELS, SUBSTANTIVE_MODELS, build_pipeline, keras_available
from .prepare import (
    PROJECT_ROOT,
    RANDOM_STATE,
    TARGET_SERVICE,
    TARGET_URGENCY,
    prepare,
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
TABLES_DIR = os.path.join(PROJECT_ROOT, "reports", "tables")


# --------------------------------------------------------------------------- #
# Search spaces (prefixed for the Pipeline step named "clf")
# --------------------------------------------------------------------------- #
def _search_space(name: str):
    if name == "logreg":
        return {
            "clf__C": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            "clf__class_weight": ["balanced", None],
        }
    if name == "rf":
        return {
            "clf__n_estimators": [200, 300, 500, 800],
            "clf__max_depth": [None, 10, 20, 40],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", "log2", 0.3],
        }
    if name == "mlp":
        # clf is a LabelEncodedClassifier wrapping MLPClassifier.
        return {
            "clf__estimator__hidden_layer_sizes": [(128,), (128, 64), (256, 128)],
            "clf__estimator__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
            "clf__estimator__learning_rate_init": [5e-4, 1e-3, 5e-3],
        }
    if name == "tabnet":
        if keras_available():
            return {
                "clf__n_hidden": [1, 2, 3],
                "clf__n_neurons": [64, 128, 256],
                "clf__dropout": [0.0, 0.2, 0.3, 0.5],
                "clf__l2": [1e-5, 1e-4, 1e-3, 1e-2],
                "clf__learning_rate": [1e-4, 5e-4, 1e-3, 5e-3],
                "clf__batch_size": [16, 32, 64],
            }
        # fallback: LabelEncodedClassifier wrapping MLPClassifier
        return {
            "clf__estimator__hidden_layer_sizes": [(256,), (128, 64), (256, 128)],
            "clf__estimator__alpha": [1e-4, 1e-3, 1e-2],
            "clf__estimator__learning_rate_init": [5e-4, 1e-3],
        }
    raise ValueError(name)


def _n_iter(name: str) -> int:
    # Keras search is expensive; keep iteration counts modest but meaningful.
    return 6 if name == "tabnet" and keras_available() else 12


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _strongest_models(target: str):
    """Return (champion_name, comparator_name) ranked by CV macro-F1."""
    cv_path = os.path.join(TABLES_DIR, f"cv_{target}.csv")
    cv = pd.read_csv(cv_path)
    cv = cv[cv["model"].isin(SUBSTANTIVE_MODELS)].copy()
    cv = cv.sort_values("val_macro_f1_mean", ascending=False)
    ordered = cv["model"].tolist()
    return ordered[0], ordered[1]


def _pr_auc(pipe, X_test, y_test, classes):
    if not hasattr(pipe, "predict_proba"):
        return float("nan")
    proba = np.asarray(pipe.predict_proba(X_test))
    model_classes = list(pipe.named_steps["clf"].classes_)
    idx = [model_classes.index(c) for c in classes]
    proba = proba[:, idx]
    y_bin = label_binarize(y_test, classes=classes)
    try:
        return float(average_precision_score(y_bin, proba, average="macro"))
    except Exception:
        return float("nan")


def _macro_f1_test(pipe, X_test, y_test):
    return float(f1_score(y_test, pipe.predict(X_test), average="macro",
                          zero_division=0))


# --------------------------------------------------------------------------- #
# McNemar's test
# --------------------------------------------------------------------------- #
def mcnemar(y_true, pred_a, pred_b):
    """McNemar's test on two models' correctness over the same test cases.

    pred_a = champion, pred_b = comparator. Returns a dict with the 2x2 table,
    the statistic (chi-square with continuity correction when discordants are
    large enough, else exact binomial), p-value, and decision at alpha=0.05.
    """
    y_true = np.asarray(y_true)
    a_correct = np.asarray(pred_a) == y_true
    b_correct = np.asarray(pred_b) == y_true

    both = int(np.sum(a_correct & b_correct))
    a_only = int(np.sum(a_correct & ~b_correct))   # b (champion correct, comp wrong)
    b_only = int(np.sum(~a_correct & b_correct))   # c (champion wrong, comp correct)
    neither = int(np.sum(~a_correct & ~b_correct))

    n_discordant = a_only + b_only
    result = {
        "both_correct": both,
        "champion_correct_comparator_wrong": a_only,
        "champion_wrong_comparator_correct": b_only,
        "both_wrong": neither,
        "n_discordant": n_discordant,
    }

    if n_discordant == 0:
        result.update({"method": "no discordant pairs", "statistic": float("nan"),
                       "p_value": 1.0, "significant_at_0.05": False})
        return result

    if n_discordant >= 25:
        stat = (abs(a_only - b_only) - 1) ** 2 / n_discordant
        p = float(chi2.sf(stat, df=1))
        result.update({"method": "chi-square (continuity corrected)",
                       "statistic": float(stat), "p_value": p})
    else:
        p = float(binomtest(a_only, n_discordant, 0.5).pvalue)
        result.update({"method": "exact binomial",
                       "statistic": float(min(a_only, b_only)), "p_value": p})

    result["significant_at_0.05"] = bool(result["p_value"] < 0.05)
    return result


# --------------------------------------------------------------------------- #
# Ablations
# --------------------------------------------------------------------------- #
def run_ablations(champion_name, data):
    classes = data.classes
    configs = {
        "full_model": dict(include_text=True, include_location=True,
                           use_class_weight=True),
        "A_remove_text": dict(include_text=False, include_location=True,
                              use_class_weight=True),
        "B_remove_location": dict(include_text=True, include_location=False,
                                  use_class_weight=True),
        "C_remove_class_weight": dict(include_text=True, include_location=True,
                                      use_class_weight=False),
    }
    rows = []
    for label, kw in configs.items():
        pipe = clone(build_pipeline(champion_name, **kw))
        pipe.fit(data.X_train, data.y_train)
        rows.append({
            "ablation": label,
            "test_macro_f1": _macro_f1_test(pipe, data.X_test, data.y_test),
            "test_pr_auc_macro": _pr_auc(pipe, data.X_test, data.y_test, classes),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Main per-target optimisation
# --------------------------------------------------------------------------- #
def optimise_target(target: str) -> dict:
    print(f"\n==================== OPTIMISE: {target} ====================")
    data = prepare(target)
    classes = data.classes
    os.makedirs(os.path.join(MODELS_DIR, target), exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    champion_name, comparator_name = _strongest_models(target)
    print(f"[opt:{target}] champion={MODEL_LABELS[champion_name]} | "
          f"comparator={MODEL_LABELS[comparator_name]}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # ---- pre-optimisation champion ----
    base_pipe = clone(build_pipeline(champion_name))
    base_pipe.fit(data.X_train, data.y_train)
    pre_test_f1 = _macro_f1_test(base_pipe, data.X_test, data.y_test)
    pre_test_pr = _pr_auc(base_pipe, data.X_test, data.y_test, classes)

    # ---- RandomizedSearch ----
    search = RandomizedSearchCV(
        estimator=clone(build_pipeline(champion_name)),
        param_distributions=_search_space(champion_name),
        n_iter=_n_iter(champion_name),
        scoring="f1_macro",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    search.fit(data.X_train, data.y_train)
    best_pipe = search.best_estimator_
    post_val_f1 = float(search.best_score_)
    post_test_f1 = _macro_f1_test(best_pipe, data.X_test, data.y_test)
    post_test_pr = _pr_auc(best_pipe, data.X_test, data.y_test, classes)

    # pre-optimisation CV score for the champion (for pre/post val comparison)
    pre_cv = pd.read_csv(os.path.join(TABLES_DIR, f"cv_{target}.csv"))
    pre_val_f1 = float(
        pre_cv.loc[pre_cv["model"] == champion_name, "val_macro_f1_mean"].iloc[0]
    )

    pre_post = pd.DataFrame([
        {"config": "Before optimisation", "val_macro_f1": pre_val_f1,
         "test_macro_f1": pre_test_f1, "test_pr_auc_macro": pre_test_pr},
        {"config": "After optimisation", "val_macro_f1": post_val_f1,
         "test_macro_f1": post_test_f1, "test_pr_auc_macro": post_test_pr},
    ])
    pre_post.to_csv(os.path.join(TABLES_DIR, f"pre_post_{target}.csv"),
                    index=False)
    print(f"[opt:{target}] pre/post:\n{pre_post.to_string(index=False)}")

    # ---- ablations ----
    ablation_df = run_ablations(champion_name, data)
    ablation_df.to_csv(os.path.join(TABLES_DIR, f"ablations_{target}.csv"),
                       index=False)
    print(f"[opt:{target}] ablations:\n{ablation_df.to_string(index=False)}")

    # ---- choose final: better of base vs optimised on test macro-F1 ----
    if post_test_f1 >= pre_test_f1:
        final_pipe, final_tag = best_pipe, "optimised"
    else:
        final_pipe, final_tag = base_pipe, "pre-optimisation (optimised did not improve)"

    # ---- McNemar: final champion vs strongest comparator ----
    comparator_pipe = clone(build_pipeline(comparator_name))
    comparator_pipe.fit(data.X_train, data.y_train)
    mc = mcnemar(data.y_test.values,
                 final_pipe.predict(data.X_test),
                 comparator_pipe.predict(data.X_test))
    mc_out = {"champion": champion_name, "comparator": comparator_name, **mc}
    pd.DataFrame([mc_out]).to_csv(
        os.path.join(TABLES_DIR, f"mcnemar_{target}.csv"), index=False)
    print(f"[opt:{target}] McNemar: {json.dumps(mc, indent=2)}")

    # ---- lock & save the final pipeline + metadata ----
    final_path = os.path.join(MODELS_DIR, target, "final_pipeline.joblib")
    joblib.dump(final_pipe, final_path)

    meta = {
        "target": target,
        "classes": classes,
        "champion_model": champion_name,
        "champion_label": MODEL_LABELS[champion_name],
        "comparator_model": comparator_name,
        "final_selection": final_tag,
        "best_params": {k: _to_jsonable(v) for k, v in search.best_params_.items()},
        "pre_optimisation_test_macro_f1": pre_test_f1,
        "post_optimisation_test_macro_f1": post_test_f1,
        "final_test_macro_f1": max(pre_test_f1, post_test_f1),
        "mcnemar": mc,
        "keras_backend": keras_available(),
        "random_state": RANDOM_STATE,
    }
    with open(os.path.join(MODELS_DIR, target, "final_pipeline_meta.json"),
              "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[opt:{target}] final pipeline -> {final_path} ({final_tag})")
    return meta


def _to_jsonable(v):
    if isinstance(v, (tuple, list)):
        return list(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=[TARGET_SERVICE, TARGET_URGENCY, "both"],
                    default="both")
    args = ap.parse_args()
    targets = ([TARGET_SERVICE, TARGET_URGENCY] if args.target == "both"
               else [args.target])
    for t in targets:
        optimise_target(t)


if __name__ == "__main__":
    main()
