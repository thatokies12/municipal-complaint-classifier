"""Evaluation of trained models on the locked held-out test set.

Loads the fitted pipelines saved by train.py and produces, per target:
  * reports/tables/test_metrics_<target>.csv  - accuracy, macro/weighted-F1,
    macro precision/recall, ROC-AUC (OvR), PR-AUC (macro) for every model
  * reports/tables/classification_report_<target>_<model>.txt
  * reports/figures/confusion_<target>_<model>.png
  * reports/figures/roc_<target>.png          - OvR ROC curves, all models
  * reports/figures/pr_<target>.png           - macro PR curves, all models
  * reports/figures/learning_curve_<target>_<model>.png (neural train/val)

Every number is computed from the real test predictions; nothing is invented.
"""
from __future__ import annotations

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from .models import ALL_MODELS, MODEL_LABELS
from .prepare import PROJECT_ROOT, TARGET_SERVICE, TARGET_URGENCY, prepare

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
TABLES_DIR = os.path.join(PROJECT_ROOT, "reports", "tables")
FIG_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
ACCENT = "#0a84ff"


def _load(target, name):
    path = os.path.join(MODELS_DIR, target, f"{name}.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def _proba_matrix(pipe, X, classes):
    """Return an (n, n_classes) probability matrix aligned to `classes`."""
    if not hasattr(pipe, "predict_proba"):
        return None
    proba = np.asarray(pipe.predict_proba(X))
    model_classes = list(pipe.named_steps["clf"].classes_)
    # Reorder columns to the canonical `classes` order.
    idx = [model_classes.index(c) for c in classes]
    return proba[:, idx]


def _safe_auc(y_true_bin, proba, average):
    try:
        return float(roc_auc_score(y_true_bin, proba, average=average,
                                   multi_class="ovr"))
    except Exception:
        return float("nan")


def evaluate_target(target: str) -> pd.DataFrame:
    print(f"\n==================== EVALUATE: {target} ====================")
    data = prepare(target)
    classes = data.classes
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    y_test = data.y_test.values
    y_bin = label_binarize(y_test, classes=classes)

    rows = []
    roc_data = {}  # name -> (fpr, tpr, auc) micro
    pr_data = {}   # name -> (recall, precision, ap) micro

    for name in ALL_MODELS:
        pipe = _load(target, name)
        if pipe is None:
            print(f"[eval:{target}] missing model '{name}', skipping")
            continue
        y_pred = pipe.predict(data.X_test)
        proba = _proba_matrix(pipe, data.X_test, classes)

        row = {
            "model": name,
            "label": MODEL_LABELS[name],
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_macro_f1": f1_score(y_test, y_pred, average="macro",
                                      zero_division=0),
            "test_weighted_f1": f1_score(y_test, y_pred, average="weighted",
                                         zero_division=0),
            "test_macro_precision": precision_score(y_test, y_pred,
                                                     average="macro",
                                                     zero_division=0),
            "test_macro_recall": recall_score(y_test, y_pred, average="macro",
                                               zero_division=0),
        }

        # Urgency: report recall for the most-urgent class.
        if target == TARGET_URGENCY and "critical" in classes:
            row["recall_critical"] = recall_score(
                y_test, y_pred, labels=["critical"], average="macro",
                zero_division=0,
            )

        if proba is not None and y_bin.shape[1] == proba.shape[1]:
            row["roc_auc_ovr_macro"] = _safe_auc(y_bin, proba, "macro")
            row["pr_auc_macro"] = float(
                average_precision_score(y_bin, proba, average="macro")
            )
            # micro-average curves for the combined figures
            fpr, tpr, _ = roc_curve(y_bin.ravel(), proba.ravel())
            roc_data[name] = (fpr, tpr, _safe_auc(y_bin, proba, "micro"))
            prec, rec, _ = precision_recall_curve(y_bin.ravel(), proba.ravel())
            pr_data[name] = (rec, prec,
                             average_precision_score(y_bin, proba,
                                                     average="micro"))
        else:
            row["roc_auc_ovr_macro"] = float("nan")
            row["pr_auc_macro"] = float("nan")

        rows.append(row)

        # Per-model classification report.
        rep = classification_report(y_test, y_pred, labels=classes,
                                    zero_division=0)
        with open(os.path.join(TABLES_DIR,
                               f"classification_report_{target}_{name}.txt"),
                  "w") as fh:
            fh.write(f"{MODEL_LABELS[name]} - target: {target}\n\n{rep}\n")

        # Confusion matrix figure.
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ConfusionMatrixDisplay(cm, display_labels=classes).plot(
            ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
        ax.set_title(f"Confusion Matrix - {MODEL_LABELS[name]}\n({target})")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"confusion_{target}_{name}.png"),
                    bbox_inches="tight")
        plt.close(fig)

        # Neural learning curves (loss) where available.
        clf = pipe.named_steps.get("clf")
        _save_learning_curve(target, name, clf)

    # Combined ROC figure.
    if roc_data:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for name, (fpr, tpr, auc) in roc_data.items():
            ax.plot(fpr, tpr, label=f"{MODEL_LABELS[name]} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"ROC Curves (micro-average OvR) - {target}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"roc_{target}.png"),
                    bbox_inches="tight")
        plt.close(fig)

    # Combined PR figure.
    if pr_data:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for name, (rec, prec, ap) in pr_data.items():
            ax.plot(rec, prec, label=f"{MODEL_LABELS[name]} (AP={ap:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Curves (micro-average) - {target}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"pr_{target}.png"),
                    bbox_inches="tight")
        plt.close(fig)

    metrics_df = pd.DataFrame(rows)
    out = os.path.join(TABLES_DIR, f"test_metrics_{target}.csv")
    metrics_df.to_csv(out, index=False)
    print(f"[eval:{target}] test metrics -> {out}")
    cols = ["label", "test_accuracy", "test_macro_f1", "test_weighted_f1",
            "roc_auc_ovr_macro", "pr_auc_macro"]
    print(metrics_df[cols].to_string(index=False))
    return metrics_df


def _save_learning_curve(target, name, clf):
    """Plot a train/validation learning curve for neural models if available."""
    history = None
    # Keras wrapper exposes .history_
    if hasattr(clf, "history_") and hasattr(clf.history_, "history"):
        history = clf.history_.history
        loss = history.get("loss")
        val_loss = history.get("val_loss")
        if loss:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(range(1, len(loss) + 1), loss, label="train loss",
                    color=ACCENT)
            if val_loss:
                ax.plot(range(1, len(val_loss) + 1), val_loss,
                        label="val loss", color="#ff9500")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(f"Learning Curve - {MODEL_LABELS[name]} ({target})")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(FIG_DIR,
                                     f"learning_curve_{target}_{name}.png"),
                        bbox_inches="tight")
            plt.close(fig)
            return
    # sklearn MLP exposes loss_curve_ and validation_scores_
    if hasattr(clf, "loss_curve_"):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(1, len(clf.loss_curve_) + 1), clf.loss_curve_,
                label="train loss", color=ACCENT)
        if getattr(clf, "validation_scores_", None):
            ax2 = ax.twinx()
            vs = clf.validation_scores_
            ax2.plot(range(1, len(vs) + 1), vs, label="val score",
                     color="#ff9500")
            ax2.set_ylabel("Validation score")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training loss")
        ax.set_title(f"Learning Curve - {MODEL_LABELS[name]} ({target})")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR,
                                 f"learning_curve_{target}_{name}.png"),
                    bbox_inches="tight")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=[TARGET_SERVICE, TARGET_URGENCY, "both"],
                    default="both")
    args = ap.parse_args()
    targets = ([TARGET_SERVICE, TARGET_URGENCY] if args.target == "both"
               else [args.target])
    for t in targets:
        evaluate_target(t)


if __name__ == "__main__":
    main()
