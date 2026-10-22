"""Inference helpers used by the Streamlit app.

Loads the *saved, complete* preprocessing-and-model pipeline (never a manually
recreated partial pipeline) and turns raw intake form values into the exact
feature frame the pipeline expects. Also provides input validation that mirrors
the assignment's responsible-use requirements.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from .prepare import (
    ALL_FEATURE_COLUMNS,
    PROJECT_ROOT,
    TARGET_SERVICE,
    TARGET_URGENCY,
)

MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")

# Reasonable defaults for optional numeric intake fields (dataset medians).
DEFAULTS = {
    "location_city": "New York",
    "location_state": "NY",
    "is_anonymous": "false",
    "area_importance": 7,
    "citizen_reports_count": 20,
    "report_hour": 12,
    "report_dayofweek": 2,
    "report_month": 6,
}

MIN_TEXT_CHARS = 12
MIN_TEXT_WORDS = 3


@dataclass
class ValidationResult:
    ok: bool
    message: str = ""


def validate_complaint(text: str) -> ValidationResult:
    """Reject empty, too-short, or non-textual complaints.

    Mirrors the assignment's invalid-case requirement (e.g. empty text or
    "12345" must be rejected with a clear message).
    """
    if text is None:
        return ValidationResult(False, "Please enter a valid municipal complaint.")
    stripped = text.strip()
    if not stripped:
        return ValidationResult(False, "Please enter a valid municipal complaint.")
    if len(stripped) < MIN_TEXT_CHARS:
        return ValidationResult(
            False,
            f"Complaint is too short. Please provide at least {MIN_TEXT_CHARS} "
            "characters describing the issue.",
        )
    letters = re.findall(r"[A-Za-z]", stripped)
    words = [w for w in re.split(r"\s+", stripped) if w]
    if len(letters) < 5 or len(words) < MIN_TEXT_WORDS:
        return ValidationResult(
            False,
            "Input does not look like a complaint description. Please enter a "
            "valid municipal complaint (e.g. 'Water pipe burst flooding the road').",
        )
    return ValidationResult(True)


def build_input_frame(
    complaint_text: str,
    location_city: Optional[str] = None,
    location_state: Optional[str] = None,
    is_anonymous: bool = False,
    area_importance: Optional[float] = None,
    citizen_reports_count: Optional[float] = None,
    report_hour: Optional[int] = None,
    report_dayofweek: Optional[int] = None,
    report_month: Optional[int] = None,
) -> pd.DataFrame:
    """Assemble a one-row DataFrame with exactly the pipeline's feature columns."""
    row = {
        "issue_description": complaint_text.strip(),
        "location_city": location_city or DEFAULTS["location_city"],
        "location_state": location_state or DEFAULTS["location_state"],
        "is_anonymous": "true" if is_anonymous else "false",
        "area_importance": (area_importance if area_importance is not None
                            else DEFAULTS["area_importance"]),
        "citizen_reports_count": (citizen_reports_count
                                  if citizen_reports_count is not None
                                  else DEFAULTS["citizen_reports_count"]),
        "report_hour": (report_hour if report_hour is not None
                        else DEFAULTS["report_hour"]),
        "report_dayofweek": (report_dayofweek if report_dayofweek is not None
                             else DEFAULTS["report_dayofweek"]),
        "report_month": (report_month if report_month is not None
                         else DEFAULTS["report_month"]),
    }
    return pd.DataFrame([row], columns=ALL_FEATURE_COLUMNS)


def model_path(target: str) -> str:
    return os.path.join(MODELS_ROOT, target, "final_pipeline.joblib")


def meta_path(target: str) -> str:
    return os.path.join(MODELS_ROOT, target, "final_pipeline_meta.json")


def load_pipeline(target: str):
    """Load the saved final pipeline for a target, or None if missing."""
    path = model_path(target)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def load_meta(target: str) -> Optional[dict]:
    path = meta_path(target)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def predict(pipe, X: pd.DataFrame) -> Dict:
    """Return the predicted label and a sorted probability dict (if available)."""
    label = pipe.predict(X)[0]
    probs = None
    if hasattr(pipe, "predict_proba"):
        proba = np.asarray(pipe.predict_proba(X))[0]
        classes = list(pipe.named_steps["clf"].classes_)
        probs = dict(sorted(zip(classes, map(float, proba)),
                            key=lambda kv: kv[1], reverse=True))
    return {"label": str(label), "probabilities": probs}


def available_targets() -> List[str]:
    """Return targets that have a saved final pipeline."""
    return [t for t in (TARGET_SERVICE, TARGET_URGENCY)
            if os.path.exists(model_path(t))]
