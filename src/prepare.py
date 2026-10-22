"""Data preparation for the Civic & Municipality Complaint Classification System.

This module is the single source of truth for:
  * loading the raw CSV,
  * cleaning / canonicalising labels and categorical values,
  * excluding leakage columns and personal information (POPIA),
  * engineering intake-only features,
  * building a reproducible scikit-learn preprocessing pipeline (ColumnTransformer),
  * producing a stratified 80/20 train/test split with a fixed seed.

All transformations that *learn* from data (TF-IDF vocabulary, imputation
statistics, scaling, one-hot categories) live inside the ColumnTransformer so
they are fit on training folds only and never see the test set. This prevents
data leakage end to end.

Nothing here is model-specific: the same prepared data and preprocessor are
reused by train.py, evaluate.py, optimise.py and app.py.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.20

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "municipal_training_set_1100.csv")

# Columns removed because they encode the target or are recorded after intake.
LEAKAGE_COLUMNS = [
    "issue_status",
    "municipal_department",
    "resolution_datetime",
    "resolution_notes",
    "priority_score",   # triage-derived; near-disjoint ranges per priority_level
    "severity_score",   # strongly encodes priority_level
]

# Personal information removed/anonymised under POPIA.
PII_COLUMNS = [
    "resident_id",
    "resident_name",
    "resident_email",
    "resident_phone",
    "location_street_address",
    "latitude",
    "longitude",
    "photo_url",
    "location_postal_code",   # near-identifying, high cardinality
    "location_country",       # constant (USA), no signal
]

# Intake feature groups.
TEXT_FEATURE = "issue_description"
CATEGORICAL_FEATURES = ["location_city", "location_state", "is_anonymous"]
NUMERIC_FEATURES = [
    "area_importance",
    "citizen_reports_count",
    "report_hour",
    "report_dayofweek",
    "report_month",
]

ALL_FEATURE_COLUMNS = [TEXT_FEATURE] + CATEGORICAL_FEATURES + NUMERIC_FEATURES

TARGET_SERVICE = "issue_type"
TARGET_URGENCY = "priority_level"
URGENCY_ORDER = ["low", "medium", "high", "critical"]

# Canonical mapping for inconsistent service-type labels.
ISSUE_TYPE_CANONICAL = {
    "water-leak": "water_leakage",
    "illegal-dumping": "garbage",
}

# Canonical mapping for inconsistent state values.
STATE_CANONICAL = {
    "California": "CA",
    "New Jersey": "NJ",
}


@dataclass
class PreparedData:
    """Container for a prepared train/test split for one target."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    target_name: str
    classes: List[str]
    feature_columns: List[str] = field(default_factory=lambda: list(ALL_FEATURE_COLUMNS))


# --------------------------------------------------------------------------- #
# Loading & cleaning
# --------------------------------------------------------------------------- #
def load_raw(data_path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV without any transformation."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. Place the CSV at the project root."
        )
    return pd.read_csv(data_path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and canonicalise the raw dataframe.

    Steps:
      * drop duplicate report_id rows,
      * canonicalise issue_type and location_state spellings,
      * parse the mixed-format datetime and engineer intake-only time features,
      * normalise is_anonymous to a string category,
      * strip whitespace from the complaint text.

    The returned frame still contains the target columns; feature selection
    happens in `build_feature_frame`.
    """
    df = df.copy()

    # Drop duplicated complaint ids (keep first occurrence).
    if "report_id" in df.columns:
        df = df.drop_duplicates(subset="report_id", keep="first")

    # Canonicalise labels / categories.
    if TARGET_SERVICE in df.columns:
        df[TARGET_SERVICE] = (
            df[TARGET_SERVICE].astype(str).str.strip().replace(ISSUE_TYPE_CANONICAL)
        )
    if "location_state" in df.columns:
        df["location_state"] = (
            df["location_state"].astype(str).str.strip().replace(STATE_CANONICAL)
        )
    if "location_city" in df.columns:
        df["location_city"] = df["location_city"].astype(str).str.strip()

    # Intake timestamp -> engineered features (intake time only; no leakage).
    if "report_datetime" in df.columns:
        dt = pd.to_datetime(df["report_datetime"], errors="coerce", format="mixed")
        df["report_hour"] = dt.dt.hour
        df["report_dayofweek"] = dt.dt.dayofweek
        df["report_month"] = dt.dt.month

    # Normalise boolean-ish anonymity flag to a clean string category.
    if "is_anonymous" in df.columns:
        df["is_anonymous"] = df["is_anonymous"].astype(str).str.strip().str.lower()

    # Clean complaint text.
    if TEXT_FEATURE in df.columns:
        df[TEXT_FEATURE] = df[TEXT_FEATURE].fillna("").astype(str).str.strip()

    return df


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the intake feature columns, dropping leakage + PII columns."""
    missing = [c for c in ALL_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Expected engineered feature columns missing: {missing}")
    return df[ALL_FEATURE_COLUMNS].copy()


# --------------------------------------------------------------------------- #
# Preprocessing pipeline
# --------------------------------------------------------------------------- #
def build_preprocessor(
    max_text_features: int = 3000,
    ngram_range=(1, 2),
    min_df: int = 2,
    include_text: bool = True,
    include_location: bool = True,
) -> ColumnTransformer:
    """Build the reproducible preprocessing ColumnTransformer.

    Parameters allow ablations (dropping text or location) while keeping every
    other setting fixed. All statistics are learned at fit time only.
    """
    transformers = []

    if include_text:
        text_pipe = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=max_text_features,
                        ngram_range=ngram_range,
                        min_df=min_df,
                        sublinear_tf=True,
                        strip_accents="unicode",
                        lowercase=True,
                        stop_words="english",
                    ),
                )
            ]
        )
        transformers.append(("text", text_pipe, TEXT_FEATURE))

    categorical_cols = list(CATEGORICAL_FEATURES)
    if not include_location:
        categorical_cols = [c for c in categorical_cols if c not in ("location_city", "location_state")]

    if categorical_cols:
        cat_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
            ]
        )
        transformers.append(("categorical", cat_pipe, categorical_cols))

    num_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    transformers.append(("numeric", num_pipe, NUMERIC_FEATURES))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
    )


# --------------------------------------------------------------------------- #
# Prepared split
# --------------------------------------------------------------------------- #
def prepare(
    target: str = TARGET_SERVICE,
    data_path: str = DEFAULT_DATA_PATH,
    min_class_count: int = 2,
) -> PreparedData:
    """Load, clean, and produce a stratified 80/20 split for the chosen target.

    `min_class_count` drops extremely rare classes that cannot be stratified
    across a train/test split (needs >= 2 members). Dropped classes are logged.
    """
    if target not in (TARGET_SERVICE, TARGET_URGENCY):
        raise ValueError(f"Unsupported target '{target}'.")

    df = clean(load_raw(data_path))
    df = df[df[target].notna()].copy()

    # Drop classes too rare to stratify.
    counts = df[target].value_counts()
    keep = counts[counts >= max(min_class_count, 2)].index
    dropped = sorted(set(counts.index) - set(keep))
    if dropped:
        print(f"[prepare] Dropping rare {target} classes (<{min_class_count}): {dropped}")
    df = df[df[target].isin(keep)].copy()

    X = build_feature_frame(df)
    y = df[target].astype(str).reset_index(drop=True)
    X = X.reset_index(drop=True)

    if target == TARGET_URGENCY:
        classes = [c for c in URGENCY_ORDER if c in set(y)]
    else:
        classes = sorted(y.unique())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return PreparedData(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        target_name=target,
        classes=classes,
    )


AUDIT_CACHE_PATH = os.path.join(PROJECT_ROOT, "reports", "audit_summary.json")


def audit_summary(data_path: str = DEFAULT_DATA_PATH) -> dict:
    """Return a dictionary of aggregate audit facts for the app Home / EDA pages.

    When the raw dataset is available, this computes the summary from it and
    caches it (aggregate counts only, no personal data) to
    ``reports/audit_summary.json``. When the raw dataset is absent — e.g. on a
    public cloud deployment where the CSV is withheld for POPIA — it falls back
    to the cached JSON so the interface still works.
    """
    if not os.path.exists(data_path):
        if os.path.exists(AUDIT_CACHE_PATH):
            import json
            with open(AUDIT_CACHE_PATH) as fh:
                return json.load(fh)
        raise FileNotFoundError(
            f"Raw dataset not found at '{data_path}' and no cached audit summary "
            f"at '{AUDIT_CACHE_PATH}'. Run `python -m src.prepare` with the "
            "dataset present to generate the cache."
        )

    raw = load_raw(data_path)
    cleaned = clean(raw)
    dt = pd.to_datetime(raw["report_datetime"], errors="coerce", format="mixed")
    summary = {
        "n_rows_raw": int(raw.shape[0]),
        "n_cols_raw": int(raw.shape[1]),
        "n_rows_clean": int(cleaned.drop_duplicates(subset="report_id").shape[0]),
        "duplicate_report_ids": int(raw["report_id"].duplicated().sum()),
        "missing_by_column": raw.isna().sum().to_dict(),
        "service_distribution": cleaned[TARGET_SERVICE].value_counts().to_dict(),
        "urgency_distribution": cleaned[TARGET_URGENCY].value_counts().to_dict(),
        "date_min": None if dt.isna().all() else str(dt.min()),
        "date_max": None if dt.isna().all() else str(dt.max()),
        "leakage_columns": list(LEAKAGE_COLUMNS),
        "pii_columns": list(PII_COLUMNS),
        "feature_columns": list(ALL_FEATURE_COLUMNS),
    }

    # Cache aggregate summary (no personal data) so the app can run without the
    # raw dataset, e.g. on a public cloud deployment.
    try:
        import json
        os.makedirs(os.path.dirname(AUDIT_CACHE_PATH), exist_ok=True)
        with open(AUDIT_CACHE_PATH, "w") as fh:
            json.dump(summary, fh, indent=2)
    except OSError:
        pass

    return summary


if __name__ == "__main__":
    import json

    print("=== Audit summary ===")
    summary = audit_summary()
    print(json.dumps({k: v for k, v in summary.items() if k != "missing_by_column"}, indent=2))
    for tgt in (TARGET_SERVICE, TARGET_URGENCY):
        pd_ = prepare(tgt)
        print(
            f"\n[{tgt}] train={pd_.X_train.shape} test={pd_.X_test.shape} "
            f"classes={pd_.classes}"
        )
