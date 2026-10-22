"""Tests for data preparation, cleaning, and leakage/PII exclusion."""
import pandas as pd
import pytest

from src import prepare


def test_canonicalisation_merges_variant_labels():
    raw = prepare.load_raw()
    cleaned = prepare.clean(raw)
    labels = set(cleaned[prepare.TARGET_SERVICE].unique())
    # Variant spellings must be merged away.
    assert "water-leak" not in labels
    assert "illegal-dumping" not in labels
    assert "water_leakage" in labels


def test_state_canonicalisation():
    raw = prepare.load_raw()
    cleaned = prepare.clean(raw)
    states = set(cleaned["location_state"].unique())
    assert "California" not in states
    assert "CA" in states


def test_engineered_time_features_exist():
    raw = prepare.load_raw()
    cleaned = prepare.clean(raw)
    for col in ("report_hour", "report_dayofweek", "report_month"):
        assert col in cleaned.columns


def test_feature_frame_excludes_leakage_and_pii():
    raw = prepare.load_raw()
    cleaned = prepare.clean(raw)
    X = prepare.build_feature_frame(cleaned)
    forbidden = set(prepare.LEAKAGE_COLUMNS) | set(prepare.PII_COLUMNS)
    assert not (set(X.columns) & forbidden), "feature frame leaked a forbidden column"
    assert list(X.columns) == prepare.ALL_FEATURE_COLUMNS


@pytest.mark.parametrize("target", [prepare.TARGET_SERVICE, prepare.TARGET_URGENCY])
def test_prepare_split_is_stratified_and_sized(target):
    data = prepare.prepare(target)
    n_total = len(data.X_train) + len(data.X_test)
    # ~20% test split.
    assert abs(len(data.X_test) / n_total - prepare.TEST_SIZE) < 0.03
    # Every class present in train is present in test (stratification).
    assert set(data.y_test.unique()).issubset(set(data.y_train.unique()))
    # No target column leaked into the feature matrix.
    assert target not in data.X_train.columns


def test_prepare_is_reproducible():
    a = prepare.prepare(prepare.TARGET_SERVICE)
    b = prepare.prepare(prepare.TARGET_SERVICE)
    assert a.X_train.index.equals(b.X_train.index)
    assert (a.y_train.values == b.y_train.values).all()


def test_audit_summary_keys():
    summary = prepare.audit_summary()
    for key in ("n_rows_clean", "service_distribution", "urgency_distribution",
                "leakage_columns", "pii_columns"):
        assert key in summary
