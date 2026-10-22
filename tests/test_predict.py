"""Tests for input validation, feature-frame assembly, and prediction."""
import os

import pytest

from src import predict
from src.prepare import ALL_FEATURE_COLUMNS, TARGET_SERVICE, TARGET_URGENCY


# --------------------------- validation ---------------------------------- #
@pytest.mark.parametrize("bad", ["", "   ", "12345", "hi", "!!!", "1 2 3"])
def test_validation_rejects_invalid(bad):
    assert predict.validate_complaint(bad).ok is False


@pytest.mark.parametrize("good", [
    "Main water pipe has burst and flooding is occurring.",
    "Streetlight has not been working for several days.",
    "Garbage collection was missed this week.",
])
def test_validation_accepts_valid(good):
    assert predict.validate_complaint(good).ok is True


def test_validation_none_input():
    assert predict.validate_complaint(None).ok is False


# --------------------------- feature frame ------------------------------- #
def test_build_input_frame_has_exact_columns():
    X = predict.build_input_frame("Pothole on main road near the school.")
    assert list(X.columns) == ALL_FEATURE_COLUMNS
    assert len(X) == 1


def test_build_input_frame_anonymous_flag():
    X = predict.build_input_frame("Broken drain cover.", is_anonymous=True)
    assert X.loc[0, "is_anonymous"] == "true"


# --------------------------- prediction ---------------------------------- #
pipelines_exist = all(
    os.path.exists(predict.model_path(t)) for t in (TARGET_SERVICE, TARGET_URGENCY)
)


@pytest.mark.skipif(not pipelines_exist,
                    reason="final pipelines not trained yet (run src.train/optimise)")
@pytest.mark.parametrize("target", [TARGET_SERVICE, TARGET_URGENCY])
def test_prediction_returns_valid_label(target):
    pipe = predict.load_pipeline(target)
    meta = predict.load_meta(target)
    X = predict.build_input_frame("Main water pipe has burst and flooding is occurring.")
    result = predict.predict(pipe, X)
    assert result["label"] in meta["classes"]
    if result["probabilities"] is not None:
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 1e-3  # probabilities sum to 1


@pytest.mark.skipif(not pipelines_exist, reason="final pipelines not trained yet")
def test_water_complaint_predicts_water_category():
    """A clear water complaint should map to a water-related service category.

    This checks the saved model behaves sensibly; it does NOT hardcode the
    answer (the label comes from the trained pipeline).
    """
    pipe = predict.load_pipeline(TARGET_SERVICE)
    X = predict.build_input_frame(
        "Main water pipe has burst and water is flooding the street.")
    result = predict.predict(pipe, X)
    assert "water" in result["label"].lower()
