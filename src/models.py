"""Model factory for the complaint classification system.

Defines the naive baseline and the four substantive models required by the
assignment, each wrapped in a full scikit-learn Pipeline that begins with the
shared preprocessor from ``prepare.py``:

    baseline  : DummyClassifier(strategy="most_frequent")
    logreg    : Logistic Regression (L2, class_weight balanced)
    rf        : Random Forest (fixed n_estimators + controlled depth)
    mlp       : scikit-learn MLPClassifier (primary neural model)
    tabnet    : a second feed-forward tabular neural network.
                Uses Keras (TensorFlow) when available with ReLU, dropout,
                L2 weight decay, Adam, cross-entropy loss and early stopping.
                Falls back to a *differently configured* sklearn MLP when
                TensorFlow is unavailable, so the whole project still runs.

Every estimator receives identical preprocessed input, satisfying the
"comparable input information" requirement. class_weight / sample weighting is
applied to the training data only (inside the estimators / fit calls).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from .prepare import RANDOM_STATE, build_preprocessor

# --------------------------------------------------------------------------- #
# Optional TensorFlow / Keras detection
# --------------------------------------------------------------------------- #
from .keras_net import LabelEncodedClassifier

try:  # pragma: no cover - exercised implicitly depending on environment
    import tensorflow as tf  # noqa: F401

    from .keras_net import KerasTabularClassifier

    _HAS_KERAS = True
except Exception:  # noqa: BLE001 - any import failure means fall back
    _HAS_KERAS = False


def keras_available() -> bool:
    """Return True when a Keras-based tabular network can be built."""
    return _HAS_KERAS


def _make_keras_classifier():
    """The assignment's second neural model: a Keras feed-forward tabular net.

    Dense input is required; the surrounding Pipeline converts the sparse
    preprocessor output via ``to_dense`` (see build_pipeline).
    """
    return KerasTabularClassifier(
        n_hidden=2,
        n_neurons=128,
        dropout=0.3,
        l2=1e-4,
        learning_rate=1e-3,
        epochs=50,
        batch_size=32,
        patience=5,
        validation_split=0.15,
        random_state=RANDOM_STATE,
        verbose=0,
    )


# --------------------------------------------------------------------------- #
# Dense conversion helper (TF-IDF produces a sparse matrix; MLPs want dense)
# --------------------------------------------------------------------------- #
from sklearn.preprocessing import FunctionTransformer


def _to_dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def _dense_step():
    return ("to_dense", FunctionTransformer(_to_dense, accept_sparse=True))


# --------------------------------------------------------------------------- #
# Pipeline builders
# --------------------------------------------------------------------------- #
def build_pipeline(name: str, include_text: bool = True,
                   include_location: bool = True,
                   use_class_weight: bool = True) -> Pipeline:
    """Build a full preprocess + estimator pipeline for ``name``.

    ``include_text`` / ``include_location`` / ``use_class_weight`` expose the
    three ablation switches while keeping all other settings fixed.
    """
    pre = build_preprocessor(include_text=include_text,
                             include_location=include_location)
    cw = "balanced" if use_class_weight else None

    if name == "baseline":
        estimator = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
        return Pipeline([("pre", pre), ("clf", estimator)])

    if name == "logreg":
        estimator = LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=2000,
            class_weight=cw,
            random_state=RANDOM_STATE,
        )
        return Pipeline([("pre", pre), ("clf", estimator)])

    if name == "rf":
        estimator = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample" if use_class_weight else None,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        return Pipeline([("pre", pre), ("clf", estimator)])

    if name == "mlp":
        estimator = LabelEncodedClassifier(MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,            # L2 weight decay
            batch_size=32,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
            random_state=RANDOM_STATE,
        ))
        return Pipeline([("pre", pre), _dense_step(), ("clf", estimator)])

    if name == "tabnet":
        if _HAS_KERAS:
            estimator = _make_keras_classifier()
            return Pipeline([("pre", pre), _dense_step(), ("clf", estimator)])
        # Fallback: a second, differently-configured sklearn MLP (single wider
        # hidden layer, stronger regularisation) so it remains a distinct model.
        estimator = LabelEncodedClassifier(MLPClassifier(
            hidden_layer_sizes=(256,),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            batch_size=32,
            learning_rate_init=5e-4,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
            random_state=RANDOM_STATE,
        ))
        return Pipeline([("pre", pre), _dense_step(), ("clf", estimator)])

    raise ValueError(f"Unknown model name '{name}'.")


# Human-readable labels for reports / the UI.
MODEL_LABELS: Dict[str, str] = {
    "baseline": "Dummy baseline (most-frequent)",
    "logreg": "Logistic Regression",
    "rf": "Random Forest",
    "mlp": "MLP (neural)",
    "tabnet": "Tabular Neural Network" + ("" if _HAS_KERAS else " (sklearn fallback)"),
}

SUBSTANTIVE_MODELS = ["logreg", "rf", "mlp", "tabnet"]
ALL_MODELS = ["baseline"] + SUBSTANTIVE_MODELS
