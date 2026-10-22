"""A self-contained Keras feed-forward tabular network with a scikit-learn API.

We implement our own thin wrapper instead of depending on scikeras, because
scikeras 0.13 is not yet compatible with scikit-learn 1.6's ``__sklearn_tags__``
API. The wrapper implements the small surface the project actually uses
(fit / predict / predict_proba / get_params / set_params / clone support) and
plugs cleanly into sklearn Pipelines, cross_val_score and clone().

Architecture / training settings follow the assignment:
  * ReLU hidden layers, softmax output
  * Dropout + L2 weight decay
  * Adam optimiser, sparse categorical cross-entropy loss
  * up to ``epochs`` with EarlyStopping on validation loss (patience=5)
  * fixed random seed for reproducibility
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PYTHONHASHSEED", "42")

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels


class KerasTabularClassifier(BaseEstimator, ClassifierMixin):
    """Feed-forward tabular neural network (Keras) with an sklearn interface."""

    def __init__(
        self,
        n_hidden: int = 2,
        n_neurons: int = 128,
        dropout: float = 0.3,
        l2: float = 1e-4,
        learning_rate: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        patience: int = 5,
        validation_split: float = 0.15,
        random_state: int = 42,
        verbose: int = 0,
    ):
        self.n_hidden = n_hidden
        self.n_neurons = n_neurons
        self.dropout = dropout
        self.l2 = l2
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.validation_split = validation_split
        self.random_state = random_state
        self.verbose = verbose

    # --- internals ------------------------------------------------------- #
    def _build(self, n_features: int, n_classes: int):
        import tensorflow as tf
        from tensorflow.keras import layers, models, regularizers

        tf.keras.utils.set_random_seed(self.random_state)
        net = models.Sequential()
        net.add(layers.Input(shape=(n_features,)))
        for _ in range(int(self.n_hidden)):
            net.add(
                layers.Dense(
                    int(self.n_neurons),
                    activation="relu",
                    kernel_regularizer=regularizers.l2(self.l2),
                )
            )
            net.add(layers.Dropout(self.dropout))
        net.add(layers.Dense(n_classes, activation="softmax"))
        net.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return net

    @staticmethod
    def _to_dense(X):
        return X.toarray() if hasattr(X, "toarray") else np.asarray(X, dtype="float32")

    # --- sklearn API ----------------------------------------------------- #
    def fit(self, X, y):
        import tensorflow as tf

        X = self._to_dense(X).astype("float32")
        self.classes_ = unique_labels(y)
        self._class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([self._class_to_idx[v] for v in y], dtype="int64")

        self.n_features_in_ = X.shape[1]
        self.model_ = self._build(self.n_features_in_, len(self.classes_))

        callbacks = []
        val_split = self.validation_split
        # EarlyStopping requires a validation split with enough samples.
        if val_split and X.shape[0] * val_split >= len(self.classes_):
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.patience,
                    restore_best_weights=True,
                )
            )
        else:
            val_split = 0.0

        self.history_ = self.model_.fit(
            X,
            y_idx,
            epochs=int(self.epochs),
            batch_size=int(self.batch_size),
            validation_split=val_split,
            callbacks=callbacks,
            verbose=self.verbose,
        )
        return self

    def predict_proba(self, X):
        X = self._to_dense(X).astype("float32")
        return np.asarray(self.model_.predict(X, verbose=0))

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return np.asarray([self.classes_[i] for i in idx])


class LabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    """Wrap a classifier so it always receives integer-encoded labels.

    scikit-learn's MLPClassifier with ``early_stopping=True`` internally calls
    ``np.isnan`` on predictions, which raises ``TypeError`` when the target is
    string-valued. Encoding the labels to integers before fitting avoids this
    while preserving the original string classes on the public API
    (``classes_``, ``predict``, ``predict_proba``).
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        from sklearn.base import clone
        from sklearn.preprocessing import LabelEncoder

        self.encoder_ = LabelEncoder()
        y_enc = self.encoder_.fit_transform(y)
        self.classes_ = self.encoder_.classes_
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y_enc)
        return self

    def predict(self, X):
        pred = self.estimator_.predict(X)
        return self.encoder_.inverse_transform(pred)

    def predict_proba(self, X):
        # estimator_.classes_ are integer-encoded 0..k-1 aligned with encoder_
        return self.estimator_.predict_proba(X)

    @property
    def loss_curve_(self):  # expose for learning-curve plotting
        return getattr(self.estimator_, "loss_curve_", None)

    @property
    def validation_scores_(self):
        return getattr(self.estimator_, "validation_scores_", None)
