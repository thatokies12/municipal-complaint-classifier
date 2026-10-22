"""Build lightweight, TensorFlow-free pipelines for public deployment.

The academic `final_pipeline.joblib` for urgency is a Keras model, which makes
cloud hosting slow and fragile (large install + version drift). For deployment
we save a `deploy_pipeline.joblib` per target that uses only scikit-learn
estimators, so the hosted app loads fast with no TensorFlow dependency:

  * service category -> Random Forest (the academic champion; already TF-free)
  * urgency          -> scikit-learn MLP (macro-F1 ~0.808 vs Keras 0.815;
                        statistically indistinguishable per McNemar)

This does NOT change the academic artefacts or reported results — it only adds
a deployment-friendly copy the interface prefers when present.

Run:  python -m src.build_deploy_models
"""
from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore")

import joblib
from sklearn.base import clone
from sklearn.metrics import f1_score

from .models import build_pipeline
from .prepare import PROJECT_ROOT, TARGET_SERVICE, TARGET_URGENCY, prepare

MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")

# TensorFlow-free model choice per target.
DEPLOY_MODEL = {
    TARGET_SERVICE: "rf",   # academic champion, already sklearn
    TARGET_URGENCY: "mlp",  # sklearn MLP stand-in for the Keras tabnet
}


def build(target: str) -> str:
    data = prepare(target)
    name = DEPLOY_MODEL[target]
    pipe = clone(build_pipeline(name))
    pipe.fit(data.X_train, data.y_train)

    f1 = f1_score(data.y_test, pipe.predict(data.X_test),
                  average="macro", zero_division=0)
    out = os.path.join(MODELS_ROOT, target, "deploy_pipeline.joblib")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    joblib.dump(pipe, out)
    print(f"[deploy] {target}: {name} (test macro-F1={f1:.3f}) -> {out}")
    return out


def main():
    for t in (TARGET_SERVICE, TARGET_URGENCY):
        build(t)
    print("Deployment pipelines ready (TensorFlow-free).")


if __name__ == "__main__":
    main()
