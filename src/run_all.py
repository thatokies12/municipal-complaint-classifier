"""Convenience runner: execute the full reproducible pipeline in order.

    python -m src.run_all

Runs EDA -> train (5-fold CV) -> evaluate (held-out test) -> optimise
(RandomizedSearch + ablations + McNemar + save final pipeline) for both targets.
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from src import eda, evaluate, optimise, train
from src.prepare import TARGET_SERVICE, TARGET_URGENCY


def main():
    print(">>> [1/4] Exploratory data analysis")
    eda.generate_all()

    for target in (TARGET_SERVICE, TARGET_URGENCY):
        print(f"\n>>> [2/4] Train + 5-fold CV ({target})")
        train.train_target(target, do_cv=True)
        print(f"\n>>> [3/4] Evaluate on held-out test ({target})")
        evaluate.evaluate_target(target)
        print(f"\n>>> [4/4] Optimise + ablations + McNemar ({target})")
        optimise.optimise_target(target)

    print("\nDone. Artefacts in models/ and reports/. Launch the app with:")
    print("    streamlit run src/app.py")


if __name__ == "__main__":
    main()
