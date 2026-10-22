"""Tests for the McNemar's-test implementation."""
import numpy as np

from src.optimise import mcnemar


def test_mcnemar_identical_predictions_not_significant():
    y = np.array(["a", "b", "a", "b", "a"])
    res = mcnemar(y, y.copy(), y.copy())
    assert res["n_discordant"] == 0
    assert res["significant_at_0.05"] is False


def test_mcnemar_counts_discordant_pairs():
    y = np.array(["a", "a", "a", "a"])
    champ = np.array(["a", "a", "b", "b"])   # 2 wrong
    comp = np.array(["b", "b", "a", "a"])    # 2 wrong, all discordant
    res = mcnemar(y, champ, comp)
    assert res["both_correct"] == 0
    assert res["champion_correct_comparator_wrong"] == 2
    assert res["champion_wrong_comparator_correct"] == 2
    assert res["n_discordant"] == 4


def test_mcnemar_uses_exact_binomial_for_small_discordant():
    y = np.array(["a"] * 10)
    champ = np.array(["a"] * 10)
    comp = np.array(["a"] * 8 + ["b", "b"])  # comparator wrong on 2
    res = mcnemar(y, champ, comp)
    assert res["method"] == "exact binomial"
    assert 0.0 <= res["p_value"] <= 1.0
