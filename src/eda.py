"""Exploratory data analysis for the municipal complaints dataset.

Generates real figures saved to reports/figures/ from the actual data. Every
chart is derived from the cleaned dataset (``prepare.clean``) so the figures
match the features and labels the models actually use.

Figures produced:
  1. service_distribution.png      - complaints by service category
  2. urgency_distribution.png      - complaints by urgency level
  3. missingness.png               - missing values by column
  4. service_vs_urgency.png        - service category x urgency heatmap
  5. complaints_over_time.png      - monthly complaint volume
  6. text_length.png               - complaint text-length distribution
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # headless backend for reproducible file output
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .prepare import (
    DEFAULT_DATA_PATH,
    PROJECT_ROOT,
    TARGET_SERVICE,
    TARGET_URGENCY,
    TEXT_FEATURE,
    URGENCY_ORDER,
    clean,
    load_raw,
)

FIG_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
ACCENT = "#0a84ff"  # macOS system blue

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
})


def _ensure_dir() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)


def _save(fig, name: str) -> str:
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_service_distribution(df: pd.DataFrame) -> str:
    counts = df[TARGET_SERVICE].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(counts.index, counts.values, color=ACCENT)
    ax.set_title("Distribution of Complaints by Service Category")
    ax.set_ylabel("Number of complaints")
    ax.set_xticklabels(counts.index, rotation=35, ha="right")
    return _save(fig, "service_distribution.png")


def plot_urgency_distribution(df: pd.DataFrame) -> str:
    order = [u for u in URGENCY_ORDER if u in set(df[TARGET_URGENCY])]
    counts = df[TARGET_URGENCY].value_counts().reindex(order)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color=ACCENT)
    ax.set_title("Distribution of Complaints by Urgency Level")
    ax.set_ylabel("Number of complaints")
    return _save(fig, "urgency_distribution.png")


def plot_missingness(raw: pd.DataFrame) -> str:
    miss = raw.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(miss.index[::-1], (miss.values[::-1] * 100), color="#8e8e93")
    ax.set_title("Missing Values by Column (%)")
    ax.set_xlabel("Percent missing")
    return _save(fig, "missingness.png")


def plot_service_vs_urgency(df: pd.DataFrame) -> str:
    order = [u for u in URGENCY_ORDER if u in set(df[TARGET_URGENCY])]
    ct = pd.crosstab(df[TARGET_SERVICE], df[TARGET_URGENCY]).reindex(columns=order)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(ct.values, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(ct.columns)))
    ax.set_xticklabels(ct.columns)
    ax.set_yticks(range(len(ct.index)))
    ax.set_yticklabels(ct.index)
    ax.set_title("Service Category vs Urgency")
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            ax.text(j, i, int(ct.values[i, j]), ha="center", va="center",
                    color="black", fontsize=8)
    fig.colorbar(im, ax=ax, label="Count")
    ax.grid(False)
    return _save(fig, "service_vs_urgency.png")


def plot_complaints_over_time(raw: pd.DataFrame) -> str:
    dt = pd.to_datetime(raw["report_datetime"], errors="coerce", format="mixed")
    monthly = dt.dt.to_period("M").value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot([str(p) for p in monthly.index], monthly.values,
            marker="o", color=ACCENT)
    ax.set_title("Complaint Volume Over Time (monthly)")
    ax.set_ylabel("Number of complaints")
    ax.set_xticklabels([str(p) for p in monthly.index], rotation=60, ha="right",
                       fontsize=7)
    return _save(fig, "complaints_over_time.png")


def plot_text_length(df: pd.DataFrame) -> str:
    lengths = df[TEXT_FEATURE].astype(str).str.len()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lengths, bins=30, color=ACCENT, alpha=0.85)
    ax.set_title("Complaint Text-Length Distribution")
    ax.set_xlabel("Characters")
    ax.set_ylabel("Number of complaints")
    return _save(fig, "text_length.png")


def generate_all(data_path: str = DEFAULT_DATA_PATH) -> list:
    _ensure_dir()
    raw = load_raw(data_path)
    df = clean(raw).drop_duplicates(subset="report_id")
    paths = [
        plot_service_distribution(df),
        plot_urgency_distribution(df),
        plot_missingness(raw),
        plot_service_vs_urgency(df),
        plot_complaints_over_time(raw),
        plot_text_length(df),
    ]
    return paths


if __name__ == "__main__":
    for p in generate_all():
        print("wrote", p)
