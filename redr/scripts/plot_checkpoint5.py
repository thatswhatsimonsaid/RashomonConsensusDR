"""Figures for the Checkpoint 5 report. Reads the saved CSVs only."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "checkpoint5" / "figures"


def _mean_sd(frame: pd.DataFrame, condition: str, rule: str, column: str) -> tuple[float, float]:
    part = frame[(frame["condition"] == condition) & (frame["rule"] == rule)][column]
    return float(part.mean()), float(part.std(ddof=1))


def plot_stretched_metrics(path: Path) -> None:
    detail = pd.read_csv(REPO / "results" / "checkpoint5" / "baselines" / "detail.csv")
    audit = pd.read_csv(REPO / "results" / "checkpoint5" / "baselines" / "metric_audit.csv")
    stretched = detail[detail["condition"] == "stretched"]
    labels = [
        "raw latent",
        "equalized latent",
        "concat 1D Isomap",
        "block-distance CoMDS",
        "averaged-distance MDS",
        "Isomap on full X",
    ]
    keys = [
        ("perfect_balanced", "perfect_isometric"),
        ("equalized_balanced", "equalized_isometric"),
        ("concat_1d_isomap", None),
        ("block_distance_comds", None),
        ("aggregated_distance_mds", None),
        ("isomap_full", None),
    ]
    balanced = []
    isometric = []
    audit_s = audit[audit["condition"] == "stretched"]
    for balanced_key, isometric_key in keys:
        if isometric_key is None:
            part = stretched[stretched["baseline"] == balanced_key]
            balanced.append(part["intrinsic_triplet_balanced"].mean())
            isometric.append(part["intrinsic_triplet_isometric"].mean())
        else:
            balanced.append(audit_s[balanced_key].mean())
            isometric.append(audit_s[isometric_key].mean())
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.barh(y + 0.16, balanced, height=0.3, color="#1f4e79", label="balanced triplet")
    ax.barh(y - 0.16, isometric, height=0.3, color="#c47b2b", label="isometric triplet")
    ax.set_yticks(y, labels)
    ax.set_xlim(0.5, 1.02)
    ax.set_xlabel("Triplet accuracy")
    ax.set_title("Stretched product, four replications")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.55, -0.18), ncol=2)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_discovery(path: Path) -> None:
    detail = pd.read_csv(REPO / "results" / "checkpoint5" / "discovery" / "detail.csv")
    rules = [
        ("all_comds", "all CoMDS"),
        ("union_comds", "union"),
        ("pareto_comds", "Pareto"),
        ("pooled_multiscale_comds", "pooled"),
        ("diversity_first_comds", "diversity first"),
        ("privileged_isomap_visible", "Isomap"),
        ("privileged_concat_1d", "concat 1D"),
    ]
    conditions = ["aligned", "rotated", "rotated_hidden", "stretched_rotated"]
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.6), sharey=True)
    for ax, condition in zip(axes, conditions):
        means = []
        sds = []
        for rule, _label in rules:
            mean, sd = _mean_sd(detail.rename(columns={"rule": "rule"}), condition, rule, "intrinsic_triplet_balanced")
            means.append(mean)
            sds.append(sd)
        y = np.arange(len(rules))
        ax.barh(y, means, xerr=sds, color="#1f4e79", height=0.7, capsize=2)
        ax.set_yticks(y, [label for _rule, label in rules])
        ax.set_xlim(0.45, 1.05)
        ax.set_title(condition.replace("_", " "))
        ax.invert_yaxis()
    axes[0].set_xlabel("")
    fig.suptitle("Balanced triplet, three replications", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_dependence(path: Path) -> None:
    detail = pd.read_csv(REPO / "results" / "checkpoint5" / "alternatives" / "detail.csv")
    rules = [
        ("mi_k3_concat", "MI concat"),
        ("mi_k3_rashomon", "MI Rashomon"),
        ("pearson_k2_concat", "Pearson concat"),
        ("stress_rashomon_pair", "stress pair"),
        ("isomap_visible", "Isomap"),
        ("privileged_concat_1d", "known blocks"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)
    for ax, condition in zip(axes, ("aligned", "rotated")):
        means, sds = [], []
        for rule, _label in rules:
            mean, sd = _mean_sd(detail, condition, rule, "intrinsic_triplet_balanced")
            means.append(mean)
            sds.append(sd)
        y = np.arange(len(rules))
        ax.barh(y, means, xerr=sds, color="#1f4e79", height=0.7, capsize=2)
        ax.set_yticks(y, [label for _rule, label in rules])
        ax.set_xlim(0.45, 1.05)
        ax.set_title(condition)
        ax.invert_yaxis()
    fig.suptitle("Column clustering versus Isomap, three replications", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plot_stretched_metrics(OUT / "stretched_metrics.png")
    plot_discovery(OUT / "discovery_balanced.png")
    plot_dependence(OUT / "dependence_balanced.png")
    print(OUT)


if __name__ == "__main__":
    main()
