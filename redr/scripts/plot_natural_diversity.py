"""Figures for the natural-diversity pilot.

    PYTHONPATH=redr python redr/scripts/plot_natural_diversity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "results" / "checkpoint8" / "natural_diversity"


def _candidates() -> pd.DataFrame:
    frames = []
    for path in sorted(ROOT.glob("*/candidates.csv")):
        frame = pd.read_csv(path)
        frame["dataset"] = path.parent.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    frame = _candidates()
    figure_dir = ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    primary = frame.loc[frame["primary"].astype(bool)]
    datasets = list(dict.fromkeys(primary["dataset"]))
    methods = list(dict.fromkeys(primary["method"]))
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.6), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        part = primary.loc[primary["dataset"].eq(dataset)]
        positions = np.arange(len(methods))
        for offset, (column, label) in enumerate((("trustworthiness", "trustworthiness"),)):
            means = [float(part.loc[part["method"].eq(method), column].mean()) for method in methods]
            ax.bar(positions, means, width=0.7, color="C0")
        ax.set_xticks(positions, methods, rotation=30, ha="right")
        ax.set_title(dataset)
        ax.set_ylim(0.5, 1.0)
    axes[0].set_ylabel("mean trustworthiness")
    fig.suptitle("Primary pool, k=15, lambda=0, three seeds")
    fig.tight_layout()
    fig.savefig(figure_dir / "trust_by_method.png", dpi=140)
    plt.close(fig)

    means = (
        frame.groupby(["dataset", "method", "k", "lambda"], sort=False)["trustworthiness"]
        .agg(["mean", "min", "max", "count"])
        .reset_index()
    )
    fig, axes = plt.subplots(len(datasets), 1, figsize=(8.5, 2.6 * len(datasets)), sharex=False)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        part = means.loc[means["dataset"].eq(dataset)].reset_index(drop=True)
        labels = [f"{row.method}\nk{int(row.k)} l{row['lambda']:g}" for _, row in part.iterrows()]
        ax.plot(range(len(part)), part["mean"], marker="o", color="C0")
        ax.fill_between(range(len(part)), part["min"], part["max"], color="C0", alpha=0.2)
        ax.set_xticks(range(len(part)), labels, fontsize=7)
        ax.set_ylabel("trustworthiness")
        ax.set_title(dataset)
    fig.suptitle("Seed range of trustworthiness. Shaded band is min to max.")
    fig.tight_layout()
    fig.savefig(figure_dir / "trust_ranges.png", dpi=140)
    plt.close(fig)

    consensus_paths = sorted(ROOT.glob("*/consensus.csv"))
    if consensus_paths:
        consensus = pd.concat([pd.read_csv(path).assign(dataset=path.parent.name) for path in consensus_paths], ignore_index=True)
        order = ["rashomon_0", "rashomon_0.05", "rashomon_0.25", "rashomon_1", "rashomon_inf"]
        fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.6), sharey=True)
        if len(datasets) == 1:
            axes = [axes]
        for ax, dataset in zip(axes, datasets):
            part = consensus.loc[consensus["dataset"].eq(dataset)]
            values = [float(part.loc[part["rule"].eq(rule), "trustworthiness"].iloc[0]) if (part["rule"] == rule).any() else np.nan for rule in order]
            ax.plot(range(len(order)), values, marker="o", color="C3", label="within-method band")
            for name, color in (("unrestricted", "C0"), ("full_gibbs", "C2"), ("trimmed", "C1"), ("best_trust", "C4")):
                hit = part.loc[part["rule"].eq(name), "trustworthiness"]
                if len(hit):
                    ax.axhline(float(hit.iloc[0]), color=color, linestyle="--", linewidth=1, label=name)
            ax.set_xticks(range(len(order)), ["0", "0.05", "0.25", "1", "inf"])
            ax.set_title(dataset)
            ax.set_xlabel("epsilon")
        axes[0].set_ylabel("consensus trustworthiness")
        axes[-1].legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(figure_dir / "epsilon_trust.png", dpi=140)
        plt.close(fig)
    print(f"wrote {figure_dir}")


if __name__ == "__main__":
    main()
