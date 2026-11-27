"""Figures for the cross-objective pilot.

    PYTHONPATH=redr python redr/scripts/plot_cross_objective.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "results" / "checkpoint9" / "cross_objective"


def _read(name: str) -> pd.DataFrame:
    frames = []
    for path in sorted(ROOT.glob(f"*/seed*/{name}")):
        frame = pd.read_csv(path)
        frame["dataset"] = path.parents[1].name
        frame["data_seed"] = path.parent.name
        frames.append(frame)
    if not frames:
        raise SystemExit(f"no {name}")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    figure_dir = ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary = _read("diversity_summary.csv")
    datasets = list(dict.fromkeys(summary["dataset"]))
    relations = ["within_config", "between_config", "between_algorithm"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for ax, column, title in (
        (axes[0], "overlap_k15", "Neighborhood overlap, k=15"),
        (axes[1], "global_disagreement", "Global distance disagreement"),
    ):
        width = 0.25
        for index, relation in enumerate(relations):
            means = []
            for dataset in datasets:
                part = summary.loc[summary["dataset"].eq(dataset) & summary["relation"].eq(relation), column]
                means.append(float(part.mean()) if len(part) else np.nan)
            ax.bar(np.arange(len(datasets)) + (index - 1) * width, means, width=width, label=relation.replace("_", " "))
        ax.set_xticks(np.arange(len(datasets)), datasets, rotation=20, ha="right")
        ax.set_title(title)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_dir / "within_between.png", dpi=140)
    plt.close(fig)

    configs = _read("config_summary.csv")
    if "mean_height" in configs:
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharex=True, sharey=True)
        for ax, dataset in zip(axes, ("s_curve", "swiss")):
            part = configs.loc[configs["dataset"].eq(dataset)]
            means = part.groupby("config_id", sort=False)[["mean_arc", "mean_height"]].mean()
            colors = []
            for name in means.index:
                if name.startswith("umap"):
                    colors.append("C0")
                elif name.startswith("pca"):
                    colors.append("C3")
                elif name.startswith("isomap"):
                    colors.append("C2")
                else:
                    colors.append("0.55")
            ax.scatter(means["mean_arc"], means["mean_height"], c=colors, s=36)
            for name in ("pca", "umap|k=5", "umap|k=15", "umap|k=50", "isomap|k=8"):
                if name in means.index:
                    ax.annotate(name, (means.loc[name, "mean_arc"], means.loc[name, "mean_height"]), fontsize=7)
            ax.axhline(0.5, color="0.8", linewidth=0.8)
            ax.axvline(0.5, color="0.8", linewidth=0.8)
            ax.set_title(dataset)
            ax.set_xlabel("arc triplet")
            ax.set_xlim(0.45, 1.0)
            ax.set_ylim(0.45, 1.0)
        axes[0].set_ylabel("height triplet")
        fig.tight_layout()
        fig.savefig(figure_dir / "axis_recovery.png", dpi=140)
        plt.close(fig)

    # Membership curve from the stored native losses. No extra consensus fits.
    import sys
    sys.path.insert(0, str(REPO / "redr"))
    from redr.cross_objective import apply_membership

    candidates = _read("candidates.csv")
    epsilons = [0.0, 0.05, 0.1, 0.25, 1.0, np.inf]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for dataset, part in candidates.groupby("dataset"):
        fractions = []
        for epsilon in epsilons:
            kept = []
            for _rep, group in part.groupby("data_seed"):
                marked, _inactive, _span = apply_membership(group, epsilon)
                kept.append(float(marked["in_rashomon"].mean()))
            fractions.append(float(np.mean(kept)))
        ax.plot(range(len(epsilons)), fractions, marker="o", label=dataset)
    ax.set_xticks(range(len(epsilons)), ["0", "0.05", "0.1", "0.25", "1", "inf"])
    ax.set_ylim(0.3, 1.02)
    ax.set_xlabel("epsilon")
    ax.set_ylabel("fraction kept")
    ax.legend(fontsize=7)
    ax.set_title("Within-configuration Rashomon membership")
    fig.tight_layout()
    fig.savefig(figure_dir / "membership.png", dpi=140)
    plt.close(fig)

    consensus_paths = list(ROOT.glob("*/seed*/consensus.csv"))
    if consensus_paths:
        consensus = _read("consensus.csv")
        order = [
            "baseline_umap", "baseline_isomap", "baseline_tsne", "baseline_pacmap", "best_trust",
            "conventional_comds", "fixed_objective_comds", "one_seed_per_config",
            "full_comds", "full_method_balanced", "full_config_balanced", "full_quality", "trimmed_trust",
            "hierarchical_comds", "full_locomds",
        ]
        fig, axes = plt.subplots(1, len(datasets), figsize=(3.4 * len(datasets), 4.6), sharey=False)
        if len(datasets) == 1:
            axes = [axes]
        for ax, dataset in zip(axes, datasets):
            part = consensus.loc[consensus["dataset"].eq(dataset)]
            column = "intrinsic_triplet_balanced" if part["intrinsic_triplet_balanced"].notna().any() else "trustworthiness"
            values = []
            for rule in order:
                hit = part.loc[part["rule"].eq(rule), column]
                values.append(float(hit.mean()) if len(hit) else np.nan)
            ax.barh(np.arange(len(order)), values, color="C0")
            ax.set_yticks(np.arange(len(order)), order, fontsize=7)
            ax.set_title(f"{dataset}\n{column.replace('_', ' ')}")
        fig.tight_layout()
        fig.savefig(figure_dir / "consensus_comparison.png", dpi=140)
        plt.close(fig)

    gallery = ROOT / "s_curve" / "seed5100"
    latent_path = gallery / "latent.npz"
    if latent_path.exists():
        latent = np.load(latent_path)["intrinsic"]
        names = [
            "umap__k5_seed0",
            "umap__k15_seed0",
            "umap__k50_seed0",
            "neg_tsne__k15__spectrum0_seed0",
            "neg_tsne__k15__spectrum1_seed0",
            "isomap__k15_seed0",
            "pca_seed0",
            "tsne__perplexity30_seed0",
        ]
        fig, axes = plt.subplots(2, len(names), figsize=(2.1 * len(names), 4.4))
        for column, name in enumerate(names):
            path = gallery / "embeddings" / f"{name}.npy"
            if not path.exists():
                continue
            embedding = np.load(path)
            axes[0, column].scatter(embedding[:, 0], embedding[:, 1], c=latent[:, 0], s=4, cmap="viridis")
            axes[1, column].scatter(embedding[:, 0], embedding[:, 1], c=latent[:, 1], s=4, cmap="coolwarm")
            axes[0, column].set_title(name.replace("__", "\n").replace("_seed0", ""), fontsize=7)
            axes[0, column].set_xticks([])
            axes[0, column].set_yticks([])
            axes[1, column].set_xticks([])
            axes[1, column].set_yticks([])
        axes[0, 0].set_ylabel("colored by arc")
        axes[1, 0].set_ylabel("colored by height")
        fig.tight_layout()
        fig.savefig(figure_dir / "s_curve_gallery.png", dpi=140)
        plt.close(fig)
    print(f"wrote {figure_dir}")


if __name__ == "__main__":
    main()
