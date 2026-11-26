"""Summarize the quality × diversity pilot and write the figures.

    PYTHONPATH=redr python redr/scripts/plot_quality_diversity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "redr"))

ROOT = REPO / "results" / "checkpoint7" / "quality_diversity"


def _rule(frame: pd.DataFrame, rule: str, epsilon: float | None = None) -> pd.DataFrame:
    chosen = frame.loc[frame["rule"].eq(rule)].copy()
    if epsilon is not None:
        if np.isinf(epsilon):
            chosen = chosen.loc[np.isinf(chosen["epsilon"])]
        else:
            chosen = chosen.loc[np.isclose(chosen["epsilon"], epsilon, equal_nan=False)]
    return chosen


def _one(frame: pd.DataFrame, rule: str, epsilon: float | None, replication: int) -> float:
    chosen = _rule(frame, rule, epsilon)
    chosen = chosen.loc[chosen["replication"].eq(replication)]
    if chosen.empty or not np.isfinite(chosen["intrinsic_triplet_balanced"]).any():
        return float("nan")
    return float(chosen["intrinsic_triplet_balanced"].iloc[0])


def paired(detail: pd.DataFrame) -> pd.DataFrame:
    """Within-replication differences. Size-matched is the mean of its three draws."""
    keys = ["quality", "diversity", "score_kind"]
    rows = []
    for key, frame in detail.groupby(keys, sort=False):
        reps = sorted(frame["replication"].unique())
        for replication in reps:
            base = {
                "quality": key[0],
                "diversity": key[1],
                "score_kind": key[2],
                "replication": int(replication),
            }
            values = {
                "rashomon": _one(frame, "rashomon_uniform", 0.05, replication),
                "unrestricted": _one(frame, "unrestricted", np.inf, replication),
                "trimmed": _one(frame, "trimmed", None, replication),
                "best": _one(frame, "best", None, replication),
                "gibbs": _one(frame, "full_gibbs", None, replication),
                "balanced_full": _one(frame, "full_balanced", None, replication),
                "procrustes": _one(frame, "unrestricted_procrustes", None, replication),
                "rashomon_procrustes": _one(frame, "rashomon_procrustes", 0.05, replication),
                "rashomon_gibbs": _one(frame, "rashomon_gibbs", 0.05, replication),
                "rashomon_balanced": _one(frame, "rashomon_balanced", 0.05, replication),
                "oracle": _one(frame, "oracle_rashomon", 0.05, replication),
                "locomds": _one(frame, "unrestricted_locomds", None, replication),
                "rashomon_locomds": _one(frame, "rashomon_locomds", 0.05, replication),
            }
            matched = frame.loc[frame["rule"].str.startswith("size_matched") & frame["replication"].eq(replication)]
            values["size_matched"] = float(matched["intrinsic_triplet_balanced"].mean()) if len(matched) else float("nan")
            row = {**base, **values}
            for name in ("unrestricted", "trimmed", "best", "gibbs", "size_matched", "procrustes", "rashomon_gibbs"):
                row[f"rashomon_minus_{name}"] = values["rashomon"] - values[name]
            row["rashomon_locomds_minus_locomds"] = values["rashomon_locomds"] - values["locomds"]
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(paired_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["quality", "diversity", "score_kind"]
    contrasts = [
        "rashomon",
        "unrestricted",
        "trimmed",
        "best",
        "gibbs",
        "size_matched",
        "procrustes",
        "rashomon_gibbs",
        "rashomon_balanced",
        "oracle",
        "locomds",
        "rashomon_locomds",
        "rashomon_minus_unrestricted",
        "rashomon_minus_trimmed",
        "rashomon_minus_best",
        "rashomon_minus_gibbs",
        "rashomon_minus_size_matched",
        "rashomon_minus_procrustes",
        "rashomon_locomds_minus_locomds",
    ]
    for key, frame in paired_frame.groupby(group_cols, sort=False):
        row = {"quality": key[0], "diversity": key[1], "score_kind": key[2], "n": int(len(frame))}
        for name in contrasts:
            values = frame[name].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[f"{name}_mean"] = float(np.mean(finite)) if len(finite) else float("nan")
            row[f"{name}_sd"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")
            if name.startswith("rashomon_minus") or name.endswith("_minus_locomds"):
                row[f"{name}_n_pos"] = int(np.sum(finite > 0)) if len(finite) else 0
        rows.append(row)
    return pd.DataFrame(rows)


def _plots(detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    informative = detail.loc[detail["score_kind"].eq("informative") & detail["rule"].eq("rashomon_uniform")]
    qualities = ["high", "mostly_high", "mixed", "severe", "shared"]
    diversities = ["identical", "moderate", "complementary", "harmful"]
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.2), sharey=True)
    for ax, diversity in zip(axes, diversities):
        for quality in qualities:
            block = informative.loc[informative["diversity"].eq(diversity) & informative["quality"].eq(quality)]
            if block.empty:
                continue
            grouped = block.groupby("epsilon", sort=True)["intrinsic_triplet_balanced"]
            means = grouped.mean()
            xs = [1e-3 if np.isinf(value) else max(value, 1e-3) for value in means.index]
            # Plot epsilon on a categorical axis so zero and infinity stay visible.
            order = [0.0, 0.05, 0.25, 1.0, np.inf]
            ys = []
            for eps in order:
                match = block.loc[np.isinf(block["epsilon"]) if np.isinf(eps) else np.isclose(block["epsilon"], eps)]
                ys.append(float(match["intrinsic_triplet_balanced"].mean()) if len(match) else np.nan)
            ax.plot([0, 1, 2, 3, 4], ys, marker="o", label=quality)
        ax.set_xticks([0, 1, 2, 3, 4], ["0", "0.05", "0.25", "1", "inf"])
        ax.set_title(diversity)
        ax.set_xlabel("epsilon")
    axes[0].set_ylabel("balanced triplet")
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Rashomon CoMDS across epsilon, informative score")
    fig.tight_layout()
    fig.savefig(figure_dir / "epsilon_curves.png", dpi=140)
    plt.close(fig)

    primary = summary.loc[summary["score_kind"].eq("informative")]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
    contrasts = [
        ("rashomon_minus_unrestricted_mean", "minus unrestricted"),
        ("rashomon_minus_gibbs_mean", "minus full-pool Gibbs"),
        ("rashomon_minus_size_matched_mean", "minus size-matched"),
    ]
    for ax, (column, title) in zip(axes, contrasts):
        grid = np.full((len(qualities), len(diversities)), np.nan)
        for i, quality in enumerate(qualities):
            for j, diversity in enumerate(diversities):
                hit = primary.loc[primary["quality"].eq(quality) & primary["diversity"].eq(diversity)]
                if len(hit):
                    grid[i, j] = float(hit.iloc[0][column])
        image = ax.imshow(grid, cmap="coolwarm", vmin=-0.42, vmax=0.16)
        ax.set_xticks(range(len(diversities)), diversities, rotation=30, ha="right")
        ax.set_yticks(range(len(qualities)), qualities, fontsize=8)
        ax.set_title(title, fontsize=9)
        for i in range(len(qualities)):
            for j in range(len(diversities)):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:+.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=axes, fraction=0.03)
    fig.suptitle("Rashomon epsilon 0.05 minus the named baseline")
    fig.savefig(figure_dir / "quality_diversity_deltas.png", dpi=140)
    plt.close(fig)


def _gallery() -> None:
    """One replication of the harmful high-quality cell. Latent color is the first axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from redr.quality_diversity import build_pool, load_config

    config = load_config()
    pool = build_pool(config, {"quality": "high", "diversity": "harmful", "score_kind": "informative"}, 0)
    fit = ROOT / "fits" / "high__harmful__informative__rep0"
    panels = [
        ("faithful m0", pool["embeddings"]["m0_d1"]),
        ("mis-paired m2", pool["embeddings"]["m2_d1"]),
        ("unrestricted", pd.read_csv(fit / "unrestricted" / "epsinf" / "gspace.csv", header=None).to_numpy()),
        ("Rashomon 0.05", pd.read_csv(fit / "rashomon_uniform" / "eps0.05" / "gspace.csv", header=None).to_numpy()),
        ("full Gibbs", pd.read_csv(fit / "full_gibbs" / "epsinf" / "gspace.csv", header=None).to_numpy()),
        ("trimmed", pd.read_csv(fit / "trimmed" / "epsnan" / "gspace.csv", header=None).to_numpy()),
    ]
    color = pool["latent"][:, 0]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.2 * len(panels), 2.5))
    for ax, (title, embedding) in zip(axes, panels):
        ax.scatter(embedding[:, 0], embedding[:, 1], c=color, s=8, cmap="viridis", linewidths=0)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("High quality, harmful method, informative score, replication 0")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "harmful_high_gallery.png", dpi=140)
    plt.close(fig)


def main() -> None:
    detail_path = ROOT / "detail.csv"
    if not detail_path.exists():
        parts = sorted((ROOT / "by_rep").glob("*.csv"))
        detail = pd.concat([pd.read_csv(path) for path in parts], ignore_index=True)
        detail.to_csv(detail_path, index=False)
    else:
        detail = pd.read_csv(detail_path)
    pairs = paired(detail)
    pairs.to_csv(ROOT / "paired.csv", index=False)
    summary = summarize(pairs)
    summary.to_csv(ROOT / "summary.csv", index=False)
    _plots(detail, summary)
    _gallery()
    confirm = REPO / "results" / "checkpoint7" / "confirm" / "detail.csv"
    if confirm.exists():
        confirm_pairs = paired(pd.read_csv(confirm))
        confirm_pairs.to_csv(confirm.parent / "paired.csv", index=False)
        summarize(confirm_pairs).to_csv(confirm.parent / "summary.csv", index=False)
    focus = summary.loc[summary["score_kind"].eq("informative"), [
        "quality",
        "diversity",
        "rashomon_mean",
        "unrestricted_mean",
        "gibbs_mean",
        "trimmed_mean",
        "best_mean",
        "size_matched_mean",
        "rashomon_minus_unrestricted_mean",
        "rashomon_minus_gibbs_mean",
        "rashomon_minus_trimmed_mean",
        "rashomon_minus_best_mean",
        "rashomon_minus_size_matched_mean",
        "rashomon_minus_unrestricted_n_pos",
        "rashomon_minus_gibbs_n_pos",
    ]]
    print(focus.to_string(index=False))


if __name__ == "__main__":
    main()
