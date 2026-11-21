"""CoMDS and LoCoMDS on the controlled complementary ensembles.

The observable score is a simulated quality signal. When it is built from
the latent error, the run is a mechanism study and is labeled with its
alignment. Latent coordinates are not an input to the embedding generators
beyond the perturbation model itself. They are not read by the selection
rule except through that pre-specified score.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from redr.complementary import aggregate
from redr.controlled import (
    _unit,
    observable_scores,
    perturb,
    sample_latent,
)
from redr.intrinsic import balance_axes, intrinsic_recovery
from redr.metrics import neighborhood_preservation
from redr.metrics import _sample_triplets
from redr.plotting import display_align
from redr.selection_rules import method_specific, pooled


def local_tear(latent: np.ndarray, method: int, rng: np.random.Generator, scale: float) -> np.ndarray:
    """One method damages a minority of points. Another squashes a whole axis."""
    drawn = latent + rng.normal(0.0, 0.2 * scale, size=latent.shape)
    if method == 0:
        mask = rng.random(latent.shape[0]) < 0.25
        drawn[mask] = drawn[mask] + rng.normal(0.0, 2.5 * scale, size=(int(mask.sum()), latent.shape[1]))
    elif method == 1:
        drawn[:, 1] = 0.15 * drawn[:, 1]
    return drawn


def scenarios() -> list[dict]:
    """Small pre-specified grid. Not expanded after seeing consensus scores."""
    base = {
        "n": 80,
        "n_triplets": 1500,
        "n_methods": 3,
        "per_method": 6,
        "scale": 0.18,
        "n_replications": 4,
    }
    specs = [
        ("comp_square_noisy", "square", "complementary", "noisy", 0.0),
        ("comp_square_none", "square", "complementary", "none", 0.0),
        ("comp_square_misleading", "square", "complementary", "misleading", 0.0),
        ("comp_square_junk", "square", "complementary", "noisy", 0.4),
        ("majority_square_noisy", "square", "majority_bias", "noisy", 0.0),
        ("majority_square_none", "square", "majority_bias", "none", 0.0),
        ("shared_square_noisy", "square", "shared_noise", "noisy", 0.0),
        ("comp_long_noisy", "long", "complementary", "noisy", 0.0),
        ("local_square_noisy", "square", "local_tear", "noisy", 0.0),
    ]
    return [{**base, "name": name, "geometry": geometry, "structure": structure, "alignment": alignment, "contamination": contamination} for name, geometry, structure, alignment, contamination in specs]


def _draw(latent, method, structure, rng, scale):
    if structure == "local_tear":
        return local_tear(latent, method, rng, scale)
    return perturb(latent, method, structure, rng, scale)


def build_pool(scenario: dict, replication: int, seed0: int):
    rng = np.random.default_rng(seed0 + replication)
    latent = sample_latent(int(scenario["n"]), scenario["geometry"], rng)
    triplets = _sample_triplets(
        latent.shape[0],
        int(scenario["n_triplets"]),
        np.random.default_rng(seed0 + 50_000 + replication),
    )
    balanced = balance_axes(latent)
    records = []
    embeddings = {}
    for method in range(int(scenario["n_methods"])):
        for draw in range(int(scenario["per_method"])):
            embedding = _draw(latent, method, scenario["structure"], rng, float(scenario["scale"]))
            if rng.random() < float(scenario["contamination"]):
                embedding = rng.normal(size=latent.shape)
            embedding_id = f"m{method}_d{draw}"
            global_error = float(
                np.sqrt(np.mean((display_align(embedding, balanced) - _unit(balanced)) ** 2))
            )
            local_error = float(1.0 - neighborhood_preservation(balanced, embedding, k=8))
            records.append(
                {
                    "embedding_id": embedding_id,
                    "method": f"m{method}",
                    "objective_group": f"m{method}",
                    "family": f"m{method}",
                    "true_global_error": global_error,
                    "true_local_error": local_error,
                }
            )
            embeddings[embedding_id] = np.asarray(embedding, dtype=np.float64)
    frame = pd.DataFrame(records)
    # The simulated score tracks global error. Local error is evaluation only.
    frame["reference_loss"] = observable_scores(
        frame["true_global_error"].to_numpy(), scenario["alignment"], rng
    )
    frame["structural_loss"] = frame["reference_loss"]
    return latent, triplets, frame, embeddings


def select_controlled(rule: str, frame: pd.DataFrame) -> pd.DataFrame:
    if rule == "all":
        return frame.copy()
    if rule == "specific":
        return method_specific(frame, 0.05)
    if rule == "pooled":
        return pooled(frame, 0.05, loss_col="reference_loss")
    if rule == "best":
        return frame.loc[[frame["reference_loss"].idxmin()]].copy()
    raise ValueError(rule)


def run_scenario(scenario: dict, out: Path, seed0: int, kinds: list[str]) -> pd.DataFrame:
    rows = []
    for replication in range(int(scenario["n_replications"])):
        latent, triplets, frame, embeddings = build_pool(scenario, replication, seed0)
        spearman_global = _spearman(frame["reference_loss"], frame["true_global_error"])
        spearman_local = _spearman(frame["reference_loss"], frame["true_local_error"])
        for rule in ("all", "specific", "pooled", "best"):
            members = select_controlled(rule, frame)
            for kind in kinds:
                if rule == "best" and kind != "procrustes":
                    continue
                row = {
                    "scenario": scenario["name"],
                    "geometry": scenario["geometry"],
                    "structure": scenario["structure"],
                    "alignment": scenario["alignment"],
                    "contamination": float(scenario["contamination"]),
                    "replication": replication,
                    "rule": rule,
                    "kind": kind,
                    "n_members": int(len(members)),
                    "score_global_spearman": spearman_global,
                    "score_local_spearman": spearman_local,
                    "minority_kept": int(members["method"].eq("m2").sum()) if len(members) else 0,
                    "methods_kept": ",".join(sorted(members["method"].unique())) if len(members) else "",
                }
                try:
                    embedding = aggregate(
                        members,
                        embeddings,
                        kind,
                        "uniform",
                        out / scenario["name"] / f"rep{replication:02d}",
                        f"{rule}_{kind}",
                    )
                    scores = intrinsic_recovery(latent, embedding, triplets, k=8)
                    scores["u_triplet"] = scores["arc_triplet_accuracy"]
                    scores["v_triplet"] = scores["height_triplet_accuracy"]
                    row.update(scores)
                    row["error"] = ""
                except Exception as exc:  # noqa: BLE001
                    row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
                print(
                    f"{scenario['name']} r{replication} {rule} {kind} "
                    f"n={row['n_members']} balanced={_fmt(row, 'intrinsic_triplet_balanced')} "
                    f"u={_fmt(row, 'u_triplet')} v={_fmt(row, 'v_triplet')} err={row['error'][:80]}",
                    flush=True,
                )
    return pd.DataFrame(rows)


def plot_mechanisms(detail: pd.DataFrame, path: Path) -> None:
    """Compare the full pool, a pooled quality filter, and the best single view."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    order = [
        "comp_square_noisy",
        "comp_square_none",
        "comp_square_misleading",
        "comp_square_junk",
        "majority_square_noisy",
        "majority_square_none",
        "shared_square_noisy",
        "local_square_noisy",
        "comp_long_noisy",
    ]
    specs = [
        ("all", "procrustes", "all, Procrustes", "#a6bddb"),
        ("all", "locomds", "all, LoCoMDS", "#045a8d"),
        ("pooled", "locomds", "pooled score, LoCoMDS", "#e34a33"),
        ("best", "procrustes", "best single score", "#fec44f"),
    ]
    fig, axis = plt.subplots(figsize=(11.2, 5.2))
    x = np.arange(len(order))
    width = 0.18
    for offset, (rule, kind, label, color) in enumerate(specs):
        means = []
        stds = []
        for name in order:
            hit = detail.loc[
                detail["scenario"].eq(name) & detail["rule"].eq(rule) & detail["kind"].eq(kind),
                "intrinsic_triplet_balanced",
            ]
            means.append(float(hit.mean()) if len(hit) else np.nan)
            stds.append(float(hit.std(ddof=1)) if len(hit) > 1 else 0.0)
        axis.bar(
            x + (offset - 1.5) * width,
            means,
            width=width,
            yerr=stds,
            capsize=2,
            color=color,
            label=label,
            error_kw={"elinewidth": 0.6},
        )
    axis.set_xticks(x)
    axis.set_xticklabels(order, rotation=40, ha="right", fontsize=8)
    axis.set_ylabel("balanced intrinsic triplet accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _fmt(row: dict, key: str) -> str:
    value = row.get(key, float("nan"))
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "nan"


def _spearman(left, right) -> float:
    from scipy.stats import spearmanr

    value = spearmanr(left, right).correlation
    if value != value:
        return float("nan")
    return float(value)
