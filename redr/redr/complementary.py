"""Complementary candidate families on a product manifold.

Selection uses reference losses and structural losses on the observations.
Latent coordinates are used only when a finished embedding is scored.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from redr.checkpoint2 import _fit_or_load
from redr.controlled import weighted_mean
from redr.diversity import pairwise_neighborhood_overlap
from redr.entropy import entropy_decomposition, knn_adjacency
from redr.intrinsic import intrinsic_recovery
from redr.metrics import _sample_triplets, evaluate_embedding, neighborhood_preservation
from redr.product import generate_product
from redr.rashomon import method_balanced_weights, uniform_weights
from redr.replication import _content_scheme
from redr.selection_rules import diversity_aware, hybrid, method_specific, pooled

import umap


STRUCTURAL_KS = (5, 15, 30)
PRIMARY_EPSILON = 0.05


def dataset_for(condition: dict, replication: int) -> dict:
    return generate_product(
        n_samples=int(condition["n_samples"]),
        aspect=tuple(condition["aspect"]),
        noise=float(condition["noise"]),
        n_nuisance=int(condition["n_nuisance"]),
        hide_second=bool(condition["hide_second"]),
        preserve_scale=bool(condition.get("preserve_scale", False)),
        random_state=int(condition["seed_offset"]) + int(replication),
    )


def fit_candidates(data: dict, triplets: np.ndarray) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Fit joint embeddings and single-axis specialists. Losses stay inside their group."""
    geodesic = _geodesic(data["X"], n_neighbors=15)
    rows = []
    embeddings: dict[str, np.ndarray] = {}
    specs = _specs()
    for spec in specs:
        matrix = _source_matrix(data, spec["source"])
        embedding, reference_loss = _fit_one(matrix, spec)
        embedding = np.asarray(embedding, dtype=np.float64)
        structural = _structural_loss(data["X"], embedding)
        geodesic_loss = _distance_loss(geodesic, embedding)
        latent = intrinsic_recovery(data["intrinsic"], embedding, triplets, k=10)
        latent["u_triplet"] = latent["arc_triplet_accuracy"]
        latent["v_triplet"] = latent["height_triplet_accuracy"]
        ambient = evaluate_embedding(data["X"], embedding, None, k=10, triplets=triplets)
        embedding_id = spec["embedding_id"]
        embeddings[embedding_id] = embedding
        rows.append(
            {
                "embedding_id": embedding_id,
                "algorithm": spec["algorithm"],
                "source": spec["source"],
                "hyper": spec["hyper"],
                "seed": int(spec["seed"]),
                "family": f"{spec['algorithm']}_{spec['source']}",
                "objective_group": f"{spec['algorithm']}|{spec['source']}|{spec['hyper']}",
                "method": f"{spec['algorithm']}_{spec['source']}",
                "reference_loss": float(reference_loss),
                "structural_loss": float(structural),
                "geodesic_loss": float(geodesic_loss),
                **latent,
                **ambient,
            }
        )
        print(
            f"  fit {embedding_id} ref={reference_loss:.3f} struct={structural:.3f} "
            f"balanced={latent['intrinsic_triplet_balanced']:.3f} "
            f"u={latent['u_triplet']:.3f} v={latent['v_triplet']:.3f}",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    return frame, embeddings


def _specs() -> list[dict]:
    specs = [
        {"algorithm": "pca", "source": "joint", "hyper": "2", "seed": 0},
        {"algorithm": "pca", "source": "u", "hyper": "2", "seed": 0},
        {"algorithm": "pca", "source": "v", "hyper": "2", "seed": 0},
    ]
    for source in ("joint", "u", "v"):
        for neighbors in (10, 20):
            specs.append(
                {
                    "algorithm": "isomap",
                    "source": source,
                    "hyper": f"k{neighbors}",
                    "seed": 0,
                    "neighbors": neighbors,
                }
            )
    for source in ("joint", "u", "v"):
        for neighbors in (10, 30):
            for seed in (0, 1):
                specs.append(
                    {
                        "algorithm": "umap",
                        "source": source,
                        "hyper": f"k{neighbors}",
                        "seed": seed,
                        "neighbors": neighbors,
                    }
                )
    for source in ("joint", "u", "v"):
        for perplexity in (15, 40):
            for seed in (0, 1):
                specs.append(
                    {
                        "algorithm": "tsne",
                        "source": source,
                        "hyper": f"p{perplexity}",
                        "seed": seed,
                        "perplexity": perplexity,
                    }
                )
    for spec in specs:
        spec["embedding_id"] = (
            f"{spec['algorithm']}_{spec['source']}_{spec['hyper']}_s{spec['seed']}"
        )
    return specs


def _source_matrix(data: dict, source: str) -> np.ndarray:
    if source == "joint":
        return np.asarray(data["X"], dtype=np.float64)
    block = data["block_u"] if source == "u" else data["block_v"]
    return StandardScaler().fit_transform(np.asarray(block, dtype=np.float64))


def _fit_one(matrix: np.ndarray, spec: dict) -> tuple[np.ndarray, float]:
    algorithm = spec["algorithm"]
    if algorithm == "pca":
        model = PCA(n_components=2, random_state=0)
        embedding = model.fit_transform(matrix)
        return embedding, float(1.0 - model.explained_variance_ratio_.sum())
    if algorithm == "isomap":
        model = Isomap(n_neighbors=int(spec["neighbors"]), n_components=2)
        embedding = model.fit_transform(matrix)
        return embedding, _stress(model.dist_matrix_, embedding)
    if algorithm == "umap":
        model = umap.UMAP(
            n_neighbors=int(spec["neighbors"]),
            n_components=2,
            random_state=int(spec["seed"]),
            metric="euclidean",
            n_jobs=1,
            verbose=False,
            compatibility_layout=True,
        )
        embedding = model.fit_transform(matrix)
        fidelity = neighborhood_preservation(matrix, embedding, k=min(15, matrix.shape[0] - 2))
        return embedding, float(1.0 - fidelity)
    if algorithm == "tsne":
        model = TSNE(
            n_components=2,
            perplexity=int(spec["perplexity"]),
            random_state=int(spec["seed"]),
            init="random",
            learning_rate="auto",
            max_iter=400,
        )
        embedding = model.fit_transform(matrix)
        return embedding, float(model.kl_divergence_)
    raise ValueError(algorithm)


def _geodesic(x: np.ndarray, n_neighbors: int) -> np.ndarray:
    model = Isomap(n_neighbors=n_neighbors, n_components=2)
    model.fit(x)
    return np.asarray(model.dist_matrix_, dtype=float)


def _stress(geodesic: np.ndarray, embedding: np.ndarray) -> float:
    distances = pairwise_distances(np.asarray(embedding, dtype=float))
    iu = np.triu_indices(distances.shape[0], k=1)
    target = np.asarray(geodesic, dtype=float)[iu]
    current = distances[iu]
    scale = float(np.dot(target, target))
    if scale < 1e-12:
        return float("nan")
    return float(np.dot(target - current, target - current) / scale)


def _distance_loss(geodesic: np.ndarray, embedding: np.ndarray) -> float:
    distances = pairwise_distances(np.asarray(embedding, dtype=float))
    iu = np.triu_indices(distances.shape[0], k=1)
    correlation = spearmanr(geodesic[iu], distances[iu]).correlation
    if correlation != correlation:
        return float("nan")
    return float(1.0 - correlation)


def _structural_loss(x: np.ndarray, embedding: np.ndarray) -> float:
    scores = [
        neighborhood_preservation(x, embedding, k=k)
        for k in STRUCTURAL_KS
        if k < x.shape[0] - 1
    ]
    return float(1.0 - np.mean(scores))


def select_rule(
    rule: str,
    frame: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
) -> pd.DataFrame:
    if rule == "all":
        return frame.copy()
    if rule == "specific":
        return method_specific(frame, PRIMARY_EPSILON)
    if rule == "pooled":
        return pooled(frame, PRIMARY_EPSILON, loss_col="structural_loss")
    if rule == "hybrid":
        return hybrid(frame, PRIMARY_EPSILON)
    if rule == "diverse":
        return diversity_aware(
            frame, embeddings, epsilon=PRIMARY_EPSILON, k=10, keep_max=6
        )
    if rule == "joint":
        return frame.loc[frame["source"].eq("joint")].copy()
    if rule == "specialist":
        return frame.loc[frame["source"].isin(["u", "v"])].copy()
    if rule == "best":
        return frame.loc[[frame["structural_loss"].idxmin()]].copy()
    raise ValueError(rule)


def aggregate(
    members: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    kind: str,
    scheme: str,
    out: Path,
    run_name: str,
) -> np.ndarray:
    ids = [str(value) for value in members["embedding_id"]]
    views = [np.asarray(embeddings[key], dtype=np.float64) for key in ids]
    if len(views) == 1 or kind == "procrustes":
        if kind != "procrustes" and len(views) == 1:
            return views[0]
        weights = _weights(members, scheme)
        if len(views) == 1:
            return views[0]
        return weighted_mean(views, weights)
    weights = _weights(members, scheme)
    fitted = _fit_or_load(
        out,
        kind,
        views,
        weights,
        None,
        ids,
        _content_scheme(f"{scheme}|{run_name}", views),
    )
    return np.asarray(fitted["embedding"], dtype=float)


def _weights(members: pd.DataFrame, scheme: str) -> np.ndarray:
    if scheme == "uniform":
        return uniform_weights(members)
    if scheme == "balanced":
        weighted = members.copy()
        weighted["method"] = weighted["family"]
        return method_balanced_weights(weighted)
    raise ValueError(scheme)


def score_embedding(data: dict, embedding: np.ndarray, triplets: np.ndarray) -> dict:
    latent = intrinsic_recovery(data["intrinsic"], embedding, triplets, k=10)
    latent["u_triplet"] = latent["arc_triplet_accuracy"]
    latent["v_triplet"] = latent["height_triplet_accuracy"]
    ambient = evaluate_embedding(data["X"], embedding, None, k=10, triplets=triplets)
    return {**latent, **ambient}


def entropy_diagnostic(
    frame: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    data: dict,
    consensus: np.ndarray,
    k: int = 10,
) -> dict:
    """Ask whether between-method vote entropy tracks latent neighborhood error.

    The entropy graph is the union of candidate neighborhoods. Latent
    coordinates are used only for the error being predicted.
    """
    ids = [str(value) for value in frame["embedding_id"]]
    views = [embeddings[key] for key in ids]
    methods = frame["family"].astype(str).tolist()
    parts = entropy_decomposition(views, methods, k)
    union = np.zeros((data["n"], data["n"]), dtype=float)
    for view in views:
        union = np.maximum(union, knn_adjacency(view, k))
    between = np.asarray(parts["between_information"], dtype=float)
    mass = union.sum(axis=1)
    point_between = (between * union).sum(axis=1) / np.maximum(mass, 1.0)
    latent_miss = _point_miss(data["intrinsic"], consensus, k)
    correlation = spearmanr(point_between, latent_miss).correlation
    return {
        "entropy_error_spearman": float(correlation) if correlation == correlation else float("nan"),
        "mean_between_information": float(parts["mean_between_information"]),
        "mean_within_entropy": float(parts["mean_within_entropy"]),
        "mean_latent_miss": float(np.mean(latent_miss)),
    }


def _point_miss(reference: np.ndarray, embedding: np.ndarray, k: int) -> np.ndarray:
    left = knn_adjacency(reference, k)
    right = knn_adjacency(embedding, k)
    overlap = (left * right).sum(axis=1) / k
    return 1.0 - overlap


def consensus_plan() -> list[tuple[str, str, str]]:
    """(run_name, rule, scheme). Kind is supplied by the caller."""
    rules = [
        ("all", "uniform"),
        ("all", "balanced"),
        ("specific", "uniform"),
        ("pooled", "uniform"),
        ("hybrid", "uniform"),
        ("diverse", "uniform"),
        ("joint", "uniform"),
        ("specialist", "uniform"),
        ("best", "uniform"),
    ]
    return [(f"{rule}_{scheme}", rule, scheme) for rule, scheme in rules]


def write_lock(path: Path) -> None:
    if path.exists():
        return
    payload = {
        "primary_metric": "intrinsic_triplet_balanced",
        "axis_metrics": ["u_triplet", "v_triplet"],
        "structural_loss": "one minus mean neighborhood preservation at k=5,15,30 on the full observed matrix",
        "primary_epsilon": PRIMARY_EPSILON,
        "contrasts": [
            "specialist uniform LoCoMDS versus joint uniform LoCoMDS",
            "pooled uniform LoCoMDS versus specialist uniform LoCoMDS",
            "specialist uniform LoCoMDS versus the single lowest structural loss",
            "method-specific uniform versus all-uniform",
            "the same contrasts for CoMDS and for Procrustes",
            "square_hidden: v_triplet near one half for every rule",
        ],
        "not_used_for_selection": ["intrinsic coordinates", "u", "v", "labels"],
        "tsne_init": "random",
        "umap_compatibility_layout": True,
        "note": "A one-replication smoke used PCA initialization for t-SNE and is archived under complementary_smoke. It is not in the paired table.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def triplets_for(n: int, seed: int) -> np.ndarray:
    return _sample_triplets(n, 3000, np.random.default_rng(seed))


def plot_candidates(frame: pd.DataFrame, path: Path) -> None:
    """Where each embedding sits on the two latent axes. Evaluation only."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    colors = {"joint": "#045a8d", "u": "#e34a33", "v": "#2ca25f"}
    conditions = list(dict.fromkeys(frame["condition"]))
    fig, axes = plt.subplots(1, len(conditions), figsize=(4.0 * len(conditions), 3.8), sharex=True, sharey=True)
    if len(conditions) == 1:
        axes = [axes]
    for axis, condition in zip(axes, conditions):
        block = frame.loc[frame["condition"].eq(condition)]
        for source, color in colors.items():
            part = block.loc[block["source"].eq(source)]
            axis.scatter(
                part["u_triplet"],
                part["v_triplet"],
                s=18,
                alpha=0.75,
                c=color,
                label=source,
                linewidths=0,
            )
        axis.plot([0.5, 1], [0.5, 1], color="#cccccc", linewidth=0.6)
        axis.set_title(condition)
        axis.set_xlim(0.45, 1.02)
        axis.set_ylim(0.45, 1.02)
    axes[0].set_xlabel("u triplet accuracy")
    axes[0].set_ylabel("v triplet accuracy")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_rules(summary: pd.DataFrame, path: Path) -> None:
    """Grouped bars of balanced triplet accuracy by rule, one panel per condition."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    rules = ["specialist", "all", "joint", "pooled", "best", "diverse"]
    kinds = ["procrustes", "comds", "locomds"]
    conditions = list(dict.fromkeys(summary["condition"]))
    fig, axes = plt.subplots(1, len(conditions), figsize=(4.2 * len(conditions), 4.6), sharey=True)
    if len(conditions) == 1:
        axes = [axes]
    colors = {"procrustes": "#a6bddb", "comds": "#2b8cbe", "locomds": "#045a8d"}
    for axis, condition in zip(axes, conditions):
        block = summary.loc[
            summary["condition"].eq(condition) & summary["scheme"].eq("uniform")
        ]
        x = np.arange(len(rules))
        width = 0.25
        for offset, kind in enumerate(kinds):
            means = []
            stds = []
            for rule in rules:
                hit = block.loc[block["rule"].eq(rule) & block["kind"].eq(kind)]
                means.append(float(hit["intrinsic_triplet_balanced"].mean()) if len(hit) else np.nan)
                stds.append(
                    float(hit["intrinsic_triplet_balanced"].std(ddof=1)) if len(hit) > 1 else 0.0
                )
            axis.bar(
                x + (offset - 1) * width,
                means,
                width=width,
                yerr=stds,
                capsize=2,
                color=colors[kind],
                label=kind,
                error_kw={"elinewidth": 0.6},
            )
        axis.set_xticks(x)
        axis.set_xticklabels(rules, rotation=40, ha="right")
        axis.set_title(condition)
        axis.set_ylim(0.0, 1.0)
    axes[0].set_ylabel("balanced intrinsic triplet accuracy")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def geometric_diversity(frame: pd.DataFrame, embeddings: dict[str, np.ndarray], k: int = 10) -> pd.DataFrame:
    """Neighborhood overlap between families, plus axis specialization for evaluation.

    Overlap does not use latent coordinates. The axis gap does, and is only
    a description of what each family recovered.
    """
    ids = [str(value) for value in frame["embedding_id"]]
    overlap = pairwise_neighborhood_overlap([embeddings[key] for key in ids], k=k)
    rows = []
    sources = frame["source"].astype(str).tolist()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            rows.append(
                {
                    "left": ids[i],
                    "right": ids[j],
                    "source_left": sources[i],
                    "source_right": sources[j],
                    "same_source": sources[i] == sources[j],
                    "neighborhood_overlap": float(overlap[i, j]),
                }
            )
    pairs = pd.DataFrame(rows)
    return pairs
