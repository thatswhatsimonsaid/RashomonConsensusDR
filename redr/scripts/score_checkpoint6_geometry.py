"""Extra geometry scores for the Checkpoint 6 embeddings.

Reads the saved classical and parametric embeddings. Does not retrain.
The end-gap and coordinate triplets use the Swiss-roll parameter only as
an evaluation target. Selection already happened.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "redr"))

from redr.diversity import pairwise_neighborhood_overlap  # noqa: E402
from redr.intrinsic import intrinsic_recovery  # noqa: E402
from redr.complementary import triplets_for  # noqa: E402
from redr.paper_views import (  # noqa: E402
    induced_graph_weights,
    mixture_of_gaussians,
    soft_jaccard_distance,
    swiss_roll,
)
from redr.train import high_d_neighbors  # noqa: E402


ROOT = REPO / "results" / "checkpoint6"


def _load_gspace(path: Path) -> np.ndarray:
    return pd.read_csv(path, header=None).to_numpy(dtype=float)


def _end_gap(embedding: np.ndarray, parameter: np.ndarray, tail: int = 30) -> float:
    order = np.argsort(parameter)
    low = embedding[order[:tail]]
    high = embedding[order[-tail:]]
    gaps = np.linalg.norm(low[:, None, :] - high[None, :, :], axis=2)
    neighbors = NearestNeighbors(n_neighbors=11).fit(embedding)
    distances, _indices = neighbors.kneighbors(embedding)
    scale = float(np.median(distances[:, 1:]))
    return float(np.mean(gaps) / max(scale, 1e-12))


def _within_between(embedding: np.ndarray, labels: np.ndarray) -> float:
    within = []
    centroids = []
    for label in np.unique(labels):
        block = embedding[labels == label]
        centroids.append(block.mean(axis=0))
        if len(block) > 1:
            gaps = np.linalg.norm(block[:, None, :] - block[None, :, :], axis=2)
            within.append(float(gaps[np.triu_indices(len(block), k=1)].mean()))
    centers = np.vstack(centroids)
    between = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    between_mean = float(between[np.triu_indices(len(centers), k=1)].mean())
    return float(np.mean(within) / max(between_mean, 1e-12))


def _swiss_rows() -> list[dict]:
    data = swiss_roll(400, random_state=331)
    triplets = triplets_for(data["n"], 16000)
    archive = np.load(ROOT / "swiss" / "classical" / "views.npz", allow_pickle=True)
    ids = list(archive["ids"])
    sources = {name: archive["embeddings"][ids.index(name)] for name in ids}
    sources["original_comds"] = _load_gspace(ROOT / "swiss" / "classical" / "comds" / "gspace.csv")
    sources["locomds_tau0.1_p0.5"] = _load_gspace(ROOT / "swiss" / "classical" / "locomds_tau0.1_p0.5" / "gspace.csv")
    sources["locomds_tau0.1_p0.3"] = _load_gspace(ROOT / "swiss" / "classical" / "locomds_tau0.1_p0.3" / "gspace.csv")
    catalog = pd.read_csv(ROOT / "swiss" / "parametric" / "candidates.csv")
    saved = np.load(ROOT / "swiss" / "parametric" / "embeddings.npz", allow_pickle=True)
    by_id = {str(name): saved["embeddings"][index] for index, name in enumerate(saved["ids"])}
    for method, group in catalog.groupby("method"):
        best = group.sort_values("native_loss").iloc[0]
        sources[f"best_{method}"] = by_id[str(best["embedding_id"])]
    for rule in ("unrestricted_comds", "unrestricted_locomds", "rashomon_uniform_comds", "graph_rashomon_comds", "quality_weighted_comds"):
        sources[rule] = _load_gspace(ROOT / "swiss" / "parametric" / rule / "gspace.csv")
    sources["common_knowledge_umap"] = np.load(ROOT / "swiss" / "parametric" / "common_knowledge.npy")
    rows = []
    for name, embedding in sources.items():
        recovery = intrinsic_recovery(data["intrinsic"], embedding, triplets, k=10)
        rows.append(
            {
                "dataset": "swiss",
                "rule": name,
                "intrinsic_triplet_balanced": recovery["intrinsic_triplet_balanced"],
                "intrinsic_triplet_isometric": recovery["intrinsic_triplet_isometric"],
                "arc_triplet_accuracy": recovery["arc_triplet_accuracy"],
                "height_triplet_accuracy": recovery["height_triplet_accuracy"],
                "end_gap": _end_gap(embedding, data["t"]),
            }
        )
    return rows


def _gaussian_rows() -> list[dict]:
    data = mixture_of_gaussians(331)
    archive = np.load(ROOT / "gaussian" / "classical" / "views.npz", allow_pickle=True)
    ids = list(archive["ids"])
    wanted = ("pca", "isomap", "tsne_perplexity_30", "umap_neighbors_30", "kpca_sigma_0.01", "laplacian")
    sources = {name: archive["embeddings"][ids.index(name)] for name in wanted}
    # Meta-Spec is not stored in the npz. Recompute from the saved views.
    from redr.paper_views import meta_spec_kpca

    sources["metaspec_kpca"] = meta_spec_kpca([archive["embeddings"][i] for i in range(len(ids))])
    sources["original_comds"] = _load_gspace(ROOT / "gaussian" / "classical" / "comds" / "gspace.csv")
    sources["locomds_tau0.1_p0.5"] = _load_gspace(ROOT / "gaussian" / "classical" / "locomds_tau0.1_p0.5" / "gspace.csv")
    rows = []
    for name, embedding in sources.items():
        rows.append(
            {
                "dataset": "gaussian",
                "rule": name,
                "within_between": _within_between(embedding, data["labels"]),
            }
        )
    return rows


def _definition_32() -> pd.DataFrame:
    rows = []
    for name, matrix in (
        ("gaussian", mixture_of_gaussians(331)["X"]),
        ("swiss", swiss_roll(400, random_state=331)["X"]),
    ):
        catalog = pd.read_csv(ROOT / name / "parametric" / "candidates.csv")
        saved = np.load(ROOT / name / "parametric" / "embeddings.npz", allow_pickle=True)
        by_id = {str(key): saved["embeddings"][index] for index, key in enumerate(saved["ids"])}
        neighbors = high_d_neighbors(matrix.astype(np.float32), 15)
        for _method, group in catalog.groupby("method", sort=False):
            ids = [str(value) for value in group["embedding_id"]]
            weights = [induced_graph_weights(by_id[key], neighbors, gamma=1.0) for key in ids]
            distance = np.zeros((len(ids), len(ids)))
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    value = soft_jaccard_distance(weights[i], weights[j])
                    distance[i, j] = distance[j, i] = value
            mean_distance = distance.sum(axis=1) / max(len(ids) - 1, 1)
            threshold = float(np.median(mean_distance))
            views = [by_id[key] for key in ids]
            overlap = pairwise_neighborhood_overlap(views, k=15)
            mean_overlap = (overlap.sum(axis=1) - 1.0) / (len(ids) - 1)
            center = int(np.argmax(mean_overlap))
            agreement = overlap[center]
            overlap_threshold = float(np.median(agreement))
            for index, embedding_id in enumerate(ids):
                rows.append(
                    {
                        "dataset": name,
                        "embedding_id": embedding_id,
                        "method": group["method"].iloc[0],
                        "mean_soft_jaccard": float(mean_distance[index]),
                        "in_definition_32_median": bool(mean_distance[index] <= threshold + 1e-12),
                        "in_overlap_median": bool(agreement[index] >= overlap_threshold - 1e-12),
                    }
                )
    return pd.DataFrame(rows)


def _plot(geometry: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    swiss = swiss_roll(400, random_state=331)
    panels = [
        ("PCA", np.load(ROOT / "swiss" / "classical" / "views.npz", allow_pickle=True)),
    ]
    archive = panels[0][1]
    ids = list(archive["ids"])
    named = [
        ("PCA", archive["embeddings"][ids.index("pca")]),
        ("Isomap", archive["embeddings"][ids.index("isomap")]),
        ("UMAP 30", archive["embeddings"][ids.index("umap_neighbors_30")]),
        ("Meta-Spec", None),
        ("CoMDS", _load_gspace(ROOT / "swiss" / "classical" / "comds" / "gspace.csv")),
        ("LoCoMDS", _load_gspace(ROOT / "swiss" / "classical" / "locomds_tau0.1_p0.5" / "gspace.csv")),
        ("Param. CoMDS", _load_gspace(ROOT / "swiss" / "parametric" / "unrestricted_comds" / "gspace.csv")),
        ("Loss Rashomon", _load_gspace(ROOT / "swiss" / "parametric" / "rashomon_uniform_comds" / "gspace.csv")),
        ("Common knowledge", np.load(ROOT / "swiss" / "parametric" / "common_knowledge.npy")),
    ]
    from redr.paper_views import meta_spec_kpca

    named[3] = ("Meta-Spec", meta_spec_kpca([archive["embeddings"][i] for i in range(len(ids))]))
    fig, axes = plt.subplots(1, len(named), figsize=(2.15 * len(named), 2.5))
    for ax, (title, embedding) in zip(axes, named):
        ax.scatter(embedding[:, 0], embedding[:, 1], c=swiss["t"], s=3, cmap="viridis", linewidths=0)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Swiss roll, parameter t")
    fig.tight_layout()
    fig.savefig(figure_dir / "swiss_comparison.png", dpi=140)
    plt.close(fig)

    gaussian = mixture_of_gaussians(331)
    garchive = np.load(ROOT / "gaussian" / "classical" / "views.npz", allow_pickle=True)
    gids = list(garchive["ids"])
    gnamed = [
        ("PCA", garchive["embeddings"][gids.index("pca")]),
        ("t-SNE 30", garchive["embeddings"][gids.index("tsne_perplexity_30")]),
        ("UMAP 30", garchive["embeddings"][gids.index("umap_neighbors_30")]),
        ("Meta-Spec", meta_spec_kpca([garchive["embeddings"][i] for i in range(len(gids))])),
        ("CoMDS", _load_gspace(ROOT / "gaussian" / "classical" / "comds" / "gspace.csv")),
        ("LoCoMDS", _load_gspace(ROOT / "gaussian" / "classical" / "locomds_tau0.1_p0.5" / "gspace.csv")),
    ]
    fig, axes = plt.subplots(1, len(gnamed), figsize=(2.15 * len(gnamed), 2.5))
    for ax, (title, embedding) in zip(axes, gnamed):
        ax.scatter(embedding[:, 0], embedding[:, 1], c=gaussian["labels"], s=4, cmap="viridis", linewidths=0)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Gaussian mixture")
    fig.tight_layout()
    fig.savefig(figure_dir / "gaussian_comparison.png", dpi=140)
    plt.close(fig)
    del geometry


def main() -> None:
    geometry = pd.DataFrame(_swiss_rows() + _gaussian_rows())
    geometry.to_csv(ROOT / "geometry.csv", index=False)
    definition = _definition_32()
    definition.to_csv(ROOT / "definition32.csv", index=False)
    _plot(geometry)
    print(geometry.to_string(index=False))
    both = definition.groupby("dataset").apply(
        lambda frame: pd.Series(
            {
                "n": len(frame),
                "definition32": int(frame["in_definition_32_median"].sum()),
                "overlap": int(frame["in_overlap_median"].sum()),
                "agree": int((frame["in_definition_32_median"] == frame["in_overlap_median"]).sum()),
            }
        ),
        include_groups=False,
    )
    print(both)


if __name__ == "__main__":
    main()
