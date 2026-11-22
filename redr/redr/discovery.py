"""Automatic candidates from a matrix whose feature blocks are not labeled.

Nothing in this module reads latent coordinates. Feature subsets, random
projections, and feature clusters are functions of the observed matrix and
a seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from redr.metrics import neighborhood_preservation, scale_adjusted_distance_error
from redr.product import generate_product


OBJECTIVE_COLUMNS = (
    "local_loss",
    "multiscale_loss",
    "geodesic_loss",
    "global_loss",
    "subset_loss",
)


def observed_matrix(data: dict, mixing: str, mix_seed: int) -> np.ndarray:
    """Return the matrix a method is allowed to see.

    A rotation mixes columns and preserves row-wise distances, so full-matrix
    Isomap is unchanged and column subsets are no longer the original blocks.
    """
    matrix = np.asarray(data["X"], dtype=float)
    if mixing == "aligned":
        return matrix
    if mixing != "rotated":
        raise ValueError(mixing)
    rng = np.random.default_rng(mix_seed)
    width = matrix.shape[1]
    orthogonal, _ = np.linalg.qr(rng.normal(size=(width, width)))
    return matrix @ orthogonal


def make_dataset(
    n_samples: int,
    mixing: str,
    hide_second: bool,
    noise: float,
    n_nuisance: int,
    preserve_scale: bool,
    aspect: tuple[float, float],
    random_state: int,
    contaminate: bool = False,
) -> dict:
    data = generate_product(
        n_samples=n_samples,
        aspect=aspect,
        noise=noise,
        n_nuisance=n_nuisance,
        hide_second=hide_second,
        preserve_scale=preserve_scale,
        random_state=random_state,
    )
    data["X_visible"] = observed_matrix(data, mixing, mix_seed=random_state + 91)
    data["mixing"] = mixing
    data["contaminate"] = contaminate
    return data


def discover_candidates(
    matrix: np.ndarray,
    random_state: int,
    contaminate: bool,
) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    """Fit embeddings that do not know which columns are informative."""
    rng = np.random.default_rng(random_state)
    matrix = np.asarray(matrix, dtype=float)
    n, width = matrix.shape
    embeddings: dict[str, np.ndarray] = {}
    subsets: list[np.ndarray] = []

    embeddings["pca_full"] = PCA(n_components=2, random_state=0).fit_transform(matrix)
    embeddings["isomap_k10"] = _isomap(matrix, 10)
    embeddings["isomap_k25"] = _isomap(matrix, 25)

    subset_size = max(3, int(round(0.45 * width)))
    for index in range(8):
        columns = np.sort(rng.choice(width, size=subset_size, replace=False))
        subsets.append(columns)
        embeddings[f"subset_{index}"] = _isomap(matrix[:, columns], min(10, subset_size - 1))

    for index in range(4):
        projection = rng.normal(size=(width, 4)) / 2.0
        projected = StandardScaler().fit_transform(matrix @ projection)
        embeddings[f"proj_{index}"] = _isomap(projected, 10)

    labels = _feature_clusters(matrix, n_clusters=2)
    for cluster in np.unique(labels):
        columns = np.flatnonzero(labels == cluster)
        if len(columns) < 3:
            continue
        subsets.append(columns)
        embeddings[f"cluster_{int(cluster)}"] = _isomap(matrix[:, columns], min(10, len(columns) - 1))

    if contaminate:
        for index in range(4):
            embeddings[f"noise_{index}"] = rng.normal(size=(n, 2))
    return embeddings, subsets


def _isomap(matrix: np.ndarray, n_neighbors: int) -> np.ndarray:
    neighbors = min(n_neighbors, matrix.shape[0] - 2, max(matrix.shape[1] - 1, 2))
    neighbors = max(2, neighbors)
    return np.asarray(
        Isomap(n_neighbors=neighbors, n_components=2).fit_transform(matrix),
        dtype=float,
    )


def _feature_clusters(matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    correlation = np.corrcoef(matrix, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0)
    distance = np.clip(1.0 - np.abs(correlation), 0.0, None)
    np.fill_diagonal(distance, 0.0)
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average",
    )
    return model.fit_predict(distance)


def score_objectives(
    matrix: np.ndarray,
    embeddings: dict[str, np.ndarray],
    subsets: list[np.ndarray],
) -> pd.DataFrame:
    """Structural losses. Lower is better. None of them use latent coordinates."""
    geodesic = _geodesic(matrix, n_neighbors=15)
    rows = []
    for embedding_id, embedding in embeddings.items():
        local = 1.0 - neighborhood_preservation(matrix, embedding, k=5)
        multiscale = 1.0 - np.mean(
            [neighborhood_preservation(matrix, embedding, k=k) for k in (5, 15, 30)]
        )
        geodesic_loss = _rank_loss(geodesic, embedding)
        global_loss = scale_adjusted_distance_error(matrix, embedding)
        subset_loss = 1.0 - _best_subset_fidelity(matrix, embedding, subsets)
        rows.append(
            {
                "embedding_id": embedding_id,
                "family": embedding_id.split("_")[0],
                "is_noise": embedding_id.startswith("noise"),
                "local_loss": float(local),
                "multiscale_loss": float(multiscale),
                "geodesic_loss": float(geodesic_loss),
                "global_loss": float(global_loss),
                "subset_loss": float(subset_loss),
            }
        )
    return pd.DataFrame(rows)


def _geodesic(matrix: np.ndarray, n_neighbors: int) -> np.ndarray:
    model = Isomap(n_neighbors=n_neighbors, n_components=2)
    model.fit(matrix)
    return np.asarray(model.dist_matrix_, dtype=float)


def _rank_loss(geodesic: np.ndarray, embedding: np.ndarray) -> float:
    distances = pairwise_distances(np.asarray(embedding, dtype=float))
    iu = np.triu_indices(len(distances), k=1)
    correlation = spearmanr(geodesic[iu], distances[iu]).correlation
    if correlation != correlation:
        return float("nan")
    return float(1.0 - correlation)


def _best_subset_fidelity(
    matrix: np.ndarray,
    embedding: np.ndarray,
    subsets: list[np.ndarray],
) -> float:
    if not subsets:
        return 0.0
    scores = []
    for columns in subsets:
        block = matrix[:, columns]
        if block.shape[1] < 2:
            continue
        scores.append(neighborhood_preservation(block, embedding, k=min(10, block.shape[0] - 2)))
    if not scores:
        return 0.0
    return float(np.max(scores))


def marginal_select(
    frame: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    matrix: np.ndarray,
    subsets: list[np.ndarray],
    gain: float = 0.05,
    keep_max: int = 6,
) -> pd.DataFrame:
    """Add an embedding when it preserves some observed subset better than the set.

    The gain is neighborhood preservation of a feature subset, not a latent
    coordinate. Noise embeddings do not clear the gain. The threshold is
    fixed before any latent score is computed.
    """
    if len(frame) == 0:
        return frame.copy()
    start = frame.sort_values("multiscale_loss", kind="mergesort").iloc[0]
    chosen = [str(start["embedding_id"])]
    remaining = [str(value) for value in frame["embedding_id"] if str(value) not in chosen]
    while remaining and len(chosen) < keep_max:
        best_id = None
        best_gain = gain
        for candidate in remaining:
            improvement = _subset_improvement(
                matrix, embeddings[candidate], [embeddings[key] for key in chosen], subsets
            )
            if improvement > best_gain:
                best_gain = improvement
                best_id = candidate
        if best_id is None:
            break
        chosen.append(best_id)
        remaining.remove(best_id)
    return frame.loc[frame["embedding_id"].astype(str).isin(chosen)].copy()


def _subset_improvement(
    matrix: np.ndarray,
    candidate: np.ndarray,
    chosen: list[np.ndarray],
    subsets: list[np.ndarray],
) -> float:
    best = 0.0
    for columns in subsets:
        block = matrix[:, columns]
        if block.shape[1] < 2:
            continue
        k = min(10, block.shape[0] - 2)
        alone = neighborhood_preservation(block, candidate, k=k)
        current = max(neighborhood_preservation(block, view, k=k) for view in chosen)
        best = max(best, alone - current)
    return float(best)


def quality_weights(frame: pd.DataFrame, column: str = "multiscale_loss") -> np.ndarray:
    """Larger weight for a smaller structural loss. Not a latent weight."""
    loss = frame[column].to_numpy(dtype=float)
    return 1.0 / (loss + 0.05)
