"""Embedding quality metrics.

Trustworthiness is sklearn's implementation. Continuity is the symmetric
Venna–Kaski score. LCMC follows Chen and Buja (2009) as used by An and Tang.
Labels are used only by the silhouette score.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.neighbors import NearestNeighbors


def knn_indices(x: np.ndarray, k: int) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(x)
    return nn.kneighbors(return_distance=False)[:, 1:]


def neighborhood_preservation(x: np.ndarray, z: np.ndarray, k: int) -> float:
    high = knn_indices(x, k)
    low = knn_indices(z, k)
    overlaps = [
        len(set(high[i]).intersection(low[i])) / k for i in range(x.shape[0])
    ]
    return float(np.mean(overlaps))


def lcmc(x: np.ndarray, z: np.ndarray, k: int) -> float:
    """Adjusted local continuity meta-criterion at a single neighborhood size."""
    n = x.shape[0]
    preservation = neighborhood_preservation(x, z, k)
    return float(preservation - (k / (n - 1)))


def continuity(x: np.ndarray, z: np.ndarray, k: int) -> float:
    """Venna and Kaski continuity. 1 means high-d neighbors stay neighbors."""
    n = x.shape[0]
    dist_z = pairwise_distances(z)
    np.fill_diagonal(dist_z, np.inf)
    rank_z = np.argsort(np.argsort(dist_z, axis=1), axis=1)
    high = knn_indices(x, k)
    low = knn_indices(z, k)
    penalty = 0.0
    for i in range(n):
        missing = set(high[i].tolist()) - set(low[i].tolist())
        for j in missing:
            penalty += rank_z[i, j] - k
    normalizer = n * k * (2 * n - 3 * k - 1)
    return float(1.0 - (2.0 / normalizer) * penalty)


def triplet_accuracy(
    x: np.ndarray,
    z: np.ndarray,
    n_triplets: int = 4000,
    seed: int = 0,
    triplets: np.ndarray | None = None,
) -> float:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    if triplets is None:
        triplets = _sample_triplets(n, n_triplets, rng)
    i, j, k = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    dx_ij = np.linalg.norm(x[i] - x[j], axis=1)
    dx_ik = np.linalg.norm(x[i] - x[k], axis=1)
    dz_ij = np.linalg.norm(z[i] - z[j], axis=1)
    dz_ik = np.linalg.norm(z[i] - z[k], axis=1)
    usable = np.abs(dx_ij - dx_ik) > 1e-8
    if not np.any(usable):
        return float("nan")
    agree = (dx_ij < dx_ik) == (dz_ij < dz_ik)
    return float(np.mean(agree[usable]))


def _sample_triplets(n: int, n_triplets: int, rng: np.random.Generator) -> np.ndarray:
    raw = rng.integers(0, n, size=(n_triplets * 2, 3))
    keep = (raw[:, 0] != raw[:, 1]) & (raw[:, 0] != raw[:, 2]) & (raw[:, 1] != raw[:, 2])
    picked = raw[keep][:n_triplets]
    if len(picked) < n_triplets:
        raise RuntimeError("Could not sample enough distinct triplets")
    return picked


def distance_rank_correlation(x: np.ndarray, z: np.ndarray) -> float:
    """Spearman correlation of the upper triangles of the two distance matrices."""
    dx = pairwise_distances(x)
    dz = pairwise_distances(z)
    iu = np.triu_indices(x.shape[0], k=1)
    corr, _p = spearmanr(dx[iu], dz[iu])
    return float(corr)


def multi_k_lcmc(x: np.ndarray, z: np.ndarray, ks: list[int] | tuple[int, ...]) -> float:
    """Mean adjusted LCMC over several neighborhood sizes."""
    return float(np.mean([lcmc(x, z, k) for k in ks]))


def knn_classification_accuracy(z: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """Leave-one-out accuracy of a kNN classifier on the embedding."""
    neighbors = knn_indices(z, k)
    predicted = []
    for i in range(len(labels)):
        votes = labels[neighbors[i]]
        values, counts = np.unique(votes, return_counts=True)
        predicted.append(values[np.argmax(counts)])
    return float(np.mean(np.asarray(predicted) == np.asarray(labels)))


def scale_adjusted_distance_error(x: np.ndarray, z: np.ndarray) -> float:
    """Mean absolute error of distances after each matrix is divided by its mean."""
    dx = pairwise_distances(np.asarray(x, dtype=float))
    dz = pairwise_distances(np.asarray(z, dtype=float))
    iu = np.triu_indices(dx.shape[0], k=1)
    dx = dx[iu]
    dz = dz[iu]
    dx = dx / max(float(dx.mean()), 1e-12)
    dz = dz / max(float(dz.mean()), 1e-12)
    return float(np.mean(np.abs(dx - dz)))


def latent_triplet_accuracy(
    latent: np.ndarray,
    z: np.ndarray,
    triplets: np.ndarray,
) -> float:
    """Triplet accuracy against a 1-d latent coordinate, using absolute differences."""
    t = np.asarray(latent, dtype=float).reshape(-1)
    high = np.column_stack([t, np.zeros(len(t))])
    return triplet_accuracy(high, z, triplets=triplets)


def evaluate_embedding(
    x: np.ndarray,
    z: np.ndarray,
    labels: np.ndarray | None,
    k: int = 10,
    triplets: np.ndarray | None = None,
) -> dict:
    z = np.asarray(z, dtype=np.float64)
    out = {
        "trustworthiness": float(trustworthiness(x, z, n_neighbors=k)),
        "continuity": continuity(x, z, k),
        "neighborhood_preservation": neighborhood_preservation(x, z, k),
        "lcmc": lcmc(x, z, k),
        "triplet_accuracy": triplet_accuracy(x, z, triplets=triplets),
        "silhouette": np.nan,
    }
    if labels is not None:
        classes = np.unique(labels)
        if 1 < len(classes) < len(labels):
            out["silhouette"] = float(silhouette_score(z, labels, metric="euclidean"))
    return out
