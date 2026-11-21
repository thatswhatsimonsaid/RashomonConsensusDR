"""Neighborhood votes and binary vote entropy.

Embeddings vote on whether j is among the k nearest neighbors of i.
Binary entropy is high when the weighted ensemble is split on that vote.
The decomposition

    H(vote) = E[H(vote | method)] + I(vote; method)

is computed for each ordered pair. Self-neighbors are excluded. Ties in
distance use the stable order returned by scikit-learn.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def binary_entropy(probability: np.ndarray) -> np.ndarray:
    """Binary entropy in bits. 0 log 0 is 0."""
    p = np.asarray(probability, dtype=float)
    out = np.zeros(p.shape, dtype=float)
    mask = (p > 0.0) & (p < 1.0)
    pp = p[mask]
    out[mask] = -(pp * np.log2(pp) + (1.0 - pp) * np.log2(1.0 - pp))
    return out


def knn_adjacency(embedding: np.ndarray, k: int) -> np.ndarray:
    """Directed adjacency. Entry (i, j) is 1 when j is a kNN of i, i != j."""
    n = embedding.shape[0]
    if k >= n:
        raise ValueError(f"k={k} is not smaller than n={n}")
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(embedding)
    idx = nn.kneighbors(return_distance=False)[:, 1:]
    adjacency = np.zeros((n, n), dtype=float)
    rows = np.repeat(np.arange(n), k)
    adjacency[rows, idx.reshape(-1)] = 1.0
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def vote_probability(
    embeddings: list[np.ndarray],
    k: int,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted frequency that j is a kNN of i."""
    if weights is None:
        weights = np.ones(len(embeddings), dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(weights) != len(embeddings):
        raise ValueError("weights and embeddings have different lengths")
    if np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError("weights must be finite and non-negative")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("weights sum to zero")
    weights = weights / total
    probability = np.zeros_like(knn_adjacency(embeddings[0], k))
    for weight, embedding in zip(weights, embeddings):
        probability += weight * knn_adjacency(embedding, k)
    return probability


def _method_masses(methods: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    masses = {}
    for method, weight in zip(methods, weights):
        masses[str(method)] = masses.get(str(method), 0.0) + float(weight)
    total = sum(masses.values())
    return {key: value / total for key, value in masses.items()}


def entropy_decomposition(
    embeddings: list[np.ndarray],
    methods: list[str] | np.ndarray,
    k: int,
    weights: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Pooled entropy, mean within-method entropy, and between-method information.

    Method a's vote is the renormalized weighted vote of embeddings from a.
    The identity is checked on every pair: total = within + mutual information.
    """
    methods = np.asarray(methods)
    if len(methods) != len(embeddings):
        raise ValueError("methods and embeddings have different lengths")
    if weights is None:
        weights = np.ones(len(embeddings), dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    pooled = vote_probability(embeddings, k, weights)
    total = binary_entropy(pooled)
    within = np.zeros_like(total)
    masses = _method_masses(methods, weights)
    per_method = {}
    for method, mass in masses.items():
        take = np.flatnonzero(methods == method)
        local = vote_probability(
            [embeddings[i] for i in take], k, weights[take]
        )
        per_method[method] = local
        within += mass * binary_entropy(local)
    mutual = total - within
    if not np.allclose(total, within + mutual, atol=1e-8):
        raise RuntimeError("Entropy decomposition failed a numerical check")
    return {
        "pooled_probability": pooled,
        "total_entropy": total,
        "within_entropy": within,
        "between_information": mutual,
        "mean_total_entropy": float(total.mean()),
        "mean_within_entropy": float(within.mean()),
        "mean_between_information": float(mutual.mean()),
    }


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    sym = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(sym, 0.0)
    return sym


def confidence_matrices(
    probability: np.ndarray,
    eta: float = 0.05,
    floor: float = 1e-3,
) -> dict[str, np.ndarray]:
    """Three symmetric CoMDS weight matrices. They are not the same rule.

    entropy_confidence downweights ambiguous pairs.
    positive_consensus upweights pairs that are often neighbors.
    combined multiplies neighbor frequency by entropy confidence.
    """
    if not 0 < eta <= 1:
        raise ValueError("eta must be in (0, 1]")
    sym = symmetrize(probability)
    entropy = binary_entropy(sym)
    confidence = eta + (1.0 - eta) * (1.0 - entropy)
    np.fill_diagonal(confidence, 0.0)
    positive = np.maximum(sym, floor)
    np.fill_diagonal(positive, 0.0)
    combined = np.maximum(sym * (1.0 - entropy), floor)
    np.fill_diagonal(combined, 0.0)
    return {
        "entropy_confidence": confidence,
        "positive_consensus": positive,
        "combined_frequency_confidence": combined,
    }


def distance_confidence(
    embeddings: list[np.ndarray],
    weights: np.ndarray | None = None,
    eta: float = 0.05,
) -> np.ndarray:
    """Confidence from the weighted variance of normalized pairwise distances.

    Each view is divided by its mean off-diagonal distance before the
    variance is taken. High variance lowers the CoMDS pair weight.
    """
    if weights is None:
        weights = np.ones(len(embeddings), dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    normalized = []
    for embedding in embeddings:
        diff = embedding[:, None, :] - embedding[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 0.0)
        scale = float(dist.sum() / (dist.shape[0] * (dist.shape[0] - 1)))
        normalized.append(dist / max(scale, 1e-12))
    stack = np.stack(normalized, axis=0)
    mean = np.tensordot(weights, stack, axes=(0, 0))
    variance = np.tensordot(weights, (stack - mean) ** 2, axes=(0, 0))
    # A normalized-distance variance above 1 is already strong disagreement.
    spread = np.clip(variance, 0.0, 1.0)
    confidence = eta + (1.0 - eta) * (1.0 - spread)
    np.fill_diagonal(confidence, 0.0)
    return confidence
