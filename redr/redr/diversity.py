"""Geometric diversity of a finite embedding pool.

Candidate count is not diversity. These scores compare neighborhoods and
normalized distances after the embeddings have been validated.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import orthogonal_procrustes

from redr.entropy import knn_adjacency


def pairwise_neighborhood_overlap(embeddings: list[np.ndarray], k: int = 10) -> np.ndarray:
    graphs = [knn_adjacency(z, k) for z in embeddings]
    m = len(graphs)
    out = np.eye(m, dtype=float)
    for i in range(m):
        for j in range(i + 1, m):
            intersection = np.minimum(graphs[i], graphs[j]).sum()
            union = np.maximum(graphs[i], graphs[j]).sum()
            value = float(intersection / union) if union > 0 else 1.0
            out[i, j] = out[j, i] = value
    return out


def pairwise_distance_disagreement(embeddings: list[np.ndarray]) -> np.ndarray:
    """Mean absolute difference of distance matrices scaled by their own mean."""
    mats = []
    for embedding in embeddings:
        diff = embedding[:, None, :] - embedding[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 0.0)
        scale = float(dist.sum() / (dist.shape[0] * (dist.shape[0] - 1)))
        mats.append(dist / max(scale, 1e-12))
    m = len(mats)
    out = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(i + 1, m):
            value = float(np.mean(np.abs(mats[i] - mats[j])))
            out[i, j] = out[j, i] = value
    return out


def procrustes_rmse(a: np.ndarray, b: np.ndarray) -> float:
    """RMSE after centering, scaling, and orthogonal Procrustes alignment."""
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    left = left - left.mean(axis=0)
    right = right - right.mean(axis=0)
    left_scale = np.sqrt(np.mean(np.sum(left**2, axis=1))) + 1e-12
    right_scale = np.sqrt(np.mean(np.sum(right**2, axis=1))) + 1e-12
    rotation, _ = orthogonal_procrustes(left / left_scale, right / right_scale)
    aligned = (left / left_scale) @ rotation
    return float(np.sqrt(np.mean((aligned - right / right_scale) ** 2)))


def pairwise_procrustes(embeddings: list[np.ndarray]) -> np.ndarray:
    m = len(embeddings)
    out = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(i + 1, m):
            value = procrustes_rmse(embeddings[i], embeddings[j])
            out[i, j] = out[j, i] = value
    return out


def redundancy_count(overlap: np.ndarray, threshold: float = 0.98) -> int:
    """Number of embeddings whose neighborhood graph nearly duplicates an earlier one."""
    redundant = 0
    for i in range(1, overlap.shape[0]):
        if np.any(overlap[i, :i] >= threshold):
            redundant += 1
    return redundant


def within_between_overlap(
    overlap: np.ndarray, methods: list[str] | np.ndarray
) -> tuple[float, float]:
    methods = np.asarray(methods)
    within = []
    between = []
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            if methods[i] == methods[j]:
                within.append(overlap[i, j])
            else:
                between.append(overlap[i, j])
    within_mean = float(np.mean(within)) if within else float("nan")
    between_mean = float(np.mean(between)) if between else float("nan")
    return within_mean, between_mean
