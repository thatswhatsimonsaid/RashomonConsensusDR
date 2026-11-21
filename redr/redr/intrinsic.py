"""Recovery scores against a known unfolded manifold.

One-dimensional ordering of a single latent coordinate is reported separately
from distances in the two-dimensional unfolding. A high score on the long
axis is not treated as recovery of the surface.
"""

from __future__ import annotations

import numpy as np

from redr.diversity import procrustes_rmse
from redr.metrics import (
    distance_rank_correlation,
    neighborhood_preservation,
    scale_adjusted_distance_error,
    triplet_accuracy,
)


def _as_2d(coord: np.ndarray) -> np.ndarray:
    values = np.asarray(coord, dtype=float).reshape(-1)
    return np.column_stack([values, np.zeros(len(values))])


def coordinate_triplet_accuracy(
    coordinate: np.ndarray,
    embedding: np.ndarray,
    triplets: np.ndarray,
) -> float:
    """Triplet accuracy against absolute differences of one latent coordinate."""
    return triplet_accuracy(_as_2d(coordinate), embedding, triplets=triplets)


def balance_axes(intrinsic: np.ndarray) -> np.ndarray:
    """Give each unfolded axis unit variance. Translation does not matter."""
    values = np.asarray(intrinsic, dtype=float)
    scale = values.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return (values - values.mean(axis=0)) / scale


def triplet_order_disagreement(
    intrinsic: np.ndarray,
    triplets: np.ndarray,
) -> float:
    """Fraction of triplets whose nearer neighbor flips when height is added.

    The reference order uses only the first intrinsic axis (arc length).
    The comparison uses Euclidean distance in the full unfolding.
    """
    intrinsic = np.asarray(intrinsic, dtype=float)
    i, j, k = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    arc = intrinsic[:, 0]
    d_arc_ij = np.abs(arc[i] - arc[j])
    d_arc_ik = np.abs(arc[i] - arc[k])
    d_ij = np.linalg.norm(intrinsic[i] - intrinsic[j], axis=1)
    d_ik = np.linalg.norm(intrinsic[i] - intrinsic[k], axis=1)
    usable = (np.abs(d_arc_ij - d_arc_ik) > 1e-8) & (np.abs(d_ij - d_ik) > 1e-8)
    if not np.any(usable):
        return float("nan")
    arc_nearer_j = d_arc_ij < d_arc_ik
    full_nearer_j = d_ij < d_ik
    return float(np.mean(arc_nearer_j[usable] != full_nearer_j[usable]))


def intrinsic_recovery(
    intrinsic: np.ndarray,
    embedding: np.ndarray,
    triplets: np.ndarray,
    k: int = 10,
) -> dict:
    """Score an embedding against the unfolded coordinates.

    ``intrinsic_triplet_isometric`` uses the raw unfolding, so a long thin
    rectangle is dominated by arc length. ``intrinsic_triplet_balanced``
    rescales each axis to unit variance before the triplet test.
    ``arc_triplet_accuracy`` and ``height_triplet_accuracy`` are
    one-dimensional orderings and are not manifold-recovery scores.
    """
    intrinsic = np.asarray(intrinsic, dtype=float)
    embedding = np.asarray(embedding, dtype=float)
    balanced = balance_axes(intrinsic)
    return {
        "intrinsic_triplet_isometric": triplet_accuracy(
            intrinsic, embedding, triplets=triplets
        ),
        "intrinsic_triplet_balanced": triplet_accuracy(
            balanced, embedding, triplets=triplets
        ),
        "arc_triplet_accuracy": coordinate_triplet_accuracy(
            intrinsic[:, 0], embedding, triplets
        ),
        "height_triplet_accuracy": coordinate_triplet_accuracy(
            intrinsic[:, 1], embedding, triplets
        ),
        "intrinsic_neighborhood": neighborhood_preservation(intrinsic, embedding, k),
        "intrinsic_neighborhood_balanced": neighborhood_preservation(
            balanced, embedding, k
        ),
        "intrinsic_rank_correlation": distance_rank_correlation(intrinsic, embedding),
        "intrinsic_rank_correlation_balanced": distance_rank_correlation(
            balanced, embedding
        ),
        "intrinsic_distance_error": scale_adjusted_distance_error(intrinsic, embedding),
        "intrinsic_procrustes": procrustes_rmse(embedding, balanced),
    }
