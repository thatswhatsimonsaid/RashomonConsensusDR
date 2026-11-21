"""Checks that a list of embeddings is safe to hand to CoMDS."""

from __future__ import annotations

import numpy as np


def validate_embeddings(
    embeddings: list[np.ndarray],
    observation_ids: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Require one shared row order, finite coordinates, and nondegenerate shape.

    Row i of every embedding is the same observation. When observation_ids is
    given, it is that shared identifier vector and must be unique.
    """
    if len(embeddings) == 0:
        raise ValueError("No embeddings were provided")
    arrays = [np.asarray(z, dtype=np.float64) for z in embeddings]
    if arrays[0].ndim != 2:
        raise ValueError(f"Embeddings must be 2-dimensional, got ndim {arrays[0].ndim}")
    n, d = arrays[0].shape
    if n < 3 or d < 1:
        raise ValueError(f"Embedding shape {(n, d)} is too small for consensus")
    for i, z in enumerate(arrays):
        if z.shape != (n, d):
            raise ValueError(f"Embedding {i} has shape {z.shape}, expected {(n, d)}")
        if not np.isfinite(z).all():
            raise ValueError(f"Embedding {i} contains a non-finite value")
        if float(np.std(z)) < 1e-8:
            raise ValueError(f"Embedding {i} is degenerate")
    if observation_ids is not None:
        ids = np.asarray(observation_ids)
        if len(ids) != n:
            raise ValueError(
                f"Observation ids have length {len(ids)} and embeddings have {n} rows"
            )
        if len(np.unique(ids)) != n:
            raise ValueError("Observation ids are not unique")
    return arrays
