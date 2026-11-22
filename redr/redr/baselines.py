"""Strong multi-view baselines for the Checkpoint 4 product manifold.

Several of these use the known feature blocks. They are privileged
integrations, not proposed methods. They exist to ask whether CoMDS of
specialist embeddings does more than concatenate or average the blocks.

Latent coordinates are not read by any constructor. They are used only by
the caller that scores a finished embedding.
"""

from __future__ import annotations

import numpy as np
from sklearn.manifold import Isomap, MDS
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


def classical_mds(distance: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Metric MDS by double centering. Negative eigenvalues are clipped."""
    squared = np.asarray(distance, dtype=float) ** 2
    n = squared.shape[0]
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ squared @ centering
    gram = 0.5 * (gram + gram.T)
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1][:n_components]
    kept = np.clip(values[order], 0.0, None)
    return vectors[:, order] * np.sqrt(kept)


def _scaled_block(block: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(np.asarray(block, dtype=float))


def _unit_distance(block: np.ndarray) -> np.ndarray:
    distance = pairwise_distances(_scaled_block(block))
    mean = float(distance.mean())
    if mean < 1e-12:
        return distance
    return distance / mean


def block_matrix(data: dict) -> np.ndarray:
    """Each known block standardized, then concatenated with the nuisance block."""
    parts = [_scaled_block(data["block_u"]), _scaled_block(data["block_v"])]
    if data.get("nuisance") is not None and np.asarray(data["nuisance"]).size:
        parts.append(_scaled_block(data["nuisance"]))
    return np.hstack(parts).astype(np.float64)


def isomap_embedding(matrix: np.ndarray, n_neighbors: int = 10, n_components: int = 2) -> np.ndarray:
    return np.asarray(
        Isomap(n_neighbors=n_neighbors, n_components=n_components).fit_transform(matrix),
        dtype=float,
    )


def concat_one_dimensional(data: dict, n_neighbors: int = 10) -> np.ndarray:
    """Isomap of each block down to one coordinate, then stacked.

    This is ordinary multi-view integration. It uses the known blocks.
    """
    columns = []
    for key in ("block_u", "block_v"):
        unfolded = isomap_embedding(_scaled_block(data[key]), n_neighbors=n_neighbors, n_components=1)
        columns.append(np.asarray(unfolded).reshape(-1))
    return np.column_stack(columns)


def aggregated_distance_mds(data: dict) -> np.ndarray:
    """Classical MDS of the mean of the two blockwise distance matrices."""
    mean = 0.5 * (_unit_distance(data["block_u"]) + _unit_distance(data["block_v"]))
    np.fill_diagonal(mean, 0.0)
    return classical_mds(mean, n_components=2)


def block_distance_embeddings(data: dict) -> list[np.ndarray]:
    """One classical MDS configuration per block. CoMDS can take these views."""
    views = []
    for key in ("block_u", "block_v"):
        distance = _unit_distance(data[key])
        np.fill_diagonal(distance, 0.0)
        views.append(classical_mds(distance, n_components=2))
    return views


def sklearn_mds_aggregated(data: dict) -> np.ndarray:
    """SMACOF on the mean blockwise distance, as a second classical aggregator."""
    mean = 0.5 * (_unit_distance(data["block_u"]) + _unit_distance(data["block_v"]))
    np.fill_diagonal(mean, 0.0)
    model = MDS(
        n_components=2,
        metric="precomputed",
        random_state=0,
        normalized_stress="auto",
        n_init=2,
        max_iter=200,
        init="classical_mds",
    )
    return np.asarray(model.fit_transform(mean), dtype=float)
