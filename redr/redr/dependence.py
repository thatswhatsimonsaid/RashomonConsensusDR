"""Cluster observed columns by dependence and embed each cluster in one dimension.

Pearson correlation does not group the coordinates of one curve: cosine and
sine of the same angle are uncorrelated. Mutual information is the dependence
used here. Nothing in this module reads latent coordinates.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler

from redr.metrics import scale_adjusted_distance_error
from redr.rashomon import membership_mask


def mutual_information(matrix: np.ndarray, random_state: int = 0) -> np.ndarray:
    """Symmetric mutual information of the columns. The diagonal is left at 0."""
    matrix = np.asarray(matrix, dtype=float)
    width = matrix.shape[1]
    scores = np.zeros((width, width), dtype=float)
    for column in range(width):
        others = [index for index in range(width) if index != column]
        values = mutual_info_regression(
            matrix[:, others],
            matrix[:, column],
            random_state=random_state,
            n_neighbors=5,
        )
        for index, value in zip(others, values):
            scores[index, column] = float(value)
    return 0.5 * (scores + scores.T)


def _pearson_affinity(matrix: np.ndarray) -> np.ndarray:
    correlation = np.corrcoef(np.asarray(matrix, dtype=float), rowvar=False)
    correlation = np.nan_to_num(np.abs(correlation), nan=0.0)
    np.fill_diagonal(correlation, 0.0)
    return correlation


def cluster_columns(
    matrix: np.ndarray,
    n_clusters: int,
    kind: str = "mutual_information",
    random_state: int = 0,
) -> list[np.ndarray]:
    """Return column-index groups. Groups smaller than one column are absent."""
    if kind == "mutual_information":
        affinity = mutual_information(matrix, random_state=random_state)
    elif kind == "pearson":
        affinity = _pearson_affinity(matrix)
    else:
        raise ValueError(kind)
    scale = float(np.max(affinity)) if affinity.size else 0.0
    distance = np.ones_like(affinity) if scale <= 0.0 else 1.0 - affinity / scale
    np.fill_diagonal(distance, 0.0)
    labels = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average",
    ).fit_predict(distance)
    groups = []
    for label in sorted(set(labels.tolist())):
        columns = np.flatnonzero(labels == label)
        if len(columns):
            groups.append(columns)
    return groups


def one_dimensional(matrix: np.ndarray, columns: np.ndarray, n_neighbors: int = 10) -> np.ndarray:
    block = StandardScaler().fit_transform(np.asarray(matrix, dtype=float)[:, columns])
    neighbors = max(2, min(n_neighbors, block.shape[0] - 2, max(block.shape[1] - 1, 2)))
    return np.asarray(
        Isomap(n_neighbors=neighbors, n_components=1).fit_transform(block),
        dtype=float,
    ).reshape(-1)


def self_view_losses(matrix: np.ndarray, groups: list[np.ndarray]) -> list[float]:
    """One-dimensional distance error on the columns that defined the view."""
    losses = []
    for columns in groups:
        if len(columns) < 3:
            losses.append(float("inf"))
            continue
        coordinate = one_dimensional(matrix, columns)
        block = StandardScaler().fit_transform(matrix[:, columns])
        losses.append(scale_adjusted_distance_error(block, coordinate.reshape(-1, 1)))
    return losses


def rashomon_groups(
    groups: list[np.ndarray],
    losses: list[float],
    epsilon: float = 0.05,
) -> list[np.ndarray]:
    """Keep groups whose own one-dimensional loss is inside the relative band.

    A group with fewer than three columns has an infinite loss and stays out
    unless every loss is infinite.
    """
    keep, _best, _tolerance = membership_mask(np.asarray(losses, dtype=float), epsilon)
    chosen = [group for group, flag in zip(groups, keep) if flag and len(group) >= 3]
    if chosen:
        return chosen
    return [group for group in groups if len(group) >= 3]


def stack_coordinates(columns: list[np.ndarray]) -> np.ndarray:
    """Standardize one-dimensional coordinates and return a two-column embedding.

    One coordinate is padded with a zero column. Three or more are reduced
    with PCA. Distances, which the triplet scores use, do not depend on the
    sign of a coordinate.
    """
    if not columns:
        raise ValueError("no coordinates")
    stacked = np.column_stack([np.asarray(column, dtype=float).reshape(-1) for column in columns])
    stacked = StandardScaler().fit_transform(stacked)
    if stacked.shape[1] == 1:
        return np.column_stack([stacked[:, 0], np.zeros(len(stacked))])
    if stacked.shape[1] == 2:
        return stacked
    return PCA(n_components=2, random_state=0).fit_transform(stacked)
