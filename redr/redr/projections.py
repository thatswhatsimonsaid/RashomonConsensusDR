"""One-dimensional embeddings of random projections.

These rules were specified after the two-dimensional discovery pool failed
to isolate complementary directions, and before the multi-seed run of this
module. Low one-dimensional stress is not assumed to mean a useful factor:
the comparison exists to test that assumption. Latent coordinates are not
read here.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler

from redr.metrics import scale_adjusted_distance_error
from redr.rashomon import membership_mask


def projection_bank(
    matrix: np.ndarray,
    n_projections: int,
    random_state: int,
    pca_dim: int | None = None,
    target_dim: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one-dimensional Isomap coordinates and their self-view stress.

    ``pca_dim`` first restricts the matrix to that many principal components.
    The projection draws are functions of ``random_state`` only.
    """
    matrix = np.asarray(matrix, dtype=float)
    rng = np.random.default_rng(random_state)
    basis = matrix
    if pca_dim is not None:
        dim = min(int(pca_dim), matrix.shape[1], matrix.shape[0] - 1)
        basis = PCA(n_components=dim, random_state=0).fit_transform(matrix)
    width = basis.shape[1]
    coordinates = []
    stresses = []
    for _index in range(n_projections):
        projected = StandardScaler().fit_transform(basis @ rng.normal(size=(width, target_dim)))
        coordinate = np.asarray(
            Isomap(n_neighbors=10, n_components=1).fit_transform(projected),
            dtype=float,
        ).reshape(-1)
        stresses.append(scale_adjusted_distance_error(projected, coordinate.reshape(-1, 1)))
        coordinates.append(coordinate)
    return np.column_stack(coordinates), np.asarray(stresses, dtype=float)


def _absolute_correlation(coordinates: np.ndarray) -> np.ndarray:
    count = coordinates.shape[1]
    correlation = np.eye(count)
    for left in range(count):
        for right in range(left + 1, count):
            value = spearmanr(coordinates[:, left], coordinates[:, right]).correlation
            correlation[left, right] = correlation[right, left] = 0.0 if value != value else abs(float(value))
    return correlation


def stress_rashomon_pair(
    coordinates: np.ndarray,
    stresses: np.ndarray,
    epsilon: float = 0.05,
) -> np.ndarray:
    """Concatenate the lowest-stress coordinate with the least similar survivor.

    Membership uses the one-dimensional stress of each projection. If the
    band contains only one projection, the second coordinate is the least
    correlated projection in the whole bank.
    """
    keep, _best, _tolerance = membership_mask(stresses, epsilon)
    order = np.argsort(stresses)
    first = int(order[0])
    correlation = _absolute_correlation(coordinates)
    pool = np.flatnonzero(keep)
    pool = pool[pool != first]
    if len(pool) == 0:
        pool = np.asarray([index for index in order if index != first], dtype=int)
    second = int(pool[np.argmin(correlation[first, pool])])
    chosen = StandardScaler().fit_transform(coordinates[:, [first, second]])
    return chosen


def agreement_pair(coordinates: np.ndarray) -> np.ndarray:
    """Split the bank into two agreement clusters and concatenate their centers.

    The center of a cluster is the coordinate with the highest mean absolute
    Spearman correlation to the other members. There is no stress gate.
    """
    correlation = _absolute_correlation(coordinates)
    distance = np.clip(1.0 - correlation, 0.0, None)
    np.fill_diagonal(distance, 0.0)
    labels = AgglomerativeClustering(
        n_clusters=2,
        metric="precomputed",
        linkage="average",
    ).fit_predict(distance)
    centers = []
    for label in (0, 1):
        members = np.flatnonzero(labels == label)
        if len(members) == 1:
            centers.append(int(members[0]))
            continue
        means = (correlation[np.ix_(members, members)].sum(axis=1) - 1.0) / (len(members) - 1)
        centers.append(int(members[int(np.argmax(means))]))
    return StandardScaler().fit_transform(coordinates[:, centers])
