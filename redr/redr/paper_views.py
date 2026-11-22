"""Data and views for the An and Tang simulations.

The Gaussian mixture and Swiss roll follow coMDS-manuscript/R/dgps.R.
NumPy's generator is not R's, so a shared seed does not reproduce the same
sample. The formulas match: three clusters of sizes 150, 200, and 250, the
stated means, and covariance Sigma + Z Z^T with Z_ij ~ N(0, 0.5^2).

PHATE, HLLE, Sammon, non-metric MDS, and multi-SNE are not installed. The
view list is the subset that scikit-learn and umap-learn can fit. Kernel
PCA uses kernlab's rbfdot convention, gamma = sigma, not 1/(2 sigma^2).
Meta-Spec's ensemble distance and the kernel-PCA step follow methods.R
and methods-utils.R. The graph weights follow Sun et al. Definitions 3.2
and 3.3.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding, SpectralEmbedding, TSNE
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from redr.baselines import classical_mds


def mixture_of_gaussians(random_state: int = 331) -> dict:
    rng = np.random.RandomState(random_state)
    sizes = (150, 200, 250)
    means = (
        np.array([-3.0, -2.0, 0.0]),
        np.array([2.0, -4.0, 1.0]),
        np.array([0.0, 6.0, 6.0]),
    )
    base = np.array(
        [[1.0, 0.3, 0.2], [0.3, 1.0, 0.4], [0.2, 0.4, 1.0]],
        dtype=float,
    )
    blocks = []
    labels = []
    for index, (size, mean) in enumerate(zip(sizes, means), start=1):
        shock = rng.normal(loc=0.0, scale=0.5, size=(3, 3))
        covariance = base + shock @ shock.T
        draw = rng.multivariate_normal(mean, covariance, size=size)
        blocks.append(draw)
        labels.append(np.full(size, index, dtype=int))
    raw = np.vstack(blocks)
    return {
        "X": StandardScaler().fit_transform(raw).astype(np.float64),
        "labels": np.concatenate(labels),
        "n": int(raw.shape[0]),
    }


def swiss_roll(
    n_samples: int = 400,
    noise: float = 0.1,
    random_state: int = 331,
) -> dict:
    rng = np.random.RandomState(random_state)
    t = rng.uniform(np.pi, 3.5 * np.pi, size=n_samples)
    width = rng.uniform(-1.0, 1.0, size=n_samples)
    raw = np.column_stack([t * np.cos(t), t * np.sin(t), width])
    raw = raw + rng.normal(scale=noise, size=raw.shape)
    return {
        "X": StandardScaler().fit_transform(raw).astype(np.float64),
        "t": t.astype(np.float64),
        "width": width.astype(np.float64),
        "intrinsic": np.column_stack([t, width]).astype(np.float64),
        "n": int(n_samples),
    }


def classical_views(matrix: np.ndarray, random_state: int = 0) -> dict[str, np.ndarray]:
    """Representative An and Tang views. Each value is an (n, 2) embedding."""
    matrix = np.asarray(matrix, dtype=np.float64)
    views: dict[str, np.ndarray] = {
        "pca": PCA(n_components=2, random_state=random_state).fit_transform(matrix),
        "mds": classical_mds(pairwise_distances(matrix), n_components=2),
        "lle": LocallyLinearEmbedding(
            n_neighbors=20, n_components=2, method="standard", random_state=random_state
        ).fit_transform(matrix),
        "isomap": np.asarray(
            Isomap(n_neighbors=20, n_components=2).fit_transform(matrix), dtype=float
        ),
        "laplacian": np.asarray(
            SpectralEmbedding(n_components=2, n_neighbors=20, random_state=random_state).fit_transform(matrix),
            dtype=float,
        ),
    }
    for sigma in (0.01, 0.001):
        # kernlab::rbfdot uses exp(-sigma ||x-x'||^2). sklearn's gamma is that sigma.
        gamma = float(sigma)
        views[f"kpca_sigma_{sigma}"] = KernelPCA(
            n_components=2, kernel="rbf", gamma=gamma, random_state=random_state
        ).fit_transform(matrix)
    for perplexity in (30, 50):
        views[f"tsne_perplexity_{perplexity}"] = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            init="pca",
            max_iter=500,
        ).fit_transform(matrix)
    import umap

    for neighbors in (30, 50):
        views[f"umap_neighbors_{neighbors}"] = np.asarray(
            umap.UMAP(n_neighbors=neighbors, n_components=2, random_state=random_state).fit_transform(matrix),
            dtype=float,
        )
    return {name: np.asarray(value, dtype=np.float64) for name, value in views.items()}


def meta_spec_distance(embeddings: list[np.ndarray]) -> np.ndarray:
    """Pointwise ensemble distance from Ma et al., as copied in methods-utils.R."""
    stacked = np.stack([pairwise_distances(np.asarray(embedding, dtype=float)) for embedding in embeddings])
    n = stacked.shape[1]
    ensemble = np.zeros((n, n), dtype=float)
    for source in range(n):
        local = stacked[:, source, :]
        norms = np.linalg.norm(local, axis=1, keepdims=True)
        normed = local / np.maximum(norms, 1e-12)
        cosine = normed @ normed.T
        _values, vectors = np.linalg.eigh(cosine)
        weights = np.abs(vectors[:, -1])
        ensemble[:, source] = normed.T @ weights
    return 0.5 * (ensemble + ensemble.T)


def standardize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Sun et al. scale: mean zero and mean squared norm one."""
    centered = np.asarray(embedding, dtype=float)
    centered = centered - centered.mean(axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    return centered / max(scale, 1e-12)


def induced_graph_weights(
    embedding: np.ndarray,
    neighbor_index: np.ndarray,
    gamma: float = 1.0,
) -> np.ndarray:
    """Definition 3.2 weights. ``neighbor_index[j]`` is the high-d kNN of point j."""
    z = standardize_embedding(embedding)
    n = z.shape[0]
    weights = np.ones((n, n), dtype=float)
    diff = z[:, None, :] - z[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    for point in range(n):
        neighbors = neighbor_index[point]
        squared = dist2[neighbors, point]
        weights[neighbors, point] = (squared + gamma) / (squared + gamma + 1.0)
    return weights


def soft_jaccard_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Definition 3.3. Identical weights have distance 0."""
    total = left + right
    safe = np.maximum(total, 1e-12)
    numerator = float(np.sum(np.minimum(left, right) / safe))
    denominator = float(np.sum(np.maximum(left, right) / safe))
    return 1.0 - numerator / max(denominator, 1e-12)


def meta_spec_kpca(embeddings: list[np.ndarray]) -> np.ndarray:
    """Kernel PCA of the Meta-Spec distance, skipping the leading eigenvector."""
    distance = meta_spec_distance(embeddings)
    scale = float(np.median(distance[np.triu_indices(distance.shape[0], k=1)]))
    kernel = np.exp(-(distance ** 2) / max(scale, 1e-12) ** 2)
    values, vectors = np.linalg.eigh(kernel)
    order = np.argsort(values)[::-1]
    chosen = order[1:3]
    kept = np.clip(values[chosen], 0.0, None)
    return vectors[:, chosen] * kept
