"""Fit method-specific candidate embeddings with the vendored trainers.

Epoch counts are caller arguments. The PaCMAP weight schedule and the CNE
losses are the upstream implementations.
"""

from __future__ import annotations

import time

import numpy as np
import torch

import cne
from parampacmap import ParamPaCMAP
from parampacmap.parampacmap import pacmap_weight_schedule
from parampacmap.utils.data import generate_pair


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def high_d_neighbors(x: np.ndarray, k: int) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(x)
    return nn.kneighbors(return_distance=False)[:, 1:]


def pacmap_pairs(x: np.ndarray, seed: int = 0):
    neighbors, mid_near, further, _tree = generate_pair(
        np.asarray(x, dtype=np.float32),
        n_neighbors=10,
        n_MN=5,
        n_FP=20,
        distance="euclidean",
        verbose=False,
        random_state=seed,
    )
    return neighbors, mid_near, further


def fit_cne(
    x: np.ndarray,
    x_pca: np.ndarray,
    loss_mode: str,
    seed: int,
    label_weight: float,
    n_epochs: int,
    k: int = 15,
    graph=None,
    early_exaggeration: bool = True,
    spectrum: float | None = None,
) -> tuple[np.ndarray, float]:
    set_seed(seed)
    kwargs = dict(
        optimizer="adam",
        parametric=True,
        loss_mode=loss_mode,
        label_weights=float(label_weight),
        seed=int(seed),
        task_type="pca",
        n_epochs=int(n_epochs),
        batch_size=min(128, x.shape[0]),
        k=k,
        negative_samples=5,
        use_keops=False,
        print_freq_epoch=None,
        early_exaggeration=bool(early_exaggeration),
    )
    if spectrum is not None:
        kwargs["s"] = float(spectrum)
    # The vendored loss returns an undefined neighbor tensor when a batch has
    # at most negative_samples * embedding_dim rows. Dropping the incomplete
    # final batch of edges avoids that path. It does not change the objective.
    from cne._cne import FastTensorRegressionDataLoader

    if not getattr(FastTensorRegressionDataLoader, "_redr_drop_last", False):
        original_init = FastTensorRegressionDataLoader.__init__

        def _drop_incomplete(self, *args, **init_kwargs):
            init_kwargs["drop_last"] = True
            return original_init(self, *args, **init_kwargs)

        FastTensorRegressionDataLoader.__init__ = _drop_incomplete
        FastTensorRegressionDataLoader._redr_drop_last = True
    model = cne.CNE(**kwargs)
    started = time.perf_counter()
    embedding = model.fit_transform(np.asarray(x, dtype=np.float32), x_pca, graph=graph)
    elapsed = time.perf_counter() - started
    return np.asarray(embedding, dtype=np.float64), float(elapsed)


def fit_pacmap(
    x: np.ndarray,
    x_pca: np.ndarray,
    seed: int,
    label_weight: float,
    n_epochs: int,
    pairs,
) -> tuple[np.ndarray, float]:
    set_seed(seed)
    neighbors, mid_near, further = pairs
    model = ParamPaCMAP(
        n_components=2,
        n_neighbors=10,
        n_MN=5,
        n_FP=20,
        model_dict={
            "backbone": "ANN",
            "layer_size": [100, 100, 100],
            "activation": "relu",
            "n_classes": 2,
        },
        weight_schedule=pacmap_weight_schedule,
        loss_coeffs=[1.0, 1.0, 1.0],
        num_workers=1,
        use_ns_loader=True,
        label_weight=float(label_weight),
        num_epochs=int(n_epochs),
        task_type="pca",
        seed=int(seed),
        verbose=False,
        batch_size=max(32, x.shape[0]),
    )
    model.pair_neighbors = neighbors
    model.pair_MN = mid_near
    model.pair_FP = further
    model._pairs_saved = True
    started = time.perf_counter()
    embedding = model.fit_transform(np.asarray(x, dtype=np.float32), x_pca)
    elapsed = time.perf_counter() - started
    return np.asarray(embedding, dtype=np.float64), float(elapsed)
