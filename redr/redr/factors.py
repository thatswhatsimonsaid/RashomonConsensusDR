"""Product manifolds with a chosen number of observed curves.

The two-curve generator in ``product.py`` is unchanged. This module adds a
third curve, a shared curve added to every block, and the same column
rotation used by automatic discovery. Latent coordinates are stored for
evaluation only.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from redr.product import curve_coordinates


def generate_curves(
    n_samples: int = 160,
    n_curves: int = 2,
    noise: float = 0.05,
    n_nuisance: int = 8,
    nuisance_scale: float = 0.35,
    hide_last: bool = False,
    shared_weight: float = 0.0,
    mixing: str = "aligned",
    random_state: int = 0,
) -> dict:
    """Independent latent coordinates, each observed as its own curve.

    ``shared_weight`` adds one extra curve to every block before noise.
    A positive weight is a shared systematic factor. ``mixing='rotated'``
    applies an orthogonal mix so the original blocks are not columns.
    """
    if n_curves < 2:
        raise ValueError("n_curves must be at least 2")
    if mixing not in ("aligned", "rotated"):
        raise ValueError(mixing)
    rng = np.random.RandomState(random_state)
    latents = [rng.uniform(0.0, 1.0, size=n_samples) for _ in range(n_curves)]
    shared = curve_coordinates(rng.uniform(0.0, 1.0, size=n_samples))
    blocks = []
    for index, latent in enumerate(latents):
        if hide_last and index == n_curves - 1:
            block = rng.normal(0.0, 1.0, size=(n_samples, 3))
        else:
            block = curve_coordinates(latent)
            if shared_weight:
                block = block + float(shared_weight) * shared
            block = _noisy(block, noise, rng)
        blocks.append(block)
    nuisance = nuisance_scale * rng.normal(0.0, 1.0, size=(n_samples, n_nuisance))
    raw = np.hstack(blocks + [nuisance])
    observed = StandardScaler().fit_transform(raw).astype(np.float64)
    visible = observed
    if mixing == "rotated":
        orthogonal, _ = np.linalg.qr(rng.normal(size=(observed.shape[1], observed.shape[1])))
        visible = observed @ orthogonal
    intrinsic = np.column_stack(latents)
    return {
        "X": observed,
        "X_visible": np.asarray(visible, dtype=np.float64),
        "blocks": blocks,
        "intrinsic": intrinsic,
        "u": latents[0],
        "v": latents[1],
        "n": int(n_samples),
        "n_curves": int(n_curves),
        "random_state": int(random_state),
        "mixing": mixing,
        "hide_last": bool(hide_last),
        "shared_weight": float(shared_weight),
    }


def _noisy(block: np.ndarray, noise: float, rng: np.random.RandomState) -> np.ndarray:
    scale = float(np.std(block))
    if scale < 1e-8:
        scale = 1.0
    return block + noise * scale * rng.normal(size=block.shape)
