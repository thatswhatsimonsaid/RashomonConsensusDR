"""Product manifolds with two intrinsic coordinates that can carry information.

Each coordinate is observed through its own nonlinear curve, plus optional
nuisance features. The latent coordinates are not passed to any embedding
method. Setting ``hide_second`` replaces the second curve with noise, so the
second coordinate is not in the observations.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


def curve_coordinates(t: np.ndarray) -> np.ndarray:
    """Map a coordinate in [0, 1] to a 3-d curve. The angle travels a full turn."""
    t = np.asarray(t, dtype=float).reshape(-1)
    angle = 2.0 * np.pi * t
    return np.column_stack([np.cos(angle), np.sin(angle), 2.0 * t])


def generate_product(
    n_samples: int = 200,
    aspect: tuple[float, float] = (1.0, 1.0),
    noise: float = 0.05,
    n_nuisance: int = 8,
    nuisance_scale: float = 0.35,
    hide_second: bool = False,
    preserve_scale: bool = False,
    random_state: int = 0,
) -> dict:
    """Independent coordinates u and v, each observed as its own curve.

    ``aspect`` scales the latent axes. A square surface uses (1, 1). A long
    surface uses a larger first entry. Evaluation should still report each
    axis separately.

    ``preserve_scale`` multiplies each observed curve by its aspect weight
    and only centers the concatenated matrix. Per-feature standardization
    would remove that weight, so a long-versus-square comparison that still
    standardizes each column is not an aspect comparison.
    """
    rng = np.random.RandomState(random_state)
    u = rng.uniform(0.0, 1.0, size=n_samples)
    v = rng.uniform(0.0, 1.0, size=n_samples)
    block_u = _noisy(curve_coordinates(u), noise, rng)
    if hide_second:
        block_v = rng.normal(0.0, 1.0, size=(n_samples, 3))
    else:
        block_v = _noisy(curve_coordinates(v), noise, rng)
    if preserve_scale:
        block_u = float(aspect[0]) * block_u
        block_v = float(aspect[1]) * block_v
    nuisance = nuisance_scale * rng.normal(0.0, 1.0, size=(n_samples, n_nuisance))
    raw = np.hstack([block_u, block_v, nuisance])
    if preserve_scale:
        observed = (raw - raw.mean(axis=0)).astype(np.float64)
    else:
        observed = StandardScaler().fit_transform(raw).astype(np.float64)
    intrinsic = np.column_stack([aspect[0] * u, aspect[1] * v])
    return {
        "X": observed,
        "block_u": block_u,
        "block_v": block_v,
        "intrinsic": intrinsic,
        "u": u,
        "v": v,
        "aspect": tuple(aspect),
        "hide_second": bool(hide_second),
        "preserve_scale": bool(preserve_scale),
        "nuisance": nuisance,
        "n": int(n_samples),
        "random_state": int(random_state),
        "observation_ids": np.arange(n_samples, dtype=np.int64),
    }


def _noisy(block: np.ndarray, noise: float, rng: np.random.RandomState) -> np.ndarray:
    scale = float(np.std(block))
    if scale < 1e-8:
        scale = 1.0
    return block + noise * scale * rng.normal(size=block.shape)
