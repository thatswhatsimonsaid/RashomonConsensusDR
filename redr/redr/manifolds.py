"""Known latent manifolds with an explicit unfolded intrinsic coordinate.

The observed coordinates include noise. The intrinsic coordinates are the
clean unfolding and are never passed to a dimension-reduction trainer.
S-curve and Swiss-roll sampling follow scikit-learn's generators, including
the RandomState draw order, so the noisy point cloud matches
``make_s_curve`` and ``make_swiss_roll`` at the same seed.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def swiss_roll_arc_length(t: np.ndarray) -> np.ndarray:
    """Arc length of (t cos t, t sin t). Speed is sqrt(1 + t^2)."""
    t = np.asarray(t, dtype=float)
    return 0.5 * (t * np.sqrt(1.0 + t * t) + np.arcsinh(t))


def s_curve_speed(t: np.ndarray) -> np.ndarray:
    """Analytic speed of the S-curve in the xz plane. It is identically 1."""
    t = np.asarray(t, dtype=float)
    return np.ones_like(t, dtype=float)


def generate_s_curve(
    n_samples: int = 400,
    noise: float = 0.05,
    random_state: int = 0,
) -> dict:
    """S-curve. The generating parameter already has unit speed."""
    generator = np.random.RandomState(random_state)
    t = np.squeeze(3 * np.pi * (generator.uniform(size=(1, n_samples)) - 0.5))
    height = 2.0 * generator.uniform(size=n_samples)
    clean = np.column_stack(
        [np.sin(t), height, np.sign(t) * (np.cos(t) - 1.0)]
    )
    noisy = clean + noise * generator.standard_normal(size=(3, n_samples)).T
    arc = np.asarray(t, dtype=float)
    return _pack("s_curve", noisy, clean, t, arc, height, noise, random_state)


def generate_swiss_roll(
    n_samples: int = 400,
    noise: float = 0.05,
    random_state: int = 0,
) -> dict:
    """Swiss roll. Arc length is the integral of sqrt(1 + t^2), not t itself."""
    generator = np.random.RandomState(random_state)
    t = 1.5 * np.pi * (1 + 2 * generator.uniform(size=n_samples))
    height = 21.0 * generator.uniform(size=n_samples)
    clean = np.column_stack([t * np.cos(t), height, t * np.sin(t)])
    noisy = clean + noise * generator.standard_normal(size=(3, n_samples)).T
    arc = swiss_roll_arc_length(t)
    return _pack("swiss", noisy, clean, t, arc, height, noise, random_state)


def generate_manifold(name: str, n_samples: int, noise: float, random_state: int) -> dict:
    name = name.lower()
    if name == "s_curve":
        return generate_s_curve(n_samples, noise, random_state)
    if name in {"swiss", "swiss_roll"}:
        return generate_swiss_roll(n_samples, noise, random_state)
    raise ValueError(f"Unknown manifold {name}")


def _pack(name, noisy, clean, parameter, arc, height, noise, random_state) -> dict:
    observed, observed_pca = standardize_for_training(noisy)
    intrinsic = np.column_stack([np.asarray(arc, dtype=float), np.asarray(height, dtype=float)])
    return {
        "name": name,
        "X_raw": np.asarray(noisy, dtype=np.float64),
        "X_clean": np.asarray(clean, dtype=np.float64),
        "X": observed,
        "X_pca": observed_pca,
        "parameter": np.asarray(parameter, dtype=np.float64),
        "arc_length": np.asarray(arc, dtype=np.float64),
        "height": np.asarray(height, dtype=np.float64),
        "intrinsic": intrinsic,
        "labels": None,
        "discrete_labels": False,
        "observation_ids": np.arange(observed.shape[0], dtype=np.int64),
        "n": int(observed.shape[0]),
        "p": int(observed.shape[1]),
        "noise": float(noise),
        "random_state": int(random_state),
    }


def standardize_for_training(x_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match Checkpoint 2: standardize observations, then a 2-d PCA target."""
    observed = StandardScaler().fit_transform(np.asarray(x_raw, dtype=np.float64))
    observed_pca = PCA(n_components=2, random_state=0).fit_transform(observed)
    return observed.astype(np.float32), observed_pca.astype(np.float32)
