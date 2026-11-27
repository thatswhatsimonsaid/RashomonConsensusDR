"""Membership and diversity summaries for one objective configuration at a time.

The loss scale is s = max(|best loss|, 1e-8). A candidate stays in its
configuration's Rashomon set when its loss is at most best + epsilon * s.
Losses from different configurations are never compared.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(path: Path | None = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "experiments" / "study12_cross_objective" / "config.yaml"
    with path.open() as handle:
        return yaml.safe_load(handle)


def config_id(method: str, spec: dict | None) -> str:
    parts = [method]
    for key in ("k", "n_neighbors", "perplexity", "spectrum"):
        if spec and spec.get(key) is not None:
            value = spec[key]
            text = f"{float(value):g}" if key == "spectrum" else str(int(value))
            parts.append(f"{key}={text}")
    return "|".join(parts)


def is_default(spec: dict | None, default: dict | None) -> bool:
    spec = spec or {}
    default = default or {}
    keys = set(spec) | set(default)
    for key in keys:
        if spec.get(key) != default.get(key):
            return False
    return True


def iter_jobs(config: dict, train_seeds: list[int]) -> list[dict]:
    jobs = []
    for method in config["methods"]:
        for spec in method["grid"]:
            spec = spec or {}
            seeds = [0] if method["kind"] in {"pca", "isomap"} else list(train_seeds)
            for seed in seeds:
                jobs.append(
                    {
                        "method": method["name"],
                        "kind": method["kind"],
                        "loss_mode": method.get("loss_mode"),
                        "early_exaggeration": bool(method.get("early_exaggeration", False)),
                        "k": spec.get("k"),
                        "n_neighbors": spec.get("n_neighbors"),
                        "perplexity": spec.get("perplexity"),
                        "spectrum": spec.get("spectrum"),
                        "seed": int(seed),
                        "config_id": config_id(method["name"], spec),
                        "is_default": is_default(spec, method.get("default")),
                    }
                )
    return jobs


def loss_scale(best: float) -> float:
    return max(abs(float(best)), 1e-8)


def neg_spec_param(n: int, spectrum: float, negative_samples: int = 5) -> float:
    """Vendor map from the negative-t-SNE spectrum s to ContrastiveLoss spec_param.

    s = 0 uses the t-SNE normalization and s = 1 uses the UMAP normalization.
    """
    n = int(n)
    z_umap = n**2 / 5.0
    z_tsne = 100.0 * n
    z_bar = z_tsne * np.exp(float(spectrum) * (np.log(z_umap) - np.log(z_tsne)))
    return float(negative_samples * z_bar / n**2)


def apply_membership(frame: pd.DataFrame, epsilon: float) -> tuple[pd.DataFrame, bool, float]:
    """Mark each configuration's own band. Returns the frame, whether every candidate is kept, and the widest relative span."""
    out = frame.copy()
    out["in_rashomon"] = False
    out["loss_scale"] = np.nan
    out["best_native"] = np.nan
    out["relative_span"] = np.nan
    spans = []
    for _config, index in out.groupby("config_id", sort=False).groups.items():
        losses = out.loc[index, "native_loss"].to_numpy(dtype=float)
        finite = np.isfinite(losses)
        if not finite.any():
            out.loc[index, "in_rashomon"] = True
            out.loc[index, "relative_span"] = 0.0
            spans.append(0.0)
            continue
        best = float(np.min(losses[finite]))
        scale = loss_scale(best)
        limit = best + float(epsilon) * scale
        keep = finite & (losses <= limit + 1e-8)
        out.loc[index, "in_rashomon"] = keep
        out.loc[index, "loss_scale"] = scale
        out.loc[index, "best_native"] = best
        span = float((np.max(losses[finite]) - best) / scale)
        out.loc[index, "relative_span"] = span
        spans.append(span)
    inactive = bool(out["in_rashomon"].all())
    return out, inactive, float(np.max(spans) if spans else 0.0)


def pair_relation(method_a: str, config_a: str, method_b: str, config_b: str) -> str:
    if config_a == config_b:
        return "within_config"
    if method_a == method_b:
        return "between_config"
    return "between_algorithm"


def quality_weights(losses: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Full-pool Gibbs tilt on one common loss. Not split by method."""
    losses = np.asarray(losses, dtype=float)
    center = losses - np.min(losses)
    scale = float(np.std(losses))
    if scale <= 1e-8:
        raw = np.ones(len(losses), dtype=float)
    else:
        raw = np.exp(-float(beta) * center / scale)
    return raw / np.sum(raw)


def weights_match(left: np.ndarray, right: np.ndarray, tolerance: float = 1e-3) -> bool:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left = left / left.sum()
    right = right / right.sum()
    return bool(np.max(np.abs(left - right)) < tolerance)
