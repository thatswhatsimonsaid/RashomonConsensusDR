"""Regime labels for a pool of trained embeddings.

The label uses native loss and trustworthiness only. Latent coordinates
and class labels are not arguments. Thresholds come from the study config
and are not fit to a dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(path: Path | None = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "experiments" / "study11_natural_diversity" / "config.yaml"
    with path.open() as handle:
        return yaml.safe_load(handle)


def primary_pool(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """One objective per method: the configured k and lambda."""
    chosen = frame.loc[
        np.isclose(frame["k"].astype(float), float(config["primary_k"]))
        & np.isclose(frame["lambda"].astype(float), float(config["primary_lambda"]))
    ].copy()
    return chosen


def relative_span(losses: np.ndarray) -> float:
    finite = np.asarray(losses, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return float("nan")
    best = float(np.min(finite))
    return float((np.max(finite) - best) / max(abs(best), 1e-8))


def classify_regime(methods: pd.DataFrame, config: dict) -> dict:
    """Label one dataset from one row per method.

    Required columns: method, span, mean_trust, within_disagreement,
    trust_tracks_loss. `between_disagreement` is repeated on every row.
    `trust_tracks_loss` is true when the highest native loss in that method
    has strictly lower trustworthiness than the lowest native loss.
    """
    span_small = float(config["span_small"])
    trust_gap = float(config["trust_gap"])
    ratio = float(config["disagreement_ratio"])
    best_trust = float(methods["mean_trust"].max())
    within = float(methods["within_disagreement"].mean())
    between = float(methods["between_disagreement"].iloc[0])
    bad = methods.loc[
        (methods["mean_trust"] <= best_trust - trust_gap) & (methods["span"] <= span_small)
    ]
    wide = methods.loc[methods["span"] > span_small]
    contaminated = len(wide) > 0 and bool(wide["trust_tracks_loss"].all())
    loss_varies = len(wide) > 0 and not contaminated
    trusts_close = bool((best_trust - methods["mean_trust"] <= trust_gap).all())
    spans_small = bool((methods["span"] <= span_small).all())
    separated = between > max(ratio * within, within + 0.05)
    exchangeable = spans_small and between <= within * 1.25 + 0.02
    complementary = trusts_close and spans_small and separated
    diverse_similar = spans_small and not exchangeable and not complementary
    flags = {
        "systematically_bad": bool(len(bad) > 0),
        "contaminated_within_method": bool(contaminated),
        "loss_varies_unstructured": bool(loss_varies),
        "complementary_candidate": bool(complementary),
        "exchangeable": bool(exchangeable),
        "diverse_similar_loss": bool(diverse_similar),
        "bad_methods": ",".join(bad["method"].astype(str)) if len(bad) else "",
        "within_disagreement": within,
        "between_disagreement": between,
        "max_span": float(methods["span"].max()),
        "trust_range": float(methods["mean_trust"].max() - methods["mean_trust"].min()),
    }
    order = (
        "systematically_bad",
        "contaminated_within_method",
        "loss_varies_unstructured",
        "complementary_candidate",
        "exchangeable",
        "diverse_similar_loss",
    )
    flags["primary"] = next((name for name in order if flags[name]), "unclassified")
    return flags
