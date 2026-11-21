"""Empirical Rashomon membership and source weights for CoMDS.

Membership is a loss threshold on the candidates that were trained. It is
not an enumeration of every embedding inside the continuous Rashomon set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


TIE_ATOL = 1e-8


def membership_mask(losses: np.ndarray, epsilon: float, tie_atol: float = TIE_ATOL):
    """Return keep mask, best finite loss, and the absolute tolerance that was applied.

    epsilon = 0 keeps numerical ties with the minimum. epsilon = inf keeps every
    finite loss. Any other epsilon uses the Checkpoint 1 relative band.
    """
    losses = np.asarray(losses, dtype=float)
    finite = np.isfinite(losses)
    keep = np.zeros(len(losses), dtype=bool)
    if not finite.any():
        return keep, float("nan"), float("nan")
    best = float(np.min(losses[finite]))
    if np.isinf(epsilon):
        keep[finite] = True
        tolerance = float("inf")
    elif float(epsilon) == 0.0:
        keep[finite & (losses <= best + tie_atol)] = True
        tolerance = float(tie_atol)
    else:
        tolerance = float(epsilon) * max(abs(best), 1e-8)
        keep[finite & (losses <= best + tolerance)] = True
    return keep, best, tolerance


def mark_by_loss(
    frame: pd.DataFrame,
    column: str,
    epsilon: float,
    group_cols: tuple[str, ...] = ("method",),
    tie_atol: float = TIE_ATOL,
) -> pd.DataFrame:
    """Mark membership separately inside each group of `group_cols`.

    Groups are objective families. Candidates that do not share a native
    objective should not be placed in the same group.
    """
    out = frame.copy()
    out["in_rashomon"] = False
    out["loss_reference"] = np.nan
    out["loss_tolerance"] = np.nan
    out["membership_loss"] = out[column]
    if len(out) == 0:
        return out
    grouped = out.groupby(list(group_cols), dropna=False, sort=False)
    for _key, group in grouped:
        keep, best, tolerance = membership_mask(
            group[column].to_numpy(dtype=float), epsilon, tie_atol=tie_atol
        )
        out.loc[group.index, "loss_reference"] = best
        out.loc[group.index, "loss_tolerance"] = tolerance
        out.loc[group.index[keep], "in_rashomon"] = True
    return out


def mark_rashomon(frame: pd.DataFrame, delta: float) -> pd.DataFrame:
    """Keep a candidate when its probe loss is within a relative band of the best."""
    return mark_by_loss(frame, "probe_loss", delta)


def _positive_weights(values: np.ndarray) -> np.ndarray:
    weights = np.asarray(values, dtype=float)
    weights = np.clip(weights, 1e-6, None)
    return weights


def uniform_weights(members: pd.DataFrame) -> np.ndarray:
    return np.ones(len(members), dtype=float)


def method_balanced_weights(members: pd.DataFrame) -> np.ndarray:
    counts = members.groupby("method")["method"].transform("size").to_numpy(dtype=float)
    n_methods = members["method"].nunique()
    return _positive_weights(1.0 / (counts * n_methods))


def gibbs_weights(
    members: pd.DataFrame,
    loss_column: str = "probe_loss",
    beta: float = 1.0,
) -> np.ndarray:
    """Within-method Gibbs tilt, then equal method mass.

    Excess loss is divided by the within-method standard deviation, so beta
    is in units of that standard deviation. beta = 0 is uniform inside each
    method. beta = 1 reproduces the Checkpoint 1 weights. This is a practical
    tilt, not a calibrated PAC-Bayes posterior.
    """
    weights = np.zeros(len(members), dtype=float)
    positions = {idx: i for i, idx in enumerate(members.index)}
    n_methods = max(int(members["method"].nunique()), 1)
    for _method, group in members.groupby("method", sort=False):
        losses = group[loss_column].to_numpy(dtype=float)
        center = losses - np.min(losses)
        scale = float(np.std(losses))
        if scale <= 1e-8 or float(beta) == 0.0:
            raw = np.ones(len(group), dtype=float)
        else:
            raw = np.exp(-float(beta) * center / scale)
        raw = raw / np.sum(raw)
        for idx, value in zip(group.index, raw):
            weights[positions[idx]] = value / n_methods
    return _positive_weights(weights)


def vote_weights(members: pd.DataFrame, embeddings: list[np.ndarray], k: int = 10) -> np.ndarray:
    """Upweight views whose neighbors agree with the majority Rashomon vote."""
    if len(embeddings) == 0:
        return np.ones(0, dtype=float)
    n = embeddings[0].shape[0]
    votes = np.zeros((n, n), dtype=float)
    neighborhoods = []
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    for embedding in embeddings:
        nn.fit(embedding)
        idx = nn.kneighbors(return_distance=False)[:, 1:]
        neighborhoods.append(idx)
        for i in range(n):
            votes[i, idx[i]] += 1.0
    majority = np.argsort(-votes, axis=1)[:, :k]
    scores = []
    for idx in neighborhoods:
        overlaps = [
            len(set(idx[i]).intersection(majority[i])) / k for i in range(n)
        ]
        scores.append(float(np.mean(overlaps)))
    return _positive_weights(np.asarray(scores, dtype=float))


def edge_weight_matrix(embeddings: list[np.ndarray], k: int = 10) -> np.ndarray:
    """Pair weight = fraction of Rashomon views in which the pair is a kNN edge."""
    n = embeddings[0].shape[0]
    votes = np.zeros((n, n), dtype=float)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    for embedding in embeddings:
        nn.fit(embedding)
        idx = nn.kneighbors(return_distance=False)[:, 1:]
        for i in range(n):
            votes[i, idx[i]] += 1.0
    frequency = votes / max(len(embeddings), 1)
    sym = 0.5 * (frequency + frequency.T)
    np.fill_diagonal(sym, 0.0)
    return sym


def mean_vote_entropy(embeddings: list[np.ndarray], k: int = 10) -> float:
    n = embeddings[0].shape[0]
    votes = np.zeros((n, n), dtype=float)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    for embedding in embeddings:
        nn.fit(embedding)
        idx = nn.kneighbors(return_distance=False)[:, 1:]
        for i in range(n):
            votes[i, idx[i]] += 1.0
    entropies = []
    for i in range(n):
        positive = votes[i][votes[i] > 0]
        prob = positive / positive.sum()
        entropies.append(float(-(prob * np.log(prob)).sum()))
    return float(np.mean(entropies))
