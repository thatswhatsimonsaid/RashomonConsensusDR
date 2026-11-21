"""Rashomon rules that do not read latent coordinates.

Method-specific membership uses a loss that is only compared inside one
objective group. Pooled membership uses one structural loss defined on the
full observed matrix. Those two losses are not treated as the same number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from redr.diversity import pairwise_neighborhood_overlap
from redr.rashomon import mark_by_loss, membership_mask


def method_specific(frame: pd.DataFrame, epsilon: float, group_col: str = "objective_group") -> pd.DataFrame:
    """Keep near-best members inside each objective group."""
    marked = mark_by_loss(frame, "reference_loss", epsilon, group_cols=(group_col,))
    return marked.loc[marked["in_rashomon"]].copy()


def pooled(frame: pd.DataFrame, epsilon: float, loss_col: str = "structural_loss") -> pd.DataFrame:
    """One threshold for every embedding, on a common structural loss."""
    keep, _best, _tol = membership_mask(frame[loss_col].to_numpy(dtype=float), epsilon)
    return frame.iloc[np.flatnonzero(keep)].copy()


def hybrid(
    frame: pd.DataFrame,
    epsilon: float,
    group_col: str = "objective_group",
    loss_col: str = "structural_loss",
) -> pd.DataFrame:
    """Method-specific survivors that also pass a structural band.

    The structural band is computed inside the method-specific set, so a
    specialist is not required to match the best joint embedding's score
    before the set is formed. It can still be dropped if it is far from the
    best structural score among survivors.
    """
    specific = method_specific(frame, epsilon, group_col=group_col)
    if len(specific) == 0:
        return specific
    keep, _best, _tol = membership_mask(specific[loss_col].to_numpy(dtype=float), epsilon)
    return specific.iloc[np.flatnonzero(keep)].copy()


def diversity_aware(
    frame: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    epsilon: float,
    k: int = 10,
    keep_max: int = 6,
    loss_col: str = "structural_loss",
) -> pd.DataFrame:
    """Greedy neighborhood diversity inside the pooled near-optimal set.

    The first member is the lowest structural loss. Later members minimize
    the maximum neighborhood overlap with embeddings already chosen.
    """
    pool = pooled(frame, epsilon, loss_col=loss_col)
    if len(pool) <= 1:
        return pool
    order = pool.sort_values(loss_col, kind="mergesort")
    chosen_ids = [str(order.iloc[0]["embedding_id"])]
    remaining = [str(value) for value in order["embedding_id"].tolist()[1:]]
    views = {key: embeddings[key] for key in [str(v) for v in pool["embedding_id"]]}
    while remaining and len(chosen_ids) < keep_max:
        overlap = pairwise_neighborhood_overlap(
            [views[key] for key in chosen_ids + remaining], k=k
        )
        n_chosen = len(chosen_ids)
        best_pos = None
        best_value = np.inf
        for offset, _embedding_id in enumerate(remaining):
            row = overlap[n_chosen + offset, :n_chosen]
            value = float(np.max(row)) if len(row) else 0.0
            if value < best_value:
                best_value = value
                best_pos = offset
        chosen_ids.append(remaining.pop(best_pos))
    return pool.loc[pool["embedding_id"].astype(str).isin(chosen_ids)].copy()
