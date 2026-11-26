"""Quality × diversity pools for a CoMDS mechanism study.

Each candidate is a perturbation of a known square. Quality chooses how hard
the draws are distorted. Diversity chooses how the three methods differ.
The selection score is a separate random variable. Latent coordinates are
used to build the perturbations and to score finished embeddings. They are
not an input to membership except in rows labeled oracle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from redr.controlled import _unit, weighted_mean
from redr.diversity import pairwise_distance_disagreement
from redr.intrinsic import intrinsic_recovery
from redr.metrics import _sample_triplets
from redr.plotting import display_align
from redr.rashomon import gibbs_weights, mark_by_loss, method_balanced_weights


def load_config(path: Path | None = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "experiments" / "study10_quality_diversity" / "config.yaml"
    raw = yaml.safe_load(Path(path).read_text())
    raw["epsilons"] = [float(value) for value in raw["epsilons"]]
    return raw


def cells(config: dict) -> list[dict]:
    """Primary factorial first, then the score slice that is not already included."""
    found = []
    seen = set()
    for quality in config["qualities"]:
        for diversity in config["diversities"]:
            key = (quality, diversity, config["primary_score"])
            seen.add(key)
            found.append({"quality": quality, "diversity": diversity, "score_kind": config["primary_score"]})
    slice_ = config["score_slice"]
    for diversity in slice_["diversities"]:
        for score_kind in slice_["scores"]:
            key = (slice_["quality"], diversity, score_kind)
            if key in seen:
                continue
            seen.add(key)
            found.append({"quality": slice_["quality"], "diversity": diversity, "score_kind": score_kind})
    return found


def _template(latent: np.ndarray, method: int, diversity: str) -> np.ndarray:
    if diversity == "identical":
        return latent.copy()
    if diversity == "moderate":
        warped = latent.copy()
        warped[:, 0] = latent[:, 0] + 0.30 * np.sin((method + 1) * np.pi * latent[:, 1])
        warped[:, 1] = latent[:, 1] + 0.30 * np.cos((method + 1) * np.pi * latent[:, 0])
        return warped
    if diversity == "complementary":
        drawn = latent.copy()
        if method == 0:
            drawn[:, 1] *= 0.15
        elif method == 1:
            drawn[:, 0] *= 0.15
        return drawn
    if diversity == "harmful":
        if method == 2:
            shift = max(latent.shape[0] // 3, 1)
            return np.roll(latent, shift, axis=0)
        return latent.copy()
    raise ValueError(diversity)


def _noise_schedule(quality: str) -> tuple:
    if quality == "high":
        return (0.05, 0.05, 0.05, 0.05)
    if quality == "mostly_high":
        return (0.05, 0.05, 0.05, 0.45)
    if quality == "mixed":
        return (0.05, 0.05, 0.50, 0.50)
    if quality == "severe":
        return (0.05, 0.05, "severe", "severe")
    if quality == "shared":
        return (0.05, 0.05, 0.05, 0.05)
    raise ValueError(quality)


def _draw(latent: np.ndarray, method: int, diversity: str, quality: str, draw: int, rng: np.random.Generator) -> np.ndarray:
    base = _template(latent, method, diversity)
    if quality == "shared":
        base = base.copy()
        base[:, 1] *= 0.25
    spec = _noise_schedule(quality)[draw]
    noisy = base + rng.normal(0.0, 0.05 if spec == "severe" else float(spec), size=base.shape)
    if spec == "severe":
        mask = rng.random(latent.shape[0]) < 0.50
        noisy[mask] = rng.normal(0.0, 1.0, size=(int(mask.sum()), latent.shape[1]))
    return noisy


def selection_scores(true_error: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    """Lower is better. Oracle copies the latent error and is not a primary score."""
    error = np.asarray(true_error, dtype=float)
    if kind == "oracle":
        return error.copy()
    if kind == "informative":
        return np.clip(error + rng.normal(0.0, 0.02, size=len(error)), 1e-4, None)
    if kind == "weak":
        return np.clip(error + rng.normal(0.0, 0.20, size=len(error)), 1e-4, None)
    if kind == "none":
        return rng.uniform(0.2, 1.2, size=len(error))
    if kind == "misleading":
        return np.clip(float(np.max(error)) - error + 0.01, 1e-4, None)
    raise ValueError(kind)


def build_pool(config: dict, cell: dict, replication: int) -> dict:
    quality_index = config["qualities"].index(cell["quality"]) if cell["quality"] in config["qualities"] else 0
    diversity_index = config["diversities"].index(cell["diversity"])
    score_index = ["informative", "weak", "none", "misleading", "oracle"].index(cell["score_kind"])
    rng = np.random.default_rng(
        int(config["seed0"]) + replication * 1000 + quality_index * 40 + diversity_index * 8 + score_index
    )
    latent = np.column_stack(
        [
            rng.uniform(0.0, 1.0, int(config["n"])),
            rng.uniform(0.0, 1.0, int(config["n"])),
        ]
    )
    triplet_rng = np.random.default_rng(int(config["seed0"]) + 500_000 + replication)
    triplets = _sample_triplets(latent.shape[0], int(config["n_triplets"]), triplet_rng)
    balanced = latent - latent.mean(axis=0)
    records = []
    embeddings: dict[str, np.ndarray] = {}
    for method in range(int(config["n_methods"])):
        for draw in range(int(config["per_method"])):
            embedding = _draw(latent, method, cell["diversity"], cell["quality"], draw, rng)
            embedding_id = f"m{method}_d{draw}"
            aligned = display_align(embedding, balanced)
            global_error = float(np.sqrt(np.mean((aligned - _unit(balanced)) ** 2)))
            records.append(
                {
                    "embedding_id": embedding_id,
                    "method": f"m{method}",
                    "draw": draw,
                    "true_global_error": global_error,
                }
            )
            embeddings[embedding_id] = np.asarray(embedding, dtype=np.float64)
    frame = pd.DataFrame(records)
    frame["score"] = selection_scores(frame["true_global_error"].to_numpy(), cell["score_kind"], rng)
    return {"latent": latent, "triplets": triplets, "frame": frame, "embeddings": embeddings}


def retained(frame: pd.DataFrame, epsilon: float, column: str = "score") -> pd.DataFrame:
    marked = mark_by_loss(frame, column, epsilon, group_cols=("method",))
    return marked.loc[marked["in_rashomon"]].copy()


def trimmed(frame: pd.DataFrame, fraction: float) -> pd.DataFrame:
    ordered = frame.sort_values(["score", "embedding_id"], kind="mergesort")
    drop = int(np.floor(fraction * len(ordered)))
    keep = max(len(ordered) - drop, 1)
    return ordered.iloc[:keep].copy()


def size_matched(frame: pd.DataFrame, count: int, rng: np.random.Generator) -> pd.DataFrame:
    count = int(min(max(count, 1), len(frame)))
    chosen = rng.choice(len(frame), size=count, replace=False)
    return frame.iloc[np.sort(chosen)].copy()


def mean_disagreement(members: pd.DataFrame, embeddings: dict[str, np.ndarray]) -> float:
    if len(members) < 2:
        return 0.0
    views = [embeddings[str(value)] for value in members["embedding_id"]]
    matrix = pairwise_distance_disagreement(views)
    upper = matrix[np.triu_indices(len(views), k=1)]
    return float(np.mean(upper))


def score_consensus(latent: np.ndarray, embedding: np.ndarray, triplets: np.ndarray) -> dict:
    recovery = intrinsic_recovery(latent, embedding, triplets, k=8)
    return {
        "intrinsic_triplet_balanced": recovery["intrinsic_triplet_balanced"],
        "intrinsic_triplet_isometric": recovery["intrinsic_triplet_isometric"],
        "arc_triplet_accuracy": recovery["arc_triplet_accuracy"],
        "height_triplet_accuracy": recovery["height_triplet_accuracy"],
        "intrinsic_rank_correlation": recovery["intrinsic_rank_correlation"],
    }


def procrustes_consensus(members: pd.DataFrame, embeddings: dict[str, np.ndarray], weights: np.ndarray | None = None) -> np.ndarray:
    views = [embeddings[str(value)] for value in members["embedding_id"]]
    if len(views) == 1:
        return views[0]
    if weights is None:
        weights = np.ones(len(views), dtype=float)
    return weighted_mean(views, weights)


def scheme_weights(members: pd.DataFrame, scheme: str, beta: float) -> np.ndarray | None:
    if scheme == "uniform":
        return None
    if scheme == "balanced":
        return method_balanced_weights(members)
    if scheme == "gibbs":
        return gibbs_weights(members, "score", beta=beta)
    raise ValueError(scheme)
