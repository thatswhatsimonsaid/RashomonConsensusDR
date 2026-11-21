"""Mechanism simulations with synthetic candidate embeddings.

These embeddings are perturbations of a known latent coordinate. They are
not outputs of UMAP, PaCMAP, or InfoNCE. An observable score is a separate
random variable. When that score is a function of the true latent error,
the run is an oracle upper bound and is labeled as such.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from redr.intrinsic import balance_axes, intrinsic_recovery
from redr.metrics import _sample_triplets
from redr.plotting import display_align
from redr.rashomon import gibbs_weights, membership_mask, method_balanced_weights, uniform_weights

EPSILONS = [0.0, 0.05, 0.2, np.inf]


def sample_latent(n: int, geometry: str, rng: np.random.Generator) -> np.ndarray:
    if geometry == "long":
        return np.column_stack([rng.uniform(0.0, 9.4, n), rng.uniform(0.0, 2.0, n)])
    if geometry == "square":
        return np.column_stack([rng.uniform(0.0, 1.0, n), rng.uniform(0.0, 1.0, n)])
    raise ValueError(geometry)


def _heterogeneous_scale(scale: float, rng: np.random.Generator) -> float:
    """Some draws from a method are much more distorted than others."""
    return float(scale) * float(rng.choice([0.25, 0.5, 1.0, 1.0, 2.0, 4.0]))


def perturb(
    latent: np.ndarray,
    method: int,
    structure: str,
    rng: np.random.Generator,
    scale: float,
) -> np.ndarray:
    n = latent.shape[0]
    drawn_scale = _heterogeneous_scale(scale, rng)
    if structure == "shared_noise":
        return latent + rng.normal(0.0, drawn_scale, size=latent.shape)
    if structure == "complementary":
        drawn = latent.copy()
        if method == 0:
            drawn[:, 1] = drawn[:, 1] + rng.normal(0.0, 3.0 * drawn_scale, size=n)
        elif method == 1:
            drawn[:, 0] = drawn[:, 0] + rng.normal(0.0, 3.0 * drawn_scale, size=n)
        else:
            drawn = drawn + rng.normal(0.0, drawn_scale, size=latent.shape)
        return drawn
    if structure == "majority_bias":
        drawn = latent.copy()
        if method < 2:
            drawn[:, 1] = 0.05 * drawn[:, 1]
            drawn[:, 0] = drawn[:, 0] + 0.35 * np.sin(np.pi * latent[:, 1] / 2.0)
            drawn = drawn + rng.normal(0.0, 0.25 * drawn_scale, size=latent.shape)
        else:
            drawn = drawn + rng.normal(0.0, 0.25 * drawn_scale, size=latent.shape)
        return drawn
    raise ValueError(structure)


def observable_scores(
    true_error: np.ndarray,
    alignment: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Lower is treated as better. Oracle copies the latent error."""
    true_error = np.asarray(true_error, dtype=float)
    noise = rng.normal(0.0, 1.0, size=len(true_error))
    centered = true_error - np.mean(true_error)
    scale = float(np.std(centered))
    z = centered / scale if scale > 1e-12 else np.zeros_like(centered)
    if alignment == "oracle":
        return true_error.copy()
    if alignment == "noisy":
        return 0.7 * z + 0.3 * noise
    if alignment == "none":
        return noise
    if alignment == "misleading":
        return -true_error
    raise ValueError(alignment)


def weighted_mean(embeddings: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)
    reference = embeddings[int(np.argmax(weights))]
    total = np.zeros_like(np.asarray(reference, dtype=float))
    for embedding, weight in zip(embeddings, weights):
        total += weight * display_align(embedding, reference)
    return total


def _weights(frame: pd.DataFrame, scheme: str) -> np.ndarray:
    if scheme == "uniform":
        return uniform_weights(frame)
    if scheme == "method_balanced":
        return method_balanced_weights(frame)
    if scheme == "gibbs":
        return gibbs_weights(frame, "score", beta=1.0)
    raise ValueError(scheme)


def run_scenario(
    scenario: dict,
    n_replications: int,
    seed0: int,
) -> pd.DataFrame:
    rows = []
    for replication in range(n_replications):
        rng = np.random.default_rng(seed0 + replication)
        latent = sample_latent(int(scenario["n"]), scenario["geometry"], rng)
        triplets = _sample_triplets(
            latent.shape[0], int(scenario["n_triplets"]), np.random.default_rng(seed0 + 10_000 + replication)
        )
        balanced = balance_axes(latent)
        records = []
        embeddings = []
        for method in range(int(scenario["n_methods"])):
            for _draw in range(int(scenario["per_method"])):
                embedding = perturb(
                    latent,
                    method,
                    scenario["structure"],
                    rng,
                    float(scenario["scale"]),
                )
                if rng.random() < float(scenario["contamination"]):
                    embedding = rng.normal(0.0, 1.5, size=latent.shape)
                records.append(
                    {
                        "method": f"m{method}",
                        "true_error": float(
                            np.sqrt(np.mean((display_align(embedding, balanced) - _unit(balanced)) ** 2))
                        ),
                    }
                )
                embeddings.append(embedding)
        frame = pd.DataFrame(records)
        frame["score"] = observable_scores(
            frame["true_error"].to_numpy(), scenario["alignment"], rng
        )
        correlation = spearmanr(frame["score"], frame["true_error"]).correlation
        for epsilon in EPSILONS:
            for scheme in ("uniform", "method_balanced", "gibbs"):
                rows.append(
                    _evaluate_selection(
                        scenario,
                        replication,
                        correlation,
                        frame,
                        embeddings,
                        latent,
                        triplets,
                        epsilon,
                        scheme,
                        "rashomon",
                    )
                )
            rows.append(
                _evaluate_selection(
                    scenario,
                    replication,
                    correlation,
                    frame,
                    embeddings,
                    latent,
                    triplets,
                    epsilon,
                    "uniform",
                    "pooled",
                )
            )
            if np.isclose(float(epsilon), 0.05) or np.isinf(epsilon):
                rows.append(
                    _matched(
                        scenario,
                        replication,
                        correlation,
                        frame,
                        embeddings,
                        latent,
                        triplets,
                        epsilon,
                        int(scenario["matched_seeds"]),
                        rng,
                    )
                )
    return pd.DataFrame(rows)


def _unit(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0)
    scale = np.sqrt(np.mean(np.sum(centered**2, axis=1))) + 1e-12
    return centered / scale


def _evaluate_selection(
    scenario,
    replication,
    correlation,
    frame,
    embeddings,
    latent,
    triplets,
    epsilon,
    scheme,
    rule,
) -> dict:
    if rule == "pooled":
        keep, _best, _tol = membership_mask(frame["score"].to_numpy(), epsilon)
        chosen = frame.iloc[np.flatnonzero(keep)]
    elif rule == "rashomon":
        chosen = _select(frame, epsilon)
    else:
        raise ValueError(rule)
    if len(chosen) == 0:
        chosen = frame.iloc[[int(frame["score"].idxmin())]]
    aggregate = weighted_mean(
        [embeddings[i] for i in chosen.index], _weights(chosen, scheme)
    )
    scores = intrinsic_recovery(latent, aggregate, triplets, k=10)
    return _row(scenario, replication, correlation, epsilon, scheme, rule, chosen, frame, scores)


def _select(frame: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    pieces = []
    for _method, group in frame.groupby("method", sort=False):
        keep, _best, _tol = membership_mask(group["score"].to_numpy(), epsilon)
        pieces.append(group.iloc[np.flatnonzero(keep)])
    if not pieces:
        return frame.iloc[0:0]
    return pd.concat(pieces)


def _matched(scenario, replication, correlation, frame, embeddings, latent, triplets, epsilon, draws, rng):
    rashomon = _select(frame, epsilon)
    collected: list[dict] = []
    last_chosen = rashomon
    for _draw in range(draws):
        picked = []
        for method, group in frame.groupby("method", sort=False):
            k = int(rashomon["method"].eq(method).sum())
            index = group.index.to_numpy()
            if k <= 0:
                continue
            if k >= len(index):
                take = index
            else:
                take = rng.choice(index, size=k, replace=False)
            picked.extend(np.atleast_1d(take).tolist())
        chosen = frame.loc[picked]
        aggregate = weighted_mean(
            [embeddings[i] for i in chosen.index], _weights(chosen, "uniform")
        )
        collected.append(intrinsic_recovery(latent, aggregate, triplets, k=10))
        last_chosen = chosen
    scores = {
        key: float(np.nanmean([item[key] for item in collected]))
        for key in collected[0]
    }
    row = _row(
        scenario,
        replication,
        correlation,
        epsilon,
        "uniform",
        "matched",
        last_chosen,
        frame,
        scores,
    )
    balanced = [item["intrinsic_triplet_balanced"] for item in collected]
    row["matched_balanced_std"] = (
        float(np.std(balanced, ddof=1)) if len(balanced) > 1 else np.nan
    )
    return row


def _row(scenario, replication, correlation, epsilon, scheme, rule, chosen, frame, scores) -> dict:
    return {
        "scenario": scenario["name"],
        "geometry": scenario["geometry"],
        "structure": scenario["structure"],
        "alignment": scenario["alignment"],
        "oracle_score": scenario["alignment"] == "oracle",
        "contamination": float(scenario["contamination"]),
        "replication": int(replication),
        "score_error_spearman": float(correlation),
        "epsilon": float(epsilon) if np.isfinite(epsilon) else np.inf,
        "scheme": scheme,
        "rule": rule,
        "n_views": int(len(chosen)),
        "retained_fraction": float(len(chosen) / max(len(frame), 1)),
        "mean_true_error": float(chosen["true_error"].mean()) if len(chosen) else np.nan,
        **scores,
    }


def scenarios() -> list[dict]:
    base = {
        "n": 200,
        "n_triplets": 1200,
        "n_methods": 3,
        "per_method": 9,
        "scale": 0.18,
        "matched_seeds": 8,
    }
    specs = [
        ("long_shared_noisy", "long", "shared_noise", "noisy", 0.0),
        ("long_shared_none", "long", "shared_noise", "none", 0.0),
        ("long_shared_misleading", "long", "shared_noise", "misleading", 0.0),
        ("long_shared_oracle", "long", "shared_noise", "oracle", 0.0),
        ("long_complementary_noisy", "long", "complementary", "noisy", 0.0),
        ("long_complementary_none", "long", "complementary", "none", 0.0),
        ("long_complementary_misleading", "long", "complementary", "misleading", 0.0),
        ("long_complementary_oracle", "long", "complementary", "oracle", 0.0),
        ("long_majority_noisy", "long", "majority_bias", "noisy", 0.0),
        ("long_majority_none", "long", "majority_bias", "none", 0.0),
        ("long_majority_misleading", "long", "majority_bias", "misleading", 0.0),
        ("long_shared_noisy_contam", "long", "shared_noise", "noisy", 0.4),
        ("long_shared_none_contam", "long", "shared_noise", "none", 0.4),
        ("long_complementary_noisy_contam", "long", "complementary", "noisy", 0.4),
        ("square_complementary_noisy", "square", "complementary", "noisy", 0.0),
    ]
    built = []
    for name, geometry, structure, alignment, contamination in specs:
        built.append(
            {
                **base,
                "name": name,
                "geometry": geometry,
                "structure": structure,
                "alignment": alignment,
                "contamination": contamination,
            }
        )
    return built


def summarize_controlled(detail: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "scenario",
        "geometry",
        "structure",
        "alignment",
        "oracle_score",
        "contamination",
        "epsilon",
        "scheme",
        "rule",
    ]
    metric = "intrinsic_triplet_balanced"
    rows = []
    for values, group in detail.groupby(keys, dropna=False):
        payload = dict(zip(keys, values))
        series = group[metric].to_numpy(dtype=float)
        payload.update(
            {
                "n": int(np.isfinite(series).sum()),
                "mean": float(np.nanmean(series)),
                "std": float(np.nanstd(series, ddof=1)) if np.isfinite(series).sum() > 1 else np.nan,
                "mean_true_error": float(group["mean_true_error"].mean()),
                "mean_spearman": float(group["score_error_spearman"].mean()),
                "mean_views": float(group["n_views"].mean()),
            }
        )
        rows.append(payload)
    return pd.DataFrame(rows)


def plot_controlled(summary: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    uniform = summary.loc[summary["scheme"].eq("uniform")].copy()
    filtered = uniform.loc[
        uniform["rule"].eq("rashomon") & np.isclose(uniform["epsilon"].astype(float), 0.05)
    ]
    matched = uniform.loc[
        uniform["rule"].eq("matched") & np.isclose(uniform["epsilon"].astype(float), 0.05)
    ]
    full = uniform.loc[uniform["rule"].eq("rashomon") & ~np.isfinite(uniform["epsilon"].astype(float))]
    names = list(dict.fromkeys(filtered["scenario"]))
    x = np.arange(len(names))

    def column(frame: pd.DataFrame) -> tuple[list[float], list[float]]:
        lookup = frame.set_index("scenario")
        means = [float(lookup["mean"].get(name, np.nan)) for name in names]
        stds = [float(lookup["std"].get(name, np.nan)) for name in names]
        return means, stds

    fig, axis = plt.subplots(figsize=(11, 5.4))
    width = 0.26
    filtered_mean, filtered_std = column(filtered)
    full_mean, full_std = column(full)
    matched_mean, matched_std = column(matched)
    axis.bar(
        x - width,
        filtered_mean,
        width=width,
        yerr=filtered_std,
        capsize=2,
        label="method-specific filter, epsilon 0.05",
        color="#045a8d",
        error_kw={"elinewidth": 0.6},
    )
    axis.bar(
        x,
        full_mean,
        width=width,
        yerr=full_std,
        capsize=2,
        label="all candidates",
        color="#2b8cbe",
        error_kw={"elinewidth": 0.6},
    )
    axis.bar(
        x + width,
        matched_mean,
        width=width,
        yerr=matched_std,
        capsize=2,
        label="size-matched draw",
        color="#a6bddb",
        error_kw={"elinewidth": 0.6},
    )
    axis.set_xticks(x)
    axis.set_xticklabels(names, rotation=55, ha="right", fontsize=8)
    axis.set_ylabel("balanced intrinsic triplet accuracy")
    axis.set_title("Coordinate average of synthetic embeddings")
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
