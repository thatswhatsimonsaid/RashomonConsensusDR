"""Checkpoint 2 candidate pools and epsilon sweeps.

The same trained pool is reused for every tolerance. Checkpoint 1 result
directories are not written.
"""

from __future__ import annotations

import hashlib
import json
import platform
import resource
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import pairwise_distances

from redr.comds_bridge import fit_consensus
from redr.data import load_dataset
from redr.diversity import (
    pairwise_neighborhood_overlap,
    redundancy_count,
    within_between_overlap,
)
from redr.entropy import confidence_matrices, distance_confidence, vote_probability
from redr.loss_validation import contrastive_reference_loss, pacmap_component_losses
from redr.metrics import _sample_triplets, evaluate_embedding, latent_triplet_accuracy
from redr.plotting import display_align
from redr.probe import contrastive_probe, pacmap_probe
from redr.rashomon import (
    gibbs_weights,
    mark_by_loss,
    method_balanced_weights,
    uniform_weights,
    vote_weights,
)
from redr.train import fit_cne, fit_pacmap, high_d_neighbors, pacmap_pairs

EPSILONS = [0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, np.inf]
ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    with Path(path).open() as handle:
        config = yaml.safe_load(handle)
    config["out"] = ROOT / "results" / "checkpoint2" / config["name"]
    return config


def _eps_label(epsilon: float) -> str:
    if np.isinf(epsilon):
        return "inf"
    return f"{epsilon:g}"


def _candidate_id(method: str, seed: int, lam: float, epochs: int) -> str:
    return f"{method}_s{seed}_l{lam:g}_e{epochs}"


def _epochs(config: dict, method: str) -> int:
    if method == "pacmap":
        return int(config["pacmap_epochs"])
    return int(config["cne_epochs"])


def write_environment(path: Path) -> None:
    import sklearn
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
    }
    path.write_text(json.dumps(payload, indent=2))


def generate_candidates(config: dict) -> pd.DataFrame:
    out = Path(config["out"])
    emb_dir = out / "candidates"
    emb_dir.mkdir(parents=True, exist_ok=True)
    data = load_dataset(config["dataset"])
    neighbors = high_d_neighbors(data["X"], k=int(config["k_neighbors"]))
    pairs = pacmap_pairs(data["X"], seed=0)
    rows = []
    store = {}
    for method in config["methods"]:
        for seed in config["seeds"]:
            for lam in config["lambdas"]:
                epochs = _epochs(config, method)
                embedding_id = _candidate_id(method, int(seed), float(lam), epochs)
                cache = emb_dir / f"{embedding_id}.npy"
                started = time.perf_counter()
                if cache.exists():
                    embedding = np.load(cache)
                    train_seconds = np.nan
                    reused = True
                else:
                    if method == "pacmap":
                        embedding, train_seconds = fit_pacmap(
                            data["X"], data["X_pca"], int(seed), float(lam), epochs, pairs
                        )
                    else:
                        embedding, train_seconds = fit_cne(
                            data["X"],
                            data["X_pca"],
                            method,
                            int(seed),
                            float(lam),
                            epochs,
                        )
                    np.save(cache, embedding)
                    reused = False
                if method == "pacmap":
                    parts = pacmap_component_losses(
                        embedding, pairs[0], pairs[1], pairs[2], data["n"]
                    )
                    native = parts["native_loss"]
                    native_std = 0.0
                    probe = parts["probe_loss"]
                    extra = {
                        "pacmap_nn": parts["pacmap_nn"],
                        "pacmap_mn": parts["pacmap_mn"],
                        "pacmap_fp": parts["pacmap_fp"],
                    }
                else:
                    native, native_std = contrastive_reference_loss(
                        embedding,
                        neighbors,
                        method,
                        negative_samples=int(config["negative_samples"]),
                        n_draws=int(config["native_draws"]),
                    )
                    probe = contrastive_probe(embedding, neighbors, method)
                    extra = {"pacmap_nn": np.nan, "pacmap_mn": np.nan, "pacmap_fp": np.nan}
                scores = evaluate_embedding(
                    data["X"],
                    embedding,
                    data["labels"] if data["discrete_labels"] else None,
                    k=int(config["k_eval"]),
                    triplets=_triplets(data),
                )
                if data["latent"] is not None:
                    scores["latent_triplet_accuracy"] = latent_triplet_accuracy(
                        data["latent"], embedding, _triplets(data)
                    )
                row = {
                    "dataset": data["name"],
                    "embedding_id": embedding_id,
                    "method": method,
                    "seed": int(seed),
                    "lambda_pca": float(lam),
                    "n_epochs": epochs,
                    "undertrained": False,
                    "objective_family": method,
                    "reused_cache": reused,
                    "train_seconds": train_seconds,
                    "native_loss": native,
                    "native_loss_std": native_std,
                    "probe_loss": probe,
                    "error": "",
                    **extra,
                    **scores,
                }
                rows.append(row)
                store[embedding_id] = np.asarray(embedding, dtype=np.float32)
                print(
                    f"  {'cache' if reused else 'train'} {embedding_id} "
                    f"native={native:.4f} probe={probe:.4f} "
                    f"trust={scores['trustworthiness']:.3f} "
                    f"({time.perf_counter() - started:.1f}s)",
                    flush=True,
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "candidates.csv", index=False)
    np.savez_compressed(
        out / "candidates.npz",
        ids=np.array(list(store), dtype=str),
        embeddings=np.stack([store[key] for key in store]),
    )
    _save_context(out, data, pairs, neighbors)
    return frame


def _save_context(out: Path, data: dict, pairs, neighbors: np.ndarray) -> None:
    np.savez_compressed(
        out / "context.npz",
        X=data["X"],
        X_pca=data["X_pca"],
        labels=np.asarray(data["labels"] if data["labels"] is not None else [], dtype=float),
        discrete=np.array([data["discrete_labels"]]),
        color=np.asarray(data["color"], dtype=float),
        latent=np.asarray(data["latent"] if data["latent"] is not None else [], dtype=float),
        observation_ids=data["observation_ids"],
        neighbors=neighbors,
        pair_neighbors=pairs[0],
        pair_mn=pairs[1],
        pair_fp=pairs[2],
    )


def _triplets(data: dict) -> np.ndarray:
    if not hasattr(_triplets, "cache"):
        _triplets.cache = {}
    if data["name"] not in _triplets.cache:
        _triplets.cache[data["name"]] = _sample_triplets(
            data["n"], 4000, np.random.default_rng(0)
        )
    return _triplets.cache[data["name"]]


def load_pool(config: dict) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict]:
    out = Path(config["out"])
    frame = pd.read_csv(out / "candidates.csv")
    archive = np.load(out / "candidates.npz")
    embeddings = {
        str(embedding_id): archive["embeddings"][i]
        for i, embedding_id in enumerate(archive["ids"])
    }
    context = dict(np.load(out / "context.npz"))
    return frame, embeddings, context


def select_members(frame: pd.DataFrame, criterion: str, epsilon: float) -> pd.DataFrame:
    if criterion == "native":
        marked = mark_by_loss(frame, "native_loss", epsilon)
    elif criterion == "probe":
        marked = mark_by_loss(frame, "probe_loss", epsilon)
    elif criterion == "quality":
        temp = frame.copy()
        temp["quality_loss"] = 1.0 - temp["neighborhood_preservation"]
        marked = mark_by_loss(temp, "quality_loss", epsilon)
    elif criterion == "multi":
        marked = mark_by_loss(frame, "native_loss", epsilon)
        keep = []
        for method, group in frame.groupby("method"):
            threshold = float(group["neighborhood_preservation"].median())
            chosen = marked.index[
                marked["method"].eq(method)
                & marked["in_rashomon"]
                & marked["neighborhood_preservation"].ge(threshold)
            ]
            keep.extend(chosen.tolist())
        marked["in_rashomon"] = False
        marked.loc[keep, "in_rashomon"] = True
    elif criterion == "conventional_native":
        marked = frame.copy()
        marked["in_rashomon"] = False
        for _method, group in marked.groupby("method"):
            marked.loc[group["native_loss"].idxmin(), "in_rashomon"] = True
    elif criterion == "all":
        marked = frame.copy()
        marked["in_rashomon"] = True
    else:
        raise ValueError(criterion)
    return marked.loc[marked["in_rashomon"]].copy()


def _scheme_weights(scheme: str, members: pd.DataFrame, embeddings: list[np.ndarray], k: int):
    if scheme == "uniform":
        return uniform_weights(members), None
    if scheme == "method_balanced":
        return method_balanced_weights(members), None
    if scheme == "gibbs":
        return gibbs_weights(members, "native_loss", beta=1.0), None
    if scheme == "gibbs_weak":
        return gibbs_weights(members, "native_loss", beta=0.25), None
    if scheme == "gibbs_strong":
        return gibbs_weights(members, "native_loss", beta=4.0), None
    if scheme == "vote":
        return vote_weights(members, embeddings, k=k), None
    if scheme == "entropy_confidence":
        probability = vote_probability(embeddings, k)
        return uniform_weights(members), confidence_matrices(probability)["entropy_confidence"]
    if scheme == "positive_consensus":
        probability = vote_probability(embeddings, k)
        return uniform_weights(members), confidence_matrices(probability)["positive_consensus"]
    if scheme == "combined_frequency_confidence":
        probability = vote_probability(embeddings, k)
        return uniform_weights(members), confidence_matrices(probability)[
            "combined_frequency_confidence"
        ]
    if scheme == "distance_confidence":
        return uniform_weights(members), distance_confidence(embeddings)
    raise ValueError(scheme)


def _digest(kind: str, scheme: str, ids: list[str], weights: np.ndarray, weightmat) -> str:
    parts = [
        kind.encode(),
        scheme.encode(),
        ",".join(ids).encode(),
        np.round(np.asarray(weights), 6).tobytes(),
    ]
    if weightmat is not None:
        parts.append(np.round(np.asarray(weightmat), 5).tobytes())
    return hashlib.sha1(b"|".join(parts)).hexdigest()[:16]


def _fit_or_load(out: Path, kind: str, embeddings, weights, weightmat, ids, scheme: str):
    digest = _digest(kind, scheme, ids, weights, weightmat)
    dest = out / "fits" / digest
    meta_path = dest / "meta.txt"
    if kind in {"procrustes_mean", "distance_mds"}:
        embedding = (
            _procrustes_mean(embeddings)
            if kind == "procrustes_mean"
            else _distance_mds(embeddings)
        )
        return {
            "embedding": embedding,
            "stress": np.nan,
            "seconds": 0.0,
            "niter": 0,
            "reused_fit": False,
            "digest": digest,
        }
    if meta_path.exists():
        meta = {}
        for line in meta_path.read_text().splitlines():
            key, value = line.split("=", 1)
            meta[key] = value
        embedding = pd.read_csv(dest / "gspace.csv", header=None).to_numpy(dtype=float)
        return {
            "embedding": embedding,
            "stress": float(meta["stress"]),
            "seconds": float(meta["seconds"]),
            "niter": int(float(meta["niter"])),
            "reused_fit": True,
            "digest": digest,
        }
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    fitted = fit_consensus(
        embeddings,
        dest,
        kind="locomds" if kind == "locomds" else "comds",
        weights=weights,
        weightmat=weightmat,
        observation_ids=np.arange(embeddings[0].shape[0]),
    )
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    fitted["reused_fit"] = False
    fitted["digest"] = digest
    fitted["maxrss_kb"] = max(before, after)
    return fitted


def _procrustes_mean(embeddings: list[np.ndarray]) -> np.ndarray:
    reference = np.asarray(embeddings[0], dtype=float)
    total = np.zeros_like(reference)
    for embedding in embeddings:
        total += display_align(embedding, reference)
    return total / len(embeddings)


def _distance_mds(embeddings: list[np.ndarray]) -> np.ndarray:
    matrices = []
    for embedding in embeddings:
        dist = pairwise_distances(np.asarray(embedding, dtype=float))
        iu = np.triu_indices(dist.shape[0], k=1)
        dist = dist / max(float(dist[iu].mean()), 1e-12)
        matrices.append(dist)
    mean = np.mean(matrices, axis=0)
    np.fill_diagonal(mean, 0.0)
    n = mean.shape[0]
    gram = np.eye(n) - np.ones((n, n)) / n
    inner = -0.5 * gram @ (mean**2) @ gram
    evals, evecs = np.linalg.eigh(inner)
    order = np.argsort(evals)[::-1][:2]
    return evecs[:, order] * np.sqrt(np.maximum(evals[order], 0.0))


def _diversity_row(members: pd.DataFrame, embeddings: dict[str, np.ndarray], k: int) -> dict:
    ids = members["embedding_id"].tolist()
    views = [embeddings[i] for i in ids]
    if len(views) < 2:
        return {
            "within_overlap": np.nan,
            "between_overlap": np.nan,
            "redundant_embeddings": 0,
        }
    overlap = pairwise_neighborhood_overlap(views, k=k)
    within, between = within_between_overlap(overlap, members["method"].tolist())
    return {
        "within_overlap": within,
        "between_overlap": between,
        "redundant_embeddings": redundancy_count(overlap),
    }


def _evaluate_consensus(context: dict, embedding: np.ndarray, k: int, triplets: np.ndarray) -> dict:
    labels = context["labels"] if bool(context["discrete"][0]) and len(context["labels"]) else None
    scores = evaluate_embedding(context["X"], embedding, labels, k=k, triplets=triplets)
    if len(context["latent"]):
        scores["latent_triplet_accuracy"] = latent_triplet_accuracy(
            context["latent"], embedding, triplets
        )
    return scores


def sweep(config: dict) -> pd.DataFrame:
    frame, embeddings, context = load_pool(config)
    out = Path(config["out"])
    k = int(config["k_eval"])
    triplets = _sample_triplets(int(context["X"].shape[0]), 4000, np.random.default_rng(0))
    jobs = _jobs(config)
    rows = []
    for job in jobs:
        try:
            row = _run_job(job, frame, embeddings, context, triplets, k, out)
        except Exception as exc:
            traceback.print_exc()
            row = {**job, "error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        label = row.get("run_name", job["run_name"])
        print(
            f"  {label} views={row.get('n_views', 'na')} "
            f"trust={row.get('trustworthiness', float('nan'))} "
            f"triplet={row.get('triplet_accuracy', float('nan'))}",
            flush=True,
        )
    result = pd.DataFrame(rows)
    result.to_csv(out / "sweep.csv", index=False)
    _write_membership_curve(frame, out)
    _plots(frame, result, embeddings, context, out)
    return result


def _jobs(config: dict) -> list[dict]:
    jobs = []

    def add(run_name, criterion, epsilon, scheme, kind, sample_seed=None):
        jobs.append(
            {
                "run_name": run_name,
                "criterion": criterion,
                "epsilon": float(epsilon),
                "scheme": scheme,
                "kind": kind,
                "sample_seed": sample_seed,
            }
        )

    for epsilon in EPSILONS:
        label = _eps_label(epsilon)
        for kind in ("comds", "locomds"):
            add(f"native_uniform_{kind}_eps{label}", "native", epsilon, "uniform", kind)
    for epsilon in (0.05, 0.1, np.inf):
        label = _eps_label(epsilon)
        for scheme in ("method_balanced", "gibbs"):
            for kind in ("comds", "locomds"):
                add(
                    f"native_{scheme}_{kind}_eps{label}",
                    "native",
                    epsilon,
                    scheme,
                    kind,
                )
    add("native_gibbs_weak_comds_eps0.05", "native", 0.05, "gibbs_weak", "comds")
    add("native_gibbs_strong_comds_eps0.05", "native", 0.05, "gibbs_strong", "comds")
    add("native_vote_comds_eps0.05", "native", 0.05, "vote", "comds")
    for scheme in (
        "entropy_confidence",
        "positive_consensus",
        "combined_frequency_confidence",
        "distance_confidence",
    ):
        add(f"native_{scheme}_comds_eps0.05", "native", 0.05, scheme, "comds")
    for kind in ("comds", "locomds"):
        add(f"probe_uniform_{kind}_eps0.05", "probe", 0.05, "uniform", kind)
        add(f"conventional_native_{kind}", "conventional_native", 0.0, "uniform", kind)
    add("quality_uniform_comds_eps0.05", "quality", 0.05, "uniform", "comds")
    add("multi_uniform_comds_eps0.05", "multi", 0.05, "uniform", "comds")
    add("native_procrustes_mean_eps0.05", "native", 0.05, "uniform", "procrustes_mean")
    add("native_distance_mds_eps0.05", "native", 0.05, "uniform", "distance_mds")
    add("all_procrustes_mean", "all", np.inf, "uniform", "procrustes_mean")
    add("all_distance_mds", "all", np.inf, "uniform", "distance_mds")
    for seed in range(int(config["random_conventional_seeds"])):
        add(
            f"random_one_per_method_comds_s{seed}",
            "random_one",
            np.inf,
            "uniform",
            "comds",
            sample_seed=seed,
        )
    for seed in range(int(config["matched_seeds"])):
        for kind in ("comds", "locomds"):
            add(
                f"matched_uniform_{kind}_eps0.05_s{seed}",
                "matched",
                0.05,
                "uniform",
                kind,
                sample_seed=seed,
            )
    return jobs


def _run_job(job, frame, embeddings, context, triplets, k, out: Path) -> dict:
    members = _members_for_job(job, frame)
    if len(members) == 0:
        return {**job, "n_views": 0, "error": "empty ensemble"}
    ids = members["embedding_id"].tolist()
    views = [np.asarray(embeddings[i], dtype=np.float64) for i in ids]
    weights, weightmat = _scheme_weights(job["scheme"], members, views, k)
    if job["kind"] == "locomds" and weightmat is not None:
        return {**job, "n_views": len(views), "error": "LoCoMDS does not take weightmat"}
    fitted = _fit_or_load(out, job["kind"], views, weights, weightmat, ids, job["scheme"])
    scores = _evaluate_consensus(context, fitted["embedding"], k, triplets)
    diversity = _diversity_row(members, embeddings, k)
    per_method = members.groupby("method").size().to_dict()
    return {
        **job,
        "n_views": len(views),
        "method_counts": json.dumps(per_method),
        "consensus_stress": fitted["stress"],
        "consensus_seconds": fitted["seconds"],
        "niter": fitted["niter"],
        "reused_fit": fitted["reused_fit"],
        "digest": fitted["digest"],
        "error": "",
        **diversity,
        **scores,
    }


def _members_for_job(job: dict, frame: pd.DataFrame) -> pd.DataFrame:
    if job["criterion"] == "matched":
        rashomon = select_members(frame, "native", job["epsilon"])
        rng = np.random.default_rng(int(job["sample_seed"]))
        chosen = []
        for _method, group in frame.groupby("method"):
            k = int(rashomon["method"].eq(group["method"].iloc[0]).sum())
            take = group.index.to_numpy()
            if k <= 0:
                continue
            if k >= len(take):
                chosen.extend(take.tolist())
            else:
                chosen.extend(rng.choice(take, size=k, replace=False).tolist())
        return frame.loc[chosen].copy()
    if job["criterion"] == "random_one":
        rng = np.random.default_rng(int(job["sample_seed"]))
        chosen = []
        for _method, group in frame.groupby("method"):
            chosen.append(int(rng.choice(group.index.to_numpy())))
        return frame.loc[chosen].copy()
    return select_members(frame, job["criterion"], job["epsilon"])


def _write_membership_curve(frame: pd.DataFrame, out: Path) -> None:
    rows = []
    for criterion, column in (("native", "native_loss"), ("probe", "probe_loss")):
        for epsilon in EPSILONS:
            marked = mark_by_loss(frame, column, epsilon)
            for method, group in marked.groupby("method"):
                rows.append(
                    {
                        "criterion": criterion,
                        "epsilon": float(epsilon) if np.isfinite(epsilon) else np.inf,
                        "method": method,
                        "n_retained": int(group["in_rashomon"].sum()),
                        "n_candidates": int(len(group)),
                        "retained_fraction": float(group["in_rashomon"].mean()),
                    }
                )
    pd.DataFrame(rows).to_csv(out / "membership_curve.csv", index=False)


def _plots(frame, sweep_frame, embeddings, context, out: Path) -> None:
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _plot_epsilon(sweep_frame, fig_dir / "epsilon_curves.png")
    _plot_loss_quality(frame, fig_dir / "native_loss_vs_trustworthiness.png")
    _plot_gallery(frame, embeddings, context, fig_dir / "umap_lambda_gallery.png")


def _plot_epsilon(sweep_frame: pd.DataFrame, path: Path) -> None:
    curve = sweep_frame[
        sweep_frame["run_name"].str.startswith("native_uniform_")
        & sweep_frame["error"].fillna("").eq("")
    ].copy()
    if curve.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for kind, color in (("comds", "#1f4e79"), ("locomds", "#b85c38")):
        part = curve[curve["kind"] == kind].sort_values("epsilon")
        x = part["epsilon"].to_numpy(dtype=float)
        x = np.where(np.isfinite(x), x, 2.0)
        axes[0].plot(x, part["n_views"], marker="o", color=color, label=kind)
        axes[1].plot(x, part["trustworthiness"], marker="o", color=color, label=kind)
        axes[2].plot(x, part["triplet_accuracy"], marker="o", color=color, label=kind)
    conventional = sweep_frame[sweep_frame["run_name"] == "conventional_native_comds"]
    if len(conventional):
        axes[2].axhline(
            float(conventional["triplet_accuracy"].iloc[0]),
            color="#1f4e79",
            linestyle="--",
            linewidth=1,
            label="one-per-method CoMDS",
        )
    axes[0].set_ylabel("Retained views")
    axes[1].set_ylabel("Trustworthiness")
    axes[2].set_ylabel("Triplet accuracy")
    for axis in axes:
        axis.set_xlabel("epsilon (2 denotes infinity)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_loss_quality(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for method, color in (("umap", "#1f4e79"), ("infonce", "#2a9d8f"), ("pacmap", "#b85c38")):
        part = frame[frame["method"] == method]
        ax.scatter(
            part["native_loss"],
            part["trustworthiness"],
            s=28,
            c=color,
            label=method,
        )
    ax.set_xlabel("Native loss")
    ax.set_ylabel("Trustworthiness")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_gallery(frame, embeddings, context, path: Path) -> None:
    part = frame[(frame["method"] == "umap") & (frame["seed"] == 0)].sort_values("lambda_pca")
    if part.empty:
        return
    fig, axes = plt.subplots(1, len(part), figsize=(2.2 * len(part), 2.4))
    reference = context["X_pca"]
    color = context["color"]
    for axis, row in zip(np.atleast_1d(axes), part.itertuples(index=False)):
        aligned = display_align(embeddings[row.embedding_id], reference)
        axis.scatter(aligned[:, 0], aligned[:, 1], c=color, cmap="tab10", s=12)
        axis.set_title(f"λ={row.lambda_pca:g}", fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def duplication_and_contamination(config: dict) -> None:
    """One duplicate-weight check and one pool that adds 2-epoch embeddings."""
    out = Path(config["out"])
    frame, embeddings, context = load_pool(config)
    k = int(config["k_eval"])
    triplets = _sample_triplets(int(context["X"].shape[0]), 4000, np.random.default_rng(0))
    members = select_members(frame, "native", 0.05)
    rows = []
    if len(members) and (members["method"] == "pacmap").any():
        pac = members[members["method"] == "pacmap"]
        duplicated = pd.concat([members, pac], ignore_index=True)
        ids = duplicated["embedding_id"].tolist()
        views = [np.asarray(embeddings[i], dtype=np.float64) for i in ids]
        for scheme, weights in (
            ("uniform", uniform_weights(duplicated)),
            ("method_balanced", method_balanced_weights(duplicated)),
        ):
            fitted = _fit_or_load(out, "comds", views, weights, None, ids, f"dup_{scheme}")
            scores = _evaluate_consensus(context, fitted["embedding"], k, triplets)
            rows.append(
                {
                    "study": "duplication",
                    "scheme": scheme,
                    "n_views": len(views),
                    "consensus_stress": fitted["stress"],
                    "consensus_seconds": fitted["seconds"],
                    **scores,
                }
            )
            print(f"  duplication {scheme} triplet={scores['triplet_accuracy']:.3f}", flush=True)
    if config.get("contamination"):
        data = load_dataset(config["dataset"])
        neighbors = high_d_neighbors(data["X"], k=int(config["k_neighbors"]))
        pairs = pacmap_pairs(data["X"], seed=0)
        bad_ids = []
        for method in config["methods"]:
            embedding_id = f"{method}_s0_l0_e2_undertrained"
            cache = out / "candidates" / f"{embedding_id}.npy"
            if method == "pacmap":
                embedding, seconds = fit_pacmap(data["X"], data["X_pca"], 0, 0.0, 2, pairs)
                parts = pacmap_component_losses(embedding, pairs[0], pairs[1], pairs[2], data["n"])
                native, probe = parts["native_loss"], parts["probe_loss"]
            else:
                embedding, seconds = fit_cne(data["X"], data["X_pca"], method, 0, 0.0, 2)
                native, _std = contrastive_reference_loss(
                    embedding, neighbors, method, n_draws=int(config["native_draws"])
                )
                probe = contrastive_probe(embedding, neighbors, method)
            np.save(cache, embedding)
            embeddings[embedding_id] = embedding
            bad_ids.append(embedding_id)
            print(f"  contamination train {embedding_id} native={native:.4f} probe={probe:.4f} ({seconds:.1f}s)", flush=True)
        ids = frame["embedding_id"].tolist() + bad_ids
        views = [np.asarray(embeddings[i], dtype=np.float64) for i in ids]
        fitted = _fit_or_load(
            out, "comds", views, np.ones(len(views)), None, ids, "contaminated_uniform"
        )
        scores = _evaluate_consensus(context, fitted["embedding"], k, triplets)
        rows.append(
            {
                "study": "contamination",
                "scheme": "uniform",
                "n_views": len(views),
                "consensus_stress": fitted["stress"],
                "consensus_seconds": fitted["seconds"],
                **scores,
            }
        )
        print(f"  contaminated uniform triplet={scores['triplet_accuracy']:.3f}", flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(out / "robustness.csv", index=False)


def run_configured(config_path: Path) -> None:
    config = load_config(config_path)
    out = Path(config["out"])
    out.mkdir(parents=True, exist_ok=True)
    write_environment(out / "environment.json")
    print(f"=== candidates {config['name']} ===", flush=True)
    generate_candidates(config)
    print("=== sweep ===", flush=True)
    sweep(config)
    print("=== duplication and contamination ===", flush=True)
    duplication_and_contamination(config)
