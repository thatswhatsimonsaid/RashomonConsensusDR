"""Independent replications of a manifold consensus experiment.

Candidate pools are trained once per replication and reused for every
epsilon and weighting. The primary recovery metric is chosen from the
geometry of the unfolding, before consensus scores are read.
"""

from __future__ import annotations

import hashlib
import json
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from redr.checkpoint2 import (
    EPSILONS,
    _diversity_row,
    _eps_label,
    _fit_or_load,
    _members_for_job,
    _scheme_weights,
)
from redr.common_knowledge import candidate_edges, edges_to_graph, mean_score_edges
from redr.intrinsic import (
    coordinate_triplet_accuracy,
    intrinsic_recovery,
    triplet_order_disagreement,
)
from redr.manifolds import generate_manifold
from redr.metrics import _sample_triplets, evaluate_embedding
from redr.train import fit_cne, fit_pacmap, high_d_neighbors, pacmap_pairs

REPO = Path(__file__).resolve().parents[2]
INTRINSIC_COLUMNS = [
    "intrinsic_triplet_isometric",
    "intrinsic_triplet_balanced",
    "arc_triplet_accuracy",
    "height_triplet_accuracy",
    "generator_parameter_triplet_accuracy",
    "intrinsic_neighborhood",
    "intrinsic_neighborhood_balanced",
    "intrinsic_rank_correlation",
    "intrinsic_rank_correlation_balanced",
    "intrinsic_distance_error",
    "intrinsic_procrustes",
]


def load_config(path: Path) -> dict:
    with Path(path).open() as handle:
        config = yaml.safe_load(handle)
    config["config_path"] = str(Path(path).resolve())
    config["out"] = REPO / "results" / "checkpoint3" / config["study"]
    return config


def replication_dir(config: dict, replication: int) -> Path:
    return Path(config["out"]) / f"rep{int(replication):02d}"


def data_seed(config: dict, replication: int) -> int:
    return int(config["data_seed_offset"]) + int(replication)


def eval_seed(config: dict, replication: int) -> int:
    return int(config["eval_seed_offset"]) + int(replication)


def lock_primary_metric(config: dict, replications: list[int]) -> dict:
    """Choose the confirmatory endpoint from unfolding geometry only."""
    rates = []
    for replication in replications:
        data = generate_manifold(
            config["manifold"],
            int(config["n_samples"]),
            float(config["noise"]),
            data_seed(config, replication),
        )
        triplets = _sample_triplets(
            data["n"],
            int(config["n_triplets"]),
            np.random.default_rng(9000 + int(replication)),
        )
        rates.append(
            {
                "replication": int(replication),
                "data_seed": data_seed(config, replication),
                "order_disagreement": triplet_order_disagreement(
                    data["intrinsic"], triplets
                ),
                "arc_span": float(np.ptp(data["arc_length"])),
                "height_span": float(np.ptp(data["height"])),
            }
        )
    mean_rate = float(np.mean([row["order_disagreement"] for row in rates]))
    threshold = float(config.get("isometric_disagreement_threshold", 0.10))
    if mean_rate < threshold:
        metric = "intrinsic_triplet_balanced"
        reason = (
            "Fewer than "
            f"{threshold:.0%} of triplet decisions change when height is added "
            "to arc length, so the raw unfolding mostly restates arc-length order. "
            "The confirmatory endpoint is the coordinate-balanced intrinsic triplet."
        )
    else:
        metric = "intrinsic_triplet_isometric"
        reason = (
            "Adding height changes at least "
            f"{threshold:.0%} of triplet decisions, so isometric unfolding "
            "distance is the confirmatory endpoint."
        )
    payload = {
        "primary_metric": metric,
        "mean_order_disagreement": mean_rate,
        "threshold": threshold,
        "reason": reason,
        "replications": rates,
        "frozen_epsilon": float(config["primary_epsilon"]),
        "comparators": [
            "conventional_native",
            "matched_uniform",
            "unrestricted_uniform",
            "unrestricted_method_balanced",
            "rashomon_uniform",
            "rashomon_method_balanced",
            "rashomon_gibbs",
        ],
    }
    path = Path(config["out"]) / "geometry_lock.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"primary_metric": metric, "mean_order_disagreement": mean_rate}))
    return payload


def load_lock(config: dict) -> dict:
    return json.loads((Path(config["out"]) / "geometry_lock.json").read_text())


def _candidate_id(method: str, seed: int, lam: float, epochs: int) -> str:
    return f"{method}_s{seed}_l{lam:g}_e{epochs}"


def _epochs(config: dict, method: str) -> int:
    if method == "pacmap":
        return int(config["pacmap_epochs"])
    return int(config["cne_epochs"])


def _score_embedding(data: dict, embedding: np.ndarray, triplets: np.ndarray, k: int) -> dict:
    ambient = evaluate_embedding(data["X"], embedding, None, k=k, triplets=triplets)
    intrinsic = intrinsic_recovery(data["intrinsic"], embedding, triplets, k=k)
    intrinsic["generator_parameter_triplet_accuracy"] = coordinate_triplet_accuracy(
        data["parameter"], embedding, triplets
    )
    return {**ambient, **intrinsic}


def generate_candidates(config: dict, replication: int) -> pd.DataFrame:
    out = replication_dir(config, replication)
    cache_dir = out / "candidates"
    cache_dir.mkdir(parents=True, exist_ok=True)
    data = _load_or_save_data(config, replication)
    neighbors = high_d_neighbors(data["X"], k=int(config["k_neighbors"]))
    pairs = pacmap_pairs(data["X"], seed=int(config.get("pair_seed", 0)))
    np.savez_compressed(
        out / "pairs.npz",
        neighbors=neighbors,
        pair_neighbors=pairs[0],
        pair_mn=pairs[1],
        pair_fp=pairs[2],
    )
    triplets = _triplets(config, replication)
    rows = []
    store = {}
    from redr.loss_validation import contrastive_reference_loss, pacmap_component_losses
    from redr.probe import contrastive_probe

    for method in config["methods"]:
        for seed in config["training_seeds"]:
            for lam in config["lambdas"]:
                epochs = _epochs(config, method)
                embedding_id = _candidate_id(method, int(seed), float(lam), epochs)
                cache = cache_dir / f"{embedding_id}.npy"
                if cache.exists():
                    embedding = np.load(cache)
                    train_seconds = np.nan
                    reused = True
                else:
                    if method == "pacmap":
                        embedding, train_seconds = fit_pacmap(
                            data["X"],
                            data["X_pca"],
                            int(seed),
                            float(lam),
                            epochs,
                            pairs,
                        )
                    else:
                        embedding, train_seconds = fit_cne(
                            data["X"],
                            data["X_pca"],
                            method,
                            int(seed),
                            float(lam),
                            epochs,
                            k=int(config["k_neighbors"]),
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
                else:
                    native, native_std = contrastive_reference_loss(
                        embedding,
                        neighbors,
                        method,
                        negative_samples=int(config["negative_samples"]),
                        n_draws=int(config["native_draws"]),
                    )
                    probe = contrastive_probe(embedding, neighbors, method)
                scores = _score_embedding(
                    data, embedding, triplets, int(config["k_eval"])
                )
                row = {
                    "replication": int(replication),
                    "data_seed": data_seed(config, replication),
                    "dataset": data["name"],
                    "embedding_id": embedding_id,
                    "method": method,
                    "seed": int(seed),
                    "lambda_pca": float(lam),
                    "n_epochs": epochs,
                    "undertrained": False,
                    "reused_cache": reused,
                    "train_seconds": train_seconds,
                    "native_loss": native,
                    "native_loss_std": native_std,
                    "probe_loss": probe,
                    "error": "",
                    **scores,
                }
                rows.append(row)
                store[embedding_id] = np.asarray(embedding, dtype=np.float32)
                print(
                    f"  rep{replication:02d} {'cache' if reused else 'train'} "
                    f"{embedding_id} native={native:.4f} "
                    f"balanced={scores['intrinsic_triplet_balanced']:.3f} "
                    f"arc={scores['arc_triplet_accuracy']:.3f}",
                    flush=True,
                )
                pd.DataFrame(rows).to_csv(out / "candidates.csv", index=False)
    frame = pd.DataFrame(rows)
    np.savez_compressed(
        out / "candidates.npz",
        ids=np.array(list(store), dtype=str),
        embeddings=np.stack([store[key] for key in store]),
    )
    return frame


def _load_or_save_data(config: dict, replication: int) -> dict:
    out = replication_dir(config, replication)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifold.npz"
    if path.exists():
        archive = np.load(path)
        data = {key: archive[key] for key in archive.files}
        data["name"] = str(data["name"])
        data["n"] = int(data["X"].shape[0])
        return data
    data = generate_manifold(
        config["manifold"],
        int(config["n_samples"]),
        float(config["noise"]),
        data_seed(config, replication),
    )
    np.savez_compressed(
        path,
        name=np.array(data["name"]),
        X=data["X"],
        X_pca=data["X_pca"],
        X_raw=data["X_raw"],
        X_clean=data["X_clean"],
        parameter=data["parameter"],
        arc_length=data["arc_length"],
        height=data["height"],
        intrinsic=data["intrinsic"],
        observation_ids=data["observation_ids"],
    )
    return data


def _triplets(config: dict, replication: int) -> np.ndarray:
    out = replication_dir(config, replication)
    path = out / "triplets.npy"
    if path.exists():
        return np.load(path)
    data = _load_or_save_data(config, replication)
    triplets = _sample_triplets(
        data["n"],
        int(config["n_triplets"]),
        np.random.default_rng(eval_seed(config, replication)),
    )
    np.save(path, triplets)
    return triplets


def build_jobs(config: dict) -> list[dict]:
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
            add(
                f"native_uniform_{kind}_eps{label}",
                "native",
                epsilon,
                "uniform",
                kind,
            )
    for epsilon in (float(config["primary_epsilon"]), np.inf):
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
    for kind in ("comds", "locomds"):
        add(f"conventional_native_{kind}", "conventional_native", 0.0, "uniform", kind)
    label = _eps_label(config["primary_epsilon"])
    add("all_procrustes_mean", "all", np.inf, "uniform", "procrustes_mean")
    add(
        f"rashomon_procrustes_mean_eps{label}",
        "native",
        float(config["primary_epsilon"]),
        "uniform",
        "procrustes_mean",
    )
    add("all_distance_mds", "all", np.inf, "uniform", "distance_mds")
    add(
        f"rashomon_distance_mds_eps{label}",
        "native",
        float(config["primary_epsilon"]),
        "uniform",
        "distance_mds",
    )
    for seed in range(int(config["random_one_seeds"])):
        for kind in ("comds", "locomds"):
            add(
                f"random_one_{kind}_s{seed}",
                "random_one",
                np.inf,
                "uniform",
                kind,
                sample_seed=seed,
            )
    for seed in range(int(config["matched_seeds"])):
        for kind in ("comds", "locomds"):
            add(
                f"matched_uniform_{kind}_eps{_eps_label(config['primary_epsilon'])}_s{seed}",
                "matched",
                float(config["primary_epsilon"]),
                "uniform",
                kind,
                sample_seed=seed,
            )
    return jobs


def _content_scheme(scheme: str, views: list[np.ndarray]) -> str:
    stacked = np.ascontiguousarray(np.stack([np.asarray(v, dtype=np.float64) for v in views]))
    digest = hashlib.sha1(stacked.tobytes()).hexdigest()[:8]
    return f"{scheme}|{digest}"


def run_consensus(config: dict, replication: int, workers: int = 2) -> pd.DataFrame:
    out = replication_dir(config, replication)
    frame = pd.read_csv(out / "candidates.csv")
    archive = np.load(out / "candidates.npz")
    embeddings = {
        str(embedding_id): archive["embeddings"][i]
        for i, embedding_id in enumerate(archive["ids"])
    }
    data = _load_or_save_data(config, replication)
    triplets = _triplets(config, replication)
    k = int(config["k_eval"])
    jobs = build_jobs(config)
    sweep_path = out / "sweep.csv"
    done = {}
    if sweep_path.exists():
        previous = pd.read_csv(sweep_path)
        done = {
            str(row["run_name"]): row.to_dict()
            for _, row in previous.iterrows()
        }
    rows = []
    pending = []
    for job in jobs:
        if job["run_name"] in done and not _failed(done[job["run_name"]]):
            rows.append(done[job["run_name"]])
        else:
            pending.append(job)
    write_lock = __import__("threading").Lock()

    def store(row: dict) -> None:
        with write_lock:
            rows.append(row)
            _write_sweep(rows, sweep_path)
        _print_row(row)

    if workers <= 1 or len(pending) <= 1:
        for job in pending:
            store(_safe_job(job, frame, embeddings, data, triplets, k, out, replication))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _safe_job, job, frame, embeddings, data, triplets, k, out, replication
                )
                for job in pending
            ]
            for future in as_completed(futures):
                store(future.result())
    _common_knowledge(config, replication, frame, embeddings, data, triplets, k)
    result = pd.read_csv(sweep_path)
    return result


_FIT_LOCKS: dict[tuple, object] = {}
_FIT_LOCKS_GUARD = __import__("threading").Lock()


def _fit_lock(key: tuple):
    with _FIT_LOCKS_GUARD:
        lock = _FIT_LOCKS.get(key)
        if lock is None:
            lock = __import__("threading").Lock()
            _FIT_LOCKS[key] = lock
        return lock


def _failed(row: dict) -> bool:
    error = row.get("error", "")
    if error is None or (isinstance(error, float) and np.isnan(error)):
        return False
    return str(error).strip() not in {"", "nan"}


def _write_sweep(rows: list[dict], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _print_row(row: dict) -> None:
    print(
        f"  {row.get('run_name')} views={row.get('n_views')} "
        f"balanced={row.get('intrinsic_triplet_balanced')} "
        f"iso={row.get('intrinsic_triplet_isometric')} "
        f"sec={row.get('consensus_seconds')}",
        flush=True,
    )


def _safe_job(job, frame, embeddings, data, triplets, k, out, replication) -> dict:
    try:
        return _run_job(job, frame, embeddings, data, triplets, k, out, replication)
    except Exception as exc:
        traceback.print_exc()
        return {
            **job,
            "replication": int(replication),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _run_job(job, frame, embeddings, data, triplets, k, out, replication) -> dict:
    members = _members_for_job(job, frame)
    if len(members) == 0:
        return {**job, "replication": int(replication), "n_views": 0, "error": "empty ensemble"}
    ids = members["embedding_id"].tolist()
    views = [np.asarray(embeddings[i], dtype=np.float64) for i in ids]
    weights, weightmat = _scheme_weights(job["scheme"], members, views, k)
    if job["kind"] == "locomds" and weightmat is not None:
        return {
            **job,
            "replication": int(replication),
            "n_views": len(views),
            "error": "LoCoMDS does not take weightmat",
        }
    fit_key = (job["kind"], job["scheme"], tuple(ids))
    with _fit_lock(fit_key):
        fitted = _fit_or_load(
            out,
            job["kind"],
            views,
            weights,
            weightmat,
            ids,
            _content_scheme(job["scheme"], views),
        )
    scores = _score_embedding(data, fitted["embedding"], triplets, k)
    diversity = _diversity_row(members, embeddings, k)
    per_method = members.groupby("method").size().to_dict()
    saved = out / "consensus" / f"{job['run_name']}.npy"
    saved.parent.mkdir(parents=True, exist_ok=True)
    np.save(saved, fitted["embedding"])
    return {
        **job,
        "replication": int(replication),
        "data_seed": data_seed_from_frame(frame),
        "n_views": len(views),
        "method_counts": json.dumps({str(key): int(value) for key, value in per_method.items()}),
        "consensus_stress": fitted["stress"],
        "consensus_seconds": fitted["seconds"],
        "niter": fitted["niter"],
        "reused_fit": fitted["reused_fit"],
        "digest": fitted["digest"],
        "error": "",
        **diversity,
        **scores,
    }


def data_seed_from_frame(frame: pd.DataFrame):
    if "data_seed" in frame.columns and len(frame):
        return int(frame["data_seed"].iloc[0])
    return np.nan


def _common_knowledge(config, replication, frame, embeddings, data, triplets, k) -> None:
    out = replication_dir(config, replication)
    sweep_path = out / "sweep.csv"
    existing = pd.read_csv(sweep_path) if sweep_path.exists() else pd.DataFrame()
    if len(existing) and (existing["run_name"] == "common_knowledge_umap").any():
        prior = existing.loc[existing["run_name"].eq("common_knowledge_umap")].iloc[0]
        if not _failed(prior.to_dict()):
            return
    from redr.checkpoint2 import select_members

    marked = select_members(frame, "native", float(config["primary_epsilon"]))
    umap_set = marked.loc[marked["method"].eq("umap")]
    row = {
        "run_name": "common_knowledge_umap",
        "criterion": "common_knowledge",
        "epsilon": float(config["primary_epsilon"]),
        "scheme": "rashomon_umap_graph",
        "kind": "umap_refit",
        "sample_seed": 0,
        "replication": int(replication),
        "n_views": int(len(umap_set)),
        "error": "",
    }
    if len(umap_set) < 2:
        row["error"] = "fewer than 2 UMAP members"
    else:
        try:
            views = [embeddings[i] for i in umap_set["embedding_id"]]
            edges = candidate_edges(data["X"], k=30)
            refined = mean_score_edges(views, edges, keep_per_point=15)
            graph = edges_to_graph(refined, int(data["X"].shape[0]))
            embedding, seconds = fit_cne(
                data["X"],
                data["X_pca"],
                "umap",
                seed=0,
                label_weight=0.0,
                n_epochs=int(config["cne_epochs"]),
                k=int(config["k_neighbors"]),
                graph=graph,
            )
            np.save(out / "consensus" / "common_knowledge_umap.npy", embedding)
            row.update(_score_embedding(data, embedding, triplets, k))
            row["consensus_seconds"] = seconds
            row["train_seconds"] = seconds
        except Exception as exc:
            traceback.print_exc()
            row["error"] = f"{type(exc).__name__}: {exc}"
    if len(existing) and (existing["run_name"] == "common_knowledge_umap").any():
        existing = existing.loc[~existing["run_name"].eq("common_knowledge_umap")]
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    updated.to_csv(sweep_path, index=False)
    _print_row(row)


def summarize(config: dict, replications: list[int]) -> pd.DataFrame:
    lock = load_lock(config)
    metric = lock["primary_metric"]
    frames = []
    candidate_frames = []
    for replication in replications:
        out = replication_dir(config, replication)
        sweep = pd.read_csv(out / "sweep.csv")
        sweep["replication"] = replication
        frames.append(sweep)
        candidates = pd.read_csv(out / "candidates.csv")
        candidate_frames.append(candidates)
    sweep = pd.concat(frames, ignore_index=True)
    candidates = pd.concat(candidate_frames, ignore_index=True)
    sweep.to_csv(Path(config["out"]) / "sweep_all.csv", index=False)
    candidates.to_csv(Path(config["out"]) / "candidates_all.csv", index=False)
    paired = _paired_table(sweep, candidates, metric, float(config["primary_epsilon"]))
    detail = paired.attrs.pop("detail")
    detail.to_csv(Path(config["out"]) / "paired_detail.csv", index=False)
    paired.to_csv(Path(config["out"]) / "paired_comparisons.csv", index=False)
    _plot_epsilon(sweep, metric, Path(config["out"]) / "figures")
    _plot_paired(paired, metric, Path(config["out"]) / "figures")
    _plot_example(config, replications[0], Path(config["out"]) / "figures")
    keep = ["algorithm", "comparison", "mean", "std", "difference_mean", "rashomon_wins", "comparator_wins", "n"]
    print(paired[keep].to_string(index=False))
    return paired


def _paired_table(sweep, candidates, metric, epsilon: float) -> pd.DataFrame:
    rows = []
    label = _eps_label(epsilon)
    specs = [
        ("locomds", "rashomon_uniform", f"native_uniform_locomds_eps{label}", "single"),
        ("locomds", "one_per_method", "conventional_native_locomds", "single"),
        ("locomds", "unrestricted_uniform", "native_uniform_locomds_epsinf", "single"),
        ("locomds", "unrestricted_method_balanced", "native_method_balanced_locomds_epsinf", "single"),
        ("locomds", "unrestricted_gibbs", "native_gibbs_locomds_epsinf", "single"),
        ("locomds", "rashomon_method_balanced", f"native_method_balanced_locomds_eps{label}", "single"),
        ("locomds", "rashomon_gibbs", f"native_gibbs_locomds_eps{label}", "single"),
        ("locomds", "matched_uniform", f"matched_uniform_locomds_eps{label}_s", "prefix"),
        ("locomds", "random_one", "random_one_locomds_s", "prefix"),
        ("comds", "rashomon_uniform", f"native_uniform_comds_eps{label}", "single"),
        ("comds", "one_per_method", "conventional_native_comds", "single"),
        ("comds", "unrestricted_uniform", "native_uniform_comds_epsinf", "single"),
        ("comds", "unrestricted_method_balanced", "native_method_balanced_comds_epsinf", "single"),
        ("comds", "unrestricted_gibbs", "native_gibbs_comds_epsinf", "single"),
        ("comds", "rashomon_method_balanced", f"native_method_balanced_comds_eps{label}", "single"),
        ("comds", "rashomon_gibbs", f"native_gibbs_comds_eps{label}", "single"),
        ("comds", "matched_uniform", f"matched_uniform_comds_eps{label}_s", "prefix"),
        ("comds", "random_one", "random_one_comds_s", "prefix"),
        ("aggregator", "procrustes_full", "all_procrustes_mean", "single"),
        ("aggregator", "distance_mds_full", "all_distance_mds", "single"),
        ("aggregator", "procrustes_rashomon", f"rashomon_procrustes_mean_eps{label}", "single"),
        ("aggregator", "distance_mds_rashomon", f"rashomon_distance_mds_eps{label}", "single"),
        ("refit", "common_knowledge_umap", "common_knowledge_umap", "single"),
    ]
    reference_name = {
        "locomds": f"native_uniform_locomds_eps{label}",
        "comds": f"native_uniform_comds_eps{label}",
    }
    for replication, group in sweep.groupby("replication"):
        candidate_group = candidates.loc[candidates["replication"].eq(replication)]
        oracle = float(candidate_group[metric].max())
        rows.append(
            {
                "replication": int(replication),
                "algorithm": "oracle_individual",
                "comparison": "best_candidate",
                "value": oracle,
                "reference": np.nan,
                "difference": np.nan,
                "within_sd": np.nan,
                "role": "oracle",
            }
        )
        for algorithm, comparison, name, how in specs:
            value, within_sd = _lookup(group, name, how, metric)
            if algorithm in reference_name:
                reference = _lookup(group, reference_name[algorithm], "single", metric)[0]
                difference = value - reference
            else:
                reference = np.nan
                difference = np.nan
            rows.append(
                {
                    "replication": int(replication),
                    "algorithm": algorithm,
                    "comparison": comparison,
                    "value": value,
                    "reference": reference,
                    "difference": difference,
                    "within_sd": within_sd,
                    "role": "confirmatory" if comparison != "best_candidate" else "oracle",
                }
            )
    detail = pd.DataFrame(rows)
    summary_rows = []
    for (algorithm, comparison), group in detail.groupby(["algorithm", "comparison"]):
        values = group["value"].to_numpy(dtype=float)
        differences = group["difference"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "algorithm": algorithm,
                "comparison": comparison,
                "n": int(np.isfinite(values).sum()),
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values, ddof=1)) if np.isfinite(values).sum() > 1 else np.nan,
                "median": float(np.nanmedian(values)),
                "difference_mean": float(np.nanmean(differences)) if np.isfinite(differences).any() else np.nan,
                "difference_std": float(np.nanstd(differences, ddof=1))
                if np.isfinite(differences).sum() > 1
                else np.nan,
                "comparator_wins": int(np.nansum(differences > 0)),
                "rashomon_wins": int(np.nansum(differences < 0)),
                "ties": int(np.nansum(np.isfinite(differences) & (differences == 0))),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.attrs["detail"] = detail
    return summary


def _lookup(group: pd.DataFrame, name: str, how: str, metric: str):
    if how == "single":
        hit = group.loc[group["run_name"].eq(name)]
        if len(hit) == 0 or metric not in hit.columns:
            return np.nan, np.nan
        return float(hit.iloc[0][metric]), np.nan
    hit = group.loc[group["run_name"].astype(str).str.startswith(name)]
    values = hit[metric].to_numpy(dtype=float) if metric in hit.columns else np.array([])
    if len(values) == 0:
        return np.nan, np.nan
    std = float(np.nanstd(values, ddof=1)) if np.isfinite(values).sum() > 1 else np.nan
    return float(np.nanmean(values)), std


def _plot_epsilon(sweep: pd.DataFrame, metric: str, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        metric,
        "arc_triplet_accuracy",
        "height_triplet_accuracy",
        "triplet_accuracy",
        "trustworthiness",
    ]
    metrics = list(dict.fromkeys(metrics))
    fig, axes = plt.subplots(len(metrics), 2, figsize=(9.5, 2.4 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = np.array([axes])
    for column, kind in enumerate(("comds", "locomds")):
        part = sweep.loc[
            sweep["run_name"].str.startswith(f"native_uniform_{kind}_eps")
            & ~sweep["run_name"].str.contains("_s")
        ].copy()
        part["eps_plot"] = part["epsilon"].map(lambda value: 2.0 if np.isinf(value) else value)
        for row_i, score in enumerate(metrics):
            axis = axes[row_i, column]
            grouped = part.groupby("eps_plot")[score]
            xs = np.array(sorted(grouped.mean().index))
            mean = grouped.mean().loc[xs]
            std = grouped.std(ddof=1).loc[xs]
            axis.plot(xs, mean, marker="o", color="#045a8d")
            axis.fill_between(xs, mean - std, mean + std, color="#045a8d", alpha=0.2)
            axis.set_title(f"{kind} {score}")
            axis.set_xscale("symlog", linthresh=0.001)
            axis.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("epsilon (2 denotes infinity)")
    axes[-1, 1].set_xlabel("epsilon (2 denotes infinity)")
    fig.tight_layout()
    fig.savefig(fig_dir / "epsilon_intrinsic_curves.png", dpi=160)
    plt.close(fig)


def _plot_paired(summary: pd.DataFrame, metric: str, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    part = summary.loc[
        summary["algorithm"].isin(["comds", "locomds"])
        & summary["difference_mean"].notna()
        & ~summary["comparison"].eq("rashomon_uniform")
    ].copy()
    if len(part) == 0:
        return
    part["label"] = part["algorithm"] + " " + part["comparison"]
    part = part.sort_values("difference_mean")
    fig, axis = plt.subplots(figsize=(8, 5.5))
    colors = ["#b2182b" if value > 0 else "#2166ac" for value in part["difference_mean"]]
    axis.barh(part["label"], part["difference_mean"], color=colors)
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel(f"mean paired difference on {metric}\n(comparator minus uniform Rashomon at epsilon 0.05)")
    axis.set_title("Positive means the comparator beat uniform Rashomon")
    fig.tight_layout()
    fig.savefig(fig_dir / "paired_differences.png", dpi=160)
    plt.close(fig)


def _plot_example(config: dict, replication: int, fig_dir: Path) -> None:
    out = replication_dir(config, replication)
    manifold = np.load(out / "manifold.npz")
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.2))
    names = [
        (f"conventional_native_locomds", "LoCoMDS one per method"),
        (f"native_uniform_locomds_eps{_eps_label(config['primary_epsilon'])}", "LoCoMDS Rashomon 0.05"),
        (f"native_uniform_locomds_epsinf", "LoCoMDS unrestricted"),
    ]
    for column, (path_name, title) in enumerate(names):
        path = out / "consensus" / f"{path_name}.npy"
        if not path.exists():
            continue
        embedding = np.load(path)
        axes[0, column].scatter(
            embedding[:, 0], embedding[:, 1], c=manifold["arc_length"], s=8, cmap="viridis"
        )
        axes[0, column].set_title(title + " by arc length")
        axes[1, column].scatter(
            embedding[:, 0], embedding[:, 1], c=manifold["height"], s=8, cmap="cividis"
        )
        axes[1, column].set_title(title + " by height")
        for axis in (axes[0, column], axes[1, column]):
            axis.set_xticks([])
            axis.set_yticks([])
    fig.tight_layout()
    fig.savefig(fig_dir / f"rep{replication:02d}_locomds_gallery.png", dpi=160)
    plt.close(fig)
