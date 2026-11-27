"""Train a cross-objective pool, describe its geometry, then aggregate it.

Native loss is scored only inside one configuration. The hierarchical
Rashomon set is the union of those within-configuration bands. If every
candidate is inside its band, the hierarchical ensemble is the full pool
and the script does not fit that ensemble a second time.

    PYTHONPATH=redr python redr/scripts/run_cross_objective.py
"""

from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE, Isomap
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "redr"))

from redr.comds_bridge import fit_consensus  # noqa: E402
from redr.cross_objective import (  # noqa: E402
    apply_membership,
    iter_jobs,
    load_config,
    neg_spec_param,
    pair_relation,
    quality_weights,
    weights_match,
)
from redr.intrinsic import intrinsic_recovery  # noqa: E402
from redr.loss_validation import contrastive_reference_loss, pacmap_component_losses  # noqa: E402
from redr.manifolds import generate_manifold, standardize_for_training  # noqa: E402
from redr.metrics import _sample_triplets, knn_indices, neighborhood_preservation, trustworthiness  # noqa: E402
from redr.quality_diversity import size_matched, trimmed  # noqa: E402
from redr.rashomon import gibbs_weights, method_balanced_weights  # noqa: E402
from redr.train import fit_cne, fit_pacmap, high_d_neighbors, pacmap_pairs  # noqa: E402


ROOT = REPO / "results" / "checkpoint9" / "cross_objective"


def _digits(n: int, seed: int) -> dict:
    bunch = load_digits()
    rng = np.random.RandomState(seed)
    per_class = max(int(n) // 10, 1)
    chosen = [rng.choice(np.flatnonzero(bunch.target == label), size=per_class, replace=False) for label in range(10)]
    index = np.sort(np.concatenate(chosen))
    observed, pca = standardize_for_training(bunch.data[index].astype(np.float64))
    return {"name": "digits", "X": observed, "X_pca": pca, "intrinsic": None, "labels": bunch.target[index].astype(int), "n": int(len(index))}


def _dataset(spec: dict, config: dict, seed: int) -> dict:
    if spec["kind"] == "digits":
        data = _digits(int(config["digits_n"]), seed)
    else:
        data = generate_manifold(spec["kind"], int(config["n"]), float(spec["noise"]), seed)
        data["labels"] = None
    nuisance = int(spec.get("nuisance") or 0)
    if nuisance:
        extra = np.random.RandomState(seed + 17).normal(size=(data["n"], nuisance))
        observed, pca = standardize_for_training(np.hstack([data["X"], extra]))
        data = dict(data)
        data["X"] = observed
        data["X_pca"] = pca
    data["name"] = spec["name"]
    data["data_seed"] = int(seed)
    return data


def _native(job: dict, embedding: np.ndarray, matrix: np.ndarray, neighbors: dict, pairs: dict, config: dict) -> tuple[float, float]:
    if job["kind"] == "cne":
        spec = 1.0
        if job["loss_mode"] == "neg":
            spec = neg_spec_param(matrix.shape[0], float(job["spectrum"]), int(config["negative_samples"]))
        return contrastive_reference_loss(
            embedding,
            neighbors[int(job["k"])],
            str(job["loss_mode"]),
            negative_samples=int(config["negative_samples"]),
            n_draws=int(config["n_draws"]),
            base_seed=int(config["reference_seed"]),
            spec_param=spec,
        )
    if job["kind"] == "pacmap":
        parts = pacmap_component_losses(embedding, *pairs[int(job["n_neighbors"])], matrix.shape[0])
        return float(parts["native_loss"]), 0.0
    if job["kind"] == "tsne":
        return float(job["kl"]), 0.0
    return 0.0, 0.0


def _structural(matrix: np.ndarray, embedding: np.ndarray, labels, ks: list[int]) -> dict:
    scores = {
        "trustworthiness": float(trustworthiness(matrix, embedding, n_neighbors=10)),
        "neighborhood_preservation": float(neighborhood_preservation(matrix, embedding, 10)),
    }
    for k in ks:
        scores[f"preservation_k{k}"] = float(neighborhood_preservation(matrix, embedding, int(k)))
    if labels is None:
        scores["eval_label_purity"] = float("nan")
        scores["eval_adjusted_rand"] = float("nan")
        return scores
    neighbor = knn_indices(embedding, 10)
    scores["eval_label_purity"] = float(np.mean([np.mean(labels[neighbor[i]] == labels[i]) for i in range(len(labels))]))
    predicted = KMeans(n_clusters=10, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(embedding))
    scores["eval_adjusted_rand"] = float(adjusted_rand_score(labels, predicted))
    return scores


def _fit_job(data: dict, job: dict, config: dict, dest: Path, pairs: dict) -> tuple[np.ndarray, float, float]:
    if dest.exists():
        kl = np.nan
        if job["kind"] == "tsne" and dest.with_suffix(".kl").exists():
            kl = float(dest.with_suffix(".kl").read_text())
        return np.load(dest), float("nan"), kl
    dest.parent.mkdir(parents=True, exist_ok=True)
    kl = float("nan")
    if job["kind"] == "cne":
        embedding, elapsed = fit_cne(
            data["X"], data["X_pca"], str(job["loss_mode"]), int(job["seed"]), 0.0, int(config["cne_epochs"]),
            k=int(job["k"]), early_exaggeration=bool(job["early_exaggeration"]),
            spectrum=None if job["spectrum"] is None else float(job["spectrum"]),
        )
    elif job["kind"] == "pacmap":
        embedding, elapsed = fit_pacmap(
            data["X"], data["X_pca"], int(job["seed"]), 0.0, int(config["pacmap_epochs"]),
            pairs[int(job["n_neighbors"])], n_neighbors=int(job["n_neighbors"]),
            n_mn=int(config["pacmap_mn"]), n_fp=int(config["pacmap_fp"]),
        )
    elif job["kind"] == "tsne":
        started = time.perf_counter()
        model = TSNE(
            n_components=2, perplexity=float(job["perplexity"]), random_state=int(job["seed"]),
            init="pca", learning_rate="auto", max_iter=int(config["tsne_max_iter"]),
        )
        embedding = np.asarray(model.fit_transform(np.asarray(data["X"], dtype=np.float64)), dtype=np.float64)
        elapsed = time.perf_counter() - started
        kl = float(model.kl_divergence_)
        dest.with_suffix(".kl").write_text(str(kl))
    elif job["kind"] == "isomap":
        started = time.perf_counter()
        embedding = np.asarray(Isomap(n_neighbors=int(job["k"]), n_components=2).fit_transform(data["X"]), dtype=np.float64)
        elapsed = time.perf_counter() - started
    else:
        started = time.perf_counter()
        embedding = np.asarray(data["X_pca"], dtype=np.float64)
        elapsed = time.perf_counter() - started
    np.save(dest, embedding)
    return embedding, float(elapsed), kl


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    k = left.shape[1]
    total = 0.0
    for row_left, row_right in zip(left, right):
        inter = len(set(row_left.tolist()) & set(row_right.tolist()))
        total += inter / (2 * k - inter)
    return float(total / len(left))


def _diversity(frame: pd.DataFrame, embeddings: dict[str, np.ndarray], matrix: np.ndarray, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = [str(value) for value in frame["embedding_id"]]
    views = [embeddings[item] for item in ids]
    ks = [int(k) for k in config["overlap_ks"]]
    knn = {k: [knn_indices(view, k) for view in views] for k in ks}
    upper = []
    for view in views:
        distance = pairwise_distances(np.asarray(view, dtype=np.float32))
        values = distance[np.triu_indices(distance.shape[0], 1)].astype(np.float64)
        upper.append(values / max(float(values.mean()), 1e-12))
    ranks = [rankdata(values) for values in upper]
    local_graph = knn_indices(matrix, int(config["local_k"]))
    local = np.zeros((matrix.shape[0], matrix.shape[0]), dtype=bool)
    local[np.repeat(np.arange(matrix.shape[0]), local_graph.shape[1]), local_graph.ravel()] = True
    local = np.triu(local | local.T, 1)
    local_index = np.flatnonzero(local[np.triu_indices(matrix.shape[0], 1)])
    rows = []
    methods = frame["method"].tolist()
    configs = frame["config_id"].tolist()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            row = {
                "left": ids[i],
                "right": ids[j],
                "relation": pair_relation(methods[i], configs[i], methods[j], configs[j]),
                "global_disagreement": float(np.mean(np.abs(upper[i] - upper[j]))),
                "local_disagreement": float(np.mean(np.abs(upper[i][local_index] - upper[j][local_index]))),
                "rank_correlation": _corr(ranks[i], ranks[j]),
            }
            for k in ks:
                row[f"overlap_k{k}"] = _jaccard(knn[k][i], knn[k][j])
            rows.append(row)
    pairs = pd.DataFrame(rows)
    summary = pairs.groupby("relation", sort=False).mean(numeric_only=True).reset_index()
    redundant = int((pairs["overlap_k15"] >= float(config["redundancy_overlap"])).sum()) if "overlap_k15" in pairs else 0
    summary["redundant_pairs_k15"] = redundant
    return pairs, summary


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean()
    right = right - right.mean()
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return 1.0
    return float(np.dot(left, right) / denom)


def _consensus_fit(views: list[np.ndarray], dest: Path, kind: str, weights: np.ndarray | None) -> tuple[np.ndarray, float]:
    dest.mkdir(parents=True, exist_ok=True)
    cached = dest / "gspace.csv"
    if cached.exists():
        seconds = float("nan")
        meta = dest / "meta.txt"
        if meta.exists():
            for line in meta.read_text().splitlines():
                if line.startswith("seconds="):
                    seconds = float(line.split("=", 1)[1])
        return pd.read_csv(cached, header=None).to_numpy(dtype=float), seconds
    fitted = fit_consensus(views, dest, kind=kind, weights=None if weights is None else np.asarray(weights, dtype=float))
    return np.asarray(fitted["embedding"], dtype=float), float(fitted.get("seconds", np.nan))


def _score_embedding(data: dict, embedding: np.ndarray, config: dict, triplets) -> dict:
    structural = _structural(np.asarray(data["X"]), embedding, data["labels"], [int(k) for k in config["overlap_ks"]])
    if data["intrinsic"] is None or triplets is None:
        structural.update({key: float("nan") for key in ("intrinsic_triplet_balanced", "intrinsic_triplet_isometric", "arc_triplet_accuracy", "height_triplet_accuracy")})
        return structural
    structural.update(intrinsic_recovery(data["intrinsic"], embedding, triplets, k=10))
    return structural


def _group_weights(frame: pd.DataFrame, column: str) -> np.ndarray:
    labeled = frame.copy()
    labeled["method"] = labeled[column].astype(str)
    return method_balanced_weights(labeled)


def _native_gibbs(frame: pd.DataFrame, beta: float) -> np.ndarray:
    labeled = frame.copy()
    labeled["method"] = labeled["config_id"].astype(str)
    labeled["score"] = labeled["native_loss"].to_numpy(dtype=float)
    return gibbs_weights(labeled, "score", beta=beta)


def _consensus(data: dict, frame: pd.DataFrame, embeddings: dict[str, np.ndarray], config: dict, dest: Path, triplets) -> pd.DataFrame:
    pool = frame.copy()
    pool["score"] = 1.0 - pool["trustworthiness"]
    marked, inactive, widest = apply_membership(pool, float(config["epsilon"]))
    specs: list[tuple] = []
    specs.append(("best_trust", pool.sort_values(["score", "embedding_id"]).iloc[:1], None, "comds", ""))
    for method, group in pool.loc[pool["is_default"].astype(bool)].groupby("method", sort=False):
        specs.append((f"baseline_{method}", group.sort_values(["native_loss", "embedding_id"]).iloc[:1], None, "comds", ""))
    if data["intrinsic"] is not None:
        oracle = pool.assign(_oracle=-pool["height_triplet_accuracy"]).sort_values(["_oracle", "embedding_id"]).iloc[:1]
        specs.append(("oracle_best_height", oracle, None, "comds", "oracle"))
    fixed = pool.loc[pool["is_default"].astype(bool)]
    one_each = pool.sort_values(["native_loss", "embedding_id"]).groupby("config_id", sort=False).head(1)
    full_weights = {
        "full_comds": None,
        "full_method_balanced": _group_weights(pool, "method"),
        "full_config_balanced": _group_weights(pool, "config_id"),
        "full_gibbs_native": _native_gibbs(pool, float(config["gibbs_beta"])),
        "full_quality": quality_weights(pool["score"].to_numpy(), float(config["gibbs_beta"])),
    }
    specs.append(("fixed_objective_comds", fixed, None, "comds", "same objective, all seeds"))
    specs.append(("fixed_objective_locomds", fixed, None, "locomds", "same objective, all seeds"))
    conventional = fixed.sort_values(["native_loss", "embedding_id"]).groupby("method", sort=False).head(1)
    specs.append(("conventional_comds", conventional, None, "comds", "one default configuration per algorithm"))
    specs.append(("conventional_locomds", specs[-1][1], None, "locomds", "one default configuration per algorithm"))
    specs.append(("one_seed_per_config", one_each, None, "comds", "every objective, one seed"))
    for name, weights in full_weights.items():
        specs.append((name, pool, weights, "comds", "all candidates"))
    specs.append(("full_locomds", pool, None, "locomds", "all candidates"))
    trimmed_pool = trimmed(pool, float(config["trim_fraction"]))
    specs.append(("trimmed_trust", trimmed_pool, None, "comds", "drop the worst quarter by trustworthiness"))
    if inactive:
        note = "Rashomon restriction inactive: every candidate is inside its configuration band"
    else:
        note = "within-configuration band"
        specs.append(("hierarchical_comds", marked.loc[marked["in_rashomon"]], None, "comds", note))
        specs.append(("hierarchical_locomds", marked.loc[marked["in_rashomon"]], None, "locomds", note))
        specs.append(("hierarchical_gibbs", marked.loc[marked["in_rashomon"]], "native_gibbs", "comds", note))
    rng = np.random.default_rng(91000 + int(data["data_seed"]))
    conventional_n = int(fixed["method"].nunique())
    for draw in range(int(config["size_matched_draws"])):
        specs.append((f"size_matched_conventional_{draw}", size_matched(pool, conventional_n, rng), None, "comds", "matched to one embedding per algorithm"))
        specs.append((f"size_matched_one_config_{draw}", size_matched(pool, len(one_each), rng), None, "comds", "matched to one seed per configuration"))
    rows = []
    seen_embeddings: dict[tuple, tuple[np.ndarray, float, str]] = {}
    for rule, members, scheme, kind, description in specs:
        members = members.copy()
        ids = [str(value) for value in members["embedding_id"]]
        views = [embeddings[item] for item in ids]
        weights = None
        identical = ""
        if isinstance(scheme, str) and scheme == "native_gibbs":
            weights = _native_gibbs(members, float(config["gibbs_beta"]))
        elif isinstance(scheme, np.ndarray):
            weights = scheme
        if rule == "full_gibbs_native" and weights_match(weights, full_weights["full_config_balanced"]):
            identical = "full_config_balanced"
        if rule == "hierarchical_comds" and inactive:
            identical = "full_comds"
        if len(views) == 1:
            embedding, seconds = views[0], 0.0
        elif identical and identical in {row["rule"] for row in rows}:
            previous = next(row for row in rows if row["rule"] == identical)
            rows.append({**previous, "rule": rule, "identical_to": identical, "description": description, "rashomon_inactive": inactive, "widest_span": widest})
            print(f"  {data['name']} {rule} identical to {identical}", flush=True)
            continue
        else:
            key = (tuple(ids), kind, None if weights is None else tuple(np.round(weights, 5)))
            if key in seen_embeddings:
                embedding, seconds, source = seen_embeddings[key]
                identical = source
            else:
                embedding, seconds = _consensus_fit(views, dest / rule, kind, weights)
                seen_embeddings[key] = (embedding, seconds, rule)
        scored = _score_embedding(data, embedding, config, triplets)
        rows.append(
            {
                "dataset": data["name"],
                "data_seed": data["data_seed"],
                "rule": rule,
                "description": description,
                "identical_to": identical,
                "n_kept": len(members),
                "n_algorithms": int(members["method"].nunique()),
                "n_configs": int(members["config_id"].nunique()),
                "rashomon_inactive": inactive,
                "widest_span": widest,
                "seconds": seconds,
                "mean_trust_kept": float(members["trustworthiness"].mean()),
                **{key: scored[key] for key in ("trustworthiness", "neighborhood_preservation", "preservation_k5", "preservation_k15", "preservation_k30", "intrinsic_triplet_balanced", "intrinsic_triplet_isometric", "arc_triplet_accuracy", "height_triplet_accuracy", "eval_label_purity", "eval_adjusted_rand")},
            }
        )
        print(
            f"  {data['name']} {rule} n={len(members)} trust={scored['trustworthiness']:.3f} "
            f"balanced={scored['intrinsic_triplet_balanced']} height={scored['height_triplet_accuracy']}",
            flush=True,
        )
    return pd.DataFrame(rows)


def run_replication(spec: dict, seed: int, config: dict) -> None:
    data = _dataset(spec, config, seed)
    out = ROOT / spec["name"] / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    if data["intrinsic"] is not None:
        np.savez(out / "latent.npz", intrinsic=data["intrinsic"])
    train_seeds = spec.get("train_seeds") or config["train_seeds"]
    jobs = iter_jobs(config, list(train_seeds))
    matrix = np.asarray(data["X"], dtype=np.float32)
    ks = sorted({int(job["k"]) for job in jobs if job["k"] is not None and job["kind"] == "cne"})
    neighbors = {k: high_d_neighbors(matrix, k) for k in ks}
    neighbor_counts = sorted({int(job["n_neighbors"]) for job in jobs if job["kind"] == "pacmap"})
    pairs = {
        count: pacmap_pairs(matrix, seed=0, n_neighbors=count, n_mn=int(config["pacmap_mn"]), n_fp=int(config["pacmap_fp"]))
        for count in neighbor_counts
    }
    catalog_path = out / "candidates.csv"
    embeddings: dict[str, np.ndarray] = {}
    if catalog_path.exists():
        frame = pd.read_csv(catalog_path)
        for embedding_id in frame["embedding_id"]:
            embeddings[str(embedding_id)] = np.load(out / "embeddings" / f"{embedding_id}.npy")
        print(f"reload {spec['name']} seed {seed} n={len(frame)}", flush=True)
    else:
        catalog = []
        for job in jobs:
            embedding_id = f"{job['config_id'].replace('|', '__').replace('=', '')}_seed{job['seed']}"
            path = out / "embeddings" / f"{embedding_id}.npy"
            print(f"train {spec['name']} seed{seed} {embedding_id}", flush=True)
            embedding, elapsed, kl = _fit_job(data, {**job, "kl": np.nan}, config, path, pairs)
            job = {**job, "kl": kl}
            native, native_std = _native(job, embedding, matrix, neighbors, pairs, config)
            structural = _structural(matrix, embedding, data["labels"], [int(k) for k in config["overlap_ks"]])
            latent = {}
            catalog.append(
                {
                    "embedding_id": embedding_id,
                    "method": job["method"],
                    "kind": job["kind"],
                    "config_id": job["config_id"],
                    "loss_mode": job["loss_mode"] if job["loss_mode"] is not None else job["kind"],
                    "k": job["k"] if job["k"] is not None else np.nan,
                    "n_neighbors": job["n_neighbors"] if job["n_neighbors"] is not None else np.nan,
                    "perplexity": job["perplexity"] if job["perplexity"] is not None else np.nan,
                    "spectrum": job["spectrum"] if job["spectrum"] is not None else np.nan,
                    "seed": int(job["seed"]),
                    "is_default": bool(job["is_default"]),
                    "epochs": int(config["cne_epochs"] if job["kind"] == "cne" else config["pacmap_epochs"] if job["kind"] == "pacmap" else config["tsne_max_iter"] if job["kind"] == "tsne" else 0),
                    "native_loss": native,
                    "native_loss_std": native_std,
                    "loss_comparable": job["kind"] in {"cne", "pacmap", "tsne"},
                    "seconds": elapsed,
                    **structural,
                    **latent,
                }
            )
            embeddings[embedding_id] = embedding
            print(f"  native={native:.4f} trust={structural['trustworthiness']:.3f} sec={elapsed}", flush=True)
        frame = pd.DataFrame(catalog)
        triplets = None
        if data["intrinsic"] is not None:
            triplets = _sample_triplets(data["n"], int(config["n_triplets"]), np.random.default_rng(8000 + seed))
            np.save(out / "triplets.npy", triplets)
            for index, row in frame.iterrows():
                recovery = intrinsic_recovery(data["intrinsic"], embeddings[str(row["embedding_id"])], triplets, k=10)
                for key, value in recovery.items():
                    frame.loc[index, key] = value
        frame.to_csv(catalog_path, index=False)
    triplets = None
    if (out / "triplets.npy").exists():
        triplets = np.load(out / "triplets.npy")
    elif data["intrinsic"] is not None:
        triplets = _sample_triplets(data["n"], int(config["n_triplets"]), np.random.default_rng(8000 + seed))
        np.save(out / "triplets.npy", triplets)
        for index, row in frame.iterrows():
            recovery = intrinsic_recovery(data["intrinsic"], embeddings[str(row["embedding_id"])], triplets, k=10)
            for key, value in recovery.items():
                frame.loc[index, key] = value
        frame.to_csv(catalog_path, index=False)
    if not (out / "diversity_summary.csv").exists():
        pairs, summary = _diversity(frame, embeddings, np.asarray(data["X"]), config)
        pairs.to_csv(out / "pairs.csv", index=False)
        summary.to_csv(out / "diversity_summary.csv", index=False)
        marked, inactive, widest = apply_membership(frame, float(config["epsilon"]))
        aggregates = {
            "n": ("embedding_id", "size"),
            "span": ("relative_span", "first"),
            "kept": ("in_rashomon", "sum"),
            "mean_trust": ("trustworthiness", "mean"),
            "mean_preservation_k5": ("preservation_k5", "mean"),
            "mean_preservation_k30": ("preservation_k30", "mean"),
        }
        if "height_triplet_accuracy" in marked:
            aggregates["mean_height"] = ("height_triplet_accuracy", "mean")
            aggregates["mean_arc"] = ("arc_triplet_accuracy", "mean")
            aggregates["mean_balanced"] = ("intrinsic_triplet_balanced", "mean")
            aggregates["mean_isometric"] = ("intrinsic_triplet_isometric", "mean")
        marked.groupby("config_id", sort=False).agg(**aggregates).reset_index().to_csv(out / "config_summary.csv", index=False)
        print(f"diversity {spec['name']} seed{seed} inactive={inactive} widest_span={widest:.4f}", flush=True)
    if not (out / "consensus.csv").exists():
        consensus = _consensus(data, frame, embeddings, config, out / "fits", triplets)
        consensus.to_csv(out / "consensus.csv", index=False)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (out / "runtime.txt").write_text(f"max_rss_kb={rss}\n")


def main() -> None:
    config = load_config()
    names = [item for item in sys.argv[1:] if not item.startswith("-")]
    ROOT.mkdir(parents=True, exist_ok=True)
    for spec in config["datasets"]:
        if names and spec["name"] not in names:
            continue
        for seed in spec["seeds"]:
            done = ROOT / spec["name"] / f"seed{seed}" / "consensus.csv"
            if done.exists() and "--force" not in sys.argv:
                print(f"skip {spec['name']} {seed}", flush=True)
                continue
            run_replication(spec, int(seed), config)


if __name__ == "__main__":
    main()
