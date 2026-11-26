"""Train a fixed grid and describe which Checkpoint 7 regime it occupies.

Native loss is computed inside one (method, k, lambda) family. The primary
consensus pool is k = 15 and lambda = 0 for every method. Other k and lambda
values are separate objectives and stay out of that band. Trimming uses
1 - trustworthiness, which is comparable across methods. Latent coordinates
and digit labels are written to evaluation columns only.

    PYTHONPATH=redr python redr/scripts/run_natural_diversity.py
    PYTHONPATH=redr python redr/scripts/run_natural_diversity.py --benchmark
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "redr"))

from redr.comds_bridge import fit_consensus  # noqa: E402
from redr.diversity import pairwise_distance_disagreement, pairwise_neighborhood_overlap, within_between_overlap  # noqa: E402
from redr.intrinsic import intrinsic_recovery  # noqa: E402
from redr.loss_validation import contrastive_reference_loss, pacmap_component_losses  # noqa: E402
from redr.manifolds import generate_manifold, standardize_for_training  # noqa: E402
from redr.metrics import _sample_triplets, continuity, neighborhood_preservation, trustworthiness  # noqa: E402
from redr.natural_diversity import classify_regime, load_config, primary_pool, relative_span  # noqa: E402
from redr.quality_diversity import retained, scheme_weights, size_matched, trimmed  # noqa: E402
from redr.rashomon import gibbs_weights  # noqa: E402
from redr.train import fit_cne, fit_pacmap, high_d_neighbors, pacmap_pairs  # noqa: E402


ROOT = REPO / "results" / "checkpoint8" / "natural_diversity"


def _digits(n: int, seed: int) -> dict:
    bunch = load_digits()
    rng = np.random.RandomState(seed)
    chosen = []
    per_class = max(n // 10, 1)
    for label in range(10):
        pool = np.flatnonzero(bunch.target == label)
        take = min(per_class, len(pool))
        chosen.append(rng.choice(pool, size=take, replace=False))
    index = np.sort(np.concatenate(chosen))
    raw = bunch.data[index].astype(np.float64)
    observed, pca = standardize_for_training(raw)
    labels = bunch.target[index].astype(int)
    return {
        "name": "digits",
        "X": observed,
        "X_pca": pca,
        "intrinsic": None,
        "labels": labels,
        "n": int(observed.shape[0]),
    }


def _dataset(spec: dict, config: dict) -> dict:
    if spec["kind"] == "digits":
        return _digits(int(config["digits_n"]), int(spec["seed"]))
    data = generate_manifold(spec["kind"], int(config["n"]), float(spec["noise"]), int(spec["seed"]))
    data["labels"] = None
    return data


def _jobs(config: dict) -> list[dict]:
    jobs = []
    for method in config["methods"]:
        jobs.append({**method, "k": int(config["primary_k"]), "lambda": float(config["primary_lambda"]), "seed": None})
    for k in config["extra_umap_k"]:
        umap = next(item for item in config["methods"] if item["name"] == "umap")
        jobs.append({**umap, "k": int(k), "lambda": float(config["primary_lambda"]), "seed": None})
    for method_name in config["lambda_slice"]["methods"]:
        method = next(item for item in config["methods"] if item["name"] == method_name)
        for seed in config["lambda_slice"]["seeds"]:
            jobs.append({**method, "k": int(config["primary_k"]), "lambda": float(config["lambda_slice"]["lambda"]), "seed": int(seed)})
    expanded = []
    for job in jobs:
        if job["seed"] is not None:
            expanded.append(job)
            continue
        for seed in config["train_seeds"]:
            expanded.append({**job, "seed": int(seed)})
    return expanded


def _fit_one(data: dict, job: dict, config: dict, dest: Path, pairs) -> tuple[np.ndarray, float]:
    if dest.exists():
        return np.load(dest), float("nan")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if job["name"] == "pacmap":
        embedding, elapsed = fit_pacmap(
            data["X"],
            data["X_pca"],
            int(job["seed"]),
            float(job["lambda"]),
            int(config["pacmap_epochs"]),
            pairs,
        )
    else:
        embedding, elapsed = fit_cne(
            data["X"],
            data["X_pca"],
            str(job["loss_mode"]),
            int(job["seed"]),
            float(job["lambda"]),
            int(config["cne_epochs"]),
            k=int(job["k"]),
            early_exaggeration=bool(job["early_exaggeration"]),
            spectrum=None if job["spectrum"] is None else float(job["spectrum"]),
        )
    np.save(dest, embedding)
    return embedding, elapsed


def _native(job: dict, embedding: np.ndarray, neighbors: dict[int, np.ndarray], pairs, n: int) -> tuple[float, float]:
    if job["name"] == "pacmap":
        parts = pacmap_component_losses(embedding, *pairs, n)
        return float(parts["native_loss"]), float(parts["native_loss_std"])
    loss, std = contrastive_reference_loss(embedding, neighbors[int(job["k"])], str(job["loss_mode"]))
    return loss, std


def _structural(matrix: np.ndarray, embedding: np.ndarray, labels: np.ndarray | None) -> dict:
    scores = {
        "trustworthiness": float(trustworthiness(matrix, embedding, n_neighbors=10)),
        "neighborhood_preservation": float(neighborhood_preservation(matrix, embedding, 10)),
        "continuity": float(continuity(matrix, embedding, 10)),
    }
    if labels is None:
        scores["eval_label_purity"] = float("nan")
        scores["eval_adjusted_rand"] = float("nan")
        return scores
    neighbor = _neighbor_purity(embedding, labels)
    scaled = StandardScaler().fit_transform(embedding)
    predicted = KMeans(n_clusters=10, n_init=10, random_state=0).fit_predict(scaled)
    scores["eval_label_purity"] = neighbor
    scores["eval_adjusted_rand"] = float(adjusted_rand_score(labels, predicted))
    return scores


def _neighbor_purity(embedding: np.ndarray, labels: np.ndarray) -> float:
    from redr.metrics import knn_indices

    neighbors = knn_indices(embedding, 10)
    hits = [float(np.mean(labels[neighbors[i]] == labels[i])) for i in range(len(labels))]
    return float(np.mean(hits))


def _describe(frame: pd.DataFrame, embeddings: dict[str, np.ndarray], config: dict) -> tuple[pd.DataFrame, dict]:
    pool = primary_pool(frame, config)
    ids = [str(value) for value in pool["embedding_id"]]
    views = [embeddings[item] for item in ids]
    distance = pairwise_distance_disagreement(views)
    overlap = pairwise_neighborhood_overlap(views, k=10)
    within_distance, between_distance = within_between_overlap(distance, pool["method"].tolist())
    within_overlap, between_overlap = within_between_overlap(overlap, pool["method"].tolist())
    rows = []
    for method, group in pool.groupby("method", sort=False):
        index = [ids.index(str(value)) for value in group["embedding_id"]]
        ordered = group.sort_values(["native_loss", "embedding_id"])
        best = ordered.iloc[0]
        worst = ordered.iloc[-1]
        rows.append(
            {
                "method": method,
                "n": len(group),
                "span": relative_span(group["native_loss"].to_numpy()),
                "mean_trust": float(group["trustworthiness"].mean()),
                "best_trust": float(best["trustworthiness"]),
                "worst_trust": float(worst["trustworthiness"]),
                "trust_tracks_loss": bool(worst["trustworthiness"] < best["trustworthiness"] - 1e-6),
                "within_disagreement": _block_mean(distance, index),
                "within_overlap": _block_mean(overlap, index),
                "between_disagreement": between_distance,
                "between_overlap": between_overlap,
                "mean_native": float(group["native_loss"].mean()),
                "seconds": float(group["seconds"].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    # The classifier's within term is the mean of the method-wise blocks.
    summary["within_disagreement"] = summary["within_disagreement"].fillna(within_distance)
    label = classify_regime(summary, config)
    label["within_overlap"] = within_overlap
    label["between_overlap"] = between_overlap
    return summary, label


def _block_mean(matrix: np.ndarray, index: list[int]) -> float:
    if len(index) < 2:
        return 0.0
    values = [matrix[i, j] for a, i in enumerate(index) for j in index[a + 1 :]]
    return float(np.mean(values)) if values else 0.0


def _fit_consensus(views: list[np.ndarray], dest: Path, kind: str, weights: np.ndarray | None) -> tuple[np.ndarray, float]:
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
    started = time.perf_counter()
    fitted = fit_consensus(views, dest, kind=kind, weights=weights)
    elapsed = float(fitted.get("seconds", time.perf_counter() - started))
    return np.asarray(fitted["embedding"], dtype=float), elapsed


def _consensus(data: dict, frame: pd.DataFrame, embeddings: dict[str, np.ndarray], config: dict, dest: Path) -> pd.DataFrame:
    pool = primary_pool(frame, config).copy()
    pool["score"] = pool["native_loss"]
    pool["trust_loss"] = 1.0 - pool["trustworthiness"]
    triplets = None
    if data["intrinsic"] is not None:
        triplets = _sample_triplets(data["n"], int(config["n_triplets"]), np.random.default_rng(int(config["datasets"][0]["seed"]) + 9000))
    specs: list[tuple] = []
    best_trust = pool.sort_values(["trust_loss", "embedding_id"]).iloc[:1]
    specs.append(("best_trust", best_trust, None, "comds"))
    for _method, group in pool.groupby("method", sort=False):
        specs.append((f"best_native_{group['method'].iloc[0]}", group.sort_values(["score", "embedding_id"]).iloc[:1], None, "comds"))
    specs.append(("unrestricted", pool, None, "comds"))
    specs.append(("unrestricted_locomds", pool, None, "locomds"))
    specs.append(("trimmed", trimmed(pool.assign(score=pool["trust_loss"]), float(config["trim_fraction"])), None, "comds"))
    specs.append(("full_gibbs", pool, gibbs_weights(pool, "score", beta=float(config["gibbs_beta"])), "comds"))
    for epsilon in config["epsilons"]:
        label = "inf" if np.isinf(float(epsilon)) else f"{float(epsilon):g}"
        specs.append((f"rashomon_{label}", retained(pool, float(epsilon), "score"), None, "comds"))
    band = retained(pool, float(config["primary_epsilon"]), "score")
    specs.append(("rashomon_balanced", band, "balanced", "comds"))
    specs.append(("rashomon_gibbs", band, "gibbs", "comds"))
    specs.append(("rashomon_locomds", band, None, "locomds"))
    match_rng = np.random.default_rng(71000)
    for draw in range(int(config["size_matched_draws"])):
        specs.append((f"size_matched_{draw}", size_matched(pool, len(band), match_rng), None, "comds"))
    rows = []
    for rule, members, scheme, kind in specs:
        ids = [str(value) for value in members["embedding_id"]]
        views = [embeddings[item] for item in ids]
        weights = None
        if isinstance(scheme, np.ndarray):
            weights = scheme
        elif scheme == "gibbs":
            weights = scheme_weights(members, "gibbs", float(config["gibbs_beta"]))
        elif scheme == "balanced":
            weights = scheme_weights(members, "balanced", 1.0)
        if len(views) == 1:
            embedding, seconds = views[0], 0.0
        else:
            embedding, seconds = _fit_consensus(views, dest / rule.replace(".", "p"), kind, weights)
        structural = _structural(data["X"], embedding, data["labels"])
        latent = {}
        if triplets is not None:
            latent = intrinsic_recovery(data["intrinsic"], embedding, triplets, k=8)
        kept_counts = members.groupby("method").size().to_dict()
        rows.append(
            {
                "dataset": data["name"],
                "rule": rule,
                "n_kept": len(members),
                "methods_kept": int(members["method"].nunique()),
                "mean_trust_kept": float(members["trustworthiness"].mean()),
                "seconds": seconds,
                **structural,
                "intrinsic_triplet_balanced": latent.get("intrinsic_triplet_balanced", float("nan")),
                "arc_triplet_accuracy": latent.get("arc_triplet_accuracy", float("nan")),
                "height_triplet_accuracy": latent.get("height_triplet_accuracy", float("nan")),
                "kept_counts": str(kept_counts),
            }
        )
        print(
            f"  {data['name']} {rule} n={len(members)} trust={structural['trustworthiness']:.3f} "
            f"balanced={rows[-1]['intrinsic_triplet_balanced']}",
            flush=True,
        )
    return pd.DataFrame(rows)


def run_dataset(spec: dict, config: dict) -> None:
    data = _dataset(spec, config)
    out = ROOT / spec["name"]
    out.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(data["X"], dtype=np.float32)
    ks = {int(config["primary_k"]), *[int(k) for k in config["extra_umap_k"]]}
    neighbors = {k: high_d_neighbors(matrix, k) for k in ks}
    pairs = pacmap_pairs(matrix, seed=0)
    catalog = []
    embeddings: dict[str, np.ndarray] = {}
    for job in _jobs(config):
        embedding_id = f"{job['name']}_k{job['k']}_l{job['lambda']:g}_s{job['seed']}"
        path = out / "embeddings" / f"{embedding_id}.npy"
        print(f"train {spec['name']} {embedding_id}", flush=True)
        embedding, elapsed = _fit_one(data, job, config, path, pairs)
        native, native_std = _native(job, embedding, neighbors, pairs, data["n"])
        structural = _structural(matrix, embedding, data["labels"])
        catalog.append(
            {
                "embedding_id": embedding_id,
                "method": job["name"],
                "loss_mode": job["loss_mode"] if job["name"] != "pacmap" else "pacmap",
                "k": int(job["k"]),
                "lambda": float(job["lambda"]),
                "seed": int(job["seed"]),
                "native_loss": native,
                "native_loss_std": native_std,
                "seconds": elapsed,
                "primary": bool(
                    int(job["k"]) == int(config["primary_k"]) and np.isclose(float(job["lambda"]), float(config["primary_lambda"]))
                ),
                **structural,
            }
        )
        embeddings[embedding_id] = embedding
        print(f"  native={native:.4f} trust={structural['trustworthiness']:.3f} sec={elapsed}", flush=True)
    frame = pd.DataFrame(catalog)
    frame.to_csv(out / "candidates.csv", index=False)
    summary, label = _describe(frame, embeddings, config)
    summary.to_csv(out / "method_summary.csv", index=False)
    pd.DataFrame([{**label, "dataset": spec["name"]}]).to_csv(out / "regime.csv", index=False)
    print(f"regime {spec['name']} {label['primary']} bad={label['bad_methods']}", flush=True)
    consensus = _consensus(data, frame, embeddings, config, out / "fits")
    consensus.to_csv(out / "consensus.csv", index=False)


def _benchmark(config: dict) -> None:
    spec = config["datasets"][0]
    data = _dataset(spec, config)
    umap = next(item for item in config["methods"] if item["name"] == "umap")
    job = {**umap, "k": int(config["primary_k"]), "lambda": 0.0, "seed": 0}
    started = time.perf_counter()
    embedding, elapsed = fit_cne(
        data["X"], data["X_pca"], "umap", 0, 0.0, int(config["cne_epochs"]),
        k=15, early_exaggeration=False,
    )
    print(f"benchmark umap epochs={config['cne_epochs']} n={data['n']} shape={embedding.shape} sec={elapsed:.1f} wall={time.perf_counter()-started:.1f}", flush=True)
    pairs = pacmap_pairs(np.asarray(data["X"], dtype=np.float32), seed=0)
    _embedding, elapsed = fit_pacmap(data["X"], data["X_pca"], 0, 0.0, int(config["pacmap_epochs"]), pairs)
    print(f"benchmark pacmap epochs={config['pacmap_epochs']} sec={elapsed:.1f}", flush=True)


def main() -> None:
    config = load_config()
    if "--benchmark" in sys.argv:
        _benchmark(config)
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    for spec in config["datasets"]:
        if (ROOT / spec["name"] / "consensus.csv").exists() and "--force" not in sys.argv:
            print(f"skip {spec['name']}", flush=True)
            continue
        run_dataset(spec, config)


if __name__ == "__main__":
    main()
