"""Checkpoint 6: reproduce An and Tang, then compare Rashomon consensus.

Classical views are the installed subset of the manuscript's 16 methods.
Parametric views use the vendored RashomonDR trainers and a short alignment
grid, lambda in {0, 1, 5} and two seeds. Native losses are not pooled across
methods. Latent labels and the Swiss-roll parameter are used only for scoring.

    PYTHONPATH=redr python redr/scripts/run_checkpoint6.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "redr"))

from redr.comds_bridge import fit_consensus  # noqa: E402
from redr.common_knowledge import candidate_edges, edges_to_graph, mean_score_edges  # noqa: E402
from redr.complementary import triplets_for  # noqa: E402
from redr.diversity import pairwise_neighborhood_overlap, procrustes_rmse  # noqa: E402
from redr.intrinsic import intrinsic_recovery  # noqa: E402
from redr.loss_validation import contrastive_reference_loss, pacmap_component_losses  # noqa: E402
from redr.metrics import knn_indices, lcmc, trustworthiness  # noqa: E402
from redr.paper_views import classical_views, meta_spec_kpca, mixture_of_gaussians, swiss_roll  # noqa: E402
from redr.rashomon import mark_by_loss, method_balanced_weights  # noqa: E402
from redr.train import fit_pacmap, high_d_neighbors, pacmap_pairs, set_seed  # noqa: E402


EPSILON = 0.05
CNE_EPOCHS = 50
PACMAP_EPOCHS = 200
LAMBDAS = (0.0, 1.0, 5.0)
SEEDS = (0, 1)
PARAMETRIC = (
    ("umap", "umap", None),
    ("infonce", "infonce", None),
    ("neg_tsne", "neg", 0.0),
    ("nce", "nce", None),
    ("pacmap", None, None),
)


def _pca_target(matrix: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA

    return PCA(n_components=2, random_state=0).fit_transform(matrix).astype(np.float32)


def _consensus(views: list[np.ndarray], dest: Path, kind: str, weights: np.ndarray | None = None, tau: float = 0.1, percentile: float = 0.5) -> tuple[np.ndarray, float]:
    dest.mkdir(parents=True, exist_ok=True)
    cached = dest / "gspace.csv"
    if cached.exists() and (dest / "meta.txt").exists():
        embedding = pd.read_csv(cached, header=None).to_numpy(dtype=float)
        seconds = 0.0
        for line in (dest / "meta.txt").read_text().splitlines():
            if line.startswith("seconds="):
                seconds = float(line.split("=", 1)[1])
        return embedding, seconds
    fitted = fit_consensus(views, dest, kind=kind, weights=weights, tau=tau, percentile=percentile)
    return np.asarray(fitted["embedding"], dtype=float), float(fitted["seconds"])


def _gaussian_scores(matrix: np.ndarray, labels: np.ndarray, embedding: np.ndarray) -> dict:
    scaled = StandardScaler().fit_transform(embedding)
    clusters = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(scaled)
    neighbors = knn_indices(embedding, 10)
    purity = float(np.mean([np.mean(labels[neighbors[i]] == labels[i]) for i in range(len(labels))]))
    return {
        "adjusted_rand": float(adjusted_rand_score(labels, clusters)),
        "silhouette_true": float(silhouette_score(embedding, labels)),
        "neighbor_purity": purity,
        "trustworthiness": float(trustworthiness(matrix, embedding, n_neighbors=10)),
        "intrinsic_triplet_balanced": np.nan,
        "lcmc_k20": float(lcmc(matrix, embedding, 20)),
    }


def _swiss_scores(data: dict, embedding: np.ndarray, triplets: np.ndarray) -> dict:
    recovery = intrinsic_recovery(data["intrinsic"], embedding, triplets, k=10)
    return {
        "adjusted_rand": np.nan,
        "silhouette_true": np.nan,
        "neighbor_purity": np.nan,
        "trustworthiness": float(trustworthiness(data["X"], embedding, n_neighbors=10)),
        "intrinsic_triplet_balanced": float(recovery["intrinsic_triplet_balanced"]),
        "lcmc_k20": float(lcmc(data["X"], embedding, 20)),
    }


def _row(dataset: str, name: str, family: str, n_views: int, seconds: float, scores: dict) -> dict:
    return {
        "dataset": dataset,
        "rule": name,
        "family": family,
        "n_views": n_views,
        "seconds": seconds,
        **scores,
    }


def _quality_weights(frame: pd.DataFrame) -> np.ndarray:
    """Inverse excess native loss inside each method, then equal method mass."""
    weights = np.zeros(len(frame), dtype=float)
    positions = {idx: i for i, idx in enumerate(frame.index)}
    n_methods = max(int(frame["method"].nunique()), 1)
    for _method, group in frame.groupby("method", sort=False):
        loss = group["native_loss"].to_numpy(dtype=float)
        raw = 1.0 / (loss - np.min(loss) + 0.05)
        raw = raw / np.sum(raw)
        for idx, value in zip(group.index, raw):
            weights[positions[idx]] = value / n_methods
    return weights


def _graph_members(frame: pd.DataFrame, embeddings: dict[str, np.ndarray]) -> pd.DataFrame:
    """Keep embeddings whose neighborhood agrees with the most typical one in the method.

    The threshold is the median overlap inside that method, fixed before scoring.
    """
    keep_ids = []
    for _method, group in frame.groupby("method", sort=False):
        ids = [str(value) for value in group["embedding_id"]]
        if len(ids) == 1:
            keep_ids.extend(ids)
            continue
        views = [embeddings[key] for key in ids]
        overlap = pairwise_neighborhood_overlap(views, k=15)
        mean_overlap = (overlap.sum(axis=1) - 1.0) / (len(ids) - 1)
        center = int(np.argmax(mean_overlap))
        agreement = overlap[center]
        threshold = float(np.median(agreement))
        keep_ids.extend(ids[i] for i in range(len(ids)) if agreement[i] >= threshold - 1e-12)
    return frame.loc[frame["embedding_id"].astype(str).isin(keep_ids)].copy()


def run_classical(root: Path) -> pd.DataFrame:
    rows = []
    specs = {
        "gaussian": mixture_of_gaussians(331),
        "swiss": swiss_roll(400, random_state=331),
    }
    for name, data in specs.items():
        print(f"classical views {name}", flush=True)
        started = time.perf_counter()
        views = classical_views(data["X"], random_state=0)
        print(f"  {len(views)} views in {time.perf_counter() - started:.1f}s", flush=True)
        out = root / name / "classical"
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "views.npz",
            ids=np.array(list(views)),
            embeddings=np.stack(list(views.values())),
            **{key: data[key] for key in data if key != "n"},
        )
        triplets = triplets_for(data["n"], 16000)
        scorer = (
            (lambda embedding: _gaussian_scores(data["X"], data["labels"], embedding))
            if name == "gaussian"
            else (lambda embedding: _swiss_scores(data, embedding, triplets))
        )
        for view_name, embedding in views.items():
            rows.append(_row(name, view_name, "individual", 1, np.nan, scorer(embedding)))
            print(f"  {name} {view_name}", flush=True)
        meta = meta_spec_kpca(list(views.values()))
        rows.append(_row(name, "metaspec_kpca", "original_consensus", 1, np.nan, scorer(meta)))
        ordered = list(views.values())
        comds, seconds = _consensus(ordered, out / "comds", "comds")
        rows.append(_row(name, "original_comds", "original_consensus", len(ordered), seconds, scorer(comds)))
        print(f"  {name} comds {seconds:.1f}s", flush=True)
        best_loco = None
        best_score = -np.inf
        for tau, percentile in ((0.1, 0.5), (1.0, 0.5), (0.1, 0.3)):
            label = f"locomds_tau{tau}_p{percentile}"
            embedding, seconds = _consensus(
                ordered, out / label, "locomds", tau=tau, percentile=percentile
            )
            scores = scorer(embedding)
            rows.append(_row(name, label, "original_consensus", len(ordered), seconds, scores))
            if scores["lcmc_k20"] > best_score:
                best_score = scores["lcmc_k20"]
                best_loco = (label, embedding, seconds, scores)
            print(f"  {name} {label} lcmc {scores['lcmc_k20']:.3f}", flush=True)
        if best_loco is not None:
            label, embedding, seconds, scores = best_loco
            chosen = dict(scores)
            rows.append(_row(name, f"locomds_lcmc_selected:{label}", "original_consensus", len(ordered), seconds, chosen))
        pd.DataFrame(rows).to_csv(root / "classical.csv", index=False)
    return pd.DataFrame(rows)


def _fit_parametric_one(matrix: np.ndarray, pca: np.ndarray, method: str, loss_mode: str | None, spectrum, seed: int, lam: float, dest: Path) -> np.ndarray:
    if dest.exists():
        return np.load(dest)
    if method == "pacmap":
        embedding, _elapsed = fit_pacmap(matrix, pca, seed, lam, PACMAP_EPOCHS, pacmap_pairs(matrix, seed))
    else:
        import cne

        set_seed(seed)
        model = cne.CNE(
            optimizer="adam",
            parametric=True,
            loss_mode=loss_mode,
            label_weights=float(lam),
            seed=int(seed),
            task_type="pca",
            n_epochs=CNE_EPOCHS,
            batch_size=min(128, matrix.shape[0]),
            k=15,
            negative_samples=5,
            use_keops=False,
            print_freq_epoch=None,
            early_exaggeration=loss_mode in {"infonce", "neg"},
            s=spectrum,
        )
        embedding = np.asarray(model.fit_transform(matrix.astype(np.float32), pca), dtype=np.float64)
    np.save(dest, embedding)
    return embedding


def run_parametric(root: Path) -> pd.DataFrame:
    rows = []
    datasets = {
        "gaussian": mixture_of_gaussians(331),
        "swiss": swiss_roll(400, random_state=331),
    }
    for name, data in datasets.items():
        matrix = data["X"].astype(np.float32)
        pca = _pca_target(matrix)
        neighbors = high_d_neighbors(matrix, 15)
        out = root / name / "parametric"
        out.mkdir(parents=True, exist_ok=True)
        catalog = []
        embeddings = {}
        pairs = pacmap_pairs(matrix, seed=0)
        for method, loss_mode, spectrum in PARAMETRIC:
            for seed in SEEDS:
                for lam in LAMBDAS:
                    embedding_id = f"{method}_seed{seed}_lam{lam:g}"
                    path = out / f"{embedding_id}.npy"
                    print(f"parametric {name} {embedding_id}", flush=True)
                    embedding = _fit_parametric_one(matrix, pca, method, loss_mode, spectrum, seed, lam, path)
                    embeddings[embedding_id] = embedding
                    if method == "pacmap":
                        losses = pacmap_component_losses(embedding, *pairs, matrix.shape[0])
                        native = losses["native_loss"]
                    else:
                        native, _std = contrastive_reference_loss(embedding, neighbors, loss_mode)
                    catalog.append(
                        {
                            "embedding_id": embedding_id,
                            "method": method,
                            "seed": seed,
                            "lambda": lam,
                            "native_loss": native,
                        }
                    )
        frame = pd.DataFrame(catalog)
        frame.to_csv(out / "candidates.csv", index=False)
        np.savez_compressed(out / "embeddings.npz", ids=frame["embedding_id"].to_numpy(), embeddings=np.stack([embeddings[i] for i in frame["embedding_id"]]))
        triplets = triplets_for(data["n"], 16000)
        scorer = (
            (lambda embedding: _gaussian_scores(data["X"], data["labels"], embedding))
            if name == "gaussian"
            else (lambda embedding: _swiss_scores(data, embedding, triplets))
        )
        marked = mark_by_loss(frame, "native_loss", EPSILON, group_cols=("method",))
        graph = _graph_members(frame, embeddings)
        rules = {
            "unrestricted_comds": (frame, None, "comds"),
            "unrestricted_locomds": (frame, None, "locomds"),
            "rashomon_uniform_comds": (marked[marked["in_rashomon"]], None, "comds"),
            "rashomon_balanced_comds": (
                marked[marked["in_rashomon"]],
                method_balanced_weights(marked[marked["in_rashomon"]]),
                "comds",
            ),
            "graph_rashomon_comds": (graph, None, "comds"),
            "quality_weighted_comds": (frame, _quality_weights(frame), "comds"),
        }
        # Size-matched draws use the loss-Rashomon count.
        rashomon_count = int(marked["in_rashomon"].sum())
        for draw in range(5):
            take = frame.sample(n=min(rashomon_count, len(frame)), random_state=100 + draw)
            rules[f"size_matched_{draw}"] = (take, None, "comds")
        for rule, (members, weights, kind) in rules.items():
            ids = [str(value) for value in members["embedding_id"]]
            views = [embeddings[key] for key in ids]
            if len(views) == 0:
                continue
            if len(views) == 1:
                embedding, seconds = views[0], 0.0
            else:
                weight_vector = None if weights is None else np.asarray(weights, dtype=float)
                embedding, seconds = _consensus(views, out / rule, kind, weights=weight_vector)
            scores = scorer(embedding)
            rows.append(_row(name, rule, "redr", len(views), seconds, scores))
            print(f"  {name} {rule} n={len(views)}", flush=True)
        # One embedding per method: lowest native loss.
        for method, group in frame.groupby("method"):
            best = group.sort_values("native_loss").iloc[0]
            embedding = embeddings[str(best["embedding_id"])]
            rows.append(_row(name, f"best_{method}", "individual", 1, np.nan, scorer(embedding)))
        # Common-knowledge refit on the loss-Rashomon pool.
        kept = [embeddings[str(value)] for value in marked.loc[marked["in_rashomon"], "embedding_id"]]
        if len(kept) >= 2:
            edges = candidate_edges(data["X"], 15)
            selected = mean_score_edges(kept, edges, keep_per_point=10)
            graph_matrix = edges_to_graph(selected, data["n"])
            import cne

            model = cne.CNE(
                optimizer="adam",
                parametric=True,
                loss_mode="umap",
                label_weights=0.0,
                seed=0,
                task_type="pca",
                n_epochs=CNE_EPOCHS,
                batch_size=min(128, data["n"]),
                k=15,
                negative_samples=5,
                use_keops=False,
                print_freq_epoch=None,
                early_exaggeration=False,
            )
            common = np.asarray(model.fit_transform(matrix, pca, graph=graph_matrix), dtype=np.float64)
            rows.append(_row(name, "common_knowledge_umap", "rashomon_dr", 1, np.nan, scorer(common)))
            np.save(out / "common_knowledge.npy", common)
        # Uncertainty: does neighbor-vote entropy track false neighbors?
        if name == "gaussian":
            rows.extend(_uncertainty_rows(name, data, frame, marked, embeddings))
        pd.DataFrame(rows).to_csv(root / "parametric.csv", index=False)
    return pd.DataFrame(rows)


def _point_entropy_and_error(labels: np.ndarray, pool: list[np.ndarray], k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    n = len(labels)
    votes = np.zeros((n, n), dtype=float)
    for embedding in pool:
        neighbors = knn_indices(embedding, k)
        for i in range(n):
            votes[i, neighbors[i]] += 1.0
    entropy = np.zeros(n)
    error = np.zeros(n)
    for i in range(n):
        chosen = np.argsort(-votes[i])[:k]
        positive = votes[i, votes[i] > 0]
        probability = positive / positive.sum()
        entropy[i] = float(-(probability * np.log(probability)).sum())
        error[i] = float(np.mean(labels[chosen] != labels[i]))
    return entropy, error


def _uncertainty_rows(name: str, data: dict, frame: pd.DataFrame, marked: pd.DataFrame, embeddings: dict[str, np.ndarray]) -> list[dict]:
    from scipy.stats import spearmanr

    rows = []
    pools = {
        "full_parametric": [embeddings[str(v)] for v in frame["embedding_id"]],
        "loss_rashomon": [embeddings[str(v)] for v in marked.loc[marked["in_rashomon"], "embedding_id"]],
    }
    for pool_name, pool in pools.items():
        if len(pool) < 2:
            continue
        entropy, error = _point_entropy_and_error(data["labels"], pool)
        correlation = spearmanr(entropy, error).correlation
        rows.append(
            {
                "dataset": name,
                "rule": f"uncertainty_{pool_name}",
                "family": "uncertainty",
                "n_views": len(pool),
                "seconds": np.nan,
                "adjusted_rand": np.nan,
                "silhouette_true": np.nan,
                "neighbor_purity": float(1.0 - np.mean(error)),
                "trustworthiness": np.nan,
                "intrinsic_triplet_balanced": np.nan,
                "lcmc_k20": np.nan,
                "entropy_error_spearman": float(correlation) if correlation == correlation else np.nan,
                "mean_vote_entropy": float(np.mean(entropy)),
            }
        )
        print(f"  uncertainty {pool_name} spearman {correlation}", flush=True)
    return rows


def run_noise(root: Path) -> pd.DataFrame:
    """One noisy copy of the Gaussian mixture. Classical views only."""
    clean = mixture_of_gaussians(331)
    rng = np.random.RandomState(7)
    noisy_matrix = StandardScaler().fit_transform(clean["X"] + rng.normal(scale=0.25, size=clean["X"].shape))
    rows = []
    for tag, matrix in (("clean", clean["X"]), ("noise_0.25", noisy_matrix)):
        views = {key: classical_views(matrix, random_state=0)[key] for key in ("pca", "isomap", "tsne_perplexity_30", "umap_neighbors_30")}
        stored = {}
        for key, embedding in views.items():
            scores = _gaussian_scores(matrix if tag == "noise_0.25" else clean["X"], clean["labels"], embedding)
            # Score every embedding against the clean observations and the true labels.
            scores = _gaussian_scores(clean["X"], clean["labels"], embedding)
            rows.append(_row(tag, key, "individual", 1, np.nan, scores))
            stored[key] = embedding
        embedding, seconds = _consensus(list(stored.values()), root / "noise" / tag / "comds", "comds")
        rows.append(_row(tag, "comds_four", "original_consensus", 4, seconds, _gaussian_scores(clean["X"], clean["labels"], embedding)))
        stored["comds_four"] = embedding
        np.savez_compressed(root / "noise" / f"{tag}.npz", **stored)
    clean_archive = np.load(root / "noise" / "clean.npz")
    noisy_archive = np.load(root / "noise" / "noise_0.25.npz")
    for key in clean_archive.files:
        distance = procrustes_rmse(clean_archive[key], noisy_archive[key])
        rows.append(
            {
                "dataset": "noise_shift",
                "rule": key,
                "family": "stability",
                "n_views": 1,
                "seconds": np.nan,
                "adjusted_rand": np.nan,
                "silhouette_true": np.nan,
                "neighbor_purity": np.nan,
                "trustworthiness": np.nan,
                "intrinsic_triplet_balanced": np.nan,
                "lcmc_k20": np.nan,
                "procrustes_to_clean": distance,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "noise.csv", index=False)
    return frame


def _plot(root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for name, color_key in (("gaussian", "labels"), ("swiss", "t")):
        archive = root / name / "classical" / "views.npz"
        if not archive.exists():
            continue
        stored = np.load(archive, allow_pickle=True)
        ids = list(stored["ids"])
        wanted = [key for key in ("pca", "isomap", "tsne_perplexity_30", "umap_neighbors_30") if key in ids]
        comds_path = root / name / "classical" / "comds" / "gspace.csv"
        locomds_path = root / name / "classical" / "locomds_tau0.1_p0.5" / "gspace.csv"
        fig, axes = plt.subplots(1, len(wanted) + 2, figsize=(2.4 * (len(wanted) + 2), 2.6))
        color = stored[color_key]
        panels = [(key, stored["embeddings"][ids.index(key)]) for key in wanted]
        if comds_path.exists():
            panels.append(("CoMDS", pd.read_csv(comds_path, header=None).to_numpy()))
        if locomds_path.exists():
            panels.append(("LoCoMDS", pd.read_csv(locomds_path, header=None).to_numpy()))
        for ax, (title, embedding) in zip(axes, panels):
            ax.scatter(embedding[:, 0], embedding[:, 1], c=color, s=4, cmap="viridis", linewidths=0)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(name)
        fig.tight_layout()
        fig.savefig(figure_dir / f"{name}_views.png", dpi=140)
        plt.close(fig)


def main() -> None:
    root = REPO / "results" / "checkpoint6"
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "classical.csv").exists():
        run_classical(root)
    else:
        print("classical cached", flush=True)
    if not (root / "parametric.csv").exists():
        run_parametric(root)
    else:
        print("parametric cached", flush=True)
    if not (root / "noise.csv").exists():
        run_noise(root)
    _plot(root)
    print("done", root, flush=True)


if __name__ == "__main__":
    main()
