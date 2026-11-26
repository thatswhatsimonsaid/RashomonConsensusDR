"""Real DR embeddings, plus one self-consistent bad family.

The selection score is 1 minus trustworthiness against the observed Swiss
roll, computed inside each family. The latent parameter is used only to
score a finished consensus. The prediction, fixed from the quality-diversity
pilot, is that a within-method 5% band keeps the permuted t-SNE family
because its losses are similarly bad, and that this consensus falls behind
unrestricted CoMDS. Gibbs and trimming are the quality-weight baselines.

    PYTHONPATH=redr python redr/scripts/run_dr_mechanism.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE, trustworthiness

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "redr"))

from redr.comds_bridge import fit_consensus  # noqa: E402
from redr.intrinsic import intrinsic_recovery  # noqa: E402
from redr.metrics import _sample_triplets  # noqa: E402
from redr.paper_views import swiss_roll  # noqa: E402
from redr.quality_diversity import retained, scheme_weights, size_matched, trimmed  # noqa: E402
from redr.rashomon import gibbs_weights  # noqa: E402


N = 180
REPS = 4
SEED0 = 3000


def _views(matrix: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    views: dict[str, tuple[str, np.ndarray]] = {}
    for neighbors in (10, 25):
        views[f"isomap_k{neighbors}"] = (
            "isomap",
            np.asarray(Isomap(n_neighbors=neighbors, n_components=2).fit_transform(matrix), dtype=float),
        )
    for perplexity, draw in ((20, 0), (20, 1), (40, 0), (40, 1)):
        embedding = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=seed + draw,
            init="pca",
            max_iter=500,
        ).fit_transform(matrix)
        views[f"tsne_p{perplexity}_s{draw}"] = ("tsne", np.asarray(embedding, dtype=float))
    try:
        import umap

        for neighbors, draw in ((10, 0), (30, 1)):
            embedding = umap.UMAP(
                n_neighbors=neighbors,
                n_components=2,
                random_state=seed + draw,
                compatibility_layout=True,
            ).fit_transform(matrix)
            views[f"umap_k{neighbors}_s{draw}"] = ("umap", np.asarray(embedding, dtype=float))
    except Exception as exc:  # noqa: BLE001
        print(f"umap unavailable: {exc}", flush=True)
    for draw in range(4):
        jitter = matrix + rng.normal(0.0, 0.05, size=matrix.shape)
        views[f"pca_j{draw}"] = ("pca", PCA(n_components=2, random_state=draw).fit_transform(jitter))
    shift = max(matrix.shape[0] // 3, 1)
    for name, (family, embedding) in list(views.items()):
        if family == "tsne":
            views[f"permuted_{name}"] = ("permuted_tsne", np.roll(embedding, shift, axis=0))
    return views


def _fit(views: list[np.ndarray], dest: Path, weights: np.ndarray | None) -> tuple[np.ndarray, float]:
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "gspace.csv").exists():
        return pd.read_csv(dest / "gspace.csv", header=None).to_numpy(dtype=float), float("nan")
    started = time.perf_counter()
    fitted = fit_consensus(views, dest, kind="comds", weights=weights)
    return np.asarray(fitted["embedding"], dtype=float), float(fitted.get("seconds", time.perf_counter() - started))


def run_replication(replication: int, root: Path) -> pd.DataFrame:
    data = swiss_roll(N, noise=0.1, random_state=SEED0 + replication)
    matrix = data["X"]
    catalog = _views(matrix, SEED0 + replication)
    records = []
    embeddings = {}
    for name, (family, embedding) in catalog.items():
        loss = 1.0 - float(trustworthiness(matrix, embedding, n_neighbors=10))
        records.append({"embedding_id": name, "method": family, "score": loss, "true_global_error": loss})
        embeddings[name] = np.asarray(embedding, dtype=float)
    frame = pd.DataFrame(records)
    triplets = _sample_triplets(N, 2000, np.random.default_rng(SEED0 + 9000 + replication))
    specs = [
        ("best", frame.sort_values(["score", "embedding_id"]).iloc[:1], None),
        ("unrestricted", frame, None),
        ("trimmed", trimmed(frame, 0.25), None),
        ("full_gibbs", frame, "full_gibbs"),
    ]
    for epsilon in (0.0, 0.05, 0.25, 1.0, np.inf):
        specs.append((f"rashomon_{epsilon}", retained(frame, epsilon), None))
    specs.append(("rashomon_gibbs_0.05", retained(frame, 0.05), "gibbs"))
    count = len(retained(frame, 0.05))
    match_rng = np.random.default_rng(SEED0 + 7000 + replication)
    for draw in range(3):
        specs.append((f"size_matched_{draw}", size_matched(frame, count, match_rng), None))
    rows = []
    for rule, members, scheme in specs:
        ids = [str(value) for value in members["embedding_id"]]
        views = [embeddings[item] for item in ids]
        weights = None
        if scheme == "gibbs":
            weights = scheme_weights(members, "gibbs", 1.0)
        elif scheme == "full_gibbs":
            weights = gibbs_weights(members, "score", beta=1.0)
        if len(views) == 1:
            embedding, seconds = views[0], 0.0
        else:
            embedding, seconds = _fit(views, root / f"rep{replication}" / rule, weights)
        recovery = intrinsic_recovery(data["intrinsic"], embedding, triplets, k=8)
        rows.append(
            {
                "replication": replication,
                "rule": rule,
                "n_kept": len(members),
                "n_permuted": int(members["method"].eq("permuted_tsne").sum()),
                "n_pca": int(members["method"].eq("pca").sum()),
                "n_methods": int(members["method"].nunique()),
                "mean_trust_loss": float(members["score"].mean()),
                "seconds": seconds,
                "trustworthiness": float(trustworthiness(matrix, embedding, n_neighbors=10)),
                "intrinsic_triplet_balanced": recovery["intrinsic_triplet_balanced"],
                "intrinsic_triplet_isometric": recovery["intrinsic_triplet_isometric"],
                "arc_triplet_accuracy": recovery["arc_triplet_accuracy"],
                "height_triplet_accuracy": recovery["height_triplet_accuracy"],
            }
        )
        print(
            f"r{replication} {rule} n={len(members)} perm={rows[-1]['n_permuted']} "
            f"balanced={recovery['intrinsic_triplet_balanced']:.3f}",
            flush=True,
        )
    frame.to_csv(root / f"rep{replication}_candidates.csv", index=False)
    return pd.DataFrame(rows)


def _plots(detail: pd.DataFrame, root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from redr.paper_views import swiss_roll

    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    labels = ["0.0", "0.05", "0.25", "1.0", "inf"]
    means, lows, highs = [], [], []
    for label in labels:
        rule = "rashomon_inf" if label == "inf" else f"rashomon_{label}"
        values = detail.loc[detail["rule"].eq(rule), "intrinsic_triplet_balanced"].to_numpy()
        means.append(float(np.mean(values)))
        lows.append(float(np.min(values)))
        highs.append(float(np.max(values)))
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(range(len(labels)), means, marker="o", color="C3", label="within-method band")
    ax.fill_between(range(len(labels)), lows, highs, color="C3", alpha=0.15)
    for name, color in (("unrestricted", "C0"), ("full_gibbs", "C2"), ("trimmed", "C1"), ("best", "C4")):
        level = float(detail.loc[detail["rule"].eq(name), "intrinsic_triplet_balanced"].mean())
        ax.axhline(level, color=color, linestyle="--", linewidth=1, label=name)
    ax.set_xticks(range(len(labels)), ["0", "0.05", "0.25", "1", "inf"])
    ax.set_xlabel("epsilon")
    ax.set_ylabel("balanced intrinsic triplet")
    ax.set_title("Swiss roll with one permuted t-SNE family")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(figure_dir / "epsilon_curve.png", dpi=140)
    plt.close(fig)

    data = swiss_roll(N, noise=0.1, random_state=SEED0)
    color = data["t"]
    panels = []
    for title, rule in (
        ("unrestricted", "unrestricted"),
        ("Rashomon 0.05", "rashomon_0.05"),
        ("Gibbs on that set", "rashomon_gibbs_0.05"),
        ("trimmed", "trimmed"),
    ):
        path = root / "rep0" / rule / "gspace.csv"
        panels.append((title, pd.read_csv(path, header=None).to_numpy(dtype=float)))
    fig, axes = plt.subplots(1, len(panels), figsize=(2.4 * len(panels), 2.6))
    for ax, (title, embedding) in zip(axes, panels):
        ax.scatter(embedding[:, 0], embedding[:, 1], c=color, s=8, cmap="viridis", linewidths=0)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Replication 0, colored by arc parameter")
    fig.tight_layout()
    fig.savefig(figure_dir / "swiss_consensus.png", dpi=140)
    plt.close(fig)


def main() -> None:
    root = REPO / "results" / "checkpoint7" / "dr_mechanism"
    root.mkdir(parents=True, exist_ok=True)
    frames = []
    for replication in range(REPS):
        dest = root / f"rep{replication}.csv"
        if dest.exists():
            frames.append(pd.read_csv(dest))
            continue
        frame = run_replication(replication, root)
        frame.to_csv(dest, index=False)
        frames.append(frame)
    detail = pd.concat(frames, ignore_index=True)
    detail.to_csv(root / "detail.csv", index=False)
    _plots(detail, root)
    print(detail.groupby("rule")[["intrinsic_triplet_balanced", "n_kept", "n_permuted", "n_pca"]].mean().round(3))


if __name__ == "__main__":
    main()
