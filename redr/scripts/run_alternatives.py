"""Follow-up formulations after the two-dimensional discovery pool.

The rules below were fixed before this multi-seed run. A one-seed probe had
already shown that one-dimensional stress prefers uninformative projections
and that agreement clustering does not split the two latent axes. This script
checks those findings on new seeds. It is not a search over thresholds.

Mutual-information clustering is the feature-clustering method that Pearson
correlation cannot be, because the coordinates of one curve are dependent
and not linearly correlated.

    PYTHONPATH=redr python redr/scripts/run_alternatives.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import SpectralEmbedding

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "redr"))

from redr.baselines import concat_one_dimensional, isomap_embedding  # noqa: E402
from redr.complementary import score_embedding, triplets_for  # noqa: E402
from redr.dependence import (  # noqa: E402
    cluster_columns,
    one_dimensional,
    rashomon_groups,
    self_view_losses,
    stack_coordinates,
)
from redr.discovery import make_dataset  # noqa: E402
from redr.factors import generate_curves  # noqa: E402
from redr.intrinsic import coordinate_triplet_accuracy  # noqa: E402
from redr.projections import agreement_pair, projection_bank, stress_rashomon_pair  # noqa: E402


N_SAMPLES = 160
N_REPS = 3
EPSILON = 0.05

PRODUCT = [
    {"name": "aligned", "mixing": "aligned", "hide_second": False, "noise": 0.05, "n_nuisance": 8, "preserve_scale": False, "aspect": (1.0, 1.0), "seed_offset": 8000},
    {"name": "rotated", "mixing": "rotated", "hide_second": False, "noise": 0.05, "n_nuisance": 8, "preserve_scale": False, "aspect": (1.0, 1.0), "seed_offset": 8100},
    {"name": "rotated_hidden", "mixing": "rotated", "hide_second": True, "noise": 0.05, "n_nuisance": 8, "preserve_scale": False, "aspect": (1.0, 1.0), "seed_offset": 8200},
    {"name": "stretched_rotated", "mixing": "rotated", "hide_second": False, "noise": 0.05, "n_nuisance": 8, "preserve_scale": True, "aspect": (5.0, 1.0), "seed_offset": 8300},
]

FACTORS = [
    {"name": "three_aligned", "n_curves": 3, "mixing": "aligned", "shared_weight": 0.0, "seed_offset": 8400},
    {"name": "three_rotated", "n_curves": 3, "mixing": "rotated", "shared_weight": 0.0, "seed_offset": 8500},
    {"name": "shared_aligned", "n_curves": 2, "mixing": "aligned", "shared_weight": 1.0, "seed_offset": 8600},
]


def _product(spec: dict, replication: int) -> dict:
    return make_dataset(
        n_samples=N_SAMPLES,
        mixing=spec["mixing"],
        hide_second=spec["hide_second"],
        noise=spec["noise"],
        n_nuisance=spec["n_nuisance"],
        preserve_scale=spec["preserve_scale"],
        aspect=spec["aspect"],
        random_state=spec["seed_offset"] + replication,
    )


def _embed_groups(matrix: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    usable = [group for group in groups if len(group) >= 3]
    if not usable:
        return isomap_embedding(matrix, n_neighbors=10)
    coordinates = [one_dimensional(matrix, group) for group in usable]
    return stack_coordinates(coordinates)


def _privileged_blocks(blocks: list[np.ndarray]) -> np.ndarray:
    coordinates = []
    for block in blocks:
        columns = np.arange(block.shape[1])
        if len(columns) < 3:
            continue
        coordinates.append(one_dimensional(block, columns))
    return stack_coordinates(coordinates)


def _row(name: str, privileged: bool, embedding: np.ndarray, data: dict, triplets: np.ndarray) -> dict:
    # Triplet scores use distances, so a 2-d embedding of a 3-factor unfolding
    # is padded with zero columns. The extra columns do not change those distances.
    # Procrustes inside the scorer requires matching widths.
    width = int(data["intrinsic"].shape[1])
    if embedding.shape[1] < width:
        padding = np.zeros((len(embedding), width - embedding.shape[1]))
        embedding = np.column_stack([embedding, padding])
    scores = score_embedding(data, embedding, triplets)
    third = float("nan")
    if data["intrinsic"].shape[1] >= 3:
        third = coordinate_triplet_accuracy(data["intrinsic"][:, 2], embedding, triplets)
    return {
        "rule": name,
        "privileged_blocks": privileged,
        "intrinsic_triplet_balanced": scores["intrinsic_triplet_balanced"],
        "intrinsic_triplet_isometric": scores["intrinsic_triplet_isometric"],
        "u_triplet": scores["u_triplet"],
        "v_triplet": scores["v_triplet"],
        "third_triplet": third,
        "trustworthiness": scores["trustworthiness"],
        "error": "",
    }


def _safe(name: str, privileged: bool, builder, data: dict, triplets: np.ndarray) -> dict:
    try:
        embedding = builder()
        row = _row(name, privileged, embedding, data, triplets)
    except Exception as exc:  # noqa: BLE001
        row = {
            "rule": name,
            "privileged_blocks": privileged,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return row


def run_product(spec: dict, replication: int) -> list[dict]:
    data = _product(spec, replication)
    matrix = data["X_visible"]
    triplets = triplets_for(data["n"], 14000 + replication)
    rows = []

    def add(name: str, privileged: bool, builder) -> None:
        row = _safe(name, privileged, builder, data, triplets)
        row.update({"condition": spec["name"], "replication": replication, "family": "product"})
        rows.append(row)
        print(
            f"{spec['name']} r{replication} {name} "
            f"bal={row.get('intrinsic_triplet_balanced', float('nan'))} "
            f"u={row.get('u_triplet', float('nan'))} v={row.get('v_triplet', float('nan'))} "
            f"err={row['error'][:80]}",
            flush=True,
        )

    add("mi_k2_concat", False, lambda: _embed_groups(matrix, cluster_columns(matrix, 2, "mutual_information", data["random_state"])))
    add("pearson_k2_concat", False, lambda: _embed_groups(matrix, cluster_columns(matrix, 2, "pearson", data["random_state"])))
    groups3 = cluster_columns(matrix, 3, "mutual_information", data["random_state"])
    add("mi_k3_concat", False, lambda: _embed_groups(matrix, groups3))
    losses = self_view_losses(matrix, groups3)
    add("mi_k3_rashomon", False, lambda: _embed_groups(matrix, rashomon_groups(groups3, losses, EPSILON)))
    add(
        "stress_rashomon_pair",
        False,
        lambda: stress_rashomon_pair(*projection_bank(matrix, 24, data["random_state"] + 17), EPSILON),
    )
    add(
        "agreement_k2",
        False,
        lambda: agreement_pair(projection_bank(matrix, 32, data["random_state"] + 19)[0]),
    )
    add(
        "pca6_agreement_k2",
        False,
        lambda: agreement_pair(projection_bank(matrix, 32, data["random_state"] + 23, pca_dim=6)[0]),
    )
    add("isomap_visible", False, lambda: isomap_embedding(matrix, n_neighbors=10))
    add(
        "spectral_visible",
        False,
        lambda: np.asarray(SpectralEmbedding(n_components=2, n_neighbors=10, random_state=0).fit_transform(matrix), dtype=float),
    )
    add("privileged_concat_1d", True, lambda: concat_one_dimensional(data))
    return rows


def run_factor(spec: dict, replication: int) -> list[dict]:
    data = generate_curves(
        n_samples=N_SAMPLES,
        n_curves=spec["n_curves"],
        noise=0.05,
        n_nuisance=8,
        shared_weight=spec["shared_weight"],
        mixing=spec["mixing"],
        random_state=spec["seed_offset"] + replication,
    )
    matrix = data["X_visible"]
    triplets = triplets_for(data["n"], 15000 + replication)
    rows = []

    def add(name: str, privileged: bool, builder) -> None:
        row = _safe(name, privileged, builder, data, triplets)
        row.update({"condition": spec["name"], "replication": replication, "family": "factor"})
        rows.append(row)
        print(
            f"{spec['name']} r{replication} {name} "
            f"bal={row.get('intrinsic_triplet_balanced', float('nan'))} "
            f"u={row.get('u_triplet', float('nan'))} v={row.get('v_triplet', float('nan'))} "
            f"w={row.get('third_triplet', float('nan'))} err={row['error'][:80]}",
            flush=True,
        )

    add("mi_k2_concat", False, lambda: _embed_groups(matrix, cluster_columns(matrix, 2, "mutual_information", data["random_state"])))
    groups = cluster_columns(matrix, spec["n_curves"], "mutual_information", data["random_state"])
    add("mi_matched_concat", False, lambda: _embed_groups(matrix, groups))
    losses = self_view_losses(matrix, groups)
    add("mi_matched_rashomon", False, lambda: _embed_groups(matrix, rashomon_groups(groups, losses, EPSILON)))
    add("isomap_visible", False, lambda: isomap_embedding(matrix, n_neighbors=10))
    add(
        "spectral_visible",
        False,
        lambda: np.asarray(SpectralEmbedding(n_components=2, n_neighbors=10, random_state=0).fit_transform(matrix), dtype=float),
    )
    add("privileged_concat_1d", True, lambda: _privileged_blocks(data["blocks"]))
    return rows


def main() -> None:
    root = REPO / "results" / "checkpoint5" / "alternatives"
    root.mkdir(parents=True, exist_ok=True)
    dest = root / "detail.csv"
    frames = []
    done = set()
    if dest.exists():
        previous = pd.read_csv(dest)
        frames.append(previous)
        done = set(zip(previous["condition"], previous["replication"]))
    jobs = [("product", spec) for spec in PRODUCT] + [("factor", spec) for spec in FACTORS]
    for kind, spec in jobs:
        for replication in range(N_REPS):
            if (spec["name"], replication) in done:
                print(f"skip {spec['name']} {replication}", flush=True)
                continue
            fresh = pd.DataFrame(run_product(spec, replication) if kind == "product" else run_factor(spec, replication))
            frames.append(fresh)
            pd.concat(frames, ignore_index=True).to_csv(dest, index=False)
    detail = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    detail.to_csv(dest, index=False)
    if len(detail) and "intrinsic_triplet_balanced" in detail.columns:
        summary = detail.groupby(["condition", "rule"])[
            ["intrinsic_triplet_balanced", "intrinsic_triplet_isometric", "u_triplet", "v_triplet", "third_triplet", "trustworthiness"]
        ].agg(["mean", "std", "count"])
        summary.to_csv(root / "summary.csv")
        print(summary.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
