#!/usr/bin/env python3
"""Re-score the Checkpoint 1 Swiss-roll PaCMAP embeddings.

The saved 220-epoch embeddings are not retrained. Two of those configurations
are also fit again for 450 epochs, which includes the full final phase.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from redr.data import load_dataset
from redr.loss_validation import pacmap_component_losses
from redr.metrics import _sample_triplets, evaluate_embedding, latent_triplet_accuracy
from redr.train import fit_pacmap, pacmap_pairs


def _score(embedding, data, pairs, triplets):
    parts = pacmap_component_losses(embedding, pairs[0], pairs[1], pairs[2], data["n"])
    scores = evaluate_embedding(data["X"], embedding, None, k=10, triplets=triplets)
    scores["latent_triplet_accuracy"] = latent_triplet_accuracy(
        data["latent"], embedding, triplets
    )
    return {**parts, **scores}


def main():
    data = load_dataset("swiss")
    pairs = pacmap_pairs(data["X"], seed=0)
    triplets = _sample_triplets(data["n"], 4000, np.random.default_rng(0))
    archive = np.load(ROOT / "results" / "swiss" / "embeddings.npz", allow_pickle=True)
    ids = [str(v) for v in archive["ids"]]
    rows = []
    for embedding_id, embedding in zip(ids, archive["embeddings"]):
        if not embedding_id.startswith("pacmap_"):
            continue
        parts = _score(embedding, data, pairs, triplets)
        rows.append({"embedding_id": embedding_id, "source": "checkpoint1_220", **parts})
        print(
            f"{embedding_id} probe={parts['probe_loss']:.4f} "
            f"terminal={parts['native_loss']:.4f} trust={parts['trustworthiness']:.3f} "
            f"latent_triplet={parts['latent_triplet_accuracy']:.3f}",
            flush=True,
        )
    for lam in (0.0, 5.0):
        embedding, seconds = fit_pacmap(
            data["X"], data["X_pca"], seed=0, label_weight=lam, n_epochs=450, pairs=pairs
        )
        parts = _score(embedding, data, pairs, triplets)
        rows.append(
            {
                "embedding_id": f"pacmap_s0_l{lam:g}_e450",
                "source": "retrained_450",
                "train_seconds": seconds,
                **parts,
            }
        )
        print(
            f"retrained λ={lam:g} probe={parts['probe_loss']:.4f} "
            f"terminal={parts['native_loss']:.4f} trust={parts['trustworthiness']:.3f} "
            f"({seconds:.1f}s)",
            flush=True,
        )
    out = ROOT / "results" / "checkpoint2" / "swiss_pacmap_anomaly.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
