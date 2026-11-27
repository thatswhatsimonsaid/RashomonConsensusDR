"""Fixed reference losses for Rashomon membership.

The training scalar mixes the DR loss with the PCA-alignment penalty, and
ContrastiveLoss draws new negatives on every resampled call. Membership uses
a reference that drops the alignment penalty and freezes the pairs.

For UMAP, InfoNCE, negative t-SNE (`loss_mode="neg"`), and NCVis
(`loss_mode="nce"`), the reference is the vendored `ContrastiveLoss` at the
training settings temperature=0.5, eps=1, spec_param=1, negative_samples=5.
Positives are the fixed high-d kNN. The same negative index tensor is reused
for every embedding. The reported native loss is the mean over those
positives and several negative draws. The standard deviation is retained so
a noisy draw is not the membership rule.

For PaCMAP, the reference is the terminal phase of `pacmap_weight_schedule`:
neighbor weight 1, further-point weight 1, mid-near weight 0. The Checkpoint 1
probe instead sums the three unweighted terms from `record_loss_pacmap`.
Both numbers are returned. Pairs come from one `generate_pair` call.

These scores are evaluations of the candidates we trained. They do not
enumerate the continuous Rashomon set.
"""

from __future__ import annotations

import numpy as np
import torch
from cne.cne import ContrastiveLoss, make_neighbor_indices

from redr.probe import _edges_from_neighbor_table, _rashomon_evaluation


def _contrastive_once(
    embedding: np.ndarray,
    positive_index: np.ndarray,
    loss_mode: str,
    negative_indices: torch.Tensor,
    negative_samples: int,
    spec_param: float = 1.0,
) -> float:
    y = torch.tensor(np.asarray(embedding), dtype=torch.float32)
    positive = y[torch.as_tensor(positive_index, dtype=torch.long)]
    features = torch.cat([y, positive], dim=0)
    criterion = ContrastiveLoss(
        negative_samples=negative_samples,
        loss_mode=loss_mode,
        metric="euclidean",
        temperature=0.5,
        eps=torch.tensor(1.0),
        spec_param=torch.tensor(float(spec_param)),
        clamp_high=1.0,
        clamp_low=1e-4,
        seed=0,
        loss_aggregation="mean",
    )
    criterion.neigh_inds = negative_indices
    loss, *_rest = criterion(
        features,
        log_Z=torch.tensor(0.0),
        log_temp=None,
        force_resample=False,
    )
    return float(loss.detach().cpu())


def frozen_negative_indices(n: int, negative_samples: int, seed: int) -> torch.Tensor:
    torch.manual_seed(int(seed))
    return make_neighbor_indices(n, negative_samples).cpu()


def contrastive_reference_loss(
    embedding: np.ndarray,
    neighbor_index: np.ndarray,
    loss_mode: str,
    negative_samples: int = 5,
    n_draws: int = 4,
    base_seed: int = 1000,
    spec_param: float = 1.0,
) -> tuple[float, float]:
    """Mean and std of ContrastiveLoss on a shared positive graph and shared negatives."""
    embedding = np.asarray(embedding, dtype=np.float64)
    neighbor_index = np.asarray(neighbor_index)
    n, dim = embedding.shape
    if negative_samples * dim >= n:
        raise ValueError(
            "ContrastiveLoss's full-batch branch is not used for this reference; "
            f"need negative_samples * dim < n, got {negative_samples * dim} >= {n}"
        )
    values = []
    for draw in range(n_draws):
        negatives = frozen_negative_indices(n, negative_samples, base_seed + draw)
        for column in range(neighbor_index.shape[1]):
            values.append(
                _contrastive_once(
                    embedding,
                    neighbor_index[:, column],
                    loss_mode,
                    negatives,
                    negative_samples,
                    spec_param,
                )
            )
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std())


def pacmap_component_losses(
    embedding: np.ndarray,
    pair_neighbors: np.ndarray,
    pair_mn: np.ndarray,
    pair_fp: np.ndarray,
    n_samples: int,
) -> dict[str, float]:
    """PaCMAP NN, mid-near, and further-point terms, plus two scalar summaries."""
    from parampacmap.training import convert_pairs

    nn_table, fp_table, mn_table = convert_pairs(
        pair_neighbors, pair_fp, pair_mn, n_samples
    )
    evaluation = _rashomon_evaluation()
    loss_nn, loss_mn, loss_fp = evaluation.record_loss_pacmap(
        np.asarray(embedding, dtype=np.float64),
        _edges_from_neighbor_table(nn_table),
        _edges_from_neighbor_table(mn_table),
        _edges_from_neighbor_table(fp_table),
        n_samples,
    )
    nn = float(np.mean(loss_nn))
    mn = float(np.mean(loss_mn))
    fp = float(np.mean(loss_fp))
    return {
        "pacmap_nn": nn,
        "pacmap_mn": mn,
        "pacmap_fp": fp,
        "probe_loss": nn + mn + fp,
        "native_loss": nn + fp,
        "native_loss_std": 0.0,
    }
