# Checkpoint 5 — Automatically discovering complementary geometry

Local pilots, 21 November 2026 (America/Los_Angeles). Checkpoint 3 and Checkpoint 4 are unchanged. No cluster array was submitted.

Checkpoint 4 showed that CoMDS can combine embeddings that were trained on known feature blocks. It did not show that a Rashomon rule was the reason. The question here is whether an unsupervised procedure can find complementary views from the observed matrix, and whether a multi-objective Rashomon rule keeps those views when a single global score would drop them.

Latent coordinates are used only to score a finished embedding. Membership, weights, clustering, and projection draws do not read them.

These pilots are small: four replications for the privileged baselines, three for the automatic rules. They are not a 25-replication confirmation. They are large enough to reject the designs that were tried.

## The balanced triplet is not an axis score

`intrinsic_triplet_balanced` compares an embedding with the latent coordinates after each latent axis has been scaled to unit variance. `intrinsic_triplet_isometric` uses the raw latent coordinates. `u_triplet` and `v_triplet` are one-coordinate orderings. On this product manifold a perfect two-dimensional embedding does not score 1 on either one-coordinate test, because the other coordinate changes about a quarter of the triplet decisions.

Four replications, same seeds as Checkpoint 4 (`results/checkpoint5/baselines/metric_audit.csv`):

| Reference embedding | Square balanced | Square isometric | Stretched balanced | Stretched isometric | Stretched u | Stretched v |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw latent coordinates | 0.995 | 1.000 | 0.782 | 1.000 | 0.970 | 0.540 |
| Each latent axis scaled to unit variance | 1.000 | 0.995 | 1.000 | 0.782 | 0.752 | 0.758 |
| Only the first latent coordinate | 0.752 | — | 0.752 | — | 1.000 | 0.510 |

On the square, adding the second axis changes 0.252 of triplet decisions relative to the first axis, so the isometric score is not an arc-length score. On the stretched surface (aspect 5 to 1, scale preserved in the observations) that disagreement is only 0.030. The raw latent embedding then scores 1.000 isometric and 0.782 balanced, and its v triplet is 0.540. Equalizing the axes reverses those two numbers. A method can win the balanced score by undoing the aspect ratio and lose the isometric score. The two scores are both reported below. Neither one is full manifold recovery by itself.

<img src="/opt/cursor/artifacts/stretched_metrics.png" alt="Balanced and isometric triplet accuracy on the stretched product" />

## Known blocks: concatenation beats CoMDS

The baselines in this section may use the known curves. Isomap on the full matrix does not. Four replications, n = 200, noise 0.05, eight nuisance features. Square seeds 4000–4003. Stretched seeds 4300–4303 with `preserve_scale`. Hidden-second seeds 4200–4203. Means are balanced / isometric / u / v.

| Method | Sees the blocks | Square | Stretched | Hidden second |
| --- | --- | --- | --- | --- |
| Concatenate 1D Isomap of each block | yes | 0.989 / 0.989 / 0.751 / 0.755 | 0.986 / 0.787 / 0.757 / 0.752 | 0.739 / 0.752 / 0.906 / 0.504 |
| CoMDS of one classical MDS view per block | yes | 0.945 / 0.945 / 0.750 / 0.747 | 0.935 / 0.790 / 0.760 / 0.738 | 0.680 / 0.689 / 0.782 / 0.493 |
| Checkpoint 4 pool, source-balanced CoMDS | yes | 0.956 / 0.957 / 0.745 / 0.756 | 0.915 / 0.860 / 0.830 / 0.676 | 0.717 / 0.728 / 0.845 / 0.501 |
| SMACOF of the averaged block distances | yes | 0.931 / 0.931 / 0.746 / 0.746 | 0.918 / 0.770 / 0.742 / 0.749 | 0.669 / 0.678 / 0.817 / 0.501 |
| Classical MDS of the averaged block distances | yes | 0.836 / 0.837 / 0.716 / 0.717 | 0.832 / 0.740 / 0.717 / 0.718 | 0.693 / 0.702 / 0.841 / 0.506 |
| Isomap after standardizing each block | yes | 0.874 / 0.874 / 0.714 / 0.724 | 0.867 / 0.751 / 0.724 / 0.717 | 0.703 / 0.713 / 0.817 / 0.501 |
| Isomap on the full matrix | no | 0.874 / 0.874 / 0.714 / 0.724 | 0.759 / 0.965 / 0.972 / 0.518 | 0.703 / 0.713 / 0.817 / 0.501 |
| PCA after standardizing each block | yes | 0.801 / 0.802 / 0.692 / 0.690 | 0.810 / 0.722 / 0.698 / 0.697 | 0.646 / 0.653 / 0.714 / 0.500 |

On the square the matrix is already standardized, so block standardization does not change Isomap. On the stretched surface it does. Isomap on the raw stretched matrix keeps the long axis (u triplet 0.972, v triplet 0.518, isometric 0.965) and scores 0.759 balanced. Standardizing each block before Isomap raises the balanced score to 0.867 and the v triplet to 0.717, and lowers the isometric score to 0.751. The Checkpoint 4 statement that joint Isomap misses the short axis is partly a scaling effect.

CoMDS of the two block configurations beats classical MDS of their averaged distances (stretched balanced 0.935 versus 0.832) and is slightly above SMACOF of that average (0.918). It loses to concatenating the two one-dimensional Isomaps (0.986 balanced). Source-balanced weights on the Checkpoint 4 pool match unrestricted CoMDS from that checkpoint and also lose to concatenation. On the hidden-second data, concatenation does not invent the missing axis (v triplet 0.504) and is the best of these rules on the axis that remains (u triplet 0.906).

CoMDS does not, on this manifold, do more than straightforward multi-view integration. The Checkpoint 4 gain of specialist CoMDS over joint embeddings is the gain from being handed the blocks, and a simpler use of those blocks scores higher. It is not a Rashomon result.

## Automatic candidates from the visible matrix

The next pilot does not pass block labels to the candidate methods. An orthogonal rotation mixes the columns and preserves row-wise distances, so full-matrix Isomap is unchanged and a column subset is no longer a curve. Candidates, all fit on the visible matrix only: PCA; Isomap with 10 and 25 neighbors; eight random subsets of about 45% of the columns, each embedded with Isomap; four Gaussian projections to four dimensions, then Isomap; two clusters from average linkage on one minus absolute Pearson correlation; and, in one condition, four standard-normal embeddings.

Objectives, all minimized, none using the latent coordinates:

- local neighborhood loss at k = 5 on the full visible matrix
- the mean of that loss at k = 5, 15, and 30
- one minus Spearman correlation with a fixed Isomap geodesic
- scale-adjusted distance error against the full visible matrix
- one minus the best neighborhood preservation against any feature subset in the bank

Rules at relative epsilon 0.05: unrestricted CoMDS, unrestricted Procrustes, pooled multiscale CoMDS, the union of the five objective-specific bands, the Pareto front of the five losses, a marginal-gain rule, diversity with no quality gate, quality-weighted CoMDS, and the single best multiscale embedding. The marginal rule starts at the lowest multiscale loss and adds a candidate only when it improves neighborhood preservation of some observed subset by more than 0.05. Diversity-first starts at the embedding with the lowest mean neighborhood overlap and keeps six. Quality weights are `1 / (multiscale loss + 0.05)`.

Privileged concatenation, block CoMDS, and distance MDS are scored on the same draws so the gap is visible. They are not automatic methods.

Three replications, n = 160. Aligned seeds 6000–6002. Rotated 6100–6102. The contaminated condition reuses the rotated seeds and adds four noise embeddings. Stretched-rotated seeds 6300–6302. Twenty nuisance features, seeds 6400–6402. Noise 0.25, seeds 6500–6502. Hidden-second rotated seeds 6200–6202.

<img src="/opt/cursor/artifacts/discovery_balanced.png" alt="Balanced triplet of automatic rules and privileged concatenation" />

Balanced triplet means on the rotated square, the setting where the blocks are not columns:

| Rule | Members | Balanced | u | v |
| --- | ---: | ---: | ---: | ---: |
| All CoMDS | 17 | 0.822 | 0.698 | 0.704 |
| Union of objective bands | 4–7 | 0.826 | 0.702 | 0.698 |
| Pareto front | 5–7 | 0.838 | 0.703 | 0.707 |
| Pooled multiscale CoMDS | 3–4 | 0.849 | 0.715 | 0.705 |
| Quality-weighted CoMDS | 17 | 0.828 | 0.701 | 0.704 |
| Marginal gain | 6 | 0.772 | 0.682 | 0.656 |
| Diversity first | 6 | 0.653 | 0.634 | 0.584 |
| Best multiscale embedding | 1 | 0.849 | 0.710 | 0.710 |
| Isomap on the visible matrix | 1 | 0.846 | 0.708 | 0.706 |
| Privileged 1D concatenation | 1 | 0.986 | 0.767 | 0.732 |
| Privileged block CoMDS | 2 | 0.938 | 0.745 | 0.741 |

Paired differences on that rotated square, three replications:

- Pooled minus all CoMDS: +0.027 (sd 0.022), better in 3 of 3. Pooled minus Isomap: +0.003 (sd 0.012), better in 2 of 3. The filter moves the pool toward the full-matrix Isomap. Both axes stay near 0.71. It does not recover a specialist pair.
- Pareto minus all CoMDS: +0.016 (sd 0.019), 3 of 3. Pareto minus Isomap: −0.008 (sd 0.024), better in 1 of 3.
- Union minus all CoMDS: +0.004 (sd 0.031), better in 1 of 3.
- Marginal minus all CoMDS: −0.051 (sd 0.045), worse in 3 of 3.
- Diversity-first minus all CoMDS: −0.169 (sd 0.004), worse in 3 of 3.
- Every automatic rule minus privileged concatenation: between −0.14 and −0.33, worse in 3 of 3.

The same pattern holds when the features are still aligned. Pareto beats all CoMDS by +0.031 (sd 0.038) in 2 of 3 and loses to concatenation by 0.096 in 3 of 3. Axis scores stay near 0.72 rather than the 0.91 / 0.50 split of a true specialist. Random 45% subsets rarely isolate a three-column curve, and Pearson correlation does not group cosine with sine of the same angle.

Other conditions, still three replications:

- Hidden second curve. Every automatic rule has v triplet about 0.50. Pooled beats all CoMDS by +0.024 (sd 0.006, 3 of 3) and matches Isomap (0.708 versus 0.701). Privileged concatenation reaches u triplet 0.909 and does not invent v (0.500).
- Four noise embeddings added to the rotated pool. All CoMDS stays at 0.821, the same as the uncontaminated rotated run. Diversity-first falls from 0.653 to 0.610, a paired loss of 0.211 (sd 0.041) against all CoMDS, 0 of 3 the other way. Quality weighting stays at 0.826. Unrestricted CoMDS already ignores a few noise views. Selecting for disagreement does not.
- Noise 0.25. Pooled beats all CoMDS by +0.043 (sd 0.021, 3 of 3) and ties Isomap (−0.001, sd 0.010). It does not beat concatenation (0.941).
- Twenty nuisance features. Automatic rules sit at 0.73–0.79. Concatenation stays at 0.983.
- Stretched and rotated. Automatic rules, including the union and the Pareto front, score balanced about 0.75 and v triplet about 0.51, the same geometry as Isomap on the visible matrix (0.761 / 0.519). Concatenation scores 0.985 balanced and 0.758 on v, and 0.765 isometric. No automatic Rashomon rule recovers the short axis.

Diversity before quality filtering hurts in every condition above. Adding a marginal-diversity step after the best multiscale embedding also hurts. Quality weighting is within 0.01 of uniform CoMDS. There is no setting in this grid where a diversity rule beat both unrestricted CoMDS and Isomap.

## A dependence clustering that works only before rotation

Pearson clustering cannot see a curve. Mutual information can, when the curve is still three columns. The follow-up fixes that measure and also tests two projection rules that had looked plausible after the two-dimensional pool failed. Those projection rules were locked before the multi-seed run: a 5% band on one-dimensional stress of a random 3-d projection, and a two-cluster agreement split of one-dimensional Isomap coordinates. Seeds 8000 and above. They do not overlap the discovery pilot. Three replications, n = 160.

<img src="/opt/cursor/artifacts/dependence_balanced.png" alt="Mutual-information clustering against Isomap, aligned and rotated" />

Balanced triplet, mean (sd):

| Rule | Aligned | Rotated | Rotated, second curve removed | Stretched and rotated |
| --- | ---: | ---: | ---: | ---: |
| Mutual information, 3 clusters, concatenate 1D Isomaps | 0.930 (0.011) | 0.657 (0.061) | 0.616 (0.027) | 0.745 (0.015) |
| Same clusters, Rashomon band on each cluster's own 1D loss | 0.799 (0.116) | 0.677 (0.050) | 0.637 (0.049) | 0.745 (0.015) |
| Pearson, 2 clusters, concatenate | 0.777 (0.082) | 0.702 (0.017) | 0.687 (0.034) | 0.727 (0.015) |
| Stress Rashomon pair | 0.546 (0.027) | 0.572 (0.009) | 0.520 (0.024) | 0.675 (0.019) |
| Agreement pair, no quality gate | 0.630 (0.045) | 0.587 (0.023) | 0.559 (0.019) | 0.694 (0.018) |
| Isomap on the visible matrix | 0.865 (0.008) | 0.837 (0.010) | 0.706 (0.009) | 0.762 (0.019) |
| 1D Isomap of the known blocks | 0.986 (0.003) | 0.983 (0.006) | 0.731 (0.008) | 0.981 (0.010) |

On aligned columns, mutual-information concatenation beats Isomap by +0.065 (sd 0.008, 3 of 3) and loses to the known blocks by 0.056 (sd 0.008, 0 of 3). The clusters are not pure: each large cluster is one curve plus some nuisance columns. That is enough for a one-dimensional Isomap. The Rashomon band then drops the second curve in two of three replications. The self-view losses on seeds 8000–8002 were 0.408 versus 0.487, 0.353 versus 0.483, and 0.442 versus 0.445 (`aligned_cluster_losses.csv`). A 5% band keeps only the first of those pairs. The kept embedding scores about 0.91 on one axis and 0.50 on the other. The replication where the two losses match keeps both, and the band does nothing. Paired, the band minus unrestricted concatenation is −0.131 (sd 0.114), worse in 2 of 3 and tied in 1. Objective-specific filtering removed the complementary cluster. It did not protect it.

After rotation the same concatenation loses to Isomap by 0.180 (sd 0.070, 0 of 3). The original blocks are not columns, and column clustering has nothing to find. Stress pairs and agreement pairs lose to Isomap by 0.25 to 0.32 on the square. Low one-dimensional stress prefers projections that are easy to flatten, including uninformative ones. Forcing a second coordinate when the second curve was replaced by noise lowers the balanced score relative to Isomap (agreement −0.147, sd 0.021, 0 of 3) and does not raise the v triplet above 0.50.

## Three directions, and a shared curve

A separate generator places two or three independent curves in the columns, optionally adds the same extra curve to every block, and optionally rotates. A two-dimensional embedding cannot hold three complementary directions. Balanced triplet against the three-dimensional unfolding, three replications:

| Rule | Three curves, aligned | Three curves, rotated |
| --- | ---: | ---: |
| Isomap | 0.741 (0.015) | 0.768 (0.020) |
| Spectral embedding | 0.750 (0.012) | 0.746 (0.008) |
| Known blocks, 1D Isomap, then PCA to 2D | 0.741 (0.018) | 0.743 (0.030) |
| Mutual information, one cluster per curve, concatenated | 0.741 (0.091) | 0.661 (0.020) |
| Rashomon band on those clusters | 0.669 (0.011) | 0.661 (0.020) |

Privileged concatenation minus Isomap on the aligned three-curve draws is −0.000 (sd 0.031). Knowing the blocks and then projecting three coordinates to two dimensions does not beat one Isomap. The Rashomon band minus the unrestricted cluster concatenation is −0.072 (sd 0.080), worse in 2 of 3. In those two replications the band keeps a single coordinate (triplet about 0.92 on that coordinate and about 0.49 on the other two).

Shared systematic bias uses weight 1: every block is its own curve plus a common curve. Even the known-block concatenation falls to balanced 0.595 (sd 0.021). Isomap is 0.543. Mutual-information clustering is 0.532. The blocks are no longer pure factors, so the integration that won the earlier sections has nothing complementary to concatenate.

## What was not found

No automatically constructed Rashomon ensemble in these pilots preserves complementary geometry better than both unrestricted CoMDS of the same pool and the non-Rashomon baselines (Isomap, and concatenation when the blocks are known). The one automatic improvement over Isomap is mutual-information concatenation on axis-aligned columns. That improvement is concatenation, the Rashomon band undoes it, and rotation removes it.

The 25-replication arrays are not submitted. The S-curve array from Checkpoint 3 remains the wrong question. The Checkpoint 4 complementary array would remeasure specialist CoMDS against joint embeddings when the blocks are handed to the specialists. Concatenation of one-dimensional Isomaps already beats that CoMDS on the same manifolds, so that array would not establish a Rashomon method.

## Relation to existing methods

Sun, Huang, Parikh, and Rudin (arXiv:2604.00485, PMLR v300) define a Rashomon set as the embeddings whose single dimension-reduction loss is within δ of the best. They also add a PCA-alignment penalty on non-neighbor pairs, and they extract neighbor edges that persist inside the set. That is one loss plus an alignment term, not a union of views that are each bad at the full matrix and good on a different substructure.

An and Tang (arXiv:2512.15802) already aggregate heterogeneous embeddings with CoMDS. A result that says the consensus of several embeddings is better than one embedding repeats that paper. Bai, Wang, Long, and Hu (AAAI 2017) already fit one embedding to several distance matrices.

Zheng, Zhu, Li, Pang, Wang, and Chen (arXiv:2004.03106) regularize multi-view subspace clustering with a shared graph and a complementary graph. The output is a clustering, not a consensus of dimension-reduction embeddings, and the views are given.

The empirical question left open by those papers was whether a Rashomon rule can discover and keep complementary geometry that unrestricted aggregation and a single global loss throw away. On the manifolds and rules in this checkpoint, it does not. The positive mechanics that do show up have ordinary names: concatenate one-dimensional representations of known blocks; standardize blocks before a joint embedding when the aspect ratio would otherwise dominate; do not select embeddings for disagreement alone.
