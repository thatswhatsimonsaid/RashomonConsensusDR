# Checkpoint 6 — Return to the original research question

Local pilots, 21 November 2026 (America/Los_Angeles). Checkpoints 3, 4, and 5 are unchanged. No cluster array was submitted.

Checkpoint 5 did not find an automatic rule that discovers complementary views. This checkpoint stops adding selection rules on those product manifolds. The question is the original one: once several dimension-reduction methods have produced their own embeddings, does restricting each method to a Rashomon set improve the CoMDS or LoCoMDS consensus?

The pilot does not support that restriction. On the Gaussian mixture every serious embedding already separates the three clusters, so consensus cannot raise adjusted Rand. On the Swiss roll the best single parametric embedding beats every consensus, and the 5% loss band changes the balanced triplet by +0.003. The recommendation is to pause the original REDR direction.

## What was reproduced, and what was not

Both papers were read, including the appendices that specify the algorithms. An and Tang, arXiv:2512.15802, Sections 3.1–3.2 and Appendices A, B, C, and E. Sun, Huang, Parikh, and Rudin, arXiv:2604.00485, Definitions 3.1–3.3, Sections 4 and 5, and Appendices B, C, E, and H.

The executable An and Tang design is `coMDS-manuscript/R/dgps.R` and the 16-method list in Section 3. This machine has the coMDS R package, scikit-learn, and umap-learn. It does not have uwot, Rtsne, phate, lle, dimRed, or multiSNE. The fitted subset is PCA, classical MDS, LLE, Isomap, Laplacian eigenmaps, kernel PCA at sigma 0.01 and 0.001, t-SNE at perplexity 30 and 50, and UMAP at 30 and 50 neighbors. Omitted, and not imputed: non-metric MDS, Sammon, HLLE, PHATE, and multi-SNE. Meta-Spec UMAP is omitted. Meta-Spec kernel PCA is ported from `methods-utils.R` and the kernel-PCA branch of `fit_metaspec`.

Kernel PCA uses kernlab's rbfdot convention, `exp(-sigma ||x-x'||^2)`, so scikit-learn's gamma equals sigma. An earlier mapping `gamma = 1/(2 sigma^2)` collapsed both kernel-PCA views. The reported classical consensus was refit after that correction.

NumPy `RandomState(331)` is not R `set.seed(331)`. The formulas match the manuscript: cluster sizes 150, 200, 250; the stated means; covariance `Sigma + Z Z^T` with `Z_ij ~ N(0, 0.5^2)`; then column scaling. The Swiss roll uses the manuscript interval `t ~ Uniform(pi, 3.5 pi)` and `n = 400`. The paper's prose uses `3.4 pi` and `n = 1000`. LoCoMDS is three settings, `(tau, percentile) in {(0.1, 0.5), (1.0, 0.5), (0.1, 0.3)}`, and the reported selection maximizes LCMC at `k = 20`. The paper tunes a 9 by 9 grid by adjusted LCMC across several neighborhood sizes.

Sun et al. build 235 embeddings per method: 5 seeds times 47 alignment weights. This pilot uses seeds `{0, 1}` and `lambda in {0, 1, 5}`, 30 embeddings per dataset. Parametric UMAP, InfoNCE, and negative t-SNE are contrastive-ne loss modes `umap`, `infonce`, and `neg`. Negative t-SNE is run at spectrum `s = 0`. NCVis in that paper is the vendored noise-contrastive loss, loss mode `nce`. There is no separate NCVis package. PaCMAP is ParamPaCMAP. CNE runs 50 epochs. PaCMAP runs 200 epochs. The native-loss reference for PaCMAP uses the seed-0 pair set for every PaCMAP embedding.

USPS, MNIST, and COIL-20 were not refit. A faithful Sun et al. grid is 235 embeddings per method, COIL-20 is not in the vendored checkout, and Checkpoint 1 already found that a 40-epoch MNIST run does not separate digits. The simulation pilot did not produce a hypothesis that would justify that grid.

## Phase 1 — An and Tang's simulations

Scores are in `results/checkpoint6/classical.csv`. One sample of each simulation. Adjusted Rand uses KMeans with three clusters on the standardized embedding and the true labels. The Swiss balanced triplet uses 3,000 triplets against `(t, width)` after each axis is scaled to unit variance. The end gap is the mean embedding distance between the 30 smallest-`t` points and the 30 largest-`t` points, divided by the median 10-nearest-neighbor distance. A larger end gap means the two ends of the roll stay apart. `within_between` is the mean within-cluster pairwise distance divided by the mean distance between cluster centroids. Labels and `t` are evaluation only.

### Gaussian mixture

| Embedding | Adjusted Rand | Neighbor purity | Trustworthiness | within / between |
| --- | ---: | ---: | ---: | ---: |
| PCA, classical MDS, both kernel PCAs | 0.972 | 0.996 | 0.973 | 0.31 |
| Isomap | 0.781 | 0.995 | 0.946 | 0.16 |
| Laplacian eigenmaps | 1.000 | 1.000 | 0.929 | 0.045 |
| t-SNE, perplexity 30 | 1.000 | 1.000 | 0.992 | 0.25 |
| UMAP, 30 neighbors | 1.000 | 1.000 | 0.984 | 0.12 |
| Meta-Spec kernel PCA | 1.000 | 1.000 | 0.966 | 0.10 |
| CoMDS of the 11 views | 1.000 | 1.000 | 0.978 | 0.20 |
| LoCoMDS, tau 0.1, percentile 0.5 | 0.976 | 1.000 | 0.973 | 0.29 |
| LoCoMDS selected by LCMC at k = 20 | 0.939 | 0.998 | 0.981 |  |

PCA already separates the clusters. CoMDS, LoCoMDS, t-SNE, UMAP, and Meta-Spec all reach adjusted Rand at or above 0.97. Consensus has no accuracy left to gain. The shape claim in the paper is visible: PCA and CoMDS keep elliptical clouds, t-SNE and UMAP pack them, and Meta-Spec kernel PCA bends them into crescents. LoCoMDS at the default local setting also keeps three clouds. Selecting LoCoMDS by LCMC at a single `k` lowers adjusted Rand from 0.976 to 0.939. That selection is our reduced stand-in, not their plurality-of-`k` procedure.

### Swiss roll

| Embedding | Balanced triplet | Arc triplet | Height triplet | End gap | Trustworthiness |
| --- | ---: | ---: | ---: | ---: | ---: |
| PCA | 0.670 | 0.608 | 0.655 | 8.7 | 0.882 |
| Isomap | 0.625 | 0.778 | 0.511 | 6.6 | 0.897 |
| LLE | 0.749 | 0.619 | 0.734 | 8.2 | 0.857 |
| UMAP, 30 neighbors | 0.771 | 0.907 | 0.543 | 38.9 | 0.997 |
| t-SNE, perplexity 30 | 0.704 | 0.806 | 0.541 | 13.2 | 0.998 |
| Meta-Spec kernel PCA | 0.647 |  |  |  | 0.939 |
| CoMDS of the 11 views | 0.672 | 0.786 | 0.543 | 13.0 | 0.982 |
| LoCoMDS, tau 0.1, percentile 0.5 | 0.654 | 0.616 | 0.637 | 8.0 | 0.902 |
| LoCoMDS, tau 0.1, percentile 0.3 | 0.644 | 0.624 | 0.622 | 6.3 | 0.899 |

PCA flattens the roll. Isomap closes it into a loop, so the height triplet falls to 0.511 and the end gap is 6.6. t-SNE tears the surface into clumps. CoMDS forms the U described in the paper: arc triplet 0.786 and end gap 13. UMAP opens the roll further (end gap 38.9, balanced triplet 0.771). LoCoMDS does the opposite of the paper's unfold on this sample: its end gap is 8.0, below CoMDS, and its balanced triplet is 0.654. LCMC at `k = 20` is 0.633 for CoMDS and 0.582 for the default LoCoMDS, so the local objective does not prefer LoCoMDS here either. The manuscript's claim that LoCoMDS unfolds the roll was not reproduced at `n = 400` with this 11-method subset and this three-point hyperparameter grid.

## Phase 2 — Parametric families

Thirty embeddings per dataset: five methods, two seeds, three alignment weights. Method-specific membership is `L <= L* + 0.05 max(|L*|, 1e-8)` on the native loss. Losses are not pooled across methods.

On the Gaussian mixture the relative loss span inside every method is at most 0.045. The 5% band keeps all 30 embeddings. On the Swiss roll it keeps 26. The four dropped embeddings are InfoNCE and PaCMAP at `lambda = 5`, both seeds. Their native losses sit 6% to 14% above the best seed of the same method. The band removes the stronger PCA-alignment settings. It does not remove a failed seed.

The graph rule used for aggregation is pre-specified and is not Definition 3.2. Inside each method, the central embedding is the one with the highest mean 15-nearest-neighbor overlap, and embeddings at or above the median overlap with that center are kept. That keeps 15 of 30 on both datasets. Definition 3.2 was scored afterwards on the same saved embeddings: standardize each embedding, build `W` from the high-dimensional 15-nearest neighbors with `gamma = 1`, and keep embeddings whose mean soft-Jaccard distance to the other embeddings of the same method is at most the median. That rule also keeps 15. It agrees with the overlap rule on 20 of 30 Gaussian embeddings and 24 of 30 Swiss embeddings (`results/checkpoint6/definition32.csv`).

Common-knowledge extraction follows the vendored `experiment_2` mean score: candidate edges are the high-dimensional 15-nearest neighbors, and each point keeps its 10 lowest-score neighbors. The refit is parametric UMAP for 50 epochs on that graph. Appendix B's calibration-test split and the population penalty `Psi` were not implemented. With six embeddings per method there is no separate calibration set that would make that split meaningful.

## Phase 3 — Head-to-head

Parametric scores are in `results/checkpoint6/parametric.csv`. Definition 3.2 consensus is in `results/checkpoint6/definition32_consensus.csv`. One draw of each dataset. Size-matched draws use the loss-band count, so on the Gaussian mixture they are the full set of 30.

### Gaussian mixture, adjusted Rand

Every parametric consensus has adjusted Rand 1.000 and trustworthiness 0.981: unrestricted CoMDS, unrestricted LoCoMDS, the loss band, method-balanced weights, the overlap graph band, quality weights, and all five size-matched draws. The best InfoNCE, NCVis (`nce`), negative t-SNE, and UMAP embeddings also have adjusted Rand 1.000. The best PaCMAP embedding has 0.991. Common-knowledge UMAP has 0.996. Definition 3.2 CoMDS has adjusted Rand 1.000 and trustworthiness 0.981. The Rashomon restriction has nothing to improve.

Wall time for CoMDS: unrestricted 58 s, loss band 58 s, overlap graph band 21 s, Definition 3.2 39 s. The graph band is faster because it keeps 15 views on a saturated task. The loss band keeps every view.

### Swiss roll, balanced triplet

| Rule | Views | Balanced triplet | Trustworthiness | Seconds |
| --- | ---: | ---: | ---: | ---: |
| Best InfoNCE | 1 | 0.792 | 0.997 |  |
| Best NCVis (`nce`) | 1 | 0.792 | 0.997 |  |
| Best negative t-SNE | 1 | 0.778 | 0.995 |  |
| Best parametric UMAP | 1 | 0.753 | 0.993 |  |
| Best PaCMAP | 1 | 0.723 | 0.983 |  |
| Unrestricted CoMDS | 30 | 0.750 | 0.991 | 36 |
| Unrestricted LoCoMDS | 30 | 0.735 | 0.991 | 72 |
| Loss-band CoMDS | 26 | 0.753 | 0.992 | 52 |
| Method-balanced loss-band CoMDS | 26 | 0.750 | 0.991 | 48 |
| Overlap graph-band CoMDS | 15 | 0.748 | 0.991 | 36 |
| Definition 3.2 median CoMDS | 15 | 0.750 | 0.991 | 34 |
| Quality-weighted CoMDS | 30 | 0.751 | 0.991 | 37 |
| Size-matched CoMDS, mean of 5 draws | 26 | 0.750 | 0.991 | 30 |
| Common-knowledge UMAP | 1 | 0.706 | 0.984 |  |
| Classical CoMDS, 11 views | 11 | 0.672 | 0.982 | 36 |
| Classical UMAP, 30 neighbors | 1 | 0.771 | 0.997 |  |

The loss band is 0.003 above unrestricted CoMDS and 0.039 below the best InfoNCE embedding. The five size-matched draws sit between 0.749 and 0.751, so the 0.003 difference is the size of the draw-to-draw variation of CoMDS on a subset of the same pool. Quality weights, method-balanced weights, the overlap band, and Definition 3.2 stay within 0.002 of unrestricted CoMDS. Common-knowledge UMAP is 0.044 below unrestricted CoMDS and 0.047 below the best parametric UMAP. Unrestricted parametric LoCoMDS is 0.015 below unrestricted CoMDS.

The loss-band fit took 52 s against 36 s for the full pool. Keeping 26 of 30 views did not reduce the aggregation cost.

## Phase 4 — Stability and whether agreement tracks error

One noisy copy of the Gaussian mixture, `RandomState(7)`, standard deviation 0.25, then column scaling. PCA, Isomap, t-SNE at perplexity 30, UMAP at 30 neighbors, and CoMDS of those four were refit. Every embedding is scored against the clean matrix and the true labels (`results/checkpoint6/noise.csv`).

| Embedding | Clean adjusted Rand | Noisy adjusted Rand | Procrustes RMSE to the clean fit |
| --- | ---: | ---: | ---: |
| PCA | 0.972 | 0.935 | 0.142 |
| Isomap | 0.781 | 0.956 | 0.385 |
| t-SNE, perplexity 30 | 1.000 | 0.962 | 0.200 |
| UMAP, 30 neighbors | 1.000 | 0.940 | 0.235 |
| CoMDS of these four | 1.000 | 0.956 | 0.148 |

CoMDS moves about as much as PCA and less than t-SNE or UMAP. Isomap's adjusted Rand rises because the clean Isomap was the distorted one. This is a property of ordinary CoMDS on four views. The loss band kept every Gaussian parametric embedding, so a Rashomon ensemble and the unrestricted ensemble are the same object on this mixture. The noise pilot does not contain a Rashomon-versus-full comparison.

Neighbor-vote entropy on the 30 parametric Gaussian embeddings has Spearman correlation 0.068 with the fraction of the top-10 voted neighbors that carry a different true label. Mean vote purity is 0.9997. The loss-band pool is the same 30 embeddings, so the two correlations match by construction. Low entropy is not evidence of correctness here: almost every voted neighbor is already correct, and the entropy that remains does not rank the rare errors.

## Phase 5 — Decision

A. Consensus quality. The best single parametric embedding is ahead of every consensus on the Swiss roll. On the Gaussian mixture the label score is saturated before any consensus is fit.

B. Robustness. The noise shift favors ordinary CoMDS over t-SNE and UMAP by a small Procrustes margin, and that comparison does not involve a Rashomon set. The loss band was inactive on the clean mixture.

C. Uncertainty. Vote entropy and neighbor error have Spearman correlation 0.068. The Rashomon pool and the full pool produce the same votes.

D. Aggregation cost. The loss band does not shrink the Gaussian pool and makes the Swiss CoMDS fit slower (52 s versus 36 s). A median graph rule keeps half the embeddings by construction. On the Gaussian mixture that half is enough for adjusted Rand 1 and the fit drops from 58 s to 21 s. On the Swiss roll the same halving leaves the fit at 36 s and the triplet at 0.748. A rule that always discards half of an already exchangeable pool is not a demonstrated saving.

E. When the filter matters. On this alignment grid the native loss barely moves, and the embeddings the 5% band does drop are the `lambda = 5` fits. Removing them changes the Swiss triplet by +0.003. Checkpoint 5 already showed the same style of band dropping a complementary view and lowering the balanced triplet by 0.131. The filter is inert when candidates are exchangeable and harmful when they carry different geometry. That description does not identify a regime in which the filter should be used.

None of A–E has support that would justify another round of selection rules or a 25-replication array. The original REDR direction should pause. CoMDS itself remains a reasonable way to average embeddings that disagree in shape, which is An and Tang's claim and is visible in the Gaussian figure. Attributing that average to a Rashomon set is not supported by these pilots.

## Figures

`results/checkpoint6/figures/gaussian_comparison.png` shows PCA, t-SNE, UMAP, Meta-Spec kernel PCA, CoMDS, and LoCoMDS. `results/checkpoint6/figures/swiss_comparison.png` adds the parametric CoMDS, the loss-band CoMDS, and the common-knowledge refit, colored by the roll parameter `t`.
