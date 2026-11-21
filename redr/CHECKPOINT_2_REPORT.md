# Checkpoint 2 report

## 1. Research motivation

The question is whether aggregating near-optimal dimension-reduction embeddings is more trustworthy and more faithful to local and global structure than consensus on one embedding per method, or than an unrestricted ensemble of the same size.

Checkpoint 1 built the pipeline and did not find a consistent gain at a single tolerance, epsilon 0.05, on small and partly undertrained pools. This checkpoint keeps that code and those results, replaces the membership score with a fixed reference loss, and sweeps the tolerance on pools that were trained through the published schedules.

## 2. Changes since Checkpoint 1

- Embeddings passed to CoMDS are checked for a shared shape, finite values, and nondegenerate coordinates. Optional observation ids must be unique and the same length as the rows.
- Source-weight checks: all-ones weights match omitted weights; permuting views with their weights preserves the fit; splitting one view and halving its weight keeps the stress close. These are in `tests/test_bridge.py`.
- Native reference losses are in `redr/loss_validation.py`. The Checkpoint 1 probe is still computed and stored.
- Epsilon membership, including 0 and infinity, is in `mark_by_loss`. Gibbs weights take an explicit beta.
- Neighborhood vote entropy, the within/between decomposition, and three symmetric `weightmat` rules are in `redr/entropy.py`.
- Iris (75 embeddings) and an S-curve (27 embeddings) were trained at CNE 200 epochs and PaCMAP 450 epochs. Outputs are under `results/checkpoint2/`. Checkpoint 1 files were not rewritten.

The run used Linux, 4 CPUs, no GPU, Python 3.12.3, PyTorch 2.15.0+cpu, NumPy 2.2.6, scikit-learn 1.9.1, R 4.3.3, smacof 2.1-7, coMDS 0.0.0.9000. Sixteen unit tests passed, including the R bridge.

## 3. Implementation audit

The upstream trees are still RashomonDR `0ab277d` and coMDS `0570fe2`, with no source edits. `setup_vendor.sh` used to treat install caches as modifications. It now ignores `__pycache__` and `*.egg-info`.

The Checkpoint 1 contrastive probe is reproducible, not a fresh random draw. `ContrastiveLoss` reseeds to 0 in its constructor, and the probe uses only the first high-d neighbor. The native UMAP and InfoNCE score instead averages all 15 neighbors and four frozen negative draws. On Iris the term-wise losses of two UMAP embeddings correlated at 0.993. The standard error of their difference was about 0.0008, while the marginal standard deviation across terms was about 0.05. The marginal standard deviation is variation across pairs. It is not the uncertainty of a comparison that uses the same pairs.

PaCMAP training uses neighbor weight 1 and further-point weight 1 after epoch 200, and mid-near weight 0. The Checkpoint 1 probe adds the mid-near term anyway. The native PaCMAP score uses the terminal weights.

Metrics on an identical embedding reach perfect neighborhood overlap. Trustworthiness, continuity, neighborhood preservation, adjusted LCMC, triplet accuracy, and silhouette were unchanged under translation, rotation, reflection, and positive scaling in the unit tests. kNN graphs drop the point itself. Ties follow scikit-learn's neighbor order.

## 4. Native-loss validation and the Swiss-roll PaCMAP case

`results/checkpoint2/swiss_pacmap_anomaly.csv` rescores the Checkpoint 1 Swiss-roll PaCMAP embeddings and refits seed 0 at lambda 0 and lambda 5 for 450 epochs.

On the saved 220-epoch runs, seed 0:

| Lambda | Probe (NN+MN+FP) | Terminal (NN+FP) | Trustworthiness | Ambient triplet | Latent triplet |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.480 | 0.356 | 0.960 | 0.807 | 0.686 |
| 5 | 0.396 | 0.344 | 0.905 | 0.836 | 0.609 |

The mid-near term is 0.124 at lambda 0 and 0.052 at lambda 5. That gap is why the Checkpoint 1 probe prefers lambda 5. The terminal objective prefers lambda 5 by a smaller margin. Trustworthiness and the manifold-parameter triplet prefer lambda 0. The ambient triplet prefers lambda 5. So the 220-epoch disagreement is not a single bug. Local ambient neighborhoods and the latent coordinate prefer the low-alignment run. The ambient triplet and the probe prefer the high-alignment run.

After 450 epochs, which completes the final phase:

| Lambda | Probe | Terminal | Trustworthiness | Ambient triplet | Latent triplet |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.522 | 0.295 | 0.975 | 0.701 | 0.790 |
| 5 | 0.416 | 0.338 | 0.918 | 0.815 | 0.636 |

The terminal objective now agrees with trustworthiness and with latent triplet order: lambda 0 is better. The Checkpoint 1 probe still prefers lambda 5, because it still adds a mid-near term the final phase does not use. Higher trustworthiness is not the same thing as a better terminal PaCMAP loss on the shorter runs, and a better ambient triplet is not the same thing as a better unrolling.

## 5. Experimental setup

Iris: 150 points, 5 seeds, lambda in {0, 0.1, 1, 5, 10}, 200 CNE epochs, 450 PaCMAP epochs, 75 embeddings. UMAP, InfoNCE, and PaCMAP only. One shared triplet sample of 4000, seed 0, k=10.

S-curve: 400 points, noise 0.05, 3 seeds, lambda in {0, 1, 5}, same epochs, 27 embeddings. The latent score is triplet accuracy against the manifold parameter, not against the ambient coordinates.

Both pools exclude 2-epoch runs. Iris then adds one 2-epoch embedding per method in a separate contamination fit. LoCoMDS uses tau 0.1 and percentile 0.5. Epsilon was not tuned and then frozen. The curve is the result. Details are in `CHECKPOINT_2_DECISIONS.md`.

Candidate rows: `results/checkpoint2/iris/candidates.csv` and `results/checkpoint2/s_curve/candidates.csv`. Consensus rows: `sweep.csv` in the same directories.

## 6. Epsilon sensitivity

Iris native uniform consensus. Trustworthiness stays between 0.970 and 0.977 for every epsilon. Triplet accuracy moves by about 0.01 for CoMDS and by about 0.04 for LoCoMDS.

| Epsilon | Views | CoMDS triplet | LoCoMDS triplet | CoMDS trust | LoCoMDS trust |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3 | 0.900 | 0.867 | 0.975 | 0.976 |
| 0.001 | 5 | 0.903 | 0.875 | 0.976 | 0.976 |
| 0.005 | 38 | 0.889 | 0.881 | 0.974 | 0.974 |
| 0.05 | 65 | 0.897 | 0.876 | 0.975 | 0.976 |
| 0.10 | 73 | 0.897 | 0.902 | 0.970 | 0.977 |
| infinity | 75 | 0.898 | 0.906 | 0.970 | 0.977 |

UMAP native losses span only 0.0035 and InfoNCE only 0.0014, so both methods are entirely inside the set by epsilon 0.05. The views that epsilon 0.05 drops are 10 high-lambda PaCMAP embeddings. Their mean triplet accuracy is 0.927, against 0.891 for the embeddings that are kept. Their mean trustworthiness is 0.976, against 0.973. The native filter removes embeddings that are worse on the terminal PaCMAP loss and better on the external metrics. The best single Iris embedding is `pacmap_s1_l10_e450`, triplet accuracy 0.933, and it is outside the epsilon-0.05 set.

The curve is flat from epsilon 0.2 upward because every Iris candidate is inside a 20% band of its method's best native loss. Figure: `results/checkpoint2/iris/figures/epsilon_curves.png`.

S-curve, same table. Ambient trustworthiness for the tight sets is 0.998. It drops when higher-loss PaCMAP views enter.

| Epsilon | Views | CoMDS triplet | LoCoMDS triplet | CoMDS latent triplet | LoCoMDS latent triplet |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3 | 0.758 | 0.760 | 0.915 | 0.917 |
| 0.05 | 21 | 0.706 | 0.750 | 0.907 | 0.927 |
| 0.20 | 24 | 0.720 | 0.761 | 0.918 | 0.920 |
| infinity | 27 | 0.727 | 0.771 | 0.925 | 0.906 |

At epsilon 0.05 the set keeps all 9 UMAP embeddings, all 9 InfoNCE embeddings, and 3 of 9 PaCMAP embeddings. The excluded PaCMAP runs have mean latent triplet accuracy 0.699 against 0.885 for the kept runs, and mean ambient triplet accuracy 0.783 against 0.741. Native filtering removes S-curve views that are worse at unrolling and better at preserving ambient triplets.

CoMDS ambient triplet is highest at epsilon 0 (0.758) and lowest at epsilon 0.05 (0.706). LoCoMDS ambient triplet is highest on the full set (0.771). CoMDS latent triplet is slightly higher on the full set (0.925) than at epsilon 0 (0.915). The metric decides whether the tight set looks better.

## 7. Baseline comparisons

Iris, epsilon 0.05, uniform weights, 65 views:

| Comparison | CoMDS triplet | LoCoMDS triplet |
| --- | ---: | ---: |
| Native Rashomon | 0.897 | 0.876 |
| Matched unrestricted, 10 seeds, mean (sd) | 0.895 (0.0003) | 0.905 (0.0018) |
| One lowest-native embedding per method | 0.900 | 0.867 |
| Random one embedding per method, 5 seeds, mean (sd) | 0.892 (0.006) | — |
| All 75 embeddings | 0.898 | 0.906 |
| Checkpoint 1 probe set, 56 views | 0.894 | 0.912 |

On CoMDS the Rashomon triplet exceeds the matched mean by 0.002, which is larger than that matched standard deviation and smaller than the gap to the best single PaCMAP (0.933). On LoCoMDS the Rashomon triplet is 0.029 below the matched mean. The full set and the probe set both beat the native epsilon-0.05 LoCoMDS fit. One embedding per method matches epsilon 0, because each method has a unique minimum.

S-curve, epsilon 0.05, 21 views:

| Comparison | CoMDS triplet | LoCoMDS triplet | CoMDS latent | LoCoMDS latent |
| --- | ---: | ---: | ---: | ---: |
| Native Rashomon | 0.706 | 0.750 | 0.907 | 0.927 |
| Matched unrestricted, 5 seeds, mean (sd) | 0.721 (0.005) | 0.761 (0.004) | 0.911 (0.002) | 0.913 (0.006) |
| One lowest-native embedding per method | 0.758 | 0.760 | 0.915 | 0.917 |
| All 27 embeddings | 0.727 | 0.771 | 0.925 | 0.906 |

The matched draws beat native Rashomon on ambient triplet for both algorithms. The lowest-native one-per-method CoMDS fit beats the 21-view Rashomon fit by 0.052 on ambient triplet. LoCoMDS native Rashomon is ahead of its matched draws on latent triplet (0.927 versus 0.913) and behind them on ambient triplet.

The best single S-curve latent score is `pacmap_s2_l0_e450` at 0.931. That embedding is the lowest-native PaCMAP run, so it is inside every native set. Consensus does not improve on it.

Procrustes averaging and classical MDS of averaged distances land in the same band as CoMDS on Iris (triplet about 0.89 to 0.90) and do not beat the one-per-method S-curve CoMDS fit.

These are two pools. The matched standard deviations describe resampling of those pools. They are not standard errors over independently trained pools.

## 8. Weighting

On Iris, method-balanced weights, Gibbs beta in {0.25, 1, 4}, vote weights, entropy confidence, combined frequency-and-confidence, and distance-variance confidence all sit within about 0.003 triplet accuracy of uniform CoMDS at epsilon 0.05. Positive-neighborhood `weightmat` does not: trustworthiness falls to 0.955 and triplet accuracy to 0.873.

On the S-curve, strong Gibbs weighting (beta 4) at epsilon 0.05 moves CoMDS triplet accuracy from 0.706 back to 0.758 and trustworthiness from 0.988 back to 0.998, matching the three-view fit. Beta 1 only reaches 0.729. Method balancing does not. The equal-size full pools make method-balanced weights a global rescaling of uniform weights, and those fits match.

Duplicating every in-set PaCMAP embedding, Iris epsilon 0.05: uniform CoMDS triplet goes from 0.897 to 0.898, and method-balanced stays at 0.897. On the S-curve the duplicated uniform fit is 0.710 against 0.706 without the copies. Method balance removes the extra PaCMAP mass by construction. The numerical change from duplication is small next to the epsilon effect on the S-curve.

Entropy decomposition holds to numerical error on the unit tests. On these pools the entropy weights did not improve the external scores over uniform source weights. Low entropy was not evidence of a better map.

## 9. Synthetic experiment

The S-curve section above is the synthetic experiment that was run. Ambient triplet accuracy and latent triplet accuracy do not rank the epsilon curve the same way. Preserving the observed Euclidean distances is not the same as recovering the parameter along the curve. No noise, dimension, or sampling-density sweep was run.

## 10. Real benchmark experiments

Iris is the real dataset that was retrained. No new MNIST, Fashion-MNIST, COIL-20, or USPS run was executed. The Checkpoint 1 MNIST subset remains the undertrained one described in `REPORT.md`. It was not reused as a fully trained benchmark.

## 11. Robustness

Iris contamination adds three 2-epoch embeddings, one per method, to the 75 full runs. Uniform CoMDS triplet accuracy is 0.901, against 0.898 on the 75 full runs alone. Three poor views out of 78 did not damage this fit. That is one proportion, not a contamination curve.

The duplicate-weight check is in section 8. Random one-per-method CoMDS on Iris has triplet standard deviation 0.006 across 5 draws. On the S-curve the three draws have latent triplet accuracy 0.866, 0.929, and 0.862. A single random representative per method is a noisy baseline on the S-curve even when ambient trustworthiness stays high.

## 12. Negative findings

Native-loss filtering does not systematically remove embeddings with worse external scores. On Iris it removes the PaCMAP runs with the best triplet accuracy. On the S-curve it removes PaCMAP runs with worse latent scores and better ambient triplet scores, and the resulting 21-view CoMDS fit is worse on ambient triplet than both the three-view fit and the size-matched draws.

LoCoMDS on Iris improves as the pool grows. The full set has triplet accuracy 0.906. The epsilon-0.05 native set has 0.876.

UMAP and InfoNCE native losses barely move across five seeds and a decade of lambda. Most of the Rashomon set, once epsilon is a few percent, is repeated geometry from those two methods plus a few PaCMAP runs. Counting candidates overstates diversity. Iris within-method neighborhood overlap at epsilon 0.05 is 0.77.

Strong Gibbs weighting can imitate a tighter set. It does not create a gain over the lowest-native one-per-method fit on the S-curve. Positive neighborhood weights made the Iris CoMDS embedding worse.

## 13. Computational cost

Iris training took 488 seconds and the consensus jobs took 477 seconds. S-curve training took 404 seconds and the consensus jobs took 2553 seconds. A 27-view CoMDS fit on 400 points took about 77 seconds. LoCoMDS at 21 views took about 88 seconds. The dense distance matrices, not the loss evaluation, dominate once the pool exists. Stress was not used to rank these fits.

## 14. Limitations

One Iris pool and one S-curve pool. No second training repetition. Lambda was not taken down to 0.001, and neighbor size was not varied. Negative t-SNE and NCVis were not trained. MNIST was not retrained at 200 or more epochs. Epsilon-matched ensembles were repeated at epsilon 0.05 only. Common-quality and multi-criterion rules were fit at that same single tolerance. The latent score uses the manifold parameter as a 1-d distance. It is the right score for unrolling and the wrong score for ambient fidelity.

## 15. Recommendations for Checkpoint 3

The component that changes the S-curve result is which PaCMAP embeddings are allowed in, and how hard they are downweighted. UMAP and InfoNCE are not producing a Rashomon set with meaningful native-loss width under seed and PCA-lambda variation.

A useful next pool varies a hyperparameter that changes geometry inside one objective family, or evaluates that family under a fixed reference after grouping by objective. Neighbor size is a candidate, and it needs its own loss group if the training graph changes.

PaCMAP's terminal loss and the external metrics disagree often enough that a native-only gate will keep discarding embeddings the evaluation prefers, or the reverse. Checkpoint 3 should keep both numbers and treat a multi-criterion rule as a predeclared alternative, not as a replacement chosen after seeing the consensus table.

LoCoMDS on Iris wanted the larger pool. Any claim that restriction helps has to be made separately for CoMDS and LoCoMDS, and separately for ambient and latent scores.

Before a larger benchmark, the consensus step needs a cheaper path than a dense CoMDS on every epsilon. The Iris curve was flat across several epsilons that shared a membership. Caching those fits is already in the driver. Subsampling points, or fitting only the epsilons where membership changes, matters more once n is above a few hundred.

## Execution summary

Completed: environment audit, metric and weight tests, native reference loss, Iris epsilon curve, matched and one-per-method baselines, Gibbs and neighborhood-weight comparisons, a three-embedding contamination fit, a duplication check, the Swiss-roll PaCMAP rescore, and one S-curve pool with the same comparisons.

Partly completed: synthetic geometry (one S-curve, no noise or dimension grid), entropy weighting (CoMDS only, one epsilon), common-quality and multi-criterion membership (one fit each).

Not attempted: a fully trained MNIST or COIL-20 benchmark, Negative t-SNE, NCVis, a LoCoMDS hyperparameter grid, entropy-weighted LoCoMDS, and a second independent training pool.

The Iris and S-curve tables are reliable as descriptions of these two pools. They are preliminary as a general answer about REDR. The consistent result across both pools is that a native-loss gate is not the same as a gate on external structure, and that widening or shrinking the ensemble moves CoMDS and LoCoMDS in different directions.
