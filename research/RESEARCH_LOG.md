# Research log

Dates are America/Los_Angeles.

## 2026-11-21 — S1 geometry lock, before any consensus fit

Experiment ID: `S1-SCURVE-REP`.

Question. Does the Checkpoint 2 LoCoMDS gain on the S-curve survive independent datasets and independent candidate pools when recovery is scored on the unfolded surface rather than on one parameter?

Design, frozen before consensus scores exist.

- Five datasets, seeds 1000–1004. Checkpoint 2 used seed 0 and is not one of these five.
- Each pool: UMAP, InfoNCE, PaCMAP; training seeds 0, 1, 2; PCA alignment λ in {0, 1, 5}; 200 CNE epochs; 450 PaCMAP epochs. Same grid as Checkpoint 2.
- Evaluation triplets: 4000, seed 5000 + replication, shared by every method in that replication.
- Primary epsilon: 0.05. The full epsilon grid is a sensitivity curve, not a search that will be reported as the method.
- Comparators, both CoMDS and LoCoMDS: one lowest-native embedding per method, random one-per-method, unrestricted uniform, unrestricted method-balanced, unrestricted Gibbs, size-matched unrestricted draws (10 seeds), Rashomon uniform, Rashomon method-balanced, Rashomon Gibbs. Common-knowledge is the Checkpoint 1 UMAP graph refit on the epsilon-0.05 UMAP set, not a fresh launch of the upstream `experiment_2.py`.

Metric rule, applied to the point clouds only.

If adding the second intrinsic axis changes fewer than 10% of triplet decisions relative to arc length, isometric triplet accuracy is an arc-length score and the confirmatory endpoint is coordinate-balanced intrinsic triplet accuracy. Otherwise the endpoint is isometric triplet accuracy.

Result of that rule.

| Manifold | Mean disagreement | Arc span | Height span | Endpoint |
| --- | ---: | ---: | ---: | --- |
| S-curve, seeds 1000–1004 | 0.033 | ~9.4 | ~2.0 | `intrinsic_triplet_balanced` |
| Swiss roll, seeds 2000–2004 | 0.041 | (arc length, not t) | 21 | `intrinsic_triplet_balanced` |

Arc-length ordering and the raw generator parameter are still reported. On the S-curve they are the same coordinate, because the curve has unit speed. On the Swiss roll they are not. One-dimensional ordering is not called manifold recovery.

Result. Five pools, 135 embeddings, both CoMDS and LoCoMDS. On coordinate-balanced triplet accuracy, uniform LoCoMDS at epsilon 0.05 is below the full pool by 0.021 (worse in 5 of 5), below the matched draw by 0.010 (5 of 5), and below one embedding per method by 0.010 (4 of 5). The epsilon curve bottoms at 0.05–0.10.

The arc-length triplet, which is the Checkpoint 2 score on this curve, moves the other way: epsilon 0.05 beats the full pool by 0.022 in 5 of 5. Height triplet accuracy and ambient triplet accuracy move with the balanced score, not with arc length. Candidate height triplet accuracy averages 0.56.

Interpretation. The exploratory LoCoMDS edge was curve ordering. It is not recovery of the rectangle. The filter drops higher-loss PaCMAP runs. Those runs hurt arc order and help the transverse coordinate. Because UMAP and InfoNCE barely vary in native loss, the band mostly edits the PaCMAP slice.

The full table is in `research/CHECKPOINT_3_REPORT.md`. The 25-replication cluster array should not be launched to confirm a gain that the pre-registered endpoint did not show.

## 2026-11-21 — S4 controlled embeddings, mechanism pilot

Experiment ID: `S4-CONTROLLED`.

Question. When does a within-method loss threshold change a coordinate average, relative to a size-matched draw that ignores the score?

The candidates are perturbations of a known plane. They are not DR methods. The observable score is either the latent Procrustes error (oracle upper bound), a noisy monotone function of that error, independent noise, or the negated error. Aggregation is a weighted Procrustes mean, not CoMDS. A pooled threshold, which can drop every embedding from a bad method, is reported separately from the method-specific rule.

Result. Twenty-five replications. The numerical contrasts are in `NEGATIVE_RESULTS.md` and `results/checkpoint3/controlled_ensembles/paired.csv`.

Interpretation. A within-method threshold is not a general improvement over averaging every candidate. It helps modestly when the score tracks latent error and a large fraction of candidates are junk. It hurts when the score is noise or is anti-aligned, because the retained set is small and the average over the full pool was already robust. It does not remove a bias shared by the best candidate of a method. A pooled threshold can remove that method, and that rule needs its own experiment before it is treated as a method. Gibbs reweighting of this coordinate average did nothing of practical size.

This does not yet say what CoMDS or LoCoMDS will do on the S-curve. It does say that a win against a size-matched draw, without a win against the full pool, is not evidence for the filter. The S-curve summary will report both.

## Next

Finish the five S-curve pools and their consensus fits. Read the balanced intrinsic triplet first. Do not expand to 25 cluster replications unless that endpoint, against the full pool and against one embedding per method, is actually promising. Swiss roll stays locked and untrained until that reading.

## 2026-11-21 — Complementary views, after the S-curve reading

Experiment IDs: `S8-COMPLEMENTARY`, `S8-CONTROLLED-COMDS`.

The S-curve reading is in the Checkpoint 3 report. The balanced endpoint did not favor epsilon 0.05. That array stays unsubmitted.

Question. If some embeddings preserve one intrinsic direction and others preserve a different one, which selection rule keeps both, and which aggregator can use them?

Design, written to `results/checkpoint4/complementary/study_lock.json` before the replication grid. A one-replication smoke with PCA-initialized t-SNE is archived under `complementary_smoke` and is not in the table. t-SNE in the pilot uses random initialization, because PCA initialization made the two seeds identical.

- Square, standardized, both curves present. Seeds 4000–4003. This is the favorable case.
- Stretched, aspect (5, 1), scale preserved in the observations. Seeds 4300–4303. Joint distances are dominated by the long axis.
- Hidden second curve. Seeds 4200–4203. The unfavorable case where the direction is not in the data.
- A condition named `long` only rescaled the evaluation coordinates. It is an extra square draw, not an aspect test.
- Structural loss is fixed: one minus multiscale neighborhood preservation on the full observed matrix. Geodesic correlation is stored and not used for membership.
- Rules: all, method-specific epsilon 0.05, pooled structural epsilon 0.05, hybrid, diversity inside the pooled set, joint only, specialist only, best single structural embedding.
- Aggregators: Procrustes, CoMDS, LoCoMDS.

Result, four replications. Full table in `research/CHECKPOINT_4_REPORT.md`.

On the square, specialist LoCoMDS balanced triplet accuracy is 0.969. It beats joint LoCoMDS by 0.090 (4 of 4), pooled LoCoMDS by 0.102 (4 of 4), and the best structural embedding by 0.136 (4 of 4). The same fits lose trustworthiness to the joint consensus by 0.029 (0 of 4). The method-specific band changes the score by +0.002. The structural band keeps joint views only.

On the stretched surface, joint CoMDS has v triplet 0.512 and u triplet 0.974. Specialist CoMDS has balanced accuracy 0.957 and beats joint CoMDS by 0.204 (4 of 4), at a cost of 0.217 on the long-axis triplet and 0.044 on trustworthiness. LoCoMDS on that specialist pool failed on two of four replications (0.693 and 0.727) while CoMDS did not.

When the second curve is noise, v triplet accuracy is 0.50 for every rule, and the specialist pool is worse than the joint pool.

Neighborhood entropy does not predict latent neighborhood error in these four replications.

The controlled CoMDS grid (n = 80, four replications) separates two mechanisms. On complementary damage, LoCoMDS of every view beats Procrustes and beats a quality filter. On a majority that has destroyed one axis, a pooled score that tracks the error beats the full pool, and a within-method band does not. An uninformative score makes the pooled rule worse. Details and the figure are in the Checkpoint 4 report.

## 2026-11-21 — Checkpoint 5, automatic complementary geometry

Question. Once the blocks are not given to the method, does a multi-objective Rashomon rule find and keep complementary directions better than unrestricted CoMDS and better than Isomap or one-dimensional concatenation?

Design, fixed before the latent scores of each pilot.

- Privileged baselines on the Checkpoint 4 seeds. Four replications. Concatenated 1D Isomap, block-standardized Isomap and PCA, classical MDS and SMACOF of averaged block distances, CoMDS of the two block configurations, and source-balanced CoMDS of the saved Checkpoint 4 pool.
- Metric audit on the same seeds. Raw latent coordinates, axis-equalized latent coordinates, and a one-axis embedding, scored on balanced, isometric, and coordinate-wise triplets.
- Automatic pool. Three replications, n = 160, seeds 6000–6502. Random subsets, random projections, Pearson feature clusters, two Isomap neighborhood sizes, PCA. Five structural losses. Rules: all CoMDS, all Procrustes, pooled multiscale, union of objective bands, Pareto, marginal subset gain, diversity with no quality gate, quality-weighted CoMDS, best single. Epsilon 0.05. Rotation, hidden second curve, four noise embeddings, aspect 5 with scale preserved, 20 nuisance features, noise 0.25.
- Follow-up, new seeds 8000–8602, after the automatic pool failed to isolate axes. Mutual-information clustering versus Pearson. A 5% band on one-dimensional projection stress. Agreement clustering with no quality gate. Three curves. A shared curve added to every block at weight 1.

Result. Full table in `research/CHECKPOINT_5_REPORT.md`.

On the stretched surface the raw latent coordinates score 0.782 balanced and 1.000 isometric. Equalizing the axes scores 1.000 balanced and 0.782 isometric. Concatenated 1D Isomap matches the equalized geometry (0.986 balanced, 0.787 isometric). Isomap on the full matrix matches the raw geometry (0.759 balanced, 0.965 isometric).

With known blocks, concatenated 1D Isomap beats CoMDS of the block configurations and beats source-balanced CoMDS of the Checkpoint 4 pool (square 0.989 versus 0.945 and 0.956). CoMDS beats classical MDS of the averaged distances and does not beat concatenation. This is not a Rashomon result.

Without block labels, on rotated features, pooled multiscale CoMDS beats unrestricted CoMDS by 0.027 (3 of 3) and ties Isomap (+0.003, 2 of 3). Union, Pareto, and quality weighting stay within 0.02 of unrestricted CoMDS. All of them lose to privileged concatenation by at least 0.14 (3 of 3). Diversity-first loses to unrestricted CoMDS by 0.169 (3 of 3), and by 0.211 when four noise embeddings are added. No automatic rule recovers the short axis of the stretched rotated surface.

Mutual-information concatenation beats Isomap by 0.065 on aligned columns (3 of 3) and loses to it by 0.180 after rotation (0 of 3). The 5% self-view band drops the second curve in 2 of 3 aligned replications (−0.131). Stress pairs and agreement pairs lose to Isomap by about 0.25. Three latent directions do not fit in a two-dimensional embedding: known-block concatenation ties Isomap, and the Rashomon band is worse by 0.072. A shared curve of weight 1 drops even the known-block concatenation to 0.595.

## Next, after Checkpoint 5

Do not submit the S-curve confirmation array. Do not submit `experiments/study8_complementary/` as a Rashomon study: concatenation of the known blocks already beats the specialist CoMDS that array would remeasure. No new 25-replication array is prepared. A later study needs a selection rule that, without block labels and after feature mixing, beats both unrestricted aggregation of the same pool and Isomap. Swiss roll stays untrained.

## 2026-11-21 — Checkpoint 6, the original consensus question

Question. On the simulations that motivated CoMDS and on the parametric families from RashomonDR, does a method-specific Rashomon set improve CoMDS or LoCoMDS relative to aggregating the same embeddings without that restriction?

Design, fixed before the parametric scores.

- An and Tang Gaussian mixture, n = 600, manuscript covariance, NumPy seed 331. Swiss roll, n = 400, manuscript parameter interval, same seed. Eleven views: PCA, classical MDS, LLE, Isomap, Laplacian eigenmaps, kernel PCA at the two manuscript sigmas, t-SNE at perplexity 30 and 50, UMAP at 30 and 50 neighbors. Meta-Spec kernel PCA. CoMDS. Three LoCoMDS settings, with the reported choice maximizing LCMC at k = 20.
- Parametric pool: UMAP, InfoNCE, negative t-SNE at s = 0, noise-contrastive estimation (the vendored NCVis loss), PaCMAP. Seeds 0 and 1. Alignment weights 0, 1, and 5. Fifty CNE epochs, 200 PaCMAP epochs. Method-specific 5% native-loss band. Median neighborhood-overlap band. Quality weights with equal method mass. Five size-matched draws. Common-knowledge refit from the vendored mean-score procedure.
- After the embeddings were saved, Definition 3.2 was applied as a median of mean soft-Jaccard distance inside each method, and that subset was aggregated with CoMDS. The median threshold matches the form used for the overlap band.
- One Gaussian copy with observation noise 0.25. Four classical views plus their CoMDS. Neighbor-vote entropy on the parametric Gaussian pool.

Result. Full table in `research/CHECKPOINT_6_REPORT.md`.

PCA on the Gaussian mixture has adjusted Rand 0.972. CoMDS, LoCoMDS, UMAP, t-SNE, and Meta-Spec kernel PCA are all at or above 0.97. The visible difference is shape: CoMDS keeps elliptical clouds, and Meta-Spec, t-SNE, and UMAP tighten or bend them. On the Swiss roll, classical CoMDS forms a U (balanced triplet 0.672, end gap 13). LoCoMDS on the reduced grid has balanced triplet 0.654 and end gap 8.0. UMAP has balanced triplet 0.771 and end gap 39. The manuscript's LoCoMDS unfold was not reproduced at this sample size and hyperparameter grid.

The 5% loss band keeps all 30 Gaussian embeddings and 26 of 30 Swiss embeddings. The Swiss drops are InfoNCE and PaCMAP at lambda 5. Loss-band CoMDS scores 0.753 against unrestricted CoMDS at 0.750 and the best InfoNCE embedding at 0.792. Definition 3.2 CoMDS scores 0.750. Common-knowledge UMAP scores 0.706. Vote entropy and false-neighbor rate have Spearman correlation 0.068.

## Next

Pause the original REDR direction. Do not submit a 25-replication array, and do not add another selection rule on the product manifolds or on these two simulations. CoMDS of heterogeneous embeddings remains An and Tang's method. These pilots do not show that a Rashomon restriction is what makes that consensus work.

## 2026-11-25 — Checkpoint 7, quality by diversity

Experiment IDs: `S10-QD`, `S10-QD-CONFIRM`, `S10-DR-MECH`.

Checkpoint 6 recommended pausing the search for a manifold on which a Rashomon band wins. This checkpoint does not resume that search. The question is which quality and diversity conditions move a within-method band, relative to unrestricted CoMDS, Gibbs weights, a global trim, the best single embedding, and a size-matched draw.

Design, frozen in `experiments/study10_quality_diversity/config.yaml` before any consensus score from this study. Square, n = 80, three methods, four draws, four replications, seed 11000. Quality is high, mostly high, mixed, severe, or a shared axis shrink. Diversity is identical, a smooth warp, complementary axis damage, or one method that is a row permutation. The selection score is a noisy copy of latent error, a weakly informative copy, uniform noise, or the reflected error. Oracle rows copy the latent error and are labeled. Epsilon grid 0, 0.05, 0.25, 1, infinity. Primary epsilon 0.05. LoCoMDS on eleven named cells only.

Result. Full table in `research/CHECKPOINT_7_REPORT.md`.

On high identical copies the band loses 0.019 to the full pool (0 of 4). On severe identical copies it gains 0.042 (4 of 4) and loses 0.009 to full-pool Gibbs (0 of 4). On high harmful it loses 0.403 (0 of 4): the band keeps 3.5 of the mis-paired method's draws out of 5.5 retained, and uniform CoMDS follows them. Method-balanced weights on that same set score 0.924. The oracle band still scores 0.671 against 0.933 for the pool that ignores the score. The complementary prediction failed: mixed complementary is −0.001 versus the full pool and −0.026 versus Gibbs. Misleading scores drop the band by 0.365 (0 of 4). Epsilon 0.05 is the worst point on the harmful curve.

Fresh seeds 21000, five frozen cells, no LoCoMDS. Severe identical stays at +0.035 versus the pool and −0.002 versus Gibbs. The harmful mean is −0.201 (1 of 4 positive). The positive replication is the one that kept a faithful majority.

Swiss roll, seeds 3000–3003, n = 180. Isomap, t-SNE, UMAP, jittered PCA, plus a row-rolled copy of each t-SNE as its own family. Score is 1 minus trustworthiness against the observations. The 5% band keeps all four permuted embeddings in 4 of 4 replications and scores 0.599 against unrestricted CoMDS at 0.638 and a global trim at 0.675. Gibbs on the band scores 0.646.

## Next

Do not submit a 25-replication array. No cell beat unrestricted CoMDS, Gibbs, the trim, the best embedding, and a size-matched draw together. The next measurement is descriptive: within-method loss span and between-method disagreement on fully trained UMAP, InfoNCE, PaCMAP, NCVis, and t-SNE, with a longer schedule than Checkpoint 6, and no new epsilon.
