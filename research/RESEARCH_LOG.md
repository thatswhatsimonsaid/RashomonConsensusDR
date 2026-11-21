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

## Next

Do not submit the S-curve confirmation array. The complementary array is described by `experiments/study8_complementary/generate_sbatch_arrays.py` and was not submitted. It repeats square, stretched, and hidden-second for 25 seeds that do not overlap this pilot, so the specialist-versus-joint gap, the structural filter, and the LoCoMDS failures can be checked independently. Swiss roll stays untrained.
