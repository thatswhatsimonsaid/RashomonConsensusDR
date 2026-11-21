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
