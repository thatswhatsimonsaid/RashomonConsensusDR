# Checkpoint 3 report: S-curve replication and a mechanism pilot

This is a record of what was run. It is not a claim that REDR improves dimension reduction.

## Question

Checkpoint 2, on one S-curve pool, reported LoCoMDS latent triplet accuracy 0.927 at native epsilon 0.05, against 0.913 for a size-matched unrestricted draw, 0.906 for the full pool, and 0.917 for one lowest-native embedding per method. That score used only the generator parameter.

The confirmatory question was whether that advantage is recovery of the unfolded surface. The surface has two coordinates: arc length along the curve, and the transverse height. On this sampling, arc length spans about 9.4 and height spans about 2.0. Adding height changes 3.3% of random triplet decisions (seeds 1000–1004). Isometric triplet accuracy is therefore an arc-length score. The rule fixed before any consensus fit, in `results/checkpoint3/s_curve_replication/geometry_lock.json`, sets the confirmatory endpoint to coordinate-balanced intrinsic triplet accuracy. Arc-length ordering is reported separately and is not called manifold recovery.

## Design

Five datasets, seeds 1000–1004. Checkpoint 2 used seed 0 and is not in these five. Each pool has UMAP, InfoNCE, and PaCMAP, training seeds 0, 1, and 2, PCA alignment λ in {0, 1, 5}, 200 CNE epochs, and 450 PaCMAP epochs. 135 embeddings. Epsilon 0.05 was fixed. The epsilon grid is a sensitivity curve. Evaluation uses 4000 shared triplets per dataset, seed 5000 plus the replication index.

LoCoMDS and CoMDS both use the Checkpoint 2 defaults (τ = 0.1, percentile = 0.5). Common knowledge is the adapted UMAP graph refit from Checkpoint 1, not a launch of the upstream script.

## Result on the confirmatory endpoint

Uniform LoCoMDS at epsilon 0.05 is worse than the full pool, worse than a size-matched draw, and worse than one lowest-native embedding per method.

Mean balanced triplet accuracy, five replications:

| LoCoMDS comparison | Mean | Rashomon minus this | Replications Rashomon is higher |
| --- | ---: | ---: | ---: |
| Uniform Rashomon, ε = 0.05 | 0.790 |  |  |
| Full pool | 0.811 | −0.021 (sd 0.012) | 0 of 5 |
| Size-matched draw | 0.800 | −0.010 (sd 0.006) | 0 of 5 |
| One lowest-native per method | 0.800 | −0.010 (sd 0.007) | 1 of 5 |
| Gibbs on the Rashomon set | 0.798 | −0.008 | 0 of 5 |
| Best single candidate (oracle) | 0.808 |  |  |

CoMDS shows the same direction. One embedding per method (0.800) beats the epsilon-0.05 set (0.769) in 5 of 5 replications, difference −0.031 (sd 0.008). The full CoMDS pool (0.785) beats it in 4 of 5.

The epsilon curve for LoCoMDS balanced accuracy is highest on the full pool (0.811 for every ε ≥ 0.5, where every candidate is kept) and at ε = 0 (0.800, one embedding per method). It is lowest at ε = 0.05 and 0.10 (0.790). The pre-specified tolerance is the bottom of this curve. It was not changed after the fact.

Method balancing does not explain the gap. At ε = 0.05 the pools are unequal (typically all or most UMAP and InfoNCE runs, and 3 or 4 of 9 PaCMAP runs). Method-balanced LoCoMDS is 0.788, no better than uniform. On the full pool every method has 9 embeddings, so method-balanced weights are a rescaling of uniform weights and the fits match (both 0.811).

## The Checkpoint 2 number does replicate, on one axis

The same LoCoMDS fits, scored only on arc length, go the other way.

| Arc-length triplet, LoCoMDS | Rashomon ε = 0.05 minus comparator | Higher in |
| --- | ---: | ---: |
| Full pool | +0.022 (sd 0.004) | 5 of 5 |
| Size-matched draw | +0.010 (sd 0.004) | 5 of 5 |
| One per method | +0.006 (sd 0.005) | 4 of 5 |

Isometric triplet accuracy, which is dominated by arc length, agrees with that column (+0.016 versus the full pool, 5 of 5). Height triplet accuracy agrees with the balanced column instead: Rashomon minus the full pool is −0.023 (sd 0.011), higher in 0 of 5. Ambient triplet accuracy is also worse (−0.016, 0 of 5). Trustworthiness stays near 0.998 for every LoCoMDS fit; the differences are a few ten-thousandths.

So the exploratory latent-triplet edge was real as curve ordering, and it is bought by giving up the transverse coordinate and ambient triplet order. That is not surface recovery.

Height is weak in the whole pool, not only in the consensus. Mean height triplet accuracy is 0.54 for UMAP, 0.56 for InfoNCE, and 0.59 for PaCMAP. The best candidate reaches 0.69. Chance on a random triplet is 0.5. None of these embeddings lays the rectangle out so that height is a coordinate. A consensus step cannot restore a direction the candidates do not contain. The gallery for replication 0 is `results/checkpoint3/s_curve_replication/figures/rep00_locomds_gallery.png`. Individual embeddings are in the preview written with the replication.

## What the filter removes

UMAP native losses span 0.015 and InfoNCE spans 0.029, so those methods are mostly inside ε = 0.05. PaCMAP spans 0.080. The sets keep 3 or 4 of 9 PaCMAP runs and almost every UMAP and InfoNCE run (18 to 22 views out of 27). Dropping the higher-loss PaCMAP runs improves arc-length order and hurts height, balanced accuracy, and ambient triplets. The full pool wants those runs for the second criterion and not for the first.

## Controlled embeddings

A separate 25-replication pilot builds candidates by perturbing a known plane and averages them with Procrustes. It is not CoMDS and not UMAP. Oracle rows copy the latent error into the score and are upper bounds.

- An uninformative score does not beat a matched draw, and it loses to the full pool. With 40% junk embeddings and an uninformative score, the filtered average is worse than keeping everyone by 0.106 (22 of 25).
- A misleading score makes filtering worse than the matched draw (23 of 25 in the complementary case).
- A noisy but correlated score (Spearman about 0.7) without junk beats the full pool by about zero.
- The same score with 40% junk beats the full pool by 0.016 to 0.025 and beats the matched draw by about 0.11. The matched draw is a weak baseline once the retained set is small, because a small random subset is likely to be mostly junk while the full average is not.
- Two methods that share a height collapse are not removed by a within-method threshold. A pooled threshold, which is a different rule, gains 0.025 by keeping about one view (25 of 25). Gibbs weights change the average by less than 0.01.

Details: `results/checkpoint3/controlled_ensembles/paired.csv`.

## Cost

Candidate training summed to 0.72 hours across 135 embeddings. Consensus summed to 5.89 hours of R time and about 2.8 hours of wall clock with two fits at once. The cluster dry-run uses 0.25 and 0.60 wall-clock hours per replication for those two stages. It was not submitted.

## What was not done

Swiss roll was locked to the same balanced endpoint (4.1% of triplets change when height is added) and was not trained. Noise, nuisance dimensions, real benchmarks, and a 25-replication cluster array were not run. Negative t-SNE and NCVis were not added. Entropy weighting was not refit. The controlled pilot does not replace a CoMDS experiment.

## Recommendation

Do not launch the 25-replication array to confirm a gain for epsilon 0.05. On the pre-registered endpoint the gain is not there.

The arc-length result is already consistent across five pools. More pools of the same design would tighten a one-axis effect and would not show surface recovery, because these candidates do not represent height.

The next informative change is a candidate family in which some embeddings preserve one intrinsic direction and others preserve the other, with a score that is not identical to either. The controlled majority-bias case says a within-method band will keep the bad family if that family is near-optimal on its own loss. A pooled rule will not. That comparison is the one worth running on real embeddings, after the embeddings actually disagree about geometry. Varying PCA alignment on UMAP and InfoNCE does not create that disagreement: their native losses barely move, and their height triplet accuracies stay near one half.
