# Checkpoint 9 — Do cross-objective Rashomon sets add geometric information?

Local pilot, 26 November 2026 (America/Los_Angeles). Checkpoints 3 through 8 are unchanged. No cluster array was submitted. No new aggregator was written.

Checkpoint 8 found that three seeds of one fixed objective have native-loss spans of at most 2.2%, so a 5% band keeps every embedding. This checkpoint asks a different question. If the candidates come from different objectives — neighbor graphs, the negative-t-SNE spectrum, t-SNE perplexity, Isomap neighborhoods — do the near-optimal embeddings carry complementary geometry, and does restricting each objective to its own Rashomon set improve the consensus?

They disagree more than seeds of one objective do. They do not preserve different intrinsic directions. The consensus of the heterogeneous pool does not beat the best single embedding, and where the Rashomon rule removes anyone it removes one or two seeds and the consensus score does not improve.

## What was frozen

The configuration is `experiments/study12_cross_objective/config.yaml`. It was fixed before any consensus score from this study.

A configuration is one algorithm with one objective setting. Native losses are compared only inside that configuration. The loss scale is \(s(a,c) = \max(|L^\star(a,c)|, 10^{-8})\), so the band is \(L^\star + \epsilon s\). At \(\epsilon = 0.05\) this is the same relative rule as Checkpoints 1 through 8. Latent coordinates and digit labels are evaluation columns.

The objective grid is the set of hyperparameters that actually change the training loss in the vendored code:

- Parametric UMAP at 5, 15, 30, and 50 neighbors. The neighbor count is the positive graph.
- Parametric InfoNCE at 15 and 50 neighbors. The spectrum is left at the library default. Alignment weight stays 0, because Checkpoint 8 found that weight 5 did not move trustworthiness.
- Parametric negative t-SNE at spectrum \(s = 0\) with 15 and 50 neighbors, and at spectrum \(s = 1\) with 15 neighbors. In the vendor map, \(s = 0\) is the t-SNE normalization and \(s = 1\) is the UMAP normalization. The reference loss uses that same map.
- NCVis (`loss_mode nce`) at 15 and 50 neighbors. The spectrum parameter is not part of this loss.
- Parametric PaCMAP at 10 and 30 neighbors, with mid-near 5 and further-point 20.
- sklearn t-SNE at perplexity 10, 30, and 50. The native loss is the KL divergence.
- Isomap at 8, 15, and 30 neighbors, and PCA. Each is a single embedding, so its Rashomon set is that embedding.

Three training seeds for every stochastic configuration. CNE runs 200 epochs and PaCMAP 450, the Checkpoint 8 schedule. t-SNE runs 1000 iterations from a PCA initialization. The reference contrastive loss uses one frozen negative draw.

Datasets, three replications except the nuisance follow-up:

- S-curve, n = 240, noise 0.08, seeds 5100–5102.
- Swiss roll, n = 240, noise 0.12, seeds 5200–5202.
- Stratified digits, 300 points, 64 dimensions, seeds 5300–5302.
- One S-curve with 10 standard-normal nuisance coordinates, seed 5400, one training seed. This was specified as a follow-up, not as a third tuning pass.

The prespecified comparisons, all on the same pool, are a single embedding chosen by trustworthiness, the default configuration of each algorithm, CoMDS and LoCoMDS of every seed of those defaults, CoMDS and LoCoMDS of one default embedding per algorithm, one seed of every configuration, the full pool under uniform, method-balanced, configuration-balanced, within-configuration Gibbs, and full-pool quality weights, a trim of the worst quarter by trustworthiness, and three random subsets matched to the one-per-algorithm count. The hierarchical Rashomon ensemble is fit only when the 5% band drops someone. Otherwise it is the full pool, and it is not fit a second time. The primary synthetic score is the balanced intrinsic triplet. Isometric triplet accuracy and the two axis-wise triplets are reported beside it. Neither axis score is used to choose a view.

Each replication has 52 embeddings, 20 configurations, and 8 algorithms. The nuisance follow-up has 20 embeddings because it has one seed. Training the 488 candidates took 2.76 hours. The consensus fits took 1.13 hours. Peak resident memory was 0.85 GB.

## The pool is diverse across objectives and redundant within them

Seeds of one configuration agree. Different configurations do not, and changing the algorithm does not add much disagreement beyond changing the configuration.

| Dataset | Within-config overlap | Between-config overlap | Between-algorithm overlap | Within rank correlation | Between-config rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: |
| S-curve | 0.823 | 0.567 | 0.548 | 0.931 | 0.653 |
| Swiss roll | 0.829 | 0.517 | 0.489 | 0.900 | 0.735 |
| Digits | 0.738 | 0.407 | 0.374 | 0.889 | 0.660 |
| S-curve plus nuisance | — | 0.210 | 0.147 | — | 0.571 |

Overlap is mean pointwise Jaccard of 15-neighborhoods. The nuisance row has no within-config entry because that follow-up has one seed. Global distance disagreement tells the same story: about 0.10 within a configuration and 0.26 to 0.34 between configurations on the clean manifolds.

<img src="/opt/cursor/artifacts/within_between.png" alt="Within-configuration neighborhood overlap is higher than between-configuration overlap on every dataset." />

Each clean replication has nine pairs with 15-neighborhood overlap of at least 0.98. Those are duplicate seeds, not duplicate algorithms.

## Different objectives do not recover different intrinsic directions

On the S-curve, every nonlinear configuration has mean height-triplet accuracy between 0.53 and 0.61. Mean arc-triplet accuracy for the default small-neighborhood methods is 0.87 to 0.90. UMAP at 50 neighbors raises height from 0.535 to 0.607 and drops arc from 0.874 to 0.619. Balanced triplet accuracy falls from 0.759 to 0.643. The larger neighborhood is not storing the short axis that the smaller neighborhood omitted. It is giving up the long axis. The same pattern appears in 5-neighborhood preservation: UMAP at 5 neighbors preserves 0.659 of the ambient 5-neighborhoods, and UMAP at 50 neighbors preserves 0.492. Preservation at 30 neighbors also falls, from 0.787 to 0.709.

PCA is the only S-curve view with a clearly higher height triplet, 0.656, and its arc triplet is 0.641 and its trustworthiness is 0.871. It is not a specialist that recovered the short axis while keeping the curve. It is a different compromise, and it is the view trustworthiness ranks last.

On the Swiss roll no configuration mean has both axes above 0.65. PCA again has the highest mean height, 0.634, with arc 0.584 and trustworthiness 0.872. UMAP at 5 neighbors has the highest mean arc, 0.672, with height 0.541. Balanced triplet accuracy of the best configuration mean is 0.631 for PCA and 0.623 for UMAP at 5 neighbors. The individual best embedding in each replication scores 0.697, 0.638, and 0.635. The full consensus scores 0.571, 0.552, and 0.583.

<img src="/opt/cursor/artifacts/axis_recovery.png" alt="Arc versus height triplet accuracy. Nonlinear methods cluster at high arc and low height on the S-curve. No configuration sits in the high-arc, high-height corner." />

The gallery for S-curve seed 5100 shows the same split. Small-neighborhood UMAP, negative t-SNE at either spectrum, Isomap, and t-SNE order the arc and mix the height. UMAP at 50 neighbors scrambles the arc. PCA shows a height gradient and a weaker arc order.

<img src="/opt/cursor/artifacts/s_curve_gallery.png" alt="S-curve embeddings colored by arc in the top row and by height in the bottom row." />

On digits the large-neighborhood configurations are worse at both scales. UMAP at 5 neighbors has trustworthiness 0.936, 5-neighborhood preservation 0.448, and 30-neighborhood preservation 0.551. UMAP at 50 neighbors has 0.864, 0.215, and 0.505. Label purity, which was not used for selection, falls in the same direction. There is no second structure that the large graph is keeping.

## The Rashomon rule is almost inactive

The widest within-configuration relative span is 0.038 on all three digits replications, so the 5% band keeps all 52 embeddings. On the S-curve the widest spans are 0.028, 0.119, and 0.075. The two spans above 0.05 are UMAP at 5 neighbors, and the band keeps 1 of 3 seeds in one replication and 2 of 3 in the other. On the Swiss roll the band drops one embedding in each replication, either one PaCMAP seed at 10 neighbors or one UMAP seed at 5 neighbors. The nuisance follow-up has one seed per configuration, so every configuration is a singleton and the band cannot remove anyone.

<img src="/opt/cursor/artifacts/membership.png" alt="Fraction of candidates kept by the within-configuration band. From epsilon 0.05 onward the fraction is above 0.97." />

At epsilon 0 the band keeps one embedding per configuration, except where the native losses tie. t-SNE's KL values tie across seeds, so those seeds all stay. From epsilon 0.05 upward the kept fraction is above 0.97. Hierarchical CoMDS at 0.05 was fit on the five replications where the set differs from the full pool. It keeps 50 or 51 embeddings. On the S-curve its balanced triplet is 0.718 against 0.728 for the full pool. On the Swiss roll it is 0.566 against 0.568. Dropping the one or two seeds outside the band does not improve the consensus, and the dropped UMAP seeds are not the distorted views. Further epsilon values were not fit. They would have removed at most those same one or two seeds, or removed nobody.

## Aggregation does not beat the best embedding

Means over the three replications. The hierarchical row is the mean of the replications in which the band removed someone.

| Rule | S-curve balanced | S-curve height | Swiss balanced | Swiss height | Digits trust | Digits label purity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Best default UMAP | 0.778 | 0.549 | 0.572 | 0.604 | 0.931 | 0.754 |
| Best default t-SNE | 0.740 | 0.544 | 0.584 | 0.597 | 0.956 | 0.792 |
| Best default PaCMAP | 0.767 | 0.530 | 0.608 | 0.538 | 0.944 | 0.797 |
| PCA | 0.709 | 0.656 | 0.631 | 0.634 | 0.814 | 0.431 |
| Best by trustworthiness | 0.772 | 0.551 | 0.584 | 0.597 | 0.956 | 0.780 |
| Fixed-objective CoMDS | 0.763 | 0.534 | 0.569 | 0.578 | 0.917 | 0.739 |
| Fixed-objective LoCoMDS | 0.767 | 0.550 | 0.569 | 0.590 | 0.920 | 0.772 |
| One default per algorithm, CoMDS | 0.759 | 0.531 | 0.568 | 0.568 | 0.910 | 0.719 |
| One seed of every configuration | 0.730 | 0.517 | 0.566 | 0.575 | 0.895 | 0.678 |
| Full pool, CoMDS | 0.728 | 0.517 | 0.568 | 0.581 | 0.895 | 0.694 |
| Full pool, configuration-balanced | 0.729 | 0.519 | 0.565 | 0.569 | 0.895 | 0.677 |
| Full pool, native Gibbs | 0.728 | 0.517 | 0.566 | 0.573 | 0.895 | 0.678 |
| Full pool, quality weights | 0.744 | 0.516 | 0.570 | 0.585 | 0.920 | 0.763 |
| Trust trim, worst quarter | 0.737 | 0.511 | 0.570 | 0.585 | 0.913 | 0.750 |
| Hierarchical CoMDS, epsilon 0.05 | 0.718 | 0.502 | 0.566 | 0.576 | — | — |
| Random 8 embeddings | 0.719 | — | 0.568 | — | 0.899 | 0.717 |

On the S-curve the full pool is worse than the fixed-objective pool by 0.035 balanced triplet accuracy, and worse than the best default UMAP by 0.050. The one-seed-per-configuration pool matches the full pool, so the extra seeds are not what hurts. Adding the non-default objectives is what hurts. A random draw of 8 embeddings scores 0.719, and the 8 prespecified defaults score 0.759. The defaults are a better subset than a random subset of the same size. That is a choice of objective, not a Rashomon filter.

On the Swiss roll the full pool, the fixed-objective pool, the trim, and the hierarchical pool sit between 0.565 and 0.572. PCA scores 0.631 and the best single embedding in the three replications scores 0.697, 0.638, and 0.635. Consensus height stays near 0.58. It does not collect PCA's height or the occasional UMAP seed whose height reached 0.71, because those views are one embedding among 52 and the others share a lower height.

On digits a single t-SNE at the default perplexity has trustworthiness 0.956 and label purity 0.792. The full pool has 0.895 and 0.694. Quality weights, which use trustworthiness rather than the incompatible native losses, reach 0.920 and 0.763 and still lose to the single embedding. The trust trim does the same. Label purity is an evaluation column.

LoCoMDS does not change the ordering. On the S-curve, LoCoMDS of the defaults scores 0.767 and LoCoMDS of the full pool scores 0.717.

The nuisance follow-up collapses the arc for every method. Mean arc-triplet accuracy falls from about 0.87 on the clean S-curve to between 0.52 and 0.60. Trustworthiness of the small-neighborhood methods falls from about 0.99 to about 0.76–0.85. The best single embedding scores 0.603 balanced, the full pool 0.590, and the fixed-objective pool 0.575. Height stays near 0.56. No objective ignores the nuisance coordinates and keeps the curve the others lose. With one seed, the Rashomon comparison is vacant. One replication is enough to see the shared failure and not enough to estimate a small difference between aggregators.

## What failed, and why

The candidate configurations share the missing geometric information. On the clean S-curve the short axis is weak in every nonlinear embedding. On the Swiss roll neither axis is reliably recovered. On digits the alternate neighbor graphs are weaker copies of the same local structure. An aggregator cannot reconstruct a direction that is absent from its inputs, and weighting cannot promote a view that is one of fifty when the other forty-nine agree on a different geometry.

The native-loss criterion is informative only as a within-configuration tie check. Spans above 5% occurred in five of nine main replications, and each time in a single configuration, for one or two seeds. Those seeds were not the geometrically bad ones. UMAP at 5 neighbors is among the better balanced embeddings on both manifolds. The band occasionally drops one of its seeds and leaves the large-neighborhood embeddings, which are the ones that give up the arc, inside their own bands. Their native losses are on a different graph, so a within-configuration rule cannot see that they are worse.

Trustworthiness has the opposite blind spot. It ranks the locally faithful embeddings above PCA. The trust trim therefore drops PCA, the view with the higher height triplet, and the balanced score does not rise. A common structural score is still the right way to compare objectives. This particular score does not measure the missing axis.

Method balance, configuration balance, and within-configuration Gibbs match the uniform full pool. The native losses inside a configuration barely vary, so Gibbs has nothing to tilt. Configuration balance stops a four-setting method from outvoting PCA, and that does not create a height signal the inputs do not carry.

## Recommendation

Do not add an aggregator and do not submit a 25-replication array. The cross-objective pool is a real heterogeneous ensemble, and the Rashomon restriction on that pool is either inactive or a one-seed edit. Neither the ensemble nor the restriction beats a strong single embedding chosen without latent labels.

A later pool is worth this comparison only if some trained objective preserves an intrinsic direction the others lose, and an unsupervised score that does not use that direction still ranks the useful view above the locally faithful but incomplete ones. Trustworthiness does not do that here. Until such a pool exists, another epsilon sweep will keep nearly every candidate and refit the same consensus.
