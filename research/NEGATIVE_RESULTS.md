# Negative results and counterexamples

Unfavorable outcomes stay in the record. They are not retired because a later setting looks better.

## Checkpoint 2, carried forward

Native-loss restriction did not consistently improve consensus.

- On Iris, epsilon 0.05 dropped high-lambda PaCMAP embeddings that had better triplet accuracy than the embeddings it kept.
- On the one S-curve pool, CoMDS ambient triplet was worse for the 21-view Rashomon set than for one embedding per method and worse than size-matched unrestricted draws.
- UMAP and InfoNCE native losses barely moved across seeds and PCA-alignment strengths, so those methods entered the set together.
- Gibbs and entropy weights did not produce a stable gain. Positive-neighborhood pair weights on Iris lowered trustworthiness.
- Adding three badly undertrained views to the Iris pool did not harm uniform CoMDS. The consensus step can already ignore a small amount of contamination.
- The S-curve LoCoMDS latent-triplet edge at epsilon 0.05 (0.927 versus matched 0.913 and one-per-method 0.917) is one pool, scored on the one-dimensional generator parameter.

## S-curve replication, five independent pools

Uniform LoCoMDS at the pre-specified epsilon 0.05 loses the confirmatory endpoint, coordinate-balanced intrinsic triplet accuracy.

- Versus the full pool: −0.021 (sd 0.012), worse in 5 of 5.
- Versus a size-matched draw: −0.010 (sd 0.006), worse in 5 of 5.
- Versus one lowest-native embedding per method: −0.010 (sd 0.007), worse in 4 of 5.

The balanced-accuracy curve is lowest at epsilon 0.05 and 0.10 and highest when every candidate is kept. CoMDS at epsilon 0.05 is worse than one embedding per method by 0.031 (sd 0.008), 5 of 5.

The Checkpoint 2 arc-length edge does replicate, and it is the wrong conclusion for surface recovery. LoCoMDS arc-length triplet accuracy at epsilon 0.05 beats the full pool by 0.022 (sd 0.004) in 5 of 5, and beats the matched draw by 0.010 in 5 of 5. The same fits lose height triplet accuracy to the full pool by 0.023 in 5 of 5, and lose ambient triplet accuracy by 0.016 in 5 of 5. Candidate height triplet accuracy averages 0.56. The best candidate reaches 0.69.

Numbers and the geometry lock are in `research/CHECKPOINT_3_REPORT.md`.

## Unfolded manifolds are not tested by one-axis triplets

On the S-curve, arc length spans about 9.4 and the transverse coordinate spans about 2.0. Across data seeds 1000–1004, adding the transverse coordinate changes only 3.3% of random triplet decisions. On the Swiss roll the rate is 4.1% (seeds 2000–2004). An embedding that orders the curve and collapses the width still scores well on isometric triplet accuracy. That score is not evidence of surface recovery. The confirmatory endpoint for these replications is the coordinate-balanced intrinsic triplet, chosen before consensus fitting. See `results/checkpoint3/s_curve_replication/geometry_lock.json`.

## Controlled embeddings, 25 replications

Synthetic perturbations, aggregated by a weighted Procrustes mean. Not CoMDS, and not UMAP. Balanced intrinsic triplet accuracy. Paired differences are in `results/checkpoint3/controlled_ensembles/paired.csv`. Epsilon is 0.05 unless noted. "Rashomon" here means a within-method score threshold.

The size-matched draw is a weak comparison when the filter keeps about 3 views out of 27 and many candidates are bad. The full pool is the comparison that matters.

- Uninformative scores. Within-method filtering does not beat a matched draw (shared noise, difference −0.001, sd 0.006). It is worse than keeping every candidate (−0.005, sd 0.006, worse in 19 of 25). With 40% junk embeddings and an uninformative score, filtering is much worse than the full pool (−0.106, sd 0.088, worse in 22 of 25). Keeping a small random subset throws away the averaging that was protecting the full pool.
- Misleading scores. Filtering is worse than the matched draw (complementary −0.016, sd 0.011, worse in 23 of 25) and worse than the full pool (−0.023, sd 0.016).
- Noisy but informative scores, no junk. Spearman correlation of score and latent error is about 0.7. The gain over the matched draw is small (shared +0.005, sd 0.005; complementary +0.010, sd 0.008). The gain over the full pool is indistinguishable from zero (shared −0.0002; complementary +0.0004).
- Informative score plus 40% junk. Filtering beats the matched draw by about +0.11 (25 of 25). It beats the full pool by less: shared +0.016 (sd 0.015, better in 21 of 25) and complementary +0.025 (sd 0.023, better in 23 of 25).
- Oracle scores. The 5% relative band keeps most candidates (about 17 to 21 of 27). The change versus the full pool is about +0.001. A perfectly aligned score does not, by itself, make a tight relative band do much.
- Majority bias. Two methods share a height collapse. Within-method filtering matches the full pool (difference +0.0005). It cannot drop a method whose own best candidate is still biased. A pooled threshold, which is not the method-specific rule, gains +0.025 (sd 0.005, 25 of 25) by keeping about one view.
- Gibbs weights on the filtered set change the coordinate average by less than 0.01, and usually slightly downward.

Square geometry with complementary errors and a noisy score is the setting where filtering beats the full pool in the point estimate (+0.009, sd 0.026) but only in 16 of 25 replications. That is not a stable advantage.

## Checkpoint 4, four-replication pilot

The product-manifold results are in `research/CHECKPOINT_4_REPORT.md`. These are the unfavorable parts. They are not cancelled by the CoMDS gain on complementary views.

- Method-specific reference-loss bands at epsilon 0.05 change square LoCoMDS by +0.002. The losses do not vary enough inside a group to be a selection rule.
- The pooled structural score keeps joint embeddings and drops every specialist. On the square that costs about 0.10 balanced triplet accuracy against specialist LoCoMDS (4 of 4). On the stretched surface it keeps the long axis (u triplet about 0.94, v triplet 0.51) and costs about 0.22 against specialist CoMDS (4 of 4).
- Diversity inside that pooled set cannot recover the specialists the score already removed.
- Procrustes averaging of the specialist pool does not combine the two directions. On the square it scores 0.778 against 0.969 for LoCoMDS on the same views.
- LoCoMDS is unstable on the stretched specialist pool: balanced scores 0.966, 0.693, 0.727, 0.933. CoMDS on the same four pools stayed between 0.945 and 0.964.
- Ambient trustworthiness moves against the complementary consensus. On the square, specialist LoCoMDS loses 0.029 to joint LoCoMDS and 0.054 to the single best structural embedding (0 of 4 the other way).
- When the second curve is replaced by noise, every rule has v triplet accuracy at 0.50. Specialist CoMDS is worse than joint CoMDS by 0.025 (0 of 4).
- Between-method neighborhood entropy does not track latent neighborhood error. Square Spearman correlations are near zero and change sign across rules and replications.
- A pooled quality threshold hurts when the score is uninformative or misleading. On the controlled complementary ensemble, full-pool LoCoMDS stays at 0.932 while the misleading filter falls to 0.53 (4 of 4).
- The condition named `long` did not change the observations. Per-feature standardization removed the aspect weight. It is not evidence about aspect ratio. The stretched condition is the aspect test.

## Checkpoint 5, local pilots

The tables are in `research/CHECKPOINT_5_REPORT.md`. These designs did not produce a Rashomon advantage.

- CoMDS of two known block configurations loses the balanced triplet to concatenating one-dimensional Isomaps of those blocks: square 0.945 versus 0.989, stretched 0.935 versus 0.986, four replications. Source-balanced CoMDS of the Checkpoint 4 pool loses the same comparison (0.956 and 0.915). The Checkpoint 4 specialist consensus is not evidence for a Rashomon method.
- On the stretched surface, winning the balanced score and winning the isometric score are different outcomes. The raw latent coordinates score 0.782 balanced and 1.000 isometric. Isomap on the full matrix scores 0.759 and 0.965. Concatenated 1D Isomap scores 0.986 and 0.787. Block standardization before Isomap is what raises the short-axis triplet, from 0.518 to 0.717.
- On rotated features, the union of five objective-specific bands changes unrestricted CoMDS by +0.004 (sd 0.031). The Pareto front loses to Isomap (−0.008, better in 1 of 3). Pooled multiscale CoMDS ties Isomap (+0.003). None of them recover both axes. Privileged concatenation remains near 0.986.
- Marginal subset gain is worse than unrestricted CoMDS by 0.051 on the rotated square (0 of 3). Diversity with no quality gate is worse by 0.169 (0 of 3). Adding four noise embeddings makes that diversity rule worse by 0.211, while unrestricted CoMDS does not move.
- The 5% Rashomon band on mutual-information clusters drops the second curve in 2 of 3 aligned replications. Unrestricted concatenation of the same clusters scores 0.930 and the band scores 0.799. After an orthogonal mix of the columns, that concatenation loses to Isomap by 0.180 (0 of 3).
- One-dimensional stress does not rank complementary projections. The stress pair scores 0.546 aligned and 0.572 rotated, against Isomap at 0.865 and 0.837.
- Agreement clustering with no quality gate loses to Isomap by about 0.25 on the square and lowers recovery when the second curve is absent.
- Three informative directions in a two-dimensional embedding: known-block concatenation ties Isomap (difference −0.000, sd 0.031). The cluster Rashomon band is worse than keeping every large cluster (−0.072, sd 0.080, worse in 2 of 3).
- A shared curve added to every block at weight 1 drops known-block concatenation to balanced 0.595. The integration fails when the blocks themselves are mixed.

## Checkpoint 6, one sample of each paper simulation

The tables are in `research/CHECKPOINT_6_REPORT.md`.

- On the Gaussian mixture, PCA has adjusted Rand 0.972. CoMDS of the eleven-view subset, LoCoMDS, UMAP, t-SNE, and Meta-Spec kernel PCA are all at or above 0.97. There is no accuracy gap for a Rashomon set to close. Meta-Spec kernel PCA tightens the clusters (within/between distance 0.10) relative to CoMDS (0.20) and PCA (0.31).
- On the Swiss roll at n = 400, LoCoMDS with the reduced grid has balanced triplet 0.654 and end gap 8.0. CoMDS has 0.672 and 13. UMAP has 0.771 and 39. The manuscript claim that LoCoMDS unfolds the roll was not reproduced with the eleven available methods and three LoCoMDS settings.
- The 5% method-specific loss band keeps every Gaussian parametric embedding. Native-loss spans inside each method are at most 4.5%. On the Swiss roll the band drops four embeddings, all of them lambda 5, and the balanced triplet moves from 0.750 to 0.753. The best InfoNCE embedding scores 0.792. Five size-matched draws of 26 embeddings sit between 0.749 and 0.751.
- Definition 3.2, applied as a median soft-Jaccard band, keeps 15 embeddings and scores 0.750 on the Swiss roll. The overlap band scores 0.748. Common-knowledge UMAP scores 0.706.
- The loss-band CoMDS fit on the Swiss roll took 52 seconds. Unrestricted CoMDS of the same 30 embeddings took 36 seconds.
- Neighbor-vote entropy on the parametric Gaussian pool has Spearman correlation 0.068 with the rate of false voted neighbors. Mean purity is 0.9997. The loss-band pool is the same set of embeddings.
- Under one noise shift of standard deviation 0.25, CoMDS of four classical views has Procrustes RMSE 0.148, next to PCA at 0.142. That comparison does not include a Rashomon filter.

## Checkpoint 7, quality by diversity

The tables are in `research/CHECKPOINT_7_REPORT.md`. These are the unfavorable parts of a factorial that also contains the cells where a 5% band beats unrestricted CoMDS. Those gains lose to full-pool Gibbs and are recorded in the same report. They are not retired by the harmful cells, and the harmful cells are not retired by the gains.

- Exchangeable high-quality copies. The band loses 0.019 to unrestricted CoMDS (sd 0.005, 0 of 4) and 0.011 to Gibbs (0 of 4). Fresh seeds: −0.013 (0 of 4).
- Shared axis shrink. Identical copies move by −0.001 (2 of 4). Filtering does not remove a distortion every candidate carries. The absolute level stays near 0.79, below the unshared copies.
- Misleading score, mixed complementary. The band scores 0.557 against 0.922 for the pool that ignores the score (−0.365, sd 0.018, 0 of 4). Fresh seeds: −0.360 (0 of 4). A null score loses 0.171 (0 of 4). Gibbs falls too, and by less.
- Complementary specialists. The prediction that a within-method band would beat a global trim did not hold. Mixed complementary: band 0.911, trim 0.924, Gibbs 0.937, full pool 0.912. High complementary loses 0.040 to the full pool (0 of 4).
- One homogeneously wrong method. High harmful: band 0.530 against full pool 0.933 (−0.403, sd 0.019, 0 of 4). The retained set averages 3.5 mis-paired draws out of 5.5. Method-balanced weights on the same set score 0.924. The oracle band, which uses latent error inside each method, still scores 0.671. LoCoMDS of the band loses 0.449 to LoCoMDS of the full pool (0 of 4). Fresh seeds: mean −0.201 (sd 0.206), positive in 1 of 4, and that replication kept a faithful majority.
- Epsilon 0.05 is the worst point on the high-harmful curve (0.530, against 0.918 at epsilon 0 and 0.933 at infinity). An intermediate tolerance is not a default optimum.
- The weak-score harmful cell gains 0.054 on the full pool and ties Gibbs at +0.004 (2 of 4). The informative version of the same cell loses. That gain is a scale artifact of a relative epsilon, not a method.
- Swiss roll with a planted permuted t-SNE family, seeds 3000–3003. The 5% band keeps all four permuted embeddings in 4 of 4 replications. Balanced triplet 0.599 against unrestricted 0.638 (−0.039, sd 0.050, better in 1 of 4), Gibbs 0.644, and a trim that drops the permuted family at 0.675. Trustworthiness of the consensus falls from 0.921 to 0.805. The bad family was planted. The result says the relative rule keeps it.

## Checkpoint 8, trained UMAP, InfoNCE, negative t-SNE, NCVis, and PaCMAP

The tables are in `research/CHECKPOINT_8_REPORT.md`.

- At k = 15 and alignment weight 0, three seeds of each method have native-loss spans of at most 0.022. The 5% band keeps all 15 embeddings on the Swiss roll, the S-curve, and digits. Uniform Rashomon CoMDS matches unrestricted CoMDS on each dataset.
- No primary method is 0.08 below the best method in trustworthiness. The contaminated regime and the homogeneously bad regime do not occur inside a fixed objective.
- The unsupervised rule calls the Swiss roll and the S-curve complementary candidates. Latent triplets do not. S-curve arc triplets sit near 0.91 and height triplets near 0.55 for every method. Swiss-roll balanced triplets sit between 0.53 and 0.66. The disagreement is layout variation on a shared missing axis.
- The best single embedding beats the consensus on the intrinsic score: Swiss PaCMAP 0.606 against unrestricted CoMDS 0.546, S-curve InfoNCE 0.804 against 0.788. Digits PaCMAP trustworthiness is 0.956 against unrestricted CoMDS 0.938.
- UMAP at 50 neighbors is weaker and internally consistent. Digits trustworthiness is 0.890 against 0.940 at 15 neighbors, label purity 0.657 against 0.851, native span 0.009. A relative band inside that family keeps every seed. The losses are not pooled with the 15-neighbor family, because the graphs differ.
- Alignment weight 5 changes trustworthiness by at most 0.006. The trust trim on the Swiss roll drops negative t-SNE over a 0.01 gap and leaves the balanced triplet at 0.546.

## Checkpoint 9, cross-objective Rashomon ensembles

The tables are in `research/CHECKPOINT_9_REPORT.md`.

- Seeds of one configuration agree (S-curve 15-neighborhood overlap 0.823). Different neighbor counts, perplexities, and spectra disagree more (overlap 0.567). That disagreement does not split the two intrinsic axes. On the S-curve every nonlinear configuration has mean height-triplet accuracy between 0.53 and 0.61.
- UMAP at 50 neighbors is not a global specialist that smaller neighborhoods omit. On the S-curve it raises height from 0.535 to 0.607 and drops arc from 0.874 to 0.619. Balanced accuracy falls from 0.759 to 0.643. On digits it is worse at 5-neighborhood preservation (0.215 against 0.448) and at 30-neighborhood preservation (0.505 against 0.551).
- The full heterogeneous pool is worse than the fixed-objective pool. S-curve balanced triplet 0.728 against 0.763, and against 0.778 for the best default UMAP. Digits trustworthiness 0.895 against 0.956 for one t-SNE. Quality weights reach 0.920 and still lose to that embedding. A random subset of eight embeddings scores 0.719, and the eight prespecified defaults score 0.759.
- The 5% within-configuration band is inactive on all three digits replications (widest span 0.038). On the S-curve and the Swiss roll it drops one or two seeds of UMAP at 5 neighbors or PaCMAP at 10 neighbors. Hierarchical CoMDS scores 0.718 against 0.728 for the full S-curve pool, and 0.566 against 0.568 on the Swiss roll. The dropped seeds are not the geometrically weak views.
- The trust trim drops PCA, the view with the highest height triplet (0.656 on the S-curve, 0.634 on the Swiss roll), because its trustworthiness is about 0.87. The balanced score does not rise.
- Ten Gaussian nuisance coordinates collapse the S-curve arc for every method, from about 0.87 to between 0.52 and 0.60. The full pool scores 0.590 and the best single embedding 0.603. The Rashomon comparison is vacant because that follow-up has one seed per configuration.
