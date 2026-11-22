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
