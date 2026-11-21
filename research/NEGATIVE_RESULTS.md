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
