# Hypotheses

These are questions, not findings. Each one names the comparison that would support or weaken it.

## H1. Intrinsic geometry

Rashomon restriction may improve recovery of a latent manifold even when it does not improve preservation of observed distances.

Support would require a gain on a two-dimensional intrinsic score, not only on ordering of one manifold parameter. The S-curve and Swiss roll are long, thin rectangles once unfolded. Raw Euclidean distance in that rectangle mostly restates arc length.

## H2. Noise

Near-optimal aggregation may matter more when the observations are noisy or contain nuisance coordinates. Not tested in the runs that are underway.

## H3. Diversity and tolerance

A small tolerance can keep too little diversity. A large tolerance can admit distorted views. The epsilon curve is not assumed to be unimodal.

## H4. Complementary errors

Method-specific Rashomon sets help when methods err in different ways. They do not remove a distortion that is shared by the near-best embedding of every method. A pooled threshold, which can drop an entire method, is a different rule and is not the original method-specific definition.

## H5. Local consensus

Any benefit of restriction may show up in LoCoMDS and not in CoMDS, or the reverse. The two algorithms are separate comparisons.

## H6. Redundancy

Uniform weights over-represent a method with more retained views or with duplicated views. Method-balanced and Gibbs weights are the checks.

## H7. Uncertainty

Neighborhood disagreement may identify unreliable relationships even if weighting by that disagreement does not improve the embedding. That diagnostic is not part of the confirmatory S-curve endpoint.

## Readings after Checkpoint 7

H3. On the harmful factor, epsilon 0.05 is the worst point of the prespecified curve, and epsilon 0 and infinity are not. The curve is not unimodal.

H4. A within-method band does not remove a shared axis shrink, and it does not remove a method whose own members are exchangeable and wrong. It also does not improve on the full pool when methods err on different axes. The cell where it beats the full pool is bad draws inside every method, and Gibbs weights match that gain.

H5. LoCoMDS has the same sign as CoMDS on the eleven cells where both were fit, including the −0.45 loss on the harmful factor.

H6. Equal method mass is what stops the bad majority. Uniform weights on the epsilon-0.05 harmful set score 0.530. Method-balanced weights on that set score 0.924.

## Readings after Checkpoint 8

H2. A 64-dimensional digits subset and two noisy manifolds were scored. Inside a fixed objective the native spans stay below 0.05, including on digits. Observation noise and dimension did not by themselves create a within-method Rashomon set.

H3. From epsilon 0.05 through infinity the trained pools are the full pool. Epsilon 0 keeps one seed per method and does not filter a bad draw, because the seeds are tied.

H4. The trained methods do not err in different intrinsic directions. On the S-curve every method keeps the arc and loses the height. A relative band cannot remove a distortion that every near-best embedding shares.
