# Checkpoint 8 — Do natural dimension-reduction pools occupy the Checkpoint 7 regimes?

Local pilot, 25 November 2026 (America/Los_Angeles). Checkpoints 3 through 7 are unchanged. No cluster array was submitted. No new selection rule was added.

Checkpoint 7 found two mechanisms for a within-method relative band. The band beats unrestricted CoMDS when bad draws sit inside every method and the score tracks geometric error, and full-pool Gibbs matches that gain. The band loses when one method is homogeneously wrong, because a relative tolerance keeps that method. This checkpoint asks whether either pattern occurs in embeddings from UMAP, InfoNCE, negative t-SNE, NCVis, and PaCMAP, trained long enough that the schedule is not the short Checkpoint 6 grid.

It does not. At a fixed objective, three seeds of each method have native-loss spans of at most 2.2%. The 5% band keeps every embedding. Uniform Rashomon CoMDS is the same fit as unrestricted CoMDS. The one homogeneously weaker family is UMAP with 50 neighbors, and that is a different objective from UMAP with 15 neighbors. A relative band inside the 50-neighbor family still keeps every seed.

## What was frozen

The configuration is `experiments/study11_natural_diversity/config.yaml`. Membership, weights, trimming, and the regime label do not read latent coordinates or digit labels. Those quantities are evaluation columns.

The primary pool is one objective per method: 15 neighbors and PCA-alignment weight 0. Training seeds are 0, 1, and 2. Native loss is the Checkpoint 6 reference: a frozen positive graph and shared negative indices for UMAP, InfoNCE, negative t-SNE at spectrum 0, and NCVis; the terminal PaCMAP phase, neighbor plus further-point, for PaCMAP. Losses are not compared across methods or across neighbor sizes.

Two slices change the objective and stay out of the primary band. UMAP at 5 and at 50 neighbors. UMAP and InfoNCE at alignment weight 5, two seeds.

Epochs were set before the grid, from a one-seed Swiss-roll diagnostic. UMAP reference loss was 0.288 at 50 epochs, 0.281 at 200, and 0.278 at 400, against a negative-draw standard deviation of about 0.057. InfoNCE was 0.388, 0.377, and 0.374, against a standard deviation of about 0.106. Two hundred epochs is past the part of the curve that the reference can see. PaCMAP's published schedule enters its terminal phase at epoch 200, so 450 epochs is 250 epochs of the loss the reference measures. Checkpoint 6 used 50 and 200. These counts were not changed after the grid.

Datasets, each generated once:

- Swiss roll, n = 240, noise 0.12, seed 4000.
- S-curve, n = 240, noise 0.08, seed 4100.
- A stratified 400-point subset of scikit-learn digits, 64 dimensions, seed 4200. This is the high-dimensional benchmark. It has no latent rectangle.

The regime rule, applied to the primary pool only:

- Systematically bad: some method's mean trustworthiness is at least 0.08 below the best method, and that method's native span is at most 0.05.
- Contaminated within method: at least one method has native span above 0.05, and in every such method the highest loss has strictly lower trustworthiness than the lowest loss.
- Complementary candidate: every native span is at most 0.05, every method is within 0.08 trustworthiness of the best, and between-method distance disagreement exceeds both 1.5 times the mean within-method disagreement and that mean plus 0.05.
- Exchangeable: every span is at most 0.05 and between-method disagreement does not exceed the within-method mean by that margin.
- Diverse but similar loss: spans are small, and the pool is neither exchangeable nor a complementary candidate.

Consensus rules on the primary pool, unchanged from Checkpoint 7 except that the global trim uses 1 minus trustworthiness. Raw native losses are not on one scale, so a trim on them would mix incompatible objectives. The rules are the best embedding by trustworthiness, the best native embedding of each method, unrestricted CoMDS, unrestricted LoCoMDS, the trust trim of the worst quarter, full-pool Gibbs, the within-method native band at epsilon 0, 0.05, 0.25, 1, and infinity, method-balanced and Gibbs weights on the epsilon-0.05 set, LoCoMDS of that set, and three size-matched draws.

On digits, the vendored contrastive loss crashes when a batch of edges has at most ten rows. Training drops that incomplete final batch. The Swiss roll and the S-curve did not hit the failing batch size, so their fits use every edge. The dropped batch is at most 127 edges of a symmetrized 15-neighbor graph.

## Within a fixed objective, the seeds agree

Primary-pool native spans and mean trustworthiness, three seeds:

| Dataset | Widest span | Trust range across methods | Regime label |
| --- | ---: | ---: | --- |
| Swiss roll | 0.014 (NCVis) | 0.010 | complementary candidate |
| S-curve | 0.022 (NCVis) | 0.019 | complementary candidate |
| Digits | 0.015 (UMAP) | 0.022 | diverse, similar loss |

Every span is below the 5% membership tolerance. On all three datasets the epsilon-0.05 band keeps 15 of 15 embeddings. From epsilon 0.05 through infinity the uniform band is the full pool, so its CoMDS fit matches unrestricted CoMDS exactly: Swiss trustworthiness 0.962 and balanced triplet 0.546, S-curve 0.994 and 0.788, digits trustworthiness 0.938. Method-balanced weights on that set are uniform, because each method contributes three embeddings. Gibbs moves Swiss trustworthiness from 0.962 to 0.960 and the S-curve balanced triplet from 0.788 to 0.774. Size-matched draws of 15 embeddings are the full pool.

The contaminated regime does not occur. The systematically bad regime does not occur inside the primary pool. There is no within-method quality variation for a relative band to use.

The epsilon curves are `results/checkpoint8/natural_diversity/figures/epsilon_trust.png`.

Epsilon 0 keeps one embedding per method. On the Swiss roll its trustworthiness is 0.970 against 0.962 for the full pool, and the balanced triplet is 0.562 against 0.546. On the S-curve it is slightly worse than the full pool (0.775 against 0.788). Keeping one seed is not a filter of bad draws. The seeds were already tied on the native loss.

## The complementary label is not axis specialization

The unsupervised rule calls the Swiss roll and the S-curve complementary candidates because between-method distance disagreement is larger than within-method disagreement, while trustworthiness is almost flat. The numbers are modest. Swiss between-method disagreement is 0.290 against a within-method mean of 0.182, and between-method neighborhood overlap is still 0.52. S-curve disagreement is 0.106 against 0.044, with between-method overlap 0.67. InfoNCE seeds are nearly copies (within-method disagreement 0.019 and 0.009). UMAP, negative t-SNE, and NCVis move more across seeds than InfoNCE does, at the same native loss.

The latent triplets, which were not used to assign the label, do not show methods preserving different axes. On the S-curve every primary method has arc-triplet accuracy from 0.896 to 0.922 and height-triplet accuracy from 0.537 to 0.570. The short axis is missing in all five methods. On the Swiss roll both axes sit between 0.53 and 0.66 for every method. The distance disagreement is layout variation among embeddings that keep the same aspect of the geometry. That is the shared-distortion regime of Checkpoint 7, seen here as a shared failure to recover the short axis, not as a bad method the band should have dropped.

The best single embedding matches or beats every consensus on the intrinsic score. Swiss PaCMAP scores 0.606 against unrestricted CoMDS at 0.546. S-curve InfoNCE scores 0.804 against unrestricted CoMDS at 0.788. Digits PaCMAP has trustworthiness 0.956 and 10-neighbor label purity 0.867, against unrestricted CoMDS at 0.938 and 0.850. Label purity is an evaluation column. It was not a weight. LoCoMDS of the full pool, which is also LoCoMDS of the band, scores 0.552 on the Swiss roll and 0.778 on the S-curve. It does not recover the missing axis either.

The trust trim drops the worst quarter by trustworthiness. On the Swiss roll that removes negative t-SNE, whose mean trustworthiness is 0.976 against 0.984 to 0.986 for the others. The balanced triplet stays at 0.546. A 0.01 gap is enough for a rank trim and is far below the 0.08 gap the regime rule requires before calling a method bad. The trim does not identify a distortion the band missed, because the band had nothing to miss.

## A different neighborhood size is a different objective

Seed ranges are in `results/checkpoint8/natural_diversity/figures/trust_ranges.png`.

Alignment weight 5, scored as its own family, changes mean trustworthiness by at most 0.006 relative to weight 0. UMAP at 5 neighbors stays within 0.02 of UMAP at 15 neighbors.

UMAP at 50 neighbors is weaker on the common structural scores, and the three seeds agree with each other:

| Dataset | UMAP k=15 trust | UMAP k=50 trust | Native span at k=50 |
| --- | ---: | ---: | ---: |
| Swiss roll | 0.984 | 0.966 | 0.041 |
| S-curve | 0.995 | 0.957 | 0.007 |
| Digits | 0.940 | 0.890 | 0.009 |

On digits the 50-neighbor embedding also drops neighborhood preservation from 0.495 to 0.327 and label purity from 0.851 to 0.657. The native losses are not comparable to the 15-neighbor losses, because the positive graph has changed: the reference values sit near 0.39 and 0.28. A within-family 5% band keeps all three 50-neighbor seeds. It cannot reject the family, for the same reason Checkpoint 7's relative band kept a method whose own members were exchangeable and wrong. The difference is that this family was produced by changing the objective, and the design kept it out of the primary band. Putting it into that band under a shared native loss would have compared incompatible graphs.

No new threshold was fit to this gap. The existing common-score trim is the rule that can drop a homogeneously weaker family, and only after that family has been placed in the pool. The primary pool's trust range is 0.01 to 0.02, so the prespecified 0.08 rule does not call any primary method bad. Lowering 0.08 until the trim looks useful would be tuning the definition to this grid.

## What this does not justify

It does not justify a new aggregator, a hierarchical weight, or a different epsilon. The relative band had no bad draws to remove and no bad method to reject. Gibbs, the trim, LoCoMDS, and the best single embedding were already in the comparison. None of them turns these pools into a recovery of both intrinsic axes, and the band does not improve on the full pool because it is the full pool.

It does not justify a 25-replication array. The membership result is the same on three independently generated datasets. A larger sample would estimate the same equality more precisely.

It does not say that a Rashomon band cannot help on some other trained pool. It says these trainers, at a converged schedule, do not produce the within-method contamination or the homogeneously bad method that Checkpoint 7 required. Building that contamination by mixing undertrained seeds back in would recreate the simulation, not test whether training produces it.

## Next

Leave the selection rule where it is. A later pool is worth a consensus comparison only if, under one fixed loss, a method's native span exceeds 5% and trustworthiness moves with that loss, or if one method is at least 0.08 worse in trustworthiness than the others while its own seeds agree. Neither happened here. Until a pool meets one of those prespecified conditions, another epsilon sweep on the same three datasets would be tuning a null result.
