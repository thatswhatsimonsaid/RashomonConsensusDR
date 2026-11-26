# Checkpoint 7 — When a within-method Rashomon band helps, and when it creates a bad majority

Local pilots, 25 November 2026 (America/Los_Angeles). Checkpoints 3 through 6 are unchanged. No cluster array was submitted.

Checkpoint 6 found no Rashomon advantage on the An and Tang simulations or on a short parametric grid, and it recommended pausing that search. This checkpoint does not reopen that search. It asks a narrower question: under which quality and diversity conditions does a within-method loss band change a CoMDS consensus, and can that change be explained by keeping fewer candidates, by ordinary quality weights, or by picking one good embedding?

The local answer is specific. A 5% within-method band beats unrestricted CoMDS when bad draws sit inside every method and the score tracks geometric error. Full-pool Gibbs weights match or beat the band in every such cell. The same band hurts, sometimes by several tenths of a balanced triplet, when one method is homogeneously wrong: the relative tolerance keeps that method, drops extra draws from the good methods, and uniform CoMDS follows the bad majority. Method-balanced weights and Gibbs weights on the same retained set do not follow it. A fresh seed and a Swiss-roll pool of real Isomap, t-SNE, UMAP, and PCA embeddings, plus one deliberately permuted t-SNE family, reproduce the harm.

## What was frozen before the consensus scores

The configuration is `experiments/study10_quality_diversity/config.yaml`. The generators are in `redr/redr/quality_diversity.py`. Embeddings are perturbations of a known square, n = 80. They are not UMAP or t-SNE. Three methods, four draws each. Four replications. Seed 11000 for the exploratory grid. Two thousand triplets per replication, drawn once per replication and shared by every rule. The primary metric is coordinate-balanced intrinsic triplet accuracy. Isometric, arc, and height triplets are stored and are not the endpoint.

Quality and diversity are separate factors.

- Quality. `high` is isotropic noise of scale 0.05. `mostly_high` uses scales 0.05, 0.05, 0.05, 0.45. `mixed` uses 0.05, 0.05, 0.50, 0.50. `severe` replaces half the rows of two draws with standard normal noise. `shared` applies the small noise and then multiplies the second axis by 0.25 on every embedding.
- Diversity, applied before that noise. `identical` copies the square. `moderate` adds a smooth warp of amplitude 0.30. `complementary` shrinks one axis in method 0, the other axis in method 1, and leaves method 2 intact. `harmful` leaves methods 0 and 1 faithful and row-rolls method 2 by n/3, so that method pairs the wrong objects.

The selection score is not the latent error. Lower is better. `informative` adds N(0, 0.02) to the latent Procrustes error. `weak` adds N(0, 0.20). `none` is uniform on (0.2, 1.2). `misleading` is the reflected error, so the worst geometry gets the best score. An `oracle` row copies the latent error and is labeled. It is not a primary selector. Membership is the historical relative rule, L ≤ L* + ε max(|L*|, 1e-8), applied inside each method. Epsilon is the prespecified grid 0, 0.05, 0.25, 1, and infinity. The primary comparison is epsilon 0.05. It was not chosen after seeing the curves.

The primary grid is the 5 by 4 quality-diversity factorial under the informative score. A score slice, fixed in the same file, repeats `mixed` quality at `complementary` and `harmful` diversity for `weak`, `none`, and `misleading`. That is 26 cells. LoCoMDS, tau 0.1 and percentile 0.5, is restricted to the eleven cells named in the config. The other cells are CoMDS only.

Rules, all fit by the installed coMDS package unless named otherwise:

1. The single best embedding by the selection score.
2. Unrestricted uniform CoMDS.
3. Three size-matched draws of the epsilon-0.05 count.
4. Uniform CoMDS of the within-method Rashomon set, at every epsilon.
5. Method-balanced CoMDS of that set, at epsilon 0 and 0.05, plus method-balanced weights on the full pool.
6. Gibbs weights on the epsilon-0.05 set. Gibbs is a within-method tilt and then equal method mass, beta 1. It is not a calibrated posterior.
7. The same Gibbs weights on the full pool.
8. A trimmed pool that drops the worst quarter of scores, nine of twelve embeddings.
9. Procrustes averaging, of the full pool and of the epsilon-0.05 set, as a secondary baseline.

A Rashomon benefit, for this checkpoint, means beating unrestricted CoMDS, the size-matched draws, the trim, the best single embedding, and full-pool Gibbs. Beating only the unrestricted pool is a statement about using fewer or better candidates.

Predictions written in the config before any consensus score from this study:

- Exchangeable high-quality copies. No benefit. A tight band can discard draws that averaging would have used.
- Shared axis shrink. Filtering cannot remove a distortion every candidate carries.
- Misleading score. The band should lose to the unrestricted pool, which never reads the score.
- Severe identical copies. Bad draws sit in every method. A score that tracks error should drop them. Trim and the best single embedding still have to be beaten.
- Complementary mixed quality. Specialists have worse global error than the intact method. A global trim can drop them. A within-method band might keep each specialist's cleaner draws.
- Harmful high quality. One method is a systematic mis-pairing. A within-method band keeps it. A global trim can drop it.

## Exploratory grid

One hundred four replications, 2,040 detail rows, no fit errors. Means below are balanced triplet accuracy. Differences are Rashomon epsilon 0.05 minus the named baseline, with the count of replications in which the band was higher. The full table is `results/checkpoint7/quality_diversity/summary.csv`. Paired rows are `paired.csv`.

| Quality | Diversity | Band | Full pool | Gibbs | Trim | Best | Size-matched | Band − pool (n/4) | Band − Gibbs (n/4) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| high | identical | 0.950 | 0.969 | 0.961 | 0.965 | 0.916 | 0.951 | −0.019, sd 0.005 (0) | −0.011 (0) |
| high | moderate | 0.866 | 0.871 | 0.870 | 0.859 | 0.846 | 0.857 | −0.005 (1) | −0.004 (1) |
| high | complementary | 0.921 | 0.961 | 0.949 | 0.906 | 0.913 | 0.928 | −0.040, sd 0.032 (0) | −0.029 (1) |
| high | harmful | 0.530 | 0.933 | 0.929 | 0.960 | 0.917 | 0.830 | −0.403, sd 0.019 (0) | −0.399 (0) |
| mostly_high | identical | 0.948 | 0.950 | 0.965 | 0.965 | 0.911 | 0.892 | −0.002 (2) | −0.017 (0) |
| mostly_high | moderate | 0.833 | 0.854 | 0.860 | 0.859 | 0.837 | 0.816 | −0.022 (1) | −0.027 (1) |
| mostly_high | complementary | 0.937 | 0.949 | 0.956 | 0.957 | 0.923 | 0.892 | −0.012 (0) | −0.019 (0) |
| mostly_high | harmful | 0.708 | 0.905 | 0.919 | 0.935 | 0.915 | 0.785 | −0.197, sd 0.150 (0) | −0.211 (0) |
| mixed | identical | 0.952 | 0.921 | 0.962 | 0.936 | 0.920 | 0.840 | +0.031, sd 0.009 (4) | −0.010 (0) |
| mixed | moderate | 0.839 | 0.827 | 0.841 | 0.838 | 0.836 | 0.744 | +0.012 (3) | −0.002 (2) |
| mixed | complementary | 0.911 | 0.912 | 0.937 | 0.924 | 0.921 | 0.806 | −0.001 (1) | −0.026 (0) |
| mixed | harmful | 0.782 | 0.868 | 0.927 | 0.906 | 0.923 | 0.736 | −0.086 (1) | −0.144 (0) |
| severe | identical | 0.958 | 0.916 | 0.967 | 0.932 | 0.919 | 0.810 | +0.042, sd 0.010 (4) | −0.009 (0) |
| severe | moderate | 0.845 | 0.810 | 0.849 | 0.822 | 0.841 | 0.749 | +0.034, sd 0.018 (4) | −0.004 (2) |
| severe | complementary | 0.918 | 0.914 | 0.945 | 0.928 | 0.923 | 0.764 | +0.004 (1) | −0.027 (0) |
| severe | harmful | 0.767 | 0.857 | 0.897 | 0.877 | 0.921 | 0.677 | −0.090 (0) | −0.130 (0) |
| shared | identical | 0.789 | 0.790 | 0.790 | 0.790 | 0.770 | 0.788 | −0.001 (2) | −0.001 (1) |
| shared | moderate | 0.679 | 0.721 | 0.719 | 0.729 | 0.746 | 0.715 | −0.042 (0) | −0.040 (0) |
| shared | complementary | 0.808 | 0.840 | 0.823 | 0.854 | 0.764 | 0.815 | −0.032 (1) | −0.015 (1) |
| shared | harmful | 0.658 | 0.698 | 0.697 | 0.736 | 0.764 | 0.689 | −0.040 (0) | −0.039 (0) |

Score slice, mixed quality. The complementary cells:

| Score | Band | Full pool | Gibbs | Band − pool (n/4) | Band − Gibbs (n/4) |
| --- | ---: | ---: | ---: | --- | --- |
| weak | 0.922 | 0.921 | 0.936 | +0.001 (2) | −0.014 (0) |
| none | 0.753 | 0.924 | 0.866 | −0.171, sd 0.118 (0) | −0.113 (0) |
| misleading | 0.557 | 0.922 | 0.730 | −0.365, sd 0.018 (0) | −0.173 (0) |

LoCoMDS has the same signs on the cells where it was run. On high identical, Rashomon LoCoMDS minus full-pool LoCoMDS is −0.019 (0 of 4). On severe identical it is −0.003 (2 of 4). On high harmful it is −0.449 (0 of 4). On mixed complementary misleading it is −0.414 (0 of 4). Switching the aggregator does not reverse the comparison.

Mean CoMDS time on the informative cells was about 1.7 seconds for the full pool and 1.6 seconds for the epsilon-0.05 band. Runtime is not where these pools differ.

## What the contrasts mean

The exchangeable negative control behaved as predicted. On high identical copies the informative score is a weak ranking, because the latent errors are tiny and N(0, 0.02) swamps them. The band keeps about 3.5 of 12 embeddings and loses 0.019 to the full pool in 4 of 4 replications. It also loses to Gibbs and to the trim. It beats the single best embedding by 0.034. Averaging the good copies was already the right aggregator, and the band threw some of those copies away.

Shared shrinkage behaved as predicted. On identical copies the band changes the score by −0.001 (2 of 4). Every candidate carries the shear, so membership cannot remove it. The absolute level is lower than the high-quality copies: about 0.79 rather than 0.97.

Misleading and null scores behaved as predicted. The unrestricted pool never reads the score, so it stays near 0.92 on the mixed complementary cell. The band falls to 0.557 under the misleading score and to 0.753 under the null score, 0 of 4 the other way. Gibbs also falls under a misleading score, to 0.730, but less than the hard band. This is not a surprising failure of a score that was built to point the wrong way.

The contaminated-within-method cells are the only place the band beats the full pool. Severe identical: +0.042 (sd 0.010, 4 of 4). Mixed identical: +0.031 (sd 0.009, 4 of 4). Severe and mixed moderate move the same way by smaller amounts. In all four of those cells the band also beats the size-matched draws, by about 0.09 to 0.15, so the gain is not "use fewer candidates." It loses to full-pool Gibbs in the identical cells (severe −0.009, 0 of 4; mixed −0.010, 0 of 4) and ties Gibbs on the moderate cells. The hard threshold is doing a worse version of the soft weighting. The oracle band on severe identical scores 0.968, next to Gibbs at 0.967. A perfect score does not open a gap that Gibbs leaves closed.

The complementary prediction did not hold. The hoped-for cell was mixed complementary: keep each specialist's cleaner draws, and beat a global trim that might drop the specialists. The band keeps all three methods (mean counts 1.75, 1.50, and 1.00) and scores 0.911. Unrestricted CoMDS scores 0.912. Gibbs scores 0.937. The trim scores 0.924. The band is 0.013 below the trim and 0.026 below Gibbs, and it is higher than the full pool in 1 of 4 replications. High complementary is worse: the band loses 0.040 to the full pool (0 of 4). Specialists are real in this generator. The within-method band is not how CoMDS uses them. Keeping every view, or weighting them, does.

The harmful cells are the result that changes the interpretation of a "successful filter." On high harmful, methods 0 and 1 are faithful and method 2 is a row permutation. Unrestricted CoMDS scores 0.933. The trim scores 0.960. Gibbs scores 0.929. The best single embedding scores 0.917. The 5% band scores 0.530 (difference −0.403, sd 0.019, 0 of 4). It keeps 5.50 embeddings on average, of which 3.50 belong to the mis-paired method. In replication 0 the retained set is two faithful views and three mis-paired views. CoMDS source stress on the unrestricted fit is about 1.3 to 1.6 for the faithful views and about 22 for the mis-paired views, and the consensus stays good. On the retained set the stress reverses: the two faithful views sit near 45 and the three mis-paired views sit near 3.5. The consensus followed the junk majority.

The mechanism is the relative tolerance, not a failure of the score to notice the junk. Faithful losses are near 0.12, so a 5% band is narrow and drops the extra good draws. The best junk loss is near 0.9, so a 5% band is wide and keeps most of the junk draws, which are exchangeable with each other. The oracle version of the same rule still scores 0.671, with 8 embeddings kept, against 0.933 for the pool that ignores the score. An oracle inside each method cannot reject a method whose own best member is bad.

Method-balanced weights on that same epsilon-0.05 set score 0.924. Gibbs weights on it score 0.921. Equal method mass removes the collapse, because the bad family no longer outvotes the other two. The damage is the combination of a relative band, which retains more junk draws than good draws, and uniform weights over embeddings. It is not an inevitable consequence of having called CoMDS, and it is not fixed by making the score perfect.

One cell in the score slice looks favorable and should not be promoted. Mixed harmful with the weak score: the band beats the full pool by +0.054 (sd 0.015, 4 of 4) and ties Gibbs at +0.004 (2 of 4). The weak noise occasionally produces a small junk score, the relative band then tightens, and fewer junk draws are kept. The informative version of the same cell goes the other way (−0.086). That is a scale artifact of a relative epsilon. It is not a new rule.

## Epsilon

The curves are `results/checkpoint7/quality_diversity/figures/epsilon_curves.png`. Nothing in the prespecified grid says that an intermediate epsilon is best.

On high harmful the uniform band scores 0.918 at epsilon 0 (one embedding per method: two faithful, one junk), 0.530 at 0.05, 0.617 at 0.25, 0.929 at 1, and 0.933 at infinity. The primary historical epsilon is the worst point on that curve. Epsilon 0 does not collapse, because one junk view cannot outvote two faithful views. The collapse starts when the wide junk band admits several junk draws and the narrow good band does not admit the extra faithful draws.

On severe identical the curve is mildly higher at small and moderate epsilon than at infinity: 0.955, 0.958, 0.968, 0.966, 0.916. That is the contaminated-within-method pattern. Full-pool Gibbs on the same cell is 0.967, level with the best point of the curve. On shared shrinkage the curves are flat and lower. On high identical they rise toward the full pool.

Retained counts at epsilon 0.05 run from about 3.3, on mixed identical, to 8.5, on shared moderate. Effective diversity of the retained set is not a monotone function of epsilon: the harmful cell becomes less faithful as epsilon grows from 0 to 0.05, then more faithful again as epsilon approaches the full pool.

## Fresh seeds

Five cells were refit with the same generators, the same epsilon, and the same rules, at seed 21000. LoCoMDS was not refit. Output is `results/checkpoint7/confirm/`. This is a confirmation of the exploratory reading, not a new search.

| Cell | Band − pool | Band − Gibbs |
| --- | --- | --- |
| severe / identical / informative | +0.035, sd 0.010 (4/4) | −0.002, sd 0.003 (1/4) |
| high / identical / informative | −0.013, sd 0.009 (0/4) | −0.008 (0/4) |
| high / harmful / informative | −0.201, sd 0.206 (1/4) | −0.195, sd 0.211 (1/4) |
| mixed / complementary / informative | +0.001, sd 0.011 (2/4) | −0.019, sd 0.007 (0/4) |
| mixed / complementary / misleading | −0.360, sd 0.014 (0/4) | −0.153 (0/4) |

Four of the five signs match the exploratory grid, including the two that were predicted to be null or negative and the one gain that Gibbs absorbs. The harmful cell is the same mechanism with a less stable magnitude. Replication 0 kept five faithful views and two junk views, and the band was 0.081 above the full pool. The other three replications kept two or three faithful views against three or four junk views, and the differences were −0.212, −0.413, and −0.258. The exploratory grid happened to build a junk majority in all four replications (mean 3.5 of 5.5). The fresh seeds show that the relative band does not guarantee that majority. When the noise in the good scores happens to keep extra faithful draws, the collapse is smaller. The mean is still negative.

## Real embeddings, one planted bad family

The factorial uses synthetic perturbations. The follow-up asks whether the bad-majority arithmetic appears in embeddings that Isomap, t-SNE, UMAP, and PCA actually produce. The script is `redr/scripts/run_dr_mechanism.py`. Four Swiss rolls, n = 180, noise 0.1, seeds 3000–3003. Views: Isomap at 10 and 25 neighbors; t-SNE at perplexity 20 and 40, two initializations, 500 iterations; UMAP at 10 and 30 neighbors; PCA on four jitters of scale 0.05. A fifth family is each t-SNE embedding row-rolled by n/3. That family is labeled `permuted_tsne` and is not a natural DR method. It is the harmful factor, built from a real t-SNE so that the junk views agree with each other.

The selection score is 1 minus trustworthiness of the embedding against the observed Swiss roll, k = 10, computed inside each family. The latent parameter is used only to score a finished consensus. The prediction, written in the script before these fits, was that the 5% band would keep the permuted family, because its trust losses are similarly bad, and that uniform CoMDS of that set would fall behind unrestricted CoMDS.

Trust losses, four replications. t-SNE sits between 0.004 and 0.009. UMAP sits near 0.01. Isomap and PCA sit near 0.09 to 0.13. The four permuted t-SNE embeddings sit between 0.466 and 0.496, and within a replication they differ by about 0.006. A 5% band around a loss of 0.47 is about 0.023 wide, so it keeps all four. A 5% band around a t-SNE loss of 0.005 is a few ten-thousandths wide, so it keeps only the best of that family.

The band kept every permuted embedding in all four replications. Mean retained count was 11.25 of 16, of which 4.00 were permuted and 3.00 were PCA.

| Rule | Balanced triplet | Trustworthiness | Permuted views kept |
| --- | ---: | ---: | ---: |
| Best single embedding | 0.679 | 0.995 | 0 |
| Trim, drop the worst quarter | 0.675 | 0.974 | 0 |
| Gibbs on the epsilon-0.05 set | 0.646 | 0.923 | 4 |
| Full-pool Gibbs | 0.644 | 0.920 | 4 |
| Unrestricted CoMDS | 0.638 | 0.921 | 4 |
| Within-method band, epsilon 0.05 | 0.599 | 0.805 | 4 |

The band minus the full pool is −0.039 (sd 0.050), higher in 1 of 4. Minus Gibbs, −0.045 (0 of 4). Minus the trim, −0.076 (1 of 4). Minus the size-matched mean, −0.040 (1 of 4). The trim drops exactly the four permuted views, because they are the four worst trust scores, and it matches the best single embedding. Gibbs does not drop them. It still gives the bad family equal method mass, about one fifth, and that is enough to stay near the unrestricted pool rather than to match the trim. Uniform weights on the band are worse than either, because the band has reduced the good families to a few draws and left all four junk draws in place.

The epsilon curve, `results/checkpoint7/dr_mechanism/figures/epsilon_curve.png`, dips at 0.05 and 0.25 (0.599 and 0.600) and returns to the full pool at epsilon 1 and infinity (0.638). Epsilon 0, which keeps fewer of the permuted views (mean 2 of 4), scores 0.629. The intermediate tolerances are the worst points, as they were on the synthetic harmful factor. Replication 0 is drawn in `figures/swiss_consensus.png`, colored by the arc parameter. The unrestricted consensus and the Gibbs consensus of the band still show the roll. The uniform band does not. The trim, which has deleted the permuted family, is the clearest of the four.

This check does not say that a natural pool of UMAP, t-SNE, InfoNCE, PaCMAP, and NCVis contains a homogeneously bad family. The bad family was planted. It says that once such a family is present, and once its members agree with each other, the relative within-method rule keeps it and uniform aggregation gets worse. Checkpoint 6 already measured the other regime on a short parametric grid: within-method native-loss spans of at most 4.5%, which is the exchangeable cell, where this band loses a small amount to averaging.

## Alternatives, and what would distinguish them

Fewer candidates. Rejected where the band gains, because the size-matched draws lose by about 0.1. Also rejected where the band collapses: the size-matched draws of the same count stay well above the band (high harmful, 0.830 against 0.530). The collapse is a change in who is kept.

Quality weighting. This is the alternative that survives. Full-pool Gibbs is at least as high as the epsilon-0.05 uniform band in every informative cell of the exploratory grid, up to a tie within 0.004. On the fresh seeds the severe-identical gain over the full pool shrinks from +0.035 to −0.002 against Gibbs. On the Swiss roll, Gibbs on the band undoes the uniform-band damage. The hard threshold is not contributing a selection that soft weights miss, when the score tracks fidelity.

One good embedding. The band beats the best single embedding in the contaminated cells, and so do Gibbs and the trim. On the harmful cell the best single embedding beats the band by 0.386. Selecting one good view is a strong baseline precisely when a whole method is wrong.

CoMDS already discounts bad views. True for a small minority. The unrestricted high-harmful fit gives the junk views stress near 22 and stays at 0.933. False once the junk views are half or more of a small pool. Stress then favors the junk, because they agree with each other.

The experiment that separates "the band excluded harmful candidates" from "the band retained complementary good candidates" is the membership count, not a new epsilon. In the cells where the band beats the pool, it is dropping severe draws that every method produced. In the harmful cell it is retaining the bad method and discarding extra good draws. Complementary specialists were already in the full pool, and restricting to the band did not improve on that pool.

## What this does not say

It does not say that Rashomon aggregation cannot help. It says that this rule, a within-method relative band followed by uniform CoMDS, does not beat Gibbs weights, a global trim, the best single embedding, and the unrestricted pool at the same time, in any cell that was prespecified. The one pattern that beats the unrestricted pool is ordinary quality weighting of a score that tracks error.

It does not say that CoMDS fails on complementary views. Checkpoint 4 already showed that CoMDS combines specialists when those views are handed to it. This grid says the within-method band is not what produces that combination. On mixed complementary the full pool and Gibbs are the better rules.

It does not license another selection rule on the product manifolds from Checkpoints 4 and 5, and it does not reopen the An and Tang comparison from Checkpoint 6. Those findings stand.

It does not use latent coordinates or labels to choose embeddings in the primary comparisons. The oracle rows are marked and are an upper bound. On the harmful factor that upper bound still loses to the pool that ignores the score.

## Next experiment

Do not submit a 25-replication array. No prespecified cell produced a Rashomon benefit over unrestricted CoMDS, Gibbs, the trim, the best embedding, and a size-matched draw together. A more precise estimate of the bad-majority harm would not change the decision.

The informative next measurement is descriptive, on fully trained UMAP, InfoNCE, PaCMAP, NCVis, and t-SNE, with a longer schedule than the 50 and 200 epochs of Checkpoint 6. For a fixed native objective, record the within-method relative span of that loss and the between-method geometric disagreement. Where a hyperparameter changes the objective, keep the families separate or score them with one declared common loss. Do not compare raw native losses across families. Do not fit a new epsilon in that run.

If the within-method spans stay inside a few percent, the relevant cell of this factorial is the exchangeable one, and a 5% band cannot help. If one family is homogeneously far from the others on the common loss, the relevant cell is the harmful one, and a relative within-method band is the rule that keeps the bad family. Either result says which row of this table a real pool occupies. It is not another search for a manifold on which the band wins.
