# Checkpoint 4 — Complementary geometry and Rashomon selection

Local pilot, 21 November 2026 (America/Los_Angeles). Four replications per condition. This is not the 25-replication cluster confirmation. Checkpoint 3 is unchanged: native-loss filtering on the S-curve does not recover the surface, and that array should not be submitted.

The question here is different. When embeddings preserve different intrinsic directions, can a consensus of those embeddings keep both directions, and which selection rule throws one of them away?

Latent coordinates are used only to score finished embeddings. They are not an input to membership or to weights.

## What was compared

Five selection rules, all using losses computed from the observations:

| Rule | What it keeps |
| --- | --- |
| all | Every embedding. Uniform weights, and a family-balanced variant. |
| specific | Method-specific band at relative epsilon 0.05 on that group's own reference loss. Groups are algorithm, source, and hyperparameter. Losses are not compared across groups. |
| pooled | One band at epsilon 0.05 on a common structural loss: one minus the mean neighborhood preservation of the embedding against the full observed matrix at k = 5, 15, and 30. |
| hybrid | The method-specific set, then the same structural band computed inside that set. |
| diverse | Inside the pooled set, keep up to 6 embeddings, starting from the lowest structural loss and greedily reducing neighborhood overlap. |
| specialist | Embeddings fit on one intrinsic block only. This is a designed complementary pool, not a loss filter. |
| joint | Embeddings fit on the full observed matrix. |
| best | The single embedding with the lowest structural loss. No consensus. |

Aggregators: Procrustes mean, CoMDS, and LoCoMDS. The coMDS package is called unmodified.

Reference losses stay inside a group. UMAP has no exposed cross-entropy in this fit, so its reference is one minus neighborhood preservation on the matrix it was trained on. t-SNE uses its KL divergence at a fixed perplexity. Isomap uses stress against its own geodesic. Those numbers are never pooled with each other.

## Candidate geometry

Each latent coordinate is observed as its own curve, plus eight nuisance features. Specialists are trained on one curve. Joint methods see the concatenated matrix. On the square, standardized surface the two blocks are exchangeable.

Across four square replications the candidates separate by source:

| Source | Mean u triplet | Mean v triplet | Mean structural loss |
| --- | ---: | ---: | ---: |
| joint | 0.69 | 0.71 | 0.60 |
| u block | 0.91 | 0.50 | 0.80 |
| v block | 0.50 | 0.91 | 0.79 |

Neighborhood overlap is 0.58 for pairs from the same source and 0.07 for pairs from different sources. The pool is geometrically diverse. It is not a stack of near-copies.

A one-axis triplet near 0.91 is not surface recovery. The balanced intrinsic triplet of a u-specialist is about 0.74, because the other axis is absent. That distinction is the point of the metric.

The condition named `long` only rescaled the evaluation coordinates. Per-feature standardization removes that scale from the observations, and the balanced metric removes it again. Those four runs are extra draws of the square observation. They are not an aspect-ratio experiment. The aspect experiment is `stretched`, which multiplies each observed curve by its aspect weight and only centers the matrix.

## Square surface

Four replications, seeds 4000–4003, n = 200. Balanced intrinsic triplet accuracy.

| Rule | Procrustes | CoMDS | LoCoMDS |
| --- | ---: | ---: | ---: |
| specialist | 0.778 | 0.965 | 0.969 |
| all | 0.875 | 0.956 | 0.973 |
| specific | 0.878 | 0.957 | 0.974 |
| joint | 0.876 | 0.877 | 0.879 |
| pooled | 0.849 | 0.862 | 0.867 |
| diverse | 0.845 | 0.859 | 0.865 |
| best single | 0.832 | 0.832 | 0.832 |

Paired differences, specialist LoCoMDS minus the comparator, balanced triplet:

| Comparator | Mean difference | sd | Better |
| --- | ---: | ---: | ---: |
| joint LoCoMDS | +0.090 | 0.028 | 4/4 |
| pooled LoCoMDS | +0.102 | 0.031 | 4/4 |
| best structural embedding | +0.136 | 0.030 | 4/4 |
| specialist Procrustes | +0.191 | 0.023 | 4/4 |
| all CoMDS, specialist minus all | +0.009 | 0.006 | 4/4 |

The same specialist LoCoMDS fits lose ambient trustworthiness to the joint LoCoMDS by 0.029 (sd 0.004, 0/4) and to the best single embedding by 0.054 (sd 0.004, 0/4). The best single embedding has trustworthiness 0.915. The specialist consensus has 0.861.

Axis triplets of the specialist LoCoMDS are about 0.75 on each axis. The specialists themselves score about 0.91 on their own axis. Consensus builds a joint geometry. It does not copy either specialist's one-dimensional ordering.

The method-specific band drops about 3 of 33 embeddings and changes LoCoMDS by +0.002. UMAP, t-SNE, and Isomap reference losses barely move inside a group, which is the same limitation Checkpoint 3 found for native UMAP and InfoNCE losses. The pooled structural band keeps about 6 joint embeddings and no specialists. Diversity-aware selection is confined to that set. The cap is 6, so it drops at most one view and stays within 0.01 of the pooled score. Hybrid also keeps only joint embeddings and lands within 0.01 of pooled. Family-balanced weights on the full pool change LoCoMDS by less than 0.01.

## Stretched surface

Aspect (5, 1), scale preserved, seeds 4300–4303. Joint embeddings now follow the long axis and lose the short one. Mean joint CoMDS: balanced 0.753, u triplet 0.974, v triplet 0.512, trustworthiness 0.989. The best structural embedding is the same kind of object: u triplet 0.942, v triplet 0.513, trustworthiness 0.995. Neighborhood preservation on the full matrix rewards the long axis, because that axis dominates the ambient distances.

Specialist CoMDS: balanced 0.957 (sd 0.009), u 0.757, v 0.748, trustworthiness 0.945.

| Contrast, specialist CoMDS | Mean difference | sd | Better |
| --- | ---: | ---: | ---: |
| minus joint CoMDS, balanced | +0.204 | 0.014 | 4/4 |
| minus pooled CoMDS, balanced | +0.217 | 0.023 | 4/4 |
| minus best single, balanced | +0.214 | 0.029 | 4/4 |
| minus joint CoMDS, u triplet | −0.217 | 0.030 | 0/4 |
| minus joint CoMDS, v triplet | +0.237 | 0.033 | 4/4 |
| minus joint CoMDS, trustworthiness | −0.044 | 0.011 | 0/4 |
| minus all CoMDS, balanced | +0.042 | 0.006 | 4/4 |

Keeping the joint views in the pool dilutes the short axis. The specialist pool is better than the unrestricted pool on this surface. On the square surface the joint views already carried both axes, and adding them did not hurt.

LoCoMDS on the specialist pool is not stable here. The four balanced scores are 0.966, 0.693, 0.727, and 0.933. CoMDS stayed between 0.945 and 0.964. Two of four LoCoMDS fits failed to return the geometry that CoMDS found from the same views. On the square surface LoCoMDS was slightly above CoMDS in all four replications. The aggregator that combines complementary views in this pilot is CoMDS. LoCoMDS needs a stability check before it is preferred.

Procrustes on the specialist pool reaches balanced 0.801. It picks up some of the short axis (v triplet 0.686 versus 0.510 for the joint Procrustes) and does not approach CoMDS.

## Missing second axis

`square_hidden` replaces the second curve with noise. Four replications. Every rule has v triplet accuracy within 0.01 of 0.50. Nothing invents the missing coordinate.

Specialist CoMDS is worse than joint CoMDS on the balanced triplet by 0.025 (sd 0.003, 0/4) and worse on trustworthiness by 0.034 (sd 0.013, 0/4). Half of the specialist pool is an embedding of noise. Including it hurts the axis that is actually observed. The unrestricted pool, which mixes joint views with those noise views, is about as good as the joint-only pool and better than the specialist pool.

## Neighborhood entropy

Between-method neighborhood entropy was computed on the union of candidate neighborhoods and compared, pointwise, with the latent k-nearest-neighbor miss of a finished consensus. It was not used as a weight.

On the square, Spearman correlations for the uniform ensembles are about −0.03 (full Procrustes), −0.10 (joint LoCoMDS), −0.15 (pooled LoCoMDS), and +0.04 (specialist LoCoMDS), each with a standard deviation larger than the mean. Signs flip across replications. The same pattern holds when the second axis is absent. Entropy is not a predictor of latent neighborhood error in this pilot, and there is no case for using it as an aggregation weight here. It remains a diagnostic to recheck on the cluster replications, not a settled negative for every dataset.

## Controlled perturbations, now with CoMDS

These are not DR methods. Three synthetic methods, six draws each, n = 80, four replications. The score is a simulated quality signal. When the alignment is `noisy`, it tracks global latent error. When it is `none` or `misleading`, it does not. That score is not a native loss. Method-specific bands are within-method thresholds on this score. Pooled bands use one threshold. Checkpoint 3's 25-replication Procrustes study is still the larger Procrustes record. The numbers below are the CoMDS extension.

Balanced triplet, means. `all` ignores the score.

| Scenario | All Procrustes | All LoCoMDS | Pooled LoCoMDS | Best single |
| --- | ---: | ---: | ---: | ---: |
| complementary, square, noisy score | 0.867 | 0.932 | 0.896 | 0.879 |
| complementary, uninformative score | 0.867 | 0.932 | 0.686 | 0.686 |
| complementary, misleading score | 0.867 | 0.932 | 0.527 | 0.526 |
| complementary, 40% junk, noisy score | 0.750 | 0.911 | 0.891 | 0.886 |
| majority bias, noisy score | 0.794 | 0.829 | 0.953 | 0.946 |
| majority bias, uninformative score | 0.794 | 0.829 | 0.710 | 0.710 |
| shared noise, noisy score | 0.905 | 0.919 | 0.884 | 0.880 |
| local damage, noisy score | 0.899 | 0.912 | 0.949 | 0.937 |
| complementary, long plane, noisy score | 0.782 | 0.788 | 0.780 | 0.783 |

Paired differences worth keeping:

- Complementary square. All LoCoMDS beats all Procrustes by 0.065 (sd 0.018, 4/4) and beats the best single view by 0.053 (sd 0.047, 4/4). The pooled filter is worse than keeping every view and using LoCoMDS, by 0.036 (sd 0.017, 0/4). Consensus, not the filter, is what combines the two damaged methods.
- Uninformative or misleading scores. The filter falls to 0.69 or 0.53. The full-pool LoCoMDS is unchanged, because it never read the score.
- Forty percent junk. LoCoMDS of the full pool beats Procrustes of the full pool by 0.161 (sd 0.044, 4/4). Procrustes is pulled down by the junk. LoCoMDS is not, at this contamination rate.
- Majority bias and an informative score. Two methods squash one axis. Pooled LoCoMDS beats the full-pool LoCoMDS by 0.124 (sd 0.026, 4/4) and matches the best single view. The method-specific band does not (difference +0.001). A within-method band cannot drop a method whose own best member is still wrong. This is the setting where a pooled quality rule adds something consensus does not.
- The same majority bias with an uninformative score. The pooled rule is worse than the full pool by 0.118 (sd 0.068, 0/4).
- A long thin plane. Complementary damage does not create a gap between Procrustes, LoCoMDS, and the filter. The balanced score still equalizes the axes, but the perturbations live on a plane whose long side dominates the un-balanced geometry, and none of the aggregators recover a large balanced gain.
- Local damage to 25% of points in one method, and a global squash in another. With an informative score, the pooled rule beats the full pool by 0.036 on the balanced triplet (4/4) and by 0.099 on the squashed axis (4/4). The full-pool LoCoMDS does not beat the best single view.

## What this does and does not support

Supported by the four-replication pilot, and the reason for the cluster array:

- CoMDS of views that preserve different directions recovers a balanced geometry that no individual view and no joint-only consensus has, including when the joint views and the structural score both ignore the short axis.
- Procrustes averaging does not do that.
- A structural neighborhood score on the full matrix selects the views that match the dominant ambient geometry and drops the complementary ones.
- A method-specific band on reference losses does not implement that selection either. The losses do not vary enough inside a group.
- If the second direction is absent from the data, consensus does not create it, and noise specialists make the result worse.
- A pooled quality threshold is useful in a different regime: a minority of views carry a direction that the majority has destroyed, and the score actually tracks that error. It is harmful when the score does not.

Not supported:

- The original REDR recipe, a within-method native-loss band, as the source of the gain.
- LoCoMDS as a uniformly safer aggregator than CoMDS. It matched or slightly beat CoMDS on the square and failed on two of four stretched specialist pools.
- Neighborhood entropy as a weight or as a calibrated map of latent error.
- A claim that ambient fidelity improves. Trustworthiness moves the other way. The best single embedding is the most faithful to the ambient neighborhoods and the worst at the missing direction.
- A 25-replication confirmation. The signs are consistent inside this pilot. They are still one design, four seeds, and one family of curves.

## Cluster array

`experiments/study8_complementary/generate_sbatch_arrays.py` describes 25 replications of square, stretched, and square_hidden. Seeds start at 5000, 5300, and 5200 so they do not reuse this pilot. The dry-run is the default. Partition, memory, modules, and mail are environment placeholders. Nothing was submitted. The Checkpoint 3 S-curve array is not the job to run.

The array can separate the explanations above: specialist versus joint versus pooled versus the full pool, CoMDS versus LoCoMDS versus Procrustes, and a short axis that is present, dominant-scale, or absent.

## Files

- `results/checkpoint4/complementary/study_lock.json` — contrasts written before the replication grid.
- `results/checkpoint4/complementary/sweep_all.csv`, `paired.csv`, `candidates_all.csv`
- `results/checkpoint4/complementary/figures/rules_balanced.png`
- `results/checkpoint4/complementary/figures/candidate_axes.png`
- `results/checkpoint4/controlled_consensus/detail.csv`
- `results/checkpoint4/controlled_consensus/figures/mechanisms.png`
- `results/checkpoint4/complementary_smoke/` — one square replication with PCA-initialized t-SNE, archived because those seeds were identical. It is not in the tables above.
