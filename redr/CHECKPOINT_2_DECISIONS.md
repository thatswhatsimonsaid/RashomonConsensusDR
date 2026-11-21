# Checkpoint 2 decisions

## Branch and results

Checkpoint 2 is a separate branch. Checkpoint 1 files under `results/iris`, `results/swiss`, `results/mnist`, `REPORT.md`, and `DECISIONS.md` are unchanged. New outputs are under `results/checkpoint2/`.

## What the audit found

The pinned checkouts are RashomonDR `0ab277d` and coMDS `0570fe2`. No upstream source file is edited. Editable installs create `__pycache__` and `*.egg-info`. `setup_vendor.sh` now ignores those generated files when it checks for local modifications. Before that change, a second install was refused even though the source was clean.

`ContrastiveLoss.__init__` calls `torch.manual_seed(self.seed)`. The Checkpoint 1 probe constructs a new module with `seed=0` on every call, so repeated probes of the same embedding return the same number. It is one frozen draw, using only the first high-d neighbor as the positive. It is not an average of the training objective.

Training still adds the PCA-alignment penalty `mean((1 - cosine)^2) * lambda` on top of that contrastive loss (`cne.py`, `task_type="pca"`). The logged training scalar is therefore not comparable across lambda. Both the Checkpoint 1 probe and the Checkpoint 2 reference omit that penalty.

PaCMAP's published schedule sets the mid-near weight to 0 after epoch 200 (`pacmap_weight_schedule`). The Checkpoint 1 probe sums the neighbor, mid-near, and further-point terms with equal weight. That sum is not the terminal objective.

RashomonDR's own `record_loss_umap`, `record_infonce_loss`, `record_tSNE_loss`, and `record_NCVis_loss` are simplified pair scores used when `experiment_1_metrics.py` is run with `metric_type="Loss"`. They are not the training loss. Checkpoint 2 does not use them as the membership rule. Negative t-SNE and NCVis in that repository are `loss_mode="neg"` and `loss_mode="nce"`. They were not added to the Checkpoint 2 pools.

CoMDS initialization is classical scaling of the source-weighted sum of distance matrices (`initConf` with `inddiff=TRUE`). Permuting views together with their weights leaves that sum unchanged. Tests on 20-point views confirmed: all-ones weights match omitted weights; a permutation of weights and views matches stress to 1e-3 and the Procrustes embedding; splitting one view into two copies with half the weight keeps stress within 0.05.

Uniform source weights are a vector of ones, as in Checkpoint 1. Method-balanced and Gibbs weights sum to 1. When every method has the same number of views, those two uniform-within-method schemes differ from all-ones weights only by a global scale, and the Iris fits matched.

## Native loss

Primary membership uses one reference per method:

- UMAP and InfoNCE: vendored `ContrastiveLoss` with `temperature=0.5`, `eps=1`, `spec_param=1`, `negative_samples=5`. Positives are all 15 Euclidean high-d neighbors. Four negative-index draws are frozen with `torch.manual_seed` and reused for every embedding. The native loss is the mean of those 60 terms. The standard deviation across terms is stored separately. It is large because different pairs have different losses. Paired differences between embeddings are much more stable: on Iris the correlation of UMAP term-wise losses was 0.993 and the standard error of a difference of means was about 0.0008, against a candidate span of 0.0035.
- PaCMAP: terminal weights, neighbor 1 and further-point 1, mid-near 0, on one `generate_pair` call with seed 0. This term is deterministic.

Lambda does not define a new objective under this reference, because the alignment penalty is excluded. Seeds and lambda values for a method share one Rashomon set. A hyperparameter that changes k or the negative-sample count would need its own group. This pool does not vary those.

Epsilon 0 keeps losses within `1e-8` of the minimum. Other finite epsilons use the Checkpoint 1 relative band `epsilon * max(|L*|, 1e-8)`. Infinity keeps every finite loss.

## Pools that were trained

Iris, the main pool: 5 seeds, lambda in `{0, 0.1, 1, 5, 10}`, UMAP and InfoNCE for 200 epochs, PaCMAP for 450 epochs. 75 embeddings. The illustrative grid `{0, 0.001, 0.01, ...}` was narrowed because an alignment penalty of size about 1, multiplied by 0.001 or 0.01, is small next to a contrastive loss near 0.2 to 0.4. Lambda 0 and 0.1 are both in the pool, and the UMAP native means at those two values differ by about 0.0007.

S-curve, a smaller synthetic follow-up: 400 points, noise 0.05, 3 seeds, lambda in `{0, 1, 5}`, same epoch budgets. 27 embeddings. It is one controlled geometry, not a sweep of noise or dimension.

The main pools contain only full-length runs. Two-epoch embeddings are a separate contamination add-on on Iris (one per method).

LoCoMDS stays at tau 0.1 and percentile 0.5. Those were not tuned.

## Aggregations

The epsilon curve is uniform source weights, CoMDS and LoCoMDS, on the native set. Method-balanced weights and Gibbs weights with beta 1 are fit at epsilon 0.05, 0.1, and infinity. Beta is in units of the within-method standard deviation of native loss. Beta 0.25 and beta 4 are fit only for CoMDS at epsilon 0.05.

Matched unrestricted ensembles resample, within each method, as many full-length candidates as the native set contains. Iris uses 10 seeds at epsilon 0.05. S-curve uses 5. This is resampling of one pool, not a new training pool.

Neighborhood entropy uses binary entropy of the weighted kNN vote at k=10. The three `weightmat` rules and the distance-variance rule are separate CoMDS runs at native epsilon 0.05. They are not a search that was then reported as the method. LoCoMDS is not given those matrices. The package builds its own local weights, and the bridge skips that combination.

Common-quality membership treats `1 - neighborhood preservation` as a loss and is fit once, at epsilon 0.05, CoMDS only. Multi-criterion membership is the native epsilon-0.05 set further restricted to embeddings whose neighborhood preservation is at least the method median. That median was fixed from the pool before looking at consensus scores. Neither rule replaces native loss.

Procrustes-averaged coordinates and classical MDS of the mean normalized distance matrix are included as non-CoMDS aggregators.

## Evaluation

External metrics stay at k=10 with 4000 triplets, seed 0, shared by every embedding of a dataset. Labels are evaluation-only. Silhouette is reported for Iris and not used to choose epsilon. S-curve and Swiss roll also report triplet accuracy against the 1-d manifold parameter. That is not an ambient Euclidean score.

Consensus stress is recorded and is not used to rank methods. Different view sets and different weights define different objectives.

Epsilon was not chosen on a development set and then frozen. The whole curve is the result. No single epsilon is presented as ordinary REDR performance.

## What was left out on purpose

Negative t-SNE, NCVis, COIL-20, MNIST at a full epoch budget, Fashion-MNIST, USPS, and the mammoth cloud were not run. LoCoMDS hyperparameters were not swept. There is one Iris pool and one S-curve pool, so the matched-seed standard deviations do not measure training-seed variation. A contamination proportion curve was not run. Only three deliberately bad Iris embeddings were added.
