# Related work and what would be new

This is a reading note, not a claim that REDR is new. The combination of two papers is not automatically a contribution.

## Already established

An and Tang (arXiv:2512.15802) aggregate several dimension-reduction embeddings, including different hyperparameters of the same algorithm, by CoMDS and LoCoMDS. The consensus step in this repository is that method. A result that only says "CoMDS on several embeddings works" repeats their paper.

Bai, Wang, Long, and Hu (AAAI 2017, doi:10.1609/aaai.v31i1.10732) fit one embedding to several distance matrices and learn view weights. Weighted multi-view MDS predates both source papers.

Sun, Huang, Parikh, and Rudin (arXiv:2604.00485) define empirical Rashomon sets of parametric DR embeddings and extract common neighborhood structure inside one method. They do not aggregate those sets with CoMDS.

Procrustes means and medians of repeated UMAP or t-SNE runs are an existing ensemble (for example arXiv:2608.14513). Stability of repeated embeddings is also measured by Procrustes correlation and rank correlation (scStability). A uniform average of a candidate pool is in this family. It is a baseline, not a proposed method.

Neighborhood disagreement across representation models is used as a reliability signal for foundation models (arXiv:2609.39848). That is a different task. It does warn against treating "Rashomon" plus "disagreement" as unused words.

## What is not settled by those papers

The open empirical question is narrower than "combine RashomonDR with CoMDS":

- Does discarding embeddings outside a native-loss band change the consensus, once the comparison is a size-matched draw from the same pool and a one-embedding-per-method consensus?
- Does any change show up in recovery of a two-dimensional unfolding, as opposed to ambient trustworthiness or a single curve parameter?
- When the score used for the band is only weakly related to latent error, does the band help, do nothing, or throw away useful views?
- Does a method-specific band keep a method that is consistently wrong, while a pooled threshold would drop it?

A publishable answer has to be one of those, with the baselines above, and it has to remain after independent replications. Checkpoint 2 did not establish a positive answer. Checkpoint 3 answered the S-curve version in the negative for the balanced surface.

## Checkpoint 4, still a pilot

The open question moved. CoMDS already aggregates heterogeneous embeddings. The empirical claim that would be new is narrower:

- When views preserve different intrinsic directions, a consensus of the complementary views recovers balanced geometry that the joint-only embeddings and a neighborhood-preservation filter both miss.
- That gain is the aggregator, not a within-method native-loss band. The band does not move.
- The same consensus does not invent a direction that no view contains.
- A pooled quality threshold is a different tool. It helps when a minority of views hold a direction the majority destroyed and the score tracks that error. It hurts when the score does not, and it hurts when the complementary views are the ones with worse full-matrix neighborhood preservation.

The four-replication pilot is consistent with that split. It is not a 25-replication confirmation, and LoCoMDS failed on two stretched pools where CoMDS did not. A paper would have to report the trustworthiness trade-off and the aggregator instability, and it would have to survive the cluster seeds in `experiments/study8_complementary/`. Beating Procrustes on complementary views is not, by itself, new: it is what CoMDS is for. The selection result is which rule keeps the complementary views and which rule deletes them.

## Consequence for the experiments

The S-curve replication compares Rashomon sets to matched unrestricted ensembles and to one embedding per method. Beating a single favorite embedding, or beating an untuned default, would not distinguish REDR from ordinary ensembling.

## Checkpoint 5

Sun, Huang, Parikh, and Rudin define the Rashomon set of a dimension-reduction loss as every parameterization within δ of the best loss (arXiv:2604.00485, Definition 3.1). Their extra term aligns non-neighbor pairs with PCA. The set is still one loss. It is not a union of embeddings that fail the full-matrix loss and succeed on different substructures.

Zheng, Zhu, Li, Pang, Wang, and Chen (arXiv:2004.03106) build a shared graph and a complementary graph for multi-view subspace clustering. The views are given, and the output is a clustering. An and Tang already aggregate given embeddings. Bai, Wang, Long, and Hu already combine given distance matrices.

Checkpoint 5 asked whether a Rashomon rule can discover those views from an unlabeled matrix and keep the ones a global score deletes. The local answer is no. Concatenating one-dimensional Isomaps of known blocks beats CoMDS of those blocks. After the columns are rotated, objective-specific bands, Pareto selection, and diversity rules do not beat Isomap, and the one clustering that beats Isomap on aligned columns is undone by its own 5% band. That is a boundary on the claim, not a new aggregator. A later claim has to beat unrestricted aggregation of the same automatic pool and beat Isomap after the blocks have been mixed, without using the latent coordinates to choose the pool.
