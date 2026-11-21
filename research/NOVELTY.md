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

A publishable answer has to be one of those, with the baselines above, and it has to remain after independent replications. Checkpoint 2 did not establish a positive answer.

## Consequence for the experiments

The S-curve replication compares Rashomon sets to matched unrestricted ensembles and to one embedding per method. Beating a single favorite embedding, or beating an untuned default, would not distinguish REDR from ordinary ensembling.
