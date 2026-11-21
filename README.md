# RashomonConsensusDR

Rashomon ensembled dimension reduction. Method-specific sets of dimension-reduction embeddings are combined with consensus multidimensional scaling.

The research code, Checkpoint 1 results, and Checkpoint 2 experiments are in [`redr/`](redr/README.md).

| Document | Contents |
| --- | --- |
| `redr/README.md` | Install and the commands that reproduce the runs |
| `redr/REPORT.md` | Checkpoint 1 findings |
| `redr/DECISIONS.md` | Checkpoint 1 design choices |
| `redr/CHECKPOINT_2_REPORT.md` | Checkpoint 2 findings |
| `redr/CHECKPOINT_2_DECISIONS.md` | Checkpoint 2 design choices |
| `research/RESEARCH_LOG.md` | Checkpoint 3 log, including the frozen intrinsic metric |
| `research/NEGATIVE_RESULTS.md` | Unfavorable findings that stay in the record |

`external/` is for local checkouts of RashomonDR and coMDS and is not part of this repository. `redr/setup_vendor.sh` clones the pinned commits into `redr/vendor/`.
