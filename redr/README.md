# Rashomon Ensembled Dimension Reduction (REDR)

Proof of concept that builds method-specific Rashomon sets of dimension-reduction embeddings and combines them with Consensus Multidimensional Scaling.

This directory is the REDR project in [RashomonConsensusDR](https://github.com/thatswhatsimonsaid/RashomonConsensusDR). The upstream checkouts of [RashomonDR](https://github.com/williamsyy/RashomonDR) and [coMDS](https://github.com/tiffanymtang/coMDS) are cloned by `setup_vendor.sh` and are not edited. Pins are in `VENDOR.lock`.

Checkpoint 1 results stay in `results/{iris,swiss,mnist}`. Checkpoint 2 adds `CHECKPOINT_2_REPORT.md`, `CHECKPOINT_2_DECISIONS.md`, and `results/checkpoint2/`. On macOS, install PyTorch from the selector at pytorch.org. The CPU wheel index below is for Linux.

Papers:

- Sun, Huang, Parikh, and Rudin. *The Rashomon Effect for Visualizing High-Dimensional Data.* arXiv:2604.00485.
- An and Tang. *Consensus dimension reduction via multi-view learning.* arXiv:2512.15802.

Findings, failures, and what was actually run are in `REPORT.md`. Design choices are in `DECISIONS.md`.

## Layout

```
redr/
  redr/                 Python integration package
  r/fit_consensus.R     Calls the installed coMDS package
  scripts/run_experiment.py
  scripts/verify_bridge.py
  vendor/               Cloned upstream repos (not committed)
  results/              CSV metrics, figures, and cached embeddings
```

## Install

Python 3.10+ and R 4.x with a C++ toolchain. On Debian/Ubuntu the R libraries used here are the `r-cran-*` packages for Rcpp, RcppArmadillo, dplyr, FNN, forcats, furrr, ggplot2, purrr, tibble, and tidyselect, plus `smacof` from CRAN.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r redr/requirements.txt
bash redr/setup_vendor.sh
```

`setup_vendor.sh` installs `contrastive-ne` and `parampacmap` editable with `--no-deps`. PyKeOps is skipped; the runs set `use_keops=False` and build the neighbor graph with Annoy. `parampacmap` imports `scanpy` and its training module imports PyTorch Lightning, so both are listed even though the ANN fits in this project do not call those code paths.

Check the CoMDS bridge without training:

```bash
PYTHONPATH=redr python redr/scripts/verify_bridge.py
```

## Run

From the repository root:

```bash
PYTHONPATH=redr python redr/scripts/run_experiment.py --datasets iris
```

The Iris defaults match the executed Iris run: methods `umap`, `infonce`, and `pacmap`; seeds `0 1 2`; PCA-alignment weights `0 0.5 5`; 100 CNE epochs; 450 PaCMAP epochs; Rashomon relative tolerance `0.05`. LoCoMDS uses `tau=0.1` and `percentile=0.5` and is not tuned on LCMC.

Other executed commands:

```bash
PYTHONPATH=redr python redr/scripts/run_experiment.py \
  --datasets swiss --seeds 0 1 --lambdas 0 5 --cne-epochs 80 --pacmap-epochs 220

PYTHONPATH=redr python redr/scripts/run_experiment.py \
  --datasets mnist --seeds 0 1 --lambdas 0 5 --cne-epochs 40 --pacmap-epochs 120
```

Outputs for each dataset:

- `results/<dataset>/metrics.csv` — candidates, consensus embeddings, and the common-knowledge UMAP
- `results/<dataset>/embeddings.png` — Procrustes-aligned display of the fits
- `results/<dataset>/consensus_metrics.png`
- `results/<dataset>/embeddings.npz` — the numeric embeddings
- `results/<dataset>/consensus/<name>/` — CoMDS `gspace.csv`, stress, and source weights
- `results/metrics_all.csv` — the three datasets stacked

Class labels are used for silhouette scores and for plot colors. Training sees a 2D PCA alignment target, not the labels.

## Checkpoint 2

Tests, then the two executed pools:

```bash
PYTHONPATH=redr python -m unittest discover -s redr/tests -v
PYTHONPATH=redr python redr/scripts/verify_bridge.py
PYTHONPATH=redr python redr/scripts/swiss_pacmap_anomaly.py
PYTHONPATH=redr python redr/scripts/run_checkpoint2.py --config redr/configs/checkpoint2_iris.yaml
PYTHONPATH=redr python redr/scripts/run_checkpoint2.py --config redr/configs/checkpoint2_s_curve.yaml
```

`run_checkpoint2.py` reuses `results/checkpoint2/<name>/candidates/*.npy` when those files are present. The Iris run trained 75 embeddings and fit 76 consensus jobs. The S-curve run trained 27 embeddings. Findings are in `CHECKPOINT_2_REPORT.md`.

## Checkpoint 3 intrinsic replications

The confirmatory endpoint is chosen from the unfolding, before consensus is fit. Run the geometry lock first, then candidates, then consensus.

```bash
PYTHONPATH=redr python redr/scripts/run_intrinsic_replication.py \
  --config experiments/study1_s_curve_replication/config.yaml \
  --reps 0,1,2,3,4 --stage diagnose
PYTHONPATH=redr python redr/scripts/run_intrinsic_replication.py \
  --config experiments/study1_s_curve_replication/config.yaml \
  --reps 0,1,2,3,4 --stage candidates
PYTHONPATH=redr python redr/scripts/run_intrinsic_replication.py \
  --config experiments/study1_s_curve_replication/config.yaml \
  --reps 0,1,2,3,4 --stage consensus --workers 2
PYTHONPATH=redr python redr/scripts/run_intrinsic_replication.py \
  --config experiments/study1_s_curve_replication/config.yaml \
  --reps 0,1,2,3,4 --stage summarize
```

The controlled-embedding mechanism pilot does not train UMAP or PaCMAP:

```bash
PYTHONPATH=redr python redr/scripts/run_controlled.py --replications 25
```

Cluster arrays are described, and not submitted, by:

```bash
python experiments/study1_s_curve_replication/generate_sbatch_arrays.py
```

## Checkpoint 4 complementary geometry

Selection rules and the product-manifold pilot are in `research/CHECKPOINT_4_REPORT.md`. Latent coordinates are not an input to selection.

```bash
PYTHONPATH=redr python redr/scripts/run_complementary.py \
  --config experiments/study8_complementary/config.yaml \
  --condition square --reps 0,1,2,3 --stage all
PYTHONPATH=redr python redr/scripts/run_complementary.py \
  --config experiments/study8_complementary/config.yaml --stage summarize
PYTHONPATH=redr python redr/scripts/run_controlled_consensus.py --reps 4
python experiments/study8_complementary/generate_sbatch_arrays.py
```

The S-curve confirmation array from Checkpoint 3 is not the study to submit. `experiments/study8_complementary/` is a dry-run for the Checkpoint 4 specialist-versus-joint comparison. Checkpoint 5 did not establish a Rashomon advantage over concatenation or Isomap, so that array is not a Rashomon confirmation and was not submitted.

## Checkpoint 5 automatic discovery

Baselines, the unlabeled candidate pool, and the dependence-clustering follow-up are in `research/CHECKPOINT_5_REPORT.md`. Latent coordinates are not an input to selection.

```bash
PYTHONPATH=redr python redr/scripts/run_baselines.py
PYTHONPATH=redr python redr/scripts/run_discovery.py
PYTHONPATH=redr python redr/scripts/run_alternatives.py
PYTHONPATH=redr python redr/scripts/plot_checkpoint5.py
```

## Checkpoint 6 original consensus question

The An and Tang subset and the parametric Rashomon comparison are in `research/CHECKPOINT_6_REPORT.md`. Labels and the Swiss-roll parameter are used only for scoring.

```bash
PYTHONPATH=redr python redr/scripts/run_checkpoint6.py
PYTHONPATH=redr python redr/scripts/score_checkpoint6_geometry.py
```

No 25-replication array was prepared. The pilot does not support a Rashomon restriction on these simulations.

## Checkpoint 7 quality and diversity

The factorial, the fresh-seed confirmation, and the Swiss-roll check with one planted bad family are in `research/CHECKPOINT_7_REPORT.md`. Latent coordinates are not an input to selection. The oracle rows are labeled and are an upper bound.

```bash
PYTHONPATH=redr python redr/scripts/run_quality_diversity.py
PYTHONPATH=redr python redr/scripts/plot_quality_diversity.py
PYTHONPATH=redr python redr/scripts/run_quality_diversity_confirm.py
PYTHONPATH=redr python redr/scripts/run_dr_mechanism.py
```

No 25-replication array was prepared. The 5% band does not beat unrestricted CoMDS and Gibbs weights together.

## Checkpoint 8 natural candidate pools

Trained UMAP, InfoNCE, negative t-SNE, NCVis, and PaCMAP embeddings are in `research/CHECKPOINT_8_REPORT.md`. Native losses are compared only inside one method, neighbor count, and alignment weight. Latent coordinates and digit labels are evaluation columns.

```bash
PYTHONPATH=redr python redr/scripts/run_natural_diversity.py
PYTHONPATH=redr python redr/scripts/plot_natural_diversity.py
```

No 25-replication array was prepared. At a fixed objective the 5% band keeps every trained seed.
