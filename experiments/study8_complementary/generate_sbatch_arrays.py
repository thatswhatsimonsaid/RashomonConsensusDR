"""Dry-run or write SLURM arrays for the complementary-geometry study.

One array task is one replication of one condition. Conditions are square,
stretched, and square_hidden. The task id maps as condition = id // 25 and
replication = id % 25. Seeds start at 5000 and do not reuse the local pilot.

    python experiments/study8_complementary/generate_sbatch_arrays.py
    REDR_SLURM_PARTITION=compute python experiments/study8_complementary/generate_sbatch_arrays.py --write

Do not submit these arrays. They are prepared so a later run can separate
specialist consensus, joint consensus, a pooled structural filter, and
Procrustes averaging. They do not repeat the S-curve epsilon study.
"""

from __future__ import annotations

import argparse
import os
import stat
from pathlib import Path

from master_config import (
    ESTIMATED_CANDIDATE_HOURS,
    ESTIMATED_CONSENSUS_HOURS,
    N_CONDITIONS,
    N_REPLICATIONS,
    SLURM_CONFIG,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
JOB_DIR = SCRIPT_DIR / "job_scripts"
LOG_DIR = SCRIPT_DIR / "slurm_logs"
CONFIG = "experiments/study8_complementary/cluster_config.yaml"
N_TASKS = N_REPLICATIONS * N_CONDITIONS


def _header(name: str, time_limit: str) -> str:
    mail = ""
    if SLURM_CONFIG["mail_user"]:
        mail = (
            f"#SBATCH --mail-type={SLURM_CONFIG['mail_type']}\n"
            f"#SBATCH --mail-user={SLURM_CONFIG['mail_user']}\n"
        )
    module = (
        f"module load {SLURM_CONFIG['module']}"
        if SLURM_CONFIG["module"]
        else "# module load <python module>"
    )
    r_module = (
        f"module load {SLURM_CONFIG['r_module']}"
        if SLURM_CONFIG["r_module"]
        else "# module load <R module>"
    )
    return f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --partition={SLURM_CONFIG['partition']}
#SBATCH --array=0-{N_TASKS - 1}
#SBATCH --cpus-per-task={SLURM_CONFIG['cpus']}
#SBATCH --mem={SLURM_CONFIG['mem']}
#SBATCH --time={time_limit}
#SBATCH --output={LOG_DIR}/{name}_%a.out
#SBATCH --error={LOG_DIR}/{name}_%a.err
{mail}
set -euo pipefail
cd {PROJECT_ROOT}
{module}
{r_module}
# source .redr-venv/bin/activate
export PYTHONPATH=redr
export OMP_NUM_THREADS={SLURM_CONFIG['cpus']}
CONDITIONS=(square stretched square_hidden)
COND="${{CONDITIONS[$((SLURM_ARRAY_TASK_ID / {N_REPLICATIONS}))]}}"
REP=$((SLURM_ARRAY_TASK_ID % {N_REPLICATIONS}))
"""


def _body(stage: str) -> str:
    return f"""
{SLURM_CONFIG['python']} -u redr/scripts/run_complementary.py \\
  --config {CONFIG} \\
  --condition "$COND" \\
  --reps "$REP" \\
  --stage {stage} \\
  --kinds procrustes,comds,locomds
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    print(f"conditions: {N_CONDITIONS}  replications: {N_REPLICATIONS}")
    print(f"array tasks per stage: {N_TASKS}")
    print("stages: candidates, then consensus. Wait for candidates to finish.")
    print(f"candidate task-hours, wall clock: {N_TASKS * ESTIMATED_CANDIDATE_HOURS:.1f}")
    print(f"consensus task-hours, wall clock: {N_TASKS * ESTIMATED_CONSENSUS_HOURS:.1f}")
    print(f"partition: {SLURM_CONFIG['partition']}")
    print(f"mem: {SLURM_CONFIG['mem']}  cpus: {SLURM_CONFIG['cpus']}")
    print("Local pilot seeds 4000/4100/4200 are not in this array.")
    print("Cluster seeds start at 5000/5300/5200. Evaluation triplets use offset 11000.")
    print("This array does not repeat the S-curve confirmation study.")
    print("sbatch experiments/study8_complementary/job_scripts/candidates.sbatch")
    print("sbatch experiments/study8_complementary/job_scripts/consensus.sbatch")
    if not args.write:
        print("Dry-run only. Pass --write after REDR_SLURM_PARTITION is set.")
        return
    if SLURM_CONFIG["partition"] == "CHANGE_ME":
        raise SystemExit("Set REDR_SLURM_PARTITION before --write.")
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for name, stage, limit in (
        ("candidates.sbatch", "candidates", SLURM_CONFIG["time_candidates"]),
        ("consensus.sbatch", "consensus", SLURM_CONFIG["time_consensus"]),
    ):
        path = JOB_DIR / name
        path.write_text(_header(f"redr_c4_{stage}", limit) + _body(stage))
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
