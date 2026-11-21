"""Write or describe SLURM arrays for the S-curve replication.

Dry-run is the default. It prints job counts and a rough CPU-hour total.
It does not contact a scheduler.

    python experiments/study1_s_curve_replication/generate_sbatch_arrays.py
    REDR_SLURM_PARTITION=compute python experiments/study1_s_curve_replication/generate_sbatch_arrays.py --write

Do not submit the arrays until the local five-replication result has been read
and the partition, account, modules, and environment path have been set.
"""

from __future__ import annotations

import argparse
import os
import stat
from pathlib import Path

from master_config import ESTIMATED_CANDIDATE_HOURS, ESTIMATED_CONSENSUS_HOURS, N_REPLICATIONS, SLURM_CONFIG

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
JOB_DIR = SCRIPT_DIR / "job_scripts"
LOG_DIR = SCRIPT_DIR / "slurm_logs"
CONFIG = "experiments/study1_s_curve_replication/cluster_config.yaml"


def _header(name: str, time_limit: str) -> str:
    mail = ""
    if SLURM_CONFIG["mail_user"]:
        mail = (
            f"#SBATCH --mail-type={SLURM_CONFIG['mail_type']}\n"
            f"#SBATCH --mail-user={SLURM_CONFIG['mail_user']}\n"
        )
    return f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --partition={SLURM_CONFIG['partition']}
#SBATCH --array=0-{N_REPLICATIONS - 1}
#SBATCH --cpus-per-task={SLURM_CONFIG['cpus']}
#SBATCH --mem={SLURM_CONFIG['mem']}
#SBATCH --time={time_limit}
#SBATCH --output={LOG_DIR}/{name}_%a.out
#SBATCH --error={LOG_DIR}/{name}_%a.err
{mail}
set -euo pipefail
cd {PROJECT_ROOT}
{f"module load {SLURM_CONFIG['module']}" if SLURM_CONFIG['module'] else "# module load <python module>"}
{f"module load {SLURM_CONFIG['r_module']}" if SLURM_CONFIG['r_module'] else "# module load <R module>"}
# source the virtualenv that has the vendored trainers and the coMDS R package.
# source .redr-venv/bin/activate
export PYTHONPATH=redr
export OMP_NUM_THREADS={SLURM_CONFIG['cpus']}
"""


def candidate_script() -> str:
    body = _header("redr_s_candidates", SLURM_CONFIG["time_candidates"])
    body += f"""
{SLURM_CONFIG['python']} -u redr/scripts/run_intrinsic_replication.py \\
  --config {CONFIG} \\
  --reps "$SLURM_ARRAY_TASK_ID" \\
  --stage candidates \\
  --workers 1
"""
    return body


def consensus_script() -> str:
    body = _header("redr_s_consensus", SLURM_CONFIG["time_consensus"])
    body += f"""
{SLURM_CONFIG['python']} -u redr/scripts/run_intrinsic_replication.py \\
  --config {CONFIG} \\
  --reps "$SLURM_ARRAY_TASK_ID" \\
  --stage consensus \\
  --workers 2
"""
    return body


def diagnose_script() -> str:
    body = _header("redr_s_diagnose", "00:30:00")
    # One process, not an array. The header still contains an array line;
    # the written file below replaces it.
    body = body.replace(
        f"#SBATCH --array=0-{N_REPLICATIONS - 1}\n",
        "",
    )
    listed = ",".join(str(i) for i in range(N_REPLICATIONS))
    body += f"""
{SLURM_CONFIG['python']} -u redr/scripts/run_intrinsic_replication.py \\
  --config {CONFIG} \\
  --reps {listed} \\
  --stage diagnose
"""
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    candidate_hours = N_REPLICATIONS * ESTIMATED_CANDIDATE_HOURS
    consensus_hours = N_REPLICATIONS * ESTIMATED_CONSENSUS_HOURS
    print(f"replications: {N_REPLICATIONS}")
    print("jobs: 1 diagnose task, then 2 arrays (candidates, consensus)")
    print(f"array tasks: {2 * N_REPLICATIONS}")
    print(f"placeholder candidate CPU-hours: {candidate_hours:.1f}")
    print(f"placeholder consensus CPU-hours: {consensus_hours:.1f}")
    print(f"partition: {SLURM_CONFIG['partition']}")
    print(f"mem: {SLURM_CONFIG['mem']}  cpus: {SLURM_CONFIG['cpus']}")
    print("Local seeds 1000-1004 are not in this array. Cluster seeds start at 1100.")
    print("Submit diagnose, then candidates, then consensus. Wait between arrays.")
    print("sbatch experiments/study1_s_curve_replication/job_scripts/diagnose.sbatch")
    print("sbatch experiments/study1_s_curve_replication/job_scripts/candidates.sbatch")
    print("sbatch experiments/study1_s_curve_replication/job_scripts/consensus.sbatch")
    if not args.write:
        print("Dry-run only. Pass --write to emit the sbatch files.")
        return
    if SLURM_CONFIG["partition"] == "CHANGE_ME":
        raise SystemExit("Set REDR_SLURM_PARTITION before --write.")
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for name, content in (
        ("diagnose.sbatch", diagnose_script()),
        ("candidates.sbatch", candidate_script()),
        ("consensus.sbatch", consensus_script()),
    ):
        path = JOB_DIR / name
        path.write_text(content)
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
