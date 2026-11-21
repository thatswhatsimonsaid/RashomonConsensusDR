"""Placeholders for a complementary-geometry SLURM array.

Partition, memory, modules, and mail are environment variables. A dry-run
on a laptop must not name a real account. Estimates are wall-clock hours
per array task from the local n=200 pilot, not core-hours.
"""

from __future__ import annotations

import os

N_REPLICATIONS = 25
N_CONDITIONS = 3

SLURM_CONFIG = {
    "partition": os.environ.get("REDR_SLURM_PARTITION", "CHANGE_ME"),
    "time_candidates": os.environ.get("REDR_SLURM_TIME_CANDIDATES", "01:00:00"),
    "time_consensus": os.environ.get("REDR_SLURM_TIME_CONSENSUS", "01:00:00"),
    "cpus": os.environ.get("REDR_SLURM_CPUS", "4"),
    "mem": os.environ.get("REDR_SLURM_MEM", "16G"),
    "mail_type": os.environ.get("REDR_SLURM_MAIL_TYPE", "FAIL"),
    "mail_user": os.environ.get("REDR_SLURM_MAIL", ""),
    "module": os.environ.get("REDR_SLURM_MODULE", ""),
    "python": os.environ.get("REDR_PYTHON", "python"),
    "r_module": os.environ.get("REDR_SLURM_R_MODULE", ""),
}

# Measured on the local pilot, n=200, 33 embeddings. A square replication
# took about 2.5 minutes of wall clock for candidates and consensus together.
# Consensus itself was about 0.025 hours. The defaults leave headroom.
ESTIMATED_CANDIDATE_HOURS = float(os.environ.get("REDR_EST_CANDIDATE_HOURS", "0.05"))
ESTIMATED_CONSENSUS_HOURS = float(os.environ.get("REDR_EST_CONSENSUS_HOURS", "0.04"))
