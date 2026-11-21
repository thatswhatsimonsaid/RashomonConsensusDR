"""Cluster settings for a later S-curve array.

Nothing here is copied from the active-learning partition table. Partition,
memory, modules, and mail are read from the environment so a laptop dry-run
cannot accidentally target a cluster account. The defaults are placeholders.
"""

from __future__ import annotations

import os

N_REPLICATIONS = 25

SLURM_CONFIG = {
    "partition": os.environ.get("REDR_SLURM_PARTITION", "CHANGE_ME"),
    "time_candidates": os.environ.get("REDR_SLURM_TIME_CANDIDATES", "04:00:00"),
    "time_consensus": os.environ.get("REDR_SLURM_TIME_CONSENSUS", "08:00:00"),
    "cpus": os.environ.get("REDR_SLURM_CPUS", "4"),
    "mem": os.environ.get("REDR_SLURM_MEM", "16G"),
    "mail_type": os.environ.get("REDR_SLURM_MAIL_TYPE", "FAIL"),
    "mail_user": os.environ.get("REDR_SLURM_MAIL", ""),
    "module": os.environ.get("REDR_SLURM_MODULE", ""),
    "python": os.environ.get("REDR_PYTHON", "python"),
    "r_module": os.environ.get("REDR_SLURM_R_MODULE", ""),
}

# Measured on the five-replication pilot, n=400, 27 embeddings per pool.
# Candidate generation: 0.72 hours of summed fit time across 135 embeddings,
# about 0.22 wall-clock hours per replication on 4 CPU threads.
# Consensus: 5.89 hours of summed R time, about 0.55 wall-clock hours per
# replication with two concurrent fits. These defaults are task wall-clock
# hours, not core-hours.
ESTIMATED_CANDIDATE_HOURS = float(os.environ.get("REDR_EST_CANDIDATE_HOURS", "0.25"))
ESTIMATED_CONSENSUS_HOURS = float(os.environ.get("REDR_EST_CONSENSUS_HOURS", "0.60"))
