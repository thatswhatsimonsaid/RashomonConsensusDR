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

# Local pilot on this machine, one S-curve pool of 27 embeddings:
# UMAP/InfoNCE about 15-40s each at n=400, consensus about 3-90s per fit
# depending on the number of views. Replace these after the pilot log
# has a finished replication.
ESTIMATED_CANDIDATE_HOURS = float(os.environ.get("REDR_EST_CANDIDATE_HOURS", "0.5"))
ESTIMATED_CONSENSUS_HOURS = float(os.environ.get("REDR_EST_CONSENSUS_HOURS", "1.5"))
