#!/usr/bin/env python3
"""Train a Checkpoint 2 candidate pool and sweep Rashomon tolerances.

Example:
  PYTHONPATH=redr python redr/scripts/run_checkpoint2.py --config redr/configs/checkpoint2_iris.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from redr.checkpoint2 import run_configured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_configured(args.config)


if __name__ == "__main__":
    main()
