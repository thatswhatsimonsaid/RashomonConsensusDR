"""Fit complementary embeddings and compare selection rules.

    PYTHONPATH=redr python redr/scripts/run_complementary.py \
        --config experiments/study8_complementary/config.yaml \
        --condition square --reps 0 --stage candidates
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "redr"))

from redr.complementary import (  # noqa: E402
    aggregate,
    consensus_plan,
    dataset_for,
    entropy_diagnostic,
    fit_candidates,
    geometric_diversity,
    plot_candidates,
    plot_rules,
    score_embedding,
    select_rule,
    triplets_for,
    write_lock,
)


def _load(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    config["_path"] = str(path)
    return config


def _condition(config: dict, name: str) -> dict:
    for item in config["conditions"]:
        if item["name"] == name:
            merged = dict(item)
            merged["n_samples"] = int(config["n_samples"])
            merged["noise"] = float(config["noise"])
            merged["n_nuisance"] = int(config["n_nuisance"])
            return merged
    raise SystemExit(f"unknown condition {name}")


def _out(condition: str, replication: int) -> Path:
    path = REPO / "results" / "checkpoint4" / "complementary" / condition / f"rep{replication:02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_embeddings(path: Path, embeddings: dict[str, np.ndarray]) -> None:
    ids = list(embeddings)
    stacked = np.stack([embeddings[key] for key in ids])
    np.savez_compressed(path, ids=np.array(ids), embeddings=stacked)


def _load_embeddings(path: Path) -> dict[str, np.ndarray]:
    archive = np.load(path)
    return {
        str(embedding_id): archive["embeddings"][i]
        for i, embedding_id in enumerate(archive["ids"])
    }


def run_candidates(config: dict, condition: dict, replication: int) -> None:
    out = _out(condition["name"], replication)
    dest = out / "candidates.csv"
    if dest.exists() and (out / "embeddings.npz").exists():
        print(f"candidates exist: {dest}", flush=True)
        return
    data = dataset_for(condition, replication)
    np.savez_compressed(out / "latent.npz", intrinsic=data["intrinsic"], X=data["X"])
    triplets = triplets_for(data["n"], int(config["eval_seed_offset"]) + replication)
    np.save(out / "triplets.npy", triplets)
    print(f"fitting {condition['name']} rep {replication}", flush=True)
    frame, embeddings = fit_candidates(data, triplets)
    frame.to_csv(dest, index=False)
    _save_embeddings(out / "embeddings.npz", embeddings)
    print(f"wrote {dest} ({len(frame)} embeddings)", flush=True)


def run_consensus(config: dict, condition: dict, replication: int, kinds: list[str]) -> None:
    out = _out(condition["name"], replication)
    frame = pd.read_csv(out / "candidates.csv")
    embeddings = _load_embeddings(out / "embeddings.npz")
    archive = np.load(out / "latent.npz")
    data = {
        "X": archive["X"],
        "intrinsic": archive["intrinsic"],
        "n": int(archive["X"].shape[0]),
    }
    triplets = np.load(out / "triplets.npy")
    sweep_path = out / "sweep.csv"
    done = set()
    if sweep_path.exists():
        previous = pd.read_csv(sweep_path)
        done = set(zip(previous["run_name"], previous["kind"]))
    rows = []
    for run_name, rule, scheme in consensus_plan():
        members = select_rule(rule, frame, embeddings)
        for kind in kinds:
            key = (run_name, kind)
            if key in done:
                print(f"skip {run_name} {kind}", flush=True)
                continue
            row = {
                "condition": condition["name"],
                "replication": replication,
                "run_name": run_name,
                "rule": rule,
                "scheme": scheme,
                "kind": kind,
                "n_members": int(len(members)),
                "n_sources": int(members["source"].nunique()) if len(members) else 0,
                "n_families": int(members["family"].nunique()) if len(members) else 0,
            }
            started = time.time()
            try:
                if len(members) == 0:
                    raise RuntimeError("empty selection")
                embedding = aggregate(members, embeddings, kind, scheme, out, run_name)
                scores = score_embedding(data, embedding, triplets)
                row.update(scores)
                row["n_joint"] = int(members["source"].eq("joint").sum()) if len(members) else 0
                row["n_u"] = int(members["source"].eq("u").sum()) if len(members) else 0
                row["n_v"] = int(members["source"].eq("v").sum()) if len(members) else 0
                if (rule, kind) in {
                    ("all", "procrustes"),
                    ("specialist", "locomds"),
                    ("joint", "locomds"),
                    ("pooled", "locomds"),
                }:
                    row.update(entropy_diagnostic(frame, embeddings, data, embedding))
                row["error"] = ""
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
            row["seconds"] = time.time() - started
            rows.append(row)
            pd.DataFrame(rows).to_csv(out / "sweep_partial.csv", index=False)
            print(
                f"  {run_name} {kind} n={row['n_members']} "
                f"balanced={row.get('intrinsic_triplet_balanced', float('nan'))} "
                f"u={row.get('u_triplet', float('nan'))} v={row.get('v_triplet', float('nan'))} "
                f"err={row['error'][:80]}",
                flush=True,
            )
    fresh = pd.DataFrame(rows)
    if sweep_path.exists():
        combined = pd.concat([pd.read_csv(sweep_path), fresh], ignore_index=True)
    else:
        combined = fresh
    if len(combined):
        combined.to_csv(sweep_path, index=False)
    partial = out / "sweep_partial.csv"
    if partial.exists():
        partial.unlink()


def summarize() -> None:
    root = REPO / "results" / "checkpoint4" / "complementary"
    frames = list(root.glob("*/rep*/sweep.csv"))
    candidates = list(root.glob("*/rep*/candidates.csv"))
    if not frames:
        print("no sweep files", flush=True)
        return
    sweep = pd.concat(
        [pd.read_csv(path).assign(path=str(path)) for path in frames],
        ignore_index=True,
    )
    cand = pd.concat(
        [
            pd.read_csv(path).assign(
                condition=path.parents[1].name,
                replication=int(path.parent.name.replace("rep", "")),
            )
            for path in candidates
        ],
        ignore_index=True,
    )
    diversity_rows = []
    for path in candidates:
        rep_dir = path.parent
        archive = rep_dir / "embeddings.npz"
        if not archive.exists():
            continue
        part = pd.read_csv(path)
        pairs = geometric_diversity(part, _load_embeddings(archive))
        pairs["condition"] = path.parents[1].name
        pairs["replication"] = int(path.parent.name.replace("rep", ""))
        diversity_rows.append(pairs)
    if diversity_rows:
        pairs = pd.concat(diversity_rows, ignore_index=True)
        summary_div = (
            pairs.groupby(["condition", "same_source"])["neighborhood_overlap"]
            .agg(["mean", "std", "count"])
        )
        summary_div.to_csv(root / "diversity_summary.csv")
        print(summary_div.round(3).to_string(), flush=True)
    sweep.to_csv(root / "sweep_all.csv", index=False)
    plot_rules(sweep, root / "figures" / "rules_balanced.png")
    if "condition" in cand.columns:
        plot_candidates(cand, root / "figures" / "candidate_axes.png")
    cand.to_csv(root / "candidates_all.csv", index=False)
    metrics = [
        "intrinsic_triplet_balanced",
        "u_triplet",
        "v_triplet",
        "trustworthiness",
        "intrinsic_procrustes",
    ]
    grouped = (
        sweep.groupby(["condition", "rule", "scheme", "kind"], dropna=False)[metrics]
        .agg(["mean", "std", "count"])
    )
    grouped.to_csv(root / "summary.csv")
    print(grouped.round(3).to_string(), flush=True)
    _paired(sweep, root)


def _paired(sweep: pd.DataFrame, root: Path) -> None:
    contrasts = [
        ("specialist", "joint"),
        ("pooled", "specialist"),
        ("specialist", "best"),
        ("specific", "all"),
        ("hybrid", "specialist"),
        ("diverse", "pooled"),
        ("all", "joint"),
    ]
    rows = []
    for (condition, kind), block in sweep.groupby(["condition", "kind"]):
        for left, right in contrasts:
            a = block.loc[(block["rule"] == left) & (block["scheme"] == "uniform")]
            b = block.loc[(block["rule"] == right) & (block["scheme"] == "uniform")]
            merged = a.merge(b, on="replication", suffixes=("_l", "_r"))
            if len(merged) == 0:
                continue
            for metric in ("intrinsic_triplet_balanced", "u_triplet", "v_triplet"):
                delta = merged[f"{metric}_l"] - merged[f"{metric}_r"]
                rows.append(
                    {
                        "condition": condition,
                        "kind": kind,
                        "contrast": f"{left}_minus_{right}",
                        "metric": metric,
                        "mean_delta": float(delta.mean()),
                        "sd_delta": float(delta.std(ddof=1)) if len(delta) > 1 else float("nan"),
                        "n": int(len(delta)),
                        "n_left_better": int((delta > 0).sum()),
                    }
                )
    for (condition, rule), block in sweep.groupby(["condition", "rule"]):
        if rule not in {"specialist", "joint", "all", "pooled"}:
            continue
        uniform = block.loc[block["scheme"].eq("uniform")]
        for left, right in (("locomds", "procrustes"), ("comds", "procrustes"), ("locomds", "comds")):
            a = uniform.loc[uniform["kind"] == left]
            b = uniform.loc[uniform["kind"] == right]
            merged = a.merge(b, on="replication", suffixes=("_l", "_r"))
            if len(merged) == 0:
                continue
            for metric in ("intrinsic_triplet_balanced", "u_triplet", "v_triplet"):
                delta = merged[f"{metric}_l"] - merged[f"{metric}_r"]
                rows.append(
                    {
                        "condition": condition,
                        "kind": f"{left}_vs_{right}",
                        "contrast": f"{rule}_{left}_minus_{right}",
                        "metric": metric,
                        "mean_delta": float(delta.mean()),
                        "sd_delta": float(delta.std(ddof=1)) if len(delta) > 1 else float("nan"),
                        "n": int(len(delta)),
                        "n_left_better": int((delta > 0).sum()),
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(root / "paired.csv", index=False)
    print(table.round(4).to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--condition", default="square")
    parser.add_argument("--reps", default="0")
    parser.add_argument("--stage", choices=("candidates", "consensus", "summarize", "all"), default="all")
    parser.add_argument("--kinds", default="procrustes,comds,locomds")
    args = parser.parse_args()
    config = _load(args.config)
    lock = REPO / "results" / "checkpoint4" / "complementary" / "study_lock.json"
    write_lock(lock)
    if args.stage == "summarize":
        summarize()
        return
    condition = _condition(config, args.condition)
    reps = [int(part) for part in args.reps.split(",") if part != ""]
    kinds = [part for part in args.kinds.split(",") if part]
    for replication in reps:
        if args.stage in {"candidates", "all"}:
            run_candidates(config, condition, replication)
        if args.stage in {"consensus", "all"}:
            run_consensus(config, condition, replication, kinds)


if __name__ == "__main__":
    main()
