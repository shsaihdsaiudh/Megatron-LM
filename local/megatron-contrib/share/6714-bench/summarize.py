"""Summarize a run.sh results directory: one line per run, then the mean per arm.

Usage: python summarize.py <results dir>
"""

import collections
import glob
import json
import os
import sys


def main() -> None:
    out_dir = sys.argv[1]
    paths = sorted(
        glob.glob(os.path.join(out_dir, "[0-9]*-*.json")),
        key=lambda p: int(os.path.basename(p).split("-", 1)[0]),
    )
    if not paths:
        sys.exit(f"no run JSON files in {out_dir}")

    meta = json.load(open(paths[0]))["meta"]
    print(
        f"gpu={meta['gpu']} world={meta['world']} mesh={tuple(meta['mesh'])} "
        f"d={meta['hidden']} layers={meta['layers']} torch={meta['torch']} "
        f"nccl={'.'.join(map(str, meta['nccl']))} "
        f"CUDA_DEVICE_MAX_CONNECTIONS={meta['cuda_device_max_connections']}"
    )
    print("Units: ms. Each value is a per-rank median over the measured steps, then the max over ranks.")
    print()

    by_arm = collections.defaultdict(lambda: collections.defaultdict(list))
    losses = {}
    header = f"{'run':12s} {'mode':6s} {'T':>5s} {'GA':>2s} {'fwd_mb0':>8s} {'exposed':>8s} {'fwd_rest':>8s} {'step':>9s}"
    print(header)
    for path in paths:
        name = os.path.basename(path)[: -len(".json")]
        arm = name.split("-", 1)[1]
        data = json.load(open(path))
        for row in data["results"]:
            m = row["max"]
            key = (row["mode"], row["tokens"], row["ga"])
            print(
                f"{name:12s} {row['mode']:6s} {row['tokens']:5d} {row['ga']:2d} "
                f"{m['fwd_mb0']:8.1f} {m.get('exposed', float('nan')):8.1f} "
                f"{m.get('fwd_rest', float('nan')):8.1f} {m['step']:9.1f}"
            )
            for k in ("fwd_mb0", "exposed", "fwd_rest", "step"):
                if k in m:
                    by_arm[(arm, key)][k].append(m[k])
            losses.setdefault(key, set()).add(tuple(row["per_rank"]["loss_sum"]))
    print()

    print("Mean per arm:")
    print(f"{'arm':8s} {'mode':6s} {'T':>5s} {'GA':>2s} {'fwd_mb0':>8s} {'exposed':>8s} {'fwd_rest':>8s} {'step':>9s}")
    for (arm, (mode, t, ga)), vals in sorted(by_arm.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        mean = {k: sum(v) / len(v) for k, v in vals.items()}
        print(
            f"{arm:8s} {mode:6s} {t:5d} {ga:2d} {mean['fwd_mb0']:8.1f} "
            f"{mean.get('exposed', float('nan')):8.1f} {mean.get('fwd_rest', float('nan')):8.1f} "
            f"{mean['step']:9.1f}"
        )
    print()

    same = all(len(v) == 1 for v in losses.values())
    print(f"Losses identical across all runs: {same}")


if __name__ == "__main__":
    main()
