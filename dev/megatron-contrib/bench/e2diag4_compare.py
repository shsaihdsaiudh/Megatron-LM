#!/usr/bin/env python3
"""E2 diagnostic: summarize 12 runs in 6 arms (2 repeats each) for Megatron-FSDP, 2 nodes x 8 GPUs (16 ranks).

Arms (dir name without the NN- prefix):
  main-on, p1-on     flag on, no extra env
  main-lo, p1-lo     NCCL_LAUNCH_ORDER_IMPLICIT=1
  main-gm0, p1-gm0   NCCL_GRAPH_MIXING_SUPPORT=0
main-* = baseline code (snapshot main-d2b0e35cc). p1-* = P1 code (snapshot p1-20261112),
outer all-gather on its own CUDA stream.

Runs (time order):
  01-main-on     02-p1-on      03-main-lo    04-p1-lo
  05-main-gm0    06-p1-gm0     07-p1-gm0     08-main-gm0
  09-p1-lo       10-main-lo    11-p1-on      12-main-on

Each run dir holds e1_d4096_L32.json, node0.sweep_d4096_L32.log, node{0,1}.sysinfo.txt and env.*.txt files.
Rows per run: train and nostep at T=1024 and T=4096, GA=4.

Sections printed as Markdown:
  1. Checks: env vars per run (env.*.txt), snapshot path per node (megatron_fsdp line in sysinfo),
     CUDA_DEVICE_MAX_CONNECTIONS in the sweep header, meta flag, loss bit-identity across all runs per row,
     peak memory per arm.
  2. Results: one table per row. Cell = mean of the two repeats (ms), delta vs main-on in percent,
     and repeat difference in percent ((second - first) / first). A "~" marks a delta whose absolute
     value is below the larger repeat difference of the arm and of its comparison base.
  2b. P1 vs matching main (p1-on vs main-on, p1-lo vs main-lo, p1-gm0 vs main-gm0) for fwd_rest and step.

Usage:
  python3 e2diag4_compare.py [--root DIR] [--out FILE.md] [--head FILE.md]
  --head: text file prepended to the output (used for the summary findings).
"""

import argparse
import glob
import json
import os
import re
import statistics
from collections import Counter

DEFAULT_ROOT = "/planning-users/yangyeyuan/megatron-contrib/results/e2/e2diag4-2node-20261113"
MODEL = "e1_d4096_L32.json"
SWEEP = "node0.sweep_d4096_L32.log"
N_NODES = 2
ENV_VARS = ("NCCL_LAUNCH_ORDER_IMPLICIT", "NCCL_GRAPH_MIXING_SUPPORT")

RUNS = [
    ("01-main-on", "main-on"),
    ("02-p1-on", "p1-on"),
    ("03-main-lo", "main-lo"),
    ("04-p1-lo", "p1-lo"),
    ("05-main-gm0", "main-gm0"),
    ("06-p1-gm0", "p1-gm0"),
    ("07-p1-gm0", "p1-gm0"),
    ("08-main-gm0", "main-gm0"),
    ("09-p1-lo", "p1-lo"),
    ("10-main-lo", "main-lo"),
    ("11-p1-on", "p1-on"),
    ("12-main-on", "main-on"),
]
RUN_NAMES = [r for r, _ in RUNS]
RUN_ARM = dict(RUNS)
ARMS = ["main-on", "p1-on", "main-lo", "p1-lo", "main-gm0", "p1-gm0"]
BASE = "main-on"
REPEATS = {arm: [r for r, a in RUNS if a == arm] for arm in ARMS}  # time order

# Expected snapshot directory name per code base (prefix of the arm name).
SNAPSHOT = {"main": "main-d2b0e35cc", "p1": "p1-20261112"}

# Expected env var value per arm. None = the variable must not appear in any env file.
ARM_ENV = {
    "main-on": {"NCCL_LAUNCH_ORDER_IMPLICIT": None, "NCCL_GRAPH_MIXING_SUPPORT": None},
    "p1-on": {"NCCL_LAUNCH_ORDER_IMPLICIT": None, "NCCL_GRAPH_MIXING_SUPPORT": None},
    "main-lo": {"NCCL_LAUNCH_ORDER_IMPLICIT": "1", "NCCL_GRAPH_MIXING_SUPPORT": None},
    "p1-lo": {"NCCL_LAUNCH_ORDER_IMPLICIT": "1", "NCCL_GRAPH_MIXING_SUPPORT": None},
    "main-gm0": {"NCCL_LAUNCH_ORDER_IMPLICIT": None, "NCCL_GRAPH_MIXING_SUPPORT": "0"},
    "p1-gm0": {"NCCL_LAUNCH_ORDER_IMPLICIT": None, "NCCL_GRAPH_MIXING_SUPPORT": "0"},
}
# Value of CUDA_DEVICE_MAX_CONNECTIONS printed in the sweep header (not set by any arm here).
EXPECT_CUDA = "None"

# (mode, T, GA) rows. Only GA=4 appears in these runs.
ROWS = [("train", 1024, 4), ("train", 4096, 4), ("nostep", 1024, 4), ("nostep", 4096, 4)]
# (JSON max field, column label)
METRICS = [("fwd_mb0", "fwd_mb0"), ("fwd_rest", "fwd_rest"), ("bwd_rest", "bwd_rest"),
           ("exposed", "exposed"), ("step", "step")]
# (p1 arm, main arm) pairs for section 2b, and the metrics compared there.
PAIRS = [("p1-on", "main-on"), ("p1-lo", "main-lo"), ("p1-gm0", "main-gm0")]
PAIR_METRICS = ("fwd_rest", "step")


def load(path):
    with open(path) as f:
        return json.load(f)


def load_all(root):
    data = {}
    for run in RUN_NAMES:
        doc = load(os.path.join(root, run, MODEL))
        data[run] = {
            "meta": doc["meta"],
            "rows": {(r["mode"], r["tokens"], r["ga"]): r for r in doc["results"]},
        }
    return data


def get_val(data, run, key, name):
    return data[run]["rows"][key]["max"][name]


def repeat_pct(data, key, name, arm):
    """Second repeat vs first repeat, in percent of the first."""
    r1, r2 = REPEATS[arm]
    v1 = get_val(data, r1, key, name)
    v2 = get_val(data, r2, key, name)
    return 100.0 * (v2 - v1) / v1


def arm_mean(data, key, name, arm):
    return statistics.mean(get_val(data, r, key, name) for r in REPEATS[arm])


def pct_delta(a, b):
    """Change from a to b in percent. Positive means b is larger (slower for times)."""
    return 100.0 * (b - a) / a


def compute_stats(data):
    rep, mean = {}, {}
    for key in ROWS:
        for name, _ in METRICS:
            for arm in ARMS:
                rep[(key, name, arm)] = repeat_pct(data, key, name, arm)
                mean[(key, name, arm)] = arm_mean(data, key, name, arm)
    return rep, mean


def snapshot_paths(path):
    """Full paths on the 'megatron_fsdp' lines of a sysinfo file. None if the file is missing."""
    if not os.path.isfile(path):
        return None
    paths = []
    with open(path) as f:
        for line in f:
            if line.startswith("megatron_fsdp "):
                paths.append(line.split()[1])
    return paths


def snapshot_name(path):
    m = re.search(r"/snapshots/([^/]+)/", path)
    return m.group(1) if m else None


def sweep_cuda_setting(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        for line in f:
            if line.startswith("[e1] world="):
                m = re.search(r"CUDA_DEVICE_MAX_CONNECTIONS=(\S+)", line)
                return m.group(1) if m else None
    return None


def env_values(path):
    """{var: value} for the ENV_VARS lines of one env.*.txt file (lines look like NAME=value)."""
    found = {}
    with open(path) as f:
        for line in f:
            key, sep, val = line.rstrip("\n").partition("=")
            if sep and key in ENV_VARS:
                found[key] = val
    return found


def describe(vals):
    """Summarize the values of one variable across the env files of a run."""
    if not vals:
        return "no files"
    counts = Counter("not set" if v is None else f"={v}" for v in vals)
    return ", ".join(f"{k} ({n}/{len(vals)})" for k, n in counts.items())


def mention_count(paths):
    hits = 0
    for p in paths:
        with open(p, errors="replace") as f:
            text = f.read()
        hits += any(v in text for v in ENV_VARS)
    return hits


def check_env(root):
    lines = ["### 1a. Env vars per run (env.*.txt, all node files)", "",
             "Expected: lo arms NCCL_LAUNCH_ORDER_IMPLICIT=1 and NCCL_GRAPH_MIXING_SUPPORT not set; "
             "gm0 arms NCCL_GRAPH_MIXING_SUPPORT=0 and NCCL_LAUNCH_ORDER_IMPLICIT not set; "
             "on arms neither set. Cell = value (number of env files with it).", "",
             "| run | arm | env files | NCCL_LAUNCH_ORDER_IMPLICIT | NCCL_GRAPH_MIXING_SUPPORT | status |",
             "|---|---|---|---|---|---|"]
    failures = 0
    for run in RUN_NAMES:
        arm = RUN_ARM[run]
        files = sorted(glob.glob(os.path.join(root, run, "env.*.txt")))
        found = [env_values(p) for p in files]
        ok = len(files) == N_NODES
        cells = []
        for var in ENV_VARS:
            vals = [f.get(var) for f in found]
            ok = ok and all(v == ARM_ENV[arm][var] for v in vals)
            cells.append(describe(vals))
        failures += 0 if ok else 1
        lines.append(f"| {run} | {arm} | {len(files)} | {cells[0]} | {cells[1]} | {'PASS' if ok else 'FAIL'} |")

    sweep_sys = []
    nccl = []
    for run in RUN_NAMES:
        sweep_sys += glob.glob(os.path.join(root, run, "node*.sweep_*.log"))
        sweep_sys += glob.glob(os.path.join(root, run, "node*.sysinfo.txt"))
        nccl += glob.glob(os.path.join(root, run, "nccl.*.log"))
    lines += ["",
              f"Sweep logs and sysinfo files that mention either variable: {mention_count(sweep_sys)} of "
              f"{len(sweep_sys)} files. NCCL nccl.*.log files that mention either variable: "
              f"{mention_count(nccl)} of {len(nccl)} files. "
              "The env.*.txt files are the only per-run record of these variables.",
              "", f"Env var failures: {failures}.", ""]
    return lines, failures


def check_snapshots(root, data):
    lines = ["### 1b. Snapshot path per node, sweep header and meta flag", "",
             "Snapshot = the `megatron_fsdp` line in node{0,1}.sysinfo.txt. Expected: main-* in "
             f"`snapshots/{SNAPSHOT['main']}/`, p1-* in `snapshots/{SNAPSHOT['p1']}/`. "
             f"Sweep header CUDA_DEVICE_MAX_CONNECTIONS expected {EXPECT_CUDA}.", "",
             "| run | arm | node0 snapshot path | node1 snapshot path | sweep CUDA_DEVICE_MAX_CONNECTIONS | "
             "meta flag | status |",
             "|---|---|---|---|---|---|---|"]
    failures = 0
    for run in RUN_NAMES:
        arm = RUN_ARM[run]
        exp = SNAPSHOT[arm.split("-")[0]]
        ok = True
        per_node = []
        for n in range(N_NODES):
            paths = snapshot_paths(os.path.join(root, run, f"node{n}.sysinfo.txt"))
            if paths is None:
                per_node.append("missing file")
                ok = False
                continue
            if not paths:
                per_node.append("no megatron_fsdp line")
                ok = False
                continue
            ok = ok and [snapshot_name(p) for p in paths] == [exp]
            per_node.append("; ".join(paths))
        cuda = sweep_cuda_setting(os.path.join(root, run, SWEEP))
        flag = data[run]["meta"]["overlap_outer_communication"]
        ok = ok and cuda == EXPECT_CUDA and flag is True
        failures += 0 if ok else 1
        lines.append(f"| {run} | {arm} | {per_node[0]} | {per_node[1]} | {cuda} | {flag} | "
                     f"{'PASS' if ok else 'FAIL'} |")
    lines += ["", f"Snapshot/header/flag failures: {failures}.", ""]
    return lines, failures


def loss_sig(row):
    return (tuple(row["per_rank"]["loss_sum"]), row["max"]["loss_sum"])


def check_losses(data):
    ref_run = RUN_NAMES[0]
    n_ranks = len(data[ref_run]["rows"][ROWS[0]]["per_rank"]["loss_sum"])
    lines = ["### 1c. Loss bit-identity across all 12 runs", "",
             f"Per row: per_rank.loss_sum ({n_ranks} values) and max.loss_sum compared by exact equality with "
             f"run {ref_run}.", "",
             "| mode | T | GA | runs identical to 01-main-on | differing runs |",
             "|---|---|---|---|---|"]
    total_bad = 0
    for key in ROWS:
        ref = loss_sig(data[ref_run]["rows"][key])
        diff = [r for r in RUN_NAMES[1:] if loss_sig(data[r]["rows"][key]) != ref]
        total_bad += len(diff)
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {len(RUN_NAMES) - len(diff)}/{len(RUN_NAMES)} | "
                     f"{', '.join(diff) if diff else 'none'} |")
    lines += ["", f"Run-row pairs that differ from run {ref_run}: {total_bad}.", ""]
    return lines, total_bad


def check_memory(data):
    lines = ["### 1d. Peak memory per arm (max.peak_mem_gb, GB)", "",
             "| mode | T | GA | arm | repeat 1 | repeat 2 |", "|---|---|---|---|---|---|"]
    for key in ROWS:
        for arm in ARMS:
            v = [get_val(data, r, key, "peak_mem_gb") for r in REPEATS[arm]]
            lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {arm} | {v[0]:.4f} | {v[1]:.4f} |")
    lines.append("")
    n_equal = 0
    for key in ROWS:
        vals = [get_val(data, r, key, "peak_mem_gb") for r in RUN_NAMES]
        spread = max(vals) - min(vals)
        n_equal += spread == 0.0
        verdict = "identical in all 12 runs" if spread == 0.0 else "differs"
        lines.append(f"- {key[0]} T={key[1]}: min {min(vals):.4f}, max {max(vals):.4f}, "
                     f"spread {spread:.4f}: {verdict}")
    lines += ["", f"Rows with identical peak memory in all 12 runs: {n_equal}/{len(ROWS)}.", ""]
    return lines, len(ROWS) - n_equal


def fmt_cell(key, name, arm, rep, mean):
    m = mean[(key, name, arm)]
    r = rep[(key, name, arm)]
    if arm == BASE:
        return f"{m:.2f} (base, rep {r:+.2f}%)"
    d = pct_delta(mean[(key, name, BASE)], m)
    th = max(abs(r), abs(rep[(key, name, BASE)]))
    mark = " ~" if abs(d) < th else ""
    return f"{m:.2f} (Δ {d:+.2f}%{mark}, rep {r:+.2f}%)"


def results_section(rep, mean):
    lines = ["## 2. Results (ms)", "",
             "Cell: mean of the two repeats (Δ vs main-on in percent, positive = larger; "
             "rep = repeat difference of the two runs in the arm, (second - first) / first). "
             "A \"~\" marks |Δ| below max(|rep| of this arm, |rep| of main-on) for the same metric.", ""]
    for key in ROWS:
        mode, t, ga = key
        lines += [f"### {mode} T={t} GA={ga}", "",
                  "| arm (runs) | " + " | ".join(lbl for _, lbl in METRICS) + " |",
                  "|---|" + "---|" * len(METRICS)]
        for arm in ARMS:
            runs = ", ".join(r.split("-")[0] for r in REPEATS[arm])
            cells = [fmt_cell(key, name, arm, rep, mean) for name, _ in METRICS]
            lines.append(f"| {arm} ({runs}) | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["### Repeat noise floor (|rep| %, all rows, all metrics)", "",
              "| arm | median | max |", "|---|---|---|"]
    for arm in ARMS:
        vals = [abs(rep[(k, n, arm)]) for k in ROWS for n, _ in METRICS]
        lines.append(f"| {arm} | {statistics.median(vals):.2f} | {max(vals):.2f} |")
    lines.append("")
    return lines


def pair_cell(key, name, arm, rep, mean):
    return f"{mean[(key, name, arm)]:.2f} (rep {rep[(key, name, arm)]:+.2f}%)"


def pair_delta(key, name, p1, main, rep, mean):
    d = pct_delta(mean[(key, name, main)], mean[(key, name, p1)])
    th = max(abs(rep[(key, name, p1)]), abs(rep[(key, name, main)]))
    mark = " ~" if abs(d) < th else ""
    return f"{d:+.2f}%{mark}"


def pairs_section(rep, mean):
    lines = ["## 2b. P1 vs matching main (ms)", "",
             "Δ = (p1 mean - main mean) / main mean, positive = p1 slower. "
             "rep = repeat difference of each arm. A \"~\" marks |Δ| below the larger of the two repeat differences.",
             ""]
    for key in ROWS:
        mode, t, ga = key
        lines += [f"### {mode} T={t} GA={ga}", "",
                  "| p1 vs main | fwd_rest main | fwd_rest p1 | Δ fwd_rest | step main | step p1 | Δ step |",
                  "|---|---|---|---|---|---|---|"]
        for p1, main in PAIRS:
            cells = []
            for name in PAIR_METRICS:
                cells += [pair_cell(key, name, main, rep, mean), pair_cell(key, name, p1, rep, mean),
                          pair_delta(key, name, p1, main, rep, mean)]
            lines.append(f"| {p1} vs {main} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def header():
    return [
        "# E2 diagnostic: 12 runs, 6 arms, d4096 L32, 2 nodes x 8 GPUs (16 ranks)",
        "",
        "Arms: main-on, p1-on (no extra env); main-lo, p1-lo (NCCL_LAUNCH_ORDER_IMPLICIT=1); "
        "main-gm0, p1-gm0 (NCCL_GRAPH_MIXING_SUPPORT=0). main-* = baseline code (snapshot "
        f"{SNAPSHOT['main']}); p1-* = P1 code (snapshot {SNAPSHOT['p1']}, outer all-gather on its own CUDA "
        "stream). Each arm has two runs.",
        "",
        "Metrics (JSON `max` values, ms): fwd_mb0, fwd_rest (microbatches 1..3 forward), bwd_rest "
        "(microbatches 1..3 backward), exposed, step.",
        "",
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default="")
    ap.add_argument("--head", default="")
    args = ap.parse_args()

    data = load_all(args.root)
    rep, mean = compute_stats(data)
    out = header()
    out.append("## 1. Checks")
    out.append("")
    s, _ = check_env(args.root)
    out += s
    s, _ = check_snapshots(args.root, data)
    out += s
    s, _ = check_losses(data)
    out += s
    s, _ = check_memory(data)
    out += s
    out += results_section(rep, mean)
    out += pairs_section(rep, mean)

    body = "\n".join(out) + "\n"
    head = ""
    if args.head:
        with open(args.head) as f:
            head = f.read().rstrip() + "\n\n---\n\n"
    text = head + body
    print(text, end="")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
