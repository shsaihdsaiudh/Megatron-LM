#!/usr/bin/env python3
"""E2 diagnostic: summarize 10 runs in 5 arms (2 repeats each) for Megatron-FSDP, 2 nodes x 8 GPUs.

Runs (time order) and arms:
  01-main-on          main        flag on                                  (arm main-on)
  02-main-dummy       main+dummy  flag on, one extra unused stream          (arm main-dummy)
  03-p1-on            P1          flag on                                  (arm p1-on)
  04-main-on-c32      main        flag on, CUDA_DEVICE_MAX_CONNECTIONS=32   (arm main-on-c32)
  05-p1-on-c32        P1          flag on, CUDA_DEVICE_MAX_CONNECTIONS=32   (arm p1-on-c32)
  06-p1-on-c32        P1          flag on, CUDA_DEVICE_MAX_CONNECTIONS=32   (arm p1-on-c32)
  07-main-on-c32      main        flag on, CUDA_DEVICE_MAX_CONNECTIONS=32   (arm main-on-c32)
  08-p1-on            P1          flag on                                  (arm p1-on)
  09-main-dummy       main+dummy  flag on, one extra unused stream          (arm main-dummy)
  10-main-on          main        flag on                                  (arm main-on)

Each run dir holds e1_d4096_L32.json, node0.sweep_d4096_L32.log and node{0,1}.sysinfo.txt.
Rows per run: train and nostep at T=1024 and T=4096, GA=4.

Sections printed as Markdown:
  1. Checks: snapshot per node matches the arm, CUDA_DEVICE_MAX_CONNECTIONS in the sweep header
     matches the arm, meta flag, losses bit-identical across all runs per row, peak memory per row.
  2. Results: one table per row. Cell = mean of the two repeats (ms), delta vs main-on in percent,
     and repeat difference in percent ((second - first) / first). A "~" marks a delta whose absolute
     value is below the larger repeat difference of the arm and of main-on.

Usage:
  python3 e2diag_compare.py [--root DIR] [--out FILE.md] [--head FILE.md]
  --head: text file prepended to the output (used for the summary conclusion).
"""

import argparse
import json
import os
import re
import statistics

DEFAULT_ROOT = "/planning-users/yangyeyuan/megatron-contrib/results/e2/e2diag-2node-20261112"
MODEL = "e1_d4096_L32.json"
SWEEP = "node0.sweep_d4096_L32.log"

RUNS = [
    ("01-main-on", "main-on"),
    ("02-main-dummy", "main-dummy"),
    ("03-p1-on", "p1-on"),
    ("04-main-on-c32", "main-on-c32"),
    ("05-p1-on-c32", "p1-on-c32"),
    ("06-p1-on-c32", "p1-on-c32"),
    ("07-main-on-c32", "main-on-c32"),
    ("08-p1-on", "p1-on"),
    ("09-main-dummy", "main-dummy"),
    ("10-main-on", "main-on"),
]
RUN_NAMES = [r for r, _ in RUNS]
RUN_ARM = dict(RUNS)
ARMS = ["main-on", "main-dummy", "p1-on", "main-on-c32", "p1-on-c32"]
BASE = "main-on"
REPEATS = {arm: [r for r, a in RUNS if a == arm] for arm in ARMS}  # time order

# Expected snapshot directory name and CUDA_DEVICE_MAX_CONNECTIONS value per arm.
EXPECT = {
    "main-on": ("main-d2b0e35cc", "None"),
    "main-dummy": ("main-dummystream-d2b0e35cc", "None"),
    "p1-on": ("p1-20261112", "None"),
    "main-on-c32": ("main-d2b0e35cc", "32"),
    "p1-on-c32": ("p1-20261112", "32"),
}

# (mode, T, GA) rows. Only GA=4 appears in these runs.
ROWS = [("train", 1024, 4), ("train", 4096, 4), ("nostep", 1024, 4), ("nostep", 4096, 4)]
# (JSON max field, column label)
METRICS = [("fwd_mb0", "fwd_mb0"), ("exposed", "E"), ("fwd_rest", "fwd_rest"), ("bwd_rest", "bwd_rest"), ("step", "step")]


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


def snapshots_in(path):
    found = set()
    with open(path) as f:
        for line in f:
            if line.startswith("megatron_fsdp "):
                m = re.search(r"/snapshots/([^/]+)/", line)
                found.add(m.group(1) if m else line.strip())
    return sorted(found)


def sweep_cuda_setting(path):
    with open(path) as f:
        for line in f:
            if line.startswith("[e1] world="):
                m = re.search(r"CUDA_DEVICE_MAX_CONNECTIONS=(\S+)", line)
                return m.group(1) if m else None
    return None


def check_env(root, data):
    lines = ["### 1a. Snapshot, sweep header and meta flag per run", "",
             "| run | arm | node0 snapshot | node1 snapshot | CUDA_DEVICE_MAX_CONNECTIONS (sweep header) | meta flag | status |",
             "|---|---|---|---|---|---|---|"]
    failures = 0
    for run in RUN_NAMES:
        arm = RUN_ARM[run]
        exp_snap, exp_cuda = EXPECT[arm]
        snaps = {n: snapshots_in(os.path.join(root, run, f"node{n}.sysinfo.txt")) for n in (0, 1)}
        cuda = sweep_cuda_setting(os.path.join(root, run, SWEEP))
        flag = data[run]["meta"]["overlap_outer_communication"]
        ok = all(snaps[n] == [exp_snap] for n in (0, 1)) and cuda == exp_cuda and flag is True
        failures += 0 if ok else 1
        lines.append(f"| {run} | {arm} | {', '.join(snaps[0])} | {', '.join(snaps[1])} | {cuda} | {flag} | "
                     f"{'PASS' if ok else 'FAIL'} |")
    lines += ["", f"Expected snapshot and CUDA setting per arm: "
                  + "; ".join(f"{a}: {s}, {c}" for a, (s, c) in EXPECT.items()) + ".",
              "", f"Snapshot/header/flag failures: {failures}.", ""]
    return lines, failures


def loss_sig(row):
    return (tuple(row["raw_rank0"]["losses"]), tuple(row["per_rank"]["loss_sum"]))


def check_losses(data):
    ref_run = RUN_NAMES[0]
    lines = ["### 1b. Loss bit-identity across all 10 runs", "",
             "Per row: raw_rank0.losses and per_rank.loss_sum (16 values) compared by exact equality with run "
             f"{ref_run}.", "",
             "| mode | T | GA | runs identical to 01-main-on | differing runs |",
             "|---|---|---|---|---|"]
    total_bad = 0
    for key in ROWS:
        ref = loss_sig(data[ref_run]["rows"][key])
        diff = [r for r in RUN_NAMES[1:] if loss_sig(data[r]["rows"][key]) != ref]
        total_bad += len(diff)
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {len(RUN_NAMES) - 1 - len(diff) + 1}/{len(RUN_NAMES)} | "
                     f"{', '.join(diff) if diff else 'none'} |")
    lines += ["", f"Rows that differ from run {ref_run} in any run: {total_bad} run-row pairs.", ""]
    return lines, total_bad


def check_memory(data):
    lines = ["### 1c. Peak memory per row (max.peak_mem_gb, GB)", "",
             "| mode | T | GA | min | max | spread | verdict |", "|---|---|---|---|---|---|---|"]
    n_equal = 0
    for key in ROWS:
        vals = [get_val(data, r, key, "peak_mem_gb") for r in RUN_NAMES]
        spread = max(vals) - min(vals)
        n_equal += spread == 0.0
        verdict = "equal in all 10 runs" if spread == 0.0 else "differs"
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {min(vals):.4f} | {max(vals):.4f} | {spread:.4f} | {verdict} |")
    lines += ["", f"Rows with equal peak memory in all 10 runs: {n_equal}/{len(ROWS)}.", ""]
    return lines, len(ROWS) - n_equal


def fmt_cell(data, key, name, arm, rep, mean):
    m = mean[(key, name, arm)]
    r = rep[(key, name, arm)]
    if arm == BASE:
        return f"{m:.2f} (base, rep {r:+.2f}%)"
    d = pct_delta(mean[(key, name, BASE)], m)
    th = max(abs(r), abs(rep[(key, name, BASE)]))
    mark = " ~" if abs(d) < th else ""
    return f"{m:.2f} (Δ {d:+.2f}%{mark}, rep {r:+.2f}%)"


def results_section(data):
    rep, mean = {}, {}
    for key in ROWS:
        for name, _ in METRICS:
            for arm in ARMS:
                rep[(key, name, arm)] = repeat_pct(data, key, name, arm)
                mean[(key, name, arm)] = arm_mean(data, key, name, arm)

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
            cells = [fmt_cell(data, key, name, arm, rep, mean) for name, _ in METRICS]
            lines.append(f"| {arm} ({runs}) | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["### Repeat noise floor (|rep| %, all rows, all metrics)", "",
              "| arm | median | max |", "|---|---|---|"]
    for arm in ARMS:
        vals = [abs(rep[(k, n, arm)]) for k in ROWS for n, _ in METRICS]
        lines.append(f"| {arm} | {statistics.median(vals):.2f} | {max(vals):.2f} |")
    lines.append("")
    return lines


def header():
    return [
        "# E2 diagnostic: 10 runs, 5 arms, d4096 L32, 2 nodes x 8 GPUs (16 ranks)",
        "",
        "Arms: main-on (main code, flag on); main-dummy (main plus one never-used stream when the flag is on); "
        "p1-on (P1 code, flag on); main-on-c32 and p1-on-c32 (same code as main-on and p1-on, with "
        "CUDA_DEVICE_MAX_CONNECTIONS=32). Each arm has two runs.",
        "",
        "Metrics (JSON `max` values, each a per-rank median, then the max over 16 ranks): fwd_mb0, E (exposed), "
        "fwd_rest (microbatches 1..3 forward), bwd_rest (microbatches 1..3 backward), step. Units: ms.",
        "",
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default="")
    ap.add_argument("--head", default="")
    args = ap.parse_args()

    data = load_all(args.root)
    out = header()
    out.append("## 1. Checks")
    out.append("")
    s1, _ = check_env(args.root, data)
    out += s1
    s2, _ = check_losses(data)
    out += s2
    s3, _ = check_memory(data)
    out += s3
    out += results_section(data)

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
