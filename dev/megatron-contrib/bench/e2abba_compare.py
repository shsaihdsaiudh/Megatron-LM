#!/usr/bin/env python3
"""E2 ABBA: analyze a 3-arm interleaved 2-node benchmark (e1_exposure.py output).

Six runs in time order (1-off, 2-main-on, 3-p1-on, 4-p1-on, 5-main-on, 6-off). Each run dir has
e1_d4096_L32.json, e1_d8192_L16.json, node0.sweep_*.log and node{0,1}.sysinfo.txt.

Arms:
  off      P1 code, overlap_outer_communication=False (runs 1-off and 6-off)
  main-on  main code, flag True: reduce-side stream split only (runs 2-main-on and 5-main-on)
  p1-on    P1 code, flag True: reduce-side split + DP-outer all-gather stream (runs 3-p1-on and 4-p1-on)

Sections printed as Markdown:
  1. Sanity: arm/snapshot/flag match, bit-identical losses across all six runs, peak memory,
     sweep logs vs JSON, E recomputed from raw rank-0 forward times.
  2. Drift: repeat-to-repeat % difference per arm (the noise floor).
  3. Main table per model: mean of the two repeats per arm, deltas, "within noise" flags.
  4. Attribution: large-T slowdown in reduce-side split (main-on vs off) and in P1 (p1-on vs main-on).
  5. Per-step IQR on rank 0 for train GA=4.

Usage:
  python3 e2abba_compare.py [--root DIR] [--out FILE.md]
"""

import argparse
import json
import os
import re
import statistics

DEFAULT_ROOT = "/planning-users/yangyeyuan/megatron-contrib/results/e2/e2abba-2node-20261112"

# (JSON file, node0 sweep log) for each model.
MODELS = [
    ("e1_d4096_L32.json", "node0.sweep_d4096_L32.log"),
    ("e1_d8192_L16.json", "node0.sweep_d8192_L16.log"),
]

# Runs in time order, with their arm.
RUNS = [
    ("1-off", "off"),
    ("2-main-on", "main-on"),
    ("3-p1-on", "p1-on"),
    ("4-p1-on", "p1-on"),
    ("5-main-on", "main-on"),
    ("6-off", "off"),
]
RUN_NAMES = [r for r, _ in RUNS]
RUN_ARM = dict(RUNS)
ARMS = ["off", "main-on", "p1-on"]
REPEATS = {arm: [r for r, a in RUNS if a == arm] for arm in ARMS}  # time order: first, second

# Expected code snapshot family (from the megatron_fsdp line in sysinfo) and meta flag per arm.
EXPECT_FAMILY = {"off": "p1", "main-on": "main", "p1-on": "p1"}
EXPECT_FLAG = {"off": False, "main-on": True, "p1-on": True}

TOKENS = [256, 1024, 4096]
LARGE_T = [1024, 4096]

# (label, mode, GA, field in JSON "max" dict)
METRICS = [
    ("E (train GA=4)", "train", 4, "exposed"),
    ("fwd_rest (train GA=4)", "train", 4, "fwd_rest"),
    ("step (train GA=1)", "train", 1, "step"),
    ("step (train GA=4)", "train", 4, "step"),
    ("step (nostep GA=4)", "nostep", 4, "step"),
]
STEP_LABELS = [m[0] for m in METRICS if m[3] == "step"]

# Attribution comparisons: (baseline arm, compared arm, name).
COMPARE = [
    ("off", "main-on", "reduce-side split (main-on vs off)"),
    ("main-on", "p1-on", "P1 (p1-on vs main-on)"),
]

LOG_LINE_RE = re.compile(r"^\[e1\] (train|nostep)\s+T=\s*(\d+)\s+GA=(\d+)\s+(.*)$")
LOG_KV_RE = re.compile(r"(\w+)=\s*(-?\d+(?:\.\d+)?)")
# (log key, JSON max field, tolerance). The log prints ms with 2 decimals and memory with 1 decimal.
LOG_FIELDS = [
    ("fwd_mb0", "fwd_mb0", 0.006),
    ("E", "exposed", 0.006),
    ("fwd_rest", "fwd_rest", 0.006),
    ("step", "step", 0.006),
    ("opt", "opt", 0.006),
    ("mem", "peak_mem_gb", 0.051),
]


def load(path):
    with open(path) as f:
        return json.load(f)


def load_all(root):
    data = {}
    for run in RUN_NAMES:
        data[run] = {}
        for model, _ in MODELS:
            doc = load(os.path.join(root, run, model))
            data[run][model] = {
                "meta": doc["meta"],
                "rows": {(r["mode"], r["tokens"], r["ga"]): r for r in doc["results"]},
            }
    return data


def get_val(data, run, model, mode, t, ga, name):
    return data[run][model]["rows"][(mode, t, ga)]["max"][name]


def arm_mean(data, model, mode, t, ga, name, arm):
    return statistics.mean(get_val(data, r, model, mode, t, ga, name) for r in REPEATS[arm])


def repeat_pct(data, model, mode, t, ga, name, arm):
    """Second repeat vs first repeat, in percent of the first."""
    r1, r2 = REPEATS[arm]
    v1 = get_val(data, r1, model, mode, t, ga, name)
    v2 = get_val(data, r2, model, mode, t, ga, name)
    return 100.0 * (v2 - v1) / v1


def build_stats(data):
    rep, mean = {}, {}
    for model, _ in MODELS:
        for label, mode, ga, name in METRICS:
            for t in TOKENS:
                for arm in ARMS:
                    rep[(model, label, t, arm)] = repeat_pct(data, model, mode, t, ga, name, arm)
                    mean[(model, label, t, arm)] = arm_mean(data, model, mode, t, ga, name, arm)
    return rep, mean


def pct_delta(a, b):
    """Change from a to b in percent. Positive means b is larger (slower for times)."""
    return 100.0 * (b - a) / a


def thr(rep, model, label, t, a, b):
    """Noise threshold for a delta between arms a and b: the larger repeat difference of the two arms."""
    return max(abs(rep[(model, label, t, a)]), abs(rep[(model, label, t, b)]))


def cell(d, th):
    return f"{d:+.2f}%" + (" (within noise)" if abs(d) < th else "")


def verdict_cell(d, th):
    if abs(d) < th:
        return f"{d:+.2f}% (within noise)"
    return f"{d:+.2f}% (slower)" if d > 0 else f"{d:+.2f}% (faster)"


def fmt(x, nd=1):
    return f"{x:.{nd}f}"


def family(snapshot):
    return snapshot.split("-")[0]


def sysinfo_snapshots(path):
    found = set()
    with open(path) as f:
        for line in f:
            if line.startswith("megatron_fsdp "):
                m = re.search(r"/snapshots/([^/]+)/", line)
                found.add(m.group(1) if m else line.strip())
    return sorted(found)


def loss_sig(row):
    return (tuple(row["raw_rank0"]["losses"]), tuple(row["per_rank"]["loss_sum"]))


def max_abs_diff(a, b):
    if len(a) != len(b):
        return float("nan")
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def parse_log(path):
    vals = {}
    with open(path) as f:
        for line in f:
            m = LOG_LINE_RE.match(line.strip())
            if not m:
                continue
            mode, t, ga, rest = m.groups()
            vals[(mode, int(t), int(ga))] = {k: float(v) for k, v in LOG_KV_RE.findall(rest)}
    return vals


def recompute_e(row):
    """E = median over steps of (fwd of microbatch 0 - median of fwd of the other microbatches)."""
    per_step = [f[0] - statistics.median(f[1:]) for f in row["raw_rank0"]["fwd"]]
    return statistics.median(per_step)


def iqr(xs):
    q = statistics.quantiles(xs, n=4, method="inclusive")
    return q[2] - q[0]


def header():
    return [
        "# E2 ABBA: 2-node benchmark, arms off / main-on / p1-on",
        "",
        "Runs in time order: 1-off, 2-main-on, 3-p1-on, 4-p1-on, 5-main-on, 6-off.",
        "",
        "Arms:",
        "- off: P1 code, overlap_outer_communication=False.",
        "- main-on: main code, flag True. Reduce-side stream split only (#6973).",
        "- p1-on: P1 code, flag True. Reduce-side split plus the DP-outer all-gather stream (P1).",
        "",
        "Units: ms. E = JSON `max.exposed`. step and fwd_rest are JSON `max` values. Each `max` value is a per-rank median, then the max over 16 ranks.",
        "",
    ]


def check_arms(root, data):
    lines = ["## 1. Sanity", "", "### 1a. Arm, code snapshot and meta flag", "",
             "| run | arm | node0 snapshot | node1 snapshot | flag (d4096 / d8192) | status |",
             "|---|---|---|---|---|---|"]
    failures = 0
    for run in RUN_NAMES:
        arm = RUN_ARM[run]
        snaps = {n: sysinfo_snapshots(os.path.join(root, run, f"node{n}.sysinfo.txt")) for n in (0, 1)}
        flags = [data[run][m]["meta"]["overlap_outer_communication"] for m, _ in MODELS]
        ok = all(len(snaps[n]) == 1 and family(snaps[n][0]) == EXPECT_FAMILY[arm] for n in (0, 1))
        ok = ok and all(f == EXPECT_FLAG[arm] for f in flags)
        failures += 0 if ok else 1
        lines.append(f"| {run} | {arm} | {', '.join(snaps[0])} | {', '.join(snaps[1])} | "
                     f"{flags[0]} / {flags[1]} | {'PASS' if ok else 'FAIL'} |")
    lines.append("")
    lines.append("Expected: off -> p1 snapshot, flag False; main-on -> main snapshot, flag True; "
                 "p1-on -> p1 snapshot, flag True.")

    for model, _ in MODELS:
        keys = {k: v for k, v in data[RUN_NAMES[0]][model]["meta"].items() if k != "overlap_outer_communication"}
        ref = json.dumps(keys, sort_keys=True)
        same = all(
            json.dumps({k: v for k, v in data[r][model]["meta"].items() if k != "overlap_outer_communication"},
                       sort_keys=True) == ref
            for r in RUN_NAMES
        )
        failures += 0 if same else 1
        lines.append(f"Meta other than the flag identical across the six runs ({model}): {'PASS' if same else 'FAIL'}.")
    lines.append("")
    lines.append(f"Arm/flag/snapshot failures: {failures}.")
    lines.append("")
    return lines


def check_losses(data):
    lines = ["### 1b. Loss bit-identity across the six runs", "",
             "Compared per (model, mode, T, GA): raw_rank0.losses and per_rank.loss_sum (16 values), exact float equality against run 1-off.", ""]
    checked = identical = 0
    bad = []
    for model, _ in MODELS:
        ref_run = RUN_NAMES[0]
        for key in sorted(data[ref_run][model]["rows"]):
            checked += 1
            ref = loss_sig(data[ref_run][model]["rows"][key])
            row_ok = True
            for run in RUN_NAMES[1:]:
                other = loss_sig(data[run][model]["rows"][key])
                if other != ref:
                    row_ok = False
                    bad.append(
                        f"| {model} | {key[0]} | {key[1]} | {key[2]} | {run} | "
                        f"{'losses ' if other[0] != ref[0] else ''}{'loss_sum' if other[1] != ref[1] else ''} | "
                        f"{max_abs_diff(other[0], ref[0]):.3g} | {max_abs_diff(other[1], ref[1]):.3g} |"
                    )
            identical += row_ok
    lines.append(f"Rows checked: {checked}. Rows bit-identical in all six runs: {identical}/{checked}.")
    if bad:
        lines += ["", "| model | mode | T | GA | run | differs | max abs diff losses | max abs diff loss_sum |",
                  "|---|---|---|---|---|---|---|---|"] + bad
    lines.append("")
    return lines


def check_memory(data):
    cols = [f"{r} ({RUN_ARM[r]})" for r in RUN_NAMES]
    lines = ["### 1c. Peak memory (JSON max.peak_mem_gb, GB)", "",
             "| model | mode | T | GA | " + " | ".join(cols) + " | spread (GB) | verdict |",
             "|" + "---|" * (4 + len(cols) + 2)]
    n_equal = n_rows = 0
    for model, _ in MODELS:
        for key in sorted(data[RUN_NAMES[0]][model]["rows"]):
            vals = [data[r][model]["rows"][key]["max"]["peak_mem_gb"] for r in RUN_NAMES]
            spread = max(vals) - min(vals)
            n_rows += 1
            n_equal += spread == 0.0
            verdict = "identical in all six runs" if spread == 0.0 else f"differs by {spread:.4f}"
            lines.append(f"| {model} | {key[0]} | {key[1]} | {key[2]} | "
                         + " | ".join(f"{v:.4f}" for v in vals) + f" | {spread:.4f} | {verdict} |")
    lines.append("")
    lines.append(f"Rows with identical peak memory in all six runs: {n_equal}/{n_rows}.")
    lines.append("")
    return lines


def check_logs_and_e(root, data):
    lines = ["### 1d. Sweep logs vs JSON, and E recomputed from raw rank-0 data", "",
             "| run | model | log rows | JSON rows | values compared | outside tolerance | max abs diff |",
             "|---|---|---|---|---|---|---|"]
    log_fail = 0
    for run in RUN_NAMES:
        for model, logname in MODELS:
            log = parse_log(os.path.join(root, run, logname))
            rows = data[run][model]["rows"]
            n = bad = 0
            worst = 0.0
            for key, kv in log.items():
                row = rows.get(key)
                if row is None:
                    bad += 1
                    continue
                for logk, jk, tol in LOG_FIELDS:
                    if logk not in kv:
                        continue
                    n += 1
                    d = abs(kv[logk] - row["max"][jk])
                    worst = max(worst, d)
                    if d > tol:
                        bad += 1
            log_fail += bad
            lines.append(f"| {run} | {model} | {len(log)} | {len(rows)} | {n} | {bad} | {worst:.4f} |")
    lines.append("")
    lines.append(f"Log-vs-JSON mismatches: {log_fail}.")
    lines.append("")

    lines += ["Recomputed E (median over steps of fwd[mb0] - median(fwd[mb1..])) vs per_rank.exposed[0], GA=4 rows:", "",
              "| run | model | rows | max abs diff (ms) |", "|---|---|---|---|"]
    e_worst = 0.0
    for run in RUN_NAMES:
        for model, _ in MODELS:
            n = 0
            worst = 0.0
            for key, row in data[run][model]["rows"].items():
                if key[2] != 4 or "exposed" not in row["per_rank"]:
                    continue
                n += 1
                worst = max(worst, abs(recompute_e(row) - row["per_rank"]["exposed"][0]))
            e_worst = max(e_worst, worst)
            lines.append(f"| {run} | {model} | {n} | {worst:.2e} |")
    lines.append("")
    lines.append(f"Largest recomputed-E difference: {e_worst:.2e} ms.")
    lines.append("")
    return lines


def check_drift(data, rep):
    lines = ["## 2. Drift: repeat-to-repeat difference per arm (noise floor)", "",
             "Each cell: first value -> second value (% difference, (second - first) / first). "
             "Repeats: off 1-off -> 6-off, main-on 2-main-on -> 5-main-on, p1-on 3-p1-on -> 4-p1-on.", ""]
    for model, _ in MODELS:
        lines += [f"### {model}", "",
                  "| metric | T | off (1 -> 6) | main-on (2 -> 5) | p1-on (3 -> 4) |", "|---|---|---|---|---|"]
        for label, mode, ga, name in METRICS:
            for t in TOKENS:
                cells = []
                for arm in ARMS:
                    r1, r2 = REPEATS[arm]
                    v1 = get_val(data, r1, model, mode, t, ga, name)
                    v2 = get_val(data, r2, model, mode, t, ga, name)
                    cells.append(f"{v1:.1f} -> {v2:.1f} ({rep[(model, label, t, arm)]:+.2f}%)")
                lines.append(f"| {label} | {t} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["Noise floor summary: absolute repeat difference (%) per arm.", "",
              "| metric | arm | cells | median abs | max abs |", "|---|---|---|---|---|"]
    for label, _, _, _ in METRICS:
        for arm in ARMS:
            vals = [abs(rep[(m, label, t, arm)]) for m, _ in MODELS for t in TOKENS]
            lines.append(f"| {label} | {arm} | {len(vals)} | {statistics.median(vals):.2f}% | {max(vals):.2f}% |")
    step_vals = [abs(rep[(m, l, t, a)]) for m, _ in MODELS for l in STEP_LABELS for t in TOKENS for a in ARMS]
    lines += ["", f"All step metrics (GA=1, GA=4, nostep GA=4; both models; all T; all arms): "
                  f"median abs {statistics.median(step_vals):.2f}%, max abs {max(step_vals):.2f}%.", ""]
    return lines


def check_main(rep, mean):
    lines = ["## 3. Main table (mean of the two repeats per arm, ms)", "",
             "Delta = (b - a) / a, positive = slower. A delta is marked (within noise) when its absolute value is below "
             "the larger of the two arms' repeat differences for the same row (section 2). Each repeat difference is a single "
             "pair, so each threshold is a rough estimate.", ""]
    for model, _ in MODELS:
        lines += [f"### {model}", "",
                  "| T | metric | off | main-on | p1-on | main-on vs off | p1-on vs main-on (P1) | p1-on vs off (total) |",
                  "|---|---|---|---|---|---|---|---|"]
        for t in TOKENS:
            for label, _, _, _ in METRICS:
                off = mean[(model, label, t, "off")]
                mo = mean[(model, label, t, "main-on")]
                p1 = mean[(model, label, t, "p1-on")]
                d1 = pct_delta(off, mo)
                d2 = pct_delta(mo, p1)
                d3 = pct_delta(off, p1)
                t1 = thr(rep, model, label, t, "off", "main-on")
                t2 = thr(rep, model, label, t, "main-on", "p1-on")
                t3 = thr(rep, model, label, t, "off", "p1-on")
                lines.append(f"| {t} | {label} | {fmt(off)} | {fmt(mo)} | {fmt(p1)} | "
                             f"{cell(d1, t1)} | {cell(d2, t2)} | {cell(d3, t3)} |")
        lines.append("")
    return lines


def large_t_verdict(rep, mean, model, label, a, b):
    parts, flags = [], []
    for t in LARGE_T:
        d = pct_delta(mean[(model, label, t, a)], mean[(model, label, t, b)])
        th = thr(rep, model, label, t, a, b)
        parts.append(f"T={t} {d:+.2f}% (noise {th:.2f}%)")
        flags.append("slow" if d > th else ("fast" if d < -th else "noise"))
    if all(f == "slow" for f in flags):
        status = "slower at both large T"
    elif all(f == "fast" for f in flags):
        status = "faster at both large T"
    else:
        status = "not slower at both large T"
    return "; ".join(parts), status


def check_attribution(rep, mean):
    lines = ["## 4. Attribution", "",
             "Step-time deltas per T. Verdict per cell: slower / faster only when the delta is beyond the noise threshold.", ""]
    for model, _ in MODELS:
        lines += [f"### {model}", "",
                  "| T | metric | main-on vs off (reduce-side split) | p1-on vs main-on (P1) | p1-on vs off (total) |",
                  "|---|---|---|---|---|"]
        for t in TOKENS:
            for label in STEP_LABELS:
                cells = []
                for a, b in [("off", "main-on"), ("main-on", "p1-on"), ("off", "p1-on")]:
                    d = pct_delta(mean[(model, label, t, a)], mean[(model, label, t, b)])
                    cells.append(verdict_cell(d, thr(rep, model, label, t, a, b)))
                lines.append(f"| {t} | {label} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["Large-T attribution (T=1024 and T=4096 both required):", ""]
    for model, _ in MODELS:
        for label in STEP_LABELS:
            for a, b, name in COMPARE:
                detail, status = large_t_verdict(rep, mean, model, label, a, b)
                lines.append(f"- {model}, {label}, {name}: {status}. {detail}.")
    lines.append("")
    return lines


def check_fwd_rest(data):
    lines = ["### 4b. fwd_rest, train and nostep, GA=4 (mean of the two repeats, ms)", "",
             "fwd_rest is the forward time of microbatches 1..3 on rank 0's max. Delta = (b - a) / a. "
             "Threshold = larger repeat difference of the two arms for the same row.", ""]
    for model, _ in MODELS:
        lines += [f"#### {model}", "",
                  "| T | mode | off | main-on | p1-on | main-on vs off | p1-on vs main-on | p1-on vs off |",
                  "|---|---|---|---|---|---|---|---|"]
        for t in TOKENS:
            for mode in ["train", "nostep"]:
                m = {a: arm_mean(data, model, mode, t, 4, "fwd_rest", a) for a in ARMS}

                def th(a, b):
                    return max(abs(repeat_pct(data, model, mode, t, 4, "fwd_rest", a)),
                               abs(repeat_pct(data, model, mode, t, 4, "fwd_rest", b)))

                d1, d2, d3 = pct_delta(m["off"], m["main-on"]), pct_delta(m["main-on"], m["p1-on"]), pct_delta(m["off"], m["p1-on"])
                lines.append(f"| {t} | {mode} | {fmt(m['off'])} | {fmt(m['main-on'])} | {fmt(m['p1-on'])} | "
                             f"{cell(d1, th('off', 'main-on'))} | {cell(d2, th('main-on', 'p1-on'))} | "
                             f"{cell(d3, th('off', 'p1-on'))} |")
        lines.append("")
    return lines


def check_iqr(data):
    lines = ["## 5. Per-step variance on rank 0", "",
             "train GA=4 step time from raw_rank0.step (8 measured steps). Cell: median / IQR, ms.", ""]
    for model, _ in MODELS:
        lines += [f"### {model}", "",
                  "| run | arm | " + " | ".join(f"T={t}" for t in TOKENS) + " |",
                  "|---|---|" + "---|" * len(TOKENS)]
        for run in RUN_NAMES:
            cells = []
            for t in TOKENS:
                steps = data[run][model]["rows"][("train", t, 4)]["raw_rank0"]["step"]
                cells.append(f"{statistics.median(steps):.1f} / {iqr(steps):.2f}")
            lines.append(f"| {run} | {RUN_ARM[run]} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    data = load_all(args.root)
    rep, mean = build_stats(data)

    out = header()
    out += check_arms(args.root, data)
    out += check_losses(data)
    out += check_memory(data)
    out += check_logs_and_e(args.root, data)
    out += check_drift(data, rep)
    out += check_main(rep, mean)
    out += check_attribution(rep, mean)
    out += check_fwd_rest(data)
    out += check_iqr(data)

    text = "\n".join(out) + "\n"
    print(text, end="")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
