#!/usr/bin/env python3
"""E1 v1 vs v2 summary: 10 runs in 5 arms (2 repeats each) for Megatron-FSDP, 2 nodes x 8 GPUs (16 ranks).

Runs (time order) and arms:
  01-v2-off    v2, overlap_outer_communication=False, DP-outer all-gather on the DP-inner all-gather stream (arm v2-off)
  02-v2-p1     v2, overlap_outer_communication=True, outer all-gather on its own stream (arm v2-p1)
  03-v1-base   v1, outer all-gather on its own stream, inner all-gather waits for it (arm v1-base)
  04-v1-5985   v1, hfsdp_param_gather_overlap=True, outer on its own stream, prefetched (arm v1-5985)
  05-v1-same   v1 patched, outer all-gather on the inner all-gather stream (arm v1-same)
  06-v1-same   v1-same
  07-v1-5985   v1-5985
  08-v1-base   v1-base
  09-v2-p1     v2-p1
  10-v2-off    v2-off

Baselines: v1-same for v1 arms, v2-off for v2 arms.

Each run dir holds e1_d4096_L32.json, node0.sweep_d4096_L32.log and node{0,1}.sysinfo.txt.
Rows per run: mode train or nostep, tokens 256, 1024, 4096, GA=4. Metrics are read from the "max" dict of each row.

Sections printed as Markdown:
  Findings: fwd_rest deltas for the two v1 arms vs v1-same and for v2-p1 vs v2-off, at T=1024 and T=4096.
  1. Checks: meta flags, sweep header flags, snapshot path per node, loss bit-identity, peak memory.
  2. Results: Table A (v1 arms), Table B (v2 arms), Table C (all 5 arms, fwd_rest and step) per (mode, T) row.
  3. Anomalies: missing files, failed runs, flag mismatches, loss mismatches within a version.

Cell: mean of the two repeats (ms). Delta is (arm - baseline) / baseline in percent, positive = slower.
rep = (second repeat - first repeat) / first repeat in percent, for the same arm and metric.
A "~" marks a delta whose absolute value is below the larger |rep| of the two arms compared.

Usage:
  python3 e1v1_compare.py [--root DIR] [--out FILE.md]
  --out defaults to DIR/SUMMARY.md.
"""

import argparse
import json
import os
import re
import statistics

DEFAULT_ROOT = "/planning-users/yangyeyuan/megatron-contrib/results/e1v1/e1v1-2node-20261115"
MODEL = "e1_d4096_L32.json"
SWEEP = "node0.sweep_d4096_L32.log"
SYSINFO = "node{}.sysinfo.txt"
NODES = (0, 1)
EXPECTED_ROWS_PER_RUN = 6

RUNS = [
    ("01-v2-off", "v2-off"),
    ("02-v2-p1", "v2-p1"),
    ("03-v1-base", "v1-base"),
    ("04-v1-5985", "v1-5985"),
    ("05-v1-same", "v1-same"),
    ("06-v1-same", "v1-same"),
    ("07-v1-5985", "v1-5985"),
    ("08-v1-base", "v1-base"),
    ("09-v2-p1", "v2-p1"),
    ("10-v2-off", "v2-off"),
]
RUN_NAMES = [r for r, _ in RUNS]
RUN_ARM = dict(RUNS)
V1_ARMS = ["v1-base", "v1-5985", "v1-same"]
V2_ARMS = ["v2-off", "v2-p1"]
ARMS = V1_ARMS + V2_ARMS
REPEATS = {arm: [r for r, a in RUNS if a == arm] for arm in ARMS}  # time order
BASE = {"v1": "v1-same", "v2": "v2-off"}
VERSION = {arm: ("v1" if arm in V1_ARMS else "v2") for arm in ARMS}
ABSENT = "<absent>"

# Expected meta values per arm. ABSENT means the key must not be in meta.
EXPECT_META = {
    "v1-base": {"version": "v1", "outer_ag_on_ag_stream": False, "hfsdp_param_gather_overlap": False},
    "v1-5985": {"version": "v1", "outer_ag_on_ag_stream": False, "hfsdp_param_gather_overlap": True},
    "v1-same": {"version": "v1", "outer_ag_on_ag_stream": True, "hfsdp_param_gather_overlap": False},
    "v2-off": {"version": ABSENT, "overlap_outer_communication": False},
    "v2-p1": {"version": ABSENT, "overlap_outer_communication": True},
}
# Sweep header key -> meta key, per version. The header line is printed by the run script.
SWEEP_TO_META = {
    "v1": {"hfsdp_param_gather_overlap": "hfsdp_param_gather_overlap", "outer_ag_is_ag_stream": "outer_ag_on_ag_stream"},
    "v2": {"overlap_outer_communication": "overlap_outer_communication"},
}
# (JSON max field, column label)
METRICS = [
    ("fwd_mb0", "fwd_mb0"),
    ("fwd_rest", "fwd_rest"),
    ("bwd_mb0", "bwd_mb0"),
    ("bwd_rest", "bwd_rest"),
    ("exposed", "exposed"),
    ("step", "step"),
]
MODE_ORDER = {"train": 0, "nostep": 1}


def load_runs(root):
    data = {}
    for run in RUN_NAMES:
        path = os.path.join(root, run, MODEL)
        if not os.path.isfile(path):
            raise SystemExit(f"missing JSON, cannot continue: {path}")
        with open(path) as f:
            doc = json.load(f)
        data[run] = {
            "meta": doc["meta"],
            "rows": {(r["mode"], r["tokens"], r["ga"]): r for r in doc["results"]},
        }
    return data


def build_rows(data, anomalies):
    union = set().union(*(set(data[r]["rows"]) for r in RUN_NAMES))
    common = set.intersection(*(set(data[r]["rows"]) for r in RUN_NAMES))
    for run in RUN_NAMES:
        missing = sorted(union - set(data[run]["rows"]))
        if missing:
            anomalies.append(f"{run}: missing rows {missing}")
        if len(data[run]["rows"]) != EXPECTED_ROWS_PER_RUN:
            anomalies.append(f"{run}: {len(data[run]['rows'])} rows, expected {EXPECTED_ROWS_PER_RUN}")
    return sorted(common, key=lambda k: (MODE_ORDER.get(k[0], 9), k[1], k[2]))


def get_val(data, run, key, name):
    return float(data[run]["rows"][key]["max"][name])


def pct(a, b):
    """Change from a to b in percent. Positive means b is larger."""
    return 100.0 * (b - a) / a


def row_label(key):
    return f"{key[0]} T={key[1]} GA={key[2]}"


def sysinfo_path(root, run, node):
    return os.path.join(root, run, SYSINFO.format(node))


def snapshot_path(path):
    """Return (full path from the megatron_fsdp line, None) or (None, reason)."""
    if not os.path.isfile(path):
        return None, "file missing"
    with open(path) as f:
        hits = [line.split(None, 1)[1].strip() for line in f if line.startswith("megatron_fsdp ")]
    if len(hits) != 1:
        return None, f"{len(hits)} megatron_fsdp lines"
    return hits[0], None


def sweep_header(path):
    """Return the dict of key=value fields on the [e1...] world= header line, or None."""
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        for line in f:
            if re.search(r"\[e1[^\]]*\] world=", line):
                return dict(re.findall(r"(\w+)=(\S+)", line))
    return None


def sweep_wrote(path):
    if not os.path.isfile(path):
        return False
    with open(path) as f:
        return any("] wrote " in line for line in f)


def meta_value(meta, key):
    return meta[key] if key in meta else ABSENT


def check_meta(data, root, anomalies):
    lines = ["### 1a. Meta flags and sweep header per run", "",
             "| run | arm | meta (expected) | meta check | sweep header vs meta | status |",
             "|---|---|---|---|---|---|"]
    for run in RUN_NAMES:
        arm = RUN_ARM[run]
        meta = data[run]["meta"]
        expect = EXPECT_META[arm]
        got = {k: meta_value(meta, k) for k in expect}
        bad = [k for k in expect if got[k] != expect[k]]
        meta_txt = ", ".join(f"{k}={got[k]}" for k in expect)
        if bad:
            anomalies.append(f"{run} ({arm}): meta mismatch on {', '.join(bad)} "
                             f"(got {', '.join(str(got[k]) for k in bad)}; expected "
                             f"{', '.join(str(expect[k]) for k in bad)})")
        header = sweep_header(os.path.join(root, run, SWEEP))
        if header is None:
            hdr_txt = "no header line"
            anomalies.append(f"{run}: no [e1...] world= header line in {SWEEP}")
            hdr_ok = False
        else:
            pairs = []
            hdr_ok = True
            for hkey, mkey in SWEEP_TO_META[VERSION[arm]].items():
                hval = header.get(hkey)
                mval = meta_value(meta, mkey)
                pairs.append(f"{hkey}={hval}")
                if hval != str(mval):
                    hdr_ok = False
                    anomalies.append(f"{run}: sweep header {hkey}={hval} but meta {mkey}={mval}")
            hdr_txt = ", ".join(pairs)
        status = "PASS" if not bad and hdr_ok else "FAIL"
        lines.append(f"| {run} | {arm} | {meta_txt} | {'ok' if not bad else 'MISMATCH: ' + ', '.join(bad)} | "
                     f"{hdr_txt} | {status} |")
    lines.append("")
    return lines


def check_snapshots(root, anomalies):
    lines = ["### 1b. megatron_fsdp snapshot path per run (sysinfo)", "",
             "| run | arm | node0 | node1 |", "|---|---|---|---|"]
    for run in RUN_NAMES:
        cells = []
        paths = {}
        for node in NODES:
            path, reason = snapshot_path(sysinfo_path(root, run, node))
            if reason:
                anomalies.append(f"{run} node{node}: sysinfo {reason}")
                cells.append(f"<{reason}>")
            else:
                paths[node] = path
                cells.append(f"`{path}`")
        if len(paths) == 2 and paths[0] != paths[1]:
            anomalies.append(f"{run}: node0 and node1 snapshot paths differ")
        lines.append(f"| {run} | {RUN_ARM[run]} | {cells[0]} | {cells[1]} |")
    lines.append("")
    return lines


def check_run_files(root, anomalies):
    for run in RUN_NAMES:
        if not os.path.isfile(os.path.join(root, run, SWEEP)):
            anomalies.append(f"{run}: sweep log missing")
        elif not sweep_wrote(os.path.join(root, run, SWEEP)):
            anomalies.append(f"{run}: sweep log has no '] wrote' line (possible failed run)")
        for node in NODES:
            if not os.path.isfile(sysinfo_path(root, run, node)):
                anomalies.append(f"{run}: node{node}.sysinfo.txt missing")


def check_finite(data, rows, anomalies):
    for run in RUN_NAMES:
        for key in rows:
            if get_val(data, run, key, "finite") != 1.0:
                anomalies.append(f"{run} {row_label(key)}: finite={get_val(data, run, key, 'finite')}")


def check_losses(data, rows, anomalies):
    groups = {"v1": [r for r in RUN_NAMES if VERSION[RUN_ARM[r]] == "v1"],
              "v2": [r for r in RUN_NAMES if VERSION[RUN_ARM[r]] == "v2"]}
    lines = ["### 1c. loss_sum (max.loss_sum)", "",
             "Bit-identity is exact float equality of the JSON values, pooled over the runs of one version "
             "(v1: 03, 04, 05, 06, 07, 08; v2: 01, 02, 09, 10). The cross-version columns are informational.", "",
             "| mode | T | GA | v1 identical | v2 identical | v1 loss_sum (first run) | v2 loss_sum (first run) | "
             "v1 == v2 | abs diff |",
             "|---|---|---|---|---|---|---|---|---|"]
    for key in rows:
        ref = {}
        ident = {}
        for ver, runs in groups.items():
            vals = {r: get_val(data, r, key, "loss_sum") for r in runs}
            first = vals[runs[0]]
            ref[ver] = first
            diff = [f"{r}={vals[r]!r}" for r in runs if vals[r] != first]
            ident[ver] = "yes" if not diff else f"NO ({len(diff)} differ)"
            if diff:
                anomalies.append(f"{row_label(key)}: v{ver[1]} loss_sum not bit-identical; first={first!r}; "
                                 f"differs: {', '.join(diff)}")
        same = ref["v1"] == ref["v2"]
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {ident['v1']} | {ident['v2']} | {ref['v1']!r} | "
                     f"{ref['v2']!r} | {'yes' if same else 'no'} | {abs(ref['v1'] - ref['v2']):.3g} |")
    lines.append("")
    return lines


def check_memory(data, rows):
    lines = ["### 1d. Peak memory per arm (max.peak_mem_gb, GB)", "",
             "Cell: repeat 1 / repeat 2 in time order. Spread is max - min over all 10 runs.", "",
             "| mode | T | " + " | ".join(ARMS) + " | spread | verdict |",
             "|---|---|" + "---|" * len(ARMS) + "---|---|"]
    for key in rows:
        cells = []
        for arm in ARMS:
            r1, r2 = REPEATS[arm]
            cells.append(f"{get_val(data, r1, key, 'peak_mem_gb'):.4f} / {get_val(data, r2, key, 'peak_mem_gb'):.4f}")
        allv = [get_val(data, r, key, "peak_mem_gb") for r in RUN_NAMES]
        spread = max(allv) - min(allv)
        verdict = "equal in all 10 runs" if spread == 0.0 else "differs"
        lines.append(f"| {key[0]} | {key[1]} | " + " | ".join(cells) +
                     f" | {spread:.4f} | {verdict} |")
    lines.append("")
    return lines


def compute_stats(data, rows):
    """(row, metric, arm) -> (mean over repeats, rep %)."""
    stats = {}
    for key in rows:
        for name, _ in METRICS:
            for arm in ARMS:
                vals = [get_val(data, r, key, name) for r in REPEATS[arm]]
                stats[(key, name, arm)] = (statistics.mean(vals), pct(vals[0], vals[1]))
    return stats


def cell(stats, key, name, arm, base):
    mean, rep = stats[(key, name, arm)]
    if arm == base:
        return f"{mean:.2f} (base, rep {rep:+.2f}%)"
    bmean, brep = stats[(key, name, base)]
    d = pct(bmean, mean)
    mark = " ~" if abs(d) < max(abs(rep), abs(brep)) else ""
    return f"{mean:.2f} (Δ {d:+.2f}%{mark}, rep {rep:+.2f}%)"


def pair_delta(stats, key, name, arm, base):
    """(delta %, threshold %, marked) for arm vs base."""
    mean, rep = stats[(key, name, arm)]
    bmean, brep = stats[(key, name, base)]
    d = pct(bmean, mean)
    th = max(abs(rep), abs(brep))
    return d, th, abs(d) < th


def findings(stats, rows):
    lines = []
    for key in rows:
        if key[1] not in (1024, 4096):
            continue
        parts = []
        for arm, base in [("v1-base", "v1-same"), ("v1-5985", "v1-same"), ("v2-p1", "v2-off")]:
            d, th, marked = pair_delta(stats, key, "fwd_rest", arm, base)
            parts.append(f"{arm} vs {base} {d:+.2f}%{' ~' if marked else ''} (threshold {th:.2f}%)")
        lines.append(f"- {key[0]}, T={key[1]}: " + "; ".join(parts) + ".")
    return lines


def table_arms(stats, key, arms, base, data, title):
    lines = [f"#### {title}", "",
             "| arm (runs) | " + " | ".join(lbl for _, lbl in METRICS) + " |",
             "|---|" + "---|" * len(METRICS)]
    for arm in arms:
        runs = ", ".join(r.split("-")[0] for r in REPEATS[arm])
        cells = [cell(stats, key, name, arm, base) for name, _ in METRICS]
        lines.append(f"| {arm} ({runs}) | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def table_cross(stats, key):
    lines = ["#### Table C: all 5 arms, mean fwd_rest and step (ms, informational)", "",
             "| metric | " + " | ".join(ARMS) + " |",
             "|---|" + "---|" * len(ARMS)]
    for name in ("fwd_rest", "step"):
        vals = [f"{stats[(key, name, arm)][0]:.2f}" for arm in ARMS]
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    lines.append("")
    return lines


def results_section(data, rows, stats):
    lines = ["## 2. Results (ms)", ""]
    for key in rows:
        lines += [f"### {row_label(key)}", ""]
        lines += table_arms(stats, key, ["v1-base", "v1-5985", "v1-same"], BASE["v1"], data,
                            "Table A: v1 arms (baseline v1-same)")
        lines += table_arms(stats, key, ["v2-off", "v2-p1"], BASE["v2"], data,
                            "Table B: v2 arms (baseline v2-off)")
        lines += table_cross(stats, key)
    return lines


def noise_section(stats, rows):
    lines = ["### Repeat noise floor (|rep| %, all rows, all metrics)", "",
             "| arm | median | max |", "|---|---|---|"]
    for arm in ARMS:
        vals = [abs(stats[(k, n, arm)][1]) for k in rows for n, _ in METRICS]
        lines.append(f"| {arm} | {statistics.median(vals):.2f} | {max(vals):.2f} |")
    lines.append("")
    return lines


def header_lines():
    return [
        "# E1 v1 vs v2: 10 runs, 5 arms, d4096 L32, 2 nodes x 8 GPUs (16 ranks), GA=4",
        "",
        "Arms: v1-base (v1, outer all-gather on own stream, inner waits); v1-5985 (v1, "
        "hfsdp_param_gather_overlap=True, outer on own stream, prefetched); v1-same (v1 patched, outer "
        "all-gather on the inner stream, baseline for v1); v2-off (v2, outer on the inner stream, baseline for "
        "v2); v2-p1 (v2, overlap_outer_communication=True, outer on own stream). Each arm has two runs.",
        "",
        "Metrics are the JSON `max` values in ms (per-rank value, max over 16 ranks): fwd_mb0, fwd_rest "
        "(microbatches 1..3 forward), bwd_mb0, bwd_rest, exposed, step.",
        "",
    ]


def findings_section(stats, rows):
    lines = ["## Findings", "",
             "fwd_rest, mean of two repeats, delta % vs the version baseline (v1-same for the v1 arms, v2-off for "
             "v2-p1). \"~\" means |delta| is below the threshold, the larger |rep| of the two arms compared. "
             "Earlier jobs showed about +8% to +10% at T=4096 (quoted from the request, not re-measured here).",
             ""]
    lines += findings(stats, rows)
    lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out_path = args.out or os.path.join(args.root, "SUMMARY.md")

    data = load_runs(args.root)
    anomalies = []
    rows = build_rows(data, anomalies)
    check_run_files(args.root, anomalies)
    check_finite(data, rows, anomalies)
    stats = compute_stats(data, rows)

    out = header_lines()
    out += findings_section(stats, rows)
    out += ["## 1. Checks", ""]
    out += check_meta(data, args.root, anomalies)
    out += check_snapshots(args.root, anomalies)
    out += check_losses(data, rows, anomalies)
    out += check_memory(data, rows)
    out += results_section(data, rows, stats)
    out += noise_section(stats, rows)
    out += ["## 3. Anomalies", ""]
    out += [f"- {a}" for a in anomalies] if anomalies else ["None."]
    out.append("")

    text = "\n".join(out) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(text)
    print(f"wrote {out_path} ({len(anomalies)} anomalies)")


if __name__ == "__main__":
    main()
