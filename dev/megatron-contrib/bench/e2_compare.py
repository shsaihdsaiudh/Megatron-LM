#!/usr/bin/env python3
"""E2 P1: compare the overlap_outer_communication "off" and "on" runs of e1_exposure.py.

Reads <root>/{off,on}/e1_d{d}_L{l}.json and prints Markdown to stdout:

1. Correctness: per (mode, T, GA) row, raw_rank0.losses (every microbatch loss on rank 0)
   and per_rank.loss_sum (fsum of all losses, all 16 ranks) must be equal between off and on.
2. Memory: max over ranks of peak_mem_gb, off vs on.
3. Performance per model: E (exposed outer AG, ms), fwd_rest, step time for GA=1 and GA=4,
   saved step fraction (step_off - step_on) / step_off, and the nostep control E.
   Values are the JSON "max" entries (per-rank medians, then max over ranks).
4. Noise: IQR across the measured steps on rank 0 (from raw_rank0), for E and step time.
   Also the per-rank min/max of E for "on".
5. Run-to-run reproducibility: off/on vs the E1 baseline (same code path, GA=4 rows).

Sanity check: E recomputed from raw_rank0 must equal per_rank.exposed[0].

Usage:
  python3 e2_compare.py [--root DIR] [--baseline DIR] [--out FILE.md]
"""

import argparse
import json
import math
import os
import statistics

MODELS = ["e1_d4096_L32.json", "e1_d8192_L16.json"]
DEFAULT_ROOT = "/planning-users/yangyeyuan/megatron-contrib/results/e2/e2p1-2node-20261112"
DEFAULT_BASELINE = "/planning-users/yangyeyuan/megatron-contrib/results/e1/e1-2node-20261111"


def load(path):
    with open(path) as f:
        return json.load(f)


def index(doc):
    return {(r["mode"], r["tokens"], r["ga"]): r for r in doc["results"]}


def iqr(xs):
    q = statistics.quantiles(xs, n=4, method="inclusive")
    return q[2] - q[0]


def per_step_exposed(row):
    """Rank-0 E per measured step: fwd(mb0) - median(fwd(mb1..))."""
    return [f[0] - statistics.median(f[1:]) for f in row["raw_rank0"]["fwd"]]


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def pct(a, b):
    return "-" if a is None or b is None or a == 0 else f"{100 * (a - b) / a:+.1f}%"


def correctness(off, on):
    lines = ["## 1. Correctness (bit-exact losses)", ""]
    total = matched = 0
    bad = []
    for key in sorted(off):
        if key not in on:
            continue
        a, b = off[key], on[key]
        total += 1
        la, lb = a["raw_rank0"]["losses"], b["raw_rank0"]["losses"]
        sa, sb = a["per_rank"]["loss_sum"], b["per_rank"]["loss_sum"]
        loss_eq = la == lb
        sum_eq = sa == sb
        if loss_eq and sum_eq:
            matched += 1
        else:
            d_loss = max(abs(x - y) for x, y in zip(la, lb)) if len(la) == len(lb) else float("nan")
            d_sum = max(abs(x - y) for x, y in zip(sa, sb))
            bad.append((key, len(la) == len(lb), loss_eq, sum_eq, d_loss, d_sum))
    lines.append(f"Rows matched (raw_rank0.losses and all 16 loss_sum equal): {matched}/{total}")
    if bad:
        lines += ["", "| mode | T | GA | same length | raw losses equal | loss_sum equal | max abs diff raw | max abs diff loss_sum |",
                  "|---|---|---|---|---|---|---|---|"]
        for key, same_len, le, se, dl, ds in bad:
            lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {same_len} | {le} | {se} | {dl:.3g} | {ds:.3g} |")
    else:
        lines.append("No mismatches.")
    return lines, matched, total


def memory(off, on):
    lines = ["## 2. Memory (peak_mem_gb, max over ranks)", ""]
    diffs = []
    for key in sorted(off):
        if key in on:
            diffs.append((abs(on[key]["max"]["peak_mem_gb"] - off[key]["max"]["peak_mem_gb"]), key))
    worst = max(diffs)
    exact = sum(1 for d, _ in diffs if d == 0.0)
    lines.append(f"Rows compared: {len(diffs)}. Exact equal: {exact}. Max abs diff: {worst[0]:.4f} GB at {worst[1]}.")
    return lines, worst[0]


def perf_table(off, on, nd_model):
    lines = [f"### {nd_model}", "",
             "| T | E off | E on | ΔE | on/off | fwd_mb0 off / on | fwd_rest off / on | step GA=1 off / on | step GA=4 off / on | saved step GA=1 / GA=4 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for t in sorted({k[1] for k in off}):
        a4, b4 = off[("train", t, 4)], on[("train", t, 4)]
        a1, b1 = off[("train", t, 1)], on[("train", t, 1)]
        ea, eb = a4["max"]["exposed"], b4["max"]["exposed"]
        sav1 = pct(a1["max"]["step"], b1["max"]["step"])
        sav4 = pct(a4["max"]["step"], b4["max"]["step"])
        lines.append(
            f"| {t} | {fmt(ea)} | {fmt(eb)} | {eb - ea:+.1f} | {eb / ea:.2f} | "
            f"{fmt(a4['max']['fwd_mb0'])} / {fmt(b4['max']['fwd_mb0'])} | "
            f"{fmt(a4['max']['fwd_rest'])} / {fmt(b4['max']['fwd_rest'])} | "
            f"{fmt(a1['max']['step'])} / {fmt(b1['max']['step'])} | "
            f"{fmt(a4['max']['step'])} / {fmt(b4['max']['step'])} | "
            f"{sav1} / {sav4} |"
        )
    return lines


def nostep_table(off, on):
    """Control: no optimizer step, so no outer AG in measured steps. Should match off/on."""
    lines = ["| T | E off / on | fwd_mb0 off / on | fwd_rest off / on | step off / on | step Δ |",
             "|---|---|---|---|---|---|"]
    for t in sorted({k[1] for k in off}):
        a, b = off[("nostep", t, 4)]["max"], on[("nostep", t, 4)]["max"]
        lines.append(
            f"| {t} | {fmt(a['exposed'])} / {fmt(b['exposed'])} | {fmt(a['fwd_mb0'])} / {fmt(b['fwd_mb0'])} | "
            f"{fmt(a['fwd_rest'])} / {fmt(b['fwd_rest'])} | {fmt(a['step'])} / {fmt(b['step'])} | {pct(b['step'], a['step'])} |"
        )
    return lines


def spread_and_noise(off, on):
    lines = ["Noise (rank 0, IQR over the 8 measured steps, ms) and per-rank spread of E (max over ranks; min/max over 16 ranks):", "",
             "| T | E IQR off | E IQR on | E on min..max over ranks | E off min..max over ranks | step GA=4 IQR off | step GA=4 IQR on | step GA=1 IQR off | step GA=1 IQR on |",
             "|---|---|---|---|---|---|---|---|---|"]
    for t in sorted({k[1] for k in off}):
        a4, b4 = off[("train", t, 4)], on[("train", t, 4)]
        a1, b1 = off[("train", t, 1)], on[("train", t, 1)]
        ea, eb = per_step_exposed(a4), per_step_exposed(b4)
        pr_b = b4["per_rank"]["exposed"]
        pr_a = a4["per_rank"]["exposed"]
        lines.append(
            f"| {t} | {iqr(ea):.2f} | {iqr(eb):.2f} | {min(pr_b):.1f}..{max(pr_b):.1f} | {min(pr_a):.1f}..{max(pr_a):.1f} | "
            f"{iqr(a4['raw_rank0']['step']):.1f} | {iqr(b4['raw_rank0']['step']):.1f} | "
            f"{iqr(a1['raw_rank0']['step']):.1f} | {iqr(b1['raw_rank0']['step']):.1f} |"
        )
    return lines


def sanity(off):
    """E from raw rank-0 data must match per_rank.exposed[0] of the JSON."""
    worst = 0.0
    n = 0
    for key, row in off.items():
        if key[2] > 1:
            n += 1
            worst = max(worst, abs(statistics.median(per_step_exposed(row)) - row["per_rank"]["exposed"][0]))
    return n, worst


def baseline_table(off_doc, e1_doc):
    lines = ["Reproducibility: E2 off (same code path as main) vs the E1 baseline, GA=4 rows.", "",
             "| mode | T | E E1 | E E2-off | fwd_rest E1 / E2-off | step E1 / E2-off | step diff % |",
             "|---|---|---|---|---|---|---|"]
    e1 = index(e1_doc)
    o = index(off_doc)
    for key in sorted(o):
        if key[2] != 4 or key not in e1:
            continue
        a, b = e1[key], o[key]
        lines.append(
            f"| {key[0]} | {key[1]} | {fmt(a['max']['exposed'])} | {fmt(b['max']['exposed'])} | "
            f"{fmt(a['max']['fwd_rest'])} / {fmt(b['max']['fwd_rest'])} | "
            f"{fmt(a['max']['step'])} / {fmt(b['max']['step'])} | {pct(a['max']['step'], b['max']['step'])} |"
        )
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--baseline", default=DEFAULT_BASELINE)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out = ["# E2 P1 off/on comparison", ""]
    all_off, all_on = {}, {}
    for model in MODELS:
        off_doc = load(os.path.join(args.root, "off", model))
        on_doc = load(os.path.join(args.root, "on", model))
        all_off[model], all_on[model] = index(off_doc), index(on_doc)
        meta = off_doc["meta"]
        assert meta["overlap_outer_communication"] is False
        assert on_doc["meta"]["overlap_outer_communication"] is True
        assert off_doc["meta"]["world"] == on_doc["meta"]["world"] == 16

    corr_lines, matched, total = [], 0, 0
    for model in MODELS:
        lines, m, t = correctness(all_off[model], all_on[model])
        corr_lines += [f"{model}:"] + lines + [""]
        matched += m
        total += t
    out += corr_lines

    out += ["## 2. Memory (peak_mem_gb, max over ranks)", ""]
    for model in MODELS:
        lines, worst = memory(all_off[model], all_on[model])
        out += [f"{model}: " + lines[-1]]
    out += [""]

    out += ["## 3. Performance (ms; E, fwd_rest, step are JSON \"max\" values)", ""]
    for model in MODELS:
        out += perf_table(all_off[model], all_on[model], model) + [""]

    out += ["## 3b. Control: nostep (no optimizer step, no outer AG in measured steps)", ""]
    for model in MODELS:
        out += [f"### {model}", ""] + nostep_table(all_off[model], all_on[model]) + [""]

    out += ["## 4. Noise and per-rank spread", ""]
    for model in MODELS:
        out += [f"### {model}", ""] + spread_and_noise(all_off[model], all_on[model]) + [""]

    out += ["## 5. Reproducibility against E1 baseline", ""]
    for model in MODELS:
        e1_path = os.path.join(args.baseline, model)
        if os.path.exists(e1_path):
            out += [f"### {model}", ""] + baseline_table(load(os.path.join(args.root, "off", model)), load(e1_path)) + [""]

    out += ["## 6. Sanity", ""]
    for model in MODELS:
        n, worst = sanity(all_off[model])
        out.append(f"{model}: recomputed rank-0 E vs per_rank.exposed[0], {n} rows, max abs diff {worst:.2e} ms")
    text = "\n".join(out) + "\n"
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
