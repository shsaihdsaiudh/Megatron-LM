"""Summarize an E4 results directory (#6714 / #9539 real-model A/B).

Parses Megatron's per-iteration log lines (printed by the last rank) from every
node<k>.<NN>-<arm>-ga<G>.log and reports, per run and per (arm, GA):
median iteration time over iterations > --skip, TFLOP/s/GPU, and the lm-loss curve.

Usage: python e4_summary.py <results dir> [--skip 5]
"""

import argparse
import collections
import glob
import os
import re
import statistics

LINE = re.compile(r"iteration\s+(\d+)/\s*\d+ \|.*?elapsed time per iteration \(ms\): ([\d.]+)")
TFLOPS = re.compile(r"throughput per GPU \(TFLOP/s/GPU\): ([\d.]+)")
LOSS = re.compile(r"lm loss: ([\d.E+-]+)")
MEM = re.compile(r"max allocated: ([\d.]+)")


def parse(path):
    iters = {}
    for line in open(path, errors="replace"):
        m = LINE.search(line)
        if not m:
            continue
        it, ms = int(m.group(1)), float(m.group(2))
        t = TFLOPS.search(line)
        loss = LOSS.search(line)
        iters[it] = (ms, float(t.group(1)) if t else float("nan"), loss.group(1) if loss else None)
    return iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    p.add_argument("--skip", type=int, default=5, help="ignore iterations <= skip")
    args = p.parse_args()

    runs = collections.defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(args.out_dir, "node*.*.log"))):
        name = os.path.basename(path).split(".", 1)[1][: -len(".log")]
        if "nsys" in name:
            continue
        it = parse(path)
        if it:
            runs[name].update(it)

    print(f"Iteration time: median over iterations > {args.skip}, ms. TFLOP/s/GPU likewise.")
    print(f"{'run':22s} {'iters':>5s} {'iter_ms':>9s} {'min':>9s} {'max':>9s} {'TFLOPs':>7s}")
    by_arm = collections.defaultdict(list)
    losses = collections.defaultdict(dict)
    for name in sorted(runs):
        it = runs[name]
        kept = [v for k, v in sorted(it.items()) if k > args.skip]
        if not kept:
            print(f"{name:22s} no iterations after skip")
            continue
        ms = [v[0] for v in kept]
        med = statistics.median(ms)
        tf = statistics.median(v[1] for v in kept)
        print(f"{name:22s} {len(it):5d} {med:9.1f} {min(ms):9.1f} {max(ms):9.1f} {tf:7.1f}")
        # "<NN>-<arm>-ga<G>", where <arm> may itself contain "-" (main-on).
        arm, ga = name.split("-", 1)[1].rsplit("-", 1)
        by_arm[(ga, arm)].append(med)
        losses[ga][name] = tuple(v[2] for _, v in sorted(it.items()))
    print()
    print("Mean of run medians per arm:")
    base = {}
    for (ga, arm), meds in sorted(by_arm.items()):
        mean = sum(meds) / len(meds)
        if arm == "off":
            base[ga] = mean
    for (ga, arm), meds in sorted(by_arm.items()):
        mean = sum(meds) / len(meds)
        rel = f"{(mean / base[ga] - 1) * 100:+6.2f}%" if ga in base else ""
        print(f"  {ga:5s} {arm:8s} n={len(meds)} {mean:9.1f} ms  vs off {rel}  runs: "
              + " ".join(f"{m:.1f}" for m in meds))
    print()
    for ga, d in sorted(losses.items()):
        same = len(set(d.values())) == 1
        print(f"lm loss identical across {ga} runs (as printed): {same}")


if __name__ == "__main__":
    main()
