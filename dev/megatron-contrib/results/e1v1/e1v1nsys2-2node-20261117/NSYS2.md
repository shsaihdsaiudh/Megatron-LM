# E1v1-nsys2: v1-base and v1-5985 traces (job t-20261117144505-xfrh1)

T=4096, GA=4, mesh (2,8), nsys on all 16 ranks. Code: snapshots/main-d2b0e35cc (unpatched v1).
Reference arm: v2-off from results/e2/e2skew-2node-20261116/01-v2-off (symlink 00-v2-off).

Reports (run with E2SKEW_ROOT=<this dir> E2SKEW_ARM_DIRS="00-v2-off <arm>"):
- RACE2-01-v1-base.md, RACE2-02-v1-5985.md (bench/e2race2_compare.py)
- SKEW-01-v1-base.md, SKEW-02-v1-5985.md (bench/e2skew_compare.py)

In these reports the label "off" is v2-off and "p1" is the v1 arm. The intro paragraph, the
"Findings" section and the "RACE.md busy-at-E" column are static text from the v2 run and do not
describe this data. Use the computed tables only.

Streams: compute 7 in all arms. Inner AG-I: v2 stream 39, v1 stream 43 (ProcessGroupNCCL internal),
both ncclDevKernel_AllGather_RING_LL gridX 24.

| metric (mb1..3 fwd) | v2-off | v2-p1 | v1-base | v1-5985 |
|---|---|---|---|---|
| fwd_rest (ms, sweep JSON max) | 268.8 | 295.8 | 297.5 | 297.2 |
| AG wins / X-gated cases | 0 / 2880 | 1457 / 2880 | 1551 / 2880 | 1497 / 2880 |
| gapN median, AG wins (us) | n/a | 12.1 | 16.0 | 13.1 |
| gapN median, GEMM wins (us) | 3.0 (all) | 4.0 | 4.0 | 3.9 |
| gapG median, AG wins (us) | n/a | 16.5 | 20.1 | 18.0 |
| AG-I start skew median, node 0 (ms) | 0.40 | 5.35 | 5.30 | 5.24 |
| slow GEMMs (>4.5 ms) per mb | 0.09 | 23.80 | 24.10 | 24.03 |
| mean duration of slow GEMMs (ms) | 4.70 | 5.27 | 5.30 | 5.30 |
| other ops in [R, min(A, G.start)] | 0 | 0 | 0 | 0 |
| wait between X and N complete before R | 2880/2880 | 2880/2880 | 2880/2880 | 2880/2880 |

v2-off and v2-p1 columns come from results/e2/e2skew-2node-20261116/{RACE2.md,SKEW.md}.
