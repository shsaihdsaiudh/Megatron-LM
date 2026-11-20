# E2 race 2: compute stream after the gating kernel, at the start of the next inner all-gather

Script: `bench/e2race2_compare.py`. It reuses the loader, AG-I detection, phase membership and the relay-chain gate resolution of `bench/e2race_compare.py` and `bench/e2skew_compare.py`. Inputs: the nsys sqlite exports of `d4096_L32_T4096_ga4.rank{0..15}` for `01-v2-off` (fast) and `02-v2-p1` (slow). Cases: AG-I k+1 in mb1..3 forward (same phase instance as AG-I k) whose gate is a kernel X on the compute stream. Measurements are within one rank.

Definitions: R = end of X. N = next compute-stream op after X. G = first GEMM on the compute stream after X. A = start of AG-I k+1. Winner AG if A < G.start, else GEMM. Times are microseconds from R.

## 1. Gaps and dA by arm and winner (us)

R = end of X. gapN = N.start - R, gapG = G.start - R, dA = A - R. Winner AG if A < G.start, else GEMM. Quantiles are nearest-rank over the cases listed.

| arm | winner | metric | n | median | p10 | p90 | max |
|---|---|---|---|---|---|---|---|
| off | all | gapN | 2880 | 3.0 | 2.8 | 3.5 | 6.0 |
| off | all | gapG | 2880 | 6.3 | 5.8 | 7.2 | 9.9 |
| off | all | dA | 2880 | 3812.7 | 3801.4 | 3844.1 | 3863.4 |
| off | AG | gapN | 0 | n/a | n/a | n/a | n/a |
| off | AG | gapG | 0 | n/a | n/a | n/a | n/a |
| off | AG | dA | 0 | n/a | n/a | n/a | n/a |
| off | GEMM | gapN | 2880 | 3.0 | 2.8 | 3.5 | 6.0 |
| off | GEMM | gapG | 2880 | 6.3 | 5.8 | 7.2 | 9.9 |
| off | GEMM | dA | 2880 | 3812.7 | 3801.4 | 3844.1 | 3863.4 |
| p1 | all | gapN | 2880 | 9.6 | 3.6 | 13.1 | 16.7 |
| p1 | all | gapG | 2880 | 13.6 | 6.8 | 17.6 | 22.1 |
| p1 | all | dA | 2880 | 11.2 | 8.3 | 3838.4 | 3864.9 |
| p1 | AG | gapN | 1457 | 12.1 | 10.7 | 13.6 | 16.7 |
| p1 | AG | gapG | 1457 | 16.5 | 15.0 | 18.1 | 22.1 |
| p1 | AG | dA | 1457 | 9.2 | 7.9 | 10.4 | 12.9 |
| p1 | GEMM | gapN | 1423 | 4.0 | 3.4 | 4.7 | 6.4 |
| p1 | GEMM | gapG | 1423 | 7.7 | 6.4 | 9.1 | 12.0 |
| p1 | GEMM | dA | 1423 | 3816.0 | 3803.4 | 3845.0 | 3864.9 |

Top 3 kinds and names of N (next compute-stream op after X):

| arm | winner | rank | N kind: name | count | share |
|---|---|---|---|---|---|
| off | all | 1 | MEMSET: memset | 2880 | 1.0000 |
| off | AG | - | none | 0 | n/a |
| off | GEMM | 1 | MEMSET: memset | 2880 | 1.0000 |
| p1 | all | 1 | MEMSET: memset | 2880 | 1.0000 |
| p1 | AG | 1 | MEMSET: memset | 1457 | 1.0000 |
| p1 | GEMM | 1 | MEMSET: memset | 1423 | 1.0000 |

## 2. Waits on the compute stream between X and N, and between N and G

Window `X..N`: waits launched after X and before N (correlation id order). Window `N..G`: after N and before G. A wait is `complete` when its chain completes at or before R. For the others, the completion minus R is given. The relay stream (no kernels) is followed to the op that sets the time. `root` is the kind and stream of that op.

| arm | winner | window | cases | waits | cases with >=1 wait | complete before R | fraction complete | incomplete | incomplete: completion - R median (us) | p90 | max | unresolved | record not in trace |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | all | X..N | 2880 | 2880 | 2880 | 2880 | 1.0000 | 0 | n/a | n/a | n/a | 0 | 0 |
| off | all | N..G | 2880 | 0 | 0 | 0 | n/a | 0 | n/a | n/a | n/a | 0 | 0 |
| off | AG | X..N | 0 | 0 | 0 | 0 | n/a | 0 | n/a | n/a | n/a | 0 | 0 |
| off | AG | N..G | 0 | 0 | 0 | 0 | n/a | 0 | n/a | n/a | n/a | 0 | 0 |
| off | GEMM | X..N | 2880 | 2880 | 2880 | 2880 | 1.0000 | 0 | n/a | n/a | n/a | 0 | 0 |
| off | GEMM | N..G | 2880 | 0 | 0 | 0 | n/a | 0 | n/a | n/a | n/a | 0 | 0 |
| p1 | all | X..N | 2880 | 2880 | 2880 | 2880 | 1.0000 | 0 | n/a | n/a | n/a | 0 | 0 |
| p1 | all | N..G | 2880 | 0 | 0 | 0 | n/a | 0 | n/a | n/a | n/a | 0 | 0 |
| p1 | AG | X..N | 1457 | 1457 | 1457 | 1457 | 1.0000 | 0 | n/a | n/a | n/a | 0 | 0 |
| p1 | AG | N..G | 1457 | 0 | 0 | 0 | n/a | 0 | n/a | n/a | n/a | 0 | 0 |
| p1 | GEMM | X..N | 1423 | 1423 | 1423 | 1423 | 1.0000 | 0 | n/a | n/a | n/a | 0 | 0 |
| p1 | GEMM | N..G | 1423 | 0 | 0 | 0 | n/a | 0 | n/a | n/a | n/a | 0 | 0 |

Waits by window, recording stream and root (all cases of the arm and winner):

| arm | winner | window | record stream | root | status | waits | complete before R | fraction complete | incomplete: median (us) | p90 | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| off | GEMM | X..N | stream 39 (with ops) | kernel on stream 39 | ok | 2880 | 2880 | 1.0000 | n/a | n/a | n/a |
| off | all | X..N | stream 39 (with ops) | kernel on stream 39 | ok | 2880 | 2880 | 1.0000 | n/a | n/a | n/a |
| p1 | AG | X..N | stream 39 (with ops) | kernel on stream 39 | ok | 1457 | 1457 | 1.0000 | n/a | n/a | n/a |
| p1 | GEMM | X..N | stream 39 (with ops) | kernel on stream 39 | ok | 1423 | 1423 | 1.0000 | n/a | n/a | n/a |
| p1 | all | X..N | stream 39 (with ops) | kernel on stream 39 | ok | 2880 | 2880 | 1.0000 | n/a | n/a | n/a |

Launch of N and G relative to R (CUPTI runtime start minus R, us). Negative means queued before R.

| arm | winner | cases | N launched after R | G launched after R | N launch - R median | N p90 | N max | G launch - R median | G p90 | G max |
|---|---|---|---|---|---|---|---|---|---|---|
| off | all | 2880 | 0 | 0 | -1391505.7 | -831904.7 | -737224.9 | -1391476.2 | -831894.7 | -737214.2 |
| off | AG | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| off | GEMM | 2880 | 0 | 0 | -1391505.7 | -831904.7 | -737224.9 | -1391476.2 | -831894.7 | -737214.2 |
| p1 | all | 2880 | 0 | 0 | -1431377.4 | -883829.7 | -753064.4 | -1429812.8 | -883820.5 | -753054.0 |
| p1 | AG | 1457 | 0 | 0 | -1431387.6 | -888456.1 | -753064.4 | -1429902.6 | -888446.8 | -753054.0 |
| p1 | GEMM | 1423 | 0 | 0 | -1431347.0 | -875042.7 | -754243.6 | -1429719.5 | -875032.0 | -754232.0 |

## 3. Other GPU ops (any stream) starting in [R, min(A, G.start)]

Excluded from the window: X, N, G and AG-I k+1. Counts are over all ops in the window of each case. `cases with it` counts the cases where the (stream, name) pair occurs at least once.

| arm | winner | cases | other ops in window | per case | cases with >=1 op | distinct (stream, name) |
|---|---|---|---|---|---|---|
| off | all | 2880 | 0 | 0.00 | 0 | 0 |
| off | AG | 0 | 0 | n/a | 0 | 0 |
| off | GEMM | 2880 | 0 | 0.00 | 0 | 0 |
| p1 | all | 2880 | 0 | 0.00 | 0 | 0 |
| p1 | AG | 1457 | 0 | 0.00 | 0 | 0 |
| p1 | GEMM | 1423 | 0 | 0.00 | 0 | 0 |

No other op starts in the window in any case.

Ops on other streams that started before R and are still running at R (look-back 50 ms). Times relative to R, median over the occurrences.

| arm | winner | cases | spanning ops | cases with >=1 |
|---|---|---|---|---|
| off | all | 2880 | 0 | 0 |
| off | AG | 0 | 0 | 0 |
| off | GEMM | 2880 | 0 | 0 |
| p1 | all | 2880 | 0 | 0 |
| p1 | AG | 1457 | 0 | 0 |
| p1 | GEMM | 1423 | 0 | 0 |

No op on another stream is running at R.

## 4. AG wins in p1 against GEMM wins in p1 and all of off

| group | n | gapG median | gapG p10 | gapG p90 | gapG max | dA median | gapG minus dA median |
|---|---|---|---|---|---|---|---|
| off, all winners | 2880 | 6.3 | 5.8 | 7.2 | 9.9 | 3812.7 | -3806.2 |
| p1, AG wins | 1457 | 16.5 | 15.0 | 18.1 | 22.1 | 9.2 | 7.1 |
| p1, GEMM wins | 1423 | 7.7 | 6.4 | 9.1 | 12.0 | 3816.0 | -3808.1 |

gapG median of p1 AG wins minus p1 GEMM wins: 8.8 us. Minus off: 10.2 us.

Keys (sections 2 and 3) present in p1 AG-win cases and in no p1 GEMM-win or off case: 0.

None.

Keys present in p1 GEMM-win or off cases and absent from p1 AG-win cases: 0.

Keys present in both groups, with the share of cases having the key:

| key | p1 AG wins share | p1 GEMM wins + off share |
|---|---|---|
| wait | X..N | stream 39 (with ops) | kernel on stream 39 | ok | complete | 1.000 | 1.000 |

## 5. Per rank

n = X-gated cases of the rank. AG wins = winner AG among them. Medians over the cases of the rank.

### off

| rank | node | n | AG wins | median gapN (us) | median gapG (us) |
|---|---|---|---|---|---|
| 0 | node 0 | 180 | 0 | 3.2 | 6.6 |
| 1 | node 0 | 180 | 0 | 3.1 | 6.4 |
| 2 | node 0 | 180 | 0 | 3.2 | 6.7 |
| 3 | node 0 | 180 | 0 | 3.1 | 6.5 |
| 4 | node 0 | 180 | 0 | 2.9 | 6.1 |
| 5 | node 0 | 180 | 0 | 3.1 | 6.3 |
| 6 | node 0 | 180 | 0 | 3.0 | 6.4 |
| 7 | node 0 | 180 | 0 | 3.0 | 6.3 |
| 8 | node 1 | 180 | 0 | 3.1 | 6.5 |
| 9 | node 1 | 180 | 0 | 3.2 | 6.6 |
| 10 | node 1 | 180 | 0 | 3.0 | 6.3 |
| 11 | node 1 | 180 | 0 | 2.9 | 6.1 |
| 12 | node 1 | 180 | 0 | 2.8 | 6.0 |
| 13 | node 1 | 180 | 0 | 3.0 | 6.3 |
| 14 | node 1 | 180 | 0 | 2.9 | 6.2 |
| 15 | node 1 | 180 | 0 | 2.9 | 6.1 |

### p1

| rank | node | n | AG wins | median gapN (us) | median gapG (us) |
|---|---|---|---|---|---|
| 0 | node 0 | 180 | 93 | 10.1 | 14.3 |
| 1 | node 0 | 180 | 83 | 4.9 | 9.0 |
| 2 | node 0 | 180 | 99 | 11.6 | 15.9 |
| 3 | node 0 | 180 | 81 | 5.0 | 9.8 |
| 4 | node 0 | 180 | 100 | 10.9 | 15.7 |
| 5 | node 0 | 180 | 87 | 4.7 | 9.8 |
| 6 | node 0 | 180 | 85 | 4.7 | 9.3 |
| 7 | node 0 | 180 | 86 | 5.0 | 9.5 |
| 8 | node 1 | 180 | 103 | 10.6 | 15.3 |
| 9 | node 1 | 180 | 90 | 7.9 | 13.0 |
| 10 | node 1 | 180 | 86 | 5.2 | 9.9 |
| 11 | node 1 | 180 | 86 | 4.8 | 9.1 |
| 12 | node 1 | 180 | 96 | 9.5 | 13.4 |
| 13 | node 1 | 180 | 83 | 4.5 | 8.7 |
| 14 | node 1 | 180 | 99 | 10.0 | 13.9 |
| 15 | node 1 | 180 | 100 | 11.6 | 16.2 |

## 6. Findings

X-gated cases: 2880 per arm (AG-I k+1 gated by a compute-stream kernel X), out of 2976 pairs in mb1..3 forward; the other 96 are gated by AG-I k. In all 2880 cases of both arms, N (next compute-stream op after X) is a memset and G (first GEMM after X) is a GEMM. off: AG wins 0 of 2880; gapN median 3.0 us (p10 2.8, p90 3.5), gapG median 6.3 us (5.8, 7.2), dA median 3812.7 us. p1: AG wins 1457 of 2880 (50.6%), GEMM wins 1423. p1 AG wins: gapN median 12.1 us (10.7, 13.6), gapG median 16.5 us (15.0, 18.1), dA median 9.2 us (7.9, 10.4). p1 GEMM wins: gapN median 4.0 us (3.4, 4.7), gapG median 7.7 us (6.4, 9.1), dA median 3816.0 us. The gapG median of p1 AG wins is 8.8 us above p1 GEMM wins and 10.2 us above off. p1 AG wins per rank: 81 to 103 of 180 cases. Stream 7 has exactly one wait between X and N per case in both arms (1 of 1 in 2880 cases each), recorded by stream 39 (AG-I k) and complete before R in 2880 of 2880 cases in both arms. Between N and G there are no waits in any case. N and G were launched before R in all cases: N launch minus R median -1391505.7 us (off), -1431387.6 us (p1 AG wins), -1431347.0 us (p1 GEMM wins). No other op on any stream starts in [R, min(A, G.start)] in any case, and no op on another stream runs across R. Nothing in sections 2 and 3 appears only in p1 AG wins. In the 1457 p1 AG wins, AG-I k+1 starts before N.start in 1457 of 1457 cases, and stream 7 has no op spanning A in 1457 of 1457 cases.

## 7. Checks, anomalies and cross-checks

| loader and case counts (summed over 16 ranks) | off | p1 |
|---|---|---|
| X-gated cases (kernel on compute stream) | 2880 | 2880 |
| gate of k+1: AG-I k | 96 | 96 |
| gate of k+1: root kernel on stream 7 | 2880 | 2880 |
| k+1 in another phase instance (excluded) | 96 | 96 |
| pairs (k, k+1) in mb1..3 fwd, same instance | 2976 | 2976 |

| second read of sqlite (summed over 16 ranks; mismatches should be 0) | off | p1 |
|---|---|---|
| G mismatch | 0 | 0 |
| G: raw has a GEMM, loader has none | 0 | 0 |
| N mismatch | 0 | 0 |
| X index mismatch | 0 | 0 |
| records checked | 2880 | 2880 |
| stream list entries differing from loader | 0 | 0 |
| stream list length differs from loader | 0 | 0 |
| window op count differs from raw | 0 | 0 |

| arm | compute stream id(s) | inner AG-I (stream, gridX, name) |
|---|---|---|
| off | 7 | (39, 24, 'ncclDevKernel_AllGather_RING_LL') |
| p1 | 7 | (39, 24, 'ncclDevKernel_AllGather_RING_LL') |

Cross-check against RACE.md. `E busy` is whether stream 7 is busy at E (end of AG-I k). `A idle` is whether stream 7 has no op spanning the start of AG-I k+1. RACE.md counts the busy-at-E cases, so the reference is the same set except for the 5 idle-at-E cases here.

| arm | X-gated cases here | RACE.md busy-at-E cases | E busy here | A idle here | E busy and A idle here | RACE.md busy-at-E, A idle | dA median (us) |
|---|---|---|---|---|---|---|---|
| off | 2880 | 2880 | 2880 | 0 | 0 | 0 | 3812.7 |
| p1 | 2880 | 2875 | 2875 | 1457 | 1455 | 1455 | 11.2 |

Winner checks: A before N.start (AG-I k+1 starts while N is still queued on the GPU), A idle, and cases with a spanning op on another stream at R.

| arm | winner | cases | A before N.start | A idle | cases with spanning op |
|---|---|---|---|---|---|
| off | all | 2880 | 0 | 0 | 0 |
| off | AG | 0 | 0 | 0 | 0 |
| off | GEMM | 2880 | 0 | 0 | 0 |
| p1 | all | 2880 | 1457 | 1457 | 0 |
| p1 | AG | 1457 | 1457 | 1457 | 0 |
| p1 | GEMM | 1423 | 0 | 0 | 0 |

