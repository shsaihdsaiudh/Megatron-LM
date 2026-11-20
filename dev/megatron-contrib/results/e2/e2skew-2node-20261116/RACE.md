# E2 race: next compute kernel versus next inner all-gather at the end of each AG-I

Script: `bench/e2race_compare.py`, which reuses the loader, AG-I detection, phase membership and raw wait/record tables of `bench/e2skew_compare.py`. Inputs: the nsys sqlite exports of `d4096_L32_T4096_ga4.rank{0..15}` for `01-v2-off` (fast) and `02-v2-p1` (slow), mb1..3 forward only. All measurements are within one rank, so no clock offset is used.

Definitions (per AG-I k on a rank, k in mb1..3 forward, k+1 the next AG-I on the inner stream, in the same phase instance):

- E: end of AG-I k. Compute state at E: busy if a compute-stream op spans E, else idle.
- C: start of the first compute-stream op starting at or after E. dC = C - E. A: start of AG-I k+1. dA = A - E.
- Event completion: a waited event (cudaStreamWaitEvent) completes when its recording stream reaches the record. On a stream with kernels that is the end of the last op before the record. A stream with no kernels (relay) completes the record after the waits before it on that stream, so the chain is followed.
- Gate: the waited event of the op that completes last. `AG-I k` means its chain is set by AG-I k.
- True race: compute idle at E and the gate of C is AG-I k. Winner: `compute` if C < A, else `AG`.

## 1. Compute-stream state at E, and winner counts

| arm | compute stream at E | AG-I k ends | share | winner compute | winner AG | median time left (busy) or since last op ended (idle), us | busy op is GEMM |
|---|---|---|---|---|---|---|---|
| off | idle | 96 | 0.0323 | 95 | 1 | 741.7 | 0 |
| off | busy | 2880 | 0.9677 | 2880 | 0 | 3486.3 | 2880 |
| off | all | 2976 | 1.0000 | 2975 | 1 |  |  |
| p1 | idle | 101 | 0.0339 | 58 | 43 | 734.9 | 0 |
| p1 | busy | 2875 | 0.9661 | 2875 | 0 | 2945.9 | 2772 |
| p1 | all | 2976 | 1.0000 | 2933 | 43 |  |  |

Winner `compute` means C starts before A. Ties (C start equal to A start) count as `AG`.

Compute stream at A (start of AG-I k+1), split by the state at E. Readiness is the completion time of AG-I k+1's last waited event, relative to E:

| arm | compute stream at E | compute stream at A | AG-I k ends | share of E state | median (GEMM end - A), us | AG-I k+1 readiness - E median, us | dA median, us | A - readiness median, us |
|---|---|---|---|---|---|---|---|---|
| off | idle | GEMM | 95 | 0.9896 | 146.5 | 0.0 | 3810.9 | 3810.9 |
| off | idle | other op | 0 | 0.0000 | n/a | n/a | n/a | n/a |
| off | idle | idle | 1 | 0.0104 | n/a | 0.0 | 2.6 | 2.6 |
| off | busy | GEMM | 2880 | 1.0000 | 145.9 | 3514.8 | 7324.4 | 3812.7 |
| off | busy | other op | 0 | 0.0000 | n/a | n/a | n/a | n/a |
| off | busy | idle | 0 | 0.0000 | n/a | n/a | n/a | n/a |
| p1 | idle | GEMM | 56 | 0.5545 | 146.6 | 0.0 | 3818.7 | 3816.0 |
| p1 | idle | other op | 0 | 0.0000 | n/a | n/a | n/a | n/a |
| p1 | idle | idle | 45 | 0.4455 | n/a | 0.0 | 3.3 | 3.3 |
| p1 | busy | GEMM | 1420 | 0.4939 | 146.0 | 3742.6 | 7564.1 | 3816.0 |
| p1 | busy | other op | 0 | 0.0000 | n/a | n/a | n/a | n/a |
| p1 | busy | idle | 1455 | 0.5061 | n/a | 3738.5 | 3747.7 | 9.2 |

## 2. True race: compute idle at E and next compute kernel gated by AG-I k

Gate: the waited event whose recording chain ends at AG-I k, and which completes last among the waited events of C. dC and dA are measured from E.

### True-race cases

| arm | metric | n | median (us) | p10 (us) | p90 (us) | min (us) | max (us) |
|---|---|---|---|---|---|---|---|
| off | dC = C - E | 96 | 1.9 | 1.6 | 2.2 | 1.5 | 5.7 |
| off | dA = A - E | 96 | 3810.8 | 3799.0 | 3844.5 | 2.6 | 3855.2 |
| p1 | dC = C - E | 96 | 3.2 | 2.2 | 6.9 | 1.9 | 8.1 |
| p1 | dA = A - E | 96 | 3802.3 | 2.8 | 3838.6 | 2.4 | 3850.5 |

| arm | cases | winner compute | winner AG | ties (C start = A start) |
|---|---|---|---|---|
| off | 96 | 95 | 1 | 0 |
| p1 | 96 | 53 | 43 | 0 |

Readiness: `ready` means every waited event of the op had completed by E. An op with no waited events counts as ready.

| arm | cases | C ready at E | AG-I k+1 ready at E | AG wins when AG ready | AG-I k+1 not ready at E | AG wins when AG not ready | AG-I k+1 readiness after E, median (us) | p90 (us) |
|---|---|---|---|---|---|---|---|---|
| off | 96 | 96 | 96 | 1 | 0 | 0 | n/a | n/a |
| p1 | 96 | 96 | 96 | 43 | 0 | 0 | n/a | n/a |


AG wins: overlap of AG-I k+1 with the first GEMM that ends after A, and that GEMM's duration.

| arm | AG wins | GEMM found | GEMM running at A | overlap > 0 | overlap median (us) | overlap p10 (us) | overlap p90 (us) | GEMM dur median (ms) | GEMM dur min (ms) | GEMM dur max (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| off | 1 | 1 | 0 | 1 | 4602.9 | 4602.9 | 4602.9 | 5.256 | 5.256 | 5.256 |
| p1 | 43 | 43 | 0 | 43 | 4560.1 | 4507.7 | 4607.2 | 5.285 | 5.254 | 5.335 |

### Timeline of the true-race cases, by winner (medians, us; GEMM = next compute op after C)

| arm | winner | n | C start - E | C dur | GEMM start - E | GEMM dur | GEMM end - A | AG-I k+1 dur | AG-I k+1 start - E |
|---|---|---|---|---|---|---|---|---|---|
| off | compute | 95 | 1.9 | 1.3 | 5.4 | 3951.4 | 146.5 | 794.1 | 3810.9 |
| off | AG | 1 | 5.7 | 1.7 | 10.0 | 5256.4 | 5263.7 | 4610.2 | 2.6 |
| p1 | compute | 53 | 2.7 | 1.4 | 6.4 | 3956.0 | 146.8 | 788.7 | 3816.0 |
| p1 | AG | 43 | 5.9 | 1.5 | 10.2 | 5284.9 | 5290.5 | 4567.1 | 3.2 |

## 3. Other cases, for comparison

### Case: busy

| arm | metric | n | median (us) | p10 (us) | p90 (us) | min (us) | max (us) |
|---|---|---|---|---|---|---|---|
| off | dC = C - E | 2880 | 3487.8 | 3424.7 | 3733.9 | 655.0 | 3787.2 |
| off | dA = A - E | 2880 | 7324.4 | 7259.3 | 7603.4 | 6786.1 | 8711.9 |
| p1 | dC = C - E | 2875 | 2948.3 | 297.8 | 3720.5 | 4.2 | 4142.7 |
| p1 | dA = A - E | 2875 | 5969.6 | 3365.2 | 8653.9 | 1532.6 | 11598.2 |

| arm | cases | winner compute | winner AG | ties (C start = A start) |
|---|---|---|---|---|
| off | 2880 | 2880 | 0 | 0 |
| p1 | 2875 | 2875 | 0 | 0 |

Readiness: `ready` means every waited event of the op had completed by E. An op with no waited events counts as ready.

| arm | cases | C ready at E | AG-I k+1 ready at E | AG wins when AG ready | AG-I k+1 not ready at E | AG wins when AG not ready | AG-I k+1 readiness after E, median (us) | p90 (us) |
|---|---|---|---|---|---|---|---|---|
| off | 2880 | 2880 | 0 | 0 | 2880 | 0 | 3514.8 | 3760.4 |
| p1 | 2875 | 2875 | 0 | 0 | 2875 | 0 | 3739.7 | 5250.3 |


AG wins: overlap of AG-I k+1 with the first GEMM that ends after A, and that GEMM's duration.

| arm | AG wins | GEMM found | GEMM running at A | overlap > 0 | overlap median (us) | overlap p10 (us) | overlap p90 (us) | GEMM dur median (ms) | GEMM dur min (ms) | GEMM dur max (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| off | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| p1 | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |

### Case: idle, gate other or none

| arm | metric | n | median (us) | p10 (us) | p90 (us) | min (us) | max (us) |
|---|---|---|---|---|---|---|---|
| off | dC = C - E | 0 | n/a | n/a | n/a | n/a | n/a |
| off | dA = A - E | 0 | n/a | n/a | n/a | n/a | n/a |
| p1 | dC = C - E | 5 | 1.1 | 0.7 | 2.1 | 0.7 | 2.1 |
| p1 | dA = A - E | 5 | 7929.5 | 4124.8 | 8091.4 | 4124.8 | 8091.4 |

| arm | cases | winner compute | winner AG | ties (C start = A start) |
|---|---|---|---|---|
| off | 0 | 0 | 0 | 0 |
| p1 | 5 | 5 | 0 | 0 |

Readiness: `ready` means every waited event of the op had completed by E. An op with no waited events counts as ready.

| arm | cases | C ready at E | AG-I k+1 ready at E | AG wins when AG ready | AG-I k+1 not ready at E | AG wins when AG not ready | AG-I k+1 readiness after E, median (us) | p90 (us) |
|---|---|---|---|---|---|---|---|---|
| off | 0 | 0 | 0 | 0 | 0 | 0 | n/a | n/a |
| p1 | 5 | 5 | 0 | 0 | 5 | 0 | 4180.6 | 4284.9 |


AG wins: overlap of AG-I k+1 with the first GEMM that ends after A, and that GEMM's duration.

| arm | AG wins | GEMM found | GEMM running at A | overlap > 0 | overlap median (us) | overlap p10 (us) | overlap p90 (us) | GEMM dur median (ms) | GEMM dur min (ms) | GEMM dur max (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| off | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| p1 | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |

No cases in `idle, gate unresolved`.

## 4. Gating dependencies

Gate: the waited event that completes last. Its time is the completion of the chain that sets it: an op end, or for a relay stream the waits and records before it (see the definitions at the top). `ready at E`: that completion time is at or before E. Readiness after E is in microseconds.

### 4a. AG-I k+1 (gate by case)

| arm | AG-I k+1 | ready at E | not ready at E | waits through a relay stream | gate = AG-I k |
|---|---|---|---|---|---|
| off | 2976 | 96 | 2880 | 2976 | 96 |
| p1 | 2976 | 96 | 2880 | 2976 | 96 |

| arm | case | gate of AG-I k+1 | AG-I k+1 | ready at E | not ready at E | readiness after E median (us) | p90 (us) | max (us) | relay-stream waits |
|---|---|---|---|---|---|---|---|---|---|
| off | busy | kernel vectorized_elementwise_kernel on stream 7 | 2880 | 0 | 2880 | 3514.8 | 3760.4 | 4918.5 | 2880 |
| off | idle, gated by AG-I k | AG-I k | 96 | 96 | 0 | n/a | n/a | n/a | 96 |
| p1 | busy | kernel vectorized_elementwise_kernel on stream 7 | 2875 | 0 | 2875 | 3739.7 | 5250.3 | 7745.9 | 2875 |
| p1 | idle, gated by AG-I k | AG-I k | 96 | 96 | 0 | n/a | n/a | n/a | 96 |
| p1 | idle, gate other or none | kernel vectorized_elementwise_kernel on stream 7 | 5 | 0 | 5 | 4180.6 | 4284.9 | 4284.9 | 5 |

### 4b. Compute kernel C (gate by case)

| arm | case | gate of C | C | ready at E | not ready at E | readiness after E median (us) | p90 (us) | max (us) | relay-stream waits |
|---|---|---|---|---|---|---|---|---|---|
| off | busy | none | 2880 | 2880 | 0 | n/a | n/a | n/a | - |
| off | idle, gated by AG-I k | AG-I k | 96 | 96 | 0 | n/a | n/a | n/a | - |
| p1 | busy | none | 2875 | 2875 | 0 | n/a | n/a | n/a | - |
| p1 | idle, gated by AG-I k | AG-I k | 96 | 96 | 0 | n/a | n/a | n/a | - |
| p1 | idle, gate other or none | none | 5 | 5 | 0 | n/a | n/a | n/a | - |

## 5. Per rank

True races: the race case of section 2. Medians of dC and dA are over the true races of that rank. AG wins are over all AG-I k of the rank (186 pairs per rank in mb1..3 forward).

### off

| rank | node | AG-I k | true races | AG wins | median dC (us) | median dA (us) |
|---|---|---|---|---|---|---|
| 0 | node 0 | 186 | 6 | 0 | 2.0 | 3839.7 |
| 1 | node 0 | 186 | 6 | 1 | 1.8 | 3799.8 |
| 2 | node 0 | 186 | 6 | 0 | 2.1 | 3805.0 |
| 3 | node 0 | 186 | 6 | 0 | 1.7 | 3834.6 |
| 4 | node 0 | 186 | 6 | 0 | 1.7 | 3804.2 |
| 5 | node 0 | 186 | 6 | 0 | 2.0 | 3811.7 |
| 6 | node 0 | 186 | 6 | 0 | 1.8 | 3814.4 |
| 7 | node 0 | 186 | 6 | 0 | 1.8 | 3813.0 |
| 8 | node 1 | 186 | 6 | 0 | 2.1 | 3849.7 |
| 9 | node 1 | 186 | 6 | 0 | 2.1 | 3805.2 |
| 10 | node 1 | 186 | 6 | 0 | 1.9 | 3836.5 |
| 11 | node 1 | 186 | 6 | 0 | 1.8 | 3797.0 |
| 12 | node 1 | 186 | 6 | 0 | 1.7 | 3840.7 |
| 13 | node 1 | 186 | 6 | 0 | 2.0 | 3811.3 |
| 14 | node 1 | 186 | 6 | 0 | 1.8 | 3805.2 |
| 15 | node 1 | 186 | 6 | 0 | 1.7 | 3802.3 |

### p1

| rank | node | AG-I k | true races | AG wins | median dC (us) | median dA (us) |
|---|---|---|---|---|---|---|
| 0 | node 0 | 186 | 6 | 3 | 4.1 | 1921.7 |
| 1 | node 0 | 186 | 6 | 4 | 5.5 | 3.7 |
| 2 | node 0 | 186 | 6 | 4 | 5.8 | 4.4 |
| 3 | node 0 | 186 | 6 | 2 | 3.0 | 3831.0 |
| 4 | node 0 | 186 | 6 | 2 | 2.3 | 3807.5 |
| 5 | node 0 | 186 | 6 | 1 | 2.6 | 3808.7 |
| 6 | node 0 | 186 | 6 | 1 | 2.7 | 3808.8 |
| 7 | node 0 | 186 | 6 | 3 | 4.4 | 1903.7 |
| 8 | node 1 | 186 | 6 | 4 | 5.3 | 3.4 |
| 9 | node 1 | 186 | 6 | 3 | 5.0 | 1903.9 |
| 10 | node 1 | 186 | 6 | 3 | 5.0 | 1919.6 |
| 11 | node 1 | 186 | 6 | 2 | 2.9 | 3802.5 |
| 12 | node 1 | 186 | 6 | 1 | 2.8 | 3838.9 |
| 13 | node 1 | 186 | 6 | 4 | 5.7 | 5.3 |
| 14 | node 1 | 186 | 6 | 4 | 5.3 | 3.9 |
| 15 | node 1 | 186 | 6 | 2 | 2.5 | 3804.6 |

## 6. Findings

Compute stream at E: idle at 96 of 2976 AG-I ends in off (3.2%) and 101 in p1 (3.4%). Winner when idle: compute 95, AG 1 (off); compute 58, AG 43 (p1). Busy at E: compute wins 2880 of 2880 (off) and 2875 of 2875 (p1).

True races (96 per arm, 6 per rank): C and AG-I k+1 are both ready at E in all 96 cases in both arms. dC median 1.9 us (p10-p90 1.6-2.2) in off and 3.2 us (2.2-6.9) in p1. dA median 3810.8 us in off and 3802.3 us in p1. In the 43 p1 AG wins, dA median is 3.2 us. In the 95 off compute wins, C is a memset of 1.3 us median, the GEMM after it starts 5.4 us after E, and AG-I k+1 starts 146.5 us before that GEMM ends. The same figures in p1 compute wins (53): C start 2.7 us, GEMM start 6.4 us, GEMM end minus A 146.8 us. In the 43 p1 AG wins, AG-I k+1 runs 4567 us median, C starts 5.9 us after E, the GEMM starts 10.2 us after E and runs 5285 us (p1 compute-win median GEMM duration: 3956 us).

Busy at E: AG-I k+1 starts with the compute stream idle in 0 of 2880 cases in off and in 1455 of 2875 cases in p1. In those p1 cases, A minus readiness is 9.2 us median. In the 2880 off cases, A minus readiness is 3812.7 us median, with a GEMM running at A (median 145.9 us before its end). In p1 the 1420 busy cases with a GEMM at A show A minus readiness 3816.0 us.

## 7. Checks, anomalies and unclassified cases

| arm | inner AG-I (stream, gridX, name) | compute stream | overlapping compute ops min | max |
|---|---|---|---|---|
| off | (39, 24, 'ncclDevKernel_AllGather_RING_LL') | 7 | 0 | 0 |
| p1 | (39, 24, 'ncclDevKernel_AllGather_RING_LL') | 7 | 0 | 0 |

| check (summed over 16 ranks) | off | p1 |
|---|---|---|
| AG-I in mb1..3 fwd | 3072 | 3072 |
| AG-I k is last of its phase instance (excluded) | 96 | 96 |
| AG-I k+1 gated by AG-I k itself | 96 | 96 |
| AG-I k+1 waits on an event recorded through a relay stream | 2976 | 2976 |
| C: gate time agrees with e2skew wait_info | 2976 | 2976 |
| ops on inner stream between AG-I k and k+1 (any kind) | 96 | 96 |
| pairs (k, k+1) in the same phase instance | 2976 | 2976 |

