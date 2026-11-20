# E1 两节点 MFSDP v2 的 AG 启动延迟：就绪性分析

脚本：`bench/e1_nsys_ready.py`（用法：`python3 e1_nsys_ready.py FILE.sqlite [--range mb0_fwd] [--rows N]`）。
数据：`/tmp/e1an/` 下四个 sqlite（T1024/T256 × rank8/rank0），每个文件取 NVTX 范围的第一次出现（mb0_fwd、mb1_fwd）。

## 结论

H1（SM 竞争）解释了 AG 启动延迟的绝大部分。H2（流依赖）只决定 AG 何时变为 ready，并不解释 ready 之后的等待。
rank8、T1024、mb0 中 63/66 个 AG 晚于 ready，合计 38.94 ms；其中 38.00 ms（97.6%）处于 GEMM 运行期间，60/63 个在 GEMM 进度 0.75–0.89 时才启动。
在 mb0 中 ready 几乎都由同一 AG 流上的前一个 AG 决定（66 个中仅 1 个由 event 绑定）。在 mb1 中 ready 由计算流上的 event 决定（33 个中 31 个），但这些 AG 在 ready 之后仍平均等待约 0.9 ms。
按 delay 的定义，H1 占 ~100%（delayed AG 的 [ready, start] 窗口中 97.6% 有 GEMM 在运行），H2 占 ~0%（ready 型 AG 的 delay 合计 0.007 ms）。这些等待会传到计算流：mb0 中 31/32 个 fc1 之前有 ≥0.05 ms 的空泡，合计 14.26 ms，其中计算忙仅 0.48 ms，所有这些空泡都由延迟的 AG 造成。

## 方法

1. AG 流 = 39，计算流 = 7（按 kernel 名自动检测，四个文件一致）。
2. 对每个 AG K，`prev_end` 为同一流上前一个 GPU 操作（kernel/memset/memcpy）的结束时间，`waits` 为 correlationId 位于前一操作与 K 之间的 stream wait（syncType 2）。
3. wait 通过 `eventSyncId` 对应到 CUDA_EVENT 记录。`CUDA_EVENT.timestamp` 在四个文件的全部 5394 条记录中都为 0，因此不用它。完成时间按流内顺序推算：记录完成 = 该流上此前入队的所有 GPU 操作及 wait 都完成（wait 的完成时间取其 event 的完成时间，因此会传递）。
4. `ready = max(prev_end, wait 完成时间)`，`delay = K.start − ready`。delay < 0.05 ms 记为 ready 型，否则为 delayed 型。
5. 对 delayed 型，统计 [ready, start] 窗口内计算流和 GEMM 的占用时间，以及 start 时 GEMM 的进度 `(t − op.start)/(op.end − op.start)`。
6. fc1 空泡：`gap_op` 为 fc1 前一个计算流操作结束到 fc1 开始的时间（任务定义）；`gap_gemm` 为前一个 GEMM 结束到 fc1 开始的时间。取该窗口内完成时间最晚的 stream wait 作为门控，若其来源为 AG，则输出该 AG 的分类。

## 汇总（每个文件 × 范围）

| 文件 | 范围 | AG 数 | delay 中位 / p90 / 最大 (ms) | delay 合计 (ms) | ready 占比 (delay<0.05) | binding=event | delayed 中 GEMM 后 40% 启动 | delayed 窗口内 GEMM 占用 / delay | fc1 前 gap_gemm 合计 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| T1024 rank8 | mb0_fwd | 66 | 0.387 / 0.896 / 0.897 | 38.94 | 3/66 (4.5%) | 1/66 | 60/63 | 38.00 / 38.94 (97.6%) | 14.26 |
| T1024 rank8 | mb1_fwd | 33 | 0.896 / 0.899 / 0.900 | 27.80 | 2/33 (6.1%) | 31/33 | 31/31 | 27.58 / 27.78 (99.3%) | 2.25 |
| T256 rank8 | mb0_fwd | 66 | 0.007 / 0.176 / 0.617 | 6.33 | 34/66 (51.5%) | 1/66 | 31/32 | 5.41 / 6.20 (87%) | 0.44 |
| T256 rank8 | mb1_fwd | 33 | 0.175 / 0.176 / 0.257 | 5.59 | 2/33 (6.1%) | 1/33 | 31/31 | 5.41 / 5.58 (97%) | 0.86 |
| T1024 rank0 | mb0_fwd | 66 | 0.402 / 0.891 / 1.028 | 34.67 | 5/66 (7.6%) | 1/66 | 60/61 | 33.36 / 34.56 (96.5%) | 10.29 |
| T1024 rank0 | mb1_fwd | 33 | 0.891 / 0.894 / 0.998 | 27.77 | 2/33 (6.1%) | 31/33 | 31/31 | 27.54 / 27.74 (99.3%) | 2.39 |
| T256 rank0 | mb0_fwd | 66 | 0.064 / 0.175 / 0.671 | 6.51 | 33/66 (50%) | 1/66 | 30/33 | 5.29 / 6.43 (82%) | 0.45 |
| T256 rank0 | mb1_fwd | 33 | 0.174 / 0.175 / 0.256 | 5.58 | 2/33 (6.1%) | 1/33 | 31/31 | 5.38 / 5.56 (97%) | 0.88 |

T256 的 mb0 中一半的 AG 不延迟（delay 中位 0.007 ms），mb1 的延迟约 0.17 ms；多数延迟型 AG 仍在 GEMM 后段启动（进度约 0.67，少数为 0.99）。GEMM 长度与 SM 占用未直接测量，因此没有解释 T256 与 T1024 差异的原因。

## 示例行（T1024 rank8 mb0_fwd，第 1–3 层，idx 3–11）

| idx | t_start (ms) | dur (ms) | ready (ms) | delay (ms) | 绑定 | 类别 | 计算流 @ready | 计算流 @start |
|---|---|---|---|---|---|---|---|---|
| 3 | 1.286 | 1.003 | 1.285 | 0.002 | prev AG | ready | idle | idle |
| 4 | 2.291 | 1.663 | 2.289 | 0.002 | prev AG | ready | idle | idle |
| 5 | 4.556 | 0.785 | 3.954 | 0.603 | prev AG | delayed | fc2@0.30 | fc2@0.75 |
| 6 | 6.237 | 0.791 | 5.341 | 0.896 | prev AG | delayed | idle | fc1@0.89 |
| 7 | 7.385 | 0.781 | 7.028 | 0.357 | prev AG | delayed | fc2@0.48 | fc2@0.75 |
| 8 | 9.061 | 0.766 | 8.165 | 0.896 | prev AG | delayed | idle | fc1@0.89 |
| 9 | 10.209 | 0.790 | 9.827 | 0.382 | prev AG | delayed | fc2@0.46 | fc2@0.75 |
| 10 | 11.894 | 0.786 | 10.999 | 0.894 | prev AG | delayed | idle | fc1@0.89 |
| 11 | 13.041 | 0.789 | 12.680 | 0.361 | prev AG | delayed | fc2@0.48 | fc2@0.75 |

fc1 空泡（同一 rank8 mb0）：

| fc1 # | gap_gemm (ms) | 门控 AG | 该 AG 类别 | 该 AG delay (ms) |
|---|---|---|---|---|
| 1 | 0.459 | AG#5 | delayed | 0.603 |
| 2 | 0.456 | AG#7 | delayed | 0.357 |
| 3 | 0.464 | AG#9 | delayed | 0.382 |
| 4 | 0.465 | AG#11 | delayed | 0.361 |

门控 AG 在 fc2 进度 0.75 时启动，并运行约 0.79 ms，结束后 fc1 才开始。

## 与 H2 的对比（mb1）

mb1 中每个 AG 的 ready 由计算流上的 `vectorized_elementwise_kernel` 结束决定，该 kernel 紧接在 fc1 之前。AG 在 ready 后约 0.9 ms 才启动，且全部在 fc1 进度 0.89 时启动。这说明 H2 决定 ready 时刻，H1 决定 ready 之后的等待。

## 未验证的假设

1. correlationId 顺序等于 API 入队顺序。只在 GPU 执行顺序上验证过（流 7、35、39 的逆序数均为 0）。跨线程入队未验证。
2. `CUDA_EVENT.timestamp` 的含义无法确定，因为四个文件中全部为 0。完成时间由 GPU 操作推算，未与任何 GPU 完成时间比对。
3. `eventSyncId` 作为事件实例标识。每个 wait 都匹配到唯一记录，且 eventId 一致。correlationId 启发式的结果与之一致：T1024 rank8 为 97/97（mb0）和 67/67（mb1）。CUPTI 语义未查证。
4. 流内顺序：记录完成 = 此前在该流上入队的全部工作（包括 wait）完成。流 151 和 153 没有任何 kernel，只通过 event 传递依赖。它们的用途（推测是 FSDP 的 unshard/reshard 流）未核实。
5. 同一流上的 AG 严格串行。这是 CUDA 流的语义假设，未在 NCCL 层面验证。
6. GEMM 占用是 SM 占用的代理，没有 CTA 驻留计数器。H1 的证据是时间重叠和启动时 GEMM 进度集中（fc1 约 0.89，fc2 约 0.75），不是直接测量的 SM 占用。
7. AG kernel 的时长包含等待对端 rank 的时间。rank0 第一个 AG 时长 340.177 ms，但它在 launch 后 20.8 µs 就开始执行，等待发生在 kernel 内部。本脚本测量的是启动延迟，不包含对端等待；但 AG 结束时间受对端影响，因此 fc1 空泡的大小中可能含有对端等待。单 rank 的 trace 无法拆开这两部分。
8. 两个节点的时钟未对齐，没有做跨 rank 的时间对齐。rank0 与 rank8 的结论一致，是各自独立计算的结果。
9. 同名 kernel 无法区分 outer 和 inner AG。fc1 的门控是空泡之前最后一个 AG，因此门控编号不等于层号。
10. 窗口内所有流都没有 kernel 的延迟（rank8 mb0 合计 0.89 ms，mb1 合计 0.16 ms；rank0 mb0 合计 1.16 ms）无法用本 trace 解释。候选原因是对端握手或启动延迟，未核实。

## 异常

- AG 数量不是 64/32。mb0 中的 66 = `MFSDP <root> unshard` 的 4 个，加上 blocks.0–30 每个 2 个。mb1 中的 33 = root 的 2 个，加上 blocks.0–30 每个 1 个。blocks.31 在两个范围内都没有 AG，原因未核实。
- rank0 中第一个 AG（mb0 idx 0）时长 340 ms，启动延迟 1.03 ms，窗口内无 kernel，属于上面第 7、10 条。
- 手工核对：用原始 SQL 和递归方式独立重算了 AG 行 1、6、13（rank8 mb0），ready 与 delay 与脚本输出完全一致。

## 文件

- 脚本：`/planning-users/yangyeyuan/megatron-contrib/bench/e1_nsys_ready.py`
- 本报告：`/planning-users/yangyeyuan/megatron-contrib/results/e1/e1-2node-20261111/READINESS.md`
- 各文件的完整输出位于 `/tmp/e1an/run_*.txt`（scratch）。
