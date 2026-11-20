# #6714 gather 侧实验方案：MFSDP v2 DP-outer / DP-inner all-gather overlap

基于 main @ d2b0e35cc 的代码阅读和 #5985、#6973 的讨论整理，日期 2026-11-10。

## 1. 任务定义

在每个优化 step 的第一个 microbatch 里，让 DP-outer all-gather（跨节点 RDMA）和
DP-inner all-gather（节点内 NVLink）并行执行，不再交替串行。这是 #6973 的镜像：
#6973 做的是 reduce 侧、最后一个 microbatch。

HFSDP 下，第一个 microbatch 的 forward 对每个 group i 要做三件事：

```
O_i : post_optimizer_model_weight [Flat,Flat] -> model_weight [Replicate,Flat]   DP-outer AG，原地写常驻 buffer
I_i : model_weight [Replicate,Flat] -> unsharded [Replicate,Replicate]          DP-inner AG
F_i : 第 i 层 forward
```

真实的数据依赖只有 `O_i -> I_i -> F_i`，`O_{i+1}` 和 `I_i` 之间没有依赖。今天两者都在
`allgather_stream` 上发出（`parameter_group.py:489-498`），因此执行成 `O1 I1 O2 I2 ...`。
`context.py:115` 的注释也写明，outer all-gather 还没有拆到独立 stream。

目标执行方式：

```
outer stream: O1 O2 O3 O4 ...
inner stream:    I1 I2 I3 ...        通信链长度从 Σ(O+I) 降到约 Σmax(O,I)
```

## 2. 背景：v1 的做法，以及它为什么默认关闭

- v1 有独立的 `outer_fsdp_group_param_gather_stream`（`param_and_grad_buffer.py:4597`），
  但默认路径里 inner stream 发完 outer AG 后立刻 `wait_stream` 等它（`:5011-5014`），
  实际上仍然串行。
- 打开 `hfsdp_param_gather_overlap` 时（#5985，`2790a532a`），outer 比 inner 多预取一个
  FSDP unit，inner 只等自己那些 bucket 的 event（`:4973-5010`），这样才形成流水。
- #5985 的讨论里，wujingyue 和 cspades 指出 v1 还有一条经由 buffer 分配的间接依赖，从
  `inner AG_i` 指向 `outer AG_{i+1}`；另外 v1 每次发 outer AG 前都执行
  `outer_stream.wait_stream(current_stream)`（`:4791`）。多预取一个 unit 就是为了绕开这些依赖。
- 默认关闭的原因没有人明说。记录里的线索有：收益小且随配置变化（Llama3-8B、GA≈1
  约 5%，作者自己也不确定 GA 的值；DSv3-MXFP8、GA=8 约 1.7%）；改变了调度和显存行为；
  review 中出现过 CUDA graph 问题；fused fp8 attention 下 loss 抖动变大；只适用于
  `outer_dp_sharding_strategy='optim'`；仓库惯例是新的性能选项默认关闭。

## 3. 收益估算

### 已有数据点

| 改动 | 配置 | 收益 |
|---|---|---|
| v1 gather 侧（#5985） | Llama3-8B，GA≈1（不确定） | ~5% |
| v1 gather 侧（#5985） | DSv3-MXFP8，GA=8 | ~1.7% |
| v2 reduce 侧（#6973） | 256 GPU，DSv3，HFSDP+EP，开 CUDA graph | 0.06%（在噪声范围内） |
| v2 reduce 侧（#6973） | 128 GB200，DSv4-Flash | ~2% |

### 上限公式

```
单步节省 ≲ min( Σ min(O_i, I_i),  max(0, Σ(O_i + I_i) − Σ F_i) )
占比     = 单步节省 / (GA × T_microbatch + T_optimizer)
```

### 按 H20 粗估（假设值，需要用 E0 实测替换）

配置：Llama-8B 规模、bf16、mesh (2,8)。假设每卡一张 400Gb/s 网卡，有效带宽约
40 GB/s；NVLink AG 有效带宽约 350 GB/s；MFU 50%，约 74 TFLOPS。

- O：每个 rank 收 `16 GB / 8 / 2 ≈ 1 GB`，约 25 ms。
- I：每个 rank 收 `16 GB × 7/8 ≈ 14 GB`，约 40 ms。
- forward：每 1k token 约 0.2 s。按层算每层约 6.7 ms，而每层 O+I 只有约 2 ms。

结论：在合理的 batch 下，gather 会被计算完全盖住，收益接近 0。只有每个 microbatch
的 token 数很小、进入通信受限区间时，收益才会出现。GB200 的算力大约是 H20 的 15 倍，
所以 NVIDIA 能在 GB200 上测出 1.7–5%。

因此，在 H20 上实验的目标是：证明机制成立；报告真实配置下的暴露比例，接近 0 也要报；
大规模下的收益请 wujingyue 和 rapatel 测。

### 需要验证的推断：v2 里只换 stream 可能就够

- v2 的 O 写入常驻的 `model_weight`（`post_optimizer_model_weight` 只是它的 view，
  `parameter_group.py:350`），不需要分配 buffer，所以不存在 v1 那条分配依赖。
- 今天 `allgather_stream` 在每次释放 unsharded 权重时都会等待计算流
  （`module.py:400-401`），这条 stream 被计算限速，和它共用 stream 的 O 也被一起限速。
- 如果 O 有独立 stream，并且这条 stream 每个 step 只等待一次 optimizer 的 cast，
  那么由于 host 通常跑在 device 前面，`O_1…O_N` 会在 outer stream 上连续执行，自然
  领先 inner，不必移植 #5985 的额外预取逻辑。这一点由 E2 的 P1 和 P2 对比来验证。

## 4. 公共设置

- **拓扑**：2 节点 × 8 张 H20，
  `init_device_mesh("cuda", (2, 8), mesh_dim_names=("dp_outer", "dp_inner"))`。
  默认 rank 按节点连续编号，outer 组是 `{i, i+8}`（跨节点），inner 组在节点内。
  用 `NCCL_DEBUG=INFO` 确认 outer communicator 走 `NET/IB`，inner 走 `P2P/NVLink`
  （或 NVLS）。
- **Placements**：和 `tests/unit_tests/distributed/mfsdp_v2/test_fully_shard.py` 里的
  `_hfsdp_placements()` 相同：

  ```python
  Placements(
      dp_axes=[0, 1],
      parameter=[Replicate(), Shard(0)],
      gradient=[Partial("avg"), Shard(0)],
      optimizer=[Shard(0), Shard(0)],
  )
  ```

- **模型**：N 个 block，每个 block 是 `Linear(d,4d) -> GELU -> Linear(4d,d)`，参数用 bf16
  创建，`MixedPrecisionPolicy(main_params_dtype=torch.float32)`，这样 `model_weight` 是
  bf16，通信字节数和真实训练一致。每个 block 和 root 各调用一次 `fully_shard`。
  d=4096、N=64 时约 8.6B 参数。不依赖 Transformer Engine。
- **导入**：`from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import ...`。
  如果 `megatron.core` 的导入链拉进了不需要的依赖，可以把
  `megatron/core/distributed/fsdp/src` 作为独立包安装，改用 `megatron_fsdp.experimental`。
- **训练循环**：照抄 `test_fully_shard.py:436-447`：

  ```python
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
  fully_shard_optimizer(optimizer)
  for step in range(num_steps):
      optimizer.zero_grad(set_to_none=False)
      for mb in range(ga):
          with finalize_gradients(context, enabled=mb == ga - 1):
              torch.cuda.nvtx.range_push(f"step{step}_mb{mb}_fwd")
              fwd_start[mb].record()
              loss = model(x[mb]).float().pow(2).mean()
              fwd_end[mb].record()
              torch.cuda.nvtx.range_pop()
              (loss / ga).backward()
      optimizer.step()
  ```

- **环境变量**：不要设置 `CUDA_DEVICE_MAX_CONNECTIONS=1`，它会让不同 stream 的 kernel
  串行执行，重叠就测不出来了。
- **计时**：在计算流上围绕每个 microbatch 的 forward 和 backward 各记录一对
  `torch.cuda.Event(enable_timing=True)`，step 时间也用 event 记录。warmup 5 步，取后面
  20 步的中位数。
- **脚本位置**：`megatron-contrib/bench/`，不放进仓库。

## 5. 实验

### E0：只测通信，求收益上限（约半天）

不使用 FSDP，用 `mesh.get_group("dp_outer")` 和 `mesh.get_group("dp_inner")` 直接做
`all_gather_into_tensor`，消息大小取单个 block 的真实大小：

- O：每个 rank 发出 `block 字节数 / 16`，收回 `/ 8`。
- I：每个 rank 发出 `/ 8`，收回整个 block。

| 方式 | 做法 |
|---|---|
| 只跑 O | 单独测 outer AG |
| 只跑 I | 单独测 inner AG |
| 两条 stream 同时跑 O 和 I | 测并行时的总时间 |

产出：

- 单层的 `O`、`I`，以及上限 `C = N × min(O, I)`。
- 并行时间和 `max(O, I)` 的差距。差距越大，说明 IB 和 NVLink 同时跑时在 SM、PCIe 或
  NIC 上争抢越严重。

对照组：在单节点上用 mesh `(2,4)` 再跑一次，这时两个轴都走 NVLink，对应 #6714 里
Achyuthan 的单节点数据。

### E1：baseline 下测暴露量（约 1 天）

> **已完成（2026-11-11）**：结果见 `results/e1/e1-2node-20261111/SUMMARY.md`。T ≤ 256 时 E ≈ N·O，
> 也就是 outer AG 完全暴露；T 大时 E 仍有 N·O 的 60% 左右，原因是 NCCL kernel 只能在 GEMM 尾部启动
> （SM 被占满）。这和下面"T 较大时 E ≈ 0"的预期不符。

用 main 的代码，不做修改，跑第 4 节的训练循环。

指标：

```
暴露量 E      = T_fwd(mb0) − T_fwd(mb1)     # 计算量相同，差值就是 outer AG 造成的暴露
可省上限      = min(E, C)
占 step 比例  = min(E, C) / T_step
```

扫描：

- **每个 microbatch 的 token 数** T ∈ {128, 256, 512, 1k, 2k, 4k}，即 batch × seq。
  这是最关键的维度。预计 T 较大时 E ≈ 0，T 很小时 E 显著。曲线的转折点要写进报告。
- **GA** ∈ {1, 4, 8}。E 和 GA 无关，但它占 step 的比例会被 GA 稀释。

nsys 只在 rank 0 和 rank 8 上开，用 `torch.cuda.profiler.start()/stop()` 只抓 2 个
稳态 step：

```bash
nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi -o rank${RANK} python bench.py ...
```

预期在 timeline 上看到：outer AG 和 inner AG 的 NCCL kernel 交替出现在同一条 stream 上，
计算流有等待 AG event 的空泡。

先确认 cml 镜像里有没有 `nsys`。如果没有，就只依赖 event 计时；也可以在所有 rank 上同时
开 torch profiler，这是离线 benchmark，所有 rank 同时开，不会出现 #6973 CI 里那种 rank
不同步的问题。

### E2：最小原型（约 1–2 天）

只为测量而改，不是正式的 PR 代码。改动 `context.py`、`module.py`、`parameter_group.py`：

1. 在 `FsdpContext` 里加 `outer_allgather_stream`。
2. 在 root 的 `pre_forward` 里，在现有的 `allgather_stream.wait_stream(current_stream)`
   （`module.py:278`）旁边，让 outer stream 也等待一次计算流。这保证 O 发生在 optimizer
   cast（`sync_model_weight_from_main_weight()`）之后，并且排在上一个 step 最后一次 inner
   AG 读完 `model_weight` 之后。
3. 在 `unshard_parameters()` 里，如果权重是 stale 的，就在 outer stream 上执行
   `post_optimizer_model_weight.redistribute(model_weight.placements, out=model_weight)`，
   然后 record 一个 event。`allgather_stream` 只 `wait_event` 这个 event，不等整条
   outer stream。

| 变体 | 含义 | 验证什么 |
|---|---|---|
| P0 | main，不改 | baseline |
| P1 | 上面的改法，outer 每个 step 只等计算流一次 | 第 3 节的推断：outer 链是否自然跑在 inner 前面 |
| P2 | 和 P1 相同，但每次发 O 前都 `wait_stream(current_stream)`，模仿 v1 默认路径 | 等待放在哪里决定了能不能重叠（可选） |

每个变体都测：

- **E1 中的指标**：`T_fwd(mb0)`、`T_fwd(mb1)`、`T_step`。理想情况下，P1 的 mb0 forward
  从 `Σ(O+I)` 降到接近 `Σmax(O,I)`。
- **重叠是否真的发生**：用 `nsys stats --report cuda_gpu_trace --format csv` 导出 kernel
  列表，写脚本按 stream 统计 outer 和 inner NCCL kernel 的时间区间有多少重叠。不要用
  进程内 profiler 断言，#6973 在 CI 里因此卡死过。
- **正确性**：用相同的 seed 和数据跑 5 步，P1 和 P0 的 loss 满足 `torch.equal`。这只是
  调度改动，必须 bit-identical。
- **显存**：`torch.cuda.max_memory_allocated()` 应该不变。确认 outer stream 上没有新分配：
  `dbuffer.py` 的注释说带 `out=` 的 allgather 不分配 scratch 存储，但要用
  `torch.cuda.memory._record_memory_history()` 实际验证一次。

### E3：CUDA graph（约 1 天）

维护者是开着 training 和 optimizer CUDA graph 跑的，所以这组必须有。做法参照
`tests/unit_tests/distributed/mfsdp_v2/test_cuda_graph.py:63-145`：在 capture stream 上
warmup 3 次，然后把一整个 step 捕获进 graph，包括 GA 个 microbatch 和
`optimizer.step()`，再测每次 replay 的耗时。

- 分别用 P0 和 P1 跑。用 graph 时没有 host 发射间隙，outer 链只受依赖关系约束，P1 的
  效果可能比 eager 模式下更好，也可能更差。
- 检查 capture 能不能成功。新 stream 必须通过 E2 第 2 步里的 `wait_stream` 加入 capture，
  并且每个 O 都有对应的 I 等待它，这样 outer stream 才会汇合回 capture stream。capture
  失败本身也是需要报告的发现。
- 验证 loss 和 eager 模式下的 P0 一致。

### E4（可选）：真实模型

如果 H20 镜像能装上 TE，可以用 Megatron GPT 跑：
`--use-megatron-fsdp --megatron-fsdp-version 2 --num-distributed-optimizer-instances 2 --outer-dp-sharding-strategy optim`，
配置参考 `tests/functional_tests/test_cases/moe/deepseek_proxy_mfsdp_v2_ep2_hfsdp_mb2/model_config.yaml`，
只看 step time。在 H20 上这组的结果大概率在噪声范围内，所以排在最后。

## 6. 发到 #6714 的内容

| T(tokens/mb) | GA | O/层 | I/层 | C | E (P0) | E (P1) | ΔT_step | 占比 | 开 graph 时 ΔT_step |
|---|---|---|---|---|---|---|---|---|---|

另外附上：

- P1 的重叠证据：重叠 kernel 的统计和 nsys 截图。
- loss bit-identical、显存不变的结论。
- 测量条件：H20 算力低，E 只在 T 小时才显著；大规模下的收益请 wujingyue 和 rapatel 在
  GB200 上测。
- 根据结果提出方案。如果 P1 已经足够，PR 就只需要加独立 stream，不必移植 #5985 的
  额外预取逻辑。
- 需要维护者决定的问题：复用 `overlap_outer_communication`（它的 docstring 和
  `context.py:115` 目前都写着只影响 reduction），还是新开一个 flag。

## 7. 之后的 PR 要求

来自 wujingyue 和 #6973 的评审：

- 改动要小，规模参照 `037b883c9`（15 个文件，约 200 行），选项默认关闭。
- 写法照搬 #6973 的拆分：`parameter_group.py` 提供 outer 和 inner 两步，`module.py`
  负责编排顺序和 stream。
- 测试：
  - bit-identical 的 loss parity：扩展现有 HFSDP overlap parity 的参数化，不要另写一套。
  - 验证重叠确实发生的测试，参考 `test_overlap.py`。非平凡的 2×2 mesh 需要 4 个 rank。
  - CUDA graph capture 覆盖：`test_cuda_graph.py` / `test_mcore_cuda_graph.py`。
  - 测试放置位置遵循 `tests/unit_tests/distributed/mfsdp_v2/AGENTS.md`：按行为选择专门的
    测试文件，`test_fully_shard.py` 只作为最后选择。
- 提交时同时加 `-s -S`，以 draft PR 从 fork 发出，这些都在 WSL 上完成。
