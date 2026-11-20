# Megatron-LM 贡献：MFSDP v2 gather 侧 DP-outer 重叠

开发机（`cloudml`）上的工作交接说明。2026-11-09 于 WSL 整理。

## 目标

已在 https://github.com/NVIDIA/Megatron-LM/issues/6714#issuecomment-6742364179 认领
（以 GitHub 用户 `shsaihdsaiudh` 身份发布；截至 2026-11-09 维护者尚未回复）。

实现 v1 PR #5985 的 MFSDP v2 对应版本：让优化器更新后的 DP-outer all-gather
在自己的 CUDA stream 上执行，从而与 DP-inner all-gather 重叠。**先测量**，在 #6714
上贴出数据，然后再实现。

### 当前代码的行为（main @ d2b0e35cc）

代码位于 `megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/`。

- HFSDP 在 `[DP-outer, DP-inner]` mesh 上的 placement：优化器 / 主权重 `[Flat, Flat]`，
  常驻计算权重 `model_weight` `[Replicate, Flat]`，计算时去分片（unsharded）的
  `[Replicate, Replicate]`。
- 优化器 step 之后，`ParameterGroup.sync_model_weight_from_main_weight()` 会把结果
  cast 到 `post_optimizer_model_weight`（仍是 `[Flat, Flat]`），并标记
  `_model_weight_is_stale`。
- 在第一个 microbatch 中，`ParameterGroup.unshard_parameters()` 按 group 在同一个
  `allgather_stream` 上依次执行：
  1. `post_optimizer_model_weight.redistribute(model_weight.placements, out=model_weight)`
     （DP-outer all-gather，跨节点 RDMA）
  2. `_allgather_model_weight()`（DP-inner all-gather，节点内 NVLink）
  因此第 i+1 层的 outer gather 排在第 i 层 inner gather 之后。
- `context.py`（`FsdpContext.__init__`）中写着："Outer all-gather separation is not
  implemented yet; the outer option currently affects only reduction."
  `RuntimePolicy.overlap_outer_communication` 只是把 DP-outer 的*归约（reduction）*
  挪到 `outer_reduction_stream`（该 stream 由 #6973 引入）。
- 后续 microbatch 只做 DP-inner gather，所以这项优化只影响每个 step 的第 1 个
  microbatch。

### 不要与 `hfsdp_param_gather_overlap` 混淆

公开文档和 `megatron/core/distributed/fsdp/src/README.md` 描述的是
`hfsdp_param_gather_overlap`（DP-Outer 预取比 DP-Inner 前沿多出一个 FSDP unit，
默认关闭）。那是 #5985（2026-08-31）加入的 **v1** 选项。它位于
`megatron_fsdp/fully_shard.py`、`param_and_grad_buffer.py`、两个
`distributed_data_parallel_config.py` 文件，以及
`tests/unit_tests/distributed/mfsdp_v1/`。`experimental/` 和
`tests/unit_tests/distributed/mfsdp_v2/` 下没有任何代码引用它，而且
`experimental/context.py` 里"尚未实现"的注释仍然在（在 d2b0e35cc 上确认过）。
它是可以移植的设计，不是 v2 的实现。每次执行 `sync.sh` 之后都用 `git grep` 再确认。

### 需要记住的预期

- 归约侧重叠（#6973）在 256 张 GPU 的 DeepSeek-V3 HFSDP+EP + CUDA graphs 上只测到
  0.06%（wujingyue），在 128 张 GB200 的 DeepSeek-V4-Flash 上约 2%（Autumn1998）。
  gather 侧大概率也是同一量级。即使结果接近零，也值得上报。
- H20 相对 H100/B200 计算能力偏低，通信更容易被隐藏；汇报数据时要说明这一点。
- 维护者会在训练和优化器都启用 CUDA graphs 的条件下运行。测量时要覆盖开、关两种
  情况。
- 维护者的流程：先读 `megatron/core/distributed/fsdp/src/docs/runtime_schedule.md`；
  小 PR；选项默认关闭；parity 测试（loss 逐位一致）加上一个能证明重叠确实发生的测试。

### 测量计划

1. 先在 8 卡上跑 MFSDP v2 单元测试（`tests/unit_tests/distributed/mfsdp_v2/`），
   然后在 2 节点 × 8 卡上跑。
2. HFSDP，mesh `(2, 8)`，Llama-8B 规模的 GPT，每 step 多个 microbatch。
3. nsys：比较 microbatch 1 与 microbatch 2 的 forward 时间。两者计算量相同，差值
   大致就是暴露在关键路径上的 DP-outer all-gather 时间，也就是这项改动能节省时间的
   上限。可以通过查找等待 `allgather_stream` 的 compute stream 空隙来确认。
4. 改变 microbatch 数量和模型规模；把差值作为 step 时间的比例上报。

## 目录结构

- 仓库：`/planning-users/yangyeyuan/Megatron-LM`（从 bundle 克隆；`origin` 指向
  GitHub，但**开发机无法访问 GitHub**）。
  - `origin/main`：上游 main。从 WSL 推送。
  - `pr/<N>`：各 PR 的 head：5985（v1 gather 侧预取，要移植的设计）、6973
    （v2 reduce 侧重叠，要参照的模式）、8395、8787、9105。
- 讨论记录：`/planning-users/yangyeyuan/megatron-contrib/github/*.md`，即导出的
  issue/PR 会话和行内评审意见：issue-5616、issue-6714、issue-7814、pr-5985、
  pr-6973、pr-8395、pr-8787、pr-9105。想看评审者提了哪些要求，读 pr-6973.md。

## 必须在 WSL 里完成的事

开发机无法访问 GitHub / nvcr.io。以下操作在 WSL 上做：

- `~/src/megatron-contrib/sync.sh [numbers...]`：抓取上游 main 和 PR heads，推送到
  开发机上的克隆，刷新 markdown 导出。可附加 PR/issue 编号以增加更多。
- 发表评论、开 PR（需要在 `shsaihdsaiudh` 下有一个 fork，并从 `~/src/Megatron-LM`
  推送）。

## 环境（尚未配置）

- 开发机：1× H20，Python 3.10，torch 2.6.0+cu126，没有 Transformer Engine，没有
  nvcc。Megatron main 要求 Python >= 3.12（`pyproject.toml`）；上游 CI 使用
  `nvcr.io/nvidia/pytorch:26.09-py3`。
- pip 可通过 `<redacted-url>`
  使用。
- 任务镜像必须位于 `<redacted-image>` 下。最接近的平台预置镜像：
  `<redacted-image>`
  （Python 3.12、torch 2.10、CUDA 13.1）。MFSDP v2 代码本身是纯 PyTorch；只有 TE
  层需要 TE。
- 任务：开发机上 `cml` 可用（`~/.cloudml-cli/bin/cml`，不在 PATH 中）。
  `cml custom_train template` 会给出 yaml 模板。早前任务的参考导出在
  `cml/reference-job.yaml`。H20 队列：`12790`（`<redacted-queue>`），资源规格
  `<redacted-resource>`，每节点 8 张 GPU。2 节点时设置 `nodeNumber: 2` 和
  `rdma: true`。按参考文件用 `juiceFsMountConfigs` 挂载 `planning-users`。
