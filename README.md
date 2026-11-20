# NVIDIA/Megatron-LM#6714 实验记录：MFSDP v2 DP-inner / DP-outer 通信重叠

Issue：https://github.com/NVIDIA/Megatron-LM/issues/6714

- PR：https://github.com/NVIDIA/Megatron-LM/pull/9539 ，分支 `shsaihdsaiudh/Megatron-LM:mfsdp-6714-outer-ag-overlap`。
- 早期 nsys trace：分支 `mfsdp-6714-nsys`。
- bench 脚本 Gist：https://gist.github.com/shsaihdsaiudh/93b6dcd3bc566626edae7406e7574039

交接说明在 `dev/megatron-contrib/README.zh-CN.md`（英文版 `README.md`），实验计划在 `dev/megatron-contrib/experiment-plan.md`。

- `dev/megatron-contrib/` 是开发机上的完整目录（比 2026-11-18 的本地快照新）：`bench/`、`cml/`、`env/`、`github/`（issue/PR 记录）、`notes/`、`patches/`、`results/`、`snapshots/`。
- `wlcb/megatron-contrib/` 是集群存储上的副本，`results/e4/`（2 节点）只在这里；`snapshots/` 和开发机重复，没有收。
- `local/megatron-contrib/share/` 是 Gist 源文件和 issue 回复草稿，`local/src/megatron-contrib/` 是 GitHub 记录的导出脚本。
- `snapshots/e4-pr-29824b2df/` 保存了本地分支 `mfsdp-v2-outer-allgather-stream` 的 `29824b2df`，这个 commit 没有推到 fork（fork 上同名分支是 `cdff95e45`）。
- 本地 `~/Megatron-LM` 的 `mfsdp-6714-outer-ag-overlap` 还有 5 个文件改了没提交，不在这里。

## 目录

| 目录 | 来源 |
|---|---|
| `local/` | WSL 本地 `~/{megatron-contrib/share,src/megatron-contrib,cloudml-dev/pulled/Megatron-LM}` |
| `dev/` | 开发机 `/planning-users/yangyeyuan/{megatron-contrib,cmdev-logs}` |
| `wlcb/` | 集群共享存储（乌兰察布）`/planning-users/yangyeyuan/megatron-contrib（不含 snapshots）` |

`EXCLUDED-<目录>.txt` 列出没有收进来的文件及大小。收集时跳过的内容：Python/CUDA 环境、venv、wheel、模型和 checkpoint、`*.sqlite`（可以从 `*.nsys-rep` 重新导出）、NeMo-RL 的 rollout 样本 `train_data_step*.jsonl`、50 MB 以上的文件、`__pycache__`。

## 脱敏

这个分支在公开 fork 上。内网 IP、主机名、内部域名和镜像地址、集群卷名、队列和资源名都替换成了 `<redacted…>`；日志里的环境变量转储（`KEY=value` 行和 `env.txt`）已删除。`*.nsys-rep` 会把进程的环境变量写进报告（3976 的 trace 里有云服务 access key），所以全部没有收，清单在 `EXCLUDED-nsys-rep.txt`（如果有）。路径 `/planning-users/yangyeyuan/...` 保留，用来对照原始位置。

整理时间：2026-11-20。
