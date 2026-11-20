# Source this file: `source /planning-users/yangyeyuan/megatron-contrib/env/mfsdp-dev.sh`
# Python 3.12.16 (uv standalone) + torch 2.14.1+cu130 venv for MFSDP v2 development.
# Does not touch the system Python (3.10 / torch 2.6).

ENVS=/planning-users/yangyeyuan/envs
DEV_REPO=/planning-users/yangyeyuan/Megatron-LM

export PATH="$ENVS/tools/uv/bin:$ENVS/mfsdp-dev/bin:$PATH"
export VIRTUAL_ENV="$ENVS/mfsdp-dev"
export UV_CACHE_DIR="$ENVS/uv-cache"
export UV_PYTHON_INSTALL_DIR="$ENVS/python"
export UV_INDEX_URL=<redacted-url>
export UV_HTTP_TIMEOUT=600
# Mirror redirects wheel downloads to this storage host; the local proxy blocks it.
# ${VAR:+...} keeps this safe under `set -u` (cml containers do not set NO_PROXY).
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}.mi-fds.net" no_proxy="${no_proxy:+$no_proxy,}.mi-fds.net"

# Megatron-LM root (for megatron.core / tests) + standalone megatron_fsdp package.
export PYTHONPATH="$DEV_REPO:$DEV_REPO/megatron/core/distributed/fsdp/src${PYTHONPATH:+:$PYTHONPATH}"
# Writing .pyc to JuiceFS is very slow (first import > 2 min); keep bytecode local.
export PYTHONPYCACHEPREFIX=/tmp/pycache-mfsdp-dev

# Transformer Engine 2.20.2 (see $ENVS/te-build.md). The image ships cuDNN 9.5 and
# NVRTC 12.6 on the default library path; TE needs the venv's cuDNN 9.24 / NVRTC 13.0.
export CUDA_HOME="$ENVS/cuda13"
export LD_LIBRARY_PATH="$ENVS/cuda13/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$ENVS/mfsdp-dev/lib/python3.12/site-packages/nvidia/cu13/lib/libnvrtc.so.13${LD_PRELOAD:+:$LD_PRELOAD}"
