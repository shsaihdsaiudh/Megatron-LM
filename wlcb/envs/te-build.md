# Transformer Engine 2.20.2 install into mfsdp-dev (PyTorch extension built from source)

Date: 2026-11-12. Host: 1x NVIDIA H20 (sm_90), driver 580.105.08.
Venv: /planning-users/yangyeyuan/envs/mfsdp-dev (Python 3.12.16, torch 2.14.1+cu130, nvidia-nccl-cu13 2.30.7). Torch and NCCL are unchanged.

## Result

| Package | Version | Source |
|---|---|---|
| transformer-engine | 2.20.2 | mirror, py3-none-any, `--no-deps` |
| transformer-engine-cu13 | 2.20.2 | mirror, prebuilt core lib, `--no-deps` |
| transformer-engine-torch | 2.20.2 | built locally from sdist (cp312, linux_x86_64) |
| nvidia-cuda-nvcc | 13.0.88 | build toolchain |
| nvidia-cuda-cccl | 13.0.85 | build toolchain |
| nvidia-cudnn-frontend | 1.32.3 | build dep (>=1.29 required) |
| nvidia-mathdx | 25.6.0 | build toolchain |
| pybind11 / ninja / cmake / pip | 3.1.0 / 1.13.2 / 4.4.4 / 26.3 | build toolchain |

Built wheel: `/planning-users/yangyeyuan/envs/wheels/transformer_engine_torch-2.20.2-cp312-cp312-linux_x86_64.whl`
sha256 `233d69633501e1f185416442bcabca22ac6ea863b81cd1bbfc9369bbf71c4838`

CUDA_HOME used for the build: `/planning-users/yangyeyuan/envs/cuda13` (symlink tree; `bin` and `nvvm` point into `nvidia/cu13`, `include` and `lib64` merge cu13, cudnn, nccl, and related wheel dirs). `nvcc --version` reports 13.0.88.

Runtime needs two env vars, which mfsdp-dev.sh does not set (not changed, per instructions):

```
export LD_LIBRARY_PATH=/planning-users/yangyeyuan/envs/cuda13/lib64:$LD_LIBRARY_PATH
export LD_PRELOAD=/planning-users/yangyeyuan/envs/mfsdp-dev/lib/python3.12/site-packages/nvidia/cu13/lib/libnvrtc.so.13
```

Without them, `import transformer_engine.pytorch` fails with the cuDNN 9.5.1 error (first var) and `LayerNormLinear` fails at NVRTC compile (second var). See Problems 2 and 3.

Build wall time: successful build 2 min 0 s (16:54:40 to 16:56:40). Earlier failed attempt about 5 min. Total build phase about 8 min.

## Commands (in order)

```
source /planning-users/yangyeyuan/megatron-contrib/env/mfsdp-dev.sh
uv pip freeze --python $PY > /planning-users/yangyeyuan/envs/freeze-before-te.txt     # 64 packages
uv pip install --python $PY --dry-run nvidia-cuda-nvcc==13.0.88 nvidia-cuda-cccl==13.0.85 pybind11 ninja cmake setuptools wheel packaging nvidia-mathdx "nvidia-cudnn-frontend>=1.29"
uv pip install --python $PY nvidia-cuda-nvcc==13.0.88 nvidia-cuda-cccl==13.0.85 pybind11 ninja cmake setuptools wheel packaging nvidia-mathdx "nvidia-cudnn-frontend>=1.29" pip
# CUDA_HOME symlink tree (bin, nvvm, include/*, lib64/*) built with ln -s; nvcc check:
/planning-users/yangyeyuan/envs/cuda13/bin/nvcc --version   # release 13.0, V13.0.88
uv pip install --python $PY --no-deps --dry-run transformer_engine==2.20.2 transformer_engine_cu13==2.20.2
uv pip install --python $PY --no-deps transformer_engine==2.20.2 transformer_engine_cu13==2.20.2
uv pip install --python $PY einops onnx onnxscript nvdlfw-inspect "pydantic<2.14" importlib-metadata packaging
# sdist: transformer_engine_torch-2.20.2.tar.gz (498055 bytes) from the mirror, checksum verified
# (sha256 4c40ced8a6261649e689076f198d4feaf02277b1ba18931a66eccdaceb33c43e), unpacked to /dev/shm/te-build/src
/dev/shm/te-build/build.sh    # runs: python -m pip wheel --no-build-isolation --no-deps -w /dev/shm/te-build/dist .
cp .../transformer_engine_torch-2.20.2-cp312-cp312-linux_x86_64.whl /planning-users/yangyeyuan/envs/wheels/
uv pip install --python $PY --no-deps /planning-users/yangyeyuan/envs/wheels/transformer_engine_torch-2.20.2-cp312-cp312-linux_x86_64.whl
uv pip freeze --python $PY > /planning-users/yangyeyuan/envs/freeze-after-te.txt
```

build.sh environment (final, successful run):

```
TMPDIR, PIP_CACHE_DIR, TORCH_EXTENSIONS_DIR  -> /dev/shm/te-build (keeps caches off the 3.9 GB root overlay)
CUDA_HOME=/planning-users/yangyeyuan/envs/cuda13 ; PATH=$CUDA_HOME/bin:$PATH
NVTE_FRAMEWORK=pytorch ; NVTE_CUDA_ARCHS=90 ; MAX_JOBS=16
NVTE_PYTORCH_FORCE_BUILD=TRUE   # setup.py otherwise tries a GitHub prebuilt wheel first
CPATH=$SP/nvidia/cudnn/include:$SP/nvidia/nccl/include:$SP/include   # see Problem 1
```

## Problems and fixes

1. **Build failed: cuDNN too old.** `static_assert ... must be built against cuDNN 9.12.0 or later headers` in `csrc/extensions/misc.cpp`.
   Cause: `build_tools/utils.py` resolves CUDA_HOME through the `bin` symlink, so the include path became `nvidia/cu13/include`, which has no cuDNN or NCCL headers. The compiler then fell back to `/usr/include/cudnn.h`, which is cuDNN 9.5. The venv has cuDNN 9.24 (`nvidia/cudnn/include`). The pytorch extension also never adds the cuDNN-frontend include, since only the JAX build does.
   Fix: `CPATH` with the venv cuDNN, NCCL, and cudnn-frontend include dirs. Also removed the failed build tree before rebuilding, so no objects built against the wrong headers survived. Failed log kept only in /dev/shm (deleted at cleanup); the error lines are quoted here.

2. **`import transformer_engine` fails: cuDNN runtime 9.5.1.**
   `RuntimeError: Transformer Engine requires cuDNN 9.12.0 or later, but the cuDNN runtime is 9.5.1.`
   Cause: the TE core library `libtransformer_engine.so` needs `libcudnn.so.9`. Its dynamic loading also opens an unversioned `libcudnn.so` by name. The venv ships only `libcudnn.so.9`, so the loader falls through to the system `/usr/lib/x86_64-linux-gnu/libcudnn.so` (9.5.1), and that copy pulls in 9.5 sub-libraries. Mixing 9.24 and 9.5 then fails with undefined symbols.
   Fix: `/planning-users/yangyeyuan/envs/cuda13/lib64/` holds unversioned symlinks (`libcudnn.so`, `libcudnn_graph.so`, etc.) to the venv 9.24 files. That directory is first on `LD_LIBRARY_PATH`. The venv `site-packages` was not modified.

3. **`LayerNormLinear` forward fails: NVRTC compile error.** `NVRTC Error: NVRTC_ERROR_COMPILATION`, 6 errors in `ln_fwd_kernel.cu`, with `cuda_fp8.hpp` syntax errors.
   Cause: TE compiles some kernels at runtime with NVRTC. The core library binds NVRTC symbols from the global scope, and the system NVRTC 12.6 (`/usr/local/cuda-12.6`, via ld cache) won over the venv's NVRTC 13.0. The 12.6 compiler then could not parse the cu13 headers.
   Fix: `LD_PRELOAD=.../nvidia/cu13/lib/libnvrtc.so.13` so NVRTC 13.0 is the global symbol provider. I also added `libnvrtc.so` and `libnvrtc-builtins.so` symlinks in the cuda13 `lib64` dir. A test showed the unversioned symlinks alone do not fix the global-scope binding, so `LD_PRELOAD` is the required part. The symlinks are harmless and were left in place.

4. **Smaller items.**
   - `pip` was missing from the venv. Installed `pip==26.3`, as the spec allowed.
   - The first `uv pip install` of the toolchain hit the 120 s tool timeout. Rerunning it finished `cmake` and the other packages, which were already cached.
   - `pydantic<2.14` (from the spec) downgraded pydantic 2.14.3 to 2.13.5 and pydantic-core 2.50.4 to 2.46.5. In the freeze diff this is the only change to a pre-existing package. Flagging it because the constraint text named only torch and nvidia-* runtime packages.
   - `pip download` of the sdist failed (it tries to build metadata). Fetched the file directly from the mirror's simple index instead.
   - The spec's test path `megatron/core/distributed/fsdp/src/tests/` does not exist. The FSDP tests are in `tests/unit_tests/distributed/mfsdp_v2/`.
   - The mirror's `transformer_engine_torch` sdist was complete (common headers and csrc present), so the GitHub fallback was not needed.
   - Used `fp8_autocast` in the verification script, which emits a deprecation warning. Harmless.

## Freeze diff (freeze-before-te.txt vs freeze-after-te.txt)

- torch: unchanged, 2.14.1 (`+cu130` local build).
- nvidia-nccl-cu13: unchanged, 2.30.7. `torch.cuda.nccl.version()` returns (2, 30, 7).
- No existing `nvidia-*` package changed version.
- Added: transformer-engine 2.20.2, transformer-engine-cu13 2.20.2, transformer-engine-torch 2.20.2 (file wheel), nvidia-cuda-nvcc 13.0.88, nvidia-cuda-cccl 13.0.85, nvidia-cuda-crt 13.5.37, nvidia-nvvm 13.5.37, nvidia-cudnn-frontend 1.32.3, nvidia-cutlass-dsl 4.8.1 (+ 4 libs-*), nvidia-mathdx 25.6.0, apache-tvm-ffi 0.1.11, cmake 4.4.4, ninja 1.13.2, pybind11 3.1.0, protobuf 7.36.2, pip 26.3, wheel 0.48.0, einops (unchanged 0.8.2), onnx 1.23.2, onnx-ir 1.1.0, onnxscript 0.7.2, nvdlfw-inspect 0.2.2, ml-dtypes 0.6.0, lazy-importer 0.1.0, importlib-metadata 10.2.0, zipp 4.1.2.
- Changed: pydantic 2.14.3 to 2.13.5, pydantic-core 2.50.4 to 2.46.5 (from the `pydantic<2.14` pin).
- Full diff: `/planning-users/yangyeyuan/envs/freeze-after-te.txt` vs `freeze-before-te.txt`.

## Verification

Script: `/dev/shm/te-build/verify.py` (deleted at cleanup; content summarized here). Run with both runtime env vars set.

```
torch 2.14.1+cu130 torch.version.cuda 13.0
nccl (2, 30, 7)
transformer_engine 2.20.2
transformer_engine_cu13 2.20.2
transformer_engine_torch 2.20.2   (module: .../transformer_engine/wheel_lib/transformer_engine_torch.cpython-312-x86_64-linux-gnu.so)
device NVIDIA H20 cc (9, 0)
te.Linear out (32, 1024) torch.bfloat16; x.grad (32, 1024) finite; weight grad (1024, 1024) finite; bias grad (1024,) finite
te.LayerNormLinear out (32, 1024) torch.bfloat16; x.grad (32, 1024) finite; layer_norm_weight/bias grads (1024,) finite; weight grad (1024, 1024) finite; bias grad (1024,) finite
VERIFY OK
```

Megatron collection (read-only, `PYTHONPYCACHEPREFIX=/tmp/pycache-mfsdp-dev`, `-p no:cacheprovider`):

```
python -m pytest --collect-only -q -p no:cacheprovider tests/unit_tests/distributed/mfsdp_v2/
440 tests collected, 0 collection errors
SKIPPED tests/unit_tests/distributed/mfsdp_v2/test_muon.py: could not import 'emerging_optimizers' (not installed; not installed by this task)
```

Per file (all collect):

| File | Tests |
|---|---|
| test_fully_shard.py | 70 |
| test_context.py | 24 |
| test_gradient_accumulation.py | 34 |
| test_optimizer.py (mfsdp_v2) | 15 |
| test_quantization.py | 23 |
| test_quantized_dbuffer.py | 27 |
| test_mcore_dense.py | 14 |

Tests were not run (they need 8 GPUs).

Note: `git status` in the Megatron-LM repo already shows modified files (e.g. `megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/*.py`, `tests/unit_tests/distributed/mfsdp_v2/test_fully_shard.py`). This task made no edits to the repo.

## Cleanup

- `/dev/shm/te-build` (sdist tree, build tree, logs, verify script) removed after this log was written.
- Kept: the wheel copy in `/planning-users/yangyeyuan/envs/wheels/`, the CUDA_HOME tree in `/planning-users/yangyeyuan/envs/cuda13`, and the two freeze files.
- Not done: `emerging_optimizers` is not installed, so `test_muon.py` is skipped at collection.
