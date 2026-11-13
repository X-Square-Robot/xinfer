# XInfer image for xinfer-serve and xinfer-router: Python 3.12, torch 2.10 with its CUDA 13.0 wheels, flash-attn,
# xcompute, and XInfer with its serving, model and GPU-image extras. Build from the repository root, with the build
# context "xcompute" a directory holding the xcompute wheel for the deployment GPUs (docs/kernels.md):
#   docker build -f examples/kubernetes/Dockerfile --build-context xcompute=/path/to/wheel-dir -t xinfer:latest .
# The GPU driver comes from the host (NVIDIA container runtime or CDI); the checkpoint and the warmup requests are
# mounted (xinfer.yaml).
FROM ubuntu:24.04 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH
RUN pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130 \
    && pip install --no-cache-dir apache-tvm-ffi==0.1.13.post3

# flash-attn has no wheel for torch 2.10 with CUDA 13: build it against this torch for the GPU architectures in
# FLASH_ATTN_CUDA_ARCHS (120: RTX 5090). Lower MAX_JOBS if the compiler runs out of memory.
FROM runtime AS flash-attn
COPY --from=nvidia/cuda:13.3.0-devel-ubuntu24.04 /usr/local/cuda-13.3 /usr/local/cuda
RUN apt-get update && apt-get install -y --no-install-recommends g++ python3-dev \
    && rm -rf /var/lib/apt/lists/*
ARG FLASH_ATTN_CUDA_ARCHS=120 TORCH_CUDA_ARCH_LIST=12.0a MAX_JOBS=8
RUN pip install --no-cache-dir ninja packaging psutil setuptools wheel \
    && CUDA_HOME=/usr/local/cuda NVCC_THREADS=2 FLASH_ATTENTION_FORCE_BUILD=TRUE \
       pip wheel --no-cache-dir --no-build-isolation --no-deps flash-attn==2.8.3.post1 -w /wheels

FROM runtime
COPY --from=flash-attn /wheels /wheels
COPY --from=xcompute xcompute-*.whl /wheels/
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels
COPY . /xinfer
RUN pip install --no-cache-dir "/xinfer[serve,models,gpu-image]"
