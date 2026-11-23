ARG FRAMEWORK_IMAGE=prologue-sglang-runtime:ubuntu22.04
FROM ${FRAMEWORK_IMAGE} AS runtime

# Control imports the same framework, without accelerator operators or weights.
RUN python3 -m pip uninstall -y torch-npu flash-attn-npu
ENV SGLANG_USE_CPU_ENGINE=1 OMP_NUM_THREADS=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0

FROM runtime

WORKDIR /build/prologue
COPY pyproject.toml ./
COPY src src
RUN python3 -m pip install --no-index --no-deps --no-build-isolation . && rm -rf /build/prologue

WORKDIR /control
ENTRYPOINT ["python3", "-u", "-m", "prologue.app.control"]
