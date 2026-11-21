ARG BASE_IMAGE=prologue-sglang:ubuntu22.04
FROM ${BASE_IMAGE}

ENV LD_PRELOAD=""

# Native control uses CPU metadata; accelerator discovery and observation plugins are not part of prediction.
RUN python3 -m pip uninstall -y torch-npu flash-attn-npu prologue-collection

COPY src/control/prologue_control /opt/prologue/control/prologue_control
COPY src/workload/prologue_workload /opt/prologue/control/prologue_workload
COPY src/adapters/prologue_adapters /opt/prologue/control/prologue_adapters
ENV PYTHONPATH=/opt/prologue/control
ENV SGLANG_USE_CPU_ENGINE=1
ENV OMP_NUM_THREADS=1

ENTRYPOINT ["python3", "-m", "prologue_control"]
