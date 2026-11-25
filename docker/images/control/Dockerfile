ARG BASE_IMAGE=prologue-sglang:ubuntu22.04
FROM ${BASE_IMAGE}

ENV LD_PRELOAD=""

# Native control uses CPU metadata; accelerator discovery and observation plugins are not part of prediction.
RUN python3 -m pip uninstall -y torch-npu flash-attn-npu prologue-collection

WORKDIR /opt/prologue/control
COPY src/prologue/runtime/io.py prologue/runtime/
COPY src/prologue/app/control_worker.py prologue/app/
COPY src/prologue/serving/control_peer.py prologue/serving/
COPY src/prologue/contracts/workload prologue/contracts/workload/
COPY src/prologue/integrations/frameworks/sglang/*.py prologue/integrations/frameworks/sglang/
COPY src/prologue/integrations/frameworks/sglang/control prologue/integrations/frameworks/sglang/control/
COPY src/prologue/integrations/runtimes/pytorch/protocol_work.py prologue/integrations/runtimes/pytorch/
COPY src/prologue/integrations/runtimes/pytorch/control prologue/integrations/runtimes/pytorch/control/
ENV PYTHONPATH=/opt/prologue/control
ENV SGLANG_USE_CPU_ENGINE=1
ENV OMP_NUM_THREADS=1

ENTRYPOINT ["python3", "-m", "prologue.app.control_worker"]
