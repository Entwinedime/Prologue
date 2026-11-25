ARG BASE_IMAGE=prologue-sglang:ubuntu22.04
FROM scratch AS control-domains
COPY src/prologue/app /prologue/app/
COPY src/prologue/serving /prologue/serving/
COPY src/prologue/contracts/workload /prologue/contracts/workload/
COPY src/prologue/integrations/frameworks/sglang/shared /prologue/integrations/frameworks/sglang/shared/
COPY src/prologue/integrations/frameworks/sglang/control /prologue/integrations/frameworks/sglang/control/
COPY src/prologue/integrations/runtimes/pytorch /prologue/integrations/runtimes/pytorch/

FROM ${BASE_IMAGE}

ENV LD_PRELOAD=""

# Native control uses CPU metadata; accelerator discovery and observation plugins are not part of prediction.
RUN python3 -m pip uninstall -y torch-npu flash-attn-npu prologue-collection

WORKDIR /opt/prologue
RUN --mount=type=bind,from=control-domains,target=/domains \
    --mount=type=bind,source=src,target=/source \
    --mount=type=bind,source=docker/images/package_runtime.py,target=/package_runtime.py \
    python3 /package_runtime.py /domains /source control prologue.app.control_worker
WORKDIR /opt/prologue/control
ENV PYTHONPATH=/opt/prologue/control
ENV SGLANG_USE_CPU_ENGINE=1
ENV OMP_NUM_THREADS=1

ENTRYPOINT ["python3", "-m", "prologue.app.control_worker"]
