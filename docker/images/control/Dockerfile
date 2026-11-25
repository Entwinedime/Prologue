ARG BASE_IMAGE=prologue-sglang:ubuntu22.04
FROM ${BASE_IMAGE}

ENV LD_PRELOAD=""

# Native control uses CPU metadata; accelerator discovery and observation plugins are not part of prediction.
RUN python3 -m pip uninstall -y torch-npu flash-attn-npu prologue-collection

WORKDIR /opt/prologue/control
COPY src/python/prologue/__init__.py ./prologue/
COPY src/python/prologue/integrations/__init__.py ./prologue/integrations/
COPY src/python/prologue/integrations/frameworks/__init__.py ./prologue/integrations/frameworks/
COPY src/python/prologue/integrations/frameworks/sglang/__init__.py ./prologue/integrations/frameworks/sglang/
COPY src/python/prologue/integrations/runtimes/__init__.py ./prologue/integrations/runtimes/
COPY src/python/prologue/integrations/runtimes/pytorch/__init__.py ./prologue/integrations/runtimes/pytorch/
COPY src/python/prologue/workers/__init__.py ./prologue/workers/
COPY src/python/prologue/workers/control.py ./prologue/workers/
COPY src/python/prologue/common ./prologue/common/
COPY src/python/prologue/integrations/frameworks/sglang/shared ./prologue/integrations/frameworks/sglang/shared/
COPY src/python/prologue/integrations/frameworks/sglang/control ./prologue/integrations/frameworks/sglang/control/
COPY src/python/prologue/integrations/runtimes/pytorch/shared ./prologue/integrations/runtimes/pytorch/shared/
COPY src/python/prologue/integrations/runtimes/pytorch/control ./prologue/integrations/runtimes/pytorch/control/
ENV PYTHONPATH=/opt/prologue/control
ENV SGLANG_USE_CPU_ENGINE=1
ENV OMP_NUM_THREADS=1

ENTRYPOINT ["python3", "-m", "prologue.workers.control"]
