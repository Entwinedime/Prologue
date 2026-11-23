ARG BASE_IMAGE=prologue-sglang-env:ubuntu22.04
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG SGLANG_REVISION
RUN [[ "$SGLANG_REVISION" =~ ^[0-9a-f]{40}$ ]]
ENV PROLOGUE_SGLANG_REVISION=${SGLANG_REVISION}
LABEL org.prologue.sglang.revision=${SGLANG_REVISION}

COPY src/prologue/integrations/frameworks/sglang/ascend/install.sh /opt/prologue/install-sglang.sh
COPY third_party/sglang /opt/prologue/sglang
RUN bash /opt/prologue/install-sglang.sh /opt/prologue/sglang && rm /opt/prologue/install-sglang.sh

ENV ASCEND_USE_FA=true ASCEND_USE_FIA=false
