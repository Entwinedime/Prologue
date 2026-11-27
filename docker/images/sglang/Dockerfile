ARG BASE_IMAGE=prologue-sglang-env:ubuntu22.04
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG SGLANG_REVISION
RUN [[ "$SGLANG_REVISION" =~ ^[0-9a-f]{40}$ ]]
ENV PROLOGUE_SGLANG_REVISION=${SGLANG_REVISION}
LABEL org.prologue.sglang.revision=${SGLANG_REVISION}

COPY docker/images/sglang/install_sglang.sh /opt/prologue/install-sglang.sh
COPY third_party/sglang /opt/prologue/sglang

RUN bash /opt/prologue/install-sglang.sh /opt/prologue/sglang && rm /opt/prologue/install-sglang.sh

# FA keeps the paged KV layout consumed by native mixed-batch attention.
ENV ASCEND_USE_FA=true ASCEND_USE_FIA=false

# Keep the SDK dumper's startup ordered and measure its queue recorder work.
COPY src/probes/ascend /opt/prologue/native/ascend
RUN bash /opt/prologue/native/ascend/profiler/install.sh

RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh \
    && cmake -S /opt/prologue/native/ascend -B /tmp/prologue-ascend -DCMAKE_BUILD_TYPE=Release \
        -DPROLOGUE_ASCEND_INCLUDE_DIR=/usr/local/Ascend/ascend-toolkit/latest/include \
    && cmake --build /tmp/prologue-ascend --parallel \
    && mkdir -p /opt/prologue/lib \
    && cp /tmp/prologue-ascend/libprologue_ascend_{observer,weights}.so /opt/prologue/lib/ \
    && rm -rf /tmp/prologue-ascend

COPY src/probes/pytorch /opt/prologue/native/pytorch
RUN cmake -S /opt/prologue/native/pytorch -B /tmp/prologue-gloo -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/prologue-gloo --parallel \
    && cp /tmp/prologue-gloo/libprologue_gloo.so /opt/prologue/lib/ \
    && rm -rf /tmp/prologue-gloo

ENV LD_PRELOAD=/opt/prologue/lib/libprologue_ascend_observer.so:/opt/prologue/lib/libprologue_gloo.so

ENV PROLOGUE_FRAMEWORK=sglang PROLOGUE_PLATFORM=ascend PROLOGUE_FRAMEWORK_REVISION=${PROLOGUE_SGLANG_REVISION}

WORKDIR /opt/prologue/collector
COPY docker/images/sglang/pyproject.toml .
COPY src/python/prologue/__init__.py ./python/prologue/
COPY src/python/prologue/integrations/__init__.py src/python/prologue/integrations/bindings.py ./python/prologue/integrations/
COPY src/python/prologue/integrations/frameworks/__init__.py ./python/prologue/integrations/frameworks/
COPY src/python/prologue/integrations/frameworks/sglang/__init__.py ./python/prologue/integrations/frameworks/sglang/
COPY src/python/prologue/integrations/platforms/__init__.py ./python/prologue/integrations/platforms/
COPY src/python/prologue/integrations/platforms/ascend/__init__.py src/python/prologue/integrations/platforms/ascend/backend.py src/python/prologue/integrations/platforms/ascend/trace_store.py ./python/prologue/integrations/platforms/ascend/
COPY src/python/prologue/integrations/platforms/ascend/abi ./python/prologue/integrations/platforms/ascend/abi/
# The capture worker records and interprets traces. Independent reference checks run on the host.
COPY src/python/prologue/integrations/platforms/ascend/runtime ./python/prologue/integrations/platforms/ascend/runtime/
COPY src/python/prologue/integrations/platforms/ascend/analysis ./python/prologue/integrations/platforms/ascend/analysis/
COPY src/python/prologue/integrations/runtimes/__init__.py ./python/prologue/integrations/runtimes/
COPY src/python/prologue/integrations/runtimes/pytorch/__init__.py src/python/prologue/integrations/runtimes/pytorch/shared.py src/python/prologue/integrations/runtimes/pytorch/capture.py ./python/prologue/integrations/runtimes/pytorch/
COPY src/python/prologue/workers/__init__.py src/python/prologue/workers/registry.py ./python/prologue/workers/
COPY src/python/prologue/workers/capture.py src/python/prologue/workers/plugin.py ./python/prologue/workers/
COPY src/python/prologue/common ./python/prologue/common/
COPY src/python/prologue/integrations/frameworks/sglang/abi ./python/prologue/integrations/frameworks/sglang/abi/
COPY src/python/prologue/integrations/frameworks/sglang/shared ./python/prologue/integrations/frameworks/sglang/shared/
COPY src/python/prologue/integrations/frameworks/sglang/observation ./python/prologue/integrations/frameworks/sglang/observation/
RUN install /opt/prologue/native/ascend/scripts/capture.sh /opt/prologue/capture.sh \
    && install /opt/prologue/native/ascend/scripts/export.sh /opt/prologue/export-profile.sh \
    && python3 -m pip install --no-compile --no-deps --no-build-isolation . && rm -rf /opt/prologue/collector

WORKDIR /workspace/prologue

CMD ["bash"]
