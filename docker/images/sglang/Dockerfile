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

# The SDK dumper must publish its running flag before starting the background thread.
COPY src/prologue/integrations/platforms/ascend/runtime/native/profiler /opt/prologue/native/ascend/profiler
RUN bash /opt/prologue/native/ascend/profiler/install.sh

COPY src/prologue/integrations/platforms/ascend/runtime/native/CMakeLists.txt \
    src/prologue/integrations/platforms/ascend/runtime/native/*.cpp \
    src/prologue/integrations/platforms/ascend/runtime/native/*.hpp /opt/prologue/native/ascend/
RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh \
    && cmake -S /opt/prologue/native/ascend -B /tmp/prologue-ascend -DCMAKE_BUILD_TYPE=Release \
        -DPROLOGUE_ASCEND_INCLUDE_DIR=/usr/local/Ascend/ascend-toolkit/latest/include \
    && cmake --build /tmp/prologue-ascend --parallel \
    && mkdir -p /opt/prologue/lib \
    && cp /tmp/prologue-ascend/libprologue_ascend_{observer,weights}.so /opt/prologue/lib/ \
    && rm -rf /tmp/prologue-ascend

COPY src/prologue/integrations/runtimes/pytorch/native /opt/prologue/native/pytorch
RUN cmake -S /opt/prologue/native/pytorch -B /tmp/prologue-gloo -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/prologue-gloo --parallel \
    && cp /tmp/prologue-gloo/libprologue_gloo.so /opt/prologue/lib/ \
    && rm -rf /tmp/prologue-gloo

ENV LD_PRELOAD=/opt/prologue/lib/libprologue_ascend_observer.so:/opt/prologue/lib/libprologue_gloo.so

COPY --chmod=755 src/prologue/integrations/platforms/ascend/runtime/export.sh /opt/prologue/export-profile.sh
COPY --chmod=755 src/prologue/integrations/platforms/ascend/runtime/capture.sh /opt/prologue/capture.sh
# Package the native worker and its actual dependencies, independently of the coordinator and control backend.
WORKDIR /opt/prologue/collector
COPY pyproject.toml .
COPY src/prologue/runtime/io.py src/prologue/runtime/
COPY src/prologue/app/capture_worker.py src/prologue/app/sglang_plugin.py src/prologue/app/
COPY src/prologue/contracts/workload src/prologue/contracts/workload/
COPY src/prologue/acquisition/observation src/prologue/acquisition/observation/
COPY src/prologue/performance/samples.py src/prologue/performance/
COPY src/prologue/integrations/frameworks/sglang/shared src/prologue/integrations/frameworks/sglang/shared/
COPY src/prologue/integrations/frameworks/sglang/observation src/prologue/integrations/frameworks/sglang/observation/
COPY src/prologue/integrations/platforms/ascend/runtime src/prologue/integrations/platforms/ascend/runtime/
COPY src/prologue/integrations/platforms/ascend/analysis src/prologue/integrations/platforms/ascend/analysis/
COPY src/prologue/integrations/runtimes/pytorch/*.py src/prologue/integrations/runtimes/pytorch/
RUN python3 -m pip install --no-deps --no-build-isolation . && rm -rf /opt/prologue/collector

WORKDIR /workspace/trace-sim

CMD ["bash"]
