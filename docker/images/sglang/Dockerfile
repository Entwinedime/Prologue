ARG BASE_IMAGE=prologue-sglang-env:ubuntu22.04
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG SGLANG_REVISION
RUN [[ "$SGLANG_REVISION" =~ ^[0-9a-f]{40}$ ]]
ENV PROLOGUE_SGLANG_REVISION=${SGLANG_REVISION}
LABEL org.prologue.sglang.revision=${SGLANG_REVISION}

COPY src/collection/native/frameworks/sglang/ascend/install.sh /opt/prologue/install-sglang.sh
COPY third_party/sglang /opt/prologue/sglang

RUN bash /opt/prologue/install-sglang.sh /opt/prologue/sglang && rm /opt/prologue/install-sglang.sh

# FA keeps the paged KV layout consumed by native mixed-batch attention.
ENV ASCEND_USE_FA=true ASCEND_USE_FIA=false

# The SDK dumper must publish its running flag before starting the background thread.
COPY src/collection/native/runtimes/ascend/profiler /opt/prologue/native/ascend/profiler
RUN bash /opt/prologue/native/ascend/profiler/install.sh

COPY src/collection/native/runtimes/ascend/CMakeLists.txt \
    src/collection/native/runtimes/ascend/*.cpp \
    src/collection/native/runtimes/ascend/*.hpp /opt/prologue/native/ascend/
RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh \
    && cmake -S /opt/prologue/native/ascend -B /tmp/prologue-ascend -DCMAKE_BUILD_TYPE=Release \
        -DPROLOGUE_ASCEND_INCLUDE_DIR=/usr/local/Ascend/ascend-toolkit/latest/include \
    && cmake --build /tmp/prologue-ascend --parallel \
    && mkdir -p /opt/prologue/lib \
    && cp /tmp/prologue-ascend/libprologue_ascend_observer.so /opt/prologue/lib/ \
    && rm -rf /tmp/prologue-ascend

ENV LD_PRELOAD=/opt/prologue/lib/libprologue_ascend_observer.so

COPY --chmod=755 src/collection/prologue_collection/adapters/runtimes/ascend/export.sh /opt/prologue/export-profile.sh
COPY --chmod=755 src/collection/prologue_collection/adapters/runtimes/ascend/capture.sh /opt/prologue/capture.sh
COPY pyproject.toml /opt/prologue/collector/pyproject.toml
COPY src/collection/prologue_collection /opt/prologue/collector/src/collection/prologue_collection
COPY src/workload/prologue_workload /opt/prologue/collector/src/workload/prologue_workload
COPY src/adapters/prologue_adapters /opt/prologue/collector/src/adapters/prologue_adapters
RUN python3 -m pip install --no-deps --no-build-isolation /opt/prologue/collector && rm -rf /opt/prologue/collector

WORKDIR /workspace/trace-sim

CMD ["bash"]
