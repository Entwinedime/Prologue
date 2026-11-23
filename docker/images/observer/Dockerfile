ARG FRAMEWORK_IMAGE=prologue-sglang-runtime:ubuntu22.04
FROM ${FRAMEWORK_IMAGE}

WORKDIR /build/prologue
COPY pyproject.toml ./
COPY src src
RUN python3 -m pip install --no-index --no-deps --no-build-isolation . && rm -rf /build/prologue

COPY --chmod=755 src/prologue/integrations/platforms/ascend/observe.sh /opt/prologue/observe
WORKDIR /observation
ENTRYPOINT ["/opt/prologue/observe"]
