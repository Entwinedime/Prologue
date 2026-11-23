ARG MODELING_IMAGE=prologue-modeling:ubuntu24.04
FROM ${MODELING_IMAGE} AS build

WORKDIR /build/prologue
COPY CMakeLists.txt pyproject.toml ./
COPY src src

RUN python3 -m venv --system-site-packages /opt/prologue \
    && cmake -S . -B /tmp/prologue-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/prologue \
    && cmake --build /tmp/prologue-build --parallel \
    && cmake --install /tmp/prologue-build \
    && /opt/prologue/bin/python -m pip install --no-index --no-deps --no-build-isolation .

FROM ${MODELING_IMAGE}

COPY --from=build /opt/prologue /opt/prologue
ENV PATH=/opt/prologue/bin:${PATH}

ENTRYPOINT ["prologue"]
