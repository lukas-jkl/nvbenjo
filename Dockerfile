# Stage 1 - build nvbandwidth. Needs nvcc -> start with heavy devel image
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS nvbandwidth-builder

ARG NVBANDWIDTH_REF=v0.10.0

# Build a fat binary so the image runs on any GPU
ARG CUDA_ARCHS="70;75;80;86;89;90;100"

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        cmake \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# Keep the clone shallow but tagged: CMakeLists.txt runs `git describe --tags`
# to stamp the version into the binary.
RUN git clone --depth 1 --branch "${NVBANDWIDTH_REF}" \
        https://github.com/NVIDIA/nvbandwidth.git /src/nvbandwidth

WORKDIR /src/nvbandwidth
RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" && \
    cmake --build build --parallel "$(nproc)"

###############################################################################
# Stage 2 - runtime. The -base image is enough
FROM nvidia/cuda:12.8.1-base-ubuntu22.04

# Pass the version 
ARG NVBENJO_VERSION=0.0.0.dev0
# Set to '[onnx-gpu]' to include the ONNX Runtime backend (~250 MB).
ARG NVBENJO_EXTRAS=""
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${NVBENJO_VERSION}

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=nvbandwidth-builder /src/nvbandwidth/build/nvbandwidth /usr/local/bin/nvbandwidth

COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /bin/
ENV PATH="/root/.local/bin:${PATH}"

# There is no system python in the -base image, so let uv fetch a managed one.
# LINK_MODE=copy: the uv cache below is a BuildKit mount so we use copy to actually get them in the image
ENV UV_PYTHON=3.12 \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_LINK_MODE=copy

COPY . /nvbenjo
RUN --mount=type=cache,target=/root/.cache/uv \
    uv tool install "/nvbenjo${NVBENJO_EXTRAS}" && \
    rm -rf /nvbenjo

CMD ["nvbenjo"]
