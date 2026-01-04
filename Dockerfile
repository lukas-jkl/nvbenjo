FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

# Nvbandwith
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    libboost-program-options-dev && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/NVIDIA/nvbandwidth.git
WORKDIR /nvbandwidth
RUN mkdir build && cd build && cmake .. && make
ENV PATH="/nvbandwidth/build:$PATH"

# get uv
COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /uvx /bin/
ENV PATH="/root/.local/bin:${PATH}"

# install nvbenjo
RUN which uv
COPY . /nvbenjo
RUN --mount=type=cache,target=/root/.cache/uv \
    uv tool install /nvbenjo

CMD ["nvbenjo"]
