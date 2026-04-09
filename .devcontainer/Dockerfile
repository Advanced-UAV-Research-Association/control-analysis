FROM mcr.microsoft.com/devcontainers/python:1-3.12

# delete stale source
RUN rm -f /etc/apt/sources.list.d/yarn.list \
# update repos
 && apt update \
# install dependencies for python control
# also install npx for mcp for coding agents
 && apt install -y \
    build-essential \
    gfortran \
    cmake \
    ninja-build \
    libopenblas-dev \
    pkg-config \
    nodejs \
    npm \
 && rm -rf /var/lib/apt/lists/*

