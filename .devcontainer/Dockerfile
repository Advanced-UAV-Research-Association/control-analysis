FROM mcr.microsoft.com/devcontainers/python:1-3.12

RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    cmake \
    ninja-build \
    libopenblas-dev \
    pkg-config \
 && rm -rf /var/lib/apt/lists/*
 
