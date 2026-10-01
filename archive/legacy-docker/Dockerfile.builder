FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Remove cuda-compat to avoid "forward compatibility was attempted on non supported HW" on Kepler sm_35 and Maxwell sm_50
RUN rm -rf /etc/ld.so.conf.d/cuda-compat*.conf /usr/local/cuda/compat && ldconfig

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ninja-build \
    python3 \
    python3-dev \
    python3-pip \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python build requirements
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
    cmake \
    astunparse \
    numpy \
    pyyaml \
    "setuptools<80.0" \
    wheel \
    typing-extensions \
    sympy \
    jinja2 \
    filelock \
    networkx \
    ninja \
    optree

RUN git config --global --add safe.directory '*'

WORKDIR /workspace
