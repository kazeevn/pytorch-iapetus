#!/usr/bin/env bash
# Build the Docker images from the repository root.
#   scripts/build_images.sh [builder|runtime|all]
# Tags can be overridden with BUILDER_TAG / RUNTIME_TAG; JOBS sets build parallelism.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

BUILDER_TAG="${BUILDER_TAG:-iapetus/builder:cuda11.8-py312}"
RUNTIME_TAG="${RUNTIME_TAG:-iapetus/pytorch:2.14.0-cuda11.8-py312}"
JOBS="${JOBS:-4}"
TARGET="${1:-all}"

git submodule update --init third_party/nccl third_party/pytorch_scatter

if [[ "$TARGET" == builder || "$TARGET" == all ]]; then
    docker build -f docker/builder.Dockerfile --build-arg JOBS="$JOBS" -t "$BUILDER_TAG" .
fi
if [[ "$TARGET" == runtime || "$TARGET" == all ]]; then
    docker build -f docker/runtime.Dockerfile --build-arg BUILDER_IMAGE="$BUILDER_TAG" -t "$RUNTIME_TAG" .
fi
