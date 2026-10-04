#!/usr/bin/env bash
# Build the Docker images from the repository root.
#   scripts/build_images.sh [builder|runtime|all]
# Tags can be overridden with BUILDER_TAG / RUNTIME_TAG; JOBS sets build parallelism.
# Publishing is done by scripts/release.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

BUILDER_TAG="${BUILDER_TAG:-iapetus/builder:cuda11.8-cudnn8.7-py312}"
RUNTIME_TAG="${RUNTIME_TAG:-iapetus/pytorch:2.14.0-cuda11.8-cudnn8.7-py312}"
# NCCL and MAGMA build in parallel, so three jobs each use about six cores total.
JOBS="${JOBS:-3}"
TARGET="${1:-all}"

git submodule update --init --recursive third_party/nccl third_party/pytorch_scatter third_party/pytorch_sparse third_party/openequivariance

if [[ "$TARGET" == builder || "$TARGET" == all ]]; then
    docker build -f docker/builder.Dockerfile --build-arg JOBS="$JOBS" -t "$BUILDER_TAG" .
fi
if [[ "$TARGET" == runtime || "$TARGET" == all ]]; then
    # Record the commit (with -dirty for uncommitted changes) so scripts/release.sh can check it.
    GIT_REVISION="$(git rev-parse HEAD)$(git diff --quiet HEAD || echo -dirty)"
    docker build -f docker/runtime.Dockerfile --build-arg BUILDER_IMAGE="$BUILDER_TAG" \
        --build-arg GIT_REVISION="$GIT_REVISION" -t "$RUNTIME_TAG" .
fi
