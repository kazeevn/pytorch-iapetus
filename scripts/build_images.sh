#!/usr/bin/env bash
# Build the Docker images from the repository root.
#   scripts/build_images.sh [builder|runtime|all|push]
# Tags can be overridden with BUILDER_TAG / RUNTIME_TAG; JOBS sets build parallelism.
# `push` tags the local runtime image as PUBLISH_TAG (and :latest) and pushes it to GHCR;
# log in first: gh auth token | docker login ghcr.io -u <user> --password-stdin
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

BUILDER_TAG="${BUILDER_TAG:-iapetus/builder:cuda11.8-py312}"
RUNTIME_TAG="${RUNTIME_TAG:-iapetus/pytorch:2.14.0-cuda11.8-py312}"
PUBLISH_TAG="${PUBLISH_TAG:-ghcr.io/kazeevn/pytorch-iapetus:2.14.0-cuda11.8-py312}"
JOBS="${JOBS:-4}"
TARGET="${1:-all}"

if [[ "$TARGET" == push ]]; then
    docker tag "$RUNTIME_TAG" "$PUBLISH_TAG"
    docker tag "$RUNTIME_TAG" "${PUBLISH_TAG%:*}:latest"
    docker push "$PUBLISH_TAG"
    docker push "${PUBLISH_TAG%:*}:latest"
    exit 0
fi

git submodule update --init third_party/nccl third_party/pytorch_scatter third_party/openequivariance

if [[ "$TARGET" == builder || "$TARGET" == all ]]; then
    docker build -f docker/builder.Dockerfile --build-arg JOBS="$JOBS" -t "$BUILDER_TAG" .
fi
if [[ "$TARGET" == runtime || "$TARGET" == all ]]; then
    docker build -f docker/runtime.Dockerfile --build-arg BUILDER_IMAGE="$BUILDER_TAG" -t "$RUNTIME_TAG" .
fi
