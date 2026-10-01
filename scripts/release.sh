#!/usr/bin/env bash
# Publish a release: push the runtime image to GHCR and create a GitHub Release with the torch wheel.
#   scripts/release.sh <revision> [changes.md]
#
# Versions mirror the official pytorch/pytorch tags: <pytorch>-cuda<cuda>-iapetus-r<revision>,
# e.g. 2.14.0-cuda11.8-iapetus-r3; the revision restarts at 1 for a new PyTorch version.
# The image is pushed as :<version> (immutable), :<pytorch>-cuda<cuda>-iapetus (newest revision) and :latest;
# the commit is tagged v<version>. changes.md (optional) is put at the top of the release notes.
#
# Prerequisites: a clean, pushed checkout; the runtime image built from it (scripts/build_images.sh runtime);
# `gh auth refresh -s write:packages` and `gh auth token | docker login ghcr.io -u <user> --password-stdin`.
# RUNTIME_TAG selects the local image; FORCE=1 skips the image/commit consistency check.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

REVISION="${1:?usage: scripts/release.sh <revision> [changes.md]}"
CHANGES="${2:-}"
RUNTIME_TAG="${RUNTIME_TAG:-iapetus/pytorch:2.14.0-cuda11.8-py312}"
REPO="${REPO:-ghcr.io/kazeevn/pytorch}"

[[ "$REVISION" =~ ^[1-9][0-9]*$ ]] || { echo "revision must be a positive integer" >&2; exit 1; }
[[ -z "$CHANGES" || -f "$CHANGES" ]] || { echo "$CHANGES: no such file" >&2; exit 1; }

# The release must point at a commit others can check out.
git diff --quiet HEAD || { echo "uncommitted changes" >&2; exit 1; }
git fetch -q origin
HEAD_SHA="$(git rev-parse HEAD)"
git merge-base --is-ancestor HEAD '@{upstream}' || { echo "HEAD is not pushed" >&2; exit 1; }

# The image must have been built from this commit (older images carry no revision label).
IMAGE_SHA="$(docker image inspect "$RUNTIME_TAG" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')"
if [[ "$IMAGE_SHA" != "$HEAD_SHA" && "${FORCE:-0}" != 1 ]]; then
    echo "$RUNTIME_TAG was built from '${IMAGE_SHA:-unknown}', not $HEAD_SHA; rebuild or set FORCE=1" >&2
    exit 1
fi

in_image() { docker run --rm --entrypoint /opt/venv312/bin/python "$RUNTIME_TAG" -c "$1"; }
TORCH_FULL="$(in_image 'import torch; print(torch.__version__)')"
TORCH="$(grep -oE '^[0-9]+\.[0-9]+\.[0-9]+' <<<"$TORCH_FULL")"
CUDA="$(in_image 'import torch; print(torch.version.cuda)')"
SERIES="$TORCH-cuda$CUDA-iapetus"
VERSION="$SERIES-r$REVISION"
TAG="v$VERSION"

git rev-parse -q --verify "refs/tags/$TAG" >/dev/null && { echo "tag $TAG exists" >&2; exit 1; }
git ls-remote --exit-code --tags origin "$TAG" >/dev/null && { echo "tag $TAG exists on origin" >&2; exit 1; }

WHEEL="dist/torch-${TORCH_FULL}-cp312-cp312-linux_x86_64.whl"
[[ -f "$WHEEL" ]] || { echo "$WHEEL not found" >&2; exit 1; }

echo "Releasing $VERSION from $HEAD_SHA (image $RUNTIME_TAG, torch $TORCH_FULL)"

for t in "$VERSION" "$SERIES" latest; do
    docker tag "$RUNTIME_TAG" "$REPO:$t"
    docker push -q "$REPO:$t"
done
DIGEST="$(docker image inspect "$REPO:$VERSION" --format '{{range .RepoDigests}}{{println .}}{{end}}' | grep "^$REPO@")"

COMPONENTS="$(docker run --rm -i --entrypoint /opt/venv312/bin/python "$RUNTIME_TAG" - <<'PY'
import importlib.metadata as md
import torch
nccl = ".".join(map(str, torch.cuda.nccl.version()))
print(f"| PyTorch | {torch.__version__} (CUDA {torch.version.cuda}, sm_35 + sm_50) |")
print(f"| NCCL | {nccl} |")
print(f"| MAGMA | {'yes' if torch.cuda.has_magma else 'no'} |")
for p in ["numpy", "metatensor-torch", "metatomic-torch", "torch-scatter", "openequivariance", "ase", "orb-models", "warp-lang"]:
    try:
        print(f"| {p} | {md.version(p)} |")
    except md.PackageNotFoundError:
        pass
PY
)"

NOTES="$(mktemp)"
trap 'rm -f "$NOTES"' EXIT
{
    [[ -n "$CHANGES" ]] && { cat "$CHANGES"; echo; }
    cat <<EOF
## Image

\`\`\`bash
docker pull $REPO:$VERSION
\`\`\`

Digest: \`${DIGEST#*@}\`. Also tagged \`$SERIES\` and \`latest\` at the time of release.
Requirements (CPU with AVX2, Kepler \`sm_35\`/Maxwell \`sm_50\` GPU, NVIDIA driver 470) are in the
[README](https://github.com/kazeevn/pytorch-iapetus/blob/$TAG/README.md#will-it-run-on-my-machine).

| Component | Version |
| :--- | :--- |
$COMPONENTS

## License

The image is a derivative of \`nvidia/cuda\` and is distributed under the NVIDIA Deep Learning Container
License (\`/NGC-DL-CONTAINER-LICENSE\` in the image) and the licenses of the software it contains; by using it
you accept those terms. This repository's own files are Apache 2.0.

## Wheel

\`$(basename "$WHEEL")\` is the PyTorch wheel installed in the image. It is not self-contained: it links
against CUDA 11.8, our patched NCCL 2.23.4, MAGMA 2.10.0, oneMKL and OpenMPI 4.1 as installed in the builder
image (\`docker/builder.Dockerfile\`), and is compiled with \`-march=native\` for Haswell. Use the image unless you
are reproducing that environment.
EOF
} >"$NOTES"

git tag -a "$TAG" -m "PyTorch $VERSION"
git push -q origin "$TAG"
gh release create "$TAG" "$WHEEL" --verify-tag --title "PyTorch $VERSION" \
    --notes-file "$NOTES" --generate-notes
echo "Released $TAG: $REPO@${DIGEST#*@}"
