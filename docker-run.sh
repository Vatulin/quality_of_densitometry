#!/bin/sh
# Linux/macOS: sh docker-run.sh
# NVIDIA host with NVIDIA Container Toolkit: GPUS=all sh docker-run.sh
# Optional: PORT=8080 IMAGE=dxa-quality:local VOLUME=dxa-uploads sh docker-run.sh
# Uploaded files persist in the volume; task metadata is lost on server restart.
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
IMAGE=${IMAGE:-dxa-quality:local}
PORT=${PORT:-8000}
VOLUME=${VOLUME:-dxa-uploads}

docker build --tag "$IMAGE" "$PROJECT_DIR"

set -- docker run --detach --rm --init \
    --publish "$PORT:8000" \
    --mount "type=volume,src=$VOLUME,dst=/app/uploads"

if [ -n "${GPUS:-}" ]; then
    set -- "$@" --gpus "$GPUS"
fi

exec "$@" "$IMAGE"
