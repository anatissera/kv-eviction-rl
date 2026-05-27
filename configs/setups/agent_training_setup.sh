#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

set -e

cd $TASK_RUNTIME_DIR

# Parse arguments
layers=""
heads=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --layers) layers="$2"; shift 2 ;;
        --heads) heads="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$layers" ] || [ -z "$heads" ]; then
    echo "Error: Missing --layers or --heads"
    echo "Usage: $0 --layers <layer_idx> --heads <head_idx>"
    exit 1
fi

echo "Setting up training for layer $layers, head $heads..."

# Load .env if exists
if [ -f .env ]; then
    echo "Loading .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# Install system dependencies for torch.compile
apt update && apt install python3-dev -y


# Install Python dependencies
uv self update
uv sync

# Download ONLY the specific layer/head data to /dev/shm for fast access
echo "Downloading data for layer $layers, head $heads to /dev/shm/data..."
uv run python scripts/download_data.py download \
    --layers "$layers" \
    --heads "$heads" \
    --target_dir /dev/shm/data

echo "Setup complete for layer $layers, head $heads"
