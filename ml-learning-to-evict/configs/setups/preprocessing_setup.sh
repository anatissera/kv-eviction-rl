#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

set -e

cd $TASK_RUNTIME_DIR

# Load .env file if it exists
if [ -f .env ]; then
    echo "Loading .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# Check required environment variables
if [ -z "$KVCOMPRESSION_MODEL_ID" ]; then
    echo "ERROR: KVCOMPRESSION_MODEL_ID environment variable is not set."
    echo "Please set it to the HuggingFace model ID, e.g.:"
    echo "  export KVCOMPRESSION_MODEL_ID=Qwen/Qwen2.5-7B-Instruct"
    exit 1
fi

echo "Using model: $KVCOMPRESSION_MODEL_ID"

# Install system dependencies for torch.compile
apt update && apt install python3-dev -y

# Install env
uv self update
uv sync

# Download model
# Extract model name from ID (e.g., "Qwen/Qwen2.5-7B-Instruct" -> "Qwen2.5-7B-Instruct")
MODEL_NAME=$(basename "$KVCOMPRESSION_MODEL_ID")
uv run tune download "$KVCOMPRESSION_MODEL_ID" --output-dir "models/$MODEL_NAME" --ignore-patterns "original/consolidated.00.pth"
