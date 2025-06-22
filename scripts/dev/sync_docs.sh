#!/bin/bash
set -e

SOURCE_FILE="$HOME/.llmcode/analytics.jsonl"
DEST_FILE="llmcode/website/assets/sample-analytics.jsonl"

if [ -f "$SOURCE_FILE" ]; then
    echo "Syncing analytics.jsonl to docs..."
    tail -n 500 "$SOURCE_FILE" > "$DEST_FILE"
else
    echo "Warning: $SOURCE_FILE not found. Skipping analytics sync."
fi
