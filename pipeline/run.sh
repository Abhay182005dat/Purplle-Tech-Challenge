#!/bin/bash
# One command to process all clips and feed to API
# Usage: ./run.sh <clips_directory> <store_id>

CLIPS_DIR=$1
STORE_ID=$2
API_URL=${3:-"http://localhost:8000"}

if [ -z "$CLIPS_DIR" ] || [ -z "$STORE_ID" ]; then
    echo "Usage: ./run.sh <clips_directory> <store_id>"
    echo "Example: ./run.sh /path/to/clips ST1008"
    exit 1
fi

echo "================================================="
echo "Store Intelligence Pipeline"
echo "================================================="
echo "Processing clips from: $CLIPS_DIR"
echo "Store ID: $STORE_ID"
echo "API URL: $API_URL"
echo "================================================="

# Process each clip
TOTAL=0
SUCCESS=0
FAILED=0

for clip in "$CLIPS_DIR"/*.mp4; do
    if [ ! -f "$clip" ]; then
        echo "No .mp4 files found in $CLIPS_DIR"
        exit 1
    fi

    TOTAL=$((TOTAL + 1))
    echo ""
    echo "[$TOTAL] Processing: $(basename "$clip")"
    echo "---"

    python3 pipeline/detect.py \
        --clip "$clip" \
        --store_id "$STORE_ID" \
        --api_url "$API_URL" \
        --output "data/events.jsonl"

    if [ $? -eq 0 ]; then
        SUCCESS=$((SUCCESS + 1))
        echo "✓ Success"
    else
        FAILED=$((FAILED + 1))
        echo "✗ Failed"
    fi
done

echo ""
echo "================================================="
echo "Pipeline Complete"
echo "================================================="
echo "Total clips: $TOTAL"
echo "Successful:  $SUCCESS"
echo "Failed:      $FAILED"
echo ""
echo "Events written to: data/events.jsonl"
echo "Events also POSTed to: $API_URL/events/ingest"
echo "================================================="
