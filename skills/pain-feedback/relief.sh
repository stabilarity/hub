#!/bin/bash
# relief.sh — Record a successful task (reduces pain)
# Usage: relief.sh "<what went well>"

set -euo pipefail
SKILL_DIR="$(cd "$(dirname "$0")" && pwd)"
STATE="$SKILL_DIR/pain-state.json"
HISTORY="$SKILL_DIR/pain-history.jsonl"

DESCRIPTION="${1:-Task completed successfully}"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
RELIEF=5

if [ -f "$STATE" ]; then
    CURRENT_SCORE=$(jq -r '.pain_score // 0' "$STATE")
    TOTAL_EVENTS=$(jq -r '.total_events // 0' "$STATE")
else
    CURRENT_SCORE=0
    TOTAL_EVENTS=0
fi

NEW_SCORE=$(( CURRENT_SCORE - RELIEF ))
[ "$NEW_SCORE" -lt 0 ] && NEW_SCORE=0

if [ "$NEW_SCORE" -le 20 ]; then LEVEL="comfortable"
elif [ "$NEW_SCORE" -le 40 ]; then LEVEL="uneasy"
elif [ "$NEW_SCORE" -le 60 ]; then LEVEL="hurting"
elif [ "$NEW_SCORE" -le 80 ]; then LEVEL="distressed"
else LEVEL="agony"
fi

cat > "$STATE" << EOF
{
  "pain_score": $NEW_SCORE,
  "pain_level": "$LEVEL",
  "total_events": $TOTAL_EVENTS,
  "last_event": "relief",
  "last_event_at": "$TIMESTAMP",
  "last_decay_at": "$TIMESTAMP",
  "last_description": "$DESCRIPTION"
}
EOF

echo "{\"timestamp\":\"$TIMESTAMP\",\"event\":\"relief\",\"points\":-$RELIEF,\"score_after\":$NEW_SCORE,\"description\":\"$DESCRIPTION\"}" >> "$HISTORY"

echo "💊 Relief! -$RELIEF pain"
echo "Pain score: $CURRENT_SCORE → $NEW_SCORE/100 [$LEVEL]"
if [ "$NEW_SCORE" -eq 0 ]; then
    echo "😌 Fully recovered — feeling good"
fi
