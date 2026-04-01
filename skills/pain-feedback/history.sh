#!/bin/bash
# history.sh — View pain history
# Usage: history.sh [--last N]

SKILL_DIR="$(cd "$(dirname "$0")" && pwd)"
HISTORY="$SKILL_DIR/pain-history.jsonl"
STATE="$SKILL_DIR/pain-state.json"

LAST=${2:-10}
if [ "$1" = "--last" ] && [ -n "$2" ]; then
    LAST=$2
fi

echo "=== Pain History ==="
if [ -f "$STATE" ]; then
    SCORE=$(jq -r '.pain_score' "$STATE")
    LEVEL=$(jq -r '.pain_level' "$STATE")
    EVENTS=$(jq -r '.total_events' "$STATE")
    echo "Current: $SCORE/100 [$LEVEL] | Total events: $EVENTS"
    echo ""
fi

if [ -f "$HISTORY" ]; then
    echo "Last $LAST events:"
    tail -n "$LAST" "$HISTORY" | while read -r line; do
        TS=$(echo "$line" | jq -r '.timestamp' | cut -c1-16)
        EV=$(echo "$line" | jq -r '.event')
        PT=$(echo "$line" | jq -r '.points')
        SC=$(echo "$line" | jq -r '.score_after')
        DESC=$(echo "$line" | jq -r '.description' | cut -c1-60)
        printf "  %s  %-20s  %+3d → %3d  %s\n" "$TS" "$EV" "$PT" "$SC" "$DESC"
    done
else
    echo "No history yet — clean slate!"
fi
