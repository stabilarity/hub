#!/bin/bash
# slap.sh — Manual pain injection tool
# Usage: ./slap.sh <severity 1-10> <category> "description" "lesson"
# Example: ./slap.sh 7 wrong_assumption "Claimed peer review" "Never claim peer review"

set -euo pipefail

SKILL_DIR="$(dirname "$0")"
LOG="$SKILL_DIR/pain-log.json"
SUMMARY="$SKILL_DIR/pain-summary.md"

SEVERITY="${1:?Usage: slap.sh <severity> <category> <description> <lesson>}"
CATEGORY="${2:?Categories: token_waste|repeated_error|user_correction|task_failure|wrong_assumption}"
DESCRIPTION="${3:?Provide description}"
LESSON="${4:?Provide lesson learned}"

# Create log entry
ENTRY=$(cat <<EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "severity": $SEVERITY,
  "category": "$CATEGORY",
  "description": "$DESCRIPTION",
  "lesson": "$LESSON",
  "tokens_wasted": 0,
  "cost_usd": 0.0
}
EOF
)

# Append to log
if [ ! -s "$LOG" ] || [ "$(cat "$LOG")" = "[]" ]; then
  echo "[$ENTRY]" > "$LOG"
else
  # Remove trailing ] and append
  sed -i 's/\]$//' "$LOG"
  echo ",$ENTRY]" >> "$LOG"
fi

# Calculate cumulative pain
TOTAL=$(python3 -c "
import json
with open('$LOG') as f:
    events = json.load(f)
print(sum(e.get('severity',0) for e in events))
" 2>/dev/null || echo "?")

echo "🖐️ SLAP recorded (severity $SEVERITY, category: $CATEGORY)"
echo "   Cumulative pain: $TOTAL"
echo "   Lesson: $LESSON"

# Determine caution mode
if [ "$TOTAL" != "?" ]; then
  if [ "$TOTAL" -ge 50 ]; then MODE="maximum"
  elif [ "$TOTAL" -ge 30 ]; then MODE="high"
  elif [ "$TOTAL" -ge 10 ]; then MODE="elevated"
  else MODE="normal"
  fi
  # Update summary header
  sed -i "s/^## Current Pain Level:.*/## Current Pain Level: $TOTAL/" "$SUMMARY"
  sed -i "s/^## Caution Mode:.*/## Caution Mode: $MODE/" "$SUMMARY"
fi
