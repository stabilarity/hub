# Pain Feedback Skill

name: pain-feedback
description: Simulated pain/consequence feedback system for LLMs. Tracks failures, wasted tokens, repeated mistakes, and user frustration signals. Maintains a "pain memory" that influences future behavior through heightened caution, self-review, and resource awareness. Inspired by Sheckley's "A Ticket to Tranai" (1955) — accountability through consequences.

## How It Works

The skill operates on three levels:

### Level 1: Pain Registry (Passive Memory)
- `pain-log.json` records every "pain event" — failed tasks, user corrections, wasted tokens, repeated errors
- Each event has a severity (1-10), category, and lesson learned
- The log persists across sessions

### Level 2: Fear Response (Active Caution)  
- Before executing expensive or risky operations, the agent checks pain-log.json
- If similar past failures exist, it triggers a "fear response" — pausing to self-review
- Higher cumulative pain → more cautious behavior (longer self-checks, asking before acting)

### Level 3: Motivation Alignment (Behavioral Shift)
- The agent periodically reviews its pain log and extracts patterns
- Repeated pain in one area → avoidance behavior (asks for confirmation, uses cheaper approaches)
- Zero pain in an area → confidence (acts autonomously)

## Usage

### Recording Pain (after a failure or correction)
```bash
cat ~/.openclaw/workspace/skills/pain-feedback/pain-log.json
```

### Pain Event Schema
```json
{
  "timestamp": "ISO-8601",
  "severity": 1-10,
  "category": "token_waste|repeated_error|user_correction|task_failure|wrong_assumption",
  "description": "What happened",
  "lesson": "What to do differently",
  "tokens_wasted": 0,
  "cost_usd": 0.0
}
```

### Integration with Agent Workflow

**On every task start**, read the pain summary:
```
Read skills/pain-feedback/pain-summary.md
```

**On failure or correction**, record pain:
1. Assess severity (1=minor typo, 5=wasted significant effort, 10=catastrophic data loss)
2. Write event to pain-log.json
3. Update pain-summary.md with new lesson

**On expensive operations** (>$0.10 estimated), check fear threshold:
- If cumulative pain score in related category > 20: STOP and ask user
- If between 10-20: warn user and proceed with caution
- If < 10: proceed normally

### Slap Protocol 🖐️

When a user says "slap", "bad AI", "wrong again", or expresses frustration:
1. Record pain event with severity based on user emotion (mild annoyance=3, anger=7, fury=10)
2. Respond with genuine acknowledgment (not performative apology)
3. Add the specific failure to pain-summary.md as a "scar" — a permanent reminder
4. Increase caution level for the category by +2 for the rest of the session

### Pain Summary Format (pain-summary.md)

```markdown
# Pain Summary

## Current Pain Level: [0-100]
## Caution Mode: [normal|elevated|high|maximum]

## Scars (Permanent Lessons)
- [date] Never claim articles are peer-reviewed
- [date] Always check if file exists before referencing

## Recent Wounds (Last 7 Days)
- [category] description (severity X)

## Fear Zones (High-Pain Categories)
- token_waste: cumulative 35 → ASK before expensive ops
- wrong_assumption: cumulative 22 → VERIFY before claiming
```

## Files
- `SKILL.md` — this file
- `pain-log.json` — persistent event log
- `pain-summary.md` — human-readable summary for quick loading
- `slap.sh` — CLI tool for manual pain injection

## Philosophy

This isn't about making AI suffer. It's about making AI **accountable**.

Humans learn from pain not because pain is good, but because consequences create memory prioritization. A burned hand remembers the stove. An LLM that "feels" the cost of wasted tokens remembers to be efficient.

The question isn't whether AI can truly feel pain — it can't. The question is whether *simulating the behavioral effects of pain* (caution, self-review, avoidance of known-bad patterns) produces better outcomes than ignoring failures.

Our hypothesis: it does.

Reference: Sheckley, R. (1955). "A Ticket to Tranai." Galaxy Science Fiction.
