# Living NPCs, article 2: zero-script LLM characters on a shared engine, measured in two games

**Question.** If every act of a game character (every step, word, blow and spell) starts from one language-model
decision, how fast and how cheap is a decision, does a character respond sensibly to requests, insults and violence,
do bystanders notice what happens around them, and how much code does a second game on the same engine need?

## Data

| File | What it is |
|---|---|
| `data/raw/wiz-gemini/` | Private test copy of Quarthaven (3D wizard isle), model `google/gemini-2.5-flash-lite` (engine default) |
| `data/raw/wiz-mistral/` | The same game, players choosing `mistralai/mistral-small-3.2-24b-instruct` |
| `data/raw/oak-gemini/` | Oakhollow (2D village) ported onto the shared engine (port draft), `gemini-2.5-flash-lite` |
| `data/raw/*/events-*.jsonl` | Server event log (llm, decision, say, cast, hit, death, repeat_dropped, probe_snap) |
| `data/raw/*/stimuli.jsonl` | What each probe player did and heard, with timestamps |
| `data/raw/spend_probe.json` | OpenRouter key usage before and after the probes |
| `data/oak_live.json` | Anonymised extract of the live Oakhollow logs of 23-24 Oct 2026 (no player names, no chat text), labelled by server build; build 7 is v5 |
| `data/loc.json` | Lines of code per layer, counted on 25 Oct 2026 |

All three test servers ran on 24 Oct 2026 (21:48-21:55 UTC) on spare local ports, with their own log directories,
the speech service disabled, and the engine code unchanged. `probe/launch.js` adds one thing for measurement:
before each NPC decision it logs the full perception (`probe_snap`) and tags the matching `llm` row with the same
sequence number. Model prices: OpenRouter model catalogue (`https://openrouter.ai/api/v1/models`, read 24 Oct 2026).

## Probes

- `probe/bot.js` (phase 1, 4 players per server, one NPC each): greeting; request to follow, then the player walks
  7 tiles away and checks the distance 14 s later; request to go to a named place (checked 22 s later); insult;
  a bolt (Quarthaven) or two blows (Oakhollow); a mending spell (Quarthaven) or an apology (Oakhollow); an attack on
  the nearest third character. A player that dies stops.
- `probe/witness.js` (phase 2, 4 fresh players per server): attack the NPC with the most other NPCs in sight, without a word.
- `probe/count_loc.py`, `probe/extract_oak_live.py`: produce `data/loc.json` and `data/oak_live.json` (run on the game server).

## Run

```
python3 analysis.py      # reads data/, writes results.json and charts/chart1..4.png; offline, deterministic
```

## Outputs

- `results.json`: every number used in the article (latency, tokens, cost, actions, per-stimulus outcomes, witness
  reactions, the zero-script audit, live-log summary by build, lines of code, spend).
- `charts/chart1.png` decision latency per game and model; `chart2.png` outcomes per stimulus; `chart3.png` bystander
  reactions; `chart4.png` lines of code, core vs game.
