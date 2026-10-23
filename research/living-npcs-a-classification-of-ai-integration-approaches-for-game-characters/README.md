# Living NPCs: a classification of AI integration approaches for game characters

Reproducible code for the Stabilarity Research Hub article of the same title (Living NPCs series, article 1).

## Questions

1. How long does one LLM-driven NPC decision take in wall-clock and in game ticks, what does it cost, and how
   reliable is the structured (JSON) output? (E1)
2. Across public AI-NPC systems, which integration surfaces and augmentation types are crowded and which are empty? (E2)

## Experiments

| ID | What | Data |
|---|---|---|
| E1 | 7 low-cost chat models via OpenRouter; a villager NPC receives a perception JSON and must return one JSON object (`action`, `target`, `say`, `opinion_delta`, `reason`). Two conditions: perception only, and perception plus a ~3,000-token, 120-entry memory log. 12 calls per model and condition, streaming, `max_tokens` 200, reasoning disabled. Measures time to first token, total latency, strict/lenient JSON parse, schema validity, billed cost, and opinion deltas for hostile vs friendly speech. | `data/latency_raw.json`, `data/model_prices.json` (collected 2026-10-23 by `collect_latency.py`) |
| E2 | 33 public systems (mods, engine SDKs, research systems, products) coded on two axes: integration surface and the level (0-3) of seven augmentation types. | `data/systems.csv` (one source URL per row) |

`collect_latency.py` needs network access and an `OPENROUTER_API_KEY` environment variable; it was run once.
`analysis.py` runs offline from the committed data: `python3 analysis.py` writes `charts/chart1-4.png` and `results.json`.

## Coding scheme (E2)

Surface: `mod` (patching a shipped game through script, plugin or memory hooks), `engine` (SDK or plugin inside a
game whose source the developer controls), `server` (agent runs on or connects to an authoritative game server or
protocol), `pixels` (agent reads the screen and emits keyboard/mouse input).

| Level | dialogue | vision | hearing | memory | time | action | social |
|---|---|---|---|---|---|---|---|
| 0 | none | none | none | none | turn-based or game paused | none | none |
| 1 | barks / fixed lines | structured state as text | game events as text | conversation window | event-triggered (talk, task) | discrete commands from a menu | implicit in memory text |
| 2 | free text | rendered images to a VLM on demand | player speech via speech-to-text | persistent store / retrieval | periodic asynchronous loop | plans or code executed by game logic | explicit opinion/relationship variable |
| 3 | voice in and out | pixels as primary input | raw audio understanding | reflection or skill library | continuous fixed-rate perception/action | low-level control (keys, joints, continuous) | (unused) |

Codes are our judgement from each system's paper or documentation at the cited URL (versions current in Oct 2026).

## Notes

- Ministral 8B returned HTTP 429 (upstream rate limit) for 43 of 48 attempts, including a sequential retry, so it
  is excluded from latency statistics; the failures are kept in the raw data.
- Prices are OpenRouter list prices on 2026-10-23; costs are the billed cost per call reported by OpenRouter.
- Latency was measured from one server in Europe with one client; network distance, provider routing and load
  change the numbers.
- Total API spend for E1: see `results.json` (`E1_summary.api_spend_usd`).
