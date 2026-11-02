# Living NPCs, article 3: can players tell who is human?

**Question.** Oakhollow (2D) and Quarthaven (3D) are built so that human players and language-model NPCs are the same kind of
entity and the client is never told which is which. Is that true from the player's side, and what gives a human or an NPC away?

## Data

| File | What it is |
|---|---|
| `data/sessions.json` | Every join on the live Oakhollow and Quarthaven servers, 23 Oct - 2 Nov 2026 (in-game pseudonyms, no IPs), with a provenance class: `scripted_test` (a development test script joined at that second), `connection_check` (joined and left within 1 s, no action), `developer_manual` (hand play by the developer, followed within minutes by a feature request about that play), `unknown` |
| `data/natural_scenes.json` | Chat transcripts of the four developer-played scenes (live builds v0 and v4, which still had routines) |
| `data/probe_scenes.json` | First 3 minutes of chat of each article-2 probe run (zero-script engine, template-scripted players) |
| `data/light_items.json` | 60 dialogues sampled (seed 20261103) from the LIGHT test split, via the Hugging Face mirror `dap-exp/light_dialog`: human crowdworkers role-playing characters in a fantasy world (Urbanek et al., EMNLP 2019). One target turn each |
| `data/npc_replies.json` | The engine's NPC mind (`/root/ai-npc-core/brain.js`, unchanged: same rules, prompt layout, model, temperature, JSON schema) answering the same 60 contexts |
| `data/judgments.json` | All LLM judge calls (prompts, verdicts, billed cost) |
| `data/engine_run.json`, `data/engine_events.jsonl` | 150 s private Quarthaven copy on a spare port (own log dir, speech off): state frames, chat, relations and profile panels as seen by a player client; name sign-up test; server log |
| `data/decision_say_pairs.json` | Every NPC line paired with the decision that produced it (article-2 logs + engine run) |
| `data/code_facts.json` | Constants and lists read from the game code (speeds, typing rule, palettes, name pools, lore) |
| `data/spend_*.json` | OpenRouter key usage counter before and after all experiments |

## Experiments (`probe/`)

- `prepare_data.py` builds the data files above from the server logs, the code and the LIGHT parquet.
- `gen_npc_replies.js` turns each LIGHT context into an NPC perception (setting = lore, persona = persona, earlier turns = memories,
  partner's last line = the triggering event) and calls the engine's own `decide()`.
- `judge.py` asks four judges (gemini-2.5-flash-lite, mistral-small-3.2, gpt-4.1-mini, claude-sonnet-4.5; temperature 0) to label
  (T1) each of 118 replies as human or AI, one reply per call, and (T2) every speaker in 7 scene transcripts. Hard spend cap.
- `engine_run.js` runs the leak measurement from the player side. `extract_typing.py` pairs decisions with spoken lines.

## Run

```
python3 analysis.py     # offline, deterministic; writes results.json and charts/chart1..4.png
```

## Outputs

- `results.json`: provenance, code facts, engine-run oracles, typing rule fit, style and reply-latency tables, judge accuracies with
  95% Wilson intervals and binomial tests, feature classifier, spend.
- `charts/chart1.png` judge and classifier accuracy on matched replies; `chart2.png` accuracy of each observable channel;
  `chart3.png` NPC typing delay vs line length; `chart4.png` movement speed of players and NPCs.

The LIGHT excerpt is used for research comparison only; see the original release for its terms.
