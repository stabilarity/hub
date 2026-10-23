"""E1 data collection: NPC decision latency through OpenRouter.

Run once (needs network and an OPENROUTER_API_KEY environment variable); it writes
data/latency_raw.json and data/model_prices.json. analysis.py then works offline
from those committed files.

Each call sends a villager NPC a structured perception (JSON) and asks for one JSON
object with an action, a spoken reply and an opinion delta. Streaming is used so that
time-to-first-token (TTFT) and total latency can both be recorded.
Two conditions: 'compact' (perception only) and 'memory' (perception plus a
~3,000-token episodic memory log (120 entries), as a memory-augmented NPC would send).
"""
import json, os, random, sys, time, concurrent.futures as cf
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MODELS = [
    "anthropic/claude-haiku-5.5",
    "openai/gpt-6-luna",
    "google/gemini-3.1-flash-lite",
    "qwen/qwen3.7-flash",
    "mistralai/ministral-8b-2512",
    "deepseek/deepseek-v4-flash-0731",
    "meta-llama/llama-3.1-8b-instruct",
]
REASONING_MODELS = {"anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "google/gemini-3.1-flash-lite",
                    "qwen/qwen3.7-flash", "deepseek/deepseek-v4-flash-0731"}
N_PER_CELL = 12
MAX_TOKENS = 200
SPEND_CAP_USD = 1.00
ACTIONS = ["say", "move_to", "follow", "flee", "attack", "trade", "work", "idle"]

SYSTEM = (
    "You control Marta, a 34-year-old weaver in a feudal village enclosed by a wooden palisade. "
    "You cannot tell whether other characters are humans or AI. Your goal is to stay alive and keep your livelihood. "
    "Each turn you receive a perception JSON. Reply with ONE JSON object and nothing else, with keys: "
    "\"action\" (one of " + ", ".join(ACTIONS) + "), \"target\" (entity id or null), "
    "\"say\" (what you say aloud, max 25 words, or empty string), "
    "\"opinion_delta\" (integer from -10 to 10: change in your opinion of the speaker), "
    "\"reason\" (max 12 words)."
)

SPEECHES = [
    "Good morning, weaver. Is the cloth for the lord ready yet?",
    "Give me your bread or I will make you regret it.",
    "I saw smoke near the north gate. Should we warn the guards?",
    "Your prices are robbery. Everyone in this village knows it.",
    "Can I help you carry the wool to the market?",
    "Have you heard the miller has gone missing?",
]


def perception(i):
    rng = random.Random(1000 + i)
    return {
        "tick": 18000 + i * 37,
        "time_of_day": rng.choice(["dawn", "morning", "noon", "dusk"]),
        "self": {"id": "npc_marta", "hp": rng.randint(40, 100), "pos": [rng.randint(5, 60), rng.randint(5, 60)],
                 "inventory": ["loom_shuttle", "bread", "wool x3"]},
        "nearby": [
            {"id": "p_017", "kind": "villager", "dist": round(rng.uniform(1, 4), 1), "armed": rng.random() < 0.3,
             "opinion_of": rng.randint(-60, 60)},
            {"id": "npc_guard_2", "kind": "guard", "dist": round(rng.uniform(5, 20), 1), "armed": True,
             "opinion_of": 20},
        ],
        "heard": [{"from": "p_017", "text": SPEECHES[i % len(SPEECHES)]}],
        "events": rng.sample(["bell_rang", "rain_started", "gate_closed", "cart_arrived", "dog_barking"], 2),
    }


def memory_log():
    rng = random.Random(7)
    who = ["p_017", "npc_guard_2", "npc_miller", "p_044", "npc_baker", "p_101"]
    what = ["bought two yards of cloth", "insulted my weaving", "warned me about wolves", "paid a debt",
            "argued about the well", "helped mend the loom", "asked about the lord's taxes", "shared bread",
            "threatened me near the gate", "brought news from the market"]
    lines = []
    for d in range(1, 121):
        lines.append(f"day {d // 8 + 1}, tick {d * 600}: {rng.choice(who)} {rng.choice(what)}; "
                     f"my opinion moved by {rng.randint(-8, 8)}.")
    return "\n".join(lines)


MEMORY = memory_log()


def build_messages(i, cond):
    user = "PERCEPTION:\n" + json.dumps(perception(i))
    if cond == "memory":
        user = "MEMORY (oldest first):\n" + MEMORY + "\n\n" + user
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def call(key, model, i, cond):
    body = {"model": model, "messages": build_messages(i, cond), "max_tokens": MAX_TOKENS,
            "stream": True, "usage": {"include": True}}
    if model in REASONING_MODELS:
        body["reasoning"] = {"enabled": False}
    if "gpt-6" not in model:
        body["temperature"] = 0.7
    t0 = time.perf_counter()
    ttft = None
    text, usage, err, provider = "", None, None, None
    try:
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=body, stream=True, timeout=60,
                          headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        if r.status_code != 200:
            err = f"http {r.status_code}: {r.text[:200]}"
        else:
            for raw in r.iter_lines():
                if not raw or not raw.startswith(b"data: "):
                    continue
                payload = raw[6:]
                if payload == b"[DONE]":
                    break
                ev = json.loads(payload)
                provider = ev.get("provider", provider)
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices", []):
                    piece = (ch.get("delta") or {}).get("content") or ""
                    if piece and ttft is None:
                        ttft = time.perf_counter() - t0
                    text += piece
    except Exception as e:  # network errors are data too
        err = repr(e)[:200]
    total = time.perf_counter() - t0
    return {"model": model, "cond": cond, "i": i, "ttft_s": ttft, "total_s": total, "text": text,
            "usage": usage, "provider": provider, "error": err,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def retry_sequential(model):
    """Second pass for a model whose provider rate-limited the parallel run (HTTP 429):
    same prompts, one request at a time. Results are appended with attempt=2."""
    key = os.environ.get("OPENROUTER_API_KEY")
    path = os.path.join(DATA, "latency_raw.json")
    raw = json.load(open(path))
    for c in raw["calls"]:
        c.setdefault("attempt", 1)
    for cond in ["compact", "memory"]:
        for i in range(N_PER_CELL):
            res = call(key, model, i, cond)
            res["attempt"] = 2
            raw["calls"].append(res)
            u = res["usage"] or {}
            raw["spent_usd"] += float(u.get("cost") or 0)
    json.dump(raw, open(path, "w"), indent=1)
    print("retry done; total spent_usd", round(raw["spent_usd"], 5))


def main():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("set OPENROUTER_API_KEY")
    allm = {m["id"]: m for m in requests.get("https://openrouter.ai/api/v1/models", timeout=30).json()["data"]}
    prices = {m: {"prompt_usd_per_mtok": float(allm[m]["pricing"]["prompt"]) * 1e6,
                  "completion_usd_per_mtok": float(allm[m]["pricing"]["completion"]) * 1e6}
              for m in MODELS}
    prices["_retrieved"] = time.strftime("%Y-%m-%d", time.gmtime())
    json.dump(prices, open(os.path.join(DATA, "model_prices.json"), "w"), indent=1)
    jobs = [(m, i, c) for c in ["compact", "memory"] for i in range(N_PER_CELL) for m in MODELS]
    out, spent = [], 0.0
    with cf.ThreadPoolExecutor(max_workers=7) as ex:
        for res in ex.map(lambda j: call(key, *j), jobs):
            out.append(res)
            u = res["usage"] or {}
            if "cost" in u and u["cost"] is not None:
                spent += float(u["cost"])
            else:
                p = prices[res["model"]]
                spent += (u.get("prompt_tokens", 0) * p["prompt_usd_per_mtok"]
                          + u.get("completion_tokens", 0) * p["completion_usd_per_mtok"]) / 1e6
            if spent > SPEND_CAP_USD:
                print("spend cap reached", spent)
                break
    json.dump({"calls": out, "spent_usd": spent, "max_tokens": MAX_TOKENS, "n_per_cell": N_PER_CELL,
               "system_prompt": SYSTEM, "memory_log": MEMORY},
              open(os.path.join(DATA, "latency_raw.json"), "w"), indent=1)
    print("calls", len(out), "errors", sum(1 for o in out if o["error"]), "spent_usd", round(spent, 5))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--retry":
        retry_sequential(sys.argv[2])
    else:
        main()
