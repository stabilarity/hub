# UIB — Universal Intelligence Benchmark

Inference-agnostic intelligence measurement for any AI model. 8 dimensions, 38 tasks, weighted composite scoring with resource normalization.

## Architecture

```
uib/
├── __init__.py        # Package init
├── models.py          # Pydantic models (request/response schemas)
├── evaluator.py       # Core benchmark engine (task banks, scoring, DB)
client.py              # Python client for the hosted API
01_quick_start.py      # Quick start example
requirements.txt       # Dependencies
```

## Dimensions

| Dimension | Tasks | Weight | Description |
|-----------|-------|--------|-------------|
| Causal Reasoning | 5 | 1.5 | Pearl's causal hierarchy, intervention, confound detection |
| Temporal & Planning | 5 | 1.2 | Long-horizon planning, scheduling, trend analysis |
| Social Cognition | 5 | 1.0 | Theory of mind, negotiation, sarcasm detection |
| Multimodal Synthesis | 5 | 1.0 | Cross-modal reasoning, sensor fusion |
| Tool Creation | 5 | 1.3 | Algorithm design, DSL creation, self-improvement |
| Domain Transfer | 5 | 1.0 | Cross-domain analogy, concept mapping |
| Embodied Intelligence | 5 | 0.8 | Physical reasoning, robot control, manipulation |
| Resource Efficiency | 3 | 1.2 | Compression theory, cost-normalized intelligence |

## Quick Start — Via API

```bash
# Run benchmark on any model
curl -X POST https://hub.stabilarity.com/api/v1/uib/run \
  -H "X-API-Key: YOUR_STABILARITY_KEY" \
  -H "Content-Type: application/json" \
  -d '{"api_key":"YOUR_MODEL_KEY","model":"openai/gpt-4o","quick_mode":true}'

# Get leaderboard
curl -H "X-API-Key: YOUR_KEY" https://hub.stabilarity.com/api/v1/uib/leaderboard

# List dimensions
curl -H "X-API-Key: YOUR_KEY" https://hub.stabilarity.com/api/v1/uib/dimensions
```

## Quick Start — Python Client

```python
from client import UIBClient

uib = UIBClient(
    stabilarity_key="YOUR_STABILARITY_KEY",
    openrouter_key="YOUR_OPENROUTER_KEY"
)

# Run benchmark
result = uib.run("openai/gpt-4o")
print(f"Composite: {result.composite_score}")
for d in result.dimensions:
    print(f"  {d.dimension}: {d.score}")

# Leaderboard
for entry in uib.leaderboard(limit=10):
    print(f"#{entry['rank']} {entry['model']}: {entry['avg_score']}")
```

## Quick Start — Local Model (Ollama)

```bash
# Start Ollama
ollama serve

# Run benchmark against local model
curl -X POST https://hub.stabilarity.com/api/v1/uib/run \
  -H "X-API-Key: YOUR_STABILARITY_KEY" \
  -H "Content-Type: application/json" \
  -d '{"api_key":"none","model":"llama3","api_base":"http://localhost:11434/v1"}'
```

## Scoring

- Each task is scored 0–100 based on regex pattern matching against expected answers
- Dimension score = average of task scores
- Composite = weighted sum across dimensions / total weight
- Efficiency = composite / total inference cost (higher = more intelligent per dollar)

## Self-Hosting

```python
pip install -r requirements.txt

# Use the engine directly
from uib.evaluator import run_full_benchmark
from uib.models import UIBRunRequest

req = UIBRunRequest(
    api_key="your-openai-key",
    model="gpt-4",
    quick_mode=True
)
result = await run_full_benchmark(req)
```

Set `UIB_DB_PATH` env var to customize the SQLite database location.

## Links

- **Web Tool:** https://hub.stabilarity.com/uib-benchmark/
- **API Docs:** https://hub.stabilarity.com/api-gateway/v1/uib/
- **Research Series:** https://hub.stabilarity.com/universal-intelligence-benchmark/
- **Latest Paper:** [The UIB Composite Score](https://hub.stabilarity.com/the-uib-composite-score-integrating-eight-intelligence-dimensions-into-a-unified-benchmark/)

## Contributing

Found a bug? Want to add tasks or dimensions? PRs welcome. Or email contact@stabilarity.com.

## License

CC BY 4.0
