"""
UIB Quick Start — Run your first benchmark
"""
# %% Setup
from uib import UIBClient

# Replace with your keys
client = UIBClient(
    stabilarity_key="YOUR_STABILARITY_KEY",
    openrouter_key="YOUR_OPENROUTER_KEY"
)

# %% Run quick benchmark on GPT-4o
result = client.run("openai/gpt-4o", quick=True)

print(f"Model: {result.model}")
print(f"Composite Score: {result.composite_score}/100")
print(f"Efficiency: {result.efficiency_normalized}")
print(f"Cost: ${result.total_cost_usd:.4f}")
print(f"Time: {result.total_latency_ms:.0f}ms")
print()
print("Dimensions:")
for d in result.dimensions:
    print(f"  {d.dimension:20s} {d.score:6.1f}/100  ({d.tasks_completed}/{d.tasks_total} tasks, ${d.cost_usd:.5f})")

# %% Check leaderboard
lb = client.leaderboard()
for entry in lb:
    print(f"#{entry['rank']} {entry['model']:40s} Score: {entry['composite_score']:.1f}  Efficiency: {entry['efficiency_score']:.1f}")
