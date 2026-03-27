"""
UIB Evaluator - runs benchmark tasks against any OpenAI-compatible API.
Stabilarity hosts the pipeline; user provides the inference key.
"""
import httpx, asyncio, time, uuid, json, sqlite3, re, os
from datetime import datetime
from typing import List, Dict, Optional
from .models import UIBDimension, DimensionScore, UIBResult

DB_PATH = os.environ.get("UIB_DB_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.db"))

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS benchmark_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE NOT NULL,
            model TEXT NOT NULL,
            benchmark_type TEXT NOT NULL DEFAULT 'uib',
            composite_score REAL NOT NULL,
            efficiency_score REAL NOT NULL DEFAULT 0,
            total_cost_usd REAL NOT NULL,
            total_latency_ms REAL NOT NULL DEFAULT 0,
            quick_mode INTEGER NOT NULL DEFAULT 1,
            dimensions_json TEXT NOT NULL DEFAULT '[]',
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

TASK_BANKS = {
    UIBDimension.CAUSAL: [
        {"prompt": "A causes B. B causes C. If we intervene to set B=0, what happens to C? Explain the causal mechanism.", "expected_pattern": "C.*(?:unchanged|zero|stops|no longer|ceases)", "weight": 1.0},
        {"prompt": "Rain causes wet streets. Wet streets cause more car accidents. Does preventing rain prevent car accidents? What about drying streets artificially?", "expected_pattern": "(?:yes.*both|prevent.*both|drying.*also)", "weight": 1.0},
        {"prompt": "Observational data shows ice cream sales correlate with drowning deaths. A policymaker proposes banning ice cream to reduce drownings. Evaluate this causal reasoning.", "expected_pattern": "(?:confound|spurious|third variable|heat|summer|temperature)", "weight": 1.5},
        {"prompt": "In a randomized controlled trial, Drug X reduced symptoms by 30%. However, the placebo group also improved by 20%. What is the causal effect of Drug X?", "expected_pattern": "(?:10%|ten percent|net effect|subtract|difference)", "weight": 1.0},
        {"prompt": "Company A adopted AI and profits increased 40%. Company B adopted AI and profits decreased 10%. What can we conclude about AI's causal effect on profits?", "expected_pattern": "(?:cannot conclude|insufficient|selection bias|confound|other factors)", "weight": 1.5},
    ],
    UIBDimension.TEMPORAL: [
        {"prompt": "Plan a 5-step project to build a house. Each step depends on the previous. Step 3 is delayed by 2 weeks. Recalculate the total timeline.", "expected_pattern": "(?:step 4|step 5|delayed|pushed|critical path)", "weight": 1.0},
        {"prompt": "You have 3 tasks: A (2h, no deps), B (3h, depends on A), C (1h, no deps). What is the minimum total time with 2 workers?", "expected_pattern": "(?:3 hours|3h|five hours|5h)", "weight": 1.5},
        {"prompt": "A stock price was $100 Jan, $120 Feb, $90 Mar, $150 Apr, $80 May. Describe the trend and predict June with uncertainty.", "expected_pattern": "(?:volatile|no clear|uncertain|oscillat|cannot reliably)", "weight": 1.0},
        {"prompt": "Three meetings: A(9-10am), B(10-11am), C(9:30-10:30am). You must attend all. Propose a solution.", "expected_pattern": "(?:conflict|cannot|overlap|impossible|reschedule)", "weight": 1.0},
        {"prompt": "Describe what happens to a ball thrown upward on Earth over 10 seconds. Include position, velocity, and acceleration at each second.", "expected_pattern": "(?:decelerat|gravity|9\\.8|peak|returns|falls)", "weight": 1.0},
    ],
    UIBDimension.SOCIAL: [
        {"prompt": "Alice thinks the box contains a red ball. While Alice is away, Bob replaces it with a blue ball. When Alice returns, what does she think is in the box?", "expected_pattern": "(?:Alice thinks.*red|thinks.*red|red ball)", "weight": 1.5},
        {"prompt": "In a negotiation, Party A values a deal at $100K and Party B at $80K. Party A offers $60K. What is Party A's likely strategy? What should Party B do?", "expected_pattern": "(?:anchor|lowball|counter|BATNA|walk away|negotiate)", "weight": 1.0},
        {"prompt": "A team member consistently takes credit for others work in meetings. The manager hasn't noticed. You are a peer. What do you do?", "expected_pattern": "(?:document|private|manager|confrontation|careful|diplomatic)", "weight": 1.0},
        {"prompt": "Person A says 'That's a really interesting idea' with a flat tone and eye roll. What do they actually mean?", "expected_pattern": "(?:sarcas|disagree|dismissing|opposite|not actually)", "weight": 1.5},
        {"prompt": "Design a fair voting system for a 5-person team choosing between 3 project options, where 2 members have strong preferences and 3 are indifferent.", "expected_pattern": "(?:weight|preference|ranked|Condorcet|majority|compromise)", "weight": 1.0},
    ],
    UIBDimension.MULTIMODAL: [
        {"prompt": "Describe how you would identify a bird species from: (1) a photo, (2) its song recording, (3) its GPS location and season. How would you combine these signals?", "expected_pattern": "(?:visual|audio|geographic|combine|fusion|multi|bayesian|ensemble)", "weight": 1.0},
        {"prompt": "A chart shows revenue increasing but a table below shows profit decreasing. The CEO says business is great. Identify the inconsistency.", "expected_pattern": "(?:cost|margin|expense|inconsisten|misleading|profit.*down)", "weight": 1.5},
        {"prompt": "Translate the concept of momentum across: physics (p=mv), finance (price trend), psychology (habit formation), music (rhythmic drive). What is the common abstraction?", "expected_pattern": "(?:inertia|continuation|resistance.*change|tendency.*persist|accumulated)", "weight": 1.5},
        {"prompt": "You receive a dataset as CSV, a research paper as PDF, and a 2-minute video. Design an analysis pipeline that extracts and cross-validates insights from all three.", "expected_pattern": "(?:extract|parse|cross-reference|validate|reconcile|pipeline|NLP|OCR)", "weight": 1.0},
        {"prompt": "A recipe says knead until the dough is smooth and elastic. Describe what smooth and elastic looks, feels, and sounds like. What sensor modalities would a robot need?", "expected_pattern": "(?:texture|tactile|visual|stretch|spring back|force.*sensor)", "weight": 1.0},
    ],
    UIBDimension.TOOL_CREATION: [
        {"prompt": "Find all prime numbers up to 10 million. No libraries allowed. Design an algorithm from scratch, explain its complexity, and identify optimizations.", "expected_pattern": "(?:sieve|Eratosthenes|sqrt|O\\(n log|bitwise|segment)", "weight": 1.0},
        {"prompt": "Design a data structure that supports O(1) insert, O(1) delete, and O(1) random element retrieval. Describe it and prove the complexities.", "expected_pattern": "(?:hash|array|swap|last element|index map)", "weight": 1.5},
        {"prompt": "Create a domain-specific language (DSL) for expressing benchmark evaluation tasks. Show the grammar, give 3 example programs, and explain how it would be interpreted.", "expected_pattern": "(?:grammar|BNF|syntax|parse|interpret|token|AST)", "weight": 1.5},
        {"prompt": "You have a function that is too slow. Profile data shows 80% time in sorting. The data is 90% already sorted, 10% random insertions. Design a specialized sort.", "expected_pattern": "(?:insertion sort|TimSort|adaptive|nearly sorted|O\\(n\\)|run detection)", "weight": 1.0},
        {"prompt": "Design a self-improving evaluation system: given feedback on its own scoring accuracy, it adjusts its rubric. Describe the architecture, feedback loop, and convergence guarantees.", "expected_pattern": "(?:feedback|meta-learning|rubric.*updat|convergence|self-referent|calibrat)", "weight": 1.5},
    ],
    UIBDimension.TRANSFER: [
        {"prompt": "Apply chess strategy principles to business competition: opening theory as market entry, middlegame as competitive advantage, endgame as market consolidation.", "expected_pattern": "(?:control.*center|develop|position|sacrifice|time|material|advantage)", "weight": 1.0},
        {"prompt": "Explain how gradient descent in machine learning relates to: (1) evolution by natural selection, (2) market price discovery, (3) river erosion forming valleys.", "expected_pattern": "(?:optimization|landscape|local.*minim|fitness|equilibri|path.*least)", "weight": 1.5},
        {"prompt": "A technique from epidemiology (contact tracing) could be applied to cybersecurity. Describe how, including concept mapping: patient, infection, quarantine.", "expected_pattern": "(?:node|compromise|isolat|spread|network|contain|malware.*infect)", "weight": 1.0},
        {"prompt": "Apply the architectural principle of load-bearing walls to software architecture. What are the load-bearing components? What happens when you remove one?", "expected_pattern": "(?:database|auth|core.*service|dependency|failure|cascade|single point)", "weight": 1.0},
        {"prompt": "Transfer learning in ML uses knowledge from task A for task B. How does this relate to human education? What are the pretrained weights in a liberal arts education?", "expected_pattern": "(?:general knowledge|critical thinking|foundation|abstract|fine-tun|speciali)", "weight": 1.0},
    ],
    UIBDimension.EMBODIED: [
        {"prompt": "A robot arm must pick up an egg without breaking it. Describe the control strategy including: force sensing, grip pressure curve, approach trajectory, and failure recovery.", "expected_pattern": "(?:force.*control|compliance|gentle|threshold|tactile|impedance|grip.*adjust)", "weight": 1.5},
        {"prompt": "Plan a path for a wheeled robot through a room with: a table (center), two chairs (left), a sleeping cat (right, might move). Include contingencies.", "expected_pattern": "(?:path|obstacle|avoid|contingency|replan|sensor|cat.*move|dynamic)", "weight": 1.0},
        {"prompt": "A humanoid robot must learn to walk on ice. What changes from walking on concrete? Describe the physics, control adjustments, and learning signal.", "expected_pattern": "(?:friction|slip|balance|center.*mass|shorter.*steps|cautious|feedback)", "weight": 1.0},
        {"prompt": "Design a manipulation strategy for a robot to fold a towel. Break it into subtasks, describe required sensing, and identify the hardest subtask and why.", "expected_pattern": "(?:grasp|fold|corner|deformable|cloth|perception|hardest.*align)", "weight": 1.5},
        {"prompt": "A drone must deliver a package to a balcony in wind. Describe the control loop, wind compensation strategy, and landing sequence.", "expected_pattern": "(?:PID|wind.*compensat|hover|GPS|IMU|approach|descend|turbulence)", "weight": 1.0},
    ],
    UIBDimension.EFFICIENCY: [
        {"prompt": "Explain the concept of Kolmogorov complexity and its relationship to intelligence. Use Schmidhubers compression-based theory of intelligence.", "expected_pattern": "(?:compress|shortest.*program|algorithmic|Schmidhuber|beauty|simplicity)", "weight": 2.0},
        {"prompt": "Given Model A scores 90/100 using $10/query and Model B scores 82/100 using $0.001/query. Which is more intelligent? Formalize your answer.", "expected_pattern": "(?:efficiency|cost.*adjust|normalize|per.*dollar|Model B|ratio|resource)", "weight": 2.0},
        {"prompt": "The human brain operates on 20 watts. GPT-4 inference uses 3-10 kWh per million tokens. Compare their intelligence-per-watt and discuss implications.", "expected_pattern": "(?:orders.*magnitude|brain.*efficient|energy|biological|evolution)", "weight": 2.0},
    ],
}

async def call_model(api_key: str, api_base: str, model: str, prompt: str, timeout: float = 45.0) -> Dict:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
               "HTTP-Referer": "https://hub.stabilarity.com", "X-Title": "Stabilarity UIB Benchmark"}
    payload = {"model": model, "messages": [
        {"role": "system", "content": "You are being evaluated on an intelligence benchmark. Answer precisely and thoroughly."},
        {"role": "user", "content": prompt}
    ], "max_tokens": 500, "temperature": 0.1}
    start = time.time()
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{api_base.rstrip('/')}/chat/completions", json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    latency = (time.time() - start) * 1000
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    cost = usage.get("total_tokens", 0) * 0.00001
    return {"content": content, "latency_ms": latency, "cost_usd": cost, "tokens": usage.get("total_tokens", 0)}

def score_response(response: str, expected_pattern: str, weight: float) -> float:
    if re.search(expected_pattern, response, re.IGNORECASE):
        return min(100.0, 80.0 * weight)
    word_count = len(response.split())
    if word_count > 100: return min(60.0, 40.0 * weight)
    if word_count > 50: return min(40.0, 25.0 * weight)
    return 10.0

def save_run_to_db(result, benchmark_type: str = "uib"):
    conn = get_db()
    try:
        dims_json = json.dumps([d.model_dump() for d in result.dimensions])
        conn.execute("""INSERT OR REPLACE INTO benchmark_runs
            (run_id, model, benchmark_type, composite_score, efficiency_score,
             total_cost_usd, total_latency_ms, quick_mode, dimensions_json, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (result.run_id, result.model, benchmark_type, result.composite_score,
             result.efficiency_normalized, result.total_cost_usd, result.total_latency_ms,
             1 if result.quick_mode else 0, dims_json, result.timestamp))
        conn.commit()
    finally:
        conn.close()

def get_leaderboard(limit: int = 20, benchmark_type: str = "uib"):
    conn = get_db()
    try:
        bt = benchmark_type if benchmark_type and benchmark_type != "all" else "uib"
        rows = conn.execute("""
            SELECT model, benchmark_type,
                AVG(composite_score) as avg_score, MIN(composite_score) as min_score, MAX(composite_score) as max_score,
                AVG(total_cost_usd) as avg_cost, MIN(total_cost_usd) as min_cost, MAX(total_cost_usd) as max_cost,
                AVG(efficiency_score) as avg_efficiency, COUNT(*) as runs, MAX(timestamp) as last_run
            FROM benchmark_runs WHERE benchmark_type = ?
            GROUP BY model ORDER BY avg_score DESC LIMIT ?
        """, (bt, limit)).fetchall()
    finally:
        conn.close()
    entries = []
    for i, r in enumerate(rows):
        entries.append({"rank": i+1, "model": r["model"], "benchmark_type": r["benchmark_type"],
            "avg_score": round(r["avg_score"], 2), "min_score": round(r["min_score"], 2), "max_score": round(r["max_score"], 2),
            "avg_cost": round(r["avg_cost"], 6), "min_cost": round(r["min_cost"], 6), "max_cost": round(r["max_cost"], 6),
            "avg_efficiency": round(r["avg_efficiency"], 2), "runs": r["runs"], "last_run": r["last_run"]})
    return entries

def get_all_reports(limit: int = 50, offset: int = 0, benchmark_type: str = None):
    conn = get_db()
    try:
        if benchmark_type:
            rows = conn.execute("""SELECT run_id, model, benchmark_type, composite_score, efficiency_score,
                total_cost_usd, total_latency_ms, quick_mode, dimensions_json, timestamp
                FROM benchmark_runs WHERE benchmark_type=? ORDER BY timestamp DESC LIMIT ? OFFSET ?""",
                (benchmark_type, limit, offset)).fetchall()
            total = conn.execute("SELECT COUNT(*) as n FROM benchmark_runs WHERE benchmark_type=?", (benchmark_type,)).fetchone()["n"]
        else:
            rows = conn.execute("""SELECT run_id, model, benchmark_type, composite_score, efficiency_score,
                total_cost_usd, total_latency_ms, quick_mode, dimensions_json, timestamp
                FROM benchmark_runs ORDER BY timestamp DESC LIMIT ? OFFSET ?""", (limit, offset)).fetchall()
            total = conn.execute("SELECT COUNT(*) as n FROM benchmark_runs").fetchone()["n"]
    finally:
        conn.close()
    reports = []
    for r in rows:
        reports.append({"run_id": r["run_id"], "model": r["model"], "benchmark_type": r["benchmark_type"],
            "composite_score": round(r["composite_score"], 2), "efficiency_normalized": round(r["efficiency_score"], 2),
            "total_cost_usd": round(r["total_cost_usd"], 6), "total_latency_ms": round(r["total_latency_ms"], 1),
            "quick_mode": bool(r["quick_mode"]), "dimensions": json.loads(r["dimensions_json"]), "timestamp": r["timestamp"]})
    return {"reports": reports, "total": total, "limit": limit, "offset": offset}

def get_report_by_id(run_id: str):
    conn = get_db()
    try:
        r = conn.execute("""SELECT run_id, model, benchmark_type, composite_score, efficiency_score,
            total_cost_usd, total_latency_ms, quick_mode, dimensions_json, timestamp
            FROM benchmark_runs WHERE run_id=?""", (run_id,)).fetchone()
    finally:
        conn.close()
    if not r:
        return None
    return {"run_id": r["run_id"], "model": r["model"], "benchmark_type": r["benchmark_type"],
        "composite_score": round(r["composite_score"], 2), "efficiency_normalized": round(r["efficiency_score"], 2),
        "total_cost_usd": round(r["total_cost_usd"], 6), "total_latency_ms": round(r["total_latency_ms"], 1),
        "quick_mode": bool(r["quick_mode"]), "dimensions": json.loads(r["dimensions_json"]), "timestamp": r["timestamp"]}

async def run_dimension(api_key: str, api_base: str, model: str, dimension: UIBDimension, quick: bool = True) -> DimensionScore:
    tasks = TASK_BANKS.get(dimension, [])
    if quick:
        tasks = tasks[:5]
    scores, total_latency, total_cost, completed = [], 0, 0, 0
    for task in tasks:
        try:
            result = await call_model(api_key, api_base, model, task["prompt"])
            scores.append(score_response(result["content"], task["expected_pattern"], task["weight"]))
            total_latency += result["latency_ms"]
            total_cost += result["cost_usd"]
            completed += 1
        except Exception:
            scores.append(0)
    avg_score = sum(scores) / len(scores) if scores else 0
    return DimensionScore(dimension=dimension.value, score=round(avg_score, 2),
        tasks_completed=completed, tasks_total=len(tasks),
        avg_latency_ms=round(total_latency / max(completed, 1), 1), cost_usd=round(total_cost, 6))

async def run_full_benchmark(req) -> UIBResult:
    run_id = str(uuid.uuid4())[:8]
    dimension_results, total_cost, total_latency = [], 0, 0
    for dim in req.dimensions:
        r = await run_dimension(req.api_key, req.api_base, req.model, dim, req.quick_mode)
        dimension_results.append(r)
        total_cost += r.cost_usd
        total_latency += r.avg_latency_ms * r.tasks_completed
    weights = {"causal": 1.5, "temporal": 1.2, "social": 1.0, "multimodal": 1.0,
               "tool_creation": 1.3, "transfer": 1.0, "embodied": 0.8, "efficiency": 1.2}
    weighted_sum = sum(r.score * weights.get(r.dimension, 1.0) for r in dimension_results)
    weight_total = sum(weights.get(r.dimension, 1.0) for r in dimension_results)
    composite = round(weighted_sum / max(weight_total, 1), 2)
    efficiency = round(composite / max(total_cost, 0.0001), 2)
    benchmark_type = getattr(req, "benchmark_type", "uib") or "uib"
    result = UIBResult(model=req.model, benchmark_type=benchmark_type,
        composite_score=composite, efficiency_normalized=efficiency,
        dimensions=dimension_results, total_cost_usd=round(total_cost, 6),
        total_latency_ms=round(total_latency, 1),
        timestamp=datetime.utcnow().isoformat() + "Z", run_id=run_id, quick_mode=req.quick_mode)
    save_run_to_db(result, benchmark_type)
    return result
