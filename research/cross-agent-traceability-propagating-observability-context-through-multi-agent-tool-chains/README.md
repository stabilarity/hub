# Cross-Agent Traceability: Propagating Observability Context Through Multi-Agent Tool Chains

Code and data for the Stabilarity Research Hub article of the same title (AI Observability & Monitoring series).

## Questions

- **RQ1.** When trace context is missing at one hop of a multi-agent tool chain (in-process hand-off, agent-to-agent
  HTTP, MCP JSON-RPC, message queue, webhook), what share of the request's spans stays in the root trace, and how
  does end-to-end continuity scale with per-hop propagation coverage?
- **RQ2.** What does carrying W3C Trace Context (and tracestate and baggage) cost per hop in bytes and latency on
  each carrier?
- **RQ3.** At a given base sampling rate, what share of complete failing multi-hop traces do head sampling and
  collector tail sampling keep when hops are asynchronous?

## Data and sources

| File | Content | Source |
|---|---|---|
| (generated) | Spans from a simulated 2-agent, 6-tool chain, OpenTelemetry Python SDK, in-memory exporter | `analysis.py` part A |
| (generated) | Round-trip and propagator timings, wire bytes | `analysis.py` part B; carrier values from the W3C and OpenTelemetry examples below |
| (generated) | Monte Carlo of 20,000 requests with an assumed async latency model | `analysis.py` part C |
| `data/coverage.csv` | 13 documents scored 0/1/2 on three propagation criteria, with URL, date read and evidence | pages listed in the file (read 2026-10-21) |

Specifications used for carriers and defaults:
W3C Trace Context Level 2 (https://www.w3.org/TR/trace-context-2/),
MCP SEP-414 (https://modelcontextprotocol.io/community/seps/414-request-meta),
OpenTelemetry MCP semantic conventions (https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/mcp.md),
OpenTelemetry messaging spans (https://opentelemetry.io/docs/specs/semconv/messaging/messaging-spans/),
tail sampling processor README (https://github.com/open-telemetry/opentelemetry-collector-contrib/blob/main/processor/tailsamplingprocessor/README.md):
`decision_wait` default 30 s, `num_traces` default 50,000, decision cache size default 0.
The BatchSpanProcessor schedule delay (5,000 ms) is the default in opentelemetry-sdk 1.45.1.

## How to run

```bash
python3 -m venv venv && ./venv/bin/pip install opentelemetry-sdk==1.45.1 numpy matplotlib
./venv/bin/python analysis.py     # offline; ~70 s; writes charts/chart1.png ... chart4.png and results.json
```

Environment of the published run: AMD EPYC virtual machine (2.0 GHz, 6 vCPU), Linux 6.8, Python 3.12.3,
opentelemetry-sdk/api 1.45.1, numpy 2.3.5, matplotlib 3.11.2. Parts A, C and D are deterministic (seed
20261021; random edge drops are a hash of seed and edge key). Part B timings vary by tens of percent between runs.

## Method

- **Topology (part A).** Orchestrator root span -> planner on a thread pool (in-process hop) -> 2 research agents
  over A2A-style JSON-RPC on localhost HTTP/1.1 (W3C headers) -> 3 MCP `tools/call` each over newline-delimited
  JSON-RPC on a socket pair (context in `params._meta`, per SEP-414) -> each tool publishes a job to a queue
  (context in message headers) -> worker `process` span + `chat model` span -> webhook POST back to the orchestrator
  (A2A-style push notification). 48 spans and 21 context hops per request. The in-process failure mode is real:
  `ThreadPoolExecutor.submit` does not copy `contextvars`.
- **Metrics (A).** Share of spans in the root trace by parent-child; share reachable through parent or span link;
  distinct trace IDs per request; share of requests fully connected; analytic p^21 for random edge drops.
- **Overhead (B).** Bytes added by traceparent, +tracestate (W3C example value), +baggage (agent id, conversation id,
  tenant) as HTTP header lines, as MCP `_meta` JSON, as queue headers JSON; inject+extract cost with the SDK
  propagators (median of 5 x 20,000 calls); 2,800 interleaved localhost round trips per arm with and without full
  context (server extracts context on every request).
- **Sampling (C).** Assumed latency model (our scenario): agent reasoning lognormal median 1.5 s, MCP call 0.3 s,
  queue wait median 1 s or 10 s (sigma 1.0), optional 5% human approval of 5-60 min, worker LLM step 4 s, exporter
  delay U(0, 5 s). 3% of requests fail in one async LLM span. Strategies: parent-based head sampling; head sampling
  re-decided at each async hop; tail sampling (trace-complete timer from first span arrival, error policy plus
  probabilistic remainder) with and without a decision cache, decision_wait 5-300 s.
- **Coverage (D).** Each document scored only from the cited page: carrier (0 none, 1 generic or vendor-specific,
  2 W3C traceparent), agent/tool span semantics (0/1/2), async or indirect hops (0/1/2).

## Outputs

- `charts/chart1.png` continuity when one hop type lacks context, and under random per-hop gaps
- `charts/chart2.png` bytes added per carrier and round-trip latency with and without context
- `charts/chart3.png` complete failing traces captured vs decision_wait at a 5% base rate
- `charts/chart4.png` documentation coverage matrix
- `results.json` every number used in the article
