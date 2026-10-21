#!/usr/bin/env python3
"""Cross-Agent Traceability: propagating observability context through multi-agent tool chains.

Four analyses, all offline and deterministic in structure (fixed seeds):

A. Continuity (RQ1). A simulated multi-agent tool chain instrumented with the OpenTelemetry Python SDK
   (in-memory exporter). One request = orchestrator -> planner (in-process, thread pool) -> 2 research
   agents (A2A-style JSON-RPC over real localhost HTTP) -> 3 MCP tools/call each (JSON-RPC over a socket
   pair, context in params._meta) -> each tool publishes a job to a message queue -> a worker runs an LLM
   step -> the worker posts the result to a webhook (A2A-style push notification, real localhost HTTP).
   We switch propagation off at one hop type at a time, use span links instead of parenting at the queue
   (the OpenTelemetry messaging default), and drop propagation at random edges with probability 1 - p.
B. Overhead (RQ2). Bytes that traceparent, tracestate and baggage add to each carrier, propagator
   inject+extract cost, and round-trip latency with and without context over localhost HTTP and the
   JSON-RPC socket pair.
C. Sampling (RQ3). Monte Carlo of 20,000 requests through the same topology with async delays: capture of
   complete failing traces under parent-based head sampling, per-hop independent head sampling and
   collector tail sampling (error policy + probabilistic remainder) as a function of decision_wait and of
   the decision cache.
D. Documentation coverage matrix of protocols, conventions and agent frameworks (data/coverage.csv).

Run:  python3 analysis.py      (writes charts/chart1-4.png and results.json)
"""
import contextvars
import csv
import json
import platform
import queue
import socket
import statistics
import threading
import time
import timeit
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import http.client
import random

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from opentelemetry import baggage, context as otctx, trace
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Link, SpanKind, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.baggage.propagation import W3CBaggagePropagator

SEED = 20261021
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CHARTS = HERE / "charts"
CHARTS.mkdir(exist_ok=True)
random.seed(SEED)

INK, MID, LIGHT, GRID, PALE = "#111111", "#555555", "#bbbbbb", "#eeeeee", "#dddddd"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MID, "axes.labelcolor": INK,
    "xtick.color": MID, "ytick.color": MID, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False,
})

# ------------------------------------------------------------------------------------------ telemetry
EXPORTER = InMemorySpanExporter()
PROVIDER = TracerProvider()
PROVIDER.add_span_processor(SimpleSpanProcessor(EXPORTER))
TRACER = PROVIDER.get_tracer("cross-agent-traceability")
PROP = CompositePropagator([TraceContextTextMapPropagator(), W3CBaggagePropagator()])

HOPS = ["inproc", "a2a", "mcp", "queue", "webhook"]
N_AGENTS, N_TOOLS = 2, 3


class Policy:
    """Which hops carry context. With p < 1 each edge instance is dropped with probability 1 - p; the draw is
    a deterministic function of (seed, edge key), so results do not depend on thread scheduling."""

    def __init__(self, off=(), p=1.0, queue_mode="parent", seed=SEED):
        self.off, self.p, self.queue_mode, self.seed = set(off), p, queue_mode, seed

    def carries(self, hop, key=""):
        if hop in self.off:
            return False
        if self.p < 1.0:
            return random.Random("%d|%s|%s" % (self.seed, hop, key)).random() < self.p
        return True


POLICY = Policy()


def inject(hop, key=""):
    carrier = {}
    if POLICY.carries(hop, key):
        PROP.inject(carrier)
    return carrier


def span(name, rid, kind=SpanKind.INTERNAL, parent_ctx=None, links=None):
    s = TRACER.start_as_current_span(name, context=parent_ctx, kind=kind, links=links or [])
    return s, rid


def set_rid(sp, rid, service):
    sp.set_attribute("app.request_id", rid)
    sp.set_attribute("service.name", service)


# ---- webhook receiver (orchestrator side, A2A-style push notification) and A2A server (research agent)
def _read_json(handler):
    n = int(handler.headers.get("Content-Length", 0))
    return json.loads(handler.rfile.read(n) or b"{}")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def setup(self):
        super().setup()
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)   # avoid 40 ms delayed-ACK stalls

    def do_POST(self):
        body = _read_json(self)
        carrier = {k.lower(): v for k, v in self.headers.items()}
        parent = PROP.extract(carrier, context=otctx.Context())
        rid = body.get("rid", "")
        if self.path == "/a2a":
            with TRACER.start_as_current_span("invoke_agent research", context=parent, kind=SpanKind.SERVER) as sp:
                set_rid(sp, rid, "research-agent")
                research_agent(rid, body.get("agent", 0), body.get("fail"))
            out = b'{"jsonrpc":"2.0","id":1,"result":{"status":"completed"}}'
        elif self.path == "/webhook":
            with TRACER.start_as_current_span("receive push_notification", context=parent, kind=SpanKind.SERVER) as sp:
                set_rid(sp, rid, "orchestrator")
            out = b"{}"
        else:
            out = b"{}"
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


SERVER = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
PORT = SERVER.server_address[1]
threading.Thread(target=SERVER.serve_forever, daemon=True).start()
_local = threading.local()


def http_post(path, body, headers):
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = _local.conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=10)
        conn.connect()
        conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    data = json.dumps(body).encode()
    h = {"Content-Type": "application/json", "Content-Length": str(len(data))}
    h.update(headers)
    conn.request("POST", path, body=data, headers=h)
    r = conn.getresponse()
    return r.read()


# ---- MCP tool server: newline-delimited JSON-RPC over a socket pair (stdio-like), context in params._meta
JOBS = queue.Queue()
DONE = queue.Queue()


def mcp_server(sock):
    f = sock.makefile("rwb")
    for line in f:
        req = json.loads(line)
        params = req.get("params", {})
        meta = params.get("_meta", {})
        parent = PROP.extract(meta, context=otctx.Context())
        rid = params.get("arguments", {}).get("rid", "")
        if req.get("method") == "tools/call":
            with TRACER.start_as_current_span("tools/call " + params.get("name", ""), context=parent,
                                              kind=SpanKind.SERVER) as sp:
                set_rid(sp, rid, "tool-server")
                with TRACER.start_as_current_span("send jobs", kind=SpanKind.PRODUCER) as ps:
                    set_rid(ps, rid, "tool-server")
                    job_key = "%s|%s" % (rid, params["arguments"].get("job", ""))
                    headers = inject("queue", job_key)
                    JOBS.put({"headers": headers, "rid": rid, "key": job_key, "fail": params["arguments"].get("fail")})
            resp = {"jsonrpc": "2.0", "id": req["id"], "result": {"content": [{"type": "text", "text": "queued"}]}}
        else:  # ping, used by the latency benchmark
            resp = {"jsonrpc": "2.0", "id": req["id"], "result": {}}
        f.write((json.dumps(resp, separators=(",", ":")) + "\n").encode())
        f.flush()


def start_mcp_pair():
    a, b = socket.socketpair()
    threading.Thread(target=mcp_server, args=(b,), daemon=True).start()
    return a.makefile("rwb")


def mcp_call(f, name, args, meta, rid_counter=[0]):
    rid_counter[0] += 1
    params = {"name": name, "arguments": args}
    if meta:
        params["_meta"] = meta
    msg = {"jsonrpc": "2.0", "id": rid_counter[0], "method": "tools/call", "params": params}
    f.write((json.dumps(msg, separators=(",", ":")) + "\n").encode())
    f.flush()
    return json.loads(f.readline())


# ---- queue worker: consumes jobs, runs an "LLM" step, posts a webhook
def worker():
    while True:
        job = JOBS.get()
        if job is None:
            break
        creation = PROP.extract(job["headers"], context=otctx.Context())
        rid = job["rid"]
        link_ctx = trace.get_current_span(creation).get_span_context()
        if POLICY.queue_mode == "link":
            parent, links = otctx.Context(), ([Link(link_ctx)] if link_ctx.is_valid else [])
        else:
            parent, links = creation, []
        with TRACER.start_as_current_span("process jobs", context=parent, kind=SpanKind.CONSUMER, links=links) as sp:
            set_rid(sp, rid, "worker")
            with TRACER.start_as_current_span("chat model", kind=SpanKind.CLIENT) as llm:
                set_rid(llm, rid, "worker")
                if job.get("fail"):
                    llm.set_status(Status(StatusCode.ERROR, "tool output rejected"))
            with TRACER.start_as_current_span("send push_notification", kind=SpanKind.CLIENT) as wh:
                set_rid(wh, rid, "worker")
                http_post("/webhook", {"rid": rid}, inject("webhook", job["key"]))
        DONE.put(rid)


_mcp_files = {}


def research_agent(rid, agent, fail):
    tid = threading.get_ident()
    if tid not in _mcp_files:
        _mcp_files[tid] = start_mcp_pair()
    f = _mcp_files[tid]
    for t in range(N_TOOLS):
        with TRACER.start_as_current_span("tools/call search_%d" % t, kind=SpanKind.CLIENT) as sp:
            set_rid(sp, rid, "research-agent")
            f_flag = 1 if (fail is not None and fail == agent * N_TOOLS + t) else 0
            mcp_call(f, "search_%d" % t, {"q": "x", "rid": rid, "fail": f_flag, "job": "%d-%d" % (agent, t)},
                     inject("mcp", "%s|%d-%d" % (rid, agent, t)))


def planner(rid, fail):
    with TRACER.start_as_current_span("invoke_agent planner") as sp:
        set_rid(sp, rid, "orchestrator")
        for a in range(N_AGENTS):
            with TRACER.start_as_current_span("send_message research_%d" % a, kind=SpanKind.CLIENT) as c:
                set_rid(c, rid, "orchestrator")
                http_post("/a2a", {"rid": rid, "agent": a, "fail": fail}, inject("a2a", "%s|%d" % (rid, a)))


POOL = ThreadPoolExecutor(max_workers=1)


def run_request(rid, fail=None):
    with TRACER.start_as_current_span("invoke_agent orchestrator", kind=SpanKind.SERVER) as root:
        set_rid(root, rid, "orchestrator")
        if POLICY.carries("inproc", rid):
            ctx = contextvars.copy_context()
            POOL.submit(ctx.run, planner, rid, fail).result()
        else:  # ThreadPoolExecutor does not copy contextvars: the planner starts a new trace
            POOL.submit(planner, rid, fail).result()
    for _ in range(N_AGENTS * N_TOOLS):
        DONE.get(timeout=10)
    return root.get_span_context().trace_id


def analyse_request(rid, root_tid, spans):
    mine = [s for s in spans if s.attributes.get("app.request_id") == rid]
    by_parent = sum(1 for s in mine if s.context.trace_id == root_tid)
    # traces reachable from the root trace through span links (one level is enough in this topology)
    linked = {root_tid}
    for s in mine:
        for l in s.links:
            if l.context.trace_id in linked:
                linked.add(s.context.trace_id)
    by_link = sum(1 for s in mine if s.context.trace_id in linked)
    return len(mine), by_parent, by_link, len({s.context.trace_id for s in mine})


WORKER = threading.Thread(target=worker, daemon=True)
WORKER.start()


def run_config(policy, n, tag):
    global POLICY
    POLICY = policy
    EXPORTER.clear()
    roots = []
    for i in range(n):
        rid = "%s-%d" % (tag, i)
        roots.append((rid, run_request(rid)))
    time.sleep(0.05)
    spans = EXPORTER.get_finished_spans()
    rows = [analyse_request(rid, tid, spans) for rid, tid in roots]
    tot = sum(r[0] for r in rows)
    return {
        "spans_per_request": rows[0][0],
        "share_in_root_trace_by_parent_pct": round(100 * sum(r[1] for r in rows) / tot, 2),
        "share_reachable_parent_or_link_pct": round(100 * sum(r[2] for r in rows) / tot, 2),
        "fragments_per_request": round(statistics.mean(r[3] for r in rows), 2),
        "fully_connected_requests_pct": round(100 * sum(1 for r in rows if r[1] == r[0]) / len(rows), 2),
    }


def part_a():
    configs = [("all hops propagate", Policy()),
               ("in-process: thread pool without context copy", Policy(off=["inproc"])),
               ("A2A HTTP headers missing", Policy(off=["a2a"])),
               ("MCP params._meta missing", Policy(off=["mcp"])),
               ("queue message headers missing", Policy(off=["queue"])),
               ("webhook headers missing", Policy(off=["webhook"])),
               ("queue consumer uses link, not parent", Policy(queue_mode="link")),
               ("no hop propagates", Policy(off=HOPS))]
    single = {}
    for name, pol in configs:
        single[name] = run_config(pol, 20, "cfg%d" % len(single))
    edges = 1 + N_AGENTS + 3 * N_AGENTS * N_TOOLS
    rand = {}
    for p in [0.80, 0.90, 0.95, 0.98, 0.99, 1.00]:
        r = run_config(Policy(p=p, seed=SEED + int(round(p * 1000))), 200, "p%d" % int(p * 100))
        r["analytic_fully_connected_pct"] = round(100 * p ** edges, 2)
        rand["%.2f" % p] = r
    return {"configs": single, "random_edge_coverage": rand, "context_edges_per_request": edges,
            "requests_per_config": 20, "requests_per_coverage_level": 200}


# ----------------------------------------------------------------------------------------- B overhead
def part_b():
    tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    ts = "rojo=00f067aa0ba902b7,congo=t61rcWkgMzE"
    bg = "gen_ai.agent.id=research-agent-1,gen_ai.conversation.id=conv_5j66UpCpwteGg4YSxUnt7lPY,tenant=t-042"
    levels = {"traceparent": {"traceparent": tp},
              "traceparent+tracestate": {"traceparent": tp, "tracestate": ts},
              "traceparent+tracestate+baggage": {"traceparent": tp, "tracestate": ts, "baggage": bg}}

    def http_bytes(h):
        return sum(len(("%s: %s\r\n" % (k, v)).encode()) for k, v in h.items())

    def mcp_msg(meta):
        params = {"name": "get_weather", "arguments": {"location": "New York"}}
        if meta:
            params["_meta"] = meta
        return len(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": params},
                              separators=(",", ":")).encode())

    def queue_msg(h):
        return len(json.dumps({"headers": h, "body": {"task": "summarise", "doc": "s3://bucket/key"}},
                              separators=(",", ":")).encode())

    a2a_body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "SendMessage", "params": {"message": {
        "role": "ROLE_USER", "parts": [{"text": "Find recent papers on trace sampling"}],
        "messageId": "9229e770-767c-417b-a0b0-f0741243c589"}}}, separators=(",", ":"))
    a2a_base = len(a2a_body.encode()) + http_bytes({"Host": "agent.example.com", "Content-Type": "application/json",
                                                    "Content-Length": "200", "A2A-Version": "1.0"}) + len(
        "POST /a2a HTTP/1.1\r\n\r\n")
    bytes_out = {"mcp_tools_call_baseline_bytes": mcp_msg(None), "a2a_http_request_baseline_bytes": a2a_base,
                 "queue_message_baseline_bytes": queue_msg({}), "carriers": {}}
    for lvl, h in levels.items():
        bytes_out["carriers"][lvl] = {
            "http_header_bytes": http_bytes(h),
            "mcp_meta_added_bytes": mcp_msg(h) - mcp_msg(None),
            "queue_header_added_bytes": queue_msg(h) - queue_msg({}),
            "mcp_added_pct_of_tools_call": round(100 * (mcp_msg(h) - mcp_msg(None)) / mcp_msg(None), 1),
            "http_added_pct_of_a2a_request": round(100 * http_bytes(h) / a2a_base, 1),
        }

    # propagator cost (inject + extract) inside an active span with baggage
    ctx = baggage.set_baggage("gen_ai.agent.id", "research-agent-1")
    ctx = baggage.set_baggage("gen_ai.conversation.id", "conv_5j66UpCpwteGg4YSxUnt7lPY", ctx)
    tok = otctx.attach(ctx)
    sp = TRACER.start_span("bench")
    tok2 = otctx.attach(trace.set_span_in_context(sp))
    tcp = TraceContextTextMapPropagator()

    def inj_ext_tc():
        c = {}
        tcp.inject(c)
        tcp.extract(c)

    def inj_ext_all():
        c = {}
        PROP.inject(c)
        PROP.extract(c)

    def new_span():
        with TRACER.start_as_current_span("child"):
            pass

    n = 20000
    cost = {}
    for name, fn in [("traceparent_inject_extract_us", inj_ext_tc), ("tracecontext_plus_baggage_inject_extract_us", inj_ext_all),
                     ("create_and_end_child_span_us", new_span)]:
        reps = [timeit.timeit(fn, number=n) / n * 1e6 for _ in range(5)]
        cost[name] = round(statistics.median(reps), 2)
    otctx.detach(tok2)
    sp.end()
    otctx.detach(tok)
    EXPORTER.clear()

    # round trips with and without context (interleaved), HTTP and JSON-RPC socket pair
    hdr = {"traceparent": tp, "tracestate": ts, "baggage": bg}
    f = start_mcp_pair()

    def ping(meta):
        msg = {"jsonrpc": "2.0", "id": 1, "method": "ping", "params": ({"_meta": meta} if meta else {})}
        f.write((json.dumps(msg, separators=(",", ":")) + "\n").encode())
        f.flush()
        f.readline()

    rt = {"http_without": [], "http_with": [], "jsonrpc_without": [], "jsonrpc_with": []}
    for i in range(3000):
        for key, fn in [("http_without", lambda: http_post("/noop", {"rid": "x"}, {})),
                        ("http_with", lambda: http_post("/noop", {"rid": "x"}, hdr)),
                        ("jsonrpc_without", lambda: ping(None)), ("jsonrpc_with", lambda: ping(hdr))]:
            t0 = time.perf_counter()
            fn()
            if i >= 200:
                rt[key].append((time.perf_counter() - t0) * 1e6)
    lat = {k + "_median_us": round(statistics.median(v), 1) for k, v in rt.items()}
    lat["http_delta_median_us"] = round(lat["http_with_median_us"] - lat["http_without_median_us"], 1)
    lat["jsonrpc_delta_median_us"] = round(lat["jsonrpc_with_median_us"] - lat["jsonrpc_without_median_us"], 1)
    lat["samples_per_arm"] = len(rt["http_with"])
    return {"bytes": bytes_out, "propagator_cost": cost, "round_trip": lat, "values": {
        "traceparent": tp, "tracestate": ts, "baggage": bg}}


# ----------------------------------------------------------------------------------------- C sampling
def simulate_traces(n, rng, queue_median_s, hitl_share=0.0):
    """Span end times (s, relative to request start) and error flags for the 48-span topology.

    Returns per request: arrays of span arrival time at the collector, span 'segment' id
    (0 = synchronous part, 1..6 = async job k) and the index of the error span (or -1).
    Assumed latency model (our scenario, not measured): A2A agent reasoning lognormal median 1.5 s
    (sigma 0.5), MCP call median 0.3 s (0.6), queue wait median Q (sigma 1.0), worker LLM step median 4 s (0.7),
    webhook 0.05 s; exporter delay U(0, 5 s) (BatchSpanProcessor default schedule delay 5,000 ms).
    """
    out = []
    for _ in range(n):
        ends, seg = [], []
        t_agents_done = []
        for a in range(N_AGENTS):
            t = 0.01 + rng.lognormal(np.log(1.5), 0.5)
            for k in range(N_TOOLS):
                d = rng.lognormal(np.log(0.3), 0.6)
                t_pub = t + d * 0.5
                ends += [t + d, t + d * 0.9, t_pub]          # MCP client, MCP server, publish
                seg += [0, 0, 0]
                job = a * N_TOOLS + k + 1
                q = rng.lognormal(np.log(queue_median_s), 1.0)
                if rng.random() < hitl_share:
                    q += rng.uniform(300, 3600)              # human approval before the job runs
                start = t_pub + q
                llm = rng.lognormal(np.log(4.0), 0.7)
                ends += [start + llm + 0.06, start + llm, start + llm + 0.05, start + llm + 0.05]
                seg += [job] * 4                             # process, chat model, webhook client/server
                t = t + d
            ends += [t + 0.01, t + 0.01]                     # A2A server, A2A client
            seg += [0, 0]
            t_agents_done.append(t)
        tr = max(t_agents_done) + 0.2
        ends += [tr - 0.1, tr]                               # planner, root
        seg += [0, 0]
        ends = np.array(ends)
        arrival = ends + rng.uniform(0, 5.0, size=ends.size)
        out.append((arrival, np.array(seg)))
    return out


def tail_capture(traces, fail_idx, r, wait, cache, rng):
    """Collector tail sampling, trace-complete strategy: decision at first arrival + wait over spans seen.

    Policy: keep if any seen span has an error, else keep with probability r. Spans arriving after the
    decision: with a decision cache they follow the cached decision; without one (default size 0) they
    form a new trace evaluated by the same policy. Returns share of failing traces captured complete,
    share with the error span stored at all, and share of all spans stored.
    """
    complete = err_kept = 0
    stored = total = 0
    for (arr, seg), fi in zip(traces, fail_idx):
        t_dec = arr.min() + wait
        early = arr <= t_dec
        has_err_early = fi >= 0 and early[fi]
        keep = has_err_early or rng.random() < r
        kept = early & keep
        late = ~early
        if late.any():
            if cache:
                kept = kept | (late & keep)
            else:
                late_err = fi >= 0 and late[fi]
                late_keep = late_err or rng.random() < r
                kept = kept | (late & late_keep)
        total += arr.size
        stored += int(kept.sum())
        if fi >= 0:
            complete += int(kept.all())
            err_kept += int(kept[fi])
    nf = int((fail_idx >= 0).sum())
    return round(100 * complete / nf, 2), round(100 * err_kept / nf, 2), round(100 * stored / total, 2)


def head_capture(traces, fail_idx, r, rng, independent):
    complete = err_kept = 0
    stored = total = 0
    for (arr, seg), fi in zip(traces, fail_idx):
        if independent:   # the sampled flag does not cross the queue: each async job decides again
            dec = {s: rng.random() < r for s in np.unique(seg)}
            kept = np.array([dec[s] for s in seg])
        else:
            kept = np.full(arr.size, rng.random() < r)
        total += arr.size
        stored += int(kept.sum())
        if fi >= 0:
            complete += int(kept.all())
            err_kept += int(kept[fi])
    nf = int((fail_idx >= 0).sum())
    return round(100 * complete / nf, 2), round(100 * err_kept / nf, 2), round(100 * stored / total, 2)


def part_c():
    rng = np.random.default_rng(SEED)
    n, fail_rate = 20000, 0.03
    res = {"requests": n, "failing_share_pct": round(100 * fail_rate, 1), "scenarios": {}}
    for qname, qmed, hitl in [("queue median 1 s", 1.0, 0.0), ("queue median 10 s", 10.0, 0.0),
                              ("queue median 10 s + 5% human approval", 10.0, 0.05)]:
        tr = simulate_traces(n, rng, qmed, hitl)
        # error span = the chat model span of a random async job (index 4 within each 7-span tool block)
        fail_idx = np.full(n, -1)
        failing = rng.random(n) < fail_rate
        for i in np.where(failing)[0]:
            a, k = rng.integers(N_AGENTS), rng.integers(N_TOOLS)
            fail_idx[i] = a * (N_TOOLS * 7 + 2) + k * 7 + 4
        durations = np.array([a.max() - a.min() for a, _ in tr])
        sc = {"trace_span_arrival_window_p50_s": round(float(np.median(durations)), 1),
              "trace_span_arrival_window_p95_s": round(float(np.percentile(durations, 95)), 1),
              "by_rate": {}}
        for r in [0.01, 0.05, 0.10]:
            rr = {"head_parent_based": head_capture(tr, fail_idx, r, rng, False),
                  "head_independent_per_async_hop": head_capture(tr, fail_idx, r, rng, True)}
            for w in [5, 10, 30, 60, 120, 300]:
                rr["tail_wait_%ds_no_cache" % w] = tail_capture(tr, fail_idx, r, w, False, rng)
                rr["tail_wait_%ds_cache" % w] = tail_capture(tr, fail_idx, r, w, True, rng)
            sc["by_rate"]["%.2f" % r] = rr
        res["scenarios"][qname] = sc
    res["metric_order"] = ["complete_failing_traces_pct", "error_span_stored_pct", "all_spans_stored_pct"]
    # collector memory arithmetic (our arithmetic on documented defaults)
    res["tail_defaults"] = {"num_traces": 50000, "decision_wait_s": 30,
                            "max_new_traces_per_s_before_eviction": round(50000 / 30, 1)}
    return res


# ----------------------------------------------------------------------------------------- D coverage
def part_d():
    rows = list(csv.DictReader(open(DATA / "coverage.csv")))
    out = []
    for r in rows:
        c = {k: int(r[k]) for k in ["carrier", "agent_tool_spans", "async_hops"]}
        out.append({"item": r["item"], "kind": r["kind"], **c, "total": sum(c.values()), "url": r["url"]})
    return out


# ----------------------------------------------------------------------------------------- charts
def chart1(a):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8.2), dpi=110, gridspec_kw={"width_ratios": [1.35, 1]})
    names = list(a["configs"].keys())
    par = [a["configs"][k]["share_in_root_trace_by_parent_pct"] for k in names]
    lnk = [a["configs"][k]["share_reachable_parent_or_link_pct"] for k in names]
    y = np.arange(len(names))[::-1]
    ax1.barh(y + 0.19, par, 0.36, color=INK, label="in root trace (parent-child)")
    ax1.barh(y - 0.19, lnk, 0.36, color=LIGHT, label="reachable via parent or span link")
    for yi, v, f in zip(y, par, [a["configs"][k]["fragments_per_request"] for k in names]):
        ax1.text(v + 1, yi + 0.19, "%.1f%%  (%.0f traces/request)" % (v, f), va="center", fontsize=9, color=INK)
    ax1.set_yticks(y, names, fontsize=10)
    ax1.set_xlim(0, 135)
    ax1.set_xticks([0, 25, 50, 75, 100])
    ax1.set_xlabel("Share of the request's %d spans, %%" % a["configs"][names[0]]["spans_per_request"])
    ax1.set_title("One hop type without context", loc="left", fontsize=12)
    ax1.legend(loc="lower right", frameon=False, fontsize=9)
    ps = sorted(a["random_edge_coverage"].keys())
    x = [float(p) for p in ps]
    meas = [a["random_edge_coverage"][p]["fully_connected_requests_pct"] for p in ps]
    share = [a["random_edge_coverage"][p]["share_in_root_trace_by_parent_pct"] for p in ps]
    xx = np.linspace(0.8, 1.0, 100)
    ax2.plot(xx, 100 * xx ** a["context_edges_per_request"], color=MID, lw=2, ls="--",
             label="analytic p^%d" % a["context_edges_per_request"])
    ax2.plot(x, meas, "o", color=INK, ms=8, label="measured: request fully connected")
    ax2.plot(x, share, "s-", color=LIGHT, ms=8, lw=2, mec=MID, label="measured: spans in root trace")
    ax2.set_xlabel("Probability p that each context hop carries context")
    ax2.set_ylabel("% (200 requests per point)")
    ax2.set_ylim(0, 105)
    ax2.set_title("Random gaps across %d context hops" % a["context_edges_per_request"], loc="left", fontsize=12)
    ax2.legend(loc="upper left", frameon=False, fontsize=9)
    import importlib.metadata as md
    fig.suptitle("Trace continuity in a simulated 2-agent, 6-tool chain (OpenTelemetry Python SDK %s)"
                 % md.version("opentelemetry-sdk"), x=0.01, ha="left", fontsize=14, color=INK)
    fig.text(0.01, 0.01, "Source: stabilarity/hub research/cross-agent-traceability..., analysis.py, results.json",
             fontsize=8, color=MID)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(CHARTS / "chart1.png", dpi=150)
    plt.close(fig)


def chart2(b):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=110)
    lv = list(b["bytes"]["carriers"].keys())
    keys = [("http_header_bytes", "HTTP header lines (A2A, webhook)"), ("mcp_meta_added_bytes", "MCP params._meta (JSON)"),
            ("queue_header_added_bytes", "queue message headers (JSON)")]
    x = np.arange(len(lv))
    cols = [INK, MID, LIGHT]
    for i, (k, lab) in enumerate(keys):
        vals = [b["bytes"]["carriers"][l][k] for l in lv]
        ax1.bar(x + (i - 1) * 0.26, vals, 0.24, color=cols[i], label=lab)
        for xi, v in zip(x, vals):
            ax1.text(xi + (i - 1) * 0.26, v + 3, str(v), ha="center", fontsize=9, color=INK)
    ax1.set_xticks(x, ["traceparent", "+ tracestate", "+ baggage"])
    ax1.set_ylabel("Bytes added per message")
    ax1.set_title("Wire bytes added by context, per carrier", loc="left", fontsize=12)
    ax1.legend(frameon=False, fontsize=9, loc="upper left")
    rt = b["round_trip"]
    labels = ["HTTP/1.1\nwithout", "HTTP/1.1\nwith", "JSON-RPC pair\nwithout", "JSON-RPC pair\nwith"]
    vals = [rt["http_without_median_us"], rt["http_with_median_us"], rt["jsonrpc_without_median_us"],
            rt["jsonrpc_with_median_us"]]
    ax2.bar(range(4), vals, 0.6, color=[LIGHT, INK, LIGHT, INK])
    for i, v in enumerate(vals):
        ax2.text(i, v + max(vals) * 0.01, "%.1f" % v, ha="center", fontsize=10, color=INK)
    ax2.set_xticks(range(4), labels)
    ax2.set_ylabel("Median round trip, microseconds (localhost)")
    ax2.set_title("Round trip with full context (traceparent, tracestate, baggage)", loc="left", fontsize=12)
    fig.suptitle("Per-hop cost of carrying trace context", x=0.01, ha="left", fontsize=14, color=INK)
    fig.text(0.01, 0.01, "Source: our measurement, analysis.py / results.json; %d interleaved samples per arm"
             % rt["samples_per_arm"], fontsize=8, color=MID)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(CHARTS / "chart2.png", dpi=150)
    plt.close(fig)


def chart3(c):
    fig, axes = plt.subplots(1, 3, figsize=(17, 7.5), dpi=110, sharey=True)
    waits = [5, 10, 30, 60, 120, 300]
    for ax, (sname, sc) in zip(axes, c["scenarios"].items()):
        rr = sc["by_rate"]["0.05"]
        nc = [rr["tail_wait_%ds_no_cache" % w][0] for w in waits]
        ca = [rr["tail_wait_%ds_cache" % w][0] for w in waits]
        ax.plot(waits, ca, "o-", color=INK, lw=2, ms=8, label="tail, decision cache on")
        ax.plot(waits, nc, "s--", color=MID, lw=2, ms=8, label="tail, no decision cache (default)")
        ax.axhline(rr["head_parent_based"][0], color=LIGHT, lw=2, label="head 5%, parent-based")
        ax.axhline(rr["head_independent_per_async_hop"][0], color=LIGHT, lw=2, ls=":",
                   label="head 5%, re-decided per async hop")
        ax.axvline(30, color=PALE, lw=1)
        ax.set_xscale("log")
        ax.set_xticks(waits, [str(w) for w in waits])
        ax.set_xlabel("decision_wait, seconds (log scale)")
        ax.set_title(sname + "\n(span arrival window p95 %.0f s)" % sc["trace_span_arrival_window_p95_s"],
                     loc="left", fontsize=11)
    axes[0].set_ylabel("Failing traces captured complete, %")
    axes[0].set_ylim(-2, 102)
    axes[0].legend(frameon=False, fontsize=9, loc="center right")
    fig.suptitle("Capture of complete failing multi-hop traces at a 5% base sampling rate (20,000 simulated requests)",
                 x=0.01, ha="left", fontsize=14, color=INK)
    fig.text(0.01, 0.01, "Source: our Monte Carlo, analysis.py / results.json; latency model is an assumed scenario; "
             "grey vertical line = collector default decision_wait 30 s", fontsize=8, color=MID)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(CHARTS / "chart3.png", dpi=150)
    plt.close(fig)


def chart4(d):
    fig, ax = plt.subplots(figsize=(14, 8.5), dpi=110)
    cols = ["carrier", "agent_tool_spans", "async_hops"]
    m = np.array([[r[k] for k in cols] for r in d])
    from matplotlib.colors import ListedColormap
    ax.imshow(m, cmap=ListedColormap(["#f3f3f3", "#bbbbbb", "#111111"]), vmin=0, vmax=2, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, ["absent", "partial", "specified"][m[i, j]], ha="center", va="center", fontsize=10,
                    color="white" if m[i, j] == 2 else INK)
    ax.set_xticks(range(3), ["Cross-process carrier\n(2 = W3C traceparent)", "Agent / tool span\nsemantics",
                             "Async or indirect hops\n(queue, push, hosted tool, HITL)"])
    ax.set_yticks(range(len(d)), ["%s  [%d/6]" % (r["item"], r["total"]) for r in d], fontsize=10)
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.grid(False)
    ax.set_title("What each document specifies about trace-context propagation (scored from the cited page, Oct 2026)",
                 loc="left", fontsize=13, pad=60)
    fig.text(0.01, 0.01, "Source: data/coverage.csv (URL and evidence per cell); 0 absent, 1 partial or vendor-specific,"
             " 2 specified", fontsize=8, color=MID)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(CHARTS / "chart4.png", dpi=150)
    plt.close(fig)


def main():
    t0 = time.time()
    a = part_a()
    b = part_b()
    c = part_c()
    d = part_d()
    import importlib.metadata as md
    env = {"python": platform.python_version(), "platform": platform.platform(),
           "cpu": "AMD EPYC (virtual, 2.0 GHz), %d vCPU" % (len(__import__("os").sched_getaffinity(0))),
           "opentelemetry_sdk": md.version("opentelemetry-sdk"), "opentelemetry_api": md.version("opentelemetry-api"),
           "numpy": np.__version__, "matplotlib": matplotlib.__version__, "seed": SEED}
    res = {"environment": env, "A_continuity": a, "B_overhead": b, "C_sampling": c, "D_coverage": d,
           "runtime_s": round(time.time() - t0, 1)}
    json.dump(res, open(HERE / "results.json", "w"), indent=2)
    chart1(a)
    chart2(b)
    chart3(c)
    chart4(d)
    JOBS.put(None)
    print(json.dumps(res, indent=1)[:6000])


if __name__ == "__main__":
    main()
