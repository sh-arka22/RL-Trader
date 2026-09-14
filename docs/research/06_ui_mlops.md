# 06 — Live UI, experiment tracking and training compute

**Author:** deep-research subagent `ui-mlops` · **Date of all live fetches: 2026-09-14** ·
**Protocol:** `docs/RESEARCH_PROTOCOL.md` (no unverified citations; every number carries its context).

> **Scope.** Part A live force-directed "EKG" visualisation · Part B streaming transport ·
> Part C charts · Part D build-vs-buy for the UI · Part E Weights & Biases + HPO ·
> Part F training compute on Prime Intellect.
> A **Verification log** is at the end. Anything I could not verify in this session is marked
> `UNVERIFIED` inline and is not used to support a recommendation.

---

## 0. Decisions at a glance

| Part | Primary recommendation | Fallback | Hard constraint that drove it |
|---|---|---|---|
| A — graph | **`@cosmos.gl/graph` (cosmos.gl), MIT, GPU force layout + WebGL render** | **Sigma.js 3 + Graphology** with a *frozen* layout and a ≤5k display subgraph | CPU force layout dies at ~100k nodes: graphology FA2 does **1.61 layout iterations/second** at 100k nodes (viz-bench, M5 Pro) vs **39.7 it/s** for cosmos.gl GPU |
| A — licence | **Use `@cosmos.gl/graph` (MIT), NOT `@cosmograph/cosmos`** | — | `@cosmograph/cosmos@3.4.1` on the npm registry is **CC-BY-NC-4.0** (non-commercial). `@cosmos.gl/graph@3.4.1` is **MIT** |
| B — transport | **One WebSocket per session + server-side 20 Hz coalescing frame pump + bounded per-client queue** | SSE for a low-rate read-only status lane | `websockets` library gives no automatic per-client backpressure on broadcast; bufferbloat is documented |
| C — equity curve | **TradingView `lightweight-charts` 5.2.1 (Apache-2.0 + mandatory attribution)** | uPlot (MIT, no attribution) if the attribution notice is unacceptable | Apache-2.0 here is **not** attribution-free — see C.2 |
| C — learning curves | **uPlot 1.6.32 (MIT, ~533 KB unpacked)** | Apache ECharts 6.1.0 if heatmaps/parallel-coords are needed | Many series, dense points, small bundle |
| D — UI shell | **Custom React + Vite + FastAPI** (one small app) | HoloViz Panel with a `ReactComponent` ESM escape hatch | No Python dashboard framework can own a continuously-animating 10k+ node WebGL scene without you writing the same JS component anyway |
| E — HPO | **Optuna 5.0.0 for search/pruning + W&B for logging & lineage** (W&B Sweeps only if you want zero infra) | Pure W&B Sweeps `bayes` + `hyperband` | W&B does not document its Bayesian surrogate; Optuna exposes samplers, pruners and an RDB study you can resume |
| F — compute | **CPU, not GPU.** A Prime Intellect **sandbox** at `$0.05/core/hr + $0.01/GB-RAM/hr` | Cheapest GPU pod (`A2000 $0.15/hr` community) only for a parallel sweep burst | A 5-asset daily-bar MLP PPO/SAC agent is a tiny workload; GPU kernel-launch overhead dominates |
| F — prime-rl | **Do not use prime-rl or the Environments Hub for this project** | Plain `stable-baselines3` on a pod/sandbox | `prime-rl` algorithms are `grpo/max_rl/rae/hierarchical_grpo/opd/sft/opsd/echo` — LLM-token RL. `verifiers` is "our library for creating environments to train and evaluate **LLMs**" |

### Honest negatives against the master brief

1. **The brief's "D3-force or Three.js frontend" suggestion does not scale to the stated 10k–100k
   node EKG.** d3-force is a CPU velocity-Verlet solver; its GitHub repo was last pushed
   **2023-12-30** and its last npm release is **3.0.0, published 2021-06-05**. It is stable but
   effectively dormant, and the measured CPU-layout ceiling below rules it out above ~10k
   *moving* nodes.
2. **Cosmograph — the obvious "GPU force graph" answer — is licence-poisoned.**
   `@cosmograph/cosmos` and `@cosmograph/react` are published under **CC-BY-NC-4.0**.
   A trading product is commercial by nature. Use the MIT-licensed `@cosmos.gl/graph` fork instead.
3. **Prime Intellect is the wrong place to *train* this agent, and prime-rl cannot train it at
   all.** The brief mandates cloud GPU training. For 5 assets on daily bars that is a waste of
   money: the useful Prime Intellect product here is the **CPU sandbox** and, at most, a cheap
   GPU pod for parallel sweep workers. `prime-rl` and `verifiers` are LLM-RL infrastructure and
   do not accept a Gymnasium continuous-control env.
4. **W&B's free tier is thin for a long project** (5 GB/month storage). The **free academic Pro
   licence** (200 GB, unlimited tracked hours, 100 seats) is the tier to apply for. If this is
   not academic work, the self-hosted "Personal" plan explicitly **forbids corporate use**.
5. **`encode/broadcaster`, the library most FastAPI pub/sub tutorials recommend, is ARCHIVED**
   (last push 2025-04-09). Do not build on it.
6. **Rendering ≠ layout.** Every marketing claim of "millions of nodes" I checked is a *rendering*
   claim. The force *simulation* is the bottleneck, and it is 20–25× cheaper on a GPU.

---

## Part A — Live "spider web" EKG visualisation

### A.1 The only gated benchmark I could find (primary evidence)

`GusEllerm/viz-bench` (MIT, created **2026-08-12**, last push **2026-08-20**) is a *gate-verified*
benchmark: every published cell must pass a real-GPU gate, a full-data-coverage gate, an
on-screen pixel-diff presentation gate, a recorded-camera-state gate, and an input-honesty gate,
or it is quarantined. I pulled its raw result JSON rather than trusting the report page.

**Test bed for all numbers below:** Apple M5 Pro (ANGLE Metal renderer), macOS, headed browser,
`pixelRatio 2`, 5-second coalesced pan, datasets = `ogbn-arxiv` year slices and `cit-Patents`,
**layouts precomputed** (ForceAtlas2 / GPU-seeded). `fps` = peak FPS during the driver window;
`fpsCont` = sustained/continuous FPS, which is the number that matters for a live dashboard.

#### A.1.1 Render throughput — `graph-bench/web/bench/results/overview-matrix.json`

| Library (tool) | 53k nodes / 148k edges | 91k / 369k | 121k / 615k | 169k / 1.16M | 3.77M / 16.5M |
|---|---|---|---|---|---|
| **deck.gl** (`deck`) | 120 / **120** fps, 10 MB | 120 / **120**, 54 MB | 120 / **120**, 71 MB | 120 / **66.8**, 115 MB | 14.3 / **6.2**, 476 MB |
| **helios-web** (`heliosfast`) | 120 / **120**, 38 MB | 119.8 / **120**, 68 MB | 119.8 / **119.9**, 93 MB | 120 / **91.4**, 205 MB | 36.9 / **8.2**, 3512 MB |
| **cosmos.gl** (`cosmosnoblend`) | 120 / **120**, 43 MB | 120 / **120**, 114 MB | 97.4 / **89.5**, 125 MB | 49.4 / **47.2**, 207 MB | 4.0 / **5.0**, 4206 MB |
| **Cosmograph** (`cosmographnoblend`) | 120 / **119.7**, 82 MB | 120 / **117.7**, 218 MB | 119.8 / **82.0**, 309 MB | 91.4 / **54.8**, 654 MB | 2.1 / **4.4**, 2675 MB |
| **Sigma.js** (`sigmapre`) | 120 / **120**, 99 MB | 120 / **120**, 283 MB | 120 / **78.3**, 344 MB | 120 / **39.6**, 710 MB | **gate FAILED** (`ok:false`) |

*(format: peak fps / **sustained fps**, JS heap MB. All cells `ok:true` except the last Sigma cell.)*

**Reading:** with a *precomputed* layout, every WebGL library here renders 121k nodes / 615k edges
above 60 fps sustained on an M5 Pro. Rendering is **not** the constraint at the project's target
scale. Sigma.js degrades fastest with edge count (78.3 → 39.6 fps sustained between 615k and 1.16M
edges) and uses 2–7× the memory of deck.gl. Everything collapses at ~3.8M nodes.

#### A.1.2 Layout throughput — `graph-bench/web/bench/results/layout-matrix.json` — **this is the real constraint**

Force-simulation iterations per second (higher is better; a "live, visibly rewiring" graph needs
**≥30 it/s** to look like physics rather than a slideshow):

| Nodes / edges | graphology ForceAtlas2 (CPU) | cosmos.gl (GPU) | Helios-Web (native) | GPU speed-up |
|---:|---:|---:|---:|---:|
| 10,000 / 20,000 | **32.4 it/s** | **120 it/s** | 120 it/s | 3.7× |
| 100,000 / 200,000 | **1.61 it/s** | **39.7 it/s** | 26.5 it/s | **24.7×** |
| 500,000 / 1,000,000 | **0.23 it/s** | **10.3 it/s** | 120 it/s † | 45× |
| 1,000,000 / 2,000,000 | not measured | **5.7 it/s** | 120 it/s † | — |
| 4,000,000 / 8,000,000 | not measured | **1.2 it/s** | not measured | — |

† The Helios 500k/1M cells both report exactly 120 it/s — the same value as its 10k cell — which
is the harness frame cap. Treat those two cells as **suspect**, not as evidence Helios is 12×
faster than cosmos.gl at 500k. The 10k and 100k Helios cells are internally consistent.

**This table is the single most decisive piece of evidence in this dossier.** A CPU force layout
delivers **1.6 iterations per second at 100k nodes**. That is not a live animation; that is one
frame every 0.6 seconds. d3-force is in the same architectural class as graphology FA2 (CPU,
Barnes-Hut quadtree, velocity Verlet) and will be in the same order of magnitude.

### A.2 Per-library scorecard (all metadata fetched live 2026-09-14)

GitHub metadata via authenticated GitHub REST API; package metadata via `registry.npmjs.org`.

| Library | npm pkg @ version (published) | Licence (SPDX, from registry) | Unpacked | GitHub ★ / last push / latest release / open issues | Layout engine | Incremental-update API | React |
|---|---|---|---|---:|---|---|---|
| **d3-force** | `d3-force@3.0.0` (2021-06-05) | **ISC** | 88 KB | 2,001 ★ · pushed **2023-12-30** · `v3.0.0` 2021-06-05 · 27 issues | CPU velocity-Verlet, Barnes-Hut `forceManyBody` | `simulation.nodes(arr)` + `forceLink.links()` + `alphaTarget()/restart()` — reheats, but re-inits all bound forces over the *whole* node set | none (hand-roll) |
| **force-graph** (2D canvas) | `force-graph@1.51.4` (2026-04-16) | MIT | 6.3 MB | 2,119 ★ · pushed 2026-04-16 · 152 issues | d3-force (CPU) | `graphData()` mutation + `warmupTicks`/`cooldownTicks`/`cooldownTime`/`pauseAnimation` | via wrapper |
| **3d-force-graph** (Three.js) | `3d-force-graph@1.80.0` (2026-04-05) | MIT | **14.2 MB** | 6,394 ★ · pushed 2026-04-05 · 250 issues | d3-force-3d or ngraph (CPU) | same `graphData()` pattern | via wrapper |
| **react-force-graph-2d / -3d** | `1.29.1` (2026-02-04) | MIT | 1.7 MB / 13.0 MB | 3,298 ★ · pushed 2026-02-04 · **218 open issues** | CPU | React props → data reconciliation; must use the imperative `ref` to avoid re-warming on every prop change | **official** |
| **`@cosmos.gl/graph`** ✅ | `@cosmos.gl/graph@3.4.1` (2026-08-13) | **MIT** | 4.6 MB | `cosmosgl/graph` 1,269 ★ · pushed **2026-09-13** · 13 open issues | **GPU** (fragment/vertex shaders, luma.gl WebGL2) | `setPointPositions(Float32Array)` + `setLinks(Float32Array)`; `start()/stop()/pause()/unpause()`; GPU transitions (`transitionDuration`, default 800 ms) | imperative; official `@cosmograph/react` is NC-licensed — write your own 60-line wrapper |
| **`@cosmograph/cosmos`** ⛔ | `@cosmograph/cosmos@3.4.1` (2026-07-31) | **CC-BY-NC-4.0** | 1.4 MB | `cosmograph-org/cosmos` → **404, repo no longer public** | GPU | same API family | `@cosmograph/react@2.5.1` also **CC-BY-NC-4.0** |
| **Sigma.js + Graphology** | `sigma@3.0.3` (2026-04-30); `graphology@0.26.0` (2025-01-26); `graphology-layout-forceatlas2@0.10.1` (**2022-10-17**) | MIT / MIT / MIT | 948 KB / 2.7 MB / 77 KB | sigma 12,164 ★ · pushed **2026-09-14** · `sigma@4.0.0-beta.5` 2026-08-20 · 13 issues | CPU ForceAtlas2 (+ worker supervisor) | `graph.addNode()/addEdge()` on the Graphology store; Sigma re-renders from the live store — the **cleanest true incremental model** of any option | no official renderer |
| **deck.gl** | `deck.gl@9.4.0` (2026-09-05) | MIT | 6.3 MB | 14,588 ★ · pushed 2026-09-14 · `v9.4.0` 2026-09-05 · 532 issues | **none** — renderer only | reactive layers + `updateTriggers`; needs stable binary attributes | **official `<DeckGL>`** |
| **Cytoscape.js** | `cytoscape@3.34.3` (2026-09-07) | MIT | 5.6 MB | 11,210 ★ · pushed 2026-09-13 · `v3.34.3` 2026-09-07 · 20 issues | CPU, Canvas renderer (no WebGL force engine found) | `cy.add()`; force layout normally re-runs on topology change → visible jumps | no first-party |
| **AntV G6 v5** | `@antv/g6@5.1.1` (2026-05-08) | MIT | 7.4 MB | 12,291 ★ · pushed 2026-07-15 · `5.1.1` 2026-04-17 · **333 issues** | CPU force | batched data add/update/remove; explicit layout control | ref/effect |
| **Reagraph** | `reagraph@4.32.0` (2026-06-25) | Apache-2.0 | 671 KB | 1,088 ★ · pushed 2026-06-25 · 11 issues | CPU (Three.js/WebGL render) | React-first data props | **React-native API** |
| **helios-web** | `helios-web@0.10.9` (2026-06-29) | MIT | 26 MB | 92 ★ · pushed 2026-06-29 · 8 issues · **no LICENSE file detected by GitHub** | native GPU force | not documented for streaming | none |

**Options I could not verify and therefore do not recommend:** Ogma, KeyLines, yFiles, Graphistry
(commercial, no public benchmark or price obtained) — `UNVERIFIED`. `drkameleon/GraphGPU` (WebGPU)
was surfaced by deep research but I did not fetch its repo metadata myself — `UNVERIFIED`.

### A.3 Node-count breaking points for **this** use case

"Breaks" = the point at which a *continuously simulated, visibly rewiring* graph drops below
~30 sustained fps on a good laptop GPU. Derived from A.1.1 + A.1.2, not from marketing.

| Option | Breaks at (live physics) | Can still *display* (frozen layout) | Why it breaks |
|---|---:|---:|---|
| d3-force / react-force-graph-2d | **~5k moving nodes** | ~50k | CPU solver; graphology FA2 (a faster CPU solver) is already at 32 it/s at 10k |
| react-force-graph-3d | **~10k moving nodes** | ~75k | same CPU solver + Three.js scene-graph overhead (13 MB bundle) |
| Sigma + Graphology FA2 worker | **~10–15k moving nodes** | **~120k** (78 fps sustained at 121k/615k) | CPU FA2 = 1.61 it/s at 100k |
| Cytoscape.js | **~5–10k moving nodes** | ~20k (official perf page tests 200→20,000) | Canvas renderer, layout re-run on `cy.add()` |
| AntV G6 v5 | **~10–15k moving nodes** | ~50k | CPU force |
| **cosmos.gl (GPU)** | **~150k moving nodes** (39.7 it/s @100k; 10.3 it/s @500k) | ~1M | GPU shader solve; falls below 30 it/s somewhere between 100k and 500k |
| deck.gl + external solver | renderer never breaks first (120 fps @121k) | **~1M** | your solver is the limit |

### A.4 Recommendation for the EKG

**Primary: `@cosmos.gl/graph` (MIT).** It is the only option in this comparison whose *layout* —
not just its rendering — is measured above 30 it/s at 100k nodes. It is actively maintained
(pushed 2026-09-13, 13 open issues), v3 moved rendering to luma.gl WebGL2, and it ships
GPU-animated position transitions (`transitionDuration: 800`, `CubicInOut`) which is exactly the
"visibly rewiring" effect the brief asks for. It also has a GPU collision force and
`setConfigPartial()` for runtime parameter changes.

**Fallback: Sigma.js 3 + Graphology**, with the layout **frozen** and only a bounded active
frontier simulated. Graphology's `addNode`/`addEdge` is the cleanest genuine incremental data
model of any library here, and Sigma sustained **78.3 fps at 121k nodes / 615k edges** with a
precomputed layout.

**Mandatory mitigations regardless of library** (the EKG will be unreadable long before it is
slow):

| Mitigation | Rule of thumb for this project |
|---|---|
| Display subgraph | Render the **top-K = 2,000–5,000** nodes by recency × importance. Keep the full 100k graph server-side in the EKG store. |
| Edge budget | Hide edges above ~3× the displayed node count; show only edges incident to the selection or to the last N trades. |
| Freeze settled nodes | Simulate only a moving frontier (new nodes + 1-hop halo). Converts an O(N)-per-tick problem into O(frontier). |
| Seed new nodes | Spawn each new node at its parent's position + jitter, then local reheat. Never global reheat on every insert. |
| Batch | One `setPointPositions`/`setLinks` call per animation frame with the whole batch — never one per streamed node. |
| LOD | Labels only on hover/selection; 1-px points when zoomed out. |

**cosmos.gl API caveat (real, plan for it):** `setPointPositions` / `setLinks` are
*replacement*-style calls over `Float32Array`s, not append calls. You must own stable, pre-grown
typed-array buffers on the JS side (grow by doubling), write new entries into the tail, preserve
the existing prefix (which preserves node positions), and re-submit. Budget a day for this buffer
manager. This is the single non-obvious engineering cost of the primary pick.

---

## Part B — Streaming transport and backend

### B.1 Transport comparison

Measured evidence. `suenot/trading-ipc-bench` (verified 200 OK; updated **2026-03-04 22:45 UTC**;
GitHub Actions `ubuntu-latest`; 1,000 discarded warm-ups; 100,000 round trips per transport; each
transport isolated in a subprocess; `time.perf_counter_ns()`). **64-byte message, loopback,
Python, round-trip**:

| Transport | p50 µs | p95 µs | p99 µs | p99.9 µs | msg/s |
|---|---:|---:|---:|---:|---:|
| Unix domain socket | 20.2 | 31.1 | 42.6 | 65.5 | 46,512 |
| TCP | 41.7 | 54.6 | 67.2 | 78.0 | 22,425 |
| ZeroMQ (TCP) | 137.6 | 158.9 | 170.7 | 186.3 | 7,149 |
| **WebSocket** | **292.5** | 345.5 | 401.6 | 468.4 | **3,345** |
| HTTP REST | 296.8 | 366.4 | 385.3 | 419.8 | 3,249 |
| NATS | 411.4 | 466.6 | 527.2 | 712.4 | 2,394 |
| Redis Pub/Sub | 513.0 | 569.8 | 617.8 | 809.7 | 1,957 |
| Redis Streams | 665.9 | 772.4 | 844.2 | 999.5 | 1,491 |

**Context / limits of this number:** loopback only, no CPU model published, no TLS, no browser, no
serialisation of real payloads. It is a *floor* for "how many individual WebSocket messages per
second can Python push", not a capacity promise. The correct engineering response is **not** to
try to send 3,345 messages/s — it is to send **20 batched frames/s** carrying 50–5,000 graph
mutations each, i.e. 1,000–100,000 logical updates/s at the application level.

**I found no credible apples-to-apples FastAPI vs Starlette vs raw-uvicorn *WebSocket broadcast*
benchmark.** FastAPI's own benchmarks page is TechEmpower **HTTP** data and must not be relabelled
as WebSocket throughput. Stated as a gap, not filled with a guess.

### B.2 WebSocket vs SSE vs polling — decision table

| | WebSocket | SSE | Long-poll | Poll |
|---|---|---|---|---|
| Direction | full duplex | server→client only | server→client | client pull |
| Framing | text **or binary** | UTF-8 text only | text | text |
| Reconnect | **you implement it** | **browser does it**, with `id` + `Last-Event-ID` + `retry` (ms) | you implement | trivial |
| Connection limit | ~none in practice | **6 per browser+origin on HTTP/1.1**; HTTP/2 raises this to the negotiated max streams (MDN cites 100 default) | same HTTP/1.1 limit | same |
| Proxy risk | low | **high** — nginx/ingress/LB buffering; needs `X-Accel-Buffering: no`, `Cache-Control: no-cache`, heartbeat comments | medium | none |
| Client→server commands | yes | no (second HTTP channel needed) | no | n/a |
| Verdict here | **primary data plane** (graph deltas, controls, pause/resume, ack, snapshot request) | optional read-only status/log lane | compatibility fallback only | fine for a *local single-user* dashboard at ≤1 Hz |

**Where SSE actually wins:** a read-only, low-rate, text lane where the browser's free reconnect
and `Last-Event-ID` cursor are worth more than binary framing — training phase changes, warnings,
log lines. **Where WebSocket is required:** anything the client must talk back on — subscribe,
`request_snapshot`, `set_rate`, `pause_graph`, `ack` — plus binary typed-array batches.

### B.3 Backpressure — the part everyone gets wrong

Verified facts from the `websockets` library documentation:

| Fact | Consequence for this design |
|---|---|
| ~**64 KiB per connection** with default compression; ~**14 KiB** with compression disabled | Disable `permessage-deflate` on a LAN; it costs CPU and tail latency on already-packed numeric data |
| Defaults `max_size = 1 MiB`, `max_queue = 16` → up to **16 MiB** of queued *incoming* frames | Bound your own inbound handling too |
| `send()` waits only when the write buffer passes the high-water mark; the docs explicitly warn that **large buffers delay backpressure and cause bufferbloat** — higher latency, not higher throughput | Keep buffers small and drop frames instead |
| The broadcast docs state that a naive serialised loop lets **one slow client block all clients**, and that the built-in broadcast path does **not** apply backpressure per client | Never `for ws in clients: await ws.send(...)` |
| Starlette's thread pool default is **40 tokens** | A blocking call inside an `async def` endpoint will starve every socket |
| `encode/broadcaster` — the usual FastAPI pub/sub recommendation — is **ARCHIVED** (last push 2025-04-09, BSD-3) | Do not adopt it. Use an in-process registry (single host) or Redis Streams (multi-host) |

**Required pattern — per-client bounded queue + dedicated sender task + explicit overload policy:**

```python
@dataclass
class Client:
    ws: WebSocket
    q: asyncio.Queue           # maxsize=4 — NOT unbounded
    dropped: int = 0

async def sender(c: Client):                 # only this task touches the socket
    while True:
        await c.ws.send_bytes(await c.q.get())

def offer(c: Client, frame: bytes, lane: str):
    try:
        c.q.put_nowait(frame)
    except asyncio.QueueFull:
        c.dropped += 1
        if lane == "graph":      # conflate: drop oldest, keep newest
            c.q.get_nowait(); c.q.put_nowait(frame)
        elif lane == "metrics":  # latest-value-wins
            pass
        elif lane == "pnl":      # lossless lane -> resync from ring buffer, never silently drop
            request_resync(c)
```

**Per-lane policy (this is the architecture, not the wire protocol):**

| Lane | Rate | Policy on overload | Loss tolerated? |
|---|---|---|---|
| `graph` | 20 Hz coalesced frames | drop oldest / conflate by node-id | yes — next frame supersedes |
| `metrics` | 1–10 Hz | latest-value-wins per metric name | yes |
| `pnl` | batched ticks, sequence-numbered | never drop — resync from server ring buffer | **no** |
| `control` | event-driven | never drop | **no** |

### B.4 Serialisation

msgspec's own benchmark (methodology published: ~2020 x86 Linux laptop, CPython 3.11; the authors
warn tight loops keep caches hot). **Decoding one ~77 MiB JSON file:**

| Library | Decode time | Extra memory |
|---|---:|---:|
| msgspec (typed `Struct`) | **176.8 ms** | **67.6 MiB** |
| msgspec (untyped) | 630.5 ms | 218.3 MiB |
| orjson | 691.7 ms | 406.3 MiB |
| stdlib `json` | 868.6 ms | 295.0 MiB |
| ujson | 1,087.0 ms | 349.1 MiB |

msgspec also reports ~12× faster than Pydantic V2 and ~85× faster than Pydantic V1 on its
structured encode/decode/validate benchmark. **Takeaway: serialisation becomes the bottleneck
before the socket does.** Use `msgspec.Struct` server-side. On the wire use **MessagePack**
(`msgpackr@2.1.0`, MIT, browser side) once measured; start with JSON for debuggability.

**Wire layout must be columnar, not one object per edge:**

```json
{"v":1,"stream":"graph","kind":"delta","seq":184203,"base_seq":184000,"ts_ns":...,
 "nodes":{"id":[...],"kind":[...],"x":[...],"y":[...]},
 "edges":{"src":[...],"dst":[...],"w":[...]},
 "removed_nodes":[...],"removed_edges":[...]}
```
Integer node IDs, `float32` coordinates, strings in a side dictionary sent once. Never repeat a
label on every edge.

### B.5 Reconnect and resync

Snapshot + delta with per-stream monotonic `seq`:
1. Client connects with its `last_seq` per stream.
2. Server replies `hello` with protocol version and the retention window of its ring buffer.
3. If `last_seq` is still in the ring buffer → replay missing deltas in order.
4. Else → send a full snapshot, then deltas after it.
5. Client applies only `seq > last_seq`; duplicates are harmless (make every message idempotent).
6. Client acks the highest applied `seq`; server disconnects clients that cannot catch up.

Ring buffer = in-memory `deque` for one process; **Redis Streams** for multi-process (append-only,
random access, consumer groups) — accepting the measured 665.9 µs p50 / 1,491 msg/s loopback cost
in exchange for replay.

### B.6 Recommended architecture

```
RL training process (SB3)  --in-process asyncio.Queue-->  aggregator/state store
   (EKG store, metrics, P&L ring buffer)
        |
        v  FastAPI 0.141.1 / Starlette / uvicorn
   WebSocket /stream  --  per-client bounded queue (maxsize=4)
                      --  20 Hz coalescing frame pump
        |
        v  React + Vite
   Web Worker decodes + merges  ->  cosmos.gl applies one batch per rAF
                                ->  uPlot / lightweight-charts append batches
```

**Capacity target (engineering target, load-test it — not a published benchmark):** 20 frames/s
per client × 50–5,000 mutations per frame; queue depth 2–4; compression off on LAN.
**Instrument:** bytes/frame, encode time, send-wait time, queue age (not just length), p50/p95/p99
end-to-end latency, dropped graph frames, reconnect recovery time, browser main-thread time.

**Internal hop:** in-process `asyncio.Queue` for one host (single-user case → this project).
Redis Streams only when the trainer and the web server are on different machines and you need
replay. Kafka is overkill. A SQLite/parquet polling loop is genuinely adequate if the visual
requirement is one update every few seconds — but it cannot drive a live force graph.

---

## Part C — Charts

### C.1 Comparison (all metadata live 2026-09-14)

| Library | npm version (published) | Licence (registry) | Unpacked | GitHub ★ / last push / latest release / open issues | Live-append API | Performance evidence |
|---|---|---|---:|---|---|---|
| **TradingView lightweight-charts** | `5.2.1` (2026-08-12) | **Apache-2.0 + attribution, see C.2** | 3.0 MB | 17,256 ★ · 2026-09-10 · `v5.2.1` 2026-08-12 · 137 issues | `series.update(bar)` per tick; `setData()` for snapshots | Canvas, financial-specific. **No reproducible max-point benchmark found** — `UNVERIFIED` |
| **uPlot** | `1.6.32` (**2025-03-14**) | MIT | 533 KB | 10,491 ★ · pushed **2026-09-14** · latest release `1.6.32` 2025-03-14 · 150 issues | `setData()` on columnar typed arrays | **Project's own claim**: 166,650-point chart in 25 ms, ~100k points/ms after. Project benchmark, not independent |
| **Plotly.js** | `4.1.0` (2026-09-08) | MIT | **91.5 MB** | 18,328 ★ · 2026-09-13 · `v4.1.0` 2026-09-08 · **772 issues** | `Plotly.extendTraces` / `Plotly.react` | Plotly docs cite ~1M points for WebGL traces; larger needs server-side aggregation |
| **Apache ECharts** | `6.1.0` (2026-05-19) | Apache-2.0 | 58.9 MB | 67,315 ★ · 2026-09-14 · `6.1.0` 2026-05-19 · **1,501 issues** | `setOption` (incremental), large-data mode | No single credible max-point number found |
| **Recharts** | `3.10.1` (2026-07-25) | MIT | 7.3 MB | 27,556 ★ · 2026-09-14 · `v3.10.1` 2026-07-25 · 446 issues | React props re-render (SVG) | Its **own** performance guide tells users with large data / rapid changes to isolate charts and disable animation; issue #1146 "Recharts is slow with large data" is open |

### C.2 ⚠ lightweight-charts licence — the trap

Apache-2.0 here is **not** attribution-free. Quoting the current `README.md` on
`tradingview/lightweight-charts` master, fetched 2026-09-14:

> "This license requires specifying TradingView as the product creator.
> You shall add the **"attribution notice" from the NOTICE file and a link to
> https://www.tradingview.com/** to the page of your website or mobile application that is
> available to your users."

The `NOTICE` file contains: `TradingView Lightweight Charts™ / Copyright (с) 2025 TradingView, Inc.
https://www.tradingview.com/`. The library provides a built-in
`LayoutOptions.attributionLogo` chart option which the docs say satisfies the link requirement.

**Decision:** acceptable — enable `attributionLogo: true` and add the NOTICE text to the app's
attributions page. If the project ever needs a clean-room, attribution-free chart, **uPlot (MIT)**
is the drop-in escape.

### C.3 Picks

| Panel | Pick | Why | Fallback |
|---|---|---|---|
| **Live P&L / equity vs baseline** | **lightweight-charts 5.2.1** | Purpose-built financial time axis, crosshair, session handling, `series.update()` per tick, small relative to Plotly (3.0 MB vs 91.5 MB unpacked), very active (pushed 2026-09-10) | **uPlot** if the TradingView attribution is unacceptable |
| **Training / learning curves** | **uPlot 1.6.32** | Smallest bundle (533 KB), columnar typed-array data model matches the wire format from B.4, fastest published point throughput, log axes | **ECharts 6.1.0** if you need heatmaps, parallel coordinates, data-zoom or a broader panel set |

**Avoid Recharts on any hot path** — SVG + declarative re-render. Keep it only for small static
summary cards. **Avoid Plotly for this app** — 91.5 MB unpacked package for functionality uPlot
and lightweight-charts already cover.

**Implementation rules:** keep raw ticks in the server ring buffer; downsample the visible range
(min/max or LTTB) before sending; call the chart's append API **once per animation frame with a
batch**; never push a tick into React state.

---

## Part D — Build vs buy for the whole UI

### D.1 The acceptance test

The right question is not "can it display 10,000 nodes?" It is: *can it add nodes while the force
simulation keeps running, hold 30–60 fps on target hardware, avoid resending the whole graph on
every update, and keep training alive when the browser disconnects?*

### D.2 Matrix (metadata live 2026-09-14)

| Option | Version / ★ / last push | Live 10k+ WebGL force graph? | Curves | Glue-code estimate* | What breaks first |
|---|---|---|---|---:|---|
| **React + Vite + FastAPI** | FastAPI `0.141.1`, 102,331 ★, pushed 2026-09-01 | **Yes** — the graph owns its own rAF loop; you control the delta protocol | excellent | 800–2,500 LOC v1; 2,000–6,000 production | your own graph UX, reconnect, buffer manager |
| **Streamlit** | `1.63.0` (2026-09-01), 45,744 ★, 1,162 issues | **No natively.** `st.fragment(run_every=...)` avoids full reruns, but the model still re-runs Python; a continuous WebGL animation needs a custom component — i.e. you write the React app anyway | good at low rate | 100–400 LOC charts; **+500–2,000 TS** for the graph component | rerun latency, serialisation, session CPU |
| **streamlit-agraph** | — | **No** — wraps `react-graph-vis`; its own README states it is not working after the agraph 2.0 update | — | low, high replacement risk | maintenance |
| **Plotly Dash** | `v4.4.1` (2026-07-21), 24,405 ★, 487 issues | **Conditional.** `dcc.Interval` + `extendData` are good for curves; Dash Cytoscape is Cytoscape.js (Canvas) — no WebGL force guarantee. Custom clientside component needed | very good | 250–900 LOC + 600–2,500 for a custom graph component | callback queue; persistent synchronous callbacks share a default 4-worker pool |
| **HoloViz Panel** | `v1.9.4` (2026-08-17), 5,771 ★, **1,113 issues** | **Yes, conditional** — `JSComponent`/`ReactComponent` ESM escape hatch genuinely lets you host cosmos.gl | good (`stream()` on ColumnDataSource, periodic callbacks) | 300–900 Python + 300–1,500 ESM/React | Bokeh document patching; and you have written a custom frontend anyway |
| **Gradio** | `6.27.0` (2026-09-11), 43,531 ★ | **No** as a dashboard architecture; possible via a custom component | ok for demos | 100–400 demo; 600–2,000 component | event queue, reconnect semantics |
| **W&B dashboards** | SDK `v0.30.0` (2026-09-09), 11,247 ★ | **No.** `wandb.Html` logs HTML; Vega-Lite custom charts are interactive — but there is **no native network/force-graph panel** and no documented continuously-connected WebGL runtime | **excellent** (this is its job) | 20–200 LOC | it is a *log store*, not a live socket |
| **Reflex** | `v0.9.11` (2026-09-11), 28,883 ★ | Yes, conditional — it *is* React + FastAPI + WebSockets | good | 500–1,500 Py + 300–1,500 JS | framework state serialisation vs imperative graph state |
| **NiceGUI** | `v3.16.0` (2026-08-12), 16,206 ★ | via custom Vue component — same burden | good | similar to Panel | same |
| **Rerun** | `0.37.2` (2026-09-11), 11,445 ★, Apache-2.0 | Not a force-graph product; it is a streaming *spatial/temporal* viewer | n/a | — | wrong data model for a knowledge graph |

\* Engineering estimates, not vendor figures.

### D.3 Recommendation, and what is lost

**Build: a small custom React + Vite frontend with a FastAPI WebSocket backend. Buy: W&B for
experiment tracking.**

The justification is not "custom is better". It is that **every** Python-first option that can
actually render the EKG requires you to write the same WebGL component — Streamlit custom
component, Dash clientside component, Panel `ReactComponent`, Gradio custom component, NiceGUI Vue
component. Once you have written it, the Python framework adds a serialisation boundary, a rerun
model and a second failure mode between your component and your data, and buys you only the parts
(sliders, layout, tables) that are the cheapest part of a React app. The Python frameworks are
worth it when the graph is *not* the product; here the graph **is** the headline requirement.

| Alternative not chosen | What is genuinely lost |
|---|---|
| Streamlit | ~1 day to a working metrics page; Python-only team could maintain it |
| Dash | Mature callback model, `extendData`, Dash Cytoscape for small graphs, enterprise support option |
| Panel | Best Python escape hatch; Datashader for huge static data; keeps everything in one language |
| Gradio | Fastest path to a shareable public demo link |
| W&B-only | Zero frontend code; free hosted history, comparison, reports, permissions |
| Reflex | Same architecture with less boilerplate — **closest runner-up**; rejected only because the graph needs imperative state that fights a reactive Python state model |

**Hybrid (the actual plan):** W&B is the system of record for every run, sweep, config, artifact
and learning curve. The custom React app renders only (a) the live EKG, (b) live equity vs
baseline, (c) a live mirror of the current run's learning curve. If the React app dies, no
experiment data is lost.
