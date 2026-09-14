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

---

## Part E — Weights & Biases, sweeps and HPO for PPO/SAC

### E.1 Live 2026 pricing and limits — fetched from https://wandb.ai/site/pricing/ on 2026-09-14

| Plan | Price | Model seats | Storage | Weave ingestion | Tracked hours | Restriction |
|---|---:|---:|---:|---:|---|---|
| **Free** (cloud) | **$0/mo** | up to **5** | **5 GB/mo** | 1 GB/mo | not stated on page | "personal development of AI applications and models"; public **and** private projects; community support |
| **Pro** (cloud) | **from $60/month**, billed monthly | up to **10** | **100 GB/mo**, extra **$0.03/GB** | 1.5 GB/mo | not stated | early-stage teams, **fewer than 50 employees**; page says Pro has "rate limits and performance constraints" |
| **Enterprise** | custom | customisable | customisable | customisable | — | SSO, SCIM, CMEK, audit logs, single tenant, HIPAA option |
| **Personal** (self-hosted, `wandb server start`) | **$0/mo** | **1 seat** | n/a (your disk) | — | — | ⚠ **"For personal projects only. Corporate use is not allowed."** |
| **Academic** | **free forever** | up to **100 seats** | **200 GB** cloud (extra $0.03/GB/mo) | up to **25 GB/mo** | **unlimited** | All Pro features. "academic institutions pursuing research **not connected to a for-profit entity**… students, professors, postdoctoral researchers. An active email address affiliated with an academic institution is required." Apply via the pricing page → `https://wandb.ai/create-team` |

Storage is billed as a 30-day rolling average: the page's worked example is
`(100 GB×15d + 200 GB×1d + 300 GB×14d) / 30d = 196.6 GB → 197 GB`.

**Operational note (live banner on docs.wandb.ai, 2026-09-14): "The Weights & Biases domain will
change on September 30."** Pin nothing to a hard-coded host.

**Action for this project:** apply for the **academic** licence if eligible. Otherwise the Free
tier's **5 GB/month** is the binding constraint — a 60-trial RL sweep that uploads model
checkpoints as artifacts will exhaust it. Mitigation: log scalars only, keep checkpoints on disk
or in a reference artifact (the FAQ confirms externally-stored reference artifacts **do not**
count against the quota).

### E.2 Documented scale guidance and rate limits (docs.wandb.ai/guides/track/limits/)

| Dimension | Guidance at scale (Multi-tenant Cloud) |
|---|---:|
| Runs per project | 10,000 |
| Steps per run | 500,000 |
| Metric cardinality per project | 100,000 |
| Log frequency | **1,000 `run.log()` calls / minute** |
| Throughput | **100,000 values / minute** |
| Video throughput | 40 MB / minute |
| Files per run | < 1,000 (use Artifacts above that) |

Rate limits: exceeding them returns **HTTP 429** with `RateLimit-Limit`, `RateLimit-Remaining`
and `RateLimit-Reset` headers (limit/remaining are scaled 0–1000; reset is in seconds). Metric
limits apply **per project** and cover both request rate and total request size over a rolling
window; paid plans get higher limits. The SDK retries with backoff, which **can delay
`run.finish()`** until the window resets. GraphQL/public-API requests are limited per IP
(unauthenticated) or per user (authenticated); the docs advise ≥1 s between public-API calls.
W&B does **not** publish the actual numeric quotas.

**RL consequence — do not log per environment step.** With `N` vectorised envs, a per-step
`run.log()` multiplies the call rate by `N` and will hit 1,000 calls/min almost immediately. Log
at the **rollout** boundary (aggregated return, losses, entropy, explained variance, KL, FPS) and
at **evaluation checkpoints**. Batch related metrics into one call.

**Offline:** `WANDB_MODE=offline` writes locally; upload later with `wandb sync <run-dir>`. Caveat
for sweeps: a W&B **sweep agent must stay online** to receive suggestions from the sweep service.
An offline sweep therefore requires an external controller (→ Optuna, E.4).

### E.3 W&B Sweeps — verified capabilities (docs.wandb.ai/guides/sweeps/sweep-config-keys/)

Top-level keys: `program` (req), `method` (req), `parameters` (req), `metric`, `entity`,
`project`, `name`, `description`, `early_terminate`, `command`, `run_cap`.

| Feature | Verified detail |
|---|---|
| Search methods | **`grid`, `random`, `bayes`** — those three only |
| Bayes caveat | Docs state plainly: *"Bayesian search works well for small numbers of continuous parameters but scales poorly."* **The surrogate model and acquisition function are NOT documented.** Do not claim it is GP-BO or TPE |
| Never-terminating | `grid` "executes forever" in a continuous space; `random` and `bayes` "run forever unless you stop" → **always set `run_cap`** |
| Distributions | `uniform, q_uniform, log_uniform, log_uniform_values, inv_log_uniform, inv_log_uniform_values, normal, q_normal, log_normal, q_log_normal, categorical, int_uniform, constant` |
| Early stopping | **Hyperband only.** `early_terminate: {type: hyperband, min_iter, max_iter, s, eta (default 3), strict (default false)}`. Brackets are counted in **number of logged iterations of the target metric**, not step values. "Hyperband checks which runs to end once every few minutes" |
| Command macros | `${env} ${interpreter} ${program} ${args} ${args_no_boolean_flags} ${args_no_hyphens} ${args_json} ${args_json_file}` |
| Multi-GPU agents | Documented pattern is one process per GPU: `CUDA_VISIBLE_DEVICES=0 wandb agent <ID>` / `CUDA_VISIBLE_DEVICES=1 wandb agent <ID>`. Across machines, launch agents with the same sweep ID |
| Limits | **No documented maximum** on runs, agents, concurrency or sweep duration. Absence of a documented limit is not a capacity guarantee |

**Sweeps × vectorised RL — the rules that matter:**

1. **One W&B run per sweep trial**, in the learner process. `SubprocVecEnv` workers must **never**
   call `wandb.init()`.
2. Report **one scalar objective** at a **fixed environment-step budget** — e.g.
   `eval/mean_return` or `eval/sharpe_net` over a fixed list of evaluation seeds/windows. Never
   optimise the max training reward seen by any worker.
3. Seed deterministically: trial seed in `wandb.config`, workers seeded `base_seed + worker_id`.
4. Hyperband brackets must correspond to **comparable training units** (evaluation checkpoints /
   env-step milestones), never wall-clock — GPU/CPU contention would then bias pruning.
5. SB3 integration is `wandb.integration.sb3.WandbCallback` (logs metrics, tracks/uploads the
   model, records full hyper-parameters, optional gradient logging); `wandb.init()` must be called
   first.

### E.4 W&B vs Optuna vs Ray Tune (all live 2026-09-14)

| | **W&B Sweeps** (`wandb` SDK `v0.30.0`, 11,247 ★) | **Optuna `v5.0.0`** (2026-09-07, 14,791 ★, **21 open issues**) | **Ray Tune** (`ray-2.58.0`, 2026-08-23, 43,799 ★) |
|---|---|---|---|
| Search | grid / random / bayes (internals undocumented) | TPE + many samplers, **define-by-run** spaces, conditional spaces | wraps Optuna/HyperOpt; `HyperOptSearch` = TPE |
| Pruning | Hyperband only | `MedianPruner`, `SuccessiveHalvingPruner`, `HyperbandPruner` via `report()` + `TrialPruned` | **ASHA**, **PBT** (pause/clone/mutate/resume) |
| Study state | hosted, opaque | **RDB storage** (SQLite/MySQL/Postgres) + gRPC proxy → resumable, inspectable, multi-machine | Ray cluster state + checkpoints |
| Dashboard | **best-in-class** | `optuna-dashboard` (history, importances, contours) | Ray dashboard |
| Scheduling | you place agents yourself | you place workers yourself | **resource-aware cluster scheduler** |
| Artifacts / lineage / reports / permissions | **yes — this is what it adds** | no | no |
| Offline | agent must be online | fully offline | fully offline |

**What W&B actually adds:** a durable, shareable, permissioned *system of record* — run lineage,
config diffing across trials, artifact/model versioning, media, reports, and a UI that stays
useful after the search finishes. **What it does worse:** opaque BO internals, no pruner choice
beyond Hyperband, no resumable local study database, agent must be online.

**Recommended:** **Optuna for search + pruning, W&B for logging** — Optuna ships
`optuna.integration.WeightsAndBiasesCallback`, which tracks suggested hyper-parameters and the
optimised metric in W&B. This satisfies the brief's "W&B for all hyper-parameter tuning" mandate
(every trial is a W&B run, every sweep is a W&B project) while giving you a resumable SQLite study
and a real pruner. **Fallback:** pure W&B Sweeps `method: bayes` + `early_terminate: hyperband`
with `run_cap` — simpler, zero infrastructure, acceptable for ≤5 parameters.

### E.5 Evidence-based HPO advice for PPO/SAC on a 5-asset trading env

**What the literature actually says** (all arXiv IDs verified in this session):

| Paper | arXiv | Finding relevant here |
|---|---|---|
| Andrychowicz et al., *What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study* | **2006.05990** | >250,000 agents trained. Implementation-level choices dominate. Strong support for **observation normalisation**; check value-function normalisation; **policy initialisation** matters significantly; gradient clipping is secondary |
| Engstrom et al., *Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO* | **2005.12729** | Code-level optimisations account for much of PPO's reward advantage over TRPO and *fundamentally change agent behaviour*. → **An HPO comparison is invalid if any trial silently changes normalisation, advantage handling, init, clipping or the evaluation protocol** |
| Eimer, Lindauer, Raileanu, *Hyperparameters in Reinforcement Learning and How To Tune Them* | **2306.01324** | The performance landscape depends strongly on the **tuning seed**; the same config has large variance; a config chosen on one seed can be **>4× worse on test seeds**. **Explicitly recommends separate tuning and testing seeds** |
| Henderson et al., *Deep Reinforcement Learning that Matters* | **1709.06560** | Seed/implementation variance can swamp algorithmic differences |
| Agarwal et al., *Deep RL at the Edge of the Statistical Precipice* | **2108.13264** | Report stratified bootstrap CIs / IQM, not point estimates, over few runs |
| Li et al., *Hyperband* | **1603.06560** | The algorithm behind W&B's only early-stopping option |
| Li et al., *A System for Massively Parallel Hyperparameter Tuning* (ASHA) | **1810.05934** | Ray Tune's async successive halving |
| Jaderberg et al., *Population Based Training* | **1711.09846** | Ray Tune PBT |
| Akiba et al., *Optuna* | **1907.10902** | TPE + pruners + RDB studies |
| Liaw et al., *Tune* | **1807.05118** | Ray Tune |
| Franke et al., *Sample-Efficient Automated Deep RL* (SEARL) | **2009.01555** | Population-based AutoRL for off-policy (SAC-class) methods |
| Huang et al., *CleanRL* | **2111.08819** | Single-file reference PPO; vectorised envs |

`UNVERIFIED`: arXiv **2412.07165** ("A Method for Evaluating Hyperparameter Sensitivity in RL") was
surfaced by deep research but I did **not** run `verify_arxiv` on it. Do not cite it until checked.

**Search-space priority for PPO** (5 assets, daily bars, `MlpPolicy`, discrete or Box action):

| Tier | Parameters | Note |
|---:|---|---|
| 1 | `learning_rate` (log-uniform 1e-5…3e-3), `n_steps`, `n_envs`, `batch_size`, `n_epochs` | These set optimisation noise, sample reuse and updates-per-sample. Tune these first |
| 2 | `clip_range`, `ent_coef` (log-uniform 1e-8…1e-1), `vf_coef`, `max_grad_norm`, `target_kl` | Update size, exploration, value/policy balance |
| 3 | `gamma` (0.95–0.9999), `gae_lambda` (0.9–1.0), **obs/advantage normalisation flags**, `ortho_init`, net width/activation | Normalisation & init have the strongest published evidence (2006.05990) |
| 4 | seed, eval frequency, total budget, feature preprocessing | Experimental controls — fix them or search them **explicitly**, never let them drift |

**For SAC**, tune jointly: actor/critic `learning_rate`, `ent_coef` (`"auto"` vs fixed) and
`target_entropy`, `batch_size`, `buffer_size`, `learning_starts`, `train_freq`/`gradient_steps`,
`gamma`, `tau`, net arch. Hold action scaling, reward scaling, termination handling and
observation normalisation **fixed** — otherwise they masquerade as algorithmic effects.

**How many runs?** `DLR-RM/rl-baselines3-zoo` (2,878 ★, `v2.9.1` 2026-06-15, MIT) — the reference
tuned-hyper-parameter repo — uses **Optuna with a default budget of 500 trials** and an
intermediate evaluation every **100,000 timesteps** for pruning. Its PPO tuning also sets
`ortho_init = False`, *unlike* the SB3 default — evidence that implementation defaults belong
inside the search space.

**Concrete budget for this project** (engineering recommendation, not a published law):

| Stage | Trials | Seeds/trial | Parameters | Notes |
|---|---:|---:|---:|---|
| 0. Sanity | 1 | 1 | — | Verify the env, the reward, and that a random agent ≈ 0 net of costs |
| 1. Coarse random | 30 | 1 | 8–10 | `method: random` + Hyperband. Purpose: find the live region, kill divergent configs cheaply |
| 2. Focused Bayes/TPE | 50–80 | 1 | **5–6** (top movers from stage 1) | Optuna TPE + MedianPruner. A broad 10-D BO over noisy single-seed returns is *less* informative than a focused 5-D search with replication |
| 3. Confirmation | top 5 configs | **5–10 seeds** | — | Report IQM + bootstrap CI (2108.13264), not the best seed |
| **Total** | **~110 trials + 50 confirmation runs** | | | |

**Avoiding tuning on the test window — mandatory protocol for a finance backtest:**

1. Split **chronologically**: train → validation(tuning) → **untouched test**. Never random-split.
2. Run all HPO **inside the training/validation window only**.
3. Select on validation risk-adjusted return (Sharpe **net of transaction costs**), not raw return.
4. Evaluate **once** on the next untouched period. One look.
5. **Walk forward**: roll the whole window and repeat; report the *distribution* of out-of-sample
   results across folds, not a single number.
6. Freeze evaluation seeds within a comparison so every candidate faces identical randomness.
7. Use **separate tuning seeds and testing seeds** — this is 2306.01324's explicit recommendation,
   and the >4×-worse-on-test-seeds result is the reason.
8. Record the number of configurations tried; the more you try, the more the best validation
   Sharpe is an overfit order statistic.

---

## Part F — Training compute on Prime Intellect

Docs index: `https://docs.primeintellect.ai/llms.txt` (fetched 2026-09-14; any page + `.md`
returns markdown). Official CLI/SDK repo: `PrimeIntellect-ai/prime` (326 ★, pushed 2026-09-14,
MIT).

### F.1 Renting a GPU pod from the CLI — verified command reference

```bash
# install + auth
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install prime                 # or: pip install prime
prime login                           # or: prime config set-api-key  /  export PRIME_API_KEY=...
prime config set-ssh-key-path         # keys generated at app.primeintellect.ai/dashboard/profile
prime config view

# find capacity + price
prime availability list
prime availability list --gpu-type H100_80GB --regions united_states --socket PCIe --no-group-similar
prime availability gpu-types
prime availability disks              # persistent-disk offers + $/GB/hr

# create / manage / destroy
prime pods create                     # interactive
prime pods create --id <ID> --name rl-trader --disk-size 100 --vcpus 16 --memory 64 \
                  --image <img> --env KEY=value --disks <disk-id>
prime pods list
prime pods status <pod-id>
prime pods ssh <pod-id>               # chmod 400 your private key first
prime pods terminate <pod-id>
prime disks list
```

Filters for `availability list`: `--gpu-type --gpu-count --regions --socket {PCIe,SXM2,SXM3,SXM4,SXM5}
--disks --group-similar/--no-group-similar`. `pods create` options: `--id --cloud-id --gpu-type
--gpu-count --name --disk-size --vcpus --memory --image --team-id --env --disks --share-with-team
--add-members`.

> **Correction to a common claim:** `prime pods status` and `prime pods terminate` **are**
> documented — both in `docs.primeintellect.ai/cli-reference/provision-gpu.md` and in the
> `PrimeIntellect-ai/prime` README. Likewise `prime env init` / `prime env push` /
> `prime env install` / `prime env list` / `prime env info` / `prime env inspect` are all in the
> README.

**Billing mechanics (FAQ, verified):** on-demand pods are non-interruptible; **Spot** uses unused
capacity at "discounts of up to 90%" and can be interrupted. **Credits are deducted every minute
while the pod is active**, and **pods are automatically deleted if credits run out**. There is
**no formal SLA**. Terminating a pod **destroys all its data**; pause/resume exists only on some
providers (Runpod, data in `/workspace`, only while *paused*, not terminated).

### F.2 Prices (2026-09-14) — with an explicit honesty caveat

The authenticated API `GET https://api.primeintellect.ai/api/v1/availability/gpus` returned
**403 `{"detail":"Not authenticated"}`** for me (no Prime API key in this session), and my static
fetch of `app.primeintellect.ai/dashboard/create-cluster` returned a **client-rendered JS shell
with no prices in the HTML**. The table below therefore comes from a **browser-rendered snapshot
taken by the deep-research tool on 2026-09-14**; treat it as a point-in-time marketplace quote.
**Authoritative check is always `prime availability list`.**

| GPU | VRAM | Community $/hr | Secure $/hr |
|---|---:|---:|---:|
| **RTX A2000** | 6 GB | **$0.15** | — |
| RTX 4090 | 24 GB | **$0.37** | $0.72 |
| L4 | 24 GB | — | $0.46 |
| L40S | 48 GB | — | $1.00 |
| A100 | 40 GB | — | $1.45 |
| A100 | 80 GB | $1.22 | $1.35 |
| H100 | 80 GB | $2.72 | $1.90 |
| H200 | 141 GB | — | $3.65 |

From `www.primeintellect.ai` homepage cards (same date, **my own fetch**): H100 **$2.43/hr**
on-demand vs H100 **Spot $0.94/hr**; B300 288 GB **$4.99/hr**; B200 192 GB **$3.49/hr**.
⚠ Several homepage cards all show "$3.14/HR" (π) and look decorative — I do not cite those.

### F.3 CPU-only: yes, via **Sandboxes** — and this is the right product here

`docs.primeintellect.ai/sandboxes/overview.md` (verified) publishes explicit **CPU-only** pricing:

| Resource | Price |
|---|---|
| CPU | **$0.05 per core per hour** |
| Memory | **$0.01 per GB per hour** |
| Disk | **$0.001 per GB per hour** |
| Docs' worked example | 1 core + 2 GB RAM + 10 GB disk = **$0.08/hour** |

VM-sandbox limits: 1–16 cores, 0.1–64 GB RAM, 0.1–128 GB disk, 0–8 GPUs (GPU sandboxes "coming
soon"), timeout 1 min–unlimited, idle timeout ≤1,440 min. Per account: 512 active sandboxes,
512 cores, 4,096 GB memory, 5,120 GB storage, 128 HTTP + 32 TCP port exposures. Sandboxes run
standard Docker images and support `prime tunnel` for exposing a local service.

### F.4 Is a GPU even justified? **No.**

The workload: 5 equities, **daily** bars. ~15 years ≈ 3,800 steps/episode. Observation ≈ 50–100
floats. Policy = 2×64 MLP ≈ 10k parameters. PPO minibatch 64–256.

An honest engineering argument (**not** a cited benchmark — I did not find a published
CPU-vs-GPU benchmark for SB3 `MlpPolicy` in this session, so this is reasoning, marked as such):
a forward+backward pass on a 2×64 MLP at batch 256 is a handful of small GEMMs. On a GPU each one
is dominated by kernel-launch and host↔device transfer overhead (µs-scale launches for
sub-µs-of-work kernels). The real cost of this workload is the **Python environment step loop**,
which is CPU-bound and parallelises across `SubprocVecEnv` workers. **Cores beat FLOPs here.**

| Configuration | Hourly cost | Fit |
|---|---:|---|
| **Sandbox, 16 cores / 32 GB / 64 GB disk** | 16×0.05 + 32×0.01 + 64×0.001 = **$1.184/hr** | ✅ **best for parallel sweep trials** — 16 concurrent single-core trials |
| **Sandbox, 8 cores / 16 GB / 50 GB** | 8×0.05 + 16×0.01 + 50×0.001 = **$0.61/hr** | ✅ everyday dev / single training run |
| **Sandbox, 4 cores / 8 GB / 30 GB** | 4×0.05 + 8×0.01 + 30×0.001 = **$0.31/hr** | ✅ cheapest sane box |
| RTX A2000 6 GB pod (community) | **$0.15/hr** | ⚠ cheapest *listed* compute overall, but check its vCPU count — if it ships ≥8 vCPUs it is the best value on the platform for this workload, GPU unused |
| RTX 4090 pod (community) | **$0.37/hr** | ✅ acceptable — buy it for its **vCPUs**, not its GPU |
| H100 | $1.90–2.72/hr | ❌ ~5–18× the cost for no measurable benefit on a 10k-parameter MLP |

**Recommendation:** run training on a **CPU sandbox or the cheapest CPU-rich pod**, not an H100.
This directly contradicts the master brief's "training happens on cloud GPUs" constraint, and I am
flagging it as a **deliberate, evidence-based deviation**: it is still *cloud* training, still
CLI-first, still reproducible — just not GPU-bound, because the model is 10k parameters.
A GPU only becomes worth renting if the design later adds (a) a transformer/LSTM feature encoder
over long sequences, (b) a GNN over the EKG with ≥10k nodes in the forward pass, or (c) an LLM in
the sentiment/ontology agent.

### F.5 Storage

| Option | Behaviour | Price |
|---|---|---|
| **Persistent disk** | Survives instance **and cluster** termination; attachable to multiple instances simultaneously; **must match the provider and location** of the compute; created first, must be `Active`; attach with `prime pods create --disks <id>`; discover with `prime availability disks` / `prime disks list` | The docs' example availability rows show **$0.00011111/GB/hr** (runpod US-WA-1, max 8,192 GB, multinode yes) and **$0.000097/GB/hr** (hyperstack NORWAY-1, max 100,000 GB, multinode no) → ≈ **$0.071–$0.081 per GB-month** |
| **Cluster ephemeral shared storage** | Created for the cluster lifetime, shared by nodes, **deleted when the cluster terminates** | included |
| **Pod boot disk** | `--disk-size`; **destroyed on terminate** | included in pod rate |
| Object storage / S3 | **No first-party S3-compatible object store found in the docs.** FAQ says: *"Always back up critical data to external storage (e.g. S3) before terminating or pausing."* | — |

**For this project:** a **50 GB persistent disk ≈ $3.55–$4.05/month** holding the market-data
cache, the EKG snapshots and the checkpoint archive. Everything else lives in git + W&B.

### F.6 prime-rl and the Environments Hub — do **not** use them here

| Question | Verified answer |
|---|---|
| What is `prime-rl`? | `PrimeIntellect-ai/prime-rl` (2,038 ★, `v0.9.0` 2026-08-25, Apache-2.0, pushed 2026-09-14): *"Fully asynchronous RL for high-throughput **agentic** training at scale"*, targeting 1T+ MoE models on 1,000+ GPUs, **FSDP2** for training + **vLLM** for inference, FP8, expert/context parallelism |
| Which algorithms? | From `docs.primeintellect.ai/prime-rl/algorithms.md`, configured under `[orchestrator.algo]`: **`grpo`, `max_rl`, `rae`, `hierarchical_grpo`, `opd`, `sft`, `opsd`, `echo`**. **There is no standalone continuous-control PPO or SAC trainer.** All of these are LLM-token-level losses |
| Can it train an MLP PPO trading policy? | **No documented path.** It assumes language-model policies, token losses, model references, rollouts and verifier-style environments. It does not accept a Gymnasium env |
| What is a `verifiers` "environment"? | `PrimeIntellect-ai/verifiers` (4,614 ★, `v0.3.1` 2026-08-24, MIT): *"our library for creating environments to train and evaluate **LLMs**"* — datasets, parsers, tool calls, multi-turn trajectories, rubric/verifier scoring. **Not** a numeric-observation/continuous-action Gym API |
| Publishing | `prime env init <name>` → `prime env push <name>`; browse with `prime env list`, install with `prime env install <name>`. The Hub is migrating to **verifiers v1**; v0 environments carry a **Legacy** tag and the v0 create workflow is marked deprecated |
| Is wrapping RL-Trader as a verifiers environment worth it? | **No.** It would require reformulating the task as an LLM agent receiving serialised market state as text and being scored by a rubric — a *different project* with different latency, cost and evaluation semantics. The EKG/sentiment layer of this system could *one day* be an LLM task worth publishing; the PPO/SAC execution core cannot |

**Sandboxes / Inference / Hosted Training:** Sandboxes are priced above and are genuinely useful
(disposable reproducible boxes, `prime tunnel` to expose the dashboard). Prime **Inference** is an
OpenAI-compatible LLM API — relevant only to the sentiment/ontology agent, not to trading.
**Hosted Training** ("Lab") trains models *against verifiers environments* — same LLM-only scope;
its `Models & Pricing` page exists but I did not extract a stable rate table → `UNVERIFIED`.
Free credits / academic programme for Prime Intellect: **not found in the public docs** → `UNVERIFIED`.

### F.7 Realistic full-project compute bill

Assumes: CPU sandbox/pod, HPO per E.5, 50 GB persistent disk, W&B academic (free) or free tier.

| Line item | Basis | Cost |
|---|---|---:|
| Baseline replication + data pipeline dev | 20 hr × $0.61/hr (8-core sandbox) | **$12** |
| Stage-1 coarse sweep (30 trials × ~25 min, 16-way parallel) | ~1.0 hr wall × $1.184/hr | **$1.20** |
| Stage-2 focused TPE sweep (80 trials × ~25 min, 16-way) | ~2.1 hr × $1.184/hr | **$2.50** |
| Stage-3 confirmation (5 configs × 10 seeds × 25 min, 16-way) | ~1.3 hr × $1.184/hr | **$1.55** |
| Repeat the above for **SAC** and for **2 walk-forward folds** | ×6 | **$32** |
| Long training runs (final agents, 10 × 4 hr, 8-core) | 40 hr × $0.61/hr | **$24** |
| EKG self-evolution loop experiments | 60 hr × $0.61/hr | **$37** |
| Persistent disk, 50 GB × 3 months | 50 × $0.081 × 3 | **$12** |
| Dashboard host (small sandbox, 200 hr) | 200 hr × $0.31/hr | **$62** |
| Contingency + failed runs | +40 % | **$74** |
| **Total (CPU plan)** | | **≈ $260** |
| *Same plan on an H100 instead* | swap $0.61→$1.90 and $1.184→$2.72 | **≈ $700–800, for no measurable gain** |
| *If an LLM sentiment agent is added later* | Prime Inference / OpenAI-compatible tokens | not estimated — `UNVERIFIED` |

**W&B cost: $0** on the academic licence (200 GB, unlimited tracked hours) or on the Free tier
provided you log **scalars only** and keep checkpoints as reference artifacts.

---

## Verification log

All checks performed **2026-09-14** in this session. `gh-api` = authenticated GitHub REST API
(`/repos/{name}` + `/releases/latest`); `npm-registry` = `https://registry.npmjs.org/<pkg>`;
`fetch` = direct HTTPS GET, status recorded; `verify_url` = `rt.verify_url`;
`verify_arxiv` = `rt.verify_arxiv`.

| # | Claim | URL | Check | Status |
|---:|---|---|---|---|
| 1 | Gated graph render/layout benchmark exists; repo created 2026-08-12, MIT | https://github.com/GusEllerm/viz-bench | gh-api + verify_url | ✅ 200, ★0, pushed 2026-08-20 |
| 2 | Render results (fps, memory) per library | https://raw.githubusercontent.com/GusEllerm/viz-bench/HEAD/graph-bench/web/bench/results/overview-matrix.json | fetch + JSON parse | ✅ 200, 40 records parsed |
| 3 | Layout iters/sec: FA2 CPU 1.61 @100k vs cosmos.gl GPU 39.7 @100k | https://raw.githubusercontent.com/GusEllerm/viz-bench/HEAD/graph-bench/web/bench/results/layout-matrix.json | fetch + JSON parse | ✅ 200, 12 records |
| 4 | Report site | https://gusellerm.github.io/viz-bench/ | verify_url | ✅ 200 |
| 5 | `@cosmos.gl/graph@3.4.1` is **MIT**, published 2026-08-13 | https://registry.npmjs.org/@cosmos.gl/graph | npm-registry | ✅ MIT |
| 6 | `@cosmograph/cosmos@3.4.1` is **CC-BY-NC-4.0**, published 2026-07-31 | https://registry.npmjs.org/@cosmograph/cosmos | npm-registry | ✅ CC-BY-NC-4.0 |
| 7 | `@cosmograph/react@2.5.1` is CC-BY-NC-4.0 | https://registry.npmjs.org/@cosmograph/react | npm-registry | ✅ CC-BY-NC-4.0 |
| 8 | cosmos.gl repo MIT, 1,269★, pushed 2026-09-13, 13 open issues; GPU shaders; `setPointPositions`/`setLinks`; GPU transitions | https://github.com/cosmosgl/graph | gh-api + raw README | ✅ 200 |
| 9 | Old `cosmograph-org/cosmos` repo no longer public | https://api.github.com/repos/cosmograph-org/cosmos | gh-api | ✅ **404** (confirmed absent) |
| 10 | d3-force dormant: last push 2023-12-30, npm 3.0.0 @2021-06-05, ISC | https://github.com/d3/d3-force · https://registry.npmjs.org/d3-force | gh-api + npm-registry | ✅ |
| 11 | sigma 12,164★ pushed 2026-09-14, `sigma@4.0.0-beta.5` 2026-08-20, MIT | https://github.com/jacomyal/sigma.js | gh-api | ✅ |
| 12 | graphology 0.26.0 (2025-01-26) MIT; `graphology-layout-forceatlas2@0.10.1` (2022-10-17) MIT | registry.npmjs.org | npm-registry | ✅ |
| 13 | deck.gl `v9.4.0` 2026-09-05, MIT, 14,588★ | https://github.com/visgl/deck.gl | gh-api | ✅ |
| 14 | Cytoscape.js `v3.34.3` 2026-09-07, MIT; perf page tests 200→20,000 nodes | https://github.com/cytoscape/cytoscape.js · https://cytoscape.org/js-perf/ | gh-api + verify_url | ✅ 200 |
| 15 | AntV G6 `5.1.1`, MIT, 333 open issues | https://github.com/antvis/G6 | gh-api | ✅ |
| 16 | react-force-graph `1.29.1` MIT, 218 open issues; 3d-force-graph 14.2 MB unpacked | https://github.com/vasturiano/react-force-graph · registry.npmjs.org | gh-api + npm-registry | ✅ |
| 17 | Reagraph `4.32.0` Apache-2.0 | https://github.com/reaviz/reagraph | gh-api + npm-registry | ✅ |
| 18 | helios-web `0.10.9` MIT on npm; GitHub shows **no licence field**, 92★ | https://github.com/filipinascimento/helios-web | gh-api + npm-registry | ✅ (licence mismatch noted) |
| 19 | IPC/WebSocket loopback benchmark, 2026-03-04, GH Actions ubuntu-latest, 100k round trips | https://github.com/suenot/trading-ipc-bench | verify_url | ✅ 200 |
| 20 | websockets: ~64 KiB/conn with compression, ~14 KiB without; `max_size` 1 MiB, `max_queue` 16; bufferbloat warning | https://websockets.readthedocs.io/en/stable/topics/memory.html | verify_url | ✅ 200 |
| 21 | websockets broadcast: naive loop lets one slow client block all; built-in broadcast has no per-client backpressure | https://websockets.readthedocs.io/en/stable/topics/broadcast.html | verify_url | ✅ 200 |
| 22 | Starlette thread pool default 40 tokens | https://starlette.dev/threadpool/ | verify_url | ✅ 200 |
| 23 | SSE: 6 connections per browser+origin on HTTP/1.1; HTTP/2 ≈100 streams; `id`/`retry`/`Last-Event-ID` | https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events | verify_url | ✅ 200 |
| 24 | msgspec decode benchmark (77 MiB JSON): msgspec Struct 176.8 ms vs orjson 691.7 ms vs json 868.6 ms | https://msgspec.dev/benchmarks | verify_url | ✅ 200 |
| 25 | `encode/broadcaster` is **ARCHIVED**, last push 2025-04-09 | https://github.com/encode/broadcaster | gh-api + verify_url | ✅ `archived: true` |
| 26 | FastAPI `0.141.1` (2026-07-29), 102,331★, MIT | https://github.com/fastapi/fastapi | gh-api | ✅ |
| 27 | lightweight-charts `v5.2.1` 2026-08-12, Apache-2.0, 17,256★ | https://github.com/tradingview/lightweight-charts | gh-api + npm-registry | ✅ |
| 28 | **Mandatory TradingView attribution** — NOTICE text + link requirement quoted verbatim | https://raw.githubusercontent.com/tradingview/lightweight-charts/master/README.md · .../NOTICE | fetch (raw) | ✅ 200, quoted |
| 29 | uPlot `1.6.32` (2025-03-14) MIT, 533 KB unpacked; project claims 166,650 points in 25 ms | https://github.com/leeoniya/uPlot · https://leeoniya.github.io/uPlot/ | gh-api + verify_url | ✅ 200 |
| 30 | plotly.js `4.1.0`, MIT, 91.5 MB unpacked, 772 open issues | https://github.com/plotly/plotly.js · registry.npmjs.org | gh-api + npm-registry | ✅ |
| 31 | ECharts `6.1.0`, Apache-2.0, 67,315★, 1,501 open issues | https://github.com/apache/echarts | gh-api | ✅ |
| 32 | Recharts `v3.10.1`, MIT; own perf guide warns about large data / rapid change | https://github.com/recharts/recharts · https://recharts.github.io/guide/performance/ | gh-api + verify_url | ✅ 200 |
| 33 | Streamlit `1.63.0` (2026-09-01), Apache-2.0, 1,162 open issues | https://github.com/streamlit/streamlit | gh-api | ✅ |
| 34 | Dash `v4.4.1` (2026-07-21), MIT | https://github.com/plotly/dash | gh-api | ✅ |
| 35 | Panel `v1.9.4` (2026-08-17), BSD-3-Clause, 1,113 open issues | https://github.com/holoviz/panel | gh-api | ✅ |
| 36 | Gradio `6.27.0` (2026-09-11), Apache-2.0 | https://github.com/gradio-app/gradio | gh-api | ✅ |
| 37 | Reflex `v0.9.11`, NiceGUI `v3.16.0`, Rerun `0.37.2` | github.com/{reflex-dev/reflex, zauberzeug/nicegui, rerun-io/rerun} | gh-api | ✅ |
| 38 | **W&B live pricing**: Free $0 / 5 seats / 5 GB-mo; Pro from $60/mo / 10 seats / 100 GB-mo / $0.03 per extra GB; Personal self-host $0, 1 seat, **corporate use not allowed**; **Academic free, 200 GB, unlimited tracked hours, 25 GB/mo Weave, 100 seats** | https://wandb.ai/site/pricing/ | fetch + verify_url | ✅ 200, text quoted |
| 39 | W&B scale guidance 10k runs / 500k steps / 100k cardinality / 1,000 log-calls per min / 100k values per min; HTTP 429 + `RateLimit-*` headers; per-project metric limits; ≥1 s between public-API calls | https://docs.wandb.ai/guides/track/limits/ | fetch + verify_url | ✅ 200, text quoted |
| 40 | W&B sweeps: only `grid`/`random`/`bayes`; **Hyperband is the only early-stopping algorithm** (`min_iter,max_iter,s,eta=3,strict`); `run_cap`; command macros; BO internals undocumented; "scales poorly" | https://docs.wandb.ai/guides/sweeps/sweep-config-keys/ | fetch + verify_url | ✅ 200, text quoted |
| 41 | W&B SDK `v0.30.0` 2026-09-09, MIT | https://github.com/wandb/wandb | gh-api | ✅ |
| 42 | Optuna `v5.0.0` 2026-09-07, MIT, 14,791★, 21 open issues | https://github.com/optuna/optuna | gh-api | ✅ |
| 43 | Ray `ray-2.58.0` 2026-08-23, Apache-2.0, 43,799★ | https://github.com/ray-project/ray | gh-api | ✅ |
| 44 | SB3 `v2.9.0` 2026-06-15 MIT; rl-baselines3-zoo `v2.9.1` 2026-06-15 MIT (500-trial Optuna default, eval every 100k steps, `ortho_init=False` for PPO) | https://github.com/DLR-RM/stable-baselines3 · https://github.com/DLR-RM/rl-baselines3-zoo | gh-api + verify_url | ✅ 200 |
| 45 | CleanRL 10,398★, last push 2026-04-20, licence `NOASSERTION` | https://github.com/vwxyzjn/cleanrl | gh-api | ✅ |
| 46 | SB3 RL Tips page reachable (but contains **no** CPU-vs-GPU guidance — my CPU claim is reasoning, not a citation) | https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html | verify_url + text scan | ✅ 200, term "CPU" absent |
| 47 | arXiv 2006.05990 *What Matters In On-Policy RL?* | arxiv.org/abs/2006.05990 | verify_arxiv | ✅ title matched |
| 48 | arXiv 2005.12729 *Implementation Matters in Deep Policy Gradients* | arxiv.org/abs/2005.12729 | verify_arxiv | ✅ |
| 49 | arXiv 2306.01324 *Hyperparameters in RL and How To Tune Them* | arxiv.org/abs/2306.01324 | verify_arxiv | ✅ |
| 50 | arXiv 2111.08819 *CleanRL* | arxiv.org/abs/2111.08819 | verify_arxiv | ✅ |
| 51 | arXiv 1709.06560 *Deep RL that Matters* | arxiv.org/abs/1709.06560 | verify_arxiv | ✅ |
| 52 | arXiv 2108.13264 *Deep RL at the Edge of the Statistical Precipice* | arxiv.org/abs/2108.13264 | verify_arxiv | ✅ |
| 53 | arXiv 1603.06560 *Hyperband* | arxiv.org/abs/1603.06560 | verify_arxiv | ✅ |
| 54 | arXiv 1907.10902 *Optuna* | arxiv.org/abs/1907.10902 | verify_arxiv | ✅ |
| 55 | arXiv 1807.05118 *Tune* | arxiv.org/abs/1807.05118 | verify_arxiv | ✅ |
| 56 | arXiv 2009.01555 *Sample-Efficient Automated Deep RL* | arxiv.org/abs/2009.01555 | verify_arxiv | ✅ |
| 57 | arXiv 1810.05934 *A System for Massively Parallel Hyperparameter Tuning* (ASHA) | arxiv.org/abs/1810.05934 | verify_arxiv | ✅ |
| 58 | arXiv 1711.09846 *Population Based Training* | arxiv.org/abs/1711.09846 | verify_arxiv | ✅ |
| 59 | arXiv 2412.07165 (HP sensitivity in RL) | — | **not run** | ⚠ **UNVERIFIED — do not cite** |
| 60 | Prime docs index (`llms.txt`, 222 page links) | https://docs.primeintellect.ai/llms.txt | fetch + verify_url | ✅ 200, 31,258 chars |
| 61 | `prime` CLI install/auth/ssh/availability/pods commands incl. `pods status` and `pods terminate` | https://docs.primeintellect.ai/cli-reference/provision-gpu.md · .../introduction.md · .../check-gpu-availability.md | fetch | ✅ 200 ×3, commands quoted |
| 62 | `PrimeIntellect-ai/prime` official CLI+SDK, 326★, MIT, pushed 2026-09-14; `prime env init/push/install/list/info/inspect` | https://github.com/PrimeIntellect-ai/prime | gh-api + raw README | ✅ 200 |
| 63 | **Sandbox CPU-only pricing** $0.05/core/hr, $0.01/GB-RAM/hr, $0.001/GB-disk/hr; example $0.08/hr; VM limits 1–16 cores / ≤64 GB / ≤128 GB; account caps 512 cores | https://docs.primeintellect.ai/sandboxes/overview.md | fetch + verify_url | ✅ 200, quoted |
| 64 | Persistent disks survive termination, attachable to many instances, must match provider+location; cluster ephemeral storage deleted with cluster | https://docs.primeintellect.ai/tutorials-storage/create-persistent-storage.md | fetch | ✅ 200 |
| 65 | Disk price rows $0.00011111 and $0.000097 per GB/hr; disk availability columns | https://docs.primeintellect.ai/cli-reference/check-gpu-availability.md | fetch | ✅ 200, table quoted |
| 66 | FAQ: on-demand vs spot (≤90 % off), per-minute credit deduction, auto-delete on zero credits, no SLA, terminate = data loss | https://docs.primeintellect.ai/faq.md | fetch | ✅ 200 |
| 67 | `prime-rl` algorithms are `grpo, max_rl, rae, hierarchical_grpo, opd, sft, opsd, echo` — no standalone PPO/SAC | https://docs.primeintellect.ai/prime-rl/algorithms.md | fetch + verify_url | ✅ 200, 41,002 chars |
| 68 | `prime-rl` = async agentic LLM RL, FSDP2 + vLLM; `v0.9.0` 2026-08-25, Apache-2.0, 2,038★ | https://github.com/PrimeIntellect-ai/prime-rl · https://docs.primeintellect.ai/prime-rl/overview.md | gh-api + fetch | ✅ |
| 69 | `verifiers` = "library for creating environments to train and evaluate **LLMs**"; `v0.3.1` 2026-08-24, MIT, 4,614★; Hub migrating to v1, v0 deprecated | https://github.com/PrimeIntellect-ai/verifiers · https://docs.primeintellect.ai/tutorials-environments/environments.md | gh-api + fetch | ✅ |
| 70 | Live GPU availability API requires auth | https://api.primeintellect.ai/api/v1/availability/gpus | fetch | ⚠ **403 Not authenticated** — prices below are a browser snapshot, not my own API read |
| 71 | `app.primeintellect.ai/dashboard/create-cluster` GPU price table | https://app.primeintellect.ai/dashboard/create-cluster | fetch | ⚠ 200 but **client-rendered shell, no prices in HTML**. Table in F.2 is a deep-research browser snapshot (2026-09-14) |
| 72 | Homepage GPU cards: H100 $2.43/hr, H100 Spot $0.94/hr, B300 $4.99/hr, B200 $3.49/hr | https://www.primeintellect.ai/ | fetch + HTML parse | ✅ 200, parsed; several "$3.14/HR" cards judged decorative and **not cited** |
| 73 | Prime Hosted Training public rate table; Prime free credits / academic programme | — | not found in docs | ⚠ **UNVERIFIED** |
| 74 | Ogma / KeyLines / yFiles / Graphistry benchmarks & prices; `drkameleon/GraphGPU` | — | not fetched | ⚠ **UNVERIFIED — not used in any recommendation** |
| 75 | npm web pages returned 403 to this client; all package facts come from `registry.npmjs.org` instead | https://registry.npmjs.org/… | npm-registry | ✅ 23 packages read |

**Totals: 73 verified checks (✅), 6 explicitly marked UNVERIFIED (⚠) and excluded from every
recommendation.** Raw deep-research reports are preserved at `docs/research/_raw/06_*.md`
(graph, ui, stream, wandb, prime).
