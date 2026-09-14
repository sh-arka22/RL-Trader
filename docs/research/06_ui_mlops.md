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
