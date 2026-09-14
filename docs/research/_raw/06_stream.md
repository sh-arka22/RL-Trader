# High-Frequency Streaming for a FastAPI ML Dashboard

## Executive recommendation

Use **one WebSocket per browser session**, with a small application protocol on top of it. Keep the GPU trainer and web process decoupled through an in-process queue for one machine, or Redis Streams/NATS for a small remote deployment. Do not send every graph mutation directly to the browser. Aggregate deltas for **10-30 display frames per second**, batch them, and maintain a bounded per-client queue.

Use binary MessagePack or Protobuf for graph batches if bandwidth or CPU matters. Use JSON with `orjson` or `msgspec` for the first implementation and for scalar control messages. Send learning curves and P&L ticks in the same WebSocket, but give them different stream names and retention policies.

For charts, pick **TradingView Lightweight Charts** for the P&L/equity curve when financial-chart interaction and time-scale behavior matter. Pick **uPlot** for learning curves when the requirement is many numeric series, small bundles, and fast Canvas rendering. Use ECharts instead when the dashboard needs a broad visualization toolbox, data transforms, or a stronger general-purpose React integration.

The most important design fact is that a browser cannot render thousands of graph mutations per second usefully. The server should preserve lossless state internally, while the browser receives a controlled visual projection: the newest coalesced graph state, a fixed-rate scalar stream, and a lossless or sampled financial stream.

## A. Transport comparison

### What the published measurements actually show

The most useful reproducible number found for this workload is the `suenot/trading-ipc-bench` loopback benchmark. It was updated **2026-03-04 22:45 UTC**, ran on GitHub Actions `ubuntu-latest`, discarded 1,000 warmups, measured 100,000 round trips for most transports, used `time.perf_counter_ns()`, isolated each transport in a subprocess, and tested 64, 256, and 1,024 byte messages. The repository does not specify a physical CPU model, so these are comparative loopback numbers, not a production capacity guarantee. [8] [8]

For its 64-byte case, the complete reported table is:

| Transport | p50 us | p95 us | p99 us | p99.9 us | messages/s |
|---|---:|---:|---:|---:|---:|
| Unix domain socket | 20.2 | 31.1 | 42.6 | 65.5 | 46,512 |
| TCP | 41.7 | 54.6 | 67.2 | 78.0 | 22,425 |
| ZeroMQ TCP | 137.6 | 158.9 | 170.7 | 186.3 | 7,149 |
| WebSocket | 292.5 | 345.5 | 401.6 | 468.4 | 3,345 |
| HTTP REST | 296.8 | 366.4 | 385.3 | 419.8 | 3,249 |
| NATS | 411.4 | 466.6 | 527.2 | 712.4 | 2,394 |
| Redis Pub/Sub | 513.0 | 569.8 | 617.8 | 809.7 | 1,957 |
| Redis Streams | 665.9 | 772.4 | 844.2 | 999.5 | 1,491 |

These numbers are explicitly reported by the benchmark, including the units and message rate. [8] They support two practical conclusions: WebSocket framing is materially more expensive than raw UDS/TCP in this particular Python loopback test, but it is close to HTTP REST for tiny messages; and Redis Streams trades latency for persistence and replay. [8]

A separate Node.js repository implements latency, broadcast, and scalability tests for SSE and WebSocket with the same JSON payload and configurable message rate. Its default is one message per second, and its scalability modes use 100, 500, 1,000, 5,000, and 10,000 clients. The repository contains one example result rather than a complete protocol comparison: 1,000 WebSocket clients, 64-byte messages, 60,000 received messages, 18 ms average latency, 5 ms minimum, 40 ms maximum, 350 MB memory, 25% CPU, and 2 ms event-loop lag. It does not state the hardware or execution date, so it should not be treated as a universal benchmark. [12] [12] [12] [12]

Another published comparison advertises measurements of wire bytes and latency at a 50 ms round trip, but its retrieved page does not expose a complete result table in the accessible text. Treat it as a methodology lead, not as evidence for exact throughput. [35]

There is no credible, apples-to-apples, FastAPI-versus-Starlette-versus-Uvicorn WebSocket throughput table in the retrieved primary sources. FastAPI itself says its benchmark page is based on independent TechEmpower HTTP benchmarks and explains the stack hierarchy: Uvicorn is the server, Starlette is the framework on top, and FastAPI adds validation and serialization. Those HTTP request benchmarks are not WebSocket broadcast benchmarks and should not be relabeled as such. [10]

### Protocol matrix

| Transport | Direction and framing | Strengths | Failure mode under high-frequency data | Best use here |
|---|---|---|---|---|
| WebSocket | Full duplex; text or binary frames | One long-lived connection, browser-native, client commands, subscriptions, acknowledgements, binary payloads | A slow browser can accumulate server and TCP buffers; naive broadcast can serialize on the slowest client | Default transport for graph deltas plus controls, scalars, and P&L |
| SSE | Server to browser only; UTF-8 text event stream | Very simple browser API, automatic reconnect, `id`, `retry`, `Last-Event-ID`, HTTP infrastructure | Text-only application format, proxy buffering, one-way semantics, HTTP/1.1 connection limit | Read-only progress, logs, alerts, and a low-rate scalar feed |
| Long polling | Repeated HTTP requests held until an event | Works through restrictive infrastructure and older clients | Request/response and header overhead for every cycle; reconnect races and duplicate delivery are application concerns | Compatibility fallback, not a kHz stream |
| Plain polling | Periodic independent HTTP requests | Simplest operational model and easiest caching/testing | Latency is at least the polling interval, plus request overhead; stale bursts and thundering herds | A local single-user dashboard with updates every few seconds, or a fallback snapshot API |

### SSE details that matter

SSE is a one-way connection: the browser cannot send events back on the same connection. [4] The browser automatically restarts a closed connection by default. An event can carry an `id`, and `retry` specifies the reconnect delay in integer milliseconds. [4] [4] This is a real advantage over WebSocket, where the application normally implements reconnect, backoff, authentication refresh, and state recovery itself.

Under HTTP/1.1, browsers commonly limit SSE to **six open connections per browser and domain**, including across tabs. Under HTTP/2, the limit becomes the negotiated maximum number of HTTP streams, with 100 cited as the default in MDN's documentation. [4] HTTP/2 removes the browser's six-connection bottleneck, but it does not remove application limits, proxy idle timeouts, server file-descriptor limits, or the cost of maintaining many streams.

SSE is preferable when the stream is genuinely server-to-client, events are text-friendly, and reconnecting from a cursor is more valuable than binary efficiency. Examples are training phase changes, status messages, occasional scalar metrics, and server logs. It is not the right primary transport for a force-directed graph that needs client commands, dynamic subscriptions, binary batches, or explicit flow-control feedback.

For production SSE, send `Cache-Control: no-cache`, heartbeat comments, and an anti-buffering response header such as `X-Accel-Buffering: no`. FastAPI's SSE documentation describes a keepalive ping every 15 seconds to prevent proxies from closing an idle stream and explicitly calls out cache and proxy-buffering behavior. [13] Also test the whole path: Nginx, an ingress controller, a cloud load balancer, and the browser can each buffer data even if the Python generator yields immediately. Flush-sized event batches and heartbeats are operational requirements, not micro-optimizations.

### WebSocket framing, compression, and flow control

WebSocket supports text and binary frames. Text is convenient and inspectable; binary is preferable for packed numeric arrays, MessagePack, Protobuf, or typed-array records. Per-message-deflate can reduce bandwidth for repetitive JSON and graph identifiers, but it consumes CPU and increases per-connection memory. The current `websockets` documentation reports approximately 64 KiB per connection with default compression settings and approximately 14 KiB when compression is disabled. [6]

Compression should therefore be measured against the actual payload. Do not compress already packed numeric data automatically. For small 10-30 Hz batches, compression overhead may cost more than it saves; for repetitive JSON graph deltas across a remote link, it can be worthwhile. Disable it on a trusted high-bandwidth LAN if CPU and tail latency are more important than bandwidth.

The `websockets` library documents a write buffer and a `write_limit`; when the high-water mark is reached, `send()` waits until the buffer drains below the low-water mark. [6] It also warns that large buffers delay backpressure and create bufferbloat, which increases latency rather than improving throughput. [6] The incoming queue defaults cited by the documentation are `max_size=1 MiB` and `max_queue=16`, which can imply up to 16 MiB for queued incoming frames before application objects and other buffers. [6]

For thousands of broadcast clients, the library's broadcast documentation makes a crucial distinction: a serialized naive loop can block all clients when one write buffer fills, while an independent broadcast lets a slow client stop blocking other clients. The built-in broadcast path does not apply backpressure to every client; its recommended operational response is generally to disconnect clients that fall too far behind, or use per-client queues when slow-client handling is important. [7] [7] [7]

## B. FastAPI and Starlette implementation

### A minimal endpoint and connection manager

FastAPI's WebSocket API is provided by Starlette. The official example accepts a connection, receives messages, sends messages, catches `WebSocketDisconnect`, and keeps a list of active connections in a `ConnectionManager`. [2] [2] [2] [2]

A production version should not have one global list with one unbounded broadcast loop. Use a client record with a bounded queue and a dedicated sender task per client:

```python
import asyncio
from dataclasses import dataclass
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

@dataclass
class Client:
    ws: WebSocket
    queue: asyncio.Queue
    dropped: int = 0

clients: set[Client] = set()

async def sender(client: Client) -> None:
    while True:
        frame = await client.queue.get()
        await client.ws.send_bytes(frame)
        client.queue.task_done()

@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    client = Client(ws=ws, queue=asyncio.Queue(maxsize=4))
    clients.add(client)
    task = asyncio.create_task(sender(client))
    try:
        while True:
            command = await ws.receive_json()
            await handle_command(client, command)
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(client)
        task.cancel()
```

The code is intentionally incomplete around cancellation and authentication, but it shows the separation that matters: the producer never writes directly to a socket. It deposits a bounded frame into a client queue, and only that client's sender performs network I/O.

For a broadcast, choose an explicit overload policy:

```python
from asyncio import QueueFull

def offer(client: Client, frame: bytes, kind: str) -> None:
    try:
        client.queue.put_nowait(frame)
    except QueueFull:
        client.dropped += 1
        if kind == "graph":
            # Drop an intermediate visual frame. The next frame is a snapshot
            # or a coalesced state, so the browser can recover.
            try:
                client.queue.get_nowait()
                client.queue.task_done()
                client.queue.put_nowait(frame)
            except QueueFull:
                pass
        elif kind == "scalar":
            # Replace the pending scalar with the newest value.
            pass
        elif kind == "pnl":
            # Prefer disconnect or durable replay, not silent loss.
            client.queue = asyncio.Queue(maxsize=4)

async def broadcast(frame: bytes, kind: str) -> None:
    for client in tuple(clients):
        offer(client, frame, kind)
```

In real code, replacing `client.queue` requires synchronization with the sender; an easier implementation uses separate queues or a small deque protected by an async lock. The policy is more important than the exact container: graph frames can be conflated, scalar metrics can be latest-value-wins, and P&L ticks should be lossless or recoverable from a sequence-numbered log.

### Do not block the event loop

An `async def` endpoint runs on the event loop. A synchronous CPU-heavy graph transform, blocking file read, model callback, or synchronous Redis client inside it prevents other sockets from being serviced. Starlette runs synchronous endpoint and background-task code through its thread pool specifically to avoid blocking the event loop, but the documented default thread-pool capacity is only 40 tokens. [19]

Use `await` for async I/O, move small blocking calls to `await asyncio.to_thread(...)` or Starlette's thread-pool helper, and move substantial CPU work to a process, the GPU worker, or a separate service. Do not solve a CPU-bound callback by adding more async syntax. The GIL and event-loop scheduling still apply.

### Backpressure patterns

1. **Bound the queue.** Use `asyncio.Queue(maxsize=N)` and `put_nowait`. Catch `QueueFull`; never allow an unbounded per-client queue.
2. **Drop oldest visual frames.** For graph rendering, an old intermediate frame has little value if a newer coalesced frame supersedes it.
3. **Conflate.** Keep one pending graph state per client. Merge new node and edge additions by ID, or replace the pending frame with a snapshot-plus-new-delta.
4. **Throttle at the display rate.** A 10-30 Hz server-side frame loop is usually more useful than attempting one browser callback per training event.
5. **Batch.** Emit one frame containing hundreds or thousands of graph changes, rather than one WebSocket frame per node or edge.
6. **Separate priority lanes.** Control, errors, and P&L sequence markers should not be dropped because a graph frame is large.
7. **Disconnect hopelessly slow clients.** The `websockets` broadcast guidance explicitly notes that synchronizing all clients to the slowest one performs poorly at scale. [7]
8. **Measure queue age.** A queue length of two can still be unhealthy if each frame takes seconds to send. Track oldest-frame age, bytes pending, send duration, dropped frames, and last acknowledged sequence.

A fixed-rate coalescer can be as simple as:

```python
latest_graph = None
latest_scalars = {}

async def frame_pump():
    global latest_graph
    period = 1.0 / 20.0  # 20 visual frames per second
    while True:
        await asyncio.sleep(period)
        graph = latest_graph
        latest_graph = None
        frame = make_frame(
            graph_delta=graph,
            scalars=latest_scalars.copy(),
            pnl_ticks=drain_pnl_batch(),
        )
        if frame:
            await fanout_with_queue_policy(frame)
```

The trainer can push into an internal bounded queue at event rate, while `frame_pump` controls browser rate. For a graph, merge node and edge IDs before serialization. For scalars, retain only the newest value for each metric in the current frame. For P&L, batch ticks but preserve their sequence numbers.

### Broadcasting choices

For one Python process, an in-memory registry plus per-client queues is the lowest-latency and simplest solution. For multiple Uvicorn workers or multiple hosts, an in-memory list is not enough: each worker sees only its own clients. Use Redis Pub/Sub for ephemeral fan-out, Redis Streams for replay and consumer groups, or NATS for a lightweight internal bus. A broadcaster-style abstraction can hide the backend, but it does not remove the need for per-client queues and overload policy.

The `websockets` documentation describes publish-subscribe and per-client queues as ways to support slow clients without duplicating every message into every socket write buffer. [7] Redis Streams are append-only structures that support random access, consumer groups, and real-time syndication. [14] Kafka is the heavier option when durable, replayable, partitioned event streams and multiple independent consumers are first-class requirements. [16]

## Serialization and message format

### Practical comparison

| Format | Best property | Cost | Graph-batch recommendation |
|---|---|---|---|
| JSON | Debuggable, browser-native, easy FastAPI integration | Verbose strings and repeated keys; parse and allocation cost | Start here for control and scalar messages; use `orjson` or `msgspec` |
| MessagePack | Compact binary and simple schema evolution | Browser decoder and explicit type conventions | Good default for graph batches when both ends are under your control |
| Protobuf | Compact, typed, explicit compatibility rules, generated code | Schema/tooling overhead; browser build integration | Best when the protocol will be shared by many services or languages |
| Apache Arrow / typed arrays | Excellent columnar numeric density and zero-copy-friendly layouts | Poor fit for arbitrary nested graph objects; more complex incremental protocol | Use for large numeric arrays, embeddings, coordinates, or bulk snapshots |

The msgspec benchmark is unusually transparent about its limitations: it ran on an approximately 2020 x86 Linux laptop with CPython 3.11, and it warns that tight loops keep instruction caches hot and branches predictable. It provides relative results rather than a claim about every application. [26]

For a large approximately 77 MiB JSON file, the reported decode results were: msgspec Structs 176.8 ms and 67.6 MiB additional memory; plain msgspec 630.5 ms and 218.3 MiB; `orjson` 691.7 ms and 406.3 MiB; stdlib `json` 868.6 ms and 295.0 MiB; `ujson` 1,087.0 ms and 349.1 MiB. [26] In the structured encode/decode/validation benchmark, msgspec was reported as approximately 6x faster than mashumaro, 10x faster than cattrs, 12x faster than Pydantic V2, and 85x faster than Pydantic V1 for that payload and benchmark version. [26]

These are not WebSocket throughput numbers. They indicate that serialization can become a bottleneck before the socket does, and that a typed `msgspec.Struct` or generated Protobuf object is worth considering for hot paths. Benchmark your real node and edge distributions, because a graph with many repeated IDs benefits more from compact dictionaries or integer IDs than from changing JSON libraries.

A useful wire layout for a graph batch is columnar rather than one object per edge:

```text
{
  "type": "graph.delta",
  "seq": 184203,
  "base": 184000,
  "nodes": {"id": [..], "x": [..], "y": [..], "label": [..]},
  "edges": {"src": [..], "dst": [..], "weight": [..]},
  "removed_nodes": [..],
  "removed_edges": [..]
}
```

In binary form, encode IDs as unsigned integers, coordinates and weights as `float32` where precision permits, and keep strings in a side dictionary. Do not send repeated labels on every edge. For a first version, MessagePack arrays or JSON arrays are easier to debug; migrate only after measuring bytes per frame and serialization CPU.

## Reconnect and state resynchronization

Treat delivery as at-least-once and make every message idempotent. Use a monotonically increasing `seq` per stream, not one global sequence if graph, scalar, and P&L retention differ.

Recommended protocol:

1. Client connects with `last_seq` values for each stream.
2. Server sends `hello` with protocol version, current snapshot IDs, and retention window.
3. If the requested sequence is still in the ring buffer, server sends missing deltas in order.
4. If it is too old, server sends a compressed snapshot followed by deltas generated after the snapshot.
5. Client applies only `seq > last_seq`; duplicate frames are harmless.
6. Client acknowledges the highest applied sequence periodically.
7. Server records lag and disconnects or downgrades a client that cannot catch up.

The ring buffer can be an in-memory deque for a single process, Redis Streams for a small multi-process deployment, or Kafka for durable multi-consumer history. Keep snapshots independently addressable. A graph snapshot should be a complete authoritative node/edge state, while scalar and P&L snapshots should include the last value, timestamp, and sequence.

## C. Internal hop and local-file alternatives

| Internal mechanism | Choose it when | Avoid it when |
|---|---|---|
| In-process `asyncio.Queue` | One trainer and one web process on one host | Processes can restart independently or scale separately |
| Redis Pub/Sub | Low-latency ephemeral fan-out to a few web workers | You need replay after disconnect |
| Redis Streams | Replay, consumer groups, and a small operational footprint | The stream will become a very large durable data platform |
| ZeroMQ | Very low-latency process-to-process messaging and explicit topologies | You need broker-managed persistence and consumer recovery |
| NATS/JetStream | Lightweight subjects, request/reply, and optional persistence | The team already operates Kafka and needs Kafka ecosystem tooling |
| Kafka | Durable partitioned history, replay, many consumers, and large-scale event pipelines | A single-user dashboard where Kafka is operational overkill |
| SQLite/parquet/file polling | One local user, low update rate, easy offline inspection | Thousands of events/s, remote access, or real-time interaction |

A file or SQLite polling loop is perfectly reasonable for a single-user local dashboard if the visual requirement is one update every few seconds and the trainer already writes checkpoints. Use atomic replace or a transaction, not a file that is being partially written. Parquet is excellent for post-run analysis and historical curves, but it is not a low-latency incremental transport. For real-time updates, append to a durable log or queue and periodically materialize snapshots to SQLite or Parquet.

## D. Recommended architecture and capacity model

```text
GPU training process
    |
    | in-process queue, Redis Streams, or NATS subject
    v
Aggregator / state store
    |-- graph state + snapshot + recent graph deltas
    |-- scalar latest values
    |-- P&L ring buffer and durable writer
    v
FastAPI / Starlette / Uvicorn
    |-- WebSocket endpoint
    |-- per-client bounded queues
    |-- 20 Hz coalescing frame pump
    v
React client
    |-- Web Worker decodes and merges frames
    |-- graph renderer applies one batch per animation frame
    |-- chart components receive typed arrays or append batches
```

Use one WebSocket message envelope:

```json
{
  "v": 1,
  "stream": "graph|metrics|pnl|control",
  "kind": "snapshot|delta|batch|error|heartbeat",
  "seq": 184203,
  "ts_ns": 1770000000000000000,
  "base_seq": 184000,
  "encoding": "json|msgpack|protobuf",
  "payload": {}
}
```

The graph payload is a batched delta with integer IDs and optional tombstones. The metrics payload is latest-value-wins by metric name. The P&L payload is an ordered array of ticks, each with `seq`, event time, price, cash, position, equity, and realized/unrealized P&L. Control messages include `subscribe`, `pause_graph`, `request_snapshot`, `ack`, and `set_rate`.

For one or a few users, start with 20 frames/s, a queue depth of 2-4 frames, batches of 50-5,000 graph mutations depending on payload size, and WebSocket compression disabled on a fast LAN. Enable compression for a remote link only after measuring CPU and tail latency. Put graph rendering and binary decoding in a Web Worker if the graph is large.

The benchmark evidence gives a lower-bound planning anchor rather than a promise: generic Python WebSocket loopback measured 3,345 64-byte round trips/s at p50 292.5 us, while raw TCP measured 22,425/s. [8] A browser-facing system has TLS, network RTT, serialization, browser parsing, React scheduling, and rendering costs, so do not convert 3,345 into a claimed FastAPI capacity. A sensible initial target is **20 WebSocket frames/s per client**, with each frame carrying 50-5,000 logical graph changes. That is 1,000-100,000 logical changes/s at the application level, but it is an engineering target that must be load-tested with the actual payload and renderer, not a published benchmark result.

Measure with: bytes per frame, encode time, send wait time, queue age, p50/p95/p99 end-to-end latency, dropped graph frames, reconnect recovery time, snapshot duration, browser main-thread time, worker time, and GPU/CPU utilization. Load-test at the expected number of clients and include one deliberately slow browser.

## Charting libraries for live financial and ML curves

### Evidence and maintenance table

| Library | Exact licence | Live update path | Performance evidence | Bundle / React / maintenance evidence |
|---|---|---|---|---|
| TradingView Lightweight Charts | Apache License 2.0 / Apache-2.0. The current npm package reports 5.2.1, published 15 days before the retrieval. [30] [30] | Use the series update API for a new bar/tick and set-data for replacement snapshots; use a thin React component wrapper or manage the chart instance in `useEffect` | Financial-focused Canvas implementation; the retrieved sources do not provide a reproducible maximum point count | Numeric package size was not stated in the retrieved npm page. Current release activity is strong: 66 versions and a recent publish. [30] [30] It requires TradingView attribution; the docs say to specify TradingView as product creator. [5] The NOTICE/link/logo requirement must be retained in the product's attribution or open-source notices. |
| uPlot | MIT. [31] | Use `setData` for replacement and the library's streaming/live-update integration or append-oriented application code; keep data in typed, columnar arrays | The project advertises an interactive 166,650-point chart in 25 ms and roughly 100,000 points/ms afterward; treat this as a project benchmark, not an independent lab result. [11] | Approximately 45 KB min. [31] Third-party React integrations exist. [31] Version 1.6.32 was reported as published a year before retrieval, so pin and test rather than assuming rapid release cadence. [31] |
| Plotly.js plus react-plotly.js | Plotly.js MIT. [28] The React wrapper is MIT. [33] | `Plotly.extendTraces` or `Plotly.react`; the wrapper updates through `Plotly.react` and exposes `onUpdate`. [33] | Plotly documents around one million points for WebGL-enabled traces; larger datasets may need Datashader or server-side aggregation. [23] | The React wrapper reports approximately 6 MB unminified and just over 2 MB minified. [33] Plotly.js 3.5.0 was reported as published 17 days before retrieval. [28] Strong feature set, but heavier than uPlot or Lightweight Charts. |
| Apache ECharts plus echarts-for-react | ECharts Apache License 2.0. [29] The `echarts-for-react` wrapper is MIT. [34] | `setOption` for incremental option changes; use ECharts APIs through `getEchartsInstance()` in the React wrapper. [34] | Supports large-data modes and broad series types, but no single credible maximum interactive point count was exposed in the retrieved sources | The official docs recommend tree-shakeable imports to reduce bundle size. [25] ECharts 6.1.0 was reported as published 3 months before retrieval. [29] The wrapper is a real React wrapper, not merely a generic DOM adapter. [34] |
| Recharts | MIT. [32] | React state/props updates redraw declarative SVG components; there is no dedicated high-rate append API in the retrieved package evidence | Official guidance says it is efficient for common cases but asks users with large datasets or rapid changes to isolate charts, reduce animation, and optimize rendering. [20] A project issue reports 4 seconds for more than 3 MB in one environment, which is a warning rather than a benchmark standard. [36] | Bundle size was not stated in the retrieved npm page. Version 3.10.1 was reported as published 2 months before retrieval, with a last publish 2 days before retrieval. [32] Excellent React ergonomics, but SVG and declarative rerendering make it a poor first choice for high-frequency, large time series. |
| Perspective | The retrieved Perspective landing page did not state an exact licence or current release metadata, so verify the repository and package before adopting it. | Streaming table/view updates and client-side analytical views are its central model | The project describes an interactive analytics and visualization component suited to large and streaming datasets. [27] | Strong candidate for pivotable live tables and multidimensional training telemetry, but not the simplest equity-line component. |

The table contains a deliberate distinction between measured evidence and project claims. There is no honest universal maximum data-point count at 60 fps: point count depends on series count, line simplification, markers, axes, browser, device, interaction, update strategy, and whether the chart redraws the whole data array. Use the cited numbers as orientation, then benchmark the exact browser and chart configuration.

### Picks

**P&L and equity curve: Lightweight Charts.** It is purpose-built for financial time series, has the right interaction model for time scales and crosshairs, has a small client-side footprint relative to Plotly, and has a current 5.2.1 release signal. [30] [30] It has the important legal caveat: Apache-2.0 does not mean "no attribution" for this project. Include the required TradingView creator notice, link/logo treatment, and the repository NOTICE file in the product's attribution page. [5]

Use one line for equity, optional lines for benchmark and drawdown, and call the series update method once per display frame with a batch of ticks. Keep raw ticks in the server ring buffer and downsample the visible range. Do not push every tick into React state.

**Learning curves: uPlot.** It has the strongest combination of a small bundle, Canvas rendering, many numeric points, and a published point-count performance claim. [11] [11] Keep each metric in a columnar array, update once per frame, and downsample older points by min/max or LTTB before sending them to the browser. For logarithmic loss or learning-rate curves, use a log y-axis and make the server tell the client whether the metric is positive and log-safe.

Choose **ECharts** instead of uPlot when the learning dashboard needs heatmaps, scatter plots, parallel coordinates, data zoom, visual mappings, or a larger component ecosystem. Choose Plotly when scientific interactivity and WebGL traces outweigh bundle size. Avoid Recharts for the hot path; retain it for small summary charts where React composition is the primary requirement.

## Synthesis

The transport and chart choices should be asymmetric. WebSocket is the right control and data-plane connection because the dashboard needs subscriptions, pause/resume, acknowledgements, binary batches, and explicit recovery. SSE is operationally attractive for a separate low-rate, read-only status channel because the browser supplies reconnect and event IDs, but its one-way text model and HTTP/1.1 connection behavior are poor defaults for high-frequency graph state. [4] [4]

The internal hop should be chosen by recovery requirements, not by an abstract latency leaderboard. Raw UDS/TCP wins the loopback benchmark, WebSocket is slower but browser-compatible, Redis Streams is slower still but supplies an append-only replayable log, and Kafka is justified only when durable multi-consumer history is a product requirement. [8] [14]

The rendering strategy is more important than the nominal wire protocol. A WebSocket that sends 3,000 individual frames per second is worse than a 20 Hz WebSocket that carries 150 coalesced updates per frame. The backpressure policy is the real architecture: bounded queues, latest-value-wins metrics, conflated graph frames, recoverable P&L sequences, and disconnects for clients that cannot catch up. The `websockets` documentation's warning about bufferbloat and slow-client broadcast behavior directly supports this design. [6] [7]

Finally, chart selection follows workload shape. Lightweight Charts optimizes financial interaction, uPlot optimizes dense time-series throughput and bundle size, ECharts optimizes breadth, Plotly optimizes scientific feature depth, and Recharts optimizes React-native composition. Do not select one library for every panel. Keep the transport and state protocol shared, but let each visual consumer receive a projection suited to its frame budget.

## References

1. *Interface: LayoutOptions | Lightweight Charts*. https://tradingview.github.io/lightweight-charts/docs/api/interfaces/LayoutOptions
2. *WebSockets - FastAPI*. https://fastapi.tiangolo.com/advanced/websockets
3. *Websockets - Uvicorn*. https://uvicorn.dev/concepts/websockets/
4. *Using server-sent events - Web APIs | MDN*. https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events
5. *Getting started | Lightweight Charts*. https://tradingview.github.io/lightweight-charts/docs/4.1
6. *Memory and buffers*. https://websockets.readthedocs.io/en/stable/topics/memory.html
7. *Broadcasting*. https://websockets.readthedocs.io/en/stable/topics/broadcast.html
8. *GitHub - suenot/trading-ipc-bench: IPC latency benchmark: TCP, UDS, ZeroMQ, WebSocket, Redis, Shared Memory, Named Pipe — round-trip latency & throughput for algo trading · GitHub*. https://github.com/suenot/trading-ipc-bench
9. *lightweight-charts/NOTICE at master · tradingview/lightweight-charts · GitHub*. https://github.com/tradingview/lightweight-charts/blob/master/NOTICE
10. *Benchmarks - FastAPI*. http://fastapi.tiangolo.com/benchmarks
11. *uPlot | 📈 A small, fast chart for time series, lines, areas, ohlc & bars*. https://leeoniya.github.io/uPlot/
12. *GitHub - muratdemirci/sse-vs-websockets: Benchmark suite comparing Server-Sent Events (SSE) and WebSocket performance - latency, throughput, scalability, and reconnection · GitHub*. https://github.com/muratdemirci/sse-vs-websockets
13. *Server-Sent Events (SSE) - FastAPI*. https://fastapi.tiangolo.com/tutorial/server-sent-events/
14. *Redis Streams*. https://redis.io/docs/latest/develop/data-types/streams/
15. *GitHub - apache/echarts: Apache ECharts is a powerful, interactive charting and data visualization library for browser · GitHub*. https://github.com/apache/echarts
16. *Introduction | Apache Kafka*. https://kafka.apache.org/intro/
17. *GitHub - plotly/plotly.js: Open-source JavaScript charting library behind Plotly and Dash · GitHub*. https://github.com/plotly/plotly.js
18. *GitHub - recharts/recharts: Redefined chart library built with React and D3 · GitHub*. https://github.com/recharts/recharts
19. *Thread Pool - Starlette*. https://starlette.dev/threadpool/
20. *Performance | Recharts*. https://recharts.github.io/guide/performance/
21. *Function reference in JavaScript*. https://plotly.com/javascript/plotlyjs-function-reference/
22. *GitHub - leeoniya/uPlot: 📈 A small, fast chart for time series, lines, areas, ohlc & bars · GitHub*. https://github.com/leeoniya/uPlot
23. *High performance visualization in Python*. https://plotly.com/python/performance/
24. *GitHub - tradingview/lightweight-charts: Performant financial charts built with HTML5 canvas · GitHub*. https://github.com/tradingview/lightweight-charts
25. *Import ECharts - Basics - Handbook - Apache ECharts*. https://echarts.apache.org/handbook/en/basics/import/
26. *Benchmarks*. https://msgspec.dev/benchmarks
27. *Perspective | Perspective*. https://perspective.finos.org/
28. *plotly.js - npm*. https://www.npmjs.com/package/plotly.js
29. *echarts - npm*. https://www.npmjs.com/package/echarts
30. *lightweight-charts - npm*. https://www.npmjs.com/package/lightweight-charts
31. *uplot - npm*. https://www.npmjs.com/package/uplot
32. *recharts - npm*. https://www.npmjs.com/package/recharts
33. *GitHub - plotly/react-plotly.js: A plotly.js React component from Plotly 📈 · GitHub*. https://github.com/plotly/react-plotly.js
34. *GitHub - hustcc/echarts-for-react: ⛳️  Apache ECharts components for React wrapper. 一个简单的 Apache echarts 的 React 封装。 · GitHub*. https://github.com/hustcc/echarts-for-react
35. *WebSocket vs SSE vs Long Polling: Measured | The Infinity*. https://theinfinity.dev/articles/websocket-vs-sse-vs-polling
36. *Recharts is slow with large data · Issue #1146 · recharts ...*. https://github.com/recharts/recharts/issues/1146
