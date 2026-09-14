# Choosing a Streaming Force Graph Stack at 100K Nodes

## Executive Summary

- **Scale Reality**: A GPU renderer can draw hundreds of thousands or millions of primitives, but a live force solve is a separate problem. Cosmograph advertises multi-million-node GPU graphs, while cosmos.gl describes real-time hundreds-of-thousands-point/link simulation on modern hardware [9][8]. -> Treat those as hardware-dependent rendering/layout claims, not a guaranteed 30 FPS SLA.
- **Best Fit**: Cosmograph / `@cosmograph/cosmos` is the strongest technical fit because its force layout and rendering run on the GPU, and its public material explicitly targets multi-million-node graphs [9][8]. -> Use it as the primary candidate, subject to a device-specific benchmark.
- **Best React-Friendly Fallback**: `react-force-graph-2d` or `3d-force-graph` is the fastest route to an attractive, interactive product. Its docs expose incremental `graphData`, warmup/cooldown controls, canvas or Three.js rendering, and a documented large-graph example around 75K elements [4][6]. -> Use it when visual polish and integration speed matter more than 100K-node physics.
- **CPU Layout Ceiling**: d3-force uses a CPU velocity-Verlet simulation and recommends a Web Worker for large static layouts [3]. -> It is excellent below roughly 5K to 10K active nodes, but should not be the 100K-node live solver.
- **Rendering Is Not Layout**: Sigma.js is explicitly aimed at graphs of thousands of nodes and edges, and deck.gl can render very large point/line buffers, but neither is a complete live force engine [14][15][17]. -> Pair them with an external worker or GPU solver and render only a display subgraph.
- **React Boundary**: Imperative graph instances are safer than putting one React component in the tree per node. The official react-force-graph packages are standalone components with an imperative graph surface [4]. -> Keep graph state in a worker or mutable store, and pass batched snapshots to one canvas/WebGL component.
- **100K Visual Quality**: Showing every edge at once produces an unreadable hairball even when the GPU can draw it. -> Hide, bundle, aggregate, or sample edges and freeze old regions; reserve full-resolution display for a selected subgraph.
- **Evidence Limitation**: The public sources found here do not provide reproducible, hardware-labelled >=30 FPS node-and-edge ceilings for most libraries. -> The numeric breakpoints below are engineering gates, not universal guarantees; benchmark on the target GPU and edge density.

## How to Read the Scale Numbers

A published "large graph" example is not automatically a 30 FPS result. I separate four quantities: renderer capacity, layout capacity, update cost, and usable visual density. A WebGL or WebGPU renderer may upload 100K positions cheaply while a CPU force solver still spends too much time per tick, and a graph that technically renders at 60 FPS may be visually unusable because every edge crosses every other edge.

The sources are unusually clear about this distinction in places. d3-force says that large static layouts should run in a Web Worker to avoid freezing the UI [3]. The force-graph documentation exposes a large example around 75K elements, but it does not promise a 30 FPS continuously reheated simulation at that size [6]. Cosmograph and cosmos.gl make the strongest GPU scale claims, but they qualify them by device hardware [9][8].

The breakpoints in the tables are therefore conservative product gates: the point at which I would stop trying to show the entire live graph and switch to LOD or a display subgraph. They are not claims that the library crashes at that exact node count.

## Per-Library Technical Comparison

### d3-force

| Criterion | Current evidence and engineering interpretation |
|---|---|
| Rendering and practical scale | d3-force does not render. It supplies coordinates to SVG, Canvas, WebGL, or another renderer. No official >=30 FPS maximum was found. Engineering gate: about **2K-5K active CPU-simulated nodes**, with **10K** possible only after simplifying forces, reducing tick work, and rendering with Canvas/WebGL. |
| Layout capacity | CPU velocity-Verlet integrator; default natural run is 300 ticks. The docs explicitly recommend a Worker for large static layouts [3]. Barnes-Hut/quadtree approximation is available through `forceManyBody` [45]. |
| Incremental updates | Call `simulation.nodes(newArray)` after adding/removing nodes; it reinitializes bound forces. Reheat with `alpha`, `alphaTarget`, and `restart` [2]. This is incremental in API shape, but force initialization and subsequent ticks touch the complete node set. |
| Physics | Link, many-body, center, collision, radial, position and custom forces; velocity decay; 2D core. `d3-force-3d` is a separate package, not a property of d3-force itself. |
| License | **ISC** [2]. No commercial restriction identified. |
| Bundle | npm reports **89.6 kB unpacked** for d3-force, but the captured Bundlephobia result did not expose minified+gzip size [2]. Measure the exact imported modules in the application bundle. |
| Maintenance | npm page showed **3.0.0**, published approximately **5 years ago** [2]. GitHub showed 268 commits and ISC license [1]; the captured page did not expose an issue count or release date. |
| React | No official React wrapper. React should own the container and controls, not individual nodes; use `useRef`, `useEffect`, and a worker or mutable simulation object. |

**Decision:** d3-force is the best low-level CPU building block and the clearest incremental API, but it is not a credible whole-graph 100K live solver. Break down for the stated use case at roughly **5K active fully simulated nodes**, or earlier if edges are dense and the simulation never cools.

### Vasturiano force-graph family

| Package | Rendering/layout scale evidence | Incremental API and physics | License, bundle, maintenance, React |
|---|---|---|---|
| `force-graph` | 2D HTML5 Canvas renderer, d3-force underneath [6]. Its own documentation labels examples around **4K** and **75K elements** [6]. That is the strongest published practical scale signal for this family, but not a >=30 FPS promise for a reheated layout. | `graphData({nodes, links})` supports incremental updates [6]. `warmupTicks`, `cooldownTicks`, `cooldownTime`, `autoPauseRedraw`, `pauseAnimation`, and `resumeAnimation` control simulation and drawing [6]. CPU d3-force uses Barnes-Hut for many-body force; no 3D. | npm showed **1.51.0**, MIT, 6.46 MB unpacked and 15 dependencies [34]. Bundlephobia gzip was not exposed in the captured page. Maintenance is active enough to have a current npm package, but the captured GitHub page did not expose current issues or release date. React bindings are linked by the official docs [6]. |
| `react-force-graph-2d` | React wrapper around the Canvas engine. Good for an attractive 2D live view, but the CPU simulation remains the ceiling. | Same graphData and simulation controls; keep the graph object stable and mutate through the imperative ref rather than creating fresh arrays on every stream event. | npm showed **1.29.0**, MIT, **1.73 MB unpacked**, 3 dependencies [36]. |
| `3d-force-graph` | Three.js/WebGL renderer with d3-force-3d or ngraph physics. The package is visually impressive, but WebGL does not remove the CPU layout cost. | 3D dimensions, configurable engine, tick callbacks and cooldown controls are documented for the family [4]. | npm showed **1.80.0**, MIT, 5 dependencies, published approximately 4 months ago [35]. |
| React behavior | Four standalone packages exist for 2D, 3D, VR and AR, each exposing a force-directed graph component [4]. | The wrapper is materially better than hand-connecting d3 to React, but React prop changes can still trigger data reconciliation and reheating. Use the ref API, batch stream events, and avoid per-event React state updates. |

**Decision:** choose 2D Canvas for about **5K-15K active, continuously moving nodes**, and 3D/WebGL for about **10K-30K** if labels, picking, link density and physics are controlled. The documented 75K example is a rendering/demo ceiling, not a safe full-physics production ceiling. At 75K to 100K, freeze old nodes or display only a subgraph.

### Cosmograph and `@cosmograph/cosmos`

| Criterion | Current evidence and engineering interpretation |
|---|---|
| Rendering/layout scale | Cosmograph advertises a device-GPU simulation, zero-copy data flow and **multi-million-node graphs** [9]. The cosmos.gl repository says computations and drawing occur in GPU shaders and describes real-time graphs with **hundreds of thousands of points and links on modern hardware** [8]. These are the best public scale claims in this comparison, but they are not a reproducible universal 30 FPS benchmark. |
| Incremental updates | The current cosmos.gl v2 API replaces broad `setData` with `setPointPositions` and `setLinks`, using WebGL-compatible typed arrays [10]. That is a much better data path for streaming than rebuilding object graphs, but the documented API is replacement-oriented rather than an append-only transaction API. Maintain stable typed buffers, append in batches, preserve existing positions, and update links at a controlled cadence. |
| Physics | GPU force layout and WebGL rendering; intended for massive networks and ML embeddings [9]. The public material does not establish that every force, collision mode, or topology mutation is equivalent to d3-force. Validate collision/charge tuning and the visual behavior of newly inserted nodes on target hardware. |
| License | npm reported **3.4.1**, **MIT**, 16 dependencies; the page also displayed CC-BY-NC-4.0 in its sidebar, so verify the exact package and distribution terms before embedding Cosmograph application assets in a commercial product [38]. cosmos.gl itself is shown as MIT on GitHub [10][8]. |
| Bundle and maintenance | The captured npm page did not expose minified+gzip size. GitHub showed **653 commits** and MIT, but not open issue count or release metadata [8]. |
| React | Cosmograph presents itself as a JavaScript and React library [9]. Prefer its imperative GPU surface and batch data updates rather than making React reconcile 100K records. |

**Decision:** primary candidate. Start by benchmarking **10K, 25K, 50K, 100K nodes and 1x, 3x, 10x edge/node ratios** on the actual target GPUs. If the GPU layout remains above 30 FPS, it is the only option in this list with a public story that plausibly reaches the requested scale without immediately hiding most nodes.

### Sigma.js + Graphology + ForceAtlas2

| Criterion | Current evidence and engineering interpretation |
|---|---|
| Rendering scale | Sigma.js is WebGL-based and explicitly aimed at graphs of **thousands of nodes and edges** [14]. This supports smooth display of a substantial graph, but the official wording is not a 100K/30 FPS guarantee. |
| Incremental data | Graphology is the mutable graph model; its mutation API includes `addNode` and `addEdge` [14]. Sigma is designed to work in symbiosis with Graphology [14], so graph mutations can be observed and rendered without reconstructing the whole renderer. Batch mutations to avoid one redraw per streamed edge. |
| Layout | `graphology-layout-forceatlas2` is CPU ForceAtlas2. Every node needs initial `x` and `y` coordinates [12]. The worker/supervisor variant keeps the solver off the main thread, but it still computes a force layout over the graph. ForceAtlas2 offers strong visual quality and Barnes-Hut-style approximation options, not GPU physics. |
| Practical scale | Renderer gate: about **10K-30K displayed nodes** with modest labels and edges. Worker ForceAtlas2 gate: about **5K-15K actively moving nodes** for a visibly responsive continuous layout; 100K should be treated as a frozen/precomputed layout or sampled view. No official >=30 FPS benchmark for the combined renderer plus live ForceAtlas2 was found. |
| License/bundle | Sigma npm showed **3.0.2**, MIT, 969 kB unpacked [39]. A search result also reported 3.0.3, so pin and re-check the registry before release. Graphology is MIT, **0.26.0**, 2.73 MB unpacked [37]. ForceAtlas2 is MIT, **0.10.1**, 78.8 kB unpacked [13]. Minified+gzip values were not exposed in the captured Bundlephobia/npm pages. |
| Maintenance/React | Sigma GitHub showed **2,065 commits**, MIT and an active main branch [15]. It has no official React renderer; use a ref and Graphology store. React integration is clean if Sigma owns the canvas and React owns only controls/selection. |

**Decision:** best non-GPU alternative when you want Graphology's data model, Sigma's WebGL renderer and ForceAtlas2 aesthetics. Break down for a full live force view at roughly **15K moving nodes**; can display more if layout is frozen and edges/labels are aggressively reduced.

### deck.gl

| Criterion | Current evidence and engineering interpretation |
|---|---|
| Rendering/layout scale | deck.gl is a GPU-powered visualization framework [46]. A third-party test reports roughly **one million points** at interactive frame rates for ScatterplotLayer on ordinary hardware, but that is point rendering, not graph layout [47]. The official performance page demonstrates a complex application with close to **100 layers**, not a force graph benchmark [19]. |
| Force layout | deck.gl is a renderer/layer system, not a force solver. There is no native general-purpose GraphLayer plus live force physics in the evidence gathered. Implement positions in a worker/GPU solver, then render nodes with ScatterplotLayer and edges with LineLayer or a custom layer. |
| Incremental updates | Its reactive layer model supports changing data and attributes; stable data references, `updateTriggers`, binary attributes and batched updates are important. Replacing a 100K-object array every stream event will create CPU and garbage-collection pressure even if GPU rendering is fast. |
| Practical scale | Renderer gate: **100K to 1M simple points/line segments** is plausible with binary attributes and no labels, but not a published force-graph guarantee. Layout gate: entirely determined by the external solver. |
| License/bundle | MIT [17]. npm showed **9.3.10**, published approximately 6 days ago and 16 dependencies [40]. The captured Bundlephobia page did not expose core gzip size; a separate `@deck.gl/js` page reported **9.0 kB minified and 3.1 kB gzip**, which is not the size of the full deck.gl distribution [48]. |
| React/maintenance | Official `DeckGL` React integration is documented. GitHub showed **5,627 commits** and MIT [17]. It is a strong rendering substrate, not a turnkey force-directed graph component. |

**Decision:** rank deck.gl only if the team is willing to build or integrate the layout engine. Break down never because of rendering first; it breaks down when the external layout or data-copy pipeline cannot maintain the target cadence.

### Cytoscape.js

| Criterion | Current evidence and engineering interpretation |
|---|---|
| Scale | The public performance page includes test networks from **200** through **20,000 nodes** [23]. This is useful evidence of the tested range, not a >=30 FPS full-force promise. Cytoscape.js uses a general graph model and optional renderer [22]. |
| Incremental updates | `cy.add()` and related graph-model operations are appropriate for adding elements, but a force layout normally has to be rerun or explicitly managed after topology changes. That can cause jumps and full-layout churn. Batch additions and preserve positions where the chosen layout supports it. |
| Physics/renderer | Mature 2D graph model, compound graphs, analysis and multiple layouts [22]. The standard browser renderer is Canvas rather than a GPU-first WebGL force engine. No GPU force solver or 3D mode was found in the official material. |
| Practical scale | About **5K-10K moving nodes** is a sensible interactive force-layout gate; the documented 20K test range can be rendered or tested with simpler layouts, but it should not be interpreted as a live 30 FPS 20K-node force result. |
| License/bundle | MIT, **3.34.3**, published approximately 2 days before the 2026-09-14 access date, with zero npm dependencies [21]. Bundlephobia gzip was not exposed in the captured page. |
| Maintenance/React | GitHub showed **6,766 commits** and MIT [20]. The core package is not a first-party React renderer; use a container ref and batch imperative mutations. |

**Decision:** a good graph-analysis and moderate-size 2D product library, not a 100K live spider-web renderer. Break down around **10K moving nodes**, sooner with compound nodes, labels, high-degree edges or frequent relayout.

### AntV G6 v5

| Criterion | Current evidence and engineering interpretation |
|---|---|
| Scale | G6 documents force-directed layout for complex relationship networks, but the captured official pages did not provide a reproducible 100K-node >=30 FPS benchmark [27]. Treat marketing or demo scale as unverified until measured. |
| Incremental updates | G6 has a comprehensive data-operation API covering query, modification and update [49]. v5 redesigned its options and modularization; unused modules are not packaged into the final build [26]. Use its add/update/remove data methods in batches and explicitly control when layout runs. |
| Physics | Force layout models attraction and repulsion; exact v5 defaults and acceleration settings should be pinned and benchmarked. The evidence does not establish a GPU force solver equivalent to Cosmograph. |
| Practical scale | About **5K-15K moving nodes** for a force view; roughly **20K-50K** may be renderable with simpler styling and frozen positions. No defensible 100K live-force ceiling was published in the sources gathered. |
| License/bundle | MIT, **5.1.1**, published approximately 19 days before the access date, 11 dependencies [44]. The official feature page claims modular packaging reduces final bundle size [26], but minified+gzip was not exposed. |
| Maintenance/React | GitHub showed **5,619 commits** and MIT [25]. Use a React ref/effect integration or an available wrapper; keep G6's imperative lifecycle outside frequent React renders. |

**Decision:** a capable feature-rich middle ground, but not the first choice for 100K continuously rewiring force physics. Break down around **15K moving nodes** unless a target-hardware benchmark proves otherwise.

## Newer Options and Commercial Alternatives

| Option | What the current evidence supports | Recommendation |
|---|---|---|
| GraphGPU | WebGPU renderer for nodes, edges, halos and labels; GitHub says the repository was created **2026-02-24**, has **129 commits**, and is MIT [31]. No verified 100K force-layout benchmark, release version or issue count was exposed. | Interesting R&D option, not production primary. Breakpoint is unknown; demand a benchmark and browser fallback. |
| Reagraph | React WebGL graph visualization project, Apache-2.0, with **1,154 commits** [29]. npm showed **4.31.0**, Apache-2.0, published approximately 11 days before access and 21 dependencies [43]. No 100K live-force benchmark was found. | Good React/WebGL ergonomics for small and medium graphs; gate at roughly **10K-20K moving nodes** until proven otherwise. |
| Ogma, KeyLines, yFiles, Graphistry | These are commercial or hosted/product offerings with potentially stronger large-graph engineering, but no comparable current benchmark, package metadata or price evidence was gathered here. | Evaluate separately if commercial support, server-side aggregation or licensed GPU engines justify the cost. Do not substitute their marketing scale claims for a target-device test. |

GraphGPU is the only clearly new 2026 WebGPU candidate found, but its repository age and lack of published scale results make it a prototype candidate rather than the primary recommendation. Reagraph is more mature as a React integration, but its evidence describes WebGL visualization rather than a GPU force solver.

## Incremental Streaming Architecture That Avoids React Churn

Use a three-stage pipeline: ingestion, layout, and display. The ingestion worker receives node and edge batches; it deduplicates IDs and maintains typed arrays or a mutable Graphology/cosmos representation. The layout worker emits positions at a fixed cadence, while the React component receives only coalesced snapshots or shared buffers.

For d3-force or react-force-graph, add nodes to the backing array, call `simulation.nodes(nodes)`, update `forceLink.links(links)` when links change, set a modest `alphaTarget`, and call `restart` [2]. Do not call React state setters for every streamed node. For Sigma, batch Graphology `addNode` and `addEdge` operations and let Sigma redraw once; run ForceAtlas2 in its worker supervisor. For cosmos, retain positions for old points and update `setPointPositions` and `setLinks` in batches rather than reconstructing object-heavy data [10].

The key policy is "new topology does not imply global visual motion." Keep old nodes' velocities near zero, seed a new node near its parent or cluster, run a short local reheat, and periodically perform a slower global relaxation. This makes the graph look alive without turning every new edge into a full-layout shock.

## Production Mitigations for 10K-100K Nodes

| Mitigation | Implementation | Why it matters |
|---|---|---|
| Level of detail | Hide labels except for selected/hovered nodes; use one-pixel points at zoomed-out levels; reduce hit-testing to a spatial index. | Text and picking often dominate before point drawing does. |
| Edge budget | Render only edges incident to the selection, recent edges, top-weight edges, or one edge per aggregated relation. Bundle or fade dense edge groups. | Edge overdraw and visual crossings are the primary hairball problem. |
| Sampling/aggregation | Collapse low-importance nodes into cluster glyphs; aggregate by layer, community, time window or model component. | Lets the user see structure instead of an unreadable complete topology. |
| Freeze old regions | Mark settled nodes fixed or nearly fixed; simulate only a moving frontier and a halo around new nodes. | Converts an O(N)-per-tick global problem into a smaller active-frontier problem. |
| Worker layout | Put d3-force, ForceAtlas2, or the data model in a Worker. d3 explicitly recommends this for large static layouts [3]. | Prevents the main thread from missing input and React deadlines. It does not make the computation cheaper. |
| Display subgraph | Keep the full graph in the worker/server, but display the top-K most recent, highest-scoring, selected or causally connected nodes. | A practical K is often **1K-10K**, tuned by the target GPU and edge density, even when the underlying graph reaches 100K. |
| Progressive warmup | Show the new node immediately at a seeded position, then animate local stabilization; run global refinement on demand. | Avoids a blank or jumping canvas during long training runs. |

These mitigations are not optional polish at 100K. They are how a technically capable renderer becomes an interpretable product. A fully rendered 100K-node, 300K-edge graph may be possible on a strong GPU, but a fully visible and continuously moving version is usually the wrong user interface.

## Ranked Recommendation and Explicit Breakpoints

| Rank | Stack | Use it when | Full live-force breakdown gate |
|---:|---|---|---:|
| 1 | **Cosmograph / cosmos.gl** | Need the best chance of GPU layout plus GPU rendering at 50K-100K and can accept typed-array APIs and device-specific testing. | No universal limit published. Start failure testing at **25K**, **50K**, **100K**; if target hardware misses 30 FPS, switch to display subgraph. |
| 2 | **react-force-graph-3d** with ngraph or d3-force-3d, plus LOD | Need impressive 3D presentation and the fastest React path. | About **10K-30K moving nodes**; the documented 75K family example is not a safe continuous-force guarantee [6]. |
| 3 | **Sigma + Graphology + ForceAtlas2 worker** | Need a strong mutable graph model, WebGL rendering and ForceAtlas2 aesthetics. | About **10K-15K moving nodes**; more can be displayed with frozen positions and hidden edges. |
| 4 | **react-force-graph-2d / force-graph** | Need simple, attractive Canvas 2D and a straightforward incremental API. | About **5K-15K moving nodes**; 75K is a demo/rendering scale, not a full live-physics target [6]. |
| 5 | **deck.gl plus a custom/GPU layout** | Team can build the solver and wants maximum rendering control. | Renderer is not the limit; external layout and CPU-to-GPU transfer are. |
| 6 | **AntV G6 v5** | Need a broad visualization feature set and managed data operations. | About **10K-15K moving nodes** pending a target-device benchmark. |
| 7 | **Cytoscape.js** | Need graph analysis, compound graphs and a mature 2D model under roughly 10K active nodes. | About **5K-10K moving nodes**; the official performance page reaches 20K test networks but does not establish live-force 30 FPS [23]. |
| 8 | **Reagraph / GraphGPU** | Need React WebGL ergonomics or are willing to run a WebGPU R&D track. | Reagraph: provisionally **10K-20K**. GraphGPU: unknown until benchmarked. |

**Primary pick:** prototype Cosmograph/cosmos.gl first, with a benchmark harness that measures layout tick time, GPU frame time, heap growth, update latency and visual stability separately. **Fallback:** react-force-graph-3d for the compelling product surface, but ship a display-subgraph policy from day one rather than hoping Three.js will make a CPU force solver scale to 100K.

## Synthesis

The libraries divide into three architectural classes. d3-force, ForceAtlas2, Cytoscape layouts and G6 are primarily CPU layout systems; their strengths are control, mature graph semantics and predictable customization. Sigma and deck.gl are primarily GPU renderers around mutable or external data, so they can display more than they can lay out. Cosmograph/cosmos.gl is the unusual end-to-end GPU-oriented option and therefore has the strongest scale story for this exact workload.

The second contrast is update semantics. d3-force exposes the clearest explicit reheat contract, Graphology exposes the clearest graph mutation model, and force-graph exposes the most accessible graphData/warmup/cooldown product API. Cosmos exposes the most promising typed-array path, but its replacement-style `setPointPositions` and `setLinks` API means the application must own stable buffers and batch updates. None of these facts removes the need to control the active frontier.

The final tension is between spectacle and comprehension. Three.js 3D can look more impressive than 2D Canvas, but depth, labels and picking make dense graphs harder to read. GPU scale can preserve frame rate while destroying meaning through edge crossings. The correct architecture is therefore hybrid: keep the complete training graph in a worker or backend, run GPU or worker physics on a bounded active set, and render a carefully selected subgraph with progressive disclosure.

## References

1. *GitHub - d3/d3-force: Force-directed graph layout using velocity Verlet integration. · GitHub*. https://github.com/d3/d3-force
2. *d3-force - npm*. https://www.npmjs.com/package/d3-force
3. *Force simulations | D3 by Observable*. https://d3js.org/d3-force/simulation
4. *react-force-graph | React component for 2D, 3D, VR and AR force directed graphs*. https://vasturiano.github.io/react-force-graph/
5. *GitHub - vasturiano/force-graph: Force-directed graph rendered on HTML5 canvas · GitHub*. https://github.com/vasturiano/force-graph
6. *force-graph | Force-directed graph rendered on HTML5 canvas*. https://vasturiano.github.io/force-graph/
7. *GitHub - vasturiano/react-force-graph: React component for 2D, 3D, VR and AR force directed graphs · GitHub*. https://github.com/vasturiano/react-force-graph
8. *GitHub - cosmosgl/graph: GPU-accelerated force graph layout and rendering · GitHub*. https://github.com/cosmosgl/graph
9. *Library | Cosmograph*. https://cosmograph.app/library/
10. *GitHub - cosmosgl/graph: GPU-accelerated force graph layout and rendering · GitHub*. https://github.com/cosmograph-org/cosmos
11. *Introduction | Cosmograph*. https://cosmograph.app/docs-general/
12. *layout-forceatlas2 | Graphology*. https://graphology.github.io/standard-library/layout-forceatlas2.html
13. *graphology-layout-forceatlas2 - npm*. https://www.npmjs.com/package/graphology-layout-forceatlas2
14. *Sigma.js*. https://www.sigmajs.org/
15. *GitHub - jacomyal/sigma.js: A JavaScript library aimed at visualizing graphs of thousands of nodes and edges · GitHub*. https://github.com/jacomyal/sigma.js/
16. *Layer Catalog Overview | deck.gl*. https://deck.gl/docs/api-reference/layers
17. *GitHub - visgl/deck.gl: WebGL2 powered visualization framework · GitHub*. https://github.com/visgl/deck.gl
18. *Using deck.gl with React | deck.gl*. https://deck.gl/docs/get-started/using-with-react
19. *Performance Optimization | deck.gl*. https://deck.gl/docs/developer-guide/performance
20. *GitHub - cytoscape/cytoscape.js: Graph theory (network) library for visualisation and analysis · GitHub*. https://github.com/cytoscape/cytoscape.js/
21. *cytoscape - npm*. https://www.npmjs.com/package/cytoscape
22. *Cytoscape.js*. https://js.cytoscape.org/
23. *Cytoscape.js performance test page*. https://cytoscape.org/js-perf/
24. *G6 Graph Visualization Framework in JavaScript | AntV*. https://g6.antv.antgroup.com/en/
25. *GitHub - antvis/G6: ♾ A Graph Visualization Framework in JavaScript. · GitHub*. https://github.com/antvis/G6
26. *Feature | G6 Graph Visualization Framework in JavaScript*. https://g6.antv.antgroup.com/en/manual/whats-new/feature
27. *Force-directed Layout | G6 Graph Visualization Framework in JavaScript*. https://g6.antv.antgroup.com/en/manual/layout/force-layout
28. *Data | G6 Graph Visualization Framework in JavaScript*. https://g6.antv.antgroup.com/en/manual/data
29. *GitHub - reaviz/reagraph: 🕸 WebGL Graph Visualizations for React. Maintained by @goodcodeus. · GitHub*. https://github.com/reaviz/reagraph
30. *Reagraph - a high-performance network graph visualization built in WebGL for React*. https://reagraph.dev/
31. *GitHub - drkameleon/GraphGPU: Fast & efficient WebGPU-accelerated graph visualization library · GitHub*. https://github.com/drkameleon/GraphGPU
32. *GraphGPU · WebGPU-accelerated graph visualization*. https://graphgpu.com/
33. *3d-force-graph ❘ Bundlephobia*. https://bundlephobia.com/package/3d-force-graph
34. *force-graph - npm*. https://www.npmjs.com/package/force-graph
35. *3d-force-graph - npm*. https://www.npmjs.com/package/3d-force-graph
36. *react-force-graph-2d - npm*. https://www.npmjs.com/package/react-force-graph-2d
37. *graphology - npm*. https://www.npmjs.com/package/graphology
38. *@cosmograph/cosmos - npm*. https://www.npmjs.com/package/@cosmograph/cosmos
39. *sigma - npm*. https://www.npmjs.com/package/sigma
40. *deck.gl - npm*. https://www.npmjs.com/package/deck.gl
41. *cytoscape ❘ Bundlephobia*. https://bundlephobia.com/package/cytoscape
42. *deck.gl ❘ Bundlephobia*. https://bundlephobia.com/package/deck.gl
43. *reagraph - npm*. https://www.npmjs.com/package/reagraph
44. *@antv/g6 - npm*. https://www.npmjs.com/package/@antv/g6
45. *Many-body force | D3 by Observable*. https://d3js.org/d3-force/many-body
46. *Home | deck.gl*. https://deck.gl/
47. *Rendering a Million Points with PyDeck ScatterplotLayer*. https://www.geo-dashboard.com/core-mapping-architecture-rendering/renderer-selection-folium-maplibre-pydeck/rendering-a-million-points-with-pydeck-scatterplotlayer/
48. *@deck.gl/js v0.0.1 Bundlephobia*. https://bundlephobia.com/package/@deck.gl/js
49. *Data | G6 Graph Visualization Framework in JavaScript*. https://g6.antv.antgroup.com/en/api/data
