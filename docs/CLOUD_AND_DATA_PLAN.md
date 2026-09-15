# Cloud Training & Dataset Plan

Written 2026-09-15, in response to a request to plan low-cost, high-efficiency, correct-result
cloud training using Nebius + `mosaicml-streaming`, and to decide the datasets for the next
stages. Every claim below was checked live, not assumed — the same protocol the rest of this repo
follows.

## 0. Bottom line

- **Nebius**: its official AI-agent integration is an **MCP server for MCP-native clients**
  (Claude Code, Cursor, Codex, Copilot, OpenCode, Windsurf) — this Prime Agent session is not one
  of those hosts, so the MCP layer itself cannot be attached here. The underlying capability is
  the **Nebius CLI**, which *is* usable directly from this session, and is now installed
  (`~/.nebius/bin/nebius`, v0.12.277). It needs its own account, separate from Prime Intellect —
  see §2.
- **`mosaicml-streaming`**: checked against this project's actual data size. **Not a fit, and
  adopting it now would add overhead, not remove it.** See §3 for the measured numbers. There is
  a concrete future trigger where it becomes worth reconsidering.
- **Compute that is already live and billable right now**: Prime Intellect, authenticated this
  session (`prime config view` confirms it), CPU nodes from **$0.05/hr**, including several
  **Nebius-provider** nodes resold through Prime Intellect's marketplace. This is the fastest path
  to actually running something in the cloud today.
- **Datasets for the next stages**: S4 (the RL agent) needs **no new dataset** — it reuses the S2
  price data already built. S6 (EKG) needs **no new dataset** — it is built from the agent's own
  trade history. S7 (sentiment) is the one stage with a real, unresolved dataset decision; see §4.

## 1. Why Nebius's own agent integration doesn't attach here, and what does instead

`nebius/mcp-server`'s `AGENT_SETUP.md` (fetched live) configures the server as a **local stdio MCP
server** registered into a specific client's own config file — `~/.codex/config.toml`,
`~/.claude.json`, `.cursor/mcp.json`, `.vscode/mcp.json`, `~/.copilot/mcp-config.json`, or
`opencode.json`, depending on the client. This Prime Agent session runs its own Python-skill tool
system (`ipython`/`bash` plus documented skills), not one of those MCP hosts, and the guide itself
says plainly: *"If the agent does not support local STDIO MCP servers, explain that the Nebius MCP
Server cannot be configured using this local installation method."* That is the honest situation
here.

This does **not** block using Nebius. The MCP server's own "Execute commands" examples (create a
compute instance, list storage buckets, get compute platforms) are a thin wrapper over the
**Nebius CLI** (`nebius`), which is a normal, fully scriptable command-line tool. It is now
installed on this machine and callable directly from `bash()` — a more direct path than going
through an MCP tool-call layer would have been anyway.

**If you separately want the MCP server available inside your own Claude Code or Cursor session**
(for interactive use outside this project), the one-liner from the guide is:
```
claude mcp add --env SAFE_MODE=true --transport stdio --scope user nebius \
  -- uvx --refresh-package nebius-mcp-server "nebius-mcp-server@git+https://github.com/nebius/mcp-server@main"
```
That is a separate tool from this session, the same way `diagram-design` was.

## 2. What's needed to actually drive Nebius from here

Nebius is a **separate account/platform from Prime Intellect** — signing into one does not grant
the other, even though Prime Intellect's marketplace resells some Nebius capacity (see §5).

`nebius profile create` is interactive in a way `prime login` was not: `prime login` completed
by printing a URL and polling for a browser callback, which worked over a piped `bash()` call.
`nebius profile create` uses a raw-keystroke terminal UI (verified: it hung waiting for typed
input and terminated on EOF when driven non-interactively). That specific flow needs a real
terminal a human is typing into.

Two ways to unblock it, your choice:

1. **You run it yourself** in a real terminal on this machine — the CLI is already installed and
   waiting at `~/.nebius/bin/nebius`; `nebius profile create` will pick you up from a normal
   Nebius.com login.
2. **Give me a service-account credentials file** — Nebius supports fully non-interactive auth via
   `nebius profile create --service-account-file <path>` (or `--service-account-id` +
   `--private-key-file` + `--public-key-id`). You'd create this once from the Nebius console
   (IAM → Service Accounts → new key) and hand me the JSON; I can then drive Nebius end-to-end
   the same way I already drive Prime Intellect.

I won't guess at pricing or capability differences between Nebius-direct and Prime-Intellect's
resold Nebius nodes until I have real access to compare — that comparison is listed as an open
item in §6, not assumed.

## 3. `mosaicml-streaming`: measured, not assumed, verdict

`StreamingDataset` (its own README, fetched live) exists to make **large-dataset, multi-node,
distributed training** fast by lazily streaming sharded data from cloud object storage (S3, GCS,
Azure, OCI, etc.), avoiding the need to hold the full dataset on local disk. Its own framing is
explicit about the problem it solves: training foundation-scale models where the dataset does not
fit on a single machine.

**Measured footprint of this project's actual data, checked just now:**

| Artefact | Size |
|---|---|
| `data/` (S2 price/actions Parquet, all providers, all tickers) | **3.0 MB** |
| `models/` (60 trained PPO checkpoints from the S4 sweep) | **18 MB** |
| `data/trials.jsonl` (84 logged evaluations) | **36 KB** |
| **Total** | **~21 MB** |

This loads into RAM in milliseconds and already does, every time `Store.bars()` is called. There
is no multi-GB, no multi-node, and no cloud-object-storage-residency problem to solve. Adopting
`mosaicml-streaming` here would mean converting this tiny dataset into MDS shards, standing up
object storage, and paying shard-fetch latency on every read — replacing a `pandas.read_parquet`
call that already takes single-digit milliseconds with a network round trip. That is a net loss
on every axis in the prompt's own framing (low cost, high efficiency): it is neither cheaper nor
faster for data this size.

**Concrete future trigger, not a blanket "never"**: if S7's sentiment ingestion (see §4) grows the
raw message corpus past roughly **1 GB** — plausible only if full historical StockTwits/Reddit
message text (not just counts) is ingested across all 5 tickers over the full multi-year window —
`mosaicml-streaming` (or a simpler chunked-Parquet/DuckDB scan, which is cheaper to operate for
data in the hundreds-of-MB-to-low-GB range) becomes worth reconsidering. Not before, and not for
RL training on this 5-asset universe, which is architecturally small by design
(`RESEARCH.md`'s own conclusion: 5 liquid names, daily bars, free data).

## 4. Datasets required, stage by stage

| Stage | New dataset needed? | Detail |
|---|---|---|
| S4 (RL agent, done) | **No** | Trained entirely on the S2 price panel already built |
| Further RL training (TD3, HPO sweep) | **No** | Same S2 panel; no new sourcing |
| S6 (EKG) | **No** | Built from `Episode`/`FeatureState` records the agent already generates during training/evaluation — an internal audit log, not an external corpus |
| S7 (sentiment) | **Yes — decision pending** | See table below, carried over from the research phase and not yet resolved |

**S7 sources, already decided in `ARCHITECTURE.md`, decision on *which to start with* still open:**

| Source | Cost | Constraint |
|---|---|---|
| StockTwits | free, keyless | live/recent stream only — **no historical archive** |
| Reddit (PRAW) | free | ToS forbids training on it — inference only |
| X/Twitter counts | ~$2 total | volume-only, no message text, but does cover history back to 2015 |
| Kaggle labelled tweets | free | encoder validation only (2021-09→2022-09) |

The unresolved question from before still stands: start collecting StockTwits now for a
forward/live validation window, or spend the ~$2 on X historical counts to get a (weaker,
volume-only) signal back through 2015. This is a decision for you, not something to default on
silently.

## 5. The concrete, staged execution plan

**Stage A — now, $0.** Finish the RL side locally: TD3 (the designated challenger,
`ARCHITECTURE.md` §3.3) and a modest Optuna hyperparameter sweep. Justification: the S4 PPO sweep
(20 seeds × 3 cost levels, 60 trained models) already completed in **~5 minutes on this machine's
CPU** — this workload is not compute-bound in a way cloud GPUs help with (small 5-asset
observation space, tiny MLP policy). Recommended default unless you want the cloud step for its
own sake.

**Stage B — optional, ~$1-5, when a larger *parallel* sweep is wanted.** Provision a Prime
Intellect CPU node (already authenticated, ready now) — an 8 vCPU/32 GB node at **$0.20/hr**, or
several cheaper $0.05/hr nodes in parallel — to run many Optuna trials concurrently rather than
sequentially. This is genuinely where cloud parallelism pays for itself: wall-clock time drops
roughly linearly with node count for embarrassingly-parallel HPO trials, for a few dollars.

**Stage C — only if triggered, per §3.** Reconsider `mosaicml-streaming` (or a lighter
chunked-read approach) specifically for S7's sentiment corpus, only once it is large enough to
justify it. Not for the RL agent's own training data at any point under the current 5-asset,
daily-bar scope.

**Stage D — Nebius, pending your credentials (§2).** Once authenticated, compare its raw pricing
against Prime Intellect's resold Nebius nodes for the same instance class, and use whichever is
cheaper for Stage B's HPO burst. Not assumed cheaper either way until checked.

## 6. Open items — need a decision from you

1. **Nebius access**: run `nebius profile create` yourself in a terminal, or send a service-account
   credentials file, so I can drive it directly (§2).
2. **S7 sentiment data**: StockTwits-forward-only vs. the ~$2 X historical-counts purchase (§4).
3. **Cloud vs. local for the next RL step**: proceed with Stage A (free, local TD3 + HPO) now, or
   go straight to Stage B (a paid Prime Intellect node) for a larger parallel sweep?
