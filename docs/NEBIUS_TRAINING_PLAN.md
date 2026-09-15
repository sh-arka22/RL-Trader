# Nebius Cloud Training — Step-by-Step Guide

Written 2026-09-15. Every claim in this document was checked live against Nebius's own
documentation on the date above (fetched directly, not from parametric knowledge), because
pricing and product scope change and this project's house rule is verify, don't assume.

## 0. The critical finding first: your two Nebius accounts are NOT interchangeable

You mentioned **Nebius AI Cloud** and **Nebius Token Factory**, 120 credits each. These are
**two different products**, and only one of them can run this project's training:

| Product | What it actually is (verified from its own docs) | Usable for RL-Trader training? |
|---|---|---|
| **Nebius AI Cloud** | General-purpose GPU/CPU virtual machines you SSH into and run arbitrary code on | **Yes — this is the one** |
| **Nebius Token Factory** | An LLM platform: OpenAI-compatible inference + fine-tuning API for open-source LLMs (Qwen3, DeepSeek-R1, Mistral, etc.), RAG building blocks, embeddings | **No.** Our PPO/TD3 agent is a small custom Gymnasium policy trained with `stable-baselines3` — it has nothing to do with LLM tokens, prompts, or fine-tuning a pretrained language model. There is no way to point Token Factory's fine-tuning API at a Gymnasium environment. |

So: **your 120 Token Factory credits cannot be spent on this project.** All of this plan uses
the AI Cloud side only. (Token Factory would become relevant if this project's later,
architecturally-separate LLM feature-extraction layer — `ARCHITECTURE.md` §3.4/3.5, explicitly
*offline and cached*, never in the RL training loop — ever needed a hosted open-source LLM. Not
now.)

## 1. Budget reality check

Nebius's own Builder Program terms describe credits in direct USD terms ("$50 in AI Cloud
credits"), so **1 credit is very likely $1 face value** — confirm this on your own Nebius
console before spending, since I cannot see your account. Assuming that's right:

**You have far more budget than this workload needs.** Measured facts, not estimates:
- The full S4 sweep (PPO, 20 seeds × 3 cost levels, 60 trained models) ran **locally in ~5
  minutes for $0**.
- Nebius AI Cloud CPU-only instances (verified pricing, `docs.nebius.com/compute/resources/pricing`
  and cross-checked against Prime Intellect's own Nebius-resold listings, which use the identical
  SKU names `cpu-d3`/`cpu-e2`):

| Tier | vCPU / RAM | Price |
|---|---|---|
| Intel Ice Lake | 2–80 vCPU, 8–320 GB | **from $0.05/hr** |
| AMD EPYC Genoa | 4–64 vCPU, 16–256 GB | **from $0.10/hr** |

A generous plan — TD3 training + a 50-trial parallel Optuna HPO sweep on an 8 vCPU/32 GB node
for several hours — costs **low single digits of dollars**, not $120. **Do not rent a GPU for
this.** `ARCHITECTURE.md` already decided CPU sandboxes over GPUs for exactly this workload, and
S4's measured result confirms it: a ~106-dimensional observation and a small MLP policy is
CPU-bound on Python/environment-stepping overhead, not matrix-multiply throughput. GPU tiers here
run **$3.85–$7.85 per GPU-hour** — that would burn your budget for zero speed benefit.

## 2. Do you need to upload a dataset? Almost certainly not — clarify one thing

You mentioned you can upload a dataset if required. **For the RL agent's existing training data,
no upload is needed.** The entire S2 pipeline is designed to be reproduced from scratch on any
machine, from free live APIs (Yahoo Finance + a keyless second source), byte-for-byte
reproducibly — that reproducibility is a pass/fail exit criterion of S2 itself
(`scripts/build_dataset.py --repro`). The cloud VM will just run `make data` and get the same
29,410-bar, point-in-time-correct dataset we already have.

**If you mean a *different* dataset** — e.g., something for the sentiment agent (S7, still an
open decision — see `docs/CLOUD_AND_DATA_PLAN.md` §4) or an external corpus not yet part of this
repo — tell me specifically what it is and I'll add an upload step. Until then, this plan assumes
no upload.

## 3. Step-by-step — what you run

### Step 1 — authenticate (you, in a real terminal; I can't drive this part)
The Nebius CLI is already installed on this machine (`~/.nebius/bin/nebius`, confirmed working).
`nebius profile create` needs a real interactive terminal — verified: it hangs when driven
non-interactively by me. Run this yourself:
```bash
export PATH="$HOME/.nebius/bin:$PATH"
nebius profile create
```
Follow the browser login against your **Nebius AI Cloud** account (not Token Factory — Token
Factory doesn't have a CLI profile the same way; it's used via API key against
`docs.tokenfactory.nebius.com`, and we're not using it here). Then verify:
```bash
nebius profile list
nebius profile active
```
Tell me when this is done, or hand me a service-account credentials JSON (Nebius console → IAM →
Service Accounts → new key) and I can take it from here instead.

### Step 2 — check availability and provision a CPU-only VM (either of us can run this once profile exists)
```bash
nebius compute platform list                       # confirm cpu-d3 / cpu-e2 availability in your region
nebius compute instance create \
  --name rl-trader-train \
  --platform cpu-e2 \
  --preset 8vcpu-32gb \
  --boot-disk-image-family ubuntu22.04-driverless \
  --boot-disk-size 50 \
  --network-id <your-network-id>
```
(Exact flags for network/subnet depend on your project's defaults — `nebius compute instance
create --help` will show what's required if this errors; I have not test-run this specific
command since it needs an authenticated profile.) Cost at $0.10–0.20/hr for an 8 vCPU/32 GB node:
running for a full 8-hour day is **under $2**.

### Step 3 — set up the environment on the VM
```bash
ssh <vm-ip>                                          # or: nebius compute instance get-connect-info rl-trader-train
sudo apt update && sudo apt install -y git curl
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
git clone https://github.com/sh-arka22/RL-Trader.git
cd RL-Trader
uv venv --python 3.13 .venv-core
VIRTUAL_ENV=.venv-core uv pip install -e ".[dev]"
```

### Step 4 — regenerate the dataset (no upload — reproduced live, matches Step 2 of this doc)
```bash
.venv-core/bin/python scripts/build_dataset.py --root data --start 2015-01-01
```
This should reproduce the same gate results we already have (0 missing sessions, cross-provider
divergence ≤1 bp, etc.) — if it doesn't, that's a real finding worth reporting, not something to
paper over.

### Step 5 — run the actual training you want
Pick one or both:
```bash
# TD3 — not yet run anywhere, the designated challenger in ARCHITECTURE.md
.venv-core/bin/python scripts/train_agent.py --algo td3 --cost-level realistic \
  --seeds $(seq 0 19) --timesteps 100000 --out data/td3_realistic_20seed.json
# (note: scripts/train_agent.py currently only implements PPO — see §4, this needs a small
#  extension before Step 5 can run TD3; flagged honestly, not glossed over)

# A larger, PARALLEL Optuna HPO sweep — this is what cloud parallelism is actually good for
.venv-core/bin/python scripts/hpo_sweep.py --n-trials 50 --n-jobs 8   # not yet written — see §4
```

### Step 6 — get results back
Either commit from the VM (needs your git credentials there) or pull down:
```bash
scp <vm-ip>:~/RL-Trader/data/*.json ./data/
scp -r <vm-ip>:~/RL-Trader/models ./models/
```
Then I write the report and commit locally, same as every other stage.

### Step 7 — shut the VM down
```bash
nebius compute instance stop rl-trader-train
nebius compute instance delete rl-trader-train     # once you've confirmed results are pulled
```
Stopped VMs aren't charged for compute (confirmed in the pricing doc); deleting also frees the
disk, which is billed separately while it exists.

## 4. Honest gap — TD3 code is written but NOT YET smoke-tested, and here's why

`scripts/train_agent.py` now supports `--algo td3` (SB3's `TD3`, `NormalActionNoise` for
exploration on the continuous action space, replay buffer sized to the training episode rather
than SB3's 1M-transition default — the default would have pre-allocated ~570x more transitions
than a full pass over our ~1,750-step training window needs).

**Attempting to smoke-test it just now failed for an environmental reason, not a code reason.**
This machine has 18 GB RAM and, at the time of testing, **~82 MB free** — dozens of Chrome
renderer processes, several concurrent Claude Code sessions, and Orca were consuming the rest.
Isolated the stall precisely: even a bare `import pandas` inside `.venv-core` hung past 90
seconds under this pressure, while the same import via plain system Python (outside the heavier
`.venv-core` environment) completed in 1.7s. That is a resource-contention symptom, not a bug in
`TradingEnv`, `TD3`, or the buffer-size fix.

**This is, ironically, itself an argument for the cloud plan**: a dedicated Nebius VM has no such
contention. Recommended order: (1) you complete `nebius profile create`, (2) the *first* thing run
on the VM is the TD3 smoke test (`--seeds 0 --timesteps 2000`, a few seconds of real compute) to
get a clean verification before committing to a full 20-seed run, (3) only then the full sweep.

**No `scripts/hpo_sweep.py` exists yet** — an Optuna wrapper (search space: learning rate,
`n_steps`, `batch_size`, the `l1_penalty` turnover coefficient; trials logged to the same
hash-chained log) is still to be written. I will write it once TD3's smoke test confirms the
environment is behaving, so both are validated together on the VM in step order rather than
guessed at locally under current contention.

## 5. What I need from you to proceed
1. Confirm: proceed with Nebius **AI Cloud** only (Token Factory sits unused, as established in §1)?
2. Should I write the TD3 + Optuna sweep scripts now (free, local) so they're ready before you
   spin up the VM?
3. Run `nebius profile create` yourself when ready, or send a service-account credentials file.
