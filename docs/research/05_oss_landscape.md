# 05 — Open-Source Landscape Audit (code-level, not marketing)

**Author:** oss-arch research subagent · **Date of all checks: 2026-09-14** · **Protocol:** `docs/RESEARCH_PROTOCOL.md`

Every repo below was cloned (`git clone --depth 1` into `/tmp/ossaudit/`) and read. Every install
claim was produced by running `uv` in a throwaway venv on this machine on 2026-09-14. Raw command
output is quoted verbatim. GitHub metadata comes from the authenticated REST API on 2026-09-14.

---

## 0. Headline: the master brief's core assumption is WRONG (but not for the reason you'd guess)

The master brief (`PRIME_INTELLECT_PROMPT.md` §1) mandates
`tensortrade-org/tensortrade` as "the execution/environment layer (Gymnasium-compatible)".

Three things are true at once, and only the third is fatal:

| Claim | Status | Evidence |
|---|---|---|
| "TensorTrade is dead / unmaintained" | **FALSE** — this is stale 2024 knowledge | Last commit 2026-02-09; PyPI `1.0.4` uploaded 2026-02-06; 31 of the last 100 commits fall in the trailing 12 months |
| "TensorTrade is Gymnasium-compatible" | **TRUE at the signature level** | `TradingEnv(gymnasium.Env)`; `reset()->(obs, info)`; `step()->(obs, reward, terminated, truncated, info)`; **SB3 2.9.0 `check_env` PASSES** |
| "TensorTrade is a sound execution layer" | **FALSE** | It simulates **zero slippage** (the slippage subsystem is dead code), **zero partial fills**, and it **pins the whole project to numpy<2 / pandas<3 / Python 3.12 only** — which forbids every other library we want. It also fails Gymnasium's own `check_env` on seeding. |

**Verdict: do NOT build on TensorTrade. Vendor its ideas, not its code.**
Recommended substitute and migration argument: §3 and §7.

---

## 1. Master comparison table

`c12/100` = how many of the last 100 default-branch commits landed after 2025-09-14 (activity
density, not total commits). "Gym API" = what the code actually returns, read from source.

| Repo | Stars | Last commit | Archived | Licence | Latest release (date) | Contrib | c12/100 | Python | Gym API | numpy2 / pandas2+ | Tests | Verdict |
|---|---:|---|---|---|---|---:|---:|---|---|---|---|---|
| tensortrade-org/tensortrade | 7,117 | 2026-02-09 | No | Apache-2.0 | v1.0.4 (2026-02-06) | 44 | 31 | **==3.12 only** | Gymnasium 5-tuple ✅ | **numpy<2, pandas<3** ❌ | 57 files, 241 pass | **AVOID as base; vendor OMS ideas** |
| TradeMaster-NTU/TradeMaster | 3,070 | 2025-06-04 | No | Apache-2.0 | v1.0.0 (2023-03-05) | 12 | **0** | 3.9-era | legacy `gym` (11 files) | ❌ | 9 files, no CI | **AVOID — does not install** |
| AI4Finance-Foundation/FinRL | 16,282 | 2026-07-12 | No | MIT | v0.3.8 (2026-03-20) | 119 | 100 | `>=3.7` (unpinned) | Gymnasium 5-tuple ✅ | loose, untested | 6 files, 1 CI | **Vendor a subset (env + baselines)** |
| AI4Finance-Foundation/FinRL-Meta | 1,938 | 2026-04-02 | No | MIT | v0.3.6 (2022-06-26) | 38 | 20 | `>=3.6` | **legacy `gym` (15 files)** | ❌ | 5 files | **Avoid — data processors only** |
| AI4Finance-Foundation/FinRL-Trading (= **FinRL-X**) | 3,698 | 2026-05-02 | No | Apache-2.0 | v1.0.0 (2026-03-25) | 14 | 100 | `>=3.11` | not a Gym env (weight-vector API) | unverified | 0 test files | **Watch — read the paper, don't depend yet** |
| freqtrade/freqtrade | 54,370 | **2026-09-14** | No | GPL-3.0 | 2026.8 (2026-08-31) | 342 | 100 | `>=3.11` (3.11–3.14) | Gymnasium ✅ (`freqai/RL`) | ✅ | 105 files, 10 CI | **Study its FreqAI RL envs; GPL blocks linking** |
| nautechsystems/nautilus_trader | 28,923 | **2026-09-14** | No | LGPL-3.0 | v2.0.0rc4 (2026-09-02) | 194 | 100 | `>=3.12,<3.15` | not a Gym env | ✅ | 146 files, 11 CI | **BUILD ON IT — validation engine (see risk)** |
| ChuaCheowHuan/gym-continuousDoubleAuction | 154 | **2026-09-13** | No | MIT | none (setup.py 2.0.0) | 3 | 100 | `>=3.12` | Gymnasium + PettingZoo/RLlib multi-agent | ✅ | 47 files, CI on 3.12 | **Vendor its matching engine if we need a LOB** |
| davebyrd/minabides | 3 | 2025-07-22 | No | NOASSERTION | none | 1 | 0 | 3.12 | custom (not Gym) | n/a | 0 | **Reference only — 1 author, 3 stars, unclear licence** |
| DLR-RM/stable-baselines3 | 13,795 | 2026-09-09 | No | MIT | v2.9.0 (2026-06-15) | 163 | 33 | 3.10–3.13 | Gymnasium native | ✅ (`numpy>=1.20,<3`) | full CI | **BUILD ON IT — RL algorithms** |
| Stable-Baselines-Team/sb3-contrib | 738 | 2026-06-15 | No | MIT | v2.9.0 (2026-06-15) | 35 | 15 | 3.10–3.13 | Gymnasium native | ✅ | full CI | **BUILD ON IT — RecurrentPPO / QR-DQN** |
| Farama-Foundation/Gymnasium | 12,531 | 2026-09-10 | No | MIT | v1.3.0 (2026-04-22) | 441 | 100 | `>=3.10` | **is** the API | ✅ | full CI | **BUILD ON IT — the env contract** |
| ray-project/ray (RLlib) | 43,799 | **2026-09-14** | No | Apache-2.0 | ray-2.58.0 (2026-08-23) | 410 | 100 | `>=3.10` | Gymnasium | ✅ | full CI | **Only if we need distributed/multi-agent** |
| vwxyzjn/cleanrl | 10,398 | 2026-04-20 | No | NOASSERTION | v1.0.0 (2022-11-14); PyPI 1.2.0 (2023-05-22) | 44 | **2** | PyPI meta `>=3.7.1,<3.11` | Gymnasium in repo | stale pins | research scripts | **Copy single files, never `pip install`** |
| microsoft/qlib | 48,551 | 2026-07-23 | No | MIT | v0.9.7 (2025-08-15) | 139 | 26 | `>=3.8` | **legacy `gym` in `qlib.rl`** | pulls `gym==0.26.2` ⚠ | 36 files, 6 CI | **Vendor the PIT data ideas, not the RL module** |
| polakowo/vectorbt | 9,086 | 2026-08-02 | No | NOASSERTION | v1.1.0 (2026-07-05) | 21 | 100 | `>=3.11,<3.15` | n/a | **requires numpy>=2.4.6, pandas>=3.0.3** | 13 files, 4 CI | **Candidate for fast vectorised metrics — but see conflict** |
| AminHP/gym-anytrading | 2,388 | 2023-08-27 | No | MIT | v2.0.0 (2023-08-27) | 5 | 0 | unpinned | **Gymnasium, textbook-correct** | ✅ **tested working on numpy 2.5.3 + pandas 3.0.5** | 0 tests | **Vendor as the reference env skeleton (~300 LOC)** |
| PrimeIntellect-ai/prime-rl | 2,038 | 2026-09-13 | No | Apache-2.0 | v0.9.0 (2026-08-25) | 71 | 100 | — | LLM-RL, not Gym | ✅ | active | **Not for the PPO core; relevant only to an LLM layer** |

**Sorted maintenance ranking (evidence, not vibes):**
`freqtrade ≈ nautilus_trader ≈ ray > Gymnasium > FinRL > SB3 > vectorbt > qlib > gym-cDA
> TensorTrade > FinRL-Meta > cleanrl > gym-anytrading > TradeMaster > minabides`.

---

## 2. TensorTrade: the full autopsy (this is the decision the brief depends on)

### 2.1 It really was dead, and it really did come back

| Fact | Value | Source |
|---|---|---|
| PyPI release history | `1.0.3` on **2021-05-10** → *4 years 9 months of silence* → `1.0.4` on **2026-02-06** | `https://pypi.org/pypi/tensortrade/json` fetched 2026-09-14 |
| Last commit | `d58afba` 2026-02-09, by Carlo Grisetti | `git log -1` on the clone |
| That commit's message | *"env/ in gitignore prevented the inclusion of `tensortrade/env` in the installed package"* | same |
| Repo version file | `__version__ = "1.0.5-dev"` | `tensortrade/version.py` |

Read that commit message again: until 2026-02-09 the **published package did not contain the
environment module** because `env/` matched a `.gitignore` rule. That is the packaging maturity
we are being asked to stand on.

`CHANGES.md` is dated `## [1.0.4] - 2025-02-04` while the release shipped 2026-02-06.
`TESTING_REPORT.md` is headed `**Date:** 2025-01-XX` and contains the phrase
*"All tests designed to pass"*. `setup.py` says `python_requires='>=3.12'`, `CHANGES.md` says
`>= 3.11.9`, and `COMPATIBILITY.md` says `Gymnasium >=0.28.1, <1.0`, while `setup.py` has no
gymnasium upper bound at all. The 1.0.4 revival looks substantially machine-generated and not
fully reconciled.

### 2.2 `pip install tensortrade` — what actually happens (run on this machine, 2026-09-14)

**Python 3.13 — resolver silently gives you a 2020 beta:**
```
$ uv venv --python 3.13 tt313 && uv pip install tensortrade
Installed 9 packages
 + gym==0.26.2          <-- LEGACY gym, not gymnasium
 + numpy==2.5.3
 + pandas==3.0.5
 + tensortrade==1.0.0b0 <-- uploaded 2020-08-24
$ ./tt313/bin/python -c "import tensortrade"
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
...
ModuleNotFoundError: No module named 'IPython'
```
So on Python 3.13 a naive `pip install tensortrade` yields a **six-year-old beta that imports
legacy `gym` and then crashes on import** because `1.0.0b0` never declared its own dependencies.

**Python 3.13, pinned to 1.0.4 — hard, unfixable resolution failure:**
```
$ uv pip install 'tensortrade==1.0.4'
  Because stochastic==0.6.0 depends on numpy>=1.17,<2.0 and stochastic>=0.7.0 depends on
  numpy>=1.19,<2.0 ... and tensorflow>=2.15.1 depends on numpy>=2.1.0, we can conclude that
  stochastic>=0.6.0 and tensorflow>=2.15.1 are incompatible.
  And because tensortrade>=1.0.4 depends on stochastic>=0.6.0 ... your requirements are unsatisfiable.
hint: You require CPython 3.13 (`cp313`), but we only found wheels for `tensorflow` (v2.19.0)
      with the following Python ABI tags: `cp39`, `cp310`, `cp311`, `cp312`
```
Two independent blockers: no TensorFlow wheels for cp313 on this platform, **and** an
irreconcilable numpy constraint between `stochastic` (`numpy<2`) and `tensorflow` (`numpy>=2.1`).

**Python 3.12 — it does install, in a numpy-1.x cage:**
```
$ uv venv --python 3.12 tt312 && uv pip install tensortrade
 + tensortrade==1.0.4   gymnasium==1.3.0   tensorflow==2.21.0
 + numpy==1.26.4        pandas==2.3.3      stochastic==0.7.0
$ ./tt312/bin/python -c "import tensortrade; print(tensortrade.__version__)"
1.0.4
```
**Python 3.12 is the only working target, and it forces `numpy==1.26.4` and `pandas<3`.**

### 2.3 Does its own test suite pass? Mostly yes.

```
$ cd tensortrade && python -m pytest tests -q
241 passed, 2 skipped, 33 warnings, 10 errors in 6.32s
ERROR tests/tensortrade/integration/rllib/test_ppo_single_iteration ... (x10, all RLlib)
```
The 10 errors are all Ray/RLlib integration tests failing to collect because `ray` is not
installed. **The core unit suite is genuinely green.** Credit where due: this is better than
FinRL, FinRL-Meta, FinRL-Trading, minabides or gym-anytrading, none of which ship a real suite.

### 2.4 Gymnasium conformance — passes SB3, fails Gymnasium

Built a real `TradingEnv` (BSH action scheme, SimpleProfit reward, 400-bar GBM feed) and ran both
checkers (`/tmp/ossaudit/tt_smoke2.py`):
```
tensortrade 1.0.4
action_space Discrete(2) obs_space Box(-inf, inf, (10, 2), float32)
reset -> (obs, info) OK; obs shape (10, 2) info keys ['step', 'net_worth']
step tuple len 5
reward -0.0016729673337873008 terminated False truncated False types <class 'bool'> <class 'bool'>
has render_mode: True | metadata: {'render_modes': []}
GYMNASIUM check_env: FAIL -> AssertionError Expects the random number generator to have been
  generated given a seed was passed to reset. Most likely the environment reset function does not
  call `super().reset(seed=seed)`.
SB3 check_env: PASS
SB3 PPO learn(256): OK in 3.4s
```

The seeding failure is real and it is in the source: `TradingEnv.reset(self, seed=None,
options=None)` accepts `seed` and **never uses it**, and `random_start_pct` draws from the
module-level `from random import randint` rather than the seeded `self.np_random`
(`tensortrade/env/generic/environment.py`). Consequence: **episode start points are not
reproducible from a seed.** For a project whose whole thesis is "match-or-beat a baseline", a
non-seedable environment is a methodology defect, not a nit. Issue **#418 "Allow seeding the
random number generators"** has been open since **2022-03-01**.

### 2.5 Execution realism: worse than the marketing, by inspection

| Feature | What the code actually does | File |
|---|---|---|
| Commission | one flat proportional rate, `ExchangeOptions(commission=0.003)` **default = 30 bps** (a crypto taker fee; ~30x too high for US equities) | `oms/exchanges/exchange.py` |
| Maker/taker, borrow, financing | **absent** | — |
| Slippage | **DEAD CODE.** `SlippageModel.adjust_trade()` exists and is **never called anywhere** in the library. `grep -rn 'adjust_trade' tensortrade/` returns only its own definition; `ExchangeOptions` has no slippage field; `Exchange.execute_order()` calls the service and fills at `quote_price` | `oms/services/slippage/*`, `oms/exchanges/exchange.py` |
| Docs correctness | `docs/tutorials/.../01-trading-basics.md` tells users to `from tensortrade.oms.services.slippage import RandomSlippageModel` — **that class does not exist** (it is `RandomUniformSlippageModel`), and wiring it in would still do nothing | docs vs. `slippage/__init__.py` |
| Partial fills | none. `filled = order.remaining.contain(...)` fills the whole remaining quantity every time | `oms/services/execution/simulated.py` |
| Order book / queue / latency | none | — |
| Market impact | none | — |

### 2.6 Same-bar execution, measured

`/tmp/ossaudit/tt_lookahead.py` feeds a price path `[100, 110, 121, ...]` (+10%/bar, unambiguous):
```
obs at reset (close the agent sees): 100.0
obs after step:                      110.0
TRADE -> step 1 side buy price 100.0 size 1000.0 commission 0.00 USD
net_worth after: 1100.0
```
The agent **observes 100.0 and is filled at 100.0**. There is zero decision-to-execution delay:
`action_scheme.perform()` runs before `observer.observe()` advances the feed, so the fill price is
the same close the policy conditioned on. This is not future leakage, but it is an optimistic
same-bar fill, and combined with zero slippage it hands the agent free alpha. FinRL's
`env_stocktrading.py` has the identical structure (trade at `self.state[index+1]`, then `self.day
+= 1`), so this is a whole-field defect, not a TensorTrade-only one.

### 2.7 The dependency cage is the real blocker

TensorTrade 1.0.4 forces `numpy==1.26.4` and `pandas<3`. Measured against what the rest of the
2026 stack now demands:

| Library we want | Its 2026 requirement | Compatible with TensorTrade's cage? |
|---|---|---|
| vectorbt 1.1.0 | `numpy>=2.4.6`, `pandas>=3.0.3,<4.0` | **NO — direct conflict** |
| pyqlib 0.9.7 (resolved) | resolves to numpy 2.5.3 / pandas 3.0.5 | **NO** |
| nautilus_trader 2.x | `requires-python >=3.12,<3.15`, modern numpy | partially (3.12 only) |
| stable-baselines3 2.9.0 | `numpy>=1.20,<3`, `torch>=2.8,<3` | yes (SB3 is permissive) |
| Python 3.13 / 3.14 | — | **NO — 3.12 ceiling via TensorFlow** |

Adopting TensorTrade means freezing the project on Python 3.12 + numpy 1.26 **forever**, carrying
a ~600 MB TensorFlow dependency we never use (we train with PyTorch/SB3), to obtain an execution
model with no slippage and no partial fills. Open PRs **#491 (numpy 2.4.2)**, **#492 (pandas
3.0.1)**, **#493 (gymnasium 1.2.3)** and **#498 (numpy 2.x + pandas 3.x + gymnasium 1.x)** have all
been sitting **unmerged with 0 comments since March/June 2026**. Someone else already tried to fix
this and got no response. That is the maintenance signal that matters.

### 2.8 Honest counter-arguments (steelman)

1. Its core unit tests pass and SB3 `check_env` passes — it is not broken software.
2. The OMS design (Exchange / Portfolio / Wallet / Ledger / Order / Broker / Quantity with
   `Decimal` precision per instrument) is genuinely good and is worth **copying**.
3. The `Stream`/`DataFeed` dataflow DSL is elegant and prevents a class of feature-alignment bugs.
4. Someone is answering issues in 2026: **#501** (2026-09-08, NaN→0.0 silently in the observer)
   and **#500** (2026-07-15, point-in-time macro features) are recent, substantive, on-topic
   issues — which also tells us the community itself now worries about leakage and PIT data.
5. The only named successor is dead: **erhardtconsulting/tensortrade-ng**, 203 stars, last commit
   **2024-08-28**, latest tag `v2.0.0.alpha2` (2024-08-26), **0 commits in the trailing 12
   months**. There is no escape hatch fork.

None of this changes the verdict, because the cost of the cage exceeds the value of the OMS —
and the OMS can be re-implemented in a few hundred lines.

---

## 3. The substitute: a ~300-LOC custom Gymnasium env, with `gym-anytrading` as the skeleton

### 3.1 The comparison that settles it

The same two checkers, same machine, same day:

| | TensorTrade 1.0.4 ("actively maintained") | gym-anytrading 2.0.0 ("stale since 2023-08-27") |
|---|---|---|
| Python tested | 3.12 **only** | **3.13** |
| Resolved stack | numpy 1.26.4, pandas 2.3.3, gymnasium 1.3.0, tensorflow 2.21.0 | numpy **2.5.3**, pandas **3.0.5**, gymnasium 1.3.0, no TF |
| `reset()->(obs,info)` | ✅ | ✅ |
| `step()->5-tuple` | ✅ | ✅ |
| Gymnasium `check_env` | ❌ **FAIL** (seeding) | ✅ **PASS** |
| SB3 2.9.0 `check_env` | ✅ PASS | ✅ PASS (warns: 2-D obs shape) |
| Install size | ~700 MB (TensorFlow) | ~60 MB |

Verbatim, from `/tmp/ossaudit/venvs/at`:
```
numpy 2.5.3 pandas 3.0.5 gymnasium 1.3.0
reset ok (10, 2)
step len 5
GYMNASIUM check_env PASS
SB3 check_env PASS
```
A three-year-stale 8-file library is **more** forward-compatible and **more** API-correct than the
"maintained" one. `gym-anytrading/envs/trading_env.py` calls `super().reset(seed=seed, options=options)`
and declares `metadata = {'render_modes': ['human'], 'render_fps': 3}` — the two things TensorTrade
gets wrong.

### 3.2 But do not *depend* on gym-anytrading either

Its accounting is a toy and its defaults are wrong for equities:
`trade_fee_bid_percent = 0.01` and `trade_fee_ask_percent = 0.005` — **100 bps and 50 bps per
side**, FX/crypto-retail numbers. `_calculate_reward` returns a raw price difference with **no
cost term at all**; `_update_profit` tracks a single all-in long/short position with no cash
balance, no share count, and no multi-asset support. It cannot express a 5-stock portfolio.
Last commit 2023-08-27, 5 contributors, 0 tests.

**Use it as the 300-line skeleton to copy, not as a dependency.** It is MIT.

### 3.3 Migration argument (why writing our own env is cheaper than adopting TensorTrade)

| Dimension | Adopt TensorTrade | Write `rltrader/envs/portfolio_env.py` |
|---|---|---|
| Effort | ~1 week wiring Streams/DataFeeds/Wallets for 5 tickers | ~2–3 days, ~300–500 LOC |
| Python ceiling | **3.12 forever** | whatever SB3 supports (3.13 today) |
| numpy/pandas | **locked at 1.26 / <3** | free — can use vectorbt, qlib, polars |
| Unused deps | TensorFlow (~600 MB) we never call | none |
| Seeding | broken (#418 open since 2022) | `super().reset(seed=seed)`, one line |
| Cost model | 1 flat rate, slippage dead code | exactly the model we choose (§4.3) |
| Partial fills / ADV cap | impossible without forking | trivial to add |
| Debuggability | 209 files, metaclass `Component` registry, `Decimal` `Quantity` algebra | one file we wrote |
| Upstream risk | 4 unmerged modernisation PRs, 0 comments | none |
| What we lose | the OMS + Stream DSL | re-implement the ~200 useful lines of the OMS |

We still **steal** from TensorTrade: the `Wallet`/`Portfolio`/`Ledger` separation, `Decimal`-based
quantities with per-instrument precision, and the component split
(ActionScheme / RewardScheme / Observer / Stopper / Informer). Ideas are free; the dependency cage
is not.

---

## 4. Execution realism ranking — the thing that actually matters

### 4.1 What each engine really models (read from source / typed stubs, 2026-09-14)

| Engine | Commission | Slippage | Partial fills / LOB | Latency / queue | Point-in-time | Verdict for us |
|---|---|---|---|---|---|---|
| **nautilus_trader 2.0.0rc5** | `fee_model` (pluggable; maker/taker, fixed) | `fill_model`, `liquidity_consumption`, `bar_adaptive_high_low_ordering` | **yes** — `book_type` (L1_MBP/L2/L3), `queue_position`, trade-driven fills capped by leaves | **`latency_model`**, `queue_position`, `use_market_order_acks` | user's responsibility (event replay) | **Best available. Use as the validation engine.** |
| gym-continuousDoubleAuction v2 | per-trade, in `Account` | emergent from the book | **yes** — real price–time-priority matching engine, random arrival order per step | discrete arrival ordering, no wall-clock latency | n/a (synthetic) | Vendor the matching engine if we ever need intraday LOB |
| minabides | fees configurable | emergent | **yes** — full Nasdaq LOBSTER order replay | **yes — agent latency + jitter + "thinking time"** | historical replay | Highest realism per line of code; 3 stars / 1 author / `NOASSERTION` licence → reference only |
| freqtrade | ratio fee at entry + exit | configurable, bar-level | no LOB | no | has a `lookahead-analysis` command | Crypto-shaped, GPL-3.0. Study, don't link. |
| vectorbt 1.1.0 | `fees` (proportional) + `fixed_fees` | `slippage=` % of price, per-order | no matching engine; vectorised bar fills | no | user's responsibility | Good for **metrics + fast sweeps**, not for realism |
| microsoft/qlib | explicit cost config + price-limit / suspension handling | simple | `qlib.rl` order-execution sim | no | **Real PIT database** (publication date vs. reporting period) | **Steal the PIT concept**; avoid `qlib.rl` (legacy `gym`) |
| FinRL `env_stocktrading` | `buy_cost_pct` / `sell_cost_pct` only | **none** | **none** — fills whole order at today's close | no | none | Baseline replication only |
| TensorTrade | 1 flat rate (default 30 bps) | **none (dead code)** | **none** | no | none | Avoid |
| gym-anytrading | 100/50 bps hardcoded | none | none | no | none | Skeleton only |
| TradeMaster | unverified — does not install | — | — | — | — | Avoid |

### 4.2 Field-level warning, verified

**arXiv:2603.29086 — "Realistic Market Impact Modeling for Reinforcement Learning Trading
Environments"** (verified 2026-09-14). Its thesis: most open-source RL trading environments assume
negligible or fixed transaction costs, so agents learn behaviour that fails under real execution.
Our own code reading above is an independent confirmation of exactly that claim for TensorTrade,
FinRL, FinRL-Meta and gym-anytrading.

> ⚠ **Metric-context rule.** I found **no** paper or repo in this audit that reports an RL trading
> Sharpe ratio together with all four required fields (asset universe, test window,
> transaction-cost assumption, named baseline). Every headline number encountered was missing at
> least one. Per `RESEARCH_PROTOCOL.md` §2, **none is quoted here as evidence.** The numeric
> comparison table belongs in the literature dossier, not this one, and it must be built only from
> numbers that carry their context.

### 4.3 The cost model we should implement (concrete)

Applied inside our own env, per trade, on notional `N` at price `P` with ADV `V`:

| Component | Value | Rationale |
|---|---|---|
| Commission | **0 bps** | zero-commission US equity brokers (Alpaca, IBKR Lite) |
| Half-spread | **1 bp** for mega-cap; **2 bps** for the rest of our 5 | conservative for liquid large caps on daily/hourly bars |
| Impact | **σ_daily · k · sqrt(N / (P·V))**, k ≈ 0.5–1.0 | square-root impact law; sensitivity-test k ∈ {0.5, 1, 2} |
| ADV cap | reject/clip any order > **1% of ADV** | keeps the sqrt-law in its valid regime |
| Execution delay | fill at **t+1 open**, not at the observed close | removes the same-bar bias measured in §2.6 |
| Borrow (if shorting) | 25–50 bps annualised for large caps | cheap to model, easy to forget |

Every reported result must be published at **three** cost levels (zero / base / 2× base). A result
that survives only at zero cost is not a result.

---

## 5. The RL library layer (deep-research + API cross-check, 2026-09-14)

| Library | Latest | Date | Python | numpy2 | torch | Fit for a custom 5-stock env |
|---|---|---|---|---|---|---|
| **stable-baselines3** | **2.9.0** | 2026-06-15 | 3.10–3.13 | `numpy>=1.20,<3` ✅ | `torch>=2.8,<3` | **Default choice.** PPO + SAC, `check_env`, VecNormalize, callbacks. Verified working here with torch 2.14.0. |
| **sb3-contrib** | **2.9.0** | 2026-06-15 | 3.10–3.13 | via SB3 | via SB3 | **RecurrentPPO** — the one thing we will want for partially-observed markets. |
| **Gymnasium** | **1.3.0** | 2026-04-22 | `>=3.10` | ✅ | optional | The contract. Gymnasium 1.0 (2024-10-08) changed vector envs, autoreset modes and wrapper names — write against 1.3, not tutorials from 2022. |
| ray / RLlib | ray-2.58.0 | 2026-08-23 | `>=3.10` | ✅ | — | New API stack (RLModule / Learner / ConnectorV2) replaced ModelV2 / RolloutWorker / Policy. Heavy. **Only** if we go multi-agent or need cluster scale. |
| PettingZoo | 1.27.0 | 2026-08-13 | 3.10–3.14 | ✅ | — | If the EKG/sentiment agents become true RL agents rather than a feature source. |
| torchrl (`pytorch/rl`) | v0.14.0 | 2026-09-10 | `>=3.10` | ✅ | `torch>=2.1` | Most actively developed alternative; more assembly required. |
| tianshou | v2.0.1 | 2026-04-02 | `>=3.11,<4` | ✅ | `torch>=2,<3` | 2.0 is explicitly **not** backward compatible. Skip. |
| skrl | 2.1.0 | 2026-05-10 | `>=3.10` | — | `torch>=1.11` | Robotics-leaning. Skip. |
| cleanrl | PyPI 1.2.0 | **2023-05-22** | PyPI meta `<3.11` | ✗ | — | **2 of last 100 commits in 12 months.** Copy `ppo_continuous_action.py` if we need a custom loop; never `pip install`. |
| ElegantRL (FinRL's trainer) | v0.3.6 | **2022-06-26** | — | — | — | Last commit 2026-02-20 but no release in 4 years, `NOASSERTION` licence. Avoid. |
| prime-rl | v0.9.0 | 2026-08-25 | — | ✅ | — | LLM post-training. Relevant only if the EKG/ontology layer becomes an LLM policy. No finance env exists in it today. |

---

## 6. Per-repo notes worth keeping

**TradeMaster-NTU/TradeMaster — AVOID, does not install.** `requirements.txt` pins
`ray[rllib]==1.13.0` (2022), `tensorflow==2.11.0`, `mmcv==1.7.1`, `pydantic==1.10.2`,
`kaleido==0.1.0`, `git+https://github.com/optuna/optuna.git`, and **both** `gym` and `gymnasium`.
`setup.py` declares `install_requires=[]`, so `pip install -e .` installs nothing. Measured:
```
$ uv pip install --dry-run -r TradeMaster/requirements.txt    # Python 3.12
  ModuleNotFoundError: No module named 'pkg_resources'
  hint: mmcv@1.7.1 depends on pkg_resources but doesn't declare it as a build dependency
```
**0 of the last 100 commits** fall in the trailing 12 months; latest release `v1.0.0` (2023-03-05);
11 files still `import gym`. The NeurIPS-paper framework is a museum piece.

**AI4Finance FinRL — vendor a subset.** `finrl/meta/env_stock_trading/env_stocktrading.py` is
clean, readable, `import gymnasium as gym`, returns the 5-tuple, models `buy_cost_pct` /
`sell_cost_pct` per asset and a `turbulence_threshold` forced-liquidation rule (a nice risk
feature worth copying). Defects: `reset()` also never calls `super().reset(seed=seed)`; `truncated`
is hardcoded `False` at both return sites; trades execute at today's close then `self.day += 1`
(same-bar, §2.6); reward is raw Δ(net worth), not risk-adjusted. **PyPI is a trap:**
```
$ uv pip install --dry-run finrl
 + finrl==0.3.7        # uploaded 2024-04-12; requires_dist: None  <-- ZERO declared dependencies
```
`pip install finrl` gives an un-importable package. Use a git checkout at a pinned SHA, or copy
the ~400 lines we need. **Use FinRL only to replicate the published PPO baseline** (the brief asks
for it) — do not build the production path on it.

**FinRL-Meta — data processors only.** 15 files still `import gym`; `python_requires>=3.6`;
release `v0.3.6` from 2022-06-26. Its value is the `data_processors/` adapters (yfinance, Alpaca,
CCXT, WRDS), not its environments.

**FinRL-Trading is now "FinRL-X"** — `arXiv:2603.21330` "FinRL-X: An AI-Native Modular
Infrastructure for Quantitative Trading" (**verified**), PyPI `finrl-trading`, `python>=3.11`,
weight-vector interface (`w_t = R(T(A(S(X))))`), `bt`-powered backtest with transaction costs, and
FMP / Yahoo / WRDS data. FinRL's own README now points production users here. **0 test files**,
14 contributors. Architecturally the most interesting idea in the AI4Finance stack — the
"target-weight vector is the only contract between strategy and execution" design is worth
adopting in our own code — but not yet a dependency.

**microsoft/qlib — steal the PIT idea, avoid `qlib.rl`.** `pip install pyqlib` resolves on Python
3.12 to numpy 2.5.3 / pandas 3.0.5 **but drags in `gym==0.26.2`**, whose own import banner says it
"does not support NumPy 2.0". So `qlib.rl` is a latent breakage. `Development Status :: 3 - Alpha`,
478 open issues, last release v0.9.7 (2025-08-15). Its genuinely valuable asset is the
**point-in-time database** that records publication date vs. reporting period so a backtest cannot
see restated fundamentals. We should implement the same discipline in our data layer.

**freqtrade — best-run project in this audit, wrong shape for us.** Last commit the day of this
audit, 342 contributors, 105 test files, 10 CI workflows, Python 3.11–3.14, `freqai/RL/` contains
proper Gymnasium envs (`BaseEnvironment`, `Base3/4/5ActionRLEnv`) that correctly use
`gymnasium.utils.seeding`. But it is crypto-exchange-shaped (`ccxt>=4.5.76`) and **GPL-3.0**, which
is a licensing problem if any of this ships. **Read `freqai/RL/BaseEnvironment.py` as the
reference implementation; do not link against it.**

**nautechsystems/nautilus_trader — build on it, with eyes open.** Only engine here with
`fill_model` + `latency_model` + `fee_model` + `book_type` + `queue_position` +
`liquidity_consumption` + `margin_model` in its public config (read from
`python/nautilus_trader/backtest/__init__.pyi`). **Risk:** it is mid-rewrite. `version.json` says
`v2.0.0rc5`; v2 is the Rust core + PyO3 package classified `Development Status :: 4 - Beta`;
`MIGRATION_V2.md` states legacy v1 lives on `develop_v1` and "accepts critical security backports
for approximately three months after the v2 cutover… does not receive new feature or parity work",
and that v1 and v2 **both import as `nautilus_trader`** so they need separate venvs. **LGPL-3.0.**
Plan: use it in an isolated venv as a *validation* engine only, pinned to one RC, behind our own
adapter.

**ChuaCheowHuan/gym-continuousDoubleAuction — genuinely revived, small but serious.** Not the 2019
repo my priors expected: rewritten through Dec 2025 – Sep 2026 (last commit 2026-09-13),
`python_requires>=3.12`, CI on 3.12 with torch 2.13 + RLlib, 47 test files, 25 numbered design
docs including a `16_verification_log.md`, real price–time-priority `OrderBook`, league-based
self-play. 154 stars, 3 contributors, no tagged release. **Vendor the matching engine** if we ever
need an endogenous LOB; do not make it a load-bearing dependency at 3 contributors.

**davebyrd/minabides — highest realism, lowest adoption.** Full Nasdaq LOBSTER order replay into a
simulated exchange, discrete-event kernel, agents delayed by their real computational "thinking
time", latency + jitter, RL agent adapted from cleanrl, train/validation/out-of-sample splits
automated, slurm support. **3 stars, 1 contributor, last commit 2025-07-22, licence
`NOASSERTION`.** By the ABIDES author. Read it for how to model latency; do not depend on it, and
resolve the licence before copying code.

**polakowo/vectorbt — useful, but it is the numpy-2 camp.** v1.1.0 (2026-07-05) requires
`numpy>=2.4.6`, `pandas>=3.0.3,<4.0`, `numba>=0.66`, Python 3.11–3.14. Exposes `fees`,
`fixed_fees` and `slippage` per order. `NOASSERTION` licence on GitHub (the author monetises
vectorbt PRO) — **check the licence text before shipping.** It cannot coexist in one venv with
TensorTrade: that single fact is the cleanest proof that the brief's stack choice is untenable.
