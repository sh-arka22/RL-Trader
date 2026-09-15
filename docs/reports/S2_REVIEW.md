# S2 Data Pipeline — Adversarial Review

Reviewer: independent adversarial read of `main` @ `e1b7526`.
Method: read-only. Every claim below was reproduced against the shipped store
(`data/`, 29,410 bars, 5 tickers x 2 providers, 2015-01-02 -> 2026-09-14) or against a
`tempfile` scratch store, using the repo's own env (`.venv-core/bin/python`).
No repo source file was modified. `make check` = 6/6 PASS and `make test` = 20 passed at
review time; that is the starting point, not the conclusion. (`make check` rewrites
`data/manifest.json` as a side effect — see **L7** — so that one artifact is dirty in git
because of this review.)

**Headline: the gate suite is weaker than it reads.** Two gates (G4, and G1's trailing
window) cannot fail on realistic bad data. Two gates (G3, G5) are scored on a statistic
that the project's own report says cannot catch the bug they exist to catch. One gate
(G5) disappears entirely on a network error and the run still exits 0. And the store's
`as_of` axis returns prices recorded ten years after the knowledge instant, which the
`store.py` docstring explicitly promises it does not.

Severity key: **C** critical (silent wrong numbers reach a consumer, or a real failure
reads as a pass), **H** high, **M** medium, **L** low.

---

## C1 — `Store.bars(as_of=...)` returns future-recorded, future-adjusted prices; the docstring says it does not

**Where:** `rltrader/data/store.py:8-11` (claim), `rltrader/data/store.py:118-129` (code),
`rltrader/data/store.py:138-147` (same defect in `actions()`).

The class docstring states:

> ``as_of`` queries pick the latest row per (ticker, date, provider) whose ``recorded_at``
> **and** ``first_public_at`` both precede the knowledge timestamp.

That is not what the code does. `as_of` filters `first_public_at` only (`store.py:122`).
`recorded_at` is filtered exclusively by the separate `recorded_before` keyword
(`store.py:127`), which defaults to `None`. The two conditions are never AND-ed.

**Reproduction (scratch store):**

```python
t.write_bars(df_close_100, "X", "yahoo", "i1", recorded_at=pd.Timestamp("2020-01-03", tz="UTC"))
t.write_bars(df_close_10,  "X", "yahoo", "i2", recorded_at=pd.Timestamp("2026-01-01", tz="UTC"))
t.bars("X", as_of=pd.Timestamp("2021-01-01", tz="UTC"))
# -> close = 10.0, recorded_at = 2026-01-01+00:00   (a row recorded 5 years after the as_of instant)
```

**Reproduction (shipped store, worse):**

```
store.bars('NVDA', as_of='2016-01-01')  -> 504 rows
every recorded_at = 2026-09-14 23:49Z   = 3,909 days (10.70 y) AFTER the as_of instant
close returned for 2015-01-02           = 0.50325
the tape actually printed on 2015-01-02 = 20.13      (ratio 40.0x)
```

The 40x gap is the 2021-07-20 4:1 and 2024-06-10 10:1 splits — neither of which had
happened at `as_of=2016-01-01`. `adjust.py:24` states the problem itself ("The stored
series is NOT immutable. The next split silently rewrites every past price"), and
`store.py:108` names `as_of` as "the axis a backtest walks". So the documented backtest
path hands back prices retro-adjusted by corporate actions from the future.

Returns survive a uniform rescale, so this is invisible to any returns-based check —
which is why G5 (`build.py:148-150`) does not see it, and G4 (`build.py:110-127`) does
not look at `recorded_at` at all. Price *levels*, position sizing, share counts, round
lots, and any dollar-denominated feature are wrong.

**Fix:** in `bars()` and `actions()`, when `as_of` is given and `recorded_before` is not,
default `recorded_before = as_of`. Callers that genuinely want the revised series pass
`recorded_before=None` explicitly. Then add a gate asserting
`bars(as_of=T)["recorded_at"].max() <= T`. If the current behaviour is intended, the
docstring must be corrected and every consumer warned — but note that `tests/test_store_pit.py:29-35`
encodes the current behaviour as *desirable*, so the intent is genuinely ambiguous and
needs an explicit decision rather than a doc patch.

---

## C2 — G3 and G5 are scored on the median, the one statistic the project already proved cannot catch this bug

**Where:** `rltrader/data/build.py:101-103` (G3), `rltrader/data/build.py:158-161` (G5).

Both gates compute `worst = max(r["median_bps"] for r in usable)` and compare it to a
1 bp tolerance. `p95_bps`, `max_bps` and `n_over_1bp` are computed
(`build.py:152-155`, `reconcile.py:37-39`) and then discarded.

`docs/reports/S2_DATA.md:27-30` describes the original double-adjustment bug:

> The symptom was a ~90,671 bp return error on *exactly the split dates and nowhere else*
> — five bad rows hidden in 14,705 good ones, with a median error of 0.0004 bp.
> **No aggregate statistic would have caught it.**

The gate that was built to prevent a recurrence is scored on exactly that statistic.

**Reproduction (G5, real data):** take the real stockanalysis reference for NVDA, multiply
one close by 10, feed it to `gate_corporate_actions`:

```
rows: {'ticker':'NVDA','n':2940,'median_bps':0.000423,'p95_bps':0.00771,
       'max_bps':93138.72,'n_over_1bp':50}
G5 passed: True
detail:    "worst median split-adjusted return divergence 0.0004 bp (tol 1.0)"
```

A 93,138 bp error — the same order as the historical 90,671 bp — passes, and the pass
message quotes 0.0004 bp.

**Reproduction (G3, real data):** corrupt one NVDA bar by 10x and summarise:

```
{'n':2941, 'median_bps':0.0, 'p95_bps':0.0, 'max_bps':9000.0, 'n_over_1bp':1}
gate metric (median) = 0.0  ->  <= 1.0  ->  PASS
```

The shipped numbers already show the blind spot in production: `S2_DATA.md:80-84` records
NVDA `max 4.880 bp / 276 rows >1bp` and META `max 17.513 bp`, and `S2_DATA.md:101-104`
records META `max 17.928 bp` — all under a gate that reports "worst median divergence
0.0004 bp".

**Fix:** gate on `n_over_1bp == 0` (or `max_bps <= tol`). Where a known, explained
residue exists (the `1 - div/prev_close` dividend model, `S2_DATA.md:107-109`), carve it
out by an explicit, dated allowlist of ex-dates with a stated bound — not by switching to
a statistic that hides everything. Keep the median as a reported metric only.

---

## C3 — A network failure on the G5 reference deletes the gate, and the run still exits 0

**Where:** `scripts/build_dataset.py:60-66`, `rltrader/data/build.py:212-213`,
`scripts/build_dataset.py:79-81`.

```python
# build_dataset.py:63-66
    try:
        ref_adj[t] = sa.adjusted_reference(t, start, end)
    except Exception as e:      # a nice-to-have reference, never a dependency
        print(f"  ref adj {t}: {type(e).__name__}: {e}")
```

If every ticker raises, `ref_adj` stays `{}`. `build.py:212` is `if ref_adj:`, so
`gate_corporate_actions` is **never appended to the gate list**. `build_dataset.py:79`
then computes `failed = [g.name for g in gates if not g.passed]` — G5 is not in `gates`,
so it cannot be in `failed`.

**Reproduction:**

```python
run_gates(s, ["NVDA"], start, end, ref_adj={})
# -> ['G1_coverage','G2_no_phantom_sessions','G3_reconcile','G4_point_in_time','G7_printed_prices']
# -> failed = []      -> "all gates passed"  -> exit 0
```

This is not hypothetical. The reference comes from an undocumented keyless endpoint that
the provider module itself documents as fragile: `stockanalysis.py:44-46` — "The endpoint
rate-limits at roughly ten rapid requests and then returns HTML instead of JSON, which
surfaces as a JSONDecodeError — observed 2026-09-14 during the first full S2 ingest."
A rate-limit during a rebuild silently converts a hard gate into no gate, and the manifest
written at `build_dataset.py:74-78` records a clean run with G5 simply absent.

The same shape exists one level down: a per-ticker failure yields `{"status": "NO_REF", "n": 0}`
(`build.py:139`), which `build.py:158` filters out of `usable`. Four of five tickers can
drop out and the gate still passes on the fifth.

**Fix:** always append the gate. On a missing reference emit
`GateResult("G5_corporate_actions", False, "reference unavailable: ...")`. Make a
per-ticker `NO_REF` a failure, or at minimum assert
`len(usable) == len(tickers)`. A gate that is absent must never read as a gate that passed.

---

## H1 — `as_traded()` returns split-*adjusted* prices for any point-in-time `as_of`, which is the documented use case

**Where:** `rltrader/data/adjust.py:63-74` and `:94-101`; filter at `adjust.py:38-44`;
`rltrader/data/schema.py:56-60`.

`unadjust_factors()` selects splits with `_known(actions, as_of, "split")`, i.e.
`first_public_at <= as_of` (`adjust.py:41`). `store.action_public_at` sets
`first_public_at` to the **ex-date** 09:30 ET (`store.py:38-40`, `schema.py:60`). So a
split is invisible to `as_traded()` until its own ex-date.

But the stored series is split-adjusted as of `recorded_at` — i.e. as of *today*
(`adjust.py:15-17`). The adjustment is already in the data. The `as_of` filter therefore
refuses to undo an adjustment that is unconditionally present. Un-adjustment is a property
of the storage basis, not of the knowledge state, and the code conflates them.

**Reproduction (real NVDA data):**

```
NVDA date 2024-05-31, stored close 109.6330, actually printed ~1096.33

as_of = 2024-06-01        factor  1.0   traded_close =  109.6330    <- wrong, 10x
as_of = 2024-06-10 13:29Z factor  1.0   traded_close =  109.6330    <- still wrong,
                                                        1 minute before action_public_at
as_of = 2024-06-10 13:31Z factor 10.0   traded_close = 1096.3300    <- correct
as_of = 2026-09-15        factor 10.0   traded_close = 1096.3300
```

At a point-in-time `as_of`, `as_traded()` is wrong by exactly the split ratio — 10x,
-9,000 bp on the price, and the answer flips on the 09:30 ET boundary of the split's own
ex-date. Share counts would be 10x too large for NVDA before 2024-06-10 and 4x too large
for AAPL before 2020-08-31. This is verbatim the failure `S2_DATA.md:38-39` warns about:

> A backtest that sizes positions off the stored NVDA price in 2024 buys ten times too
> many shares.

`as_traded()` is the function that is supposed to prevent that, and under PIT `as_of` it
reproduces it.

The report's description of the function is also not what the code does.
`S2_DATA.md:33-35` says `as_traded()` "multiplies by every split ratio with `ex_date > t`".
The code multiplies by every split with `first_public_at <= as_of` **and** `ex_date > t`.
Those agree only when `as_of` is late enough — and the only caller in the gate suite,
`build.py:177,183`, passes `as_of=now`. `tests/test_real_data.py:21` does the same.
So the sole regime in which the function is correct is the sole regime in which it is
tested.

Note `tests/test_adjust.py:45-47` (`test_unadjustment_is_point_in_time`) asserts the
current behaviour is *right* — "a future split leaked into a past knowledge state". That
test enshrines the bug. Recovering the price that was printed in the past is not
look-ahead; it is the historical record. The genuine look-ahead risk is C1, and that one
is unguarded.

**Fix:** select splits for un-adjustment by `ex_date <= recorded_at` (those actually baked
into the stored series) and `ex_date > t`, independent of `as_of`. Keep the `as_of` filter
for dividends, where it is meaningful. Replace `test_unadjustment_is_point_in_time` with a
test that `as_traded(..., as_of=pre_split_date)` still recovers the printed price. Add a
G7 anchor evaluated at a PIT `as_of`.

---

## H2 — G1 coverage truncates its own test window to the last bar received, so it cannot detect a stale or truncated feed

**Where:** `rltrader/data/build.py:70`.

```python
miss = cal.missing_sessions(list(g["date"]), start, min(end, max(g["date"])))
```

The upper bound of the coverage check is the provider's own last bar. Every session after
that is excluded from the test by construction.

**Reproduction:** wrap the real store so every series stops at 2020-01-02, then run the
gate unchanged:

```
G1 with all data truncated at 2020-01-02  ->  passed = True
detail: "0 missing sessions for every ticker/provider"
sessions actually absent after the cut: 1,684   (per ticker, per provider)
```

57% of the dataset — 6.7 years — can be missing and the gate reports zero missing
sessions. A provider that silently stops updating, an `ingest` that dies partway, or a
cached endpoint serving a stale range all read as full coverage.

This is the exact failure mode `calendar.py:4-5` says the module exists to prevent:
"Without an authoritative calendar those two are indistinguishable, which is how '0%
missing' gets silently claimed on incomplete data." The calendar is authoritative; the
gate declines to use it past the last row.

Nothing else covers it. `tests/test_real_data.py:31` asserts only
`b["date"].min() <= dt.date(2015, 1, 5)` — the start. No test or gate asserts the end.

**Fix:** pass `end` through unchanged. Report a trailing gap separately as
`stale_by_n_sessions` and fail when the last stored session is more than 1-2 sessions
behind the last expected session `<= end`.

---

## H3 — G4 is a tautology: it cannot fail on any data that `write_bars` produced

**Where:** `rltrader/data/build.py:110-127`, `rltrader/data/store.py:62`,
`rltrader/data/build.py:202`.

`store.write_bars` sets `first_public_at = bar_public_at(out["date"])` — a pure function
of the `date` column, `date + 16:00 ET`. `gate_point_in_time` then asks whether, standing
at `d 09:29 ET`, any row with `date >= d` is visible. For `date == d`,
`first_public_at = d 16:00 ET > d 09:29 ET`, so `store.py:122` already removed it. For
`date > d` it is later still. For `date < d` the `date >= d` filter excludes it. The leak
list is empty by arithmetic.

**Reproduction (arithmetic, all 2,941 real sessions):**

```
sessions: 2941
rows where first_public_at <= that session's own 09:29 probe: 0
```

**Reproduction (behavioural, independent):** a scratch store in which every bar's OHLC is
literally the *next* day's value — maximal look-ahead in the content — still yields
`G4 passed: True, "no future bar visible at any probe instant"`.

G4 validates `bar_public_at()` against arithmetic the same module just performed. It
tests no provider property, no action, and no *content*. The C1 leak — every row recorded
10.7 years after the `as_of` instant — is invisible to it, because G4 never inspects
`recorded_at`. It also never inspects the `actions` table: `store.actions` is not called
anywhere in the gate. Forging an action's `first_public_at` from 2020-08-31 back to
2015-01-01 (5.6 years of look-ahead) leaves G4 green while
`store.actions('NVDA', as_of='2015-06-01')` cheerfully returns the 2024 split.

**On the specific question — is every 250th session enough?** No, but that is the second
problem, not the first. The gate would pass at stride 1. The stride matters only for a
*hypothetical* mis-stamped row, and there the numbers are bad:

```
probe list = cal.sessions(2018-01-01, min(end, 2026-01-01))[::250]  ->  9 dates
  2018-01-02, 2018-12-31, 2019-12-27, 2020-12-23, 2021-12-21,
  2022-12-19, 2023-12-18, 2024-12-16, 2025-12-16
sessions 2015-01-01..today = 2,942      fraction probed = 0.306%
sessions in 2015-2017 never probed      = 755   (START is 2015-01-01, build.py:25)
sessions after the 2026-01-01 cap       = 176
```

A single bar mis-stamped with `first_public_at = date 00:00 UTC` is detected only when the
bad date *is* a probe date, because the leak is shorter than one session. Measured:

```
bad bar on 2019-06-03 (not a probe)  ->  G4 passed: True
same store, probing 2019-06-03       ->  G4 passed: False
   "Z @ 2019-06-03: 1 future bars visible (max date 2019-06-03)"
```

P(catch) = 9/2942 = 0.306%. One mis-stamped bar is missed roughly 326 times out of 327.

**Fix:** three changes, in order of value. (1) Make the gate test something the writer did
not compute: assert `bars(as_of=T)["recorded_at"].max() <= T` under strict PIT, and assert
`actions(as_of=T)["ex_date"].max() < T`. (2) Add a negative control — inject a deliberately
mis-stamped row into a scratch store and assert the gate FAILS. A gate with no negative
control is not a gate. (3) Only then worry about stride; probe the full `START..end` range
and sample far more densely, or check the invariant vectorised over every row instead of
sampling at all.

---

## M1 — G7 silently shrinks when a ticker is missing, and its tolerance is 25 bp

**Where:** `rltrader/data/build.py:175`, `:181-182`, `:194-196`.

`if bars.empty: continue` adds nothing to `bad` and nothing to `rows`. The anchor simply
disappears.

**Reproduction:** make NVDA bars empty:

```
G7 passed: True
detail:  "un-adjusted closes match the tape on 2 split anchors"
anchors actually checked: ['TSLA', 'AAPL']
```

The 10:1 NVDA anchor — the largest ratio in the set and the one that exposed the original
bug — vanishes without a warning, and the message reads like a success.

Separately, `tol_bps: float = 25.0` is 0.25%. `S2_DATA.md:35` claims the un-adjustment
"recovers the three tape prints above to **0.00 bp**"; the gate would accept 24 bp. For a
comparison against exact tape prints, 25 bp is not a tolerance, it is an absence of one.

**Fix:** assert `len(rows) == len(PRINTED_CLOSES)`; make a missing bar a failure; tighten
`tol_bps` to ~1 bp.

---

## M2 — no G7 anchor sits on or after an ex-date, so an off-by-one at the boundary passes the real-data gate

**Where:** `rltrader/data/build.py:168-172`.

All three anchors are the last session *strictly before* their split. The `>= ex_date`
side of the boundary is never exercised on real data.

**Reproduction:** change `adjust.py:73` from `mult.index < a["ex_date"]` to `<=` —
the classic off-by-one — and run the gate unchanged:

```
G7 passed: True
detail: "un-adjusted closes match the tape on 3 split anchors"
traded_close(2024-06-10, the EX-DATE) = 1217.90      (NVDA really printed ~120.37)
```

A 10x error on every ex-date in the dataset, and the ground-truth gate says the
un-adjustment matches the tape.

**The boundary itself is correct as written.** Verified on real NVDA data, strict `<` is
right:

```
unadjust factor 2015-01-02  = 40.0      (both splits ahead)
unadjust factor 2021-07-19  = 40.0
unadjust factor 2021-07-20  = 10.0      (ex-date of the 4:1 — factor already dropped)
unadjust factor 2024-06-07  = 10.0
unadjust factor 2024-06-10  =  1.0      (ex-date of the 10:1)
```

Credit where due: `tests/test_adjust.py:41` (`tr.loc[2, "traded_close"] == 50.0`, where
index 2 *is* the ex-date) and `tests/test_adjust.py:53` (`f.loc[EX] == 1.0`) do pin the
boundary on synthetic data. So the unit suite would catch this; the real-data gate would
not. Given that the project's stated lesson is "the ground-truth anchor caught it, no
aggregate statistic would have" (`S2_DATA.md:30`), the ground-truth anchor set should
cover the boundary it is meant to police.

**Fix:** add `("NVDA", dt.date(2024, 6, 10)): 120.37` and at least one mid-history
non-split anchor to `PRINTED_CLOSES`.

---

## M3 — `adjusted()` ignores the stored `price_basis` column; the raw-price guard is on the keyword, not on the data

**Where:** `rltrader/data/adjust.py:77-83`, `rltrader/data/store.py:60`,
`rltrader/data/providers/keyed.py:20,46`, `rltrader/data/build.py:49-50`.

`adjusted()` raises `NotImplementedError` when the `price_basis` **parameter** equals
`"raw"`. That parameter defaults to `SPLIT_ADJUSTED` and is never derived from
`bars["price_basis"]`, which `store.write_bars` faithfully records per row.

**Reproduction:**

```python
b = s.bars("NVDA", provider="yahoo").copy()
b["price_basis"] = "raw"                 # as a raw provider would have stored them
adjusted(b, s.actions("NVDA"), as_of=now)
# -> no error; adj_close[0] = 0.4819
```

`as_traded()` (`adjust.py:94`) has no guard of any kind.

This is a loaded gun, not a theoretical one. `keyed.py:20,46` register Tiingo and Alpaca
with `price_basis="raw"`, `build.py:49-50` stores that basis faithfully, and
`keyed.py:6-8` advertises the upgrade path: "They are implemented anyway and register
automatically the moment a key appears, so adding a key upgrades the reconciliation from 2
to 3 sources with no code change." The moment a key appears, genuinely raw bars enter the
store and every consumer treats them as split-adjusted. `as_traded()` would multiply
already-raw prices by the split ratios a second time — the original double-adjustment bug,
restored through the documented upgrade path, with no code change, exactly as advertised.

There is a second, quieter dependency in the same place. `dividend_factors`
(`adjust.py:59`) computes `1 - value / prev_close`. That is dimensionally consistent today
only because the dividend values yfinance returns are themselves split-adjusted as of
`recorded_at`, matching the closes. Verified in the shipped store:

```
AAPL ex 2020-08-07   stored 0.205000   printed 0.82    ratio  4.000   (4:1 ex 2020-08-31)
AAPL ex 2019-11-07   stored 0.192500   printed 0.77    ratio  4.000
NVDA ex 2015-02-24   stored 0.002125   printed 0.085   ratio 40.000   (= 4 x 10)
NVDA ex 2021-06-09   stored 0.004000   printed 0.16    ratio 40.000
NVDA ex 2024-03-05   stored 0.004000   printed 0.04    ratio 10.000

cancellation check, AAPL ex 2020-08-07:
  last close before ex = 113.9025 stored, unadjust_factor 4.0 -> printed 455.61
  0.205 / 113.9025 = 17.9978 bp      0.82 / 455.61 = 17.9978 bp      diff 0.000000 bp
```

Numerator and denominator carry the same split adjustment, so it cancels exactly. The
counterfactual shows the size of the dependency: had the values been stored as-printed,
`0.82 / 113.9025 = 72.0 bp` — a **+54.0 bp** error on that single dividend, and 40x too
large for NVDA in 2015. With a raw-price provider the denominator loses the adjustment
while the yahoo-sourced numerator keeps it, and that is exactly what happens.

This invariant is load-bearing, nowhere documented, and nowhere tested.

**Fix:** have `adjusted()` and `as_traded()` read `bars["price_basis"]`, assert it is
homogeneous, and branch or raise on the *data*. Add a test that a raw-stamped frame raises.
Document the "dividend values share the price basis" invariant in `adjust.py` and assert it.

---

## M4 — `dividend_factors()` violates its own "1.0 at the last bar" contract, and G5 structurally cannot notice

**Where:** `rltrader/data/adjust.py:47-60` (docstring at `:48`), `rltrader/data/build.py:53`.

The docstring promises "1.0 at the last bar". The loop applies any *known* dividend to
every bar with `date < ex_date` (`adjust.py:59`). If `ex_date` falls after the last bar,
every bar — including the last — is scaled, and the normalisation is lost.

**Reproduction — real data, no synthetic frames, on the normal point-in-time read
path.** Stand at 11:00 ET on a NVDA dividend ex-date:

```
as_of = 2025-03-12 15:00Z   (11:00 ET, the ex-date of the NVDA 0.01 dividend)
  store.bars(as_of)    -> last visible bar = 2025-03-11   (bar first_public_at = 16:00 ET)
  store.actions(as_of) -> already includes dividend ex 2025-03-12 value 0.01
                          (action_public_at = 09:30 ET, store.py:38-40)
  dividend_factors -> factor at the last bar = 0.9999080544   ( -0.9195 bp )
                      and CONSTANT across the whole series (last 3 bars identical)
  adjusted()       -> last bar close 108.7600 -> adj_close 108.7500, adj_factor 0.99990805

control, as_of = 2025-03-12 13:00Z (09:00 ET, before action_public_at):
  factor at the last bar = 1.0 exactly
```

The window is the 6.5 hours between `action_public_at` (09:30 ET) and
`bar_public_at` (16:00 ET) — i.e. the whole trading day, every ex-dividend day, for any
consumer reading at a PIT `as_of`. Stress version, bars truncated at 2024-03-04 with the
full action list (11 dividends after the last bar): factor at the last bar
**0.9931633170 = -68.4 bp**.

A synthetic check gives the same shape: a dividend `ex_date=2026-12-01, value=0.30,
first_public_at=2026-09-01` read at `as_of=2026-09-10` yields `f.min() == f.max() ==
0.9990993` — the entire AAPL series rescaled by 9 bp, last bar included.

The shipped on-disk store does **not** currently trigger it: NVDA, AAPL, META and XOM all
have zero dividends after their last bar, and the factor is exactly `1.0000000000` at the
last bar. Latent on the full read, live on the `as_of`-sliced read.

A uniform rescale cancels in `pct_change`, which is precisely why G5 (`build.py:148-150`,
returns-based) can never detect it. Returns stay right; levels go wrong. Combined with C1,
this is the second independent way `adj_close` *levels* can be wrong while every
levels-blind gate stays green.

**Fix:** drop dividends with `ex_date > bars["date"].max()`, or renormalise
`mult /= mult.iloc[-1]`. Then assert the invariant in a test.

---

## L1 — actions are never filtered by provider; a second action source would double-adjust by the split ratio

**Where:** `rltrader/data/store.py:148-149`, `rltrader/data/adjust.py:38-44,52,70`,
`rltrader/data/build.py:142,183`.

`Store.actions()` deduplicates on `["ticker","ex_date","kind","provider"]` — deliberately
keeping one row **per provider**. `adjust._known()` takes no provider argument and the
loops in `dividend_factors`/`unadjust_factors` iterate over every row returned. Both
call sites (`build.py:142`, `build.py:183`) pass `store.actions(t)` unfiltered.

**Reproduction (scratch store):** one 10:1 NVDA split reported by *both* yahoo and
stockanalysis:

```
store.actions('NVDA')                -> 2 rows for the same (ex_date, kind)
unadjust_factors at 2024-06-07       = 100.0        (correct: 10.0 — ratio applied twice)
traded_close  2024-06-07             = 12,088.80    (the tape printed 1,208.88)  +90,000 bp
traded_volume 2024-06-07             = 10,000       (should be 100,000 — divided by 100)
dividend factor                      = 0.9998357896 (vs 0.9999178915) — an extra
                                        -0.821 bp per duplicated dividend
```

`store.actions()` has no `provider=` argument at all (`store.py:132`), while `bars()` does
(`store.py:116-117`). Same-provider re-ingest *is* safe — a third yahoo part with a new
`ingest_id`/`recorded_at` still yields the correct single row, so `keep="last"` works as
intended. The defect is strictly cross-provider.

Dormant today, and only by luck: `stockanalysis.py:84-86`, `keyed.py:40-41` and
`keyed.py:78-79` all return empty action frames, so the shipped store's actions are 100%
yahoo (NVDA 49 rows, AAPL 48, TSLA 2, META 10, XOM 47; zero `(ex_date, kind)` pairs from
more than one provider). The day a second action source is added — the explicitly
advertised upgrade path, see M3 — G5 and G7 break silently by a factor of the split ratio.
This is the same failure class `adjust.py` was rewritten to eliminate.

**Fix:** either make actions provider-agnostic (dedup on `["ticker","ex_date","kind"]`
with an explicit, documented precedence order), or add a required `provider=` argument to
`_known()`. Either way, assert in `adjusted()`/`as_traded()` that the actions frame has at
most one row per `(ticker, ex_date, kind)`.

---

## L2 — no minimum sample size on G3 or G5

**Where:** `rltrader/data/build.py:100`, `rltrader/data/build.py:158`.

`usable = [r for r in rows if r.get("n")]` admits `n = 1`.

**Reproduction:** a 2-bar overlap summarises to `{'n': 2, 'median_bps': 0.0}` and counts as
usable. If a provider's date range collapses to a handful of overlapping sessions, the
reconciliation reads as a clean pass over a sample too small to mean anything, and the
detail string still says "worst median divergence 0.0004 bp".

This compounds the inner-join behaviour noted under "no bug found" below: `reconcile.compare`
silently drops non-overlapping dates, and nothing downstream checks how many survived.

**Fix:** require `n >= 0.95 * len(cal.sessions(start, end))` per ticker and fail otherwise.
Report `n` and the expected count in the detail string.

---

## L3 — `ingest` swallows every exception per (provider, ticker); failures are never counted

**Where:** `rltrader/data/build.py:57-58`.

```python
            except Exception as e:
                log(f"  {name:14s} {t:5s} FAILED: {type(e).__name__}: {e}")
```

Printed, never raised, never recorded in the manifest, never re-checked. Partial damage is
possible: bars are written at `build.py:49` *before* actions at `build.py:54`, so an
exception inside `prov.actions()` leaves bars stored with no actions for that ticker.

In fairness, the total-loss case *is* caught downstream — G1 reports `"no data"`. This is
therefore not itself a fail-reads-as-pass, unlike C3. But the failure count never reaches
the manifest written at `build_dataset.py:74-78`, which records gate results only, so a
run with four of ten fetches failing is indistinguishable from a clean one in the
artifact.

**Fix:** accumulate failures into a list, include it in the manifest, and fail the run if
any `(provider, ticker)` pair errored.

---

## L4 — non-deterministic tie-break on equal `recorded_at`, in exactly the regime G6 pins

**Where:** `rltrader/data/store.py:94-100` (`_read` file order), `:128-129` (sort + dedup),
`:24` (`uuid4().hex[:12]`), `scripts/build_dataset.py:43-49` (`--repro`).

`sort_values` with a list of keys goes through `np.lexsort` and *is* stable, so this is not
a quicksort problem. The consequence is subtler: ties on all four sort keys are broken by
**input order**, which is `sorted(base.glob(...))` over filenames
`part-<provider>-<ingest_id>.parquet` where `ingest_id` is a random uuid. The tie-break key
is random.

**Reproduction:** two stores with identical logical content and identical
`recorded_at=2024-01-01Z`, differing only in which uuid each part got:

```
store1: close 100 -> part-yahoo-aaaa...,  close 10 -> part-yahoo-zzzz...  ->  bars() close = 10.0
store2: close  10 -> part-yahoo-aaaa...,  close 100 -> part-yahoo-zzzz...  ->  bars() close = 100.0
200 fresh stores with real random ingest ids -> winner histogram {100.0: 102, 10.0: 98}
```

A coin flip across runs; stable within one store (20/20 identical calls), so it is a
cross-run, not cross-call, hazard.

The shipped store is safe by accident: `ingest` lets `write_bars` default `recorded_at` to
`Timestamp.now()` per call (`store.py:56`), giving 120 distinct timestamps, and 0 of 29,410
`(ticker,date,provider)` groups have more than one distinct `recorded_at`.

The G6 path is the exception. `build_dataset.py:43-49` pins
`recorded_at = 2000-01-01Z` and `ingest_id="repro"` for both runs — which is precisely the
regime that manufactures ties. Two further observations on that path:

- The pinned `recorded_at` of 2000-01-01 is **earlier than `first_public_at` for every
  bar** — bitemporally impossible (recorded before it was public). Nothing asserts
  `recorded_at >= first_public_at`.
- G6 compares `payload_digest` — a hash of parquet *bytes* (`manifest.py:38-46`) — not the
  output of `bars()`. A tie-break flip changes what queries return while leaving the byte
  digest identical, so G6 cannot detect the thing it exists to detect.
- `--repro` also only ingests `a.tickers[:1]`, so reproducibility is demonstrated on 1 of 5
  tickers.

**Fix:** add `ingest_id` as a final deterministic tie-break in the sort, or sort the files
by `(provider, recorded_at, ingest_id)` rather than by path. Assert
`recorded_at >= first_public_at` on write. Have G6 compare `bars()` output, not only file
bytes.

---

## L5 — documentation states guarantees the code does not provide

Collected, since several are load-bearing:

| Location | Claim | Reality |
|---|---|---|
| `store.py:9-11` | `as_of` filters `recorded_at` **and** `first_public_at` | Only `first_public_at`. See **C1**. |
| `S2_DATA.md:33-34` | `as_traded()` multiplies by every split with `ex_date > t` | Also gated on `first_public_at <= as_of`. See **H1**. |
| `S2_DATA.md:35` | un-adjustment "recovers the three tape prints to 0.00 bp" | True only at `as_of=now`; gate tolerance is 25 bp. See **H1**, **M1**. |
| `adjust.py:48` | dividend factor is "1.0 at the last bar" | Not when a known dividend goes ex after the last bar. See **M4**. |
| `schema.py:5` | "``bars`` raw, UNADJUSTED OHLCV exactly as the exchange printed it" | Contradicted eleven lines later at `schema.py:16-19` and by `adjust.py:15-17`. The stored series is split-adjusted. A reader who stops at the module docstring gets the wrong model of the data. |
| `S2_DATA.md:123` | "15 unit tests pass" | `make test` reports **20 passed**. |
| `build.py:8` | G5 gate is "`<= 1 bp`" | It is `median <= 1 bp`. See **C2**. |

`schema.py:5` is the one worth fixing first: it is the first thing a new reader of the
schema sees, and it asserts the exact falsehood that cost this project its most expensive
bug.

---

## L6 — undocumented calling invariants in `adjust.py`: unfiltered bars crash, and any dtype normalisation crashes

**Where:** `rltrader/data/adjust.py:50-59`, `:86`, `:97`.

Two preconditions are required by `adjusted()` / `as_traded()`, enforced by nothing and
documented nowhere.

**(a) Bars must be provider-filtered.** `store.bars(t)` without `provider=` returns one row
per provider per date — duplicate labels in the index `adjust.py` builds from
`b["date"].values`. `close.loc[label]` at `adjust.py:56` then returns a `Series`, and
`float()` raises:

```
adjusted(store.bars("NVDA"), store.actions("NVDA"), as_of=now)
-> TypeError: float() argument must be a string or a real number, not 'Series'
```

`as_traded()` happens to survive (it never indexes by label) and returns correct per-row
values, so the two functions disagree about whether the same input is legal. The gate suite
never hits this because `build.py:135,181` pass `provider="yahoo"`; any S3 consumer that
reasonably calls `store.bars(t)` will.

**(b) Dates must stay `datetime.date` objects.** After the parquet round-trip both
`bars["date"]` and `actions["ex_date"]` are **object** columns of `datetime.date`
(`schema.py:25,39` declare `date32[day][pyarrow]`). The comparison
`mult.index < a["ex_date"]` (`adjust.py:53,59,73`) is correct on that pairing —
`n_true = 2374` of 2941 for ex 2024-06-10. It is not silently wrong on any mismatch; every
mismatched combination raises loudly:

```
bars date -> datetime64, ex_date stays date   TypeError: Invalid comparison between
                                              dtype=datetime64[s] and date
ex_date -> Timestamp, bars date stays date    TypeError: Cannot compare Timestamp with
                                              datetime.date
ex_date -> ISO string                         TypeError: '<' not supported between
                                              'datetime.date' and 'str'
both -> datetime64                            works, factors 1.0 / 10.0 / 40.0 correct
```

So this is fragility, not a correctness hole — worth recording because the failure mode is
counter-intuitive: a caller doing the *tidy* thing,
`bars["date"] = pd.to_datetime(bars["date"])`, before calling `adjusted()` gets a hard
`TypeError`. Object-dtype comparison is also the slow Python path over 2,941 rows per
action.

**Fix:** normalise both sides at the top of `_known()` and the factor functions
(`pd.to_datetime(...).dt.date`, or move both to `datetime64` consistently), and assert
`bars["date"].is_unique` in `adjusted()` and `as_traded()` with a message naming
`provider=`.

---

## L7 — `--check-only` mutates the build artifact it is checking

**Where:** `scripts/build_dataset.py:55` vs `:74-78`.

`--check-only` skips `ingest` but still calls `write_manifest`, so `make check`
unconditionally rewrites `data/manifest.json` — new `generated_at`, recomputed
`payload_digest`, new gate block. Running the read-only check dirties the working tree.
(Observed during this review: `make check` changed `generated_at` from
`2026-09-14T23:49:18Z` to `...23:53:28Z`. That edit is mine, from running `make check` at
the start of the review, and is the only change to `data/` in this session.)

Minor, but it means "did the gates change?" and "did the data change?" cannot be answered
from git state, and a reviewer cannot run `make check` without side effects.

**Fix:** write the manifest to a distinct path under `--check-only`, or add
`--write-manifest/--no-write-manifest`.

---

## Checked and found clean — no bug in these

Stated explicitly rather than padded.

**Dividend values share the price basis of the closes — `dividend_factors` is
dimensionally correct.** Attacked hard, because a mismatch here would be a 40x error on
NVDA and invisible to every gate. It holds: numerator and denominator both carry the same
split adjustment and it cancels to **0.000000 bp** (evidence table in M3). Independent
corroboration from dividend-yield continuity across the split dates — `value / prev stored
close`, in bp:

```
AAPL across the Aug-2020 4:1   23.68  27.00  18.00  17.22     no 4x step
NVDA across the Jul-2021 4:1    3.45   2.29   1.76            no 4x step
NVDA across the Jun-2024 10:1   0.88   0.47   0.82            no 10x step
```

No discontinuity at any split date. The invariant is real; see M3 for why it is
nonetheless fragile.

**Multiple splits on one ticker, and iteration order.** `_known()` sorts by `ex_date`
(`adjust.py:44`), but the sort is not load-bearing. `unadjust_factors` accumulates a
product of independent multiplicative masks (`adjust.py:73`), and multiplication is
commutative. `dividend_factors` is also order-independent, because `prev` is read from the
untouched `close` series (`adjust.py:51,56`) and never from the running `mult`. Verified on
real NVDA data carrying both the 2021-07-20 4:1 and the 2024-06-10 10:1 split:

```
factors: 2015-01-02 = 40.0 | 2021-07-19 = 40.0 | 2021-07-20 = 10.0
         2024-06-07 = 10.0 | 2024-06-10 =  1.0 | 2026-09-14 =  1.0

shuffle test, 5 random permutations of the actions frame:
  unadjust series bitwise identical to the sorted run in 5/5
  dividend_factors max|diff| = 0.000e+00
  post-shuffle probes still 40 / 40 / 10 / 10 / 1 / 1
```

The sort buys determinism and readability only.

**The strict-inequality boundary at the ex-date.** `mult.index < a["ex_date"]` is correct
on both paths. The ex-date bar must *not* receive the factor, and it does not:

```
NVDA stored close  2024-06-06  120.998  factor 10.0  traded 1209.98
                   2024-06-07  120.888  factor 10.0  traded 1208.88  <- tape ground truth,
                                                                        0 bp vs PRINTED_CLOSES
                   2024-06-10  121.790  factor  1.0  traded  121.79  <- ex-date, already post-split
                   2024-06-11  120.910  factor  1.0  traded  120.91

dividend_factors: 2024-03-04 = 0.9971099204
                  2024-03-05 = 0.9971567150   (its own ex-date, == the 2024-03-06 value)
```

The ex-date bar is not adjusted for its own dividend — the correct convention. Both unit
tests pin this (`test_adjust.py:41,53`). The gap is that the real-data gate does not (M2),
not that the arithmetic is wrong.

**G5's merge alignment — the specific concern raised.** No bug. `reconcile.compare`
(`reconcile.py:20`) does `left.merge(right, on="date", how="inner")`; `ours` is sorted by
`adjusted()` (`adjust.py:80`), `ref_adj[t]` is sorted by `adjusted_reference`
(`stockanalysis.py:73`), and `build.py:147` re-sorts by date before `pct_change`. Both
series are then indexed positionally over the *same* merged rows, so `r1` and `r2` are
element-wise aligned by construction (`build.py:148-149`). A fan-out from duplicate dates
cannot occur either: `Store.bars` dedups on `(ticker, date, provider)` (`store.py:129`) and
`gate_corporate_actions` passes `provider="yahoo"` (`build.py:135`). The inner join does
silently drop dates present in only one series, and `pct_change` then spans the hole — but
both series span the same hole, so the difference still cancels and no false failure
results. The real consequence of dropped dates is a shrunken, unchecked `n`; that is L2.

**DST handling of the 16:00 ET `first_public_at`.** Correct on both sides of every
transition tested. `_localise` (`store.py:27-30`) produces:

```
2021-03-12 Fri  21:00Z (EST -5)    2021-03-15 Mon  20:00Z (EDT -4)
2021-11-05 Fri  20:00Z (EDT -4)    2021-11-08 Mon  21:00Z (EST -5)
2023-03-10/13   21:00Z / 20:00Z    2023-11-03/06   20:00Z / 21:00Z
2024-11-01/04   20:00Z / 21:00Z
```

All offsets correct, for both 16:00 and 09:30. `ambiguous=True` is dead code here: the US
fall-back ambiguity window is 01:00-01:59 local and the spring gap is 02:00-02:59 local, so
neither 16:00 nor 09:30 ever reaches those branches. It remains a latent hazard only if
`SESSION_CLOSE_LOCAL` or `ACTION_PUBLIC_LOCAL` were ever moved into 01:00-01:59, since
`ambiguous=True` means "always pick DST" rather than "raise".

**Early-close days.** On the day after Thanksgiving and Christmas Eve the exchange closes
at 13:00 ET, and `first_public_at` is still stamped 16:00 ET — 3 hours *late*
(2021-11-26, 2022-11-25, 2023-11-24, 2024-11-29, 2021-12-23, 2024-12-24, 2023-07-03,
2024-07-03: 21:00Z or 20:00Z vs the true 18:00Z or 17:00Z). Later visibility is the
conservative direction and cannot create look-ahead. Worth a line in `schema.py:50-52`,
which currently admits the opposite concern (late tape corrections) but not this one.

**`actions()` dedup sort key.** `store.py:148-149` sorts by
`[ticker, ex_date, kind, recorded_at]` while deduplicating on
`[ticker, ex_date, kind, provider]`. Omitting `provider` from the sort key looks wrong but
is harmless: a sequence sorted ascending by `recorded_at` remains ascending when restricted
to any one provider, so the last row per `(..., provider)` is still that provider's maximum
`recorded_at`. Verified over 300 randomised trials with 3 providers and 6 revisions each:
0 mismatches. The defects in `actions()` are L1 (no provider filter downstream) and the
equal-`recorded_at` tie-break shared with L4.

**`compare()` denominator guard.** `reconcile.py:24` maps a zero denominator to `NaN`
rather than dividing by zero. Correct.

**Independence check.** `reconcile.independence_evidence` (`reconcile.py:45-61`) genuinely
tests what it claims, and the 3.49% bit-identical figure is real evidence rather than an
assumption. This is the strongest single piece of the suite.

---

## Suggested order of work

1. **C3** — one-line change, removes a fail-reads-as-pass. Always append G5.
2. **C2** — gate on `max_bps` / `n_over_1bp`. The data to do it is already computed.
3. **H2** — delete `min(end, ...)` in `build.py:70`.
4. **C1** — decide the `as_of` semantics, then make the code and the docstring agree.
   This one needs a design decision, not just an edit, because `tests/test_store_pit.py:29-35`
   currently asserts the present behaviour.
5. **H1** — select splits by `ex_date <= recorded_at`, and replace
   `test_unadjustment_is_point_in_time`.
6. **H3** — give G4 a negative control; a gate that cannot fail is worse than no gate,
   because it is reported as evidence.
7. **M1-M4, L1-L7** as capacity allows. **M3** and **L1** should be fixed *before* any
   Tiingo or Alpaca key is added, since that is the trigger for both, and each turns a
   documented "no code change" upgrade into a silent 10x-100x price error.

---

## Note on the review itself

The pipeline's *arithmetic* is in better shape than its *gates*. The split and dividend
factor math is correct, order-independent, correct at the boundary, and correct across two
splits on one ticker — I attacked it hard and it held. The dividend/close split-basis
invariant holds today. DST is right. The independence check is honest.

The exposure is that the six green gates certify considerably less than they appear to.
Four of them (G1's trailing window, G3, G4, G5) will not fail on a realistic instance of
the failure they name, and two of them (G5, G7) can quietly reduce their own scope to
nothing. The project's own report already contains the lesson —
"No aggregate statistic would have caught it. The ground-truth anchor did"
(`S2_DATA.md:30`) — and the gate suite has one ground-truth anchor set, `PRINTED_CLOSES`,
with three entries, a 25 bp tolerance, and permission to skip itself.
