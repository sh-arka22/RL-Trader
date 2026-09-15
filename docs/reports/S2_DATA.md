# S2 — Data Pipeline: build report

Built 2026-09-14. Universe NVDA, TSLA, AAPL, META, XOM. Window 2015-01-02 → 2026-09-14.
**2,941 sessions per ticker per provider, 29,410 bars, 0 missing sessions.**
Payload digest `ed726afc125aeca4`, 130 Parquet files, 2255 KB.

Reproduce: `make data`. Gate-only: `make check`. Everything below is machine-generated
into `data/manifest.json`; nothing here is asserted by hand.

## 1. The finding that mattered

**Both free providers return split-adjusted prices while claiming to return raw ones,
and the standard way of asking for unadjusted data does not work.**

yfinance with `auto_adjust=False` withholds only the *dividend* adjustment. Its `Close`
column is still split-adjusted. stockanalysis.com behaves identically. Verified against
the tape:

| Ticker | Session | Stored "raw" close | Actually printed | Split |
|---|---|---|---|---|
| NVDA | 2024-06-07 | 120.888 | **1208.88** | 10:1 ex 2024-06-10 |
| TSLA | 2022-08-24 | 297.097 | **891.29** | 3:1 ex 2022-08-25 |
| AAPL | 2020-08-28 | 124.808 | **499.23** | 4:1 ex 2020-08-31 |

Three consequences, all now enforced in code:

1. **Do not re-apply splits.** The first version of `adjust.py` did, and double-counted
   them. The symptom was a ~90,671 bp return error on *exactly the split dates and
   nowhere else* — five bad rows hidden in 14,705 good ones, with a median error of
   0.0004 bp. No aggregate statistic would have caught it. The ground-truth anchor did.
2. **The stored series is not immutable.** The next split silently rewrites every past
   price. This is why every row carries `price_basis` and `recorded_at`.
3. **Point-in-time prices require *un*-adjustment**, not adjustment. `as_traded()`
   multiplies by every split ratio with `ex_date > t`. It recovers the three tape prints
   above to **0.00 bp**. Use it for share counts, round lots and any claim about price
   levels; use the stored series for returns.

A backtest that sizes positions off the stored NVDA price in 2024 buys ten times too
many shares.

## 2. Two time axes, not one

The first store collapsed them and a test caught it:

| Axis | Field | Question it answers | Used by |
|---|---|---|---|
| Valid time (`as_of`) | `first_public_at` | What did the *market* know at T? | the backtest |
| Transaction time (`recorded_before`) | `recorded_at` | What did *our store* contain at T? | the audit |

A bar public on 2024-01-02 but fetched on 2024-01-05 is visible to a backtest standing on
2024-01-03 and invisible to an audit reconstructing the store as of 2024-01-04. Filtering
both on one timestamp answers neither question. A daily bar becomes public at 16:00
America/New_York. Corporate actions are dated at the **ex-date open**, not the
announcement — free tiers do not carry announcement dates, so the system learns of an
action *later* than a real trader would. That error is in the conservative direction.

## 3. Providers

No API keys exist in this environment, so the planned Tiingo/Alpaca pair was unavailable
(Tiingo without a token returns HTTP 403 `{"detail":"Please supply a token"}`). Both
adapters are implemented and register themselves the moment a key appears.

The second source is **stockanalysis.com**, found by live probing on 2026-09-14:
`GET /api/symbol/s/<ticker>/history?range=Max&period=Daily`, keyless, no registration.
Verified depth: XOM to 1968-01-03, AAPL to 1982-10-05, NVDA to 1999-01-25, TSLA to
2010-06-30, META to 2012-05-21. Undocumented, and it rate-limits at roughly ten rapid
requests by returning HTML instead of JSON — hence one cached fetch per ticker per
process plus exponential backoff.

Rejected while probing: Stooq (still dead, returns a `noindex` stub), WSJ download
(HTTP 401), Nasdaq Data Link (403), FMP/Twelve Data/Alpha Vantage (key or demo-symbol
only), Yahoo's direct chart endpoint (HTTP 429 from this IP).

## 4. G3 — cross-provider reconciliation

Gate: median absolute close divergence ≤ 1 bp.

| Ticker | Bars | Median bp | p95 bp | Max bp | Rows >1bp | Bit-identical | Worst date |
|---|---|---|---|---|---|---|---|
| NVDA | 2941 | 0.00035 | 1.4033 | 4.880 | 276 | 1.77% | 2016-05-13 |
| TSLA | 2941 | 0.00037 | 0.2365 | 0.483 | 0 | 2.01% | 2015-02-23 |
| AAPL | 2941 | 0.00027 | 0.1764 | 7.259 | 1 | 4.32% | 2020-07-09 |
| META | 2941 | 0.00022 | 0.0005 | 17.513 | 1 | 4.90% | 2016-07-05 |
| XOM | 2941 | 0.00020 | 0.0004 | 0.001 | 0 | 4.45% | 2021-11-04 |

**Independence: INDEPENDENT_LIKELY.** Only
3.49% of bars are
bit-identical. This check exists because a reconciliation between a source and its own
mirror proves nothing, and two sources that agree to the last decimal on every bar are
one source. `reconcile.independence_evidence()` fails loudly at >99.9% identical.

NVDA carries 276 rows above 1 bp and a 4.88 bp worst case (2016-05-13); NVDA traded near
$40 then, so a half-cent of rounding is ~1 bp. The 17.5 bp META outlier is 2016-07-05.

## 5. G5 — recomputed total return vs the vendor's own adjusted close

Compared as *returns*, not levels, because vendors normalise differently.

| Ticker | Median bp | p95 bp | Max bp | Rows >1bp |
|---|---|---|---|---|
| NVDA | 0.00042 | 0.0076 | 2.706 | 48 |
| TSLA | 0.00027 | 0.0009 | 0.258 | 0 |
| AAPL | 0.00025 | 0.0007 | 7.342 | 2 |
| META | 0.00026 | 0.0008 | 17.928 | 2 |
| XOM | 0.00025 | 0.0007 | 0.001 | 0 |

Residual >1bp rows cluster on dividend ex-dates, where our simple
`1 - div/prev_close` model differs slightly from the vendor's. TSLA, which pays no
dividend, has **zero**.

## 6. Gate results

| Gate | Result | Detail |
|---|---|---|
| G1_coverage | PASS | 0 missing sessions for every ticker/provider |
| G2_no_phantom_sessions | PASS | no bars on non-session days |
| G3_reconcile | PASS | worst median divergence 0.0004 bp (tol 1.0); independence: INDEPENDENT_LIKELY — only 3.4886% of bars are bit-identical — sources round/clean different |
| G4_point_in_time | PASS | no future bar visible at any probe instant |
| G5_corporate_actions | PASS | worst median split-adjusted return divergence 0.0004 bp (tol 1.0) |
| G7_printed_prices | PASS | un-adjusted closes match the tape on 3 split anchors |
| G6_reproducible | PASS | two ingests → identical payload digest `3381bfc0f11949f4` |

15 unit tests pass. `tests/test_adjust.py::test_splits_are_not_reapplied` is the
regression guard for §1.

## 7. Honest limitations

- **No truly raw price source.** `adjusted(..., price_basis="raw")` raises
  `NotImplementedError` rather than silently doing the wrong thing. Fix = one free Tiingo
  or Alpaca key.
- **Two sources, not three.** Both are free and keyless; neither is a primary exchange feed.
- **Announcement dates are absent**, so action timing is conservative by design.
- **Survivorship bias is not addressed here.** The universe was chosen in the research
  phase and all five names survived. S3 must evaluate on universes that include delisted
  names — `RESEARCH.md` flags this as a headline risk.
- **`first_public_at` = 16:00 ET ignores late consolidated-tape corrections.** Marginally
  optimistic; the environment's 1-bar execution delay absorbs it.

## 8. Adversarial review and what it changed

An independent reviewer (`s2-reviewer`) attacked this pipeline and filed 14 findings in
`docs/reports/S2_REVIEW.md`: 3 critical, 3 high, 4 medium, 4 low. Its verdict is worth
quoting, because it is the real lesson of S2:

> "The arithmetic is in better shape than the gates. I attacked the split/dividend math
> hard and it held. The exposure is that six green gates certify much less than they
> appear to — four will not fail on a realistic instance of the failure they name, and
> two can quietly reduce their own scope to nothing."

That is the same failure mode as §1 at one level up. In §1 a real bug hid behind a
healthy median. Here, healthy-looking gates hid the fact that they could not fail.

### Fixed

| ID | Defect | Evidence | Fix |
|---|---|---|---|
| C1 | `bars(as_of=…)` returned the latest vintage. NVDA at `as_of=2016-01-01` gave a 2015-01-02 close of 0.50325 when the tape printed 20.13 — 40x, from two splits that had not happened. Returns survive a uniform rescale, so every returns-based gate stayed green. | live store | New `Store.tape()` returns tape-correct levels (now 20.13). `bars()` docstring states plainly that it is correct for returns and wrong for levels. |
| C2 | G3/G5 scored on `max(median_bps)` only. One 10x-corrupted bar → `max_bps` 93,138 and **PASS**. The gate built to catch the §1 bug was scored on the exact statistic that hid it. | mutation | `_score()` judges median **and** p95 **and** max **and** the fraction above 1 bp. |
| C3 | The G5 reference fetch sat behind a bare `except`; on failure `ref_adj` stayed `{}` and `if ref_adj:` skipped the gate — "all gates passed", exit 0. The endpoint is *documented* to rate-limit. | reproduced | G5 is appended unconditionally and fails on a missing reference. |
| H1 | `as_traded()` gated un-adjustment on `as_of`, returning split-adjusted prices for a point-in-time query: NVDA 2024-05-31 → 109.63 vs 1096.33 printed. It reproduced the very 10x error §1 warns about. | reproduced | Un-adjustment follows the data vintage, not the backtest clock. Now 1096.33. |
| H2 | G1 truncated its own window to `min(end, max(date))`. Truncating every series at 2020-01-02 still reported "0 missing sessions" while 1,684 sessions per series were gone. | mutation | Window runs to `calendar.last_complete_session()`; the expected count is reported. |
| M1 | G7 skipped absent tickers. Deleting NVDA → **PASS** "on 2 split anchors"; the 10:1 anchor simply vanished. | mutation | A missing anchor ticker fails the gate. |
| M2 | No anchor sat on or after an ex-date, so flipping `<` to `<=` in `unadjust_factors` left G7 green. | analysis | Three post-split anchors added (NVDA 121.79, TSLA 296.07, AAPL 129.04). |
| M3/L1 | `adjusted()` checked a keyword, not `bars["price_basis"]`; `store.actions()` had no provider filter, so two providers reporting one 10:1 split would give factor 100 (+90,000 bp) — triggered precisely by adding the Tiingo/Alpaca key the code advertises. | analysis | Basis is read from the data; actions are de-duplicated on `(ex_date, kind)`; `provider=` filter added. |
| M4 | `dividend_factors` broke its own "1.0 at the last bar" contract when an action post-dated the final bar (−68.4 bp), invisible to G5 because a uniform rescale cancels in `pct_change`. | reproduced | Normalised to exactly 1.0 at the last bar. |
| L4 | Ties on equal `recorded_at` were broken by glob order over a UUID filename — a coin flip (102/98 over 200 stores). `--repro` pins `recorded_at`, manufacturing exactly those ties. | reproduced | Sort key includes `ingest_id`. |
| G4 | **The gate was a tautology.** `write_bars` stamps `first_public_at = date + 16:00 ET`, and the gate asked whether bars dated ≥ D are hidden at D 09:29 — testing the filter that produced the data. A store whose every bar held the *next* session's OHLC passed. It also probed only 9 of 2,942 sessions (0.3%) and never touched the actions table. | proved | Rewritten as a stamp audit over **all 29,410 bars** plus every action stamp. |

### Verified by mutation, not by assertion

Every gate is now run against a planted instance of the defect it names:

| Planted defect | Gate | Result |
|---|---|---|
| truncate all series at 2020-01-02 | G1 | fails |
| delete 5 interior sessions | G1 | fails |
| one NVDA bar of provider 2 ×10 | G3 | fails |
| provider 2 = exact mirror of provider 1 | G3 | fails |
| G5 reference rate-limited to `{}` | G5 | fails |
| NVDA removed entirely | G7 | fails |
| one bar stamped 8 h early | G4 | fails |

### Accepted, not fixed

- **A2 / review answer 4** — the G5 inner join has no minimum sample size; a 2-bar overlap
  counts as usable. Deferred to S3, where fold sizes are defined.
- **`ambiguous=True` is dead code** in the 16:00/09:30 stamping (the DST ambiguity window
  is 01:00–01:59). Harmless; left in place. DST was checked across the 2021, 2023 and 2024
  transitions and is correct.
- **Early closes (13:00 ET) are stamped 16:00** — three hours late, conservative, cannot leak.

### Confirmed clean under attack

The split/dividend arithmetic. Multiple splits compose correctly (NVDA 40 / 40 / 10 / 10 / 1
across the 2021 4:1 and 2024 10:1); five random permutations of the actions frame give
bitwise-identical output; the strict `<` at the ex-date is right on both paths; vendor
dividend values arrive in the same split-adjusted basis as the closes, so `1 - div/prev_close`
cancels to 0.000000 bp. DST handling, the G5 merge alignment, and `drop_duplicates` per axis
were each attacked and held.
