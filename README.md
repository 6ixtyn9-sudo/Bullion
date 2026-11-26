# Bullion

Lean, zero-bloat, ESPN-native quantitative paper-ticket engine for the three
non-basketball ESPN sports `Ma_Golide_Satellites` cannot cover:

| Lane | ESPN path | Markets | Publication cutoff (SAST) |
|---|---|---|---|
| `nhl` | `hockey/nhl` | 2-way ML incl. OT/SO, ±1.5 puckline, O/U totals | 16:00, bounded by earliest tip − 120m |
| `nfl` | `football/nfl` | 2-way ML, spread, total | 16:00, bounded by earliest tip − 120m |
| `cfb` | `football/college-football` | 2-way ML, spread, total | 16:00, bounded by earliest tip − 120m |
| `eng.1` `esp.1` `ger.1` `ita.1` `fra.1` `uefa.champions` | `soccer/<slug>` | 3-way 1X2, O/U totals, spreads when quoted | 12:00, bounded by earliest kickoff − 120m |

Everything is paper-only: 1.00u per position, no real money, no claims of edge
before 30 settled observations exist per segment.

## What this repo will not do

`Edge-Factory` failed in specific, avoidable ways. Bullion removes each one:

* **No tipster scrapers.** 100% keyless ESPN (`site.api.espn.com`,
  `sports.core.api.espn.com`): schedules, finals, boxscores, predictor, and the
  named-book quotes embedded in scoreboard/summary `odds`/`pickcenter`.
* **No basketball math on hockey.** NHL is regularised Poisson goals with an
  explicit OT/SO split and the overtime goal carried into final totals; soccer is
  time-decayed Dixon-Coles bivariate Poisson; football is ESPN FPI plus an
  empirical margin/total surface resolved by measured Brier skill.
* **No uncalibrated legs.** A leg needs ≥30 walk-forward settled observations in
  its segment (`league, market, period, source`), a Brier cap, positive calibrated
  EV, a pre-tip named-book price, and a positive prior settled ROI band.
* **No longshot chasing.** Leg prices are hard-bounded to `[1.20, 3.00]`; two-leg
  accas land naturally in the `@3.00–@4.00` band.
* **No 2→5 acca dilution.** Default `limit=3` per lane per day, disjoint legs —
  no `#4/#5` filler.
* **No unfaithful history.** Every receipt freezes a decision-time
  `candidate_pool`; replay rebuilds tickets from that pool via the *same*
  `build_accas` the live runner uses, and `--parity` proves it.
* **No silent assumptions.** See below.

## Anti-hallucination contract

This sandbox cannot reach ESPN. Live claims rest only on
`localdata/_probes/espn_probe_latest.json`, produced by `scripts/probe_espn.py`
on GitHub Actions:

* The only live capture so far is **`LIVE_VERIFIED 2026-11-26T12:00:17Z`**
  (`localdata/_probes/espn_probe_latest.json`, frozen extract in
  `tests/fixtures/espn_probe_2026-11-26T12-00-17Z.json`). Everything else that
  looks like a payload is `SYNTHETIC_FIXTURE`; the test suite blocks sockets
  (`tests/conftest.py`).
* What that capture actually proves is narrow (see below): the ten scoreboard
  and core-index routes resolve, the summary carries `pickcenter`, and the
  season event index is pageable — but it contains **zero pre-match events**, so
  it proves no odds path and no predictor key.
* Extrapolating the proven basketball adapter (`Ma_Golide_Satellites`,
  LIVE_VERIFIED 2026-11-06 for NBA/WNBA/NCAA) is **not** verification. Hockey
  `pointSpread` puckline nesting, soccer three-way `moneyline.draw` /
  `drawOdds` paths, NFL/CFB `/predictor` statistics keys, hockey shot keys, and
  per-league pre-kickoff named-book coverage are **unverified** until a probe
  window contains upcoming fixtures.
* `scripts/daily.py` refuses to freeze a ticket unless a committed live probe
  report shows `status: ok`, a sample of ≥3 pre-match events, and ≥50%
  pre-kickoff named-book coverage for that market. A coverage number without a
  sample is not evidence: absent/thin probe → `research_only` (forecast and
  grade, never ticket).
* Missing/malformed fields fail closed: a machine-readable `coverage_gap`
  (written under `localdata/_coverage/` and `localdata/_runs/`) or a raised
  `EspnIdentityError`. No imputed `0.5`, no invented odds, no home/away swap.

## Live probe evidence

Two captures exist. The first was a window bug; the second is the one that
matters.

### `LIVE_VERIFIED 2026-11-26T12:13:48Z` — the priced window

Probed SAST days 2026-11-26/27/28 (ET 2026-11-25..27), summarised from
`localdata/_probes/espn_probe_latest.json` and frozen as a fixture in
`tests/fixtures/espn_live_2026-11-26T12-13-48Z.json`.

| lane | events (pre) | books | ML / spread / total coverage | draw coverage as captured | backfill |
|---|---|---|---|---|---|
| cfb | 67 (67) | Draft Kings | 100% / 100% / 100% (8/8 sampled) | n/a (2-way) | 950 |
| nfl | 5 (4) | Draft Kings | 100% / 100% / 100% (4/4) | n/a (2-way) | 320 |
| eng.1 | 4 (4) | DraftKings | 100% / 100% / 100% | 0% (label-counted — see below) | 374 |
| esp.1 | 5 (5) | DraftKings | 100% / 100% / 100% | 0% (label-counted) | 381 |
| fra.1 | 6 (6) | DraftKings | 100% / 100% / 100% | 0% (label-counted) | 288 |
| ita.1 | 4 (4) | DraftKings | 100% / 100% / 100% | 0% (label-counted) | 374 |
| ger.1 | 6 (6) | Bet 365, DraftKings | 83% / 83% / 83% (5/6) | 0% (label-counted) | 297 |
| nhl | 42 (28) | — | 0% (no lines posted) | n/a | 1,478 |
| uefa.champions | 9 (0) | — | unmeasured (between matchdays) | n/a | 189 |

Four defects the capture proved, now fixed and tested:

1. **Draw prices were being counted, not read.** Every sampled soccer
   `pickcenter[0]` carried `moneyline.draw.close.odds`, `moneyline.draw.open.odds`
   *and* `drawOdds.moneyLine`, but coverage said `moneyline_draw = 0.00` because
   the probe counted market *labels* instead of priced *sides*. Coverage is now
   computed with `quote_market_presence`, and the flat `drawOdds.moneyLine`
   variant is an extraction fallback. UCL stays unmeasured (no pre-match
   fixtures until matchday 6).
2. **NFL FPI failed on tie mass only.** NFL payloads sum to 100 only when
   `teamChanceTie` is included (`99.6480` without it). The parser now accepts the
   tie, prices the two-way side conditioned on no tie (`prob_home`), and reports
   `prob_tie` / `tie_rule: "push_conditional"` rather than hiding the mass. CFB
   has no tie key and is unchanged (`3/3` samples parsed).
3. **Book labels differ per sport.** `"Draft Kings"` (NFL/CFB) vs `"DraftKings"`
   (soccer) vs `"Bet 365"` (Bundesliga) are one book; `canonical_book` normalises
   them, and the probe keeps the raw labels for audit.
4. **`/predictor` does not exist outside football.** NHL and every soccer lane
   return `HTTP 400`; `League.predictor` now gates the request so the probe stops
   inventing coverage gaps for a route ESPN does not serve.

Still unverified, and therefore still not claimed:

* NHL prices — 0 offers on this window (US Thanksgiving); re-probe on the next
  slate. The summary does expose `goalies`, `injuries` and `pickcenter`, and
  `pickcenter` was present but empty (`0` entries) for all 8 sampled fixtures.
* the ger.1 fixture whose `pickcenter` was empty and whose `odds[0]` used the
  flat `homeTeamOdds.odds.value` shape — the probe records
  `odds.shape_unrecognised` with the observed keys instead of guessing.
* `teamExpectedPts`/`oppExpectedPts` were never live-confirmed, so FPI expected
  scores and FPI totals stay `None`; the FPI margin falls back to inverting the
  win probability. `teamPredPtDiff` is now captured and surfaced as
  `predicted_margin` only when the two sides reconcile.
* whether a book pushes or loses a two-way moneyline on a tie. The model prices
  push-conditional because the live NFL quote carries only `home`/`away` sides,
  and the assumption is recorded in `tie_rule` rather than buried in code.

### `LIVE_VERIFIED 2026-11-26T12:00:17Z` — the window bug

The first run asked ESPN for ET `D-2/D-1` instead of `D-1/D`, so every probed
event was already final: 0 pre-match events, empty scoreboard odds, no predictor
attempt. `et_days_for_sast_day` is fixed and pinned by
`tests/test_live_probe.py`. Core season-index refs also carry a
`?lang=en&region=us` suffix that defeated the old `isdigit()` event-id parse;
`event_id_from_ref` strips it.

### Re-probe before enabling a lane

```
# Actions -> probe-espn -> Run workflow, or:
python3 scripts/probe_espn.py --date 2026-11-27 --days-ahead 2 --season 2026
```

Read `coverage.named_book_pre_tip.<market>`: `status` must be `ok` (sample ≥3
pre-match events) before any percentage means anything, and `markets_priced`
inside `summary.field_presence[]` shows which sides each quote actually carried.

## Layout

```text
src/bullion/
  models.py        sport-aware dataclasses, league registry, UTC/SAST helpers
  store.py         stdlib sqlite3 warehouse (games, quotes, predictor, forecasts)
  espn.py          keyless client + fail-closed parsers + probe-permission contract
  calibration.py   walk-forward Platt scaling, rolling Brier, segment gates
  engines/         nhl.py (Poisson + OT/SO), football.py (FPI vs empirical),
                   soccer.py (Dixon-Coles)
  tickets.py       candidate pool, gates, build_accas, immutable receipts, settlement
  replay.py        parity, variants, kelly/daily_pct sizing, paired A/B bootstrap
  runner.py        per-lane ingest→forecast→freeze→grade orchestration
scripts/
  probe_espn.py    live ESPN probe → localdata/_probes/espn_probe_latest.json
  backfill.py      historical finals (ratings warm-up only; never historical prices)
  daily.py         end-to-end daily runner
  replay_harness.py CLI over bullion.replay
docs/workflows/    copy-paste GitHub Actions YAML (Arena cannot push workflows)
```

## Commands

```bash
python3 -m pip install -e ".[dev]"     # pytest + ruff (runtime deps: none)
ruff check .
pytest -q

# 1) capture real ESPN evidence (needs network; run in Actions)
python3 scripts/probe_espn.py --out localdata/_probes/espn_probe_latest.json

# 2) warm the ratings (finals only; newest season first, resumable)
python3 scripts/backfill.py --league all --seasons 2024,2025,2026 --max-events 400
python3 scripts/backfill.py --league all --workers 6 --max-seconds 1800  # bounded cold run

# 3) daily lane runs (freeze only at/after cutoff, only with a probe report)
python3 scripts/daily.py --league all --date 2026-11-28

# 4) replay
python3 scripts/replay_harness.py --parity
python3 scripts/replay_harness.py --mode cross-league --legs 2 --limit 3
python3 scripts/replay_harness.py --stake-mode kelly --kelly 0.25 --slots 3
python3 scripts/replay_harness.py --variant legs=3 --against legs=2 --ab
```

`python3 scripts/replay_harness.py` on a clean checkout prints
`zero settled bet-days: no profit, edge, or readiness claim is supported`.

The backfill is budgeted and resumable: seasons are processed newest-first (so a
budget stop never costs the current season), summary fetches run on a bounded
pool while every SQLite write stays on the main thread, and `--max-seconds`
(default 1800 = 30 min, deliberately under the workflow's `timeout-minutes`)
stops cleanly with exit code 0, leaving the manifest and stored
finals behind for the next run to continue from. The manifest records
`elapsed_seconds`, `processed_per_second` and `projected_remaining_seconds`, so
the next budget decision is based on the observed rate rather than a guess.
Transient CDN failures —
including chunked-transfer `IncompleteRead`, which is a `HTTPException` and not
an `OSError` — are retried with gzip-encoded requests before becoming an
`EspnFetchError`.

## Receipt contract (`Ma_Golide_Satellites`-compatible)

`localdata/<lane>/auto_tickets_YYYY-MM-DD_decision.json`:

* `schema_version: "magolide.paper_ticket_receipt"`, `status: "frozen"`,
  `paper_only: true`, `stake_units_per_acca/position: 1.0`;
* `accas`, `singles`, `candidate_pool`, `selection_policy`, `ticket_decision`;
* every leg carries `forecast_id`, `pick_id`, `league`, `game_id`,
  `scheduled_utc`, `home`, `away`, `market`, `period: "full_game"`,
  `selection`, `line`, `raw_probability`, `calibrated_probability`, `book{...}`
  (with `selected_decimal_price` + `side_prices`), `calibration`, `eligibility`.

`auto_tickets_performance.json` (`magolide.paper_ticket_performance`) carries
`history[*].candidate_pool_outcomes` and `history[*].positions[*].legs[*].outcome`
so `Ma_Golide_Satellites/scripts/replay_harness.py` can load a Bullion
`localdata/` tree unchanged. This was verified directly against the reference
checkout: `forecasting.replay.load_localdata_evidence(/path/to/bullion/localdata)`
returned the day, its 4-row candidate pool, and all settled outcomes, and
`_performance_outcomes_by_receipt` read them via the documented fields.

Semantic notes for cross-sport readers: soccer 1X2 is
`market="moneyline"` with `selection` in `{home, draw, away}`; hockey pucklines
and soccer handicaps are `market="spread"` with a **home-oriented** `line`;
integer lines carry a real `push_probability`.

## Immutability and sizing

* Receipts and `auto_tickets_<day>_no_bet.json` locks are write-once. Later scans
  grade into `auto_tickets_performance.json` and refresh `cards/*.txt`; they never
  rotate a frozen selection or price.
* Live paper tickets are flat 1.00u. Aggressive sizing (`--stake-mode daily_pct`,
  `kelly`, `skim`, `--daily-pct 0.25`) exists only in the replay harness and is
  flagged `[LOW SAMPLE: n < 30]` until 30 clean settled bet-days accumulate.

## Status

* Offline unit tests cover: side resolution from `homeAway`, odds nesting and
  coverage gaps, `$ref` predictor orientation and contradiction handling,
  post-tip rejection, summary/OT period parsing, engine probability identities,
  Dixon-Coles rho behaviour, calibration leakage guards, gate rejection paths,
  receipt schema/immutability, settlement, 100% replay parity, paired A/B NO-OP
  refusal, cross-sport acca assembly, and zero-settled-day handling.
* `scripts/replay_harness.py --parity` rebuilds each frozen receipt and diffs
  ticket id, combined price, and every leg price, not just the selected legs; it
  exits 1 on any mismatch, and reports `vacuous: true` when there is no receipt
  to check (an empty vault is not a parity proof).
* Summary `pickcenter` is now a quote source: when a scoreboard `odds` array is
  empty, one summary fetch per fixture supplies the freeze, and the discovery
  gap stays in the coverage file so the empty scoreboard array remains visible.
* **Verified live (2026-11-26T12:13:48Z):** pre-match named-book coverage for
  CFB (100%), NFL (100%) and five soccer lanes (100%/100%/100%/100%/83%), the
  soccer draw paths (`moneyline.draw.close|open.odds`, `drawOdds.moneyLine`), the
  NFL `teamChanceTie` sum, CFB FPI parsing, and the absence of a `/predictor`
  route for NHL and soccer (HTTP 400).
* **Still unverified:** NHL prices (no lines posted on the captured day), UCL
  pre-match coverage (between matchdays), hockey shot keys, FPI expected scores,
  and whether a two-way moneyline pushes or loses on a tie. Lanes open tickets
  only where the committed probe shows `status: ok` on ≥3 pre-match events and
  ≥50% coverage: NHL and UCL remain `research_only`.
