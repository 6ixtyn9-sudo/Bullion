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

## Live probe evidence: 2026-11-26T12:00:17Z

Nine lanes probed across the ET days the *old* window derived
(`dates=20261124`, `dates=20261125`) for SAST day 2026-11-26.

| lane | events | pre-match | offers | books | named-book pre-tip coverage | predictor |
|---|---|---|---|---|---|---|
| nhl | 15 | 0 | 0 | 0 | unmeasured (0 pre events) | not attempted |
| nfl | 1 | 0 | 0 | 0 | unmeasured | not attempted |
| cfb | 2 | 0 | 0 | 0 | unmeasured | not attempted |
| eng.1 | 0 | 0 | 0 | 0 | unmeasured (no fixtures) | not attempted |
| esp.1 | 0 | 0 | 0 | 0 | unmeasured (no fixtures) | not attempted |
| ger.1 | 0 | 0 | 0 | 0 | unmeasured (no fixtures) | not attempted |
| ita.1 | 0 | 0 | 0 | 0 | unmeasured (no fixtures) | not attempted |
| fra.1 | 0 | 0 | 0 | 0 | unmeasured (no fixtures) | not attempted |
| uefa.champions | 18 | 0 | 0 | 0 | unmeasured | not attempted |

Verdict: **all nine lanes stay `research_only`.** Coverage was measured over
zero pre-match events, which is an absent measurement, not a 0% result.

Three real defects the capture exposed, now fixed and regression-tested:

1. **Wrong ET window.** `et_days_for_sast_day` anchored on SAST midnight and
   subtracted a day, asking ESPN for ET `D-2/D-1`; a SAST day occupies ET
   `D-1/D` (SAST `D` = `D-1T22:00Z .. DT21:59Z`). Every probed event was
   therefore already final, which is why no odds or predictor existed.
   `tests/test_live_probe.py` pins the captured URLs against the corrected map.
2. **Backfill refs were unparsed.** Real refs arrive as
   `.../events/401856879?lang=en&region=us`; splitting on `/` and testing
   `isdigit()` dropped the query suffix check entirely, so a backfill would
   have stored zero games. `event_id_from_ref` now strips the suffix.
3. **Summary odds were ignored.** Every sampled summary (NHL, NFL, CFB, UCL)
   carried `pickcenter` (soccer also `odds` and `hasOdds`) while the scoreboard
   `odds` array was empty, so the lane declared games unpriceable when a real
   quote source existed. `parse_pickcenter` reads those containers through the
   proven offer paths and fails closed with `pickcenter.shape_unrecognised`
   (plus the observed key list) for anything else.

Also captured: core season index counts (`nhl` 1478/2 pages, `cfb` 950,
`eng.1` 374, `esp.1` 381, `ita.1` 374, `ger.1` 297, `fra.1` 288,
`uefa.champions` 189, `nfl` 320) — the backfill route works.

### Re-probe before enabling any lane

The next probe must land on a window that actually contains upcoming fixtures,
and it records the ET dates it asked for:

```
# Actions -> probe-espn -> Run workflow, or:
python3 scripts/probe_espn.py --date 2026-12-01 --days-ahead 2 --season 2026
```

Read `coverage.named_book_pre_tip.<market>`: `status` must be `ok` (sample ≥3
pre-match events) before any percentage means anything. `status:
"no_pre_match_events"` means the window was wrong or empty — re-probe, do not
interpret it as zero coverage.

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

# 2) warm the ratings (finals only)
python3 scripts/backfill.py --league all --seasons 2024,2025,2026 --max-events 400

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
* **Verified live (2026-11-26T12:00:17Z):** route resolution for all nine lanes,
  the summary `pickcenter`/`odds`/`hasOdds` containers, `has_predictor: false` on
  every sampled summary, core season-index counts, and `$ref` query suffixes.
* **Still unverified:** every odds *path*, every predictor *key*, per-league
  pre-kickoff coverage, hockey shot keys, and soccer draw pricing. All nine lanes
  are `research_only`; `bullion_daily.yml` will forecast and grade but cannot
  ticket until a pre-match probe window is captured and inspected.
