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

This sandbox cannot reach ESPN. Nothing in this repo is a captured live ESPN
response, and no parser is labelled live-verified:

* Every committed payload-like fixture is `SYNTHETIC_FIXTURE` and the test suite
  blocks sockets (`tests/conftest.py`).
* URL shapes, `homeAway` side resolution, odds nesting, American→decimal
  conversion, and the predictor `$ref` orientation rule are ported from the
  proven basketball adapter in `Ma_Golide_Satellites` (LIVE_VERIFIED 2026-11-06
  for NBA/WNBA/NCAA). Extrapolating a proven pattern is **not** verification.
* Hockey `pointSpread` puckline nesting, soccer three-way `moneyline.draw` /
  `drawOdds` paths, NFL/CFB `/predictor` statistics keys, and per-league
  pre-kickoff named-book coverage are **unverified** until
  `scripts/probe_espn.py` has run and its JSON has been inspected.
* `scripts/daily.py` refuses to freeze a ticket for a lane unless a committed
  live probe report proves ≥50% pre-kickoff named-book coverage for that market.
  Absent probe → `research_only`: forecast and grade, never ticket.
* Missing/malformed fields fail closed: a machine-readable `coverage_gap`
  (written under `localdata/_coverage/` and `localdata/_runs/`) or a raised
  `EspnIdentityError`. No imputed `0.5`, no invented odds, no home/away swap.

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
* **Unverified until `probe_espn.yml` runs on GitHub Actions:** every live ESPN
  field, path, and coverage percentage. Do not enable `bullion_daily.yml` before
  the probe report has been inspected.
