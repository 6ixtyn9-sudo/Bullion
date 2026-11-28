# Bullion

Lean, zero-bloat, ESPN-native quantitative paper-ticket engine covering **13
lanes across four sports**: Hockey, American Football, Top-6 Soccer, and US
Basketball.

| Sport | Lane | ESPN path | Markets | Publication cutoff (SAST) |
|---|---|---|---|---|
| Hockey | `nhl` | `hockey/nhl` | 2-way ML incl. OT/SO, ±1.5 puckline, O/U totals | 16:00, bounded by earliest tip − 120m |
| Football | `nfl` | `football/nfl` | 2-way ML, spread, total | 16:00, bounded by earliest tip − 120m |
| Football | `cfb` | `football/college-football` | 2-way ML, spread, total | 16:00, bounded by earliest tip − 120m |
| Soccer | `eng.1` `esp.1` `ger.1` `ita.1` `fra.1` `uefa.champions` | `soccer/<slug>` | 3-way 1X2, O/U totals, spreads when quoted | 12:00, bounded by earliest kickoff − 120m |
| Basketball | `nba` | `basketball/nba` | 2-way ML, spread, total | 20:00, bounded by earliest tip − 120m |
| Basketball | `ncaa_m` | `basketball/mens-college-basketball` (`groups=50`) | 2-way ML, spread, total | 20:00, bounded by earliest tip − 120m |
| Basketball | `ncaa_w` | `basketball/womens-college-basketball` (`groups=50`) | 2-way ML, spread, total | 20:00, bounded by earliest tip − 120m |
| Basketball | `wnba` | `basketball/wnba` | 2-way ML, spread, total | 20:00, bounded by earliest tip − 120m |

The basketball cutoff has two jobs, because a US slate spans two SAST days.
Overnight ET evening tips (19:00–22:00 EST = `00:00–05:00 SAST` on the next SAST
day) are bounded by `earliest tip − 120m`, which lands them at **`20:00Z–23:00Z`
(`22:00–01:00 SAST`)** — the freeze happens before midnight, on the SAST day the
fixture is filed under. The `20:00 SAST` cap then bounds ET afternoon matinees
(12:00–16:30 EST), which do fall on the same SAST day and must not wait for the
evening window.

Everything is paper-only: 1.00u per position, no real money, no claims of edge
before 30 settled observations exist per segment.

## What this repo will not do

`Edge-Factory` failed in specific, avoidable ways. Bullion removes each one:

* **No tipster scrapers.** 100% keyless ESPN (`site.api.espn.com`,
  `sports.core.api.espn.com`): schedules, finals, boxscores, predictor, and the
  named-book quotes embedded in scoreboard/summary `odds`/`pickcenter`.
* **No basketball math on hockey.** Each sport gets its own likelihood: NHL is
  regularised Poisson goals with an explicit OT/SO split and the overtime goal
  carried into final totals; soccer is time-decayed Dixon-Coles bivariate
  Poisson. Football and basketball each run a **paired surface**: an empirical
  margin/total fit (`empirical_score`) beside ESPN's own provider numbers
  (`espn_fpi` / `espn_bpi`), with the lane pricing off whichever surface wins its
  walk-forward moneyline Brier comparison and a deterministic tiebreak when the
  two are inside the noise band.
* **No uncalibrated legs.** A leg needs ≥30 walk-forward settled observations in
  its segment (`league, market, period, source`), a Brier cap, positive calibrated
  EV, a pre-tip named-book price, and a positive prior settled ROI band.
* **No longshot chasing.** Leg prices are hard-bounded to `[1.20, 3.00]`, and a
  ticket must combine into `[@3.30, @5.00]`: short enough to be a bet, long
  enough to be worth freezing.
* **No 2→5 acca dilution.** Two- and three-leg accas compete in one pool and the
  best three win (`limit=3` per lane per day, disjoint legs) — no `#4/#5` filler.
* **Accas are ranked by exact log growth.** `kelly_log_growth` scores the whole
  ticket by the growth of its full-Kelly stake, `p·ln(pO) + (1−p)·ln(O(1−p)/(O−1))`,
  so at equal edge a shorter ticket outranks a longer one instead of being
  punished for a low price. `probability_price_blend` remains available, and is
  what replay uses for receipts frozen before this policy.
* **Two-tier edge tripwire.** A claimed `p·O − 1` above 15% is tagged
  `[EDGE>15% AUDIT]` on the card and counted per acca; above 40% the leg is
  refused outright (`edge_above_sanity_cap`). That magnitude is a broken input,
  not a signal — and the cap never retro-filters a receipt frozen without it.
* **No unfaithful history.** Every receipt freezes a decision-time
  `candidate_pool`; replay rebuilds tickets from that pool via the *same*
  `build_accas` the live runner uses, and `--parity` proves it.
* **No silent assumptions.** See below.

## US basketball: four lanes, one honest surface choice

`nba`, `ncaa_m`, `ncaa_w` and `wnba` are registered as live lanes, not a
research side-project. They reuse the football lane's shape — two surfaces per
fixture, resolved per lane by measured walk-forward Brier skill — with the
basketball-specific pieces the sport actually needs:

* **`espn_bpi` vs `empirical_score`.** The provider surface reads ESPN's own win
  probability and projected scores from `/predictor`; it emits nothing when the
  route is absent, and no total at all when the payload carries no expected
  points (a margin can be priced from a projection, a total cannot). The
  empirical surface fits per-team scoring rates from settled finals with
  per-lane shrinkage (`nba` 4.0, college 6.0 pseudo-games), per-lane time decay
  (90 days pro, 60 college) and a schedule-fatigue multiplier that counts only
  **strictly past** days (back-to-back ×0.975, third-in-four ×0.99). Whichever
  surface wins the moneyline Brier comparison prices the whole lane; a tie is
  broken deterministically and the receipt says so.
* **Anchor once, at the mean, college only.** An NCAA spread/total mean is
  `0.20 × model + 0.80 × (−main_line)`; the NBA and WNBA are not anchored at all,
  and the moneyline is never anchored. The anchor is applied once, then every
  main and alternate line is scored against that same mean, so a ladder stays
  monotonic instead of every line becoming its own model.
* **One main line per `(game, market, book)`.** The provider's own
  `main`/`mainLine` flag wins when present; otherwise a deterministic rule picks
  the closest handicap to zero and the middle total. Two mains in one group
  would make which line anchors the model depend on payload order.
* **Scoreboard finals, not summary-per-game.** History comes from
  `scoreboard?dates=…&limit=500[&groups=50]`, one request per ET day, and the
  parser returns **only** `post` events. A missing `linescores` block is
  tolerated (many mid-major finals publish none); a non-empty block that
  contradicts the final is dropped and recorded as a coverage gap. Resume state
  lives **inside `bullion.db`** (`scoreboard_backfill_days`, keyed
  `(lane, ET day)`), never in a committed manifest: the warehouse is gitignored
  and cache-restored, so a manifest in Git would outlive the database it
  describes and could declare an empty warehouse already swept. Past ET days are
  skipped; `today_ET` never is.
* **Seeding refits weekly.** `seed_walk_forward_ml` scores historical matchdays
  with a frozen fit per 7-day window for basketball only — a per-day fit across
  a 400-day backfill over ~360 D1 teams is the one cost that does not fit the
  job budget. The window also computes its residual sigmas and its team→days
  index **once** (a per-fixture sigma scan made the pass quadratic), while
  fatigue is still evaluated on each **matchday** against every final strictly
  before it — including finals that settled inside the window. Every other lane
  keeps its per-day cadence, and the seeding summary records
  `refit_cadence_days`, `refits` and the sampled refit days.
* **`wnba` stays `research_only` until it is measured.** In the off-season the
  probe reports `no_pre_match_events`, which is an absent measurement rather
  than a measured zero, so the lane grades and does not ticket. Nothing about
  the WNBA is special-cased: it is the same gate every lane passes.
* **Cross-lane ticketing is still off.** `CROSS_LANE_ENABLED` is not
  implemented; Phase 2 is a separate change.

## Anti-hallucination contract

This sandbox cannot reach ESPN. Every live claim rests on committed probe
captures in `localdata/_probes/espn_probe_latest.json`, written by
`scripts/probe_espn.py` on GitHub Actions:

* **A lane may only ticket on measured evidence.** `scripts/daily.py` asks
  `lane_permission()` for each lane, and that function requires, per market,
  `status: "ok"`, a sample of **≥ 3 pre-match events**, and **≥ 0.50** named-book
  pre-tip coverage. Anything less keeps the lane `research_only` — it forecasts
  and grades, it does not ticket. A coverage number without a sample is not a
  measurement, and `no_pre_match_events` is an absent measurement rather than a
  measured zero.
* **Subset probes merge, never overwrite.** `--leagues basketball` (or any
  subset) folds its lanes into the existing report: lanes probed now win, every
  other lane keeps its last captured block **and its own `captured_utc`**, and
  the file records `probed_lanes` / `carried_over_lanes` /
  `carried_over_from_utc`. A lane measured yesterday is never re-dated by a probe
  that did not look at it (legacy reports without per-lane stamps are dated by
  the previous file timestamp, not by the new run).
* **Missing or malformed fields fail closed.** An ambiguous side, a half-closed
  offer, an unknown odds shape, or a league mismatch produces a machine-readable
  `CoverageGap` (written under `localdata/_coverage/` and `localdata/_runs/`) or
  raises `EspnIdentityError`. No imputed `0.5`, no invented odds, no home/away
  swap — a `×` in a table below means the probe did not measure it, not that it
  was assumed.
* **A payload is not a contract.** A scoreboard route that resolves does not
  imply that a book posts lines on it, that `/predictor` exists, or that a
  summary carries `pickcenter`; each is measured separately, and
  `League.predictor` gates the predictor request so the probe never invents
  coverage gaps for a route ESPN does not serve.

## Live probe evidence

### `LIVE_VERIFIED 2026-11-28T07:12:44Z` — all 13 lanes, `cca6bfb`

Probed SAST days 2026-11-28/29/30 (ET 2026-11-27..29), summarised directly from
`localdata/_probes/espn_probe_latest.json`. Coverage is
`coverage.named_book_pre_tip.<market>`; `events (pre)` counts scoreboard events
and those still pre-tip at capture time.

| lane | events (pre) | books | ML / spread / total | predictor | season index |
|---|---|---|---|---|---|
| `nhl` | 36 (21) | DraftKings | 100% / 100% / 100% (8/8) | not requested (HTTP 400) | 2026: 1,478 |
| `nfl` | 12 (11) | DraftKings | 100% / 100% / 100% (8/8) | 3/3 `prob_home` + `predicted_margin`, tie `push_conditional` | 2026: 320 |
| `cfb` | 66 (53) | DraftKings | 100% / 100% / 100% (8/8) | 3/3 `prob_home` + `predicted_margin` | 2026: 950 |
| `eng.1` | 10 (9) | DraftKings | 100% / 100% / 100% (8/8) | not requested (HTTP 400) | 2026: 374 |
| `esp.1` | 10 (9) | DraftKings | 100% / 100% / 100% (8/8) | not requested (HTTP 400) | 2026: 381 |
| `ger.1` | 9 (8) | Bet365, DraftKings | 87.5% / 87.5% / 87.5% (7/8) | not requested (HTTP 400) | 2026: 297 |
| `ita.1` | 10 (9) | DraftKings | 100% / 100% / 100% (8/8) | not requested (HTTP 400) | 2026: 374 |
| `fra.1` | 9 (8) | DraftKings | 100% / 100% / 100% (8/8) | not requested (HTTP 400) | 2026: 288 |
| `nba` | 33 (22) | DraftKings | 62.5% / 62.5% / 62.5% (5/8) | 3/3 `prob_home`, `predicted_margin`, `expected_home`/`expected_away` | 2027: 725 |
| `ncaa_m` | 196 (141) | DraftKings | 75% (6/8) / 87.5% (7/8) / 87.5% (7/8) | 3/3 `prob_home` only | 2027: 390 |
| `ncaa_w` | 251 (173) | — | 0% / 0% / 0% (0/8) | 3 attempts, 1 × HTTP 400 | 2027: 314 |
| `wnba` | 0 (0) | — | — (`no_pre_match_events`) | not sampled (no pre-match events) | 2026: 374 |
| `uefa.champions` | 0 (0) | — | — (`no_pre_match_events`) | route absent (HTTP 400) | 2026: 189 |

Running `lane_permission()` over that file yields exactly **10 `tickets` lanes**
(`nhl`, `nfl`, `cfb`, `eng.1`, `esp.1`, `ger.1`, `ita.1`, `fra.1`, `nba`,
`ncaa_m`) and **3 `research_only`**:

* **`ncaa_w`** — `named_book_coverage_below_50pct`: 0% over 8 pre-tip events at
  `07:12Z` (`02:12 EST`). This is a timing fact, not an absence of women's
  lines: settled game `401903575` returned a full DraftKings `pickcenter`
  (`moneyline`, `spread`, `total` all priced, 3 quotes), i.e. DraftKings posts
  women's lines closer to tip than the overnight probe window. Re-probe in
  `17:00Z–20:00Z` before drawing any conclusion about this lane.
* **`wnba`** — `no_pre_match_events` (off-season; the season index still holds
  `2026: 374` finals for when it returns).
* **`uefa.champions`** — `no_pre_match_events` (between matchdays).

**Live-verified `/predictor` shapes per sport:**

* `nba` — 3/3 samples with `prob_home`, `predicted_margin` and both expected
  point totals (`teamExpectedPts`/`oppExpectedPts`), so BPI can price a spread
  *and* a total.
* `nfl` / `cfb` — 3/3 samples with `prob_home` and `predicted_margin`, no
  expected points: the FPI spread is priced, the FPI total stays `None` rather
  than being invented. NFL payloads sum to 100 only with `teamChanceTie`
  included, so the parser conditions the two-way price on no tie and records
  `prob_tie` / `tie_rule: "push_conditional"`.
* `ncaa_m` / `ncaa_w` — `prob_home` only: the margin is inverted through
  `normal_ppf` and no total is emitted. Small women's matchups can return
  **HTTP 400** and are skipped, not retried into a fabricated number.
* `nhl` and every soccer lane — HTTP 400 (no route); skipped via
  `League.predictor`.

Four defects the captures fixed, each with a test:

1. **Draw prices were counted, not read.** Soccer `pickcenter` carries
   `moneyline.draw.close|open.odds` *and* `drawOdds.moneyLine`; coverage said
   `moneyline_draw = 0.00` because the probe counted market labels instead of
   priced sides. Coverage is now computed by `quote_market_presence`, the flat
   `drawOdds.moneyLine` variant is an extraction fallback, and every soccer lane
   below reports draw coverage from priced sides.
2. **NFL FPI failed on tie mass only** (`99.6480` without `teamChanceTie`) —
   fixed as above; CFB has no tie key and is unchanged.
3. **Book labels differ per sport.** `"Draft Kings"` (NFL/CFB/basketball) vs
   `"DraftKings"` (soccer) vs `"Bet 365"` (Bundesliga) are one book;
   `canonical_book` normalises them (`ger.1` records
   `book_normalisation {"Bet 365": "Bet365"}`) and the probe keeps the raw
   labels for audit.
4. **`/predictor` does not exist outside football and basketball.**

Still unverified, and therefore not claimed:

* one `ger.1` Bet365 offer still arrives in the flat `homeTeamOdds` /
  `awayTeamOdds` shape with no nested `moneyline` / `pointSpread` block. The
  parser records `odds.shape_unrecognised` with the observed keys instead of
  guessing, which is why `ger.1` measures 7/8 rather than 8/8.
* pre-tip coverage for `uefa.champions` (between matchdays), `wnba` (off-season)
  and `ncaa_w` (needs a probe inside the `17:00Z–20:00Z` window above).
* whether a book pushes or loses a two-way moneyline on a tie. The model prices
  push-conditional because the live NFL quote carries only `home`/`away` sides,
  and that assumption is recorded in `tie_rule` rather than buried in code.
* FPI expected scores for NFL/CFB (absent from the payload), so FPI totals stay
  `None` for those two lanes only.

`scripts/probe_espn.py` re-measures all of this; `scripts/daily.py` re-reads the
committed file on every run, so a lane whose coverage lapses stops ticketing
without anyone editing code.

## Layout

```text
src/bullion/
  models.py        sport-aware dataclasses, 13-lane registry, UTC/SAST helpers
  store.py         stdlib sqlite3 warehouse (games, quotes, predictor, forecasts,
                   and scoreboard_backfill_days — DB-backed resume state, so
                   "already swept" is evicted with the finals it describes; no
                   file manifest)
  espn.py          keyless client + fail-closed parsers + probe-permission contract
  calibration.py   walk-forward Platt scaling, rolling Brier, segment gates
  engines/         nhl.py (Poisson + OT/SO), football.py (FPI vs empirical),
                   basketball.py (BPI vs empirical, college line anchoring),
                   soccer.py (Dixon-Coles)
  tickets.py       candidate pool, gates, build_accas, immutable receipts, settlement
  replay.py        parity, variants, kelly/daily_pct sizing, paired A/B bootstrap
  runner.py        per-lane ingest→forecast→freeze→grade orchestration
scripts/
  probe_espn.py    live ESPN probe → localdata/_probes/espn_probe_latest.json
  backfill.py      historical finals (ratings warm-up only; never historical prices)
  daily.py         end-to-end daily runner
  replay_harness.py CLI over bullion.replay
tests/             offline suite; sockets are blocked, every payload is synthetic
                   or a committed capture
.github/workflows/ (read-only here: the Agent App lacks `workflows` permission)
  bullion_daily.yml  scheduled ticks: backfill, lanes, cards, parity gate, commit
  probe_espn.yml     live evidence capture (full or subset), committed to main
```

## External schedule

GitHub's `schedule:` cron is deliberately unused (it delays or drops ticks). An
external scheduler posts a `repository_dispatch` event
`bullion-scheduled-tick`; `.github/workflows/bullion_daily.yml` documents the
ticks it expects:

| UTC | SAST | what it is for |
|---|---|---|
| `06:15Z` | 08:15 | overnight US finals refresh + grading |
| `10:05Z` | 12:05 | after the 12:00 SAST Top-6 soccer cutoff |
| `14:15Z` | 16:15 | after the 16:00 SAST NFL/CFB/NHL cutoff |
| `15:10Z` | 17:10 | basketball noon-ET matinees (12:00 EST / 13:00 EDT) |
| `21:15Z` | 23:15 | EU soccer grading + basketball D+1 early-ET tips (17:00-18:15 EST / 18:00-19:15 EDT) |
| `22:15Z` | 00:15 | basketball prime ET slate (18:30-19:15 EST / 19:30-20:15 EDT) |
| `23:15Z` | 01:15 | basketball late ET slate (19:30-20:15 EST / 20:30-21:15 EDT) |
| `18:15Z` *(optional)* | 20:15 | weekend 15:00-16:30 EST NCAA afternoon matinees |

Each run is idempotent and write-once at cutoff, so a missed tick costs a pre-tip
observation, never a rotated receipt. `.github/workflows/probe_espn.yml` is
dispatched separately (full sweep, or `leagues` for a subset merge) whenever a
lane's evidence needs re-measuring.

## Commands

```bash
python3 -m pip install -e ".[dev]"     # pytest + ruff (runtime deps: none)
ruff check .
pytest -q

# 1) capture real ESPN evidence (needs network; run in Actions)
python3 scripts/probe_espn.py --out localdata/_probes/espn_probe_latest.json
python3 scripts/probe_espn.py --leagues basketball        # subset: merges, never overwrites
python3 scripts/probe_espn.py --leagues ncaa_w --date 2026-11-30  # re-probe near tip

# 2) warm the ratings (finals only; newest season first, resumable)
python3 scripts/backfill.py --league all --seasons 2024,2025,2026 --max-events 400
python3 scripts/backfill.py --league all --workers 6 --max-seconds 1200  # bounded cold run

# basketball history is by ET day, one request per day, not per game
python3 scripts/backfill.py --league basketball --days-back 400 --max-seconds 600
python3 scripts/backfill.py --league basketball --detect-cold   # JSON: lanes still short

# 3) daily lane runs (freeze only at/after cutoff, only with a probe report)
python3 scripts/daily.py --league all --date 2026-11-28
python3 scripts/daily.py --league basketball --date 2026-11-28  # merges into daily_<day>.json
python3 scripts/daily.py --league nba,wnba --date 2026-11-28    # comma lists work everywhere

# 4) replay
python3 scripts/replay_harness.py --parity
python3 scripts/replay_harness.py --lanes nba,ncaa_m --mode cross-league --legs 2
python3 scripts/replay_harness.py --stake-mode kelly --kelly 0.25 --slots 3
python3 scripts/replay_harness.py --variant legs=3 --against legs=2 --ab
```

`python3 scripts/replay_harness.py` on a clean checkout prints
`zero settled bet-days: no profit, edge, or readiness claim is supported`.

The backfill is budgeted and resumable. Football/hockey/soccer walk the season
event index newest-first with a bounded summary pool; basketball walks **ET-day
scoreboards** instead (one request per day, finals only) and records completed
days in `scoreboard_backfill_days` inside the warehouse, so the resume state is
evicted with the finals it describes. `--max-seconds` (1200 for the mixed cold
run, 600 for the basketball step) stops cleanly with exit code 0 and leaves
stored finals behind for the next run; `elapsed_seconds`,
`processed_per_second` and `projected_remaining_seconds` make the next budget
decision observed rather than guessed. Transient CDN failures — including
chunked-transfer `IncompleteRead`, which is an `HTTPException` and not an
`OSError` — are retried with gzip-encoded requests before becoming an
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

Human-readable mirrors sit next to their JSON: `auto_tickets_<day>.txt` is the
mutable card for one frozen day (re-rendered on every settlement pass, so a
settled leg shows its status and official score instead of the `PENDING` it was
frozen with) and `auto_tickets_performance.txt` is the rolling paper ledger
(bank, W-L-P record, per-day `BET-DAYS`, `HELD` and recent `SETTLED` tickets).
Neither ever feeds a gate: every decision and replay reads the JSON.

Semantic notes for cross-sport readers: soccer 1X2 is
`market="moneyline"` with `selection` in `{home, draw, away}`; hockey pucklines
and soccer handicaps are `market="spread"` with a **home-oriented** `line`;
integer lines carry a real `push_probability`.

## Settlement, calibration and acca rules

* **Settlement follows the market's cardinality, not the sport's.** A two-way
  moneyline (NFL/CFB, where no draw price exists) voids to `push` on a level
  game; a three-way soccer `moneyline` pays `draw` and loses both sides.
  `resolve_outcome(..., has_draw=...)` is told which one it is grading.
* **Calibration counts fixtures, and weights them equally.** One game emits a
  row per side and per market, so the gate reads `n_games` (unique settled
  `game_id`s) and each row carries `1 / rows_for_game` inside the Brier score
  and the Platt fit. Both numbers appear in every report (`n`, `n_games`). A
  settled row with **no `game_id`** cannot be weighted at all: `n_games` drops
  to 0, `observations_missing_game_id` is reported, and the segment fails
  closed instead of being measured on an inflated row count.
* **Brier capacity follows market cardinality.** 0.22 applies to a 3-way
  `moneyline` only; binary `spread`/`total` markets in every league use 0.25.
  Every segment is also required to beat its own base-rate Brier.
* **Accas are built from distinct fixtures.** `eligible_rows()` keeps every
  eligible market per game (collapsing to one row per game discarded the
  `@1.90-@2.10` legs that reach the `@3.00` acca floor), collapses a duplicate
  wager from a second model surface to its **more confident** row rather than
  whichever arrived first, and `build_accas()` refuses any combination with two
  legs from the same fixture.
* **The Kelly search is exact.** Legs are sorted by price, so the last leg of a
  ticket is the contiguous price window that lands the product inside the band,
  found with `bisect`; every admissible 2- and 3-leg combination is scored, with
  no hop budget and no capped greedy. A last leg beaten on both probability and
  price by another in-band leg cannot be optimal (`g*` rises with both), which
  keeps a 130-fixture slate fast. The subtlety that makes it exact rather than
  merely fast: on a multi-market board a leg from a fixture already in the
  ticket cannot be chosen, so it must not prune anything either — it is stepped
  over without raising the bar, which is why the frontier is walked with a
  segment tree instead of a precomputed "previous greater probability" chain.
* **The legacy blend path is untouched.** Two-leg accas stay enumerated rather
  than searched, because a best-first walk cannot be both complete and bounded:
  on a 67-fixture board the below-band favourite pairs ahead of the first
  in-band pair by more than the pop budget, which returned zero accas on exactly
  the big slates the lane exists for. Three or more legs keep the bounded walk
  with a budget that scales with the candidate pool (`max(5000, 200 × candidates)`).
  That code path is what every pre-Option-1 receipt must replay through
  byte-for-byte, so it is the one thing this policy deliberately did not touch.
* **`prior_roi` ignores pushes**, because a pushed stake returns exactly its own
  unit and is not evidence of edge.
* **Shape is fail-closed, in every league.** A soccer `moneyline` without its
  draw price is a draw-no-bet price and is not quoted as 1X2. Quarter lines
  (`.25`/`.75`) are split stakes that binary grading would settle wrongly, so
  they are recorded as `pickcenter.line.split` gaps instead of being priced —
  for a hockey puckline or football spread exactly as for a soccer handicap.
* **A forecast's identity is the wager, never the price.** `forecast_id` hashes
  `league|game|market|period|selection|line|source`. The book and observation
  time used to be part of that hash, though one forecast is recorded before a
  book is attached and frozen after, which produced two ids for one
  observation — so restoring history from receipts duplicated every row it
  meant to heal. Price lives on the receipt's `price_evidence`.
* **Rehydration runs on every daily run, not only on a cold cache.** The
  workflow's backfill step is skipped when `localdata/bullion.db` exists, so a
  restored-but-stale warehouse would stay short of history forever;
  `scripts/daily.py` now rehydrates at startup and reports the counters under
  `rehydrated` in the run summary. It is idempotent: a warm warehouse restores
  0 rows and its `n`/`n_games`/Brier do not move.
* **A refrozen fixture is one pick, not two.** `settled_pick_history()` keys a
  settled pick on `(game_id, market, period, selection, line)` and keeps the
  earliest receipt, so a postponement that is refunded and frozen again on a
  later day cannot inflate `prior_roi` or its sample.
* **Prices never mix phases.** All sides of a market come from the same
  `close`/`open` phase; a half-closed offer is not quoted at all.
* **NHL overtime adds exactly one goal.** Overtime is sudden death and a
  shootout winner is credited one goal, so `OT_GOAL_DISTRIBUTION = {1: 1.0}` and
  a regulation draw is never settled at the regulation total.
* **ESPN's projected margin is used when it is all FPI publishes.**
  `teamPredPtDiff` prices the FPI spread surface, while `expected_total` stays
  `None` rather than being invented from a missing expected score.
* **Walk-forward history survives a cache eviction.** `scripts/backfill.py`
  rehydrates `forecast_snapshots` from the committed receipts on every run
  (`INSERT OR IGNORE`, so it is idempotent and additive), and the warehouse
  migrates new columns onto an older cached `bullion.db`.
* **A replay settles what settled.** A pending leg on one ticket no longer
  discards a same-day win on another, and one lane's exception no longer aborts
  the other lanes (`lane_error`, exit 1 after every lane has run).

## Immutability and sizing

* Receipts and `auto_tickets_<day>_no_bet.json` locks are write-once. Later scans
  grade into `auto_tickets_performance.json` / `.txt` and refresh
  `auto_tickets_<day>.txt` beside its receipt; they never rotate a frozen
  selection or price.
* A no-bet lock carries the `candidate_pool` and `selection_policy` it declined,
  and settlement grades that pool (`positions: []`, so ticket P&L is untouched).
  Replay treats such a day as a legitimate 0-ticket day and `--parity` still has
  to agree that no ticket was due — during the calibration warm-up this is the
  only pre-tip evidence archive that accumulates in Git.
* `prior_roi` scores only the **model's own +EV side**, priced inside the frozen
  ticket band, at most one row per `(game, market)`: the pool holds both sides of
  every market, so averaging all of it would measure the bookmaker's vig
  (~-4.5%) and block every lane forever.
* Settlement never depends on today having a slate: every lane run refreshes
  recently tipped games from the summary (discovery only asks for the current
  SAST day's ET dates) and then settles, so the 06:15Z overnight cron grades
  yesterday instead of returning early on `no_slate`.
* Live paper tickets are flat 1.00u. Aggressive sizing (`--stake-mode daily_pct`,
  `kelly`, `skim`, `--daily-pct 0.25`) exists only in the replay harness and is
  flagged `[LOW SAMPLE: n < 30]` until 30 clean settled bet-days accumulate.

## Status

* **250 offline unit tests** (`pytest -q`), sockets blocked. They cover: side
  resolution from `homeAway`, odds nesting and coverage gaps, `$ref` predictor
  orientation and contradiction handling, post-tip rejection, summary/OT period
  parsing, engine probability identities, Dixon-Coles rho behaviour, basketball
  prior/half-life shrinkage, college line anchoring, calibration leakage guards,
  gate rejection paths, receipt schema/immutability, settlement, 100% replay
  parity, paired A/B NO-OP refusal, cross-sport acca assembly, zero-settled-day
  handling, and the probe's per-league `captured_utc` stamping and
  non-destructive merge (carried-over lanes never look fresher than their
  measurement).
* `scripts/replay_harness.py --parity` rebuilds each frozen receipt and diffs
  ticket id, combined price, and every leg price, not just the selected legs; it
  exits 1 on any mismatch, and reports `vacuous: true` when there is no receipt
  to check (an empty vault is not a parity proof). Current vault: **9/9
  receipts matched, `mismatched: 0`**.
* Summary `pickcenter` is a quote source: when a scoreboard `odds` array is
  empty, one summary fetch per fixture supplies the freeze, and the discovery
  gap stays in the coverage file so the empty scoreboard array remains visible.
* **Verified live (`LIVE_VERIFIED 2026-11-28T07:12:44Z`, 13 lanes):** all four
  sports measured in one capture — see the table above. **10 lanes ticket**
  (`nhl`, `nfl`, `cfb`, `eng.1`, `esp.1`, `ger.1`, `ita.1`, `fra.1`, `nba`,
  `ncaa_m`); **3 stay `research_only`** (`uefa.champions` and `wnba` on
  `no_pre_match_events`, `ncaa_w` on 0% pre-tip coverage at `07:12Z` — a
  DraftKings timing fact that a `17:00Z–20:00Z` probe can settle).
* **Still unverified:** the `ger.1` Bet365 flat-shape offer, `uefa.champions` /
  `wnba` / `ncaa_w` pre-tip windows, FPI expected scores for NFL/CFB, and
  whether a book pushes or loses a two-way moneyline on a tie (priced
  push-conditional, and labelled as such in `tie_rule`). Lanes open tickets only
  where the committed probe shows `status: ok` on ≥3 pre-match events and ≥50%
  coverage.
* **No edge claim is made anywhere.** The receipts are paper tickets and the
  ledger is a paper ledger; `prior_roi` and the Brier segments exist to decide
  whether a lane may ticket at all, not to advertise returns.
