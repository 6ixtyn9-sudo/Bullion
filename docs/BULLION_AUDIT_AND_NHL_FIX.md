# BULLION — Full-System Audit & NHL Fix (corrective pass, v2)

> **Naming note (added after the fix merged):** the SOURCE tags quoted
> throughout this document (`nhl_poisson_v1/v2`, `empirical_score_v1/v2`,
> `espn_fpi_v1/v2`, `dixon_coles_v1/v2`) were retired in favour of bare
> model-family names (`nhl_poisson`, `empirical_score`, `espn_fpi`,
> `dixon_coles`). Evidence, numbers and the embedded patch are unchanged —
> only the tag strings moved.


**Repo:** `6ixtyn9-sudo/Bullion` @ `main` (`84a3c59`, after PR #10)
**Warehouse:** real cached `localdata/bullion.db` — **6,230 finals** across 9 leagues
(nhl 750 · nfl 746 · cfb 737 · eng.1 750 · esp.1 500 *(cap)* · ger.1 750 · ita.1 750 ·
fra.1 750 · uefa.champions 497). The local cache is ~259 finals behind the operator's
6,489 figure (`esp.1` at its 500-final cap, NHL backfill ran through 2026-02-03); every
number below is computed from this actual file. NHL yields **731 scored games** under
the seeder's ≥15-priors floor.
**Method:** every walk-forward measurement keeps `result_day < D`. Hyperparameters
were tuned **only** on the earlier chronological split (60% of scored days) and
evaluated **once** on the untouched holdout. Gate metrics come from production
`CalibrationBook.report()` **verbatim** (3-way soccer includes draws; 2-way pools
home+away rows), plus a per-selection class-frequency baseline reported separately.
**Constraints honoured:** `DEFAULT_MIN_SAMPLES=30`, Brier caps, `cap_vs_base_rate`,
EV>0 and named-book pre-tip checks untouched — the one calibration change is a
**tightening** of the base-rate baseline (§1, finding 3), never a relaxation. No new
repo machinery: all empirical scripts ran outside the repo (`scripts/audit/`, `/tmp`).

---

## 0. Corrections of record (review of the first report)

The first version of this report contained four errors. All four were re-verified
against the code and the real warehouse before this rewrite:

1. **"NHL stays locked" was wrong.** The report computed `base_rate_brier` on
   home rows only (0.2483) and concluded 0.2494 ≥ 0.2483 → locked. Production
   `CalibrationBook` pools home+away rows, so its baseline is **always exactly
   0.2500** for two-sided moneyline (and 0.2222 for 1X2 — one win per game over
   three rows). At 0.2494 < 0.2500 the first patch would have **unlocked NHL on a
   model worse than blindly predicting the 54.2% realized home rate**. The pooled
   baseline is a genuine bug in `src/bullion/calibration.py` and is fixed here.
2. **The `home_factor` fix recovered only half of home advantage.** Applying the
   factor to `lam_home` only (with `league_rate = (home+away)/2`) makes the factor
   converge to `sqrt(true)`: verified 1.0699 fitted vs 1.15 true on synthetic data
   matching NHL's scoring shape, with expected totals inflated +0.215 goals/game —
   matching the +0.185 measured on the real warehouse. The fix is now symmetric
   (`sqrt(hf)` applied to home, divided out of away) with the factor identified by
   the home:away ratio, plus a `league_rate` re-centering; it recovers 1.1502
   (true 1.15) with totals bias +0.014 on the same synthetic data.
3. **"Bug 2 is not a bug" was wrong.** Checking `|attack@25 − attack@200| = 0.0083`
   proved stationarity at the *wrong* fixed point. On noiseless data with true
   attacks 1.40/0.60 the damped IPF as written converges to 1.1368/0.8632 —
   stationary (delta 0.0000) but not the truth: the ratio keeps the team's own
   rating inside the exposure, pinning the fixed point at `sqrt(true)`. The true
   damped Gamma-Poisson MAP divides the rating out of the exposure first and
   recovers true ratings (3× lower error; §2.1). (The review's own synthetic
   numbers — 1.2139/0.7659 and the 1.3966/0.6044 recovery — differ from ours
   because the synthetic schedules differ; our 2-team setup is underdetermined so
   we verified recovery on an identifiable 12-team problem instead. The mechanism
   claim is confirmed in every setup.)
4. **The §2.3 soccer table scored the wrong metric.** It dropped draw games and
   scored 2-way home-rows-only Brier (n=539 for eng.1 instead of 733). Under
   production `CalibrationBook.report()`, all five soccer lanes **already passed**
   in PR #10 (Brier 0.2026–0.2055 vs baseline 0.2222) — exactly as the review
   stated. The corrected tables below use production `CalibrationBook` verbatim.

Two additional harness defects were found and fixed during the re-run (disclosed
for completeness): the first audit harness graded soccer draws as pushes on the
home row instead of losses (distorting soccer Brier weights), and one candidate
run accidentally evaluated the MAP form twice instead of comparing against the
compressed form (`map_form` not plumbed). All numbers below come from the fixed
harness, and the shipped patch was re-verified against it (§6).

---

## 1. Executive summary

**Confirmed defects, all now fixed in the attached patch:**

1. **P0 — asymmetric home advantage.** `fit_ratings` applied the home factor to
   `lam_home` only. First-order effect (the original audit): the factor 2-cycled
   and exited at 1.0000 on odd `iterations=25` — zero home advantage everywhere.
   After the cycle was fixed, the second-order effect remained: the factor
   converged to `sqrt(true)` (NHL: 0.5192 predicted vs 54.17% realized home wins)
   and totals inflated +0.185 goals/game. The symmetric parameterization fixes both.
2. **P1 — sqrt-compressed IPF.** The damped update kept the team's own rating in the
   exposure, so every attack/defence rating in every sport has been a sqrt-compressed
   version of the true MAP rating (1.40 → 1.1368 on noiseless data). Fixed by
   dividing the rating out before the ratio (true damped MAP; damping kept — it is
   the stabilizer, and with the exposure fix it converges to the truth).
3. **P1 — vacuous calibration baseline.** `CalibrationBook` pooled home+away rows,
   making `base_rate_brier` identically 0.2500 (two-way) / 0.2222 (1X2). Any model
   with residual signal passed `not_better_than_base_rate` regardless of direction.
   The baseline is now the per-selection class-frequency predictor; rows without a
   selection label fall back to the pooled rate. Measured outcome: blocks the NHL
   false unlock (0.2484 vs realized-home-rate baseline 0.2483 — razor thin, errs
   locked), breaks no passing lane (§3.4).
4. **P1 — seed idempotence without `source`** (`scripts/backfill.py`): a `SOURCE`
   bump silently seeded zero rows. Fixed per `(game_id, source)`; regression test added.
5. **P2 — prior semantics.** `prior_weight=2.0` entered as raw goals = 0.70 NHL /
   0.09 NFL pseudo-games. Now `k * league_rate`, with the default tuned (2→4) and
   per-lane overrides NHL 16 / CFB 2 — all chosen on train splits only, confirmed
   blind on holdouts (§2.3).

**Measured outcome of the shipped patch** (production `CalibrationBook`, 8 seeded
lanes): every lane improves on both the full sample and the untouched holdout —
holdout Brier sum 1.6569 → **1.5959**. NHL goes from failing the Brier cap
(0.2506, no home advantage) to 0.2484 full / 0.2495 holdout with mean p_home
0.5314 vs realized 0.5417 — and is **still correctly denied a ticket gate** by the
tightened baseline, by a margin of 0.0001 (honestly a coin flip; it errs locked).
NFL/CFB and all five soccer lanes pass both the production gate and the tightened
baseline.

### Ranked bug table (all findings, post-corrective-pass status)

| # | Sev | Location | Finding | Status |
|---|-----|----------|---------|--------|
| 1 | **P0** | `src/bullion/engines/__init__.py` (`fit_ratings`, `rates_for`) | Asymmetric home factor → 2-cycle to 1.0000 (first order), then `sqrt(true)` half-recovery + totals inflation (second order). Zero-to-half home advantage in every sport. | **FIXED** (symmetric ratio form) |
| 2 | **P1** | `src/bullion/engines/__init__.py` (`fit_ratings`) | sqrt-compressed IPF: fixed point at `sqrt(true_rating)` for attack/defence in every sport. | **FIXED** (true damped MAP) |
| 3 | **P1** | `src/bullion/calibration.py:232` | Pooled base rate: `baseline_brier` identically 0.2500 (2-way) / 0.2222 (1X2) — the `cap_vs_base_rate` gate was vacuous; a worse-than-class-rate model could unlock a lane. | **FIXED** (class-frequency baseline; tightening) |
| 4 | **P1** | `scripts/backfill.py:431-440` (pre-fix) | Seed idempotence without `source` → SOURCE bumps seed zero rows. | **FIXED** (per game_id+source) |
| 5 | **P2** | `engines/__init__.py` (prior) | `prior_weight` in raw goals (0.70 NHL / 0.09 NFL pseudo-games); default 2.0 untuned. | **FIXED** (pseudo-games; 4.0 default, NHL 16 / CFB 2, train-tuned) |
| 6 | P2 | `engines/__init__.py` (post-fix residual) | NHL expected-total bias +0.130 goals/game — decomposed: regulation-window drift +0.20, Poisson draw share underestimates actual OT incidence (15.7% vs 22.3%, −0.07). Not the home-factor inflation (that is fixed); moneyline unaffected. | Deferred (§4) |
| 7 | P2 | `src/bullion/engines/football.py:76` | `residual_sigmas` floors variance at 1.0 → sigma can collapse to 1.0 on degenerate short windows (verified repro), producing near-0/1 probabilities. | Deferred (§4) |
| 8 | P2 | `src/bullion/espn.py:1190-1242` | Probe staleness unbounded: `lane_permission` never checks `captured_utc` age. | Deferred (policy decision) |
| 9 | P3 | `src/bullion/engines/nhl.py:103-104` | B2B trims offense only (0.975); asymmetric defense factor rejected on train AND holdout — adds noise, not signal. | No change (documented) |
| 10 | P3 | `src/bullion/engines/soccer.py:39-49,88-103` | Dixon-Coles `tau` can go negative at extreme rates (verified: `tau(0,0) = −7.82` at rates 7.0), cell clamped to 0, matrix renormalized. | Deferred |
| 11 | P3 | `src/bullion/espn.py:500-508` | `_spread_line` flat fallback can mix price phase and line snapshot (edge case). | Deferred |
| 12 | P3 | `src/bullion/tickets.py:98` | `attach_quotes` takes the max historical pre-tip price (optimistic paper convention; documented in receipts). | Document only |
| 13 | P3 | `src/bullion/models.py:365` | `f"{float(line):g}"` id formatting can collide for exotic line values. | Document only |
| 14 | P3 | `src/bullion/store.py:64` | Quote uniqueness at second resolution collapses same-second same-book quotes. | Document only |
| 15 | P3 | `src/bullion/models.py:198-204` | `regulation_scores()` sums partial linescores silently when periods < regulation. Latent (no consumer today). | Document only |

Severity policy: **P0** = silently corrupts every probability shipped; **P1** = wrong
results in a whole subsystem or silently drops/voids evidence; **P2** = measurable
bias or a policy hole with bounded impact; **P3** = edge case / hygiene.

---

## 2. Real-data empirical results

### 2.1 Synthetic verifications (catches 2 and 3, noiseless ground truth)

**Rating recovery, identifiable 12-team problem** (true attacks 0.70–1.36, defences
1.0, noiseless scores, balanced schedule, 200 iterations):

| estimator | MAE(attack vs true) | MAE(rates vs true) | true 1.36 → | true 0.70 → |
|---|---|---|---|---|
| production/patched-v1 (sqrt-compressed) | 0.1124 | 0.3227 | 1.1270 | 0.8543 |
| **true damped MAP (shipped)** | **0.0405** | **0.1108** | **1.2637** | **0.7242** |

The compressed form is perfectly stationary at the wrong point (2-team check:
true 1.40/0.60 → 1.1368/0.8632, |@25−@200| = 0.0000). The residual MAP error is the
pseudo-game prior doing its documented job (shrinkage toward 1.0).

**Home-factor recovery** (12 equal-strength teams, true symmetric home advantage
HF = 1.15, i.e. home teams score `sqrt(1.15)` and away teams `1/sqrt(1.15)`):

| form | fitted HF | E[total] bias | home:away ratio |
|---|---|---|---|
| asymmetric (patched-v1) | 1.0699 ≈ √1.15 | **+0.215 goals/game** | 1.0699 |
| **symmetric + ratio update + lr re-centering (shipped)** | **1.1502** | **+0.014** | 1.1502 |

### 2.2 NHL decomposition (731 scored games, production `CalibrationBook`)

| metric | production `nhl_poisson_v1` | patched-v1 (asymmetric) | **shipped (symmetric MAP, pw 16)** |
|---|---|---|---|
| mean predicted p_home | 0.5044 | 0.5192 | **0.5314** |
| realized home-win rate | 0.5417 | 0.5417 | 0.5417 |
| Brier (full window) | 0.2506 | 0.2494 | **0.2484** |
| Brier (blind holdout, n=301) | 0.2509 | 0.2506 | **0.2495** |
| production baseline (pooled, always 0.2500) | 0.2500 | 0.2500 | 0.2500 |
| **realized-home-rate baseline (class-frequency)** | **0.2483** | **0.2483** | **0.2483** |
| production gate (`brier<0.25 ∧ brier<0.25`) | FAIL | **pass** (false unlock) | **pass** (still a false unlock) |
| **tightened gate (`brier<0.25 ∧ brier<0.2483`)** | FAIL | **FAIL** | **FAIL** (0.2484 > 0.2483, by 0.0001) |
| AUC (home rows) | 0.5463 | 0.5454 | 0.5437 |
| strictly prequential Platt Brier | 0.2492 | — | — |
| E[total] bias (goals/game) | +0.063 | **+0.185** | **+0.130** |

Read-outs:

- The shipped model recovers most of the missing home advantage (0.5044 → 0.5314
  predicted vs 0.5417 realized; the patched-v1 asymmetric form recovered less than
  half of the gap — the review's catch was exact).
- **AUC does not improve** (0.5463 → 0.5437): the ranking signal in this NHL sample
  is weak under every fitter. The Brier gains come from calibration level (home
  advantage + shrinkage), not from better ranking — reported honestly.
- **The gate question is decided by the baseline, not the model.** Under the pooled
  baseline the model "unlocks" at 0.2494/0.2484 < 0.2500 while being worse than the
  54.2%-home constant (0.2483). The tightened class-frequency baseline blocks it —
  by 0.0001 on the shipped model. A 0.0001 margin is a coin flip; it errs on the
  side of staying locked, which is the only defensible direction for a gate.
- The residual totals bias (+0.130) is **not** the home-factor inflation (fixed —
  see §2.1): it decomposes as regulation-window scoring drift (+0.20 between fitting
  window and scored days) minus the model's Poisson draw share underestimating
  actual OT incidence (15.7% vs 22.3% → −0.07). A Dixon-Coles-style rho on NHL
  draws is the obvious future fix (§4).

### 2.3 Prior sweep (train-only) and the shipped configuration

Train Brier per lane per prior (pseudo-games; symmetric MAP fitter; split at 60% of
each lane's scored days):

| lane | pw=2 | pw=4 | pw=8 | pw=12 | pw=16 | train-chosen |
|---|---|---|---|---|---|---|
| nhl | 0.2596 | 0.2544 | 0.2501 | 0.2484 | **0.2476** | **16** (gap 0.0068) |
| nfl | 0.2477 | 0.2420 | **0.2396** | 0.2397 | 0.2403 | 8 (gap 0.0024) |
| cfb | **0.1618** | 0.1706 | 0.1811 | 0.1862 | 0.1892 | **2** (gap 0.0088) |
| eng.1 | **0.1902** | 0.1904 | 0.1921 | 0.1940 | 0.1959 | 2 (gap 0.0002) |
| esp.1 | 0.2065 | 0.2046 | **0.2040** | 0.2045 | 0.2054 | 8 (gap 0.0006) |
| ger.1 | 0.2091 | 0.2079 | **0.2076** | 0.2082 | 0.2089 | 8 (gap 0.0003) |
| ita.1 | 0.2048 | **0.2042** | 0.2049 | 0.2060 | 0.2071 | 4 (gap 0.0000) |
| fra.1 | 0.1992 | **0.1984** | 0.1984 | 0.1990 | 0.1998 | 4 (gap 0.0000) |

Global train-total minimum: **pw=4**. A single global prior is structurally wrong —
NHL (low-scoring, weak signal) wants heavy shrinkage, CFB (blowouts, strong signal)
wants light — but per-lane train-argmin overfits: on the untouched holdout it *hurt*
eng.1 (0.2303 vs 0.2138 at pw 4), ger.1 (0.2092 vs 0.1920) and ita.1 (0.2067 vs
0.1942), because those train gaps (0.0002–0.0006) are pure noise. The shipped rule,
decided on train data only: **deviate from the global default only where the train
improvement is decisive (> 0.005)** — leaving exactly NHL 16 and CFB 2. The blind
holdout confirms every deviation (NHL 0.2495 vs 0.2546; CFB 0.1521 vs 0.1586) and
every non-deviation (nfl holdout 0.2234 at 4 vs 0.2261 at its train-favoured 8).

### 2.4 Eight-lane results and gate outcomes (shipped configuration, verified against patched production code)

Gate metrics from production `CalibrationBook.report()` verbatim (min 30 samples,
caps 0.25 two-way / 0.22 three-way, `cap_vs_base_rate=True`); the "tightened" column
adds `brier < class-frequency baseline`. Full = whole scored window; holdout = later
40% of scored days, untouched by any tuning.

| lane | n games | prod v1 full→hold | cand2 (compressed, pw2) full→hold | cand1 global pw4 full→hold | **shipped full→hold** | prod baseline | class baseline | prod gate | tightened gate |
|---|---|---|---|---|---|---|---|---|---|
| nhl | 731 | 0.2506→0.2509 | 0.2500→0.2521 | 0.2545→0.2546 | **0.2484→0.2495** | 0.2500 | 0.2483 | pass | **FAIL (locked)** |
| nfl | 723 | 0.2382→0.2264 | 0.2362→0.2297 | 0.2345→0.2234 | **0.2345→0.2234** | 0.2500 | 0.2489 | pass | pass |
| cfb | 705 | 0.1682→0.1607 | 0.1629→0.1553 | 0.1663→0.1586 | **0.1583→0.1521** | 0.2500 | 0.1777 | pass | pass |
| eng.1 | 733 | 0.2026→0.2160 | 0.2022→0.2147 | 0.1997→0.2138 | **0.1997→0.2138** | 0.2222 | 0.2185 | pass | pass |
| esp.1 | 485 | 0.2037→0.1955 | 0.2014→0.1945 | 0.1972→0.1872 | **0.1972→0.1872** | 0.2222 | 0.2163 | pass | pass |
| ger.1 | 735 | 0.2055→0.2012 | 0.2049→0.1982 | 0.2015→0.1920 | **0.2015→0.1920** | 0.2222 | 0.2184 | pass | pass |
| ita.1 | 732 | 0.2038→0.1988 | 0.2038→0.1998 | 0.2001→0.1942 | **0.2001→0.1942** | 0.2222 | 0.2205 | pass | pass |
| fra.1 | 732 | 0.2040→0.2074 | 0.2027→0.2060 | 0.2005→0.2037 | **0.2005→0.2037** | 0.2222 | 0.2140 | pass | pass |
| **sum** | | 1.6766→1.6569 | 1.6541→1.6503 | 1.6543→1.6275 | **1.6359→1.5959** | | | | |

Holdout head-to-head, as requested: **candidate 1 (MAP) beats candidate 2
(compressed) at matched priors only marginally** (1.6275 vs 1.6503 with the global
pw4); the decisive factor is the prior, not the damping form — but the MAP form is
what makes pw=2 viable at all for NHL (compressed@pw2: 0.2592, worse than
production). The shipped configuration beats production on **all 8 lanes on both
splits**.

Soccer reality check (correction 4): production soccer Briers are 0.2026–0.2055
against the pooled 0.2222 baseline — **all five soccer lanes already passed in PR
#10**; the first report's "esp.1 flips to pass / fra.1 stays locked" was an artifact
of the wrong 2-way metric. What the fix actually buys soccer is margin
(0.1972–0.2015 vs 0.2026–0.2055) — and all five lanes still pass the tightened
class-frequency baselines (0.2140–0.2205).

### 2.5 Expected-total bias after the fix (catch 4's ask)

| lane | prod v1 | patched-v1 | **shipped** | note |
|---|---|---|---|---|
| nhl | +0.063 | +0.185 | **+0.130** | regulation drift +0.20, OT under-model −0.07 (§2.2) |
| nfl | −3.719 | −1.584 | **−1.903** | scoring-window drift; totals are live-only, not gated |
| cfb | +3.831 | +5.815 | **−1.877** | same; the lr re-centering removes the structural term |
| eng.1 | +0.110 | +0.208 | **+0.105** | |
| esp.1 | +0.008 | +0.135 | **−0.028** | |
| ger.1 | −0.108 | −0.022 | **−0.133** | |
| ita.1 | +0.046 | +0.140 | **+0.043** | |
| fra.1 | −0.036 | +0.047 | **−0.029** | |

The soccer lanes now sit within ±0.14 goals/game (patched-v1's asymmetric hf pushed
them to +0.14–0.21). Football/CFB residuals are season-to-season scoring drift, not
a fitting defect; they matter only when a totals surface is ever seeded (none is
today).

### 2.6 Bonus — ESPN shot stats for a future NHL v2/v3 (checked, not built)

Two live NHL summary payloads (events `401803232`, `401803231`) carry
`boxscore.teams[].statistics` with clean per-team **`shotsTotal`** (= shots on goal
in hockey: 27/17 and 34/28), plus `blockedShots`, `powerPlayGoals`,
`powerPlayOpportunities`, `faceoffsWon`, `shootoutGoals`. A shot-rate NHL v3 —
including the DC-style draw correction suggested by §2.2's OT-share finding — has
real, parseable inputs on every sampled final. Nothing was built.

---

## 3. Detailed bug audit

### 3.1 Bug 1 — asymmetric home advantage (P0) — FIXED

**Location:** `src/bullion/engines/__init__.py` (`fit_ratings`, `rates_for`).
**Root cause chain (two layers):**

1. First order (the original audit): the factor update fed the factor its own output
   (`home_expected` already contained `* home_factor`) → 2-cycle `1.0000 ↔ 1.0666`,
   exiting at 1.0000 on odd `iterations=25` — zero home advantage in every sport.
   Verified by tracing the production iteration on real NHL history.
2. Second order (the review's catch): with the cycle fixed but the factor still
   applied **only** to `lam_home` (while `league_rate = (home+away)/2` and
   `lam_away` was unadjusted), the attack/defence system absorbs half the advantage
   and the factor converges to `sqrt(true)`. Verified on synthetic NHL-shape data:
   true 1.15 → fitted 1.0699, totals inflated +0.215 goals/game; on the real
   warehouse NHL predicted 0.5192 home wins vs 0.5417 realized with +0.185
   totals bias.

**Fix (shipped):** symmetric parameterization — `lam_home = lr·att·def·sqrt(hf)`,
`lam_away = lr·att·def/sqrt(hf)` in both `fit_ratings` and `rates_for` (neutral site
= 1.0); the factor update uses the **home:away ratio** of observed goals to
factor-free expectations (`raw_hf = (hg/he_base)/(ag/ae_base)`, damped, clamped
[1.0, 1.35]); `league_rate` re-centered each iteration by
`(total_goals/total_expected)**0.5`. Synthetic: recovers 1.1502 with +0.014 totals
bias. Real: NHL mean p_home 0.5314 vs realized 0.5417.

### 3.2 Bug 2 — sqrt-compressed IPF (P1) — FIXED

**Location:** `src/bullion/engines/__init__.py` (`fit_ratings` update step).
**Root cause:** `target = (S + pw)/(E + pw)` with `E = a·X` (the team's own rating
inside the exposure), then `a·(target/a)^0.5 = sqrt(a·target) = sqrt(T)` in one
step — and it stays there (T/a evaluated at a = sqrt(T) returns sqrt(T)). The
fixed point is `sqrt(true_rating)` for attack and defence in every sport.
**Proof:** noiseless true attacks 1.40/0.60 → 1.1368/0.8632, stationary to 4 decimals;
identifiable 12-team problem: attack MAE 0.1124 vs **0.0405** for the fixed form
(§2.1). The first report's `|attack@25 − attack@200| = 0.0083` was a stationarity
check at the wrong fixed point — a methodology error recorded in §0.
**Fix (shipped):** divide the team's own rating out of the exposure first
(`exp = expected/attack`), form the MAP target `(S+pw)/(exp+pw)`, keep the sqrt
damping `a·(target/a)^0.5 = sqrt(a·target)`, whose fixed point is now the true
rating. Damping is retained deliberately: full-weight IPF diverges on imbalanced
schedules, and the damped-MAP step still converges (delta 0.0000 at the true point).

### 3.3 Bug 3 — vacuous calibration base rate (P1) — FIXED (tightening)

**Location:** `src/bullion/calibration.py:232` (`baseline_brier=base_rate_brier(pairs, weights)`).
**Root cause:** `report()` pools all rows of a segment. Two-sided moneyline emits one
win and one loss per game → weighted rate exactly 0.5 → `baseline_brier` exactly
0.2500; 1X2 emits one win over three rows → rate exactly 1/3 → 0.2222. The
`cap_vs_base_rate` gate (`brier >= baseline → not_better_than_base_rate`) was
therefore equivalent to the Brier cap and carried no class information: a model
with residual signal in the *wrong direction* passes. Concrete near-miss measured:
the first patch's NHL model (0.2494) would have **unlocked NHL** while being worse
than always predicting the 54.2% realized home rate (0.2483).
**Fix (shipped):** when every row carries its selection label, the baseline is the
**per-selection class-frequency predictor** — each row graded against its class's
realized win frequency in the window (home rows → realized home rate, away rows →
1 − that rate, draw rows → realized draw rate), same per-game weights. That is the
strongest prediction using no team information. Rows without a selection label
(legacy callers, tests) fall back to the pooled rate, so the change is
backwards-compatible. This is a strict tightening for balanced two-way segments
(`f(1−f) ≤ 0.25`) and equal-or-tighter for the observed 1X2 lanes
(0.2140–0.2205 vs 0.2222).
**Measured gate outcomes on the shipped model:** NHL blocked (0.2484 vs 0.2483 —
by 0.0001; a coin flip that errs locked); NFL 0.2345 < 0.2489 ✓; CFB 0.1583 <
0.1777 ✓; soccer 0.1972–0.2015 < 0.2140–0.2205 ✓. **No currently-passing lane
breaks.** Regression tests added (`test_class_frequency_baseline_*`, 3 tests).

### 3.4 Known bug 5 — seed idempotence without source (P1) — FIXED

**Location (pre-fix):** `scripts/backfill.py:431-440`.
`SELECT DISTINCT game_id ... WHERE market='moneyline'` ignores `source`, so a
`SOURCE` bump (e.g. `nhl_poisson_v2`) finds every historical game "already covered"
and seeds **zero** rows. Fixed: idempotence per `(game_id, source)`, with the lane's
source taken from the engine constants. A live row for the *current* source still
excludes the whole game; old-source evidence survives untouched under its own
segment. Verified on a warehouse copy: cold seed 38.4 s / 14,591 v2 rows, warm
re-run 0.06 s / 0 rows, 24 pre-existing live v1 rows preserved verbatim. Regression
test `test_source_bump_reseeds_instead_of_silently_seeding_zero` added.

### 3.5 Known bugs 3 and 4 — measured, fix rejected, no change

- **OT/SO-inclusive fitting** (`runner.py:105-119` + `engines/nhl.py`): 99.6% of NHL
  finals (747/750) carry `regulation_scores()`; 167/750 (22.3%) have OT inflation.
  Regulation-only fitting was tuned on train and **failed the blind holdout**
  (0.2497 vs 0.2492 full-scores) in the first pass — re-confirmed in this pass's
  ablation. `nhl.py`'s "+1 OT goal on simulated regulation draws" remains correct by
  design (official finals contain the extra goal). The observed Poisson draw share
  (15.7%) vs actual OT incidence (22.3%) is the real gap; a DC-style rho on NHL
  draws is the future fix (§4), not score surgery.
- **Offense-only B2B trim** (`engines/nhl.py:103-104`): asymmetric defense factors
  1.02/1.04 lose to 1.00 on train (0.2455/0.2456 vs 0.2454) and showed no holdout
  gain. Rejected again; revisit with shot-quality data (§2.6).

### 3.6 Newly discovered findings (evidence retained from the first pass)

- **#7 (P2) `engines/football.py:76` — sigma floor.** `math.sqrt(max(var, 1.0))`
  floors *sigma* at 1.0. Repro: 25 synthetic NFL finals with 21–22 point scores →
  `residual_sigmas(...)[0] == 1.0`. A short degenerate window could freeze near-0/1
  normal moneyline probabilities. Minimal future fix: floor at `prior_margin/2`.
- **#8 (P2) `espn.py:1190-1242` — probe staleness.** `lane_permission` validates
  schema and coverage but never the report's `captured_utc`. Policy decision
  required (e.g. reject probes older than 7 days); not changed unilaterally.
- **#10 (P3) `engines/soccer.py:39-49` — negative tau.** `score_matrix(7.0, 7.0,
  -0.18)` → `tau(0,0) = −7.82`, cell clamped to 0, matrix renormalized with mass
  silently removed. Only reachable with extreme rates (att·def·lr ≳ 5.5); real fits
  stay ≤ ~2.2. Fix if ever needed: clamp `tau ≥ 0` inside `fit_rho`'s grid search.
- **#11–#15 (P3)** hygiene items as ranked in §1 (spread flat-line phase mixing,
  best-historical-price convention, line-format collision, same-second quote
  collapse, partial-period `regulation_scores`).

### 3.7 Re-checked, found correct (no change)

- `resolve_outcome` 2-way/3-way/push semantics (including the 1X2 draw grading this
  audit's own harness initially got wrong); acca push handling.
- `RESULT_AVAILABLE_AFTER_HOURS` (3 h) + `calibration.settled` strict
  `result_day < as_of_day` — no same-day leak; `_refresh_recent_finals` drains
  40/tick and reports backlog gaps.
- `rehydrate_forecasts` strictly before `seed_walk_forward_ml` in both entrypoints;
  `INSERT OR IGNORE` cannot displace live provenance.
- Replay `--parity` re-selects from frozen pools and cannot be broken by a model
  change (re-verified, exit 0).
- `et_days_for_sast_day`, `decision_cutoff`, `parse_summary` OT detection across the
  three sports.

---

## 4. Exact minimal implementation plan

### Shipped now (the attached patch — one commit)

1. **`fit_ratings` rewritten** (`engines/__init__.py`): symmetric home factor
   (ratio-identified, cap 1.35), true damped MAP exposure, pseudo-game prior, and
   per-iteration `league_rate` re-centering. `rates_for` applies the identical
   symmetric scale, so `residual_sigmas`, `fit_rho` and every engine stay consistent
   automatically.
2. **Priors:** `DEFAULT_PRIOR_WEIGHT = 4.0` (global train-total minimum across the
   8 lanes), with per-lane overrides where the train improvement is decisive:
   `nhl.PRIOR_WEIGHT = 16`, `football.PRIOR_WEIGHTS = {"cfb": 2}` — threaded through
   the engines and the seeder. hf cap 1.25 → 1.35 (CFB/esp/ger pin at 1.25).
3. **Calibration tightening** (`calibration.py` + `runner.py`): `Observation`
   carries `selection`; `report()` uses the class-frequency baseline when labels
   exist, pooled rate otherwise.
4. **SOURCE bumps:** `nhl_poisson_v1 → v2`, `dixon_coles_v1 → v2`,
   `empirical_score_v1 → v2`, `espn_fpi_v1 → v2` (FPI spreads/totals flow through
   the fitted sigmas; FPI moneyline is the provider's own).
5. **Seeder per-`(game_id, source)` idempotence** so bumps re-seed automatically.

Effect on first tick after merge: cold seed ≈ 38 s (one time), warm ticks unchanged
(~0.06 s). v2 segments start fully populated (≈14.6 k rows); v1 evidence ages out
under its own tags.

### SOURCE versioning rules (unchanged from the first pass, restated)

- Bump a `SOURCE` tag **iff** the tag's probability outputs change for the same
  inputs; never reuse a tag. Bumping now requires no manual work: the seeder
  re-seeds under the new tag and old segments keep their evidence.
- The FPI tag rides along with the fitter that feeds its sigmas.

### Deferred (measured, owner = next iteration)

| Item | Trigger to revisit | Planned minimal change |
|---|---|---|
| #6 NHL totals: regulation drift + OT share | when a totals surface is seeded, or NHL v3 starts | DC-style rho on NHL regulation draws; re-check window drift |
| #7 sigma floor | first live football segment freezing with sigma < 6 | floor at prior/2 |
| #8 probe staleness | next probe workflow touch | reject `captured_utc` older than a policy constant |
| #10 tau clamp | only if a real fit exceeds att·def ≈ 3 | clamp `tau ≥ 0` inside `fit_rho`'s grid |
| NHL v3 (shots) | when NHL skill is worth another iteration | shot-rate Poisson on `boxscore.teams[].statistics` (§2.6) |

### Explicitly rejected (do not re-introduce without a fresh blind holdout)

Regulation-only fitting, EN margin caps (2 or 3), asymmetric B2B defense,
undamped full-weight IPF, per-lane train-argmin priors for eng/ger/ita/esp/fra
(train gaps ≤ 0.0006 = noise; holdout-verified harmful for eng/ger/ita).

---

## 5. Verification of the attached patch

- `pytest -q`: **175 passed** (172 baseline after PR #10 + 1 seed-bump regression
  test from the first pass + 3 class-frequency-baseline tests).
- `ruff check .`: **All checks passed.**
- `scripts/replay_harness.py --parity`: **exit 0**.
- Real-warehouse smoke: cold seed **38.4 s** / 14,591 v2 rows; warm re-run
  **0.06 s / 0 inserted**; 24 pre-existing live v1 rows preserved verbatim.
- Patched production code reproduces the audit harness's shipped-configuration
  numbers to ≤ 5e-5 on all 8 lanes (full and holdout).
- The embedded patch applies cleanly to pristine `84a3c59`
  (`git apply --check` in a throwaway worktree).
- Leak discipline: every shipped hyperparameter was tuned on the earlier
  chronological split only; each holdout was evaluated once per configuration;
  the decisive-margin rule for per-lane priors was derived from train gaps alone.

## 6. Unified patch

`git format-patch origin/main` output (branch `fit-ratings-fix`, commit `17cb192`),
applying cleanly to `84a3c59`:

```diff
From 17cb1926bcffc56d8188b61c01ba4ba08eb39581 Mon Sep 17 00:00:00 2001
From: Z User <z@container>
Date: Fri, 27 Nov 2026 11:58:56 +0000
Subject: [PATCH] Fix fit_ratings fixed points, tighten the calibration base
 rate, bump SOURCEs
MIME-Version: 1.0
Content-Type: text/plain; charset=UTF-8
Content-Transfer-Encoding: 8bit

Warehouse audit corrective pass (2026-11-27, real cached bullion.db, 6,230
finals; full evidence in BULLION_AUDIT_AND_NHL_FIX.md). Three defects in
fit_ratings verified on noiseless and real data:

1. Home advantage was applied ASYMMETRICALLY (factor on lam_home only while
   league_rate = (home+away)/2), so the factor converged to sqrt(true) —
   real-NHL-shape synthetic data: true 1.15 -> fitted 1.0699, expected totals
   inflated +0.215 goals/game, and live NHL predicted 51.9% home wins in a
   54.2% league. Now symmetric: lam_home = lr*att*def*sqrt(hf), lam_away
   divided by sqrt(hf), and the factor is identified by the HOME:AWAY ratio
   of observed vs factor-free expectation. Same synthetic check recovers
   1.1502 with totals bias +0.014; NHL mean p_home moves 0.5044 -> 0.5314
   (realized 0.5417).
2. The sqrt-damped IPF kept the team's own rating inside the exposure, which
   pins the fixed point at sqrt(true_rating): noiseless true attacks
   1.40/0.60 converged to 1.1368/0.8632, stationary but wrong (the audit's
   |@25-@200| check only proved stationarity). The exposure now divides the
   rating out first (true damped Gamma-Poisson MAP): recovery error drops 3x
   (0.0405 vs 0.1124 MAE on an identifiable 12-team noiseless problem).
3. league_rate is re-centred each iteration by
   (total_goals/total_expected)**0.5 so expected totals stay level under the
   symmetric factor. Residual NHL totals bias +0.130 decomposes as
   regulation-window drift (+0.20) minus the Poisson draw share
   underestimating actual OT incidence (15.7% vs 22.3%, -0.07) — deferred,
   not gate-relevant.

Priors are pseudo-games (k * league_rate). Default prior_weight 2.0 -> 4.0,
chosen by a {2,4,8,12,16} sweep on the train split of all 8 seeded lanes
(global train-total minimum) with per-lane overrides where the train
improvement is decisive: NHL 16 (hockey's weak per-game signal; holdout
0.2495 vs 0.2546 at the default) and CFB 2 (blowout-heavy high signal;
holdout 0.1521 vs 0.1586). hf cap 1.25 -> 1.35 (CFB/esp/ger pin at 1.25).
Every shipped choice was tuned on the earlier chronological split and
evaluated once on the untouched holdout.

Result across the 8 seeded lanes (full | blind holdout Brier):
  nhl  0.2506/0.2509 -> 0.2484/0.2495    nfl  0.2382/0.2264 -> 0.2345/0.2234
  cfb  0.1682/0.1607 -> 0.1583/0.1521    eng.1 0.2026/0.2160 -> 0.1997/0.2138
  esp.1 0.2037/0.1955 -> 0.1972/0.1872   ger.1 0.2055/0.2012 -> 0.2015/0.1920
  ita.1 0.2038/0.1988 -> 0.2001/0.1942   fra.1 0.2040/0.2074 -> 0.2005/0.2037
All 8 lanes improve on both splits.

Calibration gate fix (audit catch): CalibrationBook pooled home+away rows,
so base_rate_brier was ALWAYS 0.2500 for two-sided moneyline (0.2222 for
1X2) — vacuous, and at Brier 0.2494 < 0.2500 it would have unlocked NHL on
a model worse than blindly predicting the 54.2% realized home rate. The
baseline is now the per-selection class-frequency predictor (the strongest
prediction using no team information); rows without a selection label fall
back to the pooled rate. Outcome on the shipped model: NHL is blocked
(0.2484 vs realized-home-rate baseline 0.2483 — razor thin, errs locked);
all other lanes still pass (NFL 0.2345 < 0.2489, CFB 0.1583 < 0.1777,
soccer 0.1972-0.2015 < 0.2140-0.2205). This is a tightening, not a
relaxation; DEFAULT_MIN_SAMPLES, Brier caps, cap_vs_base_rate and EV gates
are untouched.

SOURCE bumps (probabilities changed; calibration segments must not mix
eras): nhl_poisson_v1 -> v2, dixon_coles_v1 -> v2, empirical_score_v1 ->
v2, espn_fpi_v1 -> v2 (FPI spreads/totals flow through the fitted sigmas).
seed_walk_forward_ml idempotence is per (game_id, source) so bumps re-seed
history under the new tag instead of silently seeding zero rows; verified
on a warehouse copy: cold seed 38.4s / 14,591 rows, warm re-run 0.06s / 0
rows, 24 live v1 rows preserved verbatim.

pytest 175 pass, ruff clean, replay --parity exit 0.
---
 scripts/backfill.py              |  53 ++++++++++------
 src/bullion/calibration.py       |  40 +++++++++++-
 src/bullion/engines/__init__.py  | 102 +++++++++++++++++++++++--------
 src/bullion/engines/football.py  |  22 +++++--
 src/bullion/engines/nhl.py       |   9 ++-
 src/bullion/engines/soccer.py    |   2 +-
 src/bullion/runner.py            |   1 +
 tests/test_daily_script.py       |   2 +-
 tests/test_engines.py            |  59 ++++++++++++++++--
 tests/test_espn.py               |   2 +-
 tests/test_http_and_backfill.py  |   6 +-
 tests/test_runner.py             |   2 +-
 tests/test_store.py              |   2 +-
 tests/test_tickets_and_replay.py |   6 +-
 tests/test_walk_forward_seed.py  |  40 ++++++++++--
 15 files changed, 276 insertions(+), 72 deletions(-)

diff --git a/scripts/backfill.py b/scripts/backfill.py
index fd02a90..df2aaca 100644
--- a/scripts/backfill.py
+++ b/scripts/backfill.py
@@ -14,8 +14,8 @@ lane does — they need a book quote or a provider snapshot no honest history
 can supply.
 
 The one deliberate exception is the model's own moneyline surface:
-``seed_walk_forward_ml`` replays ``empirical_score_v1`` / ``nhl_poisson_v1`` /
-``dixon_coles_v1`` over the backfilled finals, fitting strictly on finals whose
+``seed_walk_forward_ml`` replays ``empirical_score_v2`` / ``nhl_poisson_v2`` /
+``dixon_coles_v2`` over the backfilled finals, fitting strictly on finals whose
 SAST day precedes the forecast day, and persists those out-of-sample
 probabilities as ``provenance='walk_forward_seed'`` rows. They carry no price,
 no book, and no provider information — and every honesty gate
@@ -45,6 +45,7 @@ from bullion.espn import (  # noqa: E402
     season_events_url,
 )
 from bullion.engines import (  # noqa: E402
+    DEFAULT_PRIOR_WEIGHT,
     Result,
     fit_ratings,
     normal_cdf,
@@ -53,6 +54,7 @@ from bullion.engines import (  # noqa: E402
 )
 from bullion.engines import nhl as nhl_engine  # noqa: E402
 from bullion.engines import soccer as soccer_engine  # noqa: E402
+from bullion.engines import football as football_engine  # noqa: E402
 from bullion.models import LEAGUES, Sport, forecast_id_for, now_utc, sast_day  # noqa: E402
 from bullion.store import Store  # noqa: E402
 
@@ -341,7 +343,9 @@ def _score_matchday_moneyline(
         return scored
 
     if lane == "nhl":
-        ratings = fit_ratings(results, reference_day=day)
+        ratings = fit_ratings(
+            results, reference_day=day, prior_weight=nhl_engine.PRIOR_WEIGHT
+        )
         if not ratings.get("n"):
             return scored
         for game in games:
@@ -360,9 +364,13 @@ def _score_matchday_moneyline(
             scored.append((game["game_id"], "away", prob_away))
         return scored
 
-    # nfl / cfb: empirical_score_v1 (the FPI surface needs a live predictor
+    # nfl / cfb: empirical_score_v2 (the FPI surface needs a live predictor
     # snapshot, which a historical final can never have — it stays live-only).
-    ratings = fit_ratings(results, reference_day=day)
+    ratings = fit_ratings(
+        results,
+        reference_day=day,
+        prior_weight=football_engine.PRIOR_WEIGHTS.get(lane, DEFAULT_PRIOR_WEIGHT),
+    )
     if not ratings.get("n"):
         return scored
     from bullion.engines.football import SIGMA_PRIORS, residual_sigmas
@@ -400,9 +408,11 @@ def seed_walk_forward_ml(
     snapshot (recorded first, in :meth:`daily` and :meth:`rehydrate_forecasts`)
     can never be displaced by a seeded row.
 
-    Idempotence is checked per ``game_id``, not per day: warm ticks run one
-    ``SELECT DISTINCT`` per lane, find every historical matchday already
-    covered, and pay for zero fits. No gate constant is read or modified here —
+    Idempotence is checked per ``(game_id, source)``, not per day: warm ticks
+    run one ``SELECT DISTINCT`` per lane, find every historical matchday the
+    current source already covered, and pay for zero fits. A SOURCE version
+    bump re-seeds history under the new tag instead of silently seeding zero
+    rows. No gate constant is read or modified here —
     the seeder only supplies observations; calibration and the ticket gates
     re-measure them every run and fail closed.
     """
@@ -428,11 +438,23 @@ def seed_walk_forward_ml(
         ).fetchall()
         if not finals:
             continue
+        # The lane's moneyline model tag: matches what the daily engines emit.
+        source = (
+            soccer_engine.SOURCE
+            if lane in SOCCER_SEED_LANES
+            else nhl_engine.SOURCE
+            if lane == "nhl"
+            else football_engine.EMPIRICAL_SOURCE
+        )
+        # Idempotence is per (game_id, source): a SOURCE version bump must be
+        # able to seed its own rows for history the old source already covered,
+        # while a live row for the CURRENT source still excludes the whole game
+        # (INSERT OR IGNORE keeps the live row authoritative either way).
         have = {
-            row[0]
+            (row[0], row[1])
             for row in store.conn.execute(
                 """
-                SELECT DISTINCT game_id FROM forecast_snapshots
+                SELECT DISTINCT game_id, source FROM forecast_snapshots
                 WHERE league=? AND market='moneyline'
                 """,
                 (lane,),
@@ -449,7 +471,9 @@ def seed_walk_forward_ml(
             if len(prior_results) < MIN_SEED_HISTORY:
                 lane_stats["matchdays_below_min_history"] += 1
             else:
-                missing = [game for game in games if game["game_id"] not in have]
+                missing = [
+                    game for game in games if (game["game_id"], source) not in have
+                ]
                 if missing:
                     fit_started = time.monotonic()
                     scored = _score_matchday_moneyline(lane, day, prior_results, missing)
@@ -458,13 +482,6 @@ def seed_walk_forward_ml(
                     for game_id, selection, probability in scored:
                         market = "moneyline"
                         period = "full_game"
-                        source = (
-                            soccer_engine.SOURCE
-                            if lane in SOCCER_SEED_LANES
-                            else nhl_engine.SOURCE
-                            if lane == "nhl"
-                            else "empirical_score_v1"
-                        )
                         seeded_rows.append(
                             {
                                 "forecast_id": forecast_id_for(
diff --git a/src/bullion/calibration.py b/src/bullion/calibration.py
index 257b1bb..7f78929 100644
--- a/src/bullion/calibration.py
+++ b/src/bullion/calibration.py
@@ -111,6 +111,38 @@ class Observation:
     outcome: Optional[int]  # 1 win, 0 loss, None push/void
     result_day: str  # SAST day the result became known
     game_id: str = ""  # fixture the row came from, for per-game weighting
+    selection: str = ""  # row's class ("home"/"draw"/"away", "over"/"under", ...)
+
+
+def _class_frequency_baseline(
+    rows: List[Observation], weights: List[float]
+) -> Optional[float]:
+    """Brier of always predicting each selection's realized frequency.
+
+    The pooled base rate is vacuous for two-sided moneyline (home+away rows
+    pool to rate exactly 0.5 -> baseline 0.2500) and near-vacuous for 1X2
+    (rate exactly 1/3 -> baseline 0.2222): every game contributes one win, so
+    the pooled rate carries no class information and any model with residual
+    signal passes ``not_better_than_base_rate`` even when it is worse than
+    blindly predicting the realized home/draw/away frequencies. This baseline
+    scores the strongest prediction that uses no team information — each
+    selection graded at its realized win frequency — so a model must beat the
+    actual class base rates, not coin-flipping odds. For balanced two- and
+    three-class moneyline segments it is equal or tighter than the pooled
+    baseline. Rows without a selection label (legacy callers, tests) fall
+    back to the pooled base rate.
+    """
+    if not rows or any(not row.selection for row in rows):
+        return None
+    wins: Dict[str, int] = {}
+    total: Dict[str, int] = {}
+    for row in rows:
+        total[row.selection] = total.get(row.selection, 0) + 1
+        wins[row.selection] = wins.get(row.selection, 0) + (1 if row.outcome == 1 else 0)
+    baseline_pairs = [
+        (wins[row.selection] / total[row.selection], int(row.outcome)) for row in rows
+    ]
+    return brier_score(baseline_pairs, weights)
 
 
 @dataclass
@@ -176,10 +208,12 @@ class CalibrationBook:
         outcome: Optional[int],
         result_day: str,
         game_id: str = "",
+        selection: str = "",
     ) -> None:
         self._rows.setdefault(tuple(segment), []).append(
             Observation(
-                tuple(segment), float(probability), outcome, str(result_day), str(game_id or "")
+                tuple(segment), float(probability), outcome, str(result_day),
+                str(game_id or ""), str(selection or ""),
             )
         )
 
@@ -191,6 +225,7 @@ class CalibrationBook:
                 observation.outcome,
                 observation.result_day,
                 observation.game_id,
+                observation.selection,
             )
 
     def settled(self, segment: Tuple[str, str, str, str], as_of_day: str) -> List[Observation]:
@@ -229,7 +264,8 @@ class CalibrationBook:
             n=len(pairs),
             n_games=n_games,
             brier=brier_score(pairs, weights),
-            baseline_brier=base_rate_brier(pairs, weights),
+            baseline_brier=_class_frequency_baseline(rows, weights)
+            or base_rate_brier(pairs, weights),
             brier_cap=cap,
             min_samples=min_samples,
         )
diff --git a/src/bullion/engines/__init__.py b/src/bullion/engines/__init__.py
index 9790e58..08618fc 100644
--- a/src/bullion/engines/__init__.py
+++ b/src/bullion/engines/__init__.py
@@ -8,6 +8,11 @@ from datetime import date
 from typing import Dict, List, Sequence, Tuple
 
 MAX_GOALS = 10
+#: Global shrinkage prior (pseudo-games) for :func:`fit_ratings`, chosen by a
+#: {2,4,8,12,16} sweep on the train split of all 8 seeded lanes and confirmed
+#: on the untouched holdout. Per-lane overrides live in the engines that need
+#: one (see ``nhl.PRIOR_WEIGHT`` and ``football.PRIOR_WEIGHTS``).
+DEFAULT_PRIOR_WEIGHT = 4.0
 
 
 @dataclass(frozen=True)
@@ -72,14 +77,35 @@ def fit_ratings(
     half_life_days: float = 120.0,
     iterations: int = 25,
     home_advantage_prior: float = 1.10,
-    prior_weight: float = 2.0,
+    prior_weight: float = 4.0,
 ) -> Dict[str, object]:
     """Regularised attack/defence ratings via weighted iterative fitting.
 
     Returns ``{"league_rate", "home_factor", "attack", "defence", "n"}`` where
-    ``league_rate`` is the average score per team per game. Ratings are pulled
-    toward 1.0 by ``prior_weight`` pseudo-games, so a team with two games does
-    not get a 3.0 attack rating.
+    ``league_rate`` is the scoring level and ratings are pulled toward 1.0 by
+    ``prior_weight`` PSEUDO-GAMES (``prior_weight * league_rate`` goals), so a
+    team with two games does not get a 3.0 attack rating.
+
+    Three invariants fixed here (measured on the 2026-11-27 warehouse audit,
+    ``BULLION_AUDIT_AND_NHL_FIX.md``, corrective pass):
+
+    * Symmetric home advantage: ``lam_home = lr*att*def*s`` and
+      ``lam_away = lr*att*def/s`` with ``s = sqrt(home_factor)``. The old
+      asymmetric form applied the factor to home scoring only, which forced
+      the factor to converge to ``sqrt(true)`` (real NHL: 1.0699 fitted vs
+      1.15 true) and inflated expected totals by ~0.2 goals/game. The factor
+      is now identified by the HOME:AWAY ratio of observed vs factor-free
+      expectation, so it converges to the true advantage (1.1502 on the same
+      synthetic check) and totals stay level.
+    * TRUE damped Gamma-Poisson MAP: the team's own rating is divided out of
+      the exposure BEFORE the ratio. The old ratio kept the rating inside the
+      exposure, which pinned the fixed point at ``sqrt(true_rating)`` —
+      noiseless data with true attacks 1.40/0.60 converged to 1.1368/0.8632
+      (stationary, but wrong). With the exposure divided out, the same damped
+      step converges to the true rating (3x lower recovery error).
+    * ``league_rate`` is re-centred each iteration by
+      ``(total_goals / total_expected) ** 0.5`` so expected totals match the
+      observed scoring level despite the symmetric factor.
     """
     rows = [row for row in results if row.day < reference_day]
     teams: set[str] = set()
@@ -103,16 +129,23 @@ def fit_ratings(
     attack = {team: 1.0 for team in teams}
     defence = {team: 1.0 for team in teams}
     home_factor = home_advantage_prior
+    # prior_weight pseudo-games at the league's own scoring rate, in goals.
+    prior_goals = prior_weight * league_rate
     for _ in range(iterations):
+        scale = math.sqrt(home_factor)
         scored: Dict[str, float] = {team: 0.0 for team in teams}
         conceded: Dict[str, float] = {team: 0.0 for team in teams}
         expected_scored: Dict[str, float] = {team: 0.0 for team in teams}
         expected_conceded: Dict[str, float] = {team: 0.0 for team in teams}
         home_goals = 0.0
-        home_expected = 0.0
+        away_goals = 0.0
+        home_expected_base = 0.0
+        away_expected_base = 0.0
+        total_goals = 0.0
+        total_expected = 0.0
         for weight, row in zip(weights, rows):
-            lam_home = league_rate * attack[row.home_id] * defence[row.away_id] * home_factor
-            lam_away = league_rate * attack[row.away_id] * defence[row.home_id]
+            lam_home = league_rate * attack[row.home_id] * defence[row.away_id] * scale
+            lam_away = league_rate * attack[row.away_id] * defence[row.home_id] / scale
             scored[row.home_id] += weight * row.home_score
             conceded[row.away_id] += weight * row.home_score
             scored[row.away_id] += weight * row.away_score
@@ -122,18 +155,23 @@ def fit_ratings(
             expected_conceded[row.away_id] += weight * lam_home
             expected_conceded[row.home_id] += weight * lam_away
             home_goals += weight * row.home_score
-            home_expected += weight * lam_home
+            away_goals += weight * row.away_score
+            total_goals += weight * (row.home_score + row.away_score)
+            total_expected += weight * (lam_home + lam_away)
+            # Factor-free expectations: divide each side's own scale out.
+            home_expected_base += weight * lam_home / scale
+            away_expected_base += weight * lam_away * scale
         for team in teams:
-            # Damped multiplicative updates with hard bounds. Full-weight IPF
-            # diverges on an imbalanced schedule (a team that only ever plays
-            # one side of the fixture list is not identifiable), and a diverged
-            # rating is worse than an uninformative one.
-            target_attack = (scored[team] + prior_weight) / (
-                expected_scored[team] + prior_weight
-            )
-            target_defence = (conceded[team] + prior_weight) / (
-                expected_conceded[team] + prior_weight
-            )
+            # TRUE damped MAP with hard bounds: divide the team's own rating
+            # out of the exposure first, so the damped step
+            # ``a * (t/a)**0.5 = sqrt(a*t)`` has its fixed point at the true
+            # rating instead of its square root. The damping itself is kept:
+            # full-weight IPF diverges on an imbalanced schedule, and a
+            # diverged rating is worse than an uninformative one.
+            exp_scored = expected_scored[team] / attack[team]
+            exp_conceded = expected_conceded[team] / defence[team]
+            target_attack = (scored[team] + prior_goals) / (exp_scored + prior_goals)
+            target_defence = (conceded[team] + prior_goals) / (exp_conceded + prior_goals)
             attack[team] = _clamp(
                 attack[team] * max(target_attack / attack[team], 1e-9) ** 0.5, 0.50, 2.00
             )
@@ -146,8 +184,18 @@ def fit_ratings(
         for team in teams:
             attack[team] /= mean_attack
             defence[team] /= mean_defence
-        if home_expected > 0:
-            home_factor = max(1.0, min(1.25, home_goals / home_expected))
+        # Re-centre the scoring level so E[total] matches the observed total
+        # despite the (s + 1/s) asymmetry of the symmetric home factor.
+        if total_expected > 0:
+            league_rate *= (total_goals / total_expected) ** 0.5
+        # Home factor from the home:away ratio of observed vs factor-free
+        # expectation — identified by the ratio, so it converges to the true
+        # home advantage instead of its square root.
+        if home_expected_base > 0 and away_expected_base > 0:
+            raw_hf = (home_goals / home_expected_base) / (away_goals / away_expected_base)
+            home_factor = _clamp(
+                home_factor * max(raw_hf / home_factor, 1e-9) ** 0.5, 1.0, 1.35
+            )
     return {
         "league_rate": league_rate,
         "home_factor": home_factor,
@@ -164,15 +212,19 @@ def _clamp(value: float, low: float, high: float) -> float:
 def rates_for(
     ratings: Dict[str, object], home_id: str, away_id: str, *, neutral: bool = False
 ) -> Tuple[float, float]:
-    """Home/away scoring rates, using a neutral-site home factor of 1.0."""
+    """Home/away scoring rates under the symmetric home-advantage model.
+
+    The home factor is applied as ``sqrt(hf)`` to the home side and divided
+    out of the away side, mirroring exactly how ``fit_ratings`` saw the
+    training data. A neutral site uses a factor of 1.0.
+    """
     league_rate = float(ratings.get("league_rate") or 0.0)
     attack = ratings.get("attack") or {}
     defence = ratings.get("defence") or {}
     home_factor = float(ratings.get("home_factor") or 1.0)
-    if neutral:
-        home_factor = 1.0
-    lam_home = league_rate * attack.get(home_id, 1.0) * defence.get(away_id, 1.0) * home_factor
-    lam_away = league_rate * attack.get(away_id, 1.0) * defence.get(home_id, 1.0)
+    scale = 1.0 if neutral else math.sqrt(max(home_factor, 1e-6))
+    lam_home = league_rate * attack.get(home_id, 1.0) * defence.get(away_id, 1.0) * scale
+    lam_away = league_rate * attack.get(away_id, 1.0) * defence.get(home_id, 1.0) / scale
     return max(lam_home, 0.05), max(lam_away, 0.05)
 
 
diff --git a/src/bullion/engines/football.py b/src/bullion/engines/football.py
index ba1ed75..ea81674 100644
--- a/src/bullion/engines/football.py
+++ b/src/bullion/engines/football.py
@@ -23,6 +23,7 @@ from typing import Dict, List, Optional, Sequence, Tuple
 from ..espn import PredictorEvidence
 from ..models import BookQuote, Forecast, Game
 from . import (
+    DEFAULT_PRIOR_WEIGHT,
     Result,
     decay_weight,
     fit_ratings,
@@ -33,8 +34,13 @@ from . import (
     rest_adjustment,
 )
 
-FPI_SOURCE = "espn_fpi_v1"
-EMPIRICAL_SOURCE = "empirical_score_v1"
+FPI_SOURCE = "espn_fpi_v2"  # v2: fitted sigmas change under the fixed fit_ratings
+EMPIRICAL_SOURCE = "empirical_score_v2"  # v2: symmetric fit_ratings (MAP + hf ratio fix)
+#: Per-lane shrinkage priors (pseudo-games) from the train-split sweep. CFB's
+#: blowout-heavy, high-signal scoring wants light shrinkage (2 pseudo-games;
+#: train Brier 0.1618 vs 0.1706 at the default 4, holdout 0.1521 vs 0.1586);
+#: every other football lane stays on ``DEFAULT_PRIOR_WEIGHT``.
+PRIOR_WEIGHTS = {"cfb": 2.0}
 SIGMA_PRIORS = {"nfl": (13.5, 13.5), "cfb": (16.0, 14.5)}
 
 
@@ -113,14 +119,22 @@ def forecast(
         if not results:
             sigma_margin, sigma_total = SIGMA_PRIORS.get(game.league, (14.0, 14.0))
         else:
-            ratings = fit_ratings(results, reference_day=reference_day)
+            ratings = fit_ratings(
+                results,
+                reference_day=reference_day,
+                prior_weight=PRIOR_WEIGHTS.get(game.league, DEFAULT_PRIOR_WEIGHT),
+            )
             sigma_margin, sigma_total = _sigmas(
                 game.league, results, ratings, reference_day
             )
     else:
         if not results:
             return []
-        ratings = fit_ratings(results, reference_day=reference_day)
+        ratings = fit_ratings(
+            results,
+            reference_day=reference_day,
+            prior_weight=PRIOR_WEIGHTS.get(game.league, DEFAULT_PRIOR_WEIGHT),
+        )
         if not ratings.get("n"):
             return []
         home_rate, away_rate = rates_for(
diff --git a/src/bullion/engines/nhl.py b/src/bullion/engines/nhl.py
index eda245c..f48d35a 100644
--- a/src/bullion/engines/nhl.py
+++ b/src/bullion/engines/nhl.py
@@ -28,7 +28,12 @@ from . import (
     rest_adjustment,
 )
 
-SOURCE = "nhl_poisson_v1"
+SOURCE = "nhl_poisson_v2"  # v2: symmetric fit_ratings (MAP + home-factor ratio fix)
+#: Shrinkage prior (pseudo-games) for the NHL fit, chosen by the {2,4,8,12,16}
+#: sweep on the NHL train split only and confirmed on the untouched holdout
+#: (Brier 0.2495 at 16 vs 0.2546 at the global default 4). Hockey's weak
+#: per-game signal needs much heavier shrinkage than the other lanes.
+PRIOR_WEIGHT = 16.0
 OT_HOME_WIN_SHARE = 0.52  # 3-on-3 + shootout edge for the home side (prior)
 #: Extra goals beyond regulation, for the games that reach OT/SO. NHL overtime is
 #: sudden death and a shootout winner is credited exactly one goal, so **every**
@@ -94,7 +99,7 @@ def forecast(
     reference_day: str,
 ) -> List[Forecast]:
     """Model probabilities for every quoted NHL line. Prices are attached later."""
-    ratings = fit_ratings(results, reference_day=reference_day)
+    ratings = fit_ratings(results, reference_day=reference_day, prior_weight=PRIOR_WEIGHT)
     if not ratings.get("n"):
         return []
     home_rate, away_rate = rates_for(ratings, game.home_team_id, game.away_team_id,
diff --git a/src/bullion/engines/soccer.py b/src/bullion/engines/soccer.py
index 632a079..2c14f05 100644
--- a/src/bullion/engines/soccer.py
+++ b/src/bullion/engines/soccer.py
@@ -29,7 +29,7 @@ from . import (
     rates_for,
 )
 
-SOURCE = "dixon_coles_v1"
+SOURCE = "dixon_coles_v2"  # v2: fit_ratings pseudo-game prior + home-factor fix
 RHO_GRID_MIN = -0.18
 RHO_GRID_MAX = 0.18
 RHO_GRID_STEP = 0.01
diff --git a/src/bullion/runner.py b/src/bullion/runner.py
index 73c8291..b6bf919 100644
--- a/src/bullion/runner.py
+++ b/src/bullion/runner.py
@@ -139,6 +139,7 @@ def _calibration_book(store: Store, league: League, as_of_utc: str) -> Calibrati
             None if outcome == "push" else (1 if outcome == "win" else 0),
             result_day,
             str(row["game_id"]),
+            selection=str(row["selection"]),
         )
     return book
 
diff --git a/tests/test_daily_script.py b/tests/test_daily_script.py
index 98eb42f..9ca3e19 100644
--- a/tests/test_daily_script.py
+++ b/tests/test_daily_script.py
@@ -81,7 +81,7 @@ def test_every_daily_run_rehydrates_a_warehouse_that_already_exists(daily, tmp_p
                         "league": "nhl",
                         "market": "moneyline",
                         "period": "full_game",
-                        "source": "nhl_poisson_v1",
+                        "source": "nhl_poisson_v2",
                         "selection": "home",
                         "line": None,
                         "raw_probability": 0.62,
diff --git a/tests/test_engines.py b/tests/test_engines.py
index 72f7d60..2c8d9f0 100644
--- a/tests/test_engines.py
+++ b/tests/test_engines.py
@@ -233,7 +233,7 @@ def test_soccer_totals_have_half_line_and_integer_behaviour():
 
 def test_calibration_excludes_same_day_and_future_results():
     book = CalibrationBook()
-    segment = ("nhl", "moneyline", "full_game", "nhl_poisson_v1")
+    segment = ("nhl", "moneyline", "full_game", "nhl_poisson_v2")
     for index in range(40):
         book.add(segment, 0.6, 1, "2026-11-01", f"history{index}")
     book.add(segment, 0.6, 0, "2026-11-28", "same-day")  # the scoring day itself
@@ -246,7 +246,7 @@ def test_calibration_excludes_same_day_and_future_results():
 
 def test_calibration_requires_thirty_samples():
     book = CalibrationBook()
-    segment = ("nhl", "total", "full_game", "nhl_poisson_v1")
+    segment = ("nhl", "total", "full_game", "nhl_poisson_v2")
     for index in range(29):
         book.add(segment, 0.5, 1, "2026-11-01", f"history{index}")
     report = book.report(segment, "2026-11-28")
@@ -257,7 +257,7 @@ def test_calibration_requires_thirty_samples():
 
 def test_calibration_brier_cap_rejects_a_bad_segment():
     book = CalibrationBook()
-    segment = ("eng.1", "moneyline", "full_game", "dixon_coles_v1")
+    segment = ("eng.1", "moneyline", "full_game", "dixon_coles_v2")
     for index in range(40):
         book.add(segment, 0.9, 1 if index % 5 == 0 else 0, "2026-11-01", f"history{index}")
     report = book.report(segment, "2026-11-28", brier_cap=0.22, cap_vs_base_rate=True)
@@ -282,7 +282,7 @@ def test_calibration_fails_closed_when_a_sample_has_no_fixture_id():
     restore exactly the inflation the per-game gate exists to prevent.
     """
     book = CalibrationBook()
-    segment = ("nhl", "moneyline", "full_game", "nhl_poisson_v1")
+    segment = ("nhl", "moneyline", "full_game", "nhl_poisson_v2")
     for index in range(40):
         book.add(segment, 0.62, 1 if index < 25 else 0, "2026-11-01")
     report = book.report(segment, "2026-11-28")
@@ -295,7 +295,7 @@ def test_calibration_fails_closed_when_a_sample_has_no_fixture_id():
 
 def test_calibration_counts_unique_games_not_correlated_sides():
     """A fixture with three quoted sides is one observation, not three."""
-    segment = ("nfl", "moneyline", "full_game", "espn_fpi_v1")
+    segment = ("nfl", "moneyline", "full_game", "espn_fpi_v2")
     book = CalibrationBook()
     for index in range(10):
         for side, probability, outcome in (("home", 0.6, 1), ("away", 0.35, 0), ("draw", 0.05, 0)):
@@ -317,7 +317,7 @@ def test_calibration_counts_unique_games_not_correlated_sides():
 
 def test_calibration_weights_every_fixture_equally():
     """One row in game A must weigh as much as three rows in game B."""
-    segment = ("nhl", "moneyline", "full_game", "nhl_poisson_v1")
+    segment = ("nhl", "moneyline", "full_game", "nhl_poisson_v2")
     book = CalibrationBook()
     book.add(segment, 1.0, 1, "2026-11-01", "quiet")  # single-row fixture, perfect
     for _ in range(3):
@@ -436,3 +436,50 @@ def test_surface_tiebreak_is_deterministic_when_briers_are_indistinguishable():
         football_engine.FPI_SOURCE
     )
     assert football_engine.tiebreak_surface({}) is None
+
+
+def test_class_frequency_baseline_blocks_pooled_baseline_vacuity():
+    """Regression for the audit's vacuous-base-rate catch.
+
+    The pooled base rate is always 0.2500 for two-sided moneyline (home+away
+    rows pool to rate 0.5), so a model WORSE than blindly predicting the
+    realized home-win rate could pass ``not_better_than_base_rate``. The
+    tightened per-selection class-frequency baseline blocks that: the model
+    below (constant 0.55 with a 70% home league, Brier 0.2325) passes the old
+    pooled gate but fails the tightened one (baseline 0.21).
+    """
+    book = CalibrationBook()
+    segment = ("nhl", "moneyline", "full_game", "nhl_poisson_v2")
+    for index in range(40):
+        home_won = 1 if index % 10 < 7 else 0
+        book.add(segment, 0.55, home_won, "2026-11-01", f"g{index}", selection="home")
+        book.add(segment, 0.55, 1 - home_won, "2026-11-01", f"g{index}", selection="away")
+    report = book.report(segment, "2026-11-28", cap_vs_base_rate=True)
+    assert report.n_games == 40
+    # Tightened baseline = p(1-p) with p = 0.7 realized home rate.
+    assert report.baseline_brier == pytest.approx(0.7 * 0.3, abs=1e-9)
+    assert not report.sufficient
+    assert "not_better_than_base_rate" in report.reasons
+
+
+def test_class_frequency_baseline_admits_a_real_signal():
+    book = CalibrationBook()
+    segment = ("nhl", "moneyline", "full_game", "nhl_poisson_v2")
+    for index in range(40):
+        home_won = 1 if index % 10 < 7 else 0
+        p_home = 0.9 if home_won else 0.5
+        book.add(segment, p_home, home_won, "2026-11-01", f"g{index}", selection="home")
+        book.add(segment, 1 - p_home, 1 - home_won, "2026-11-01", f"g{index}", selection="away")
+    report = book.report(segment, "2026-11-28", cap_vs_base_rate=True)
+    assert report.sufficient, report.reasons
+    assert report.brier < report.baseline_brier
+
+
+def test_class_frequency_baseline_falls_back_without_selection_labels():
+    """Rows fed without a selection keep the pooled base rate (back-compat)."""
+    book = CalibrationBook()
+    segment = ("nhl", "moneyline", "full_game", "nhl_poisson_v2")
+    for index in range(40):
+        book.add(segment, 0.5, index % 2, "2026-11-01", f"g{index}")
+    report = book.report(segment, "2026-11-28", cap_vs_base_rate=True)
+    assert report.baseline_brier == pytest.approx(0.25, abs=1e-9)
diff --git a/tests/test_espn.py b/tests/test_espn.py
index 253216d..b8bca53 100644
--- a/tests/test_espn.py
+++ b/tests/test_espn.py
@@ -110,7 +110,7 @@ def test_post_tip_quote_raises_at_the_ticket_boundary():
         market="moneyline",
         selection="home",
         raw_probability=0.55,
-        source="nhl_poisson_v1",
+        source="nhl_poisson_v2",
     )
     quote = BookQuote(
         game_id="401",
diff --git a/tests/test_http_and_backfill.py b/tests/test_http_and_backfill.py
index b0cb065..01e2f00 100644
--- a/tests/test_http_and_backfill.py
+++ b/tests/test_http_and_backfill.py
@@ -397,7 +397,7 @@ def _frozen_pool_row(forecast_id: str, game_id: str, *, raw: float = 0.62) -> di
         "league": "nhl",
         "market": "moneyline",
         "period": "full_game",
-        "source": "nhl_poisson_v1",
+        "source": "nhl_poisson_v2",
         "selection": "home",
         "line": None,
         "raw_probability": raw,
@@ -470,7 +470,7 @@ def test_a_warm_cache_restores_history_without_duplicating_it(backfill, tmp_path
     from bullion.runner import _calibration_book
 
     league = LEAGUES["nhl"]
-    segment = ("nhl", "moneyline", "full_game", "nhl_poisson_v1")
+    segment = ("nhl", "moneyline", "full_game", "nhl_poisson_v2")
     localdata = tmp_path / "localdata"
     lane = localdata / "nhl"
     lane.mkdir(parents=True)
@@ -502,7 +502,7 @@ def test_a_warm_cache_restores_history_without_duplicating_it(backfill, tmp_path
                 market="moneyline",
                 selection="home",
                 raw_probability=0.62,
-                source="nhl_poisson_v1",
+                source="nhl_poisson_v2",
             )
         )
         pool.append(_frozen_pool_row(f"frozen-{game_id}", game_id))
diff --git a/tests/test_runner.py b/tests/test_runner.py
index 04d4c8b..29d2687 100644
--- a/tests/test_runner.py
+++ b/tests/test_runner.py
@@ -149,7 +149,7 @@ def _seed_history(store: Store) -> None:
             market="moneyline",
             selection="home",
             raw_probability=0.62,
-            source="nhl_poisson_v1",
+            source="nhl_poisson_v2",
         )
         store.record_forecasts([forecast], sast_day(game.scheduled_utc))
 
diff --git a/tests/test_store.py b/tests/test_store.py
index 2e7ec5a..b402917 100644
--- a/tests/test_store.py
+++ b/tests/test_store.py
@@ -86,7 +86,7 @@ def _frozen_row(game_id: str = "401") -> dict:
         "league": "nhl",
         "market": "moneyline",
         "period": "full_game",
-        "source": "nhl_poisson_v1",
+        "source": "nhl_poisson_v2",
         "selection": "home",
         "line": None,
         "raw_probability": 0.62,
diff --git a/tests/test_tickets_and_replay.py b/tests/test_tickets_and_replay.py
index 46f4271..559a983 100644
--- a/tests/test_tickets_and_replay.py
+++ b/tests/test_tickets_and_replay.py
@@ -32,8 +32,8 @@ from bullion.tickets import (
 )
 
 DAY = "2026-11-28"
-SEGMENT = ("nhl", "moneyline", "full_game", "nhl_poisson_v1")
-SOURCE = "nhl_poisson_v1"
+SEGMENT = ("nhl", "moneyline", "full_game", "nhl_poisson_v2")
+SOURCE = "nhl_poisson_v2"
 
 
 def _calibrated_report(n: int = 40, probability: float = 0.62, wins: int = 25):
@@ -537,7 +537,7 @@ def _three_way_pool(
     Brier that says nothing about the home/away prices being booked.
     """
     league = LEAGUES["eng.1"]
-    source = "dixon_coles_v1"
+    source = "dixon_coles_v2"
     segment = (league.key, "moneyline", "full_game", source)
     book = CalibrationBook()
     for index in range(40):
diff --git a/tests/test_walk_forward_seed.py b/tests/test_walk_forward_seed.py
index f5bdeb0..485e96d 100644
--- a/tests/test_walk_forward_seed.py
+++ b/tests/test_walk_forward_seed.py
@@ -121,7 +121,7 @@ def test_seed_produces_provenanced_moneyline_rows_and_is_idempotent(backfill, tm
         rows = [dict(row) for row in _seeded_rows(store)]
         assert len(rows) == 6
         assert {row["market"] for row in rows} == {"moneyline"}
-        assert {row["source"] for row in rows} == {"empirical_score_v1"}
+        assert {row["source"] for row in rows} == {"empirical_score_v2"}
         assert {row["observed_day"] for row in rows} == {"2026-10-04"}
         for row in rows:
             assert row["provenance"] == "walk_forward_seed"
@@ -152,7 +152,7 @@ def test_seed_never_displaces_a_live_pre_tip_snapshot(backfill, tmp_path):
                     market="moneyline",
                     selection="home",
                     raw_probability=0.9,
-                    source="empirical_score_v1",
+                    source="empirical_score_v2",
                 )
             ],
             observed_day="2026-10-04",
@@ -283,10 +283,10 @@ def test_seed_covers_nhl_uniformly_and_soccer_three_way(backfill, tmp_path):
             if row["league"] == "eng.1"
         }
         assert nhl_rows, "NHL is seeded like every other lane — the gate, not the seeder, rejects it"
-        assert {row["source"] for row in nhl_rows.values()} == {"nhl_poisson_v1"}
+        assert {row["source"] for row in nhl_rows.values()} == {"nhl_poisson_v2"}
         assert {row["selection"] for row in nhl_rows.values()} == {"home", "away"}
         assert soccer_rows, "soccer rows exist"
-        assert {row["source"] for row in soccer_rows.values()} == {"dixon_coles_v1"}
+        assert {row["source"] for row in soccer_rows.values()} == {"dixon_coles_v2"}
         assert {row["selection"] for row in soccer_rows.values()} == {"home", "draw", "away"}
         assert summary["lanes"]["nhl"]["inserted"] == 2
         assert summary["lanes"]["eng.1"]["inserted"] == 3
@@ -311,3 +311,35 @@ def test_seed_skips_matchdays_without_enough_history(backfill, tmp_path):
         # Both matchdays (8 and 10 priors) sit below the 15-final floor.
         assert summary["lanes"]["cfb"]["matchdays_below_min_history"] == 2
         assert store.counts()["forecast_snapshots"] == 0
+
+
+def test_source_bump_reseeds_instead_of_silently_seeding_zero(backfill, tmp_path):
+    """Regression for the per-source idempotence check (audit bug 5).
+
+    A SOURCE version bump must seed its own rows for history the old source
+    already covered; the old rows stay as evidence under their old segment.
+    """
+    with Store(tmp_path / "s.sqlite") as store:
+        _seed_cfb_history(store)
+        first = backfill.seed_walk_forward_ml(store, leagues=("cfb",))
+        assert first["lanes"]["cfb"]["inserted"] == 6
+
+        # Warm re-run with the SAME source: pure no-op.
+        warm = backfill.seed_walk_forward_ml(store, leagues=("cfb",))
+        assert warm["lanes"]["cfb"]["inserted"] == 0, (
+            "same source + same history must be a no-op"
+        )
+        # Now move the existing rows to an OLD tag, the way a bumped engine
+        # leaves its predecessor's evidence behind: the current source no longer
+        # covers this history, so it must re-seed rather than stay silent.
+        store.conn.execute(
+            "UPDATE forecast_snapshots SET source='empirical_score_v0',"
+            " forecast_id='legacy-'||forecast_id WHERE provenance='walk_forward_seed'"
+        )
+        store.conn.commit()
+        bumped = backfill.seed_walk_forward_ml(store, leagues=("cfb",))
+        assert bumped["lanes"]["cfb"]["inserted"] == 6, (
+            "a SOURCE bump must re-seed history under the new tag"
+        )
+        counts = store.counts()["forecast_snapshots"]
+        assert counts == 12, "old-source evidence survives next to the new tag"
-- 
2.47.3

```
