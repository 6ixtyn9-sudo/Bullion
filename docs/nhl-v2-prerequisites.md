# Before any `nhl_poisson_v2`: three things this repo already knows

Findings from the 2026-11-27 engine review and the read-only reproductions that
followed. None of them is a bug in the diagnostic PR they shipped alongside;
all of them will silently waste a `v2` if they are not handled first.

## 1. A `SOURCE` bump alone seeds nothing

`scripts/backfill.py` decides which fixtures the walk-forward seeder still owes
by game id, per lane, **without looking at the model source**:

```python
have = {
    row[0]
    for row in store.conn.execute(
        "SELECT DISTINCT game_id FROM forecast_snapshots WHERE league=? AND market='moneyline'",
        (lane,),
    )
}
```

So after bumping `nhl_poisson_v2`, every historical fixture already has a
`v1` moneyline row, the covered set still contains all of them, and the seeder
writes **zero** `v2` rows. The lane then has live-only history, stays locked,
and the engine fix looks like it did nothing. `INSERT OR IGNORE` does not save
you here — the row is never even generated.

Required, in the same change as the bump: key the covered set on
`(game_id, source)` (or purge `v1` seeded rows, or evict the cache). Pin it
with a test that asserts a source bump re-seeds the full history.

## 2. Fixing the home factor alone makes Brier worse

Reproduced over 16 synthetic NHL-like leagues (paired, same seeds):

| variant | ΔBrier vs production | seeds improved |
|---|---|---|
| home factor fixed only | **+0.0013 ± 0.0004 (worse)** | 2/16 |
| home factor + `α=1` IPF + pseudo-game prior | **−0.0027 ± 0.0010** | 12/16 |

The shipped `fit_ratings` converges `attack` to `sqrt(true)` (noiseless fit:
true 1.400 → 1.1753, `sqrt(1.4) = 1.1832`; true 0.700 → 0.8396,
`sqrt(0.7) = 0.8367`). Applying a correct 1.10 home factor on top of
compressed ratings over-predicts the home side. The two fixes are one change,
not two.

## 3. The estimator bug is arithmetic, not tuning

* `home_factor = home_goals / home_expected`, where `home_expected` already
  contains `home_factor`, is the map `h <- K/h`: a two-cycle, clamped at 1.0
  from below, and `iterations=25` is odd. The fit exits at exactly `1.0000` —
  zero home advantage — and even the high phase only reaches ~1.02, not the
  ~1.10 the league actually shows.
* `attack * (target / attack) ** 0.5`, with `target ∝ 1/attack`, has the fixed
  point `sqrt(scored / C)`. The exponent is the estimator; `1.0` is the correct
  IPF step, and it needs pseudo-game shrinkage (`k * league_rate`, not `k` raw
  goals) in the same change or the map destabilises.

## What must be true before `v2` ships

1. `scripts/diagnose_engines.py` run against the real cached warehouse, with
   the NHL split tuned on the earlier portion only and the later portion
   reported untouched.
2. Holdout Brier **below 0.25 with skill at or above +0.010** — not +0.001
   hovering on noise. The synthetic ablation suggests the estimator fixes land
   around **−0.0027**, i.e. 0.2507 → ~0.2480, skill ≈ +0.008. That clears the
   gate but sits under the durability bar; shots on goal are the likely next
   lever if the real numbers land in the same place.
3. The `(game_id, source)` covered-set fix and its test, in the same PR.
4. No gate constant, cap or `min_samples` touched in that PR.
