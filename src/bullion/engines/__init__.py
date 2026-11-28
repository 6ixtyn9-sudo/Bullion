"""Shared, dependency-free numeric helpers for the three sport engines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Sequence, Tuple

MAX_GOALS = 10

# Engine ``SOURCE`` tags name a model FAMILY, not a revision. Seeding and
# calibration are keyed on ``(game_id, source)``, so a materially different
# fit must ship under its own tag (the seeder then re-seeds history under it);
# a refit of the same family keeps the tag and its accumulated evidence.

#: Global shrinkage prior (pseudo-games) for :func:`fit_ratings`, chosen by a
#: {2,4,8,12,16} sweep on the train split of all 8 seeded lanes and confirmed
#: on the untouched holdout. Per-lane overrides live in the engines that need
#: one (see ``nhl.PRIOR_WEIGHT`` and ``football.PRIOR_WEIGHTS``).
DEFAULT_PRIOR_WEIGHT = 4.0


@dataclass(frozen=True)
class Result:
    """One settled fixture used for fitting. Goals/points are official finals."""

    home_id: str
    away_id: str
    home_score: float
    away_score: float
    day: str  # SAST result day (used for walk-forward filtering)

    @property
    def total(self) -> float:
        return self.home_score + self.away_score

    @property
    def margin(self) -> float:
        return self.home_score - self.away_score


def quoted_lines(quotes: Sequence, market: str) -> List[float]:
    """Distinct pre-match lines quoted for one market (spread/total)."""
    return sorted(
        {
            round(float(quote.line), 3)
            for quote in quotes
            if quote.market == market and quote.line is not None and quote.pre_match
        }
    )


def poisson_pmf(lam: float, k: int) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * lam**k / math.factorial(k)


def poisson_matrix(home_rate: float, away_rate: float, max_goals: int = MAX_GOALS) -> List[List[float]]:
    """Joint score matrix under independent Poisson scoring."""
    home = [poisson_pmf(home_rate, k) for k in range(max_goals + 1)]
    away = [poisson_pmf(away_rate, k) for k in range(max_goals + 1)]
    return [[h * a for a in away] for h in home]


def total_days_between(earlier: str, later: str) -> int:
    return (date.fromisoformat(later) - date.fromisoformat(earlier)).days


def decay_weight(day: str, reference_day: str, half_life_days: float) -> float:
    """Exponential time decay; matches played on/after the reference weigh 1."""
    if half_life_days <= 0:
        return 1.0
    age = max(total_days_between(day, reference_day), 0)
    return 0.5 ** (age / half_life_days)


def fit_ratings(
    results: Sequence[Result],
    *,
    reference_day: str,
    half_life_days: float = 120.0,
    iterations: int = 25,
    home_advantage_prior: float = 1.10,
    prior_weight: float = 4.0,
) -> Dict[str, object]:
    """Regularised attack/defence ratings via weighted iterative fitting.

    Returns ``{"league_rate", "home_factor", "attack", "defence", "n"}`` where
    ``league_rate`` is the scoring level and ratings are pulled toward 1.0 by
    ``prior_weight`` PSEUDO-GAMES (``prior_weight * league_rate`` goals), so a
    team with two games does not get a 3.0 attack rating.

    Three invariants fixed here (measured on the 2026-11-27 warehouse audit,
    ``BULLION_AUDIT_AND_NHL_FIX.md``, corrective pass):

    * Symmetric home advantage: ``lam_home = lr*att*def*s`` and
      ``lam_away = lr*att*def/s`` with ``s = sqrt(home_factor)``. The old
      asymmetric form applied the factor to home scoring only, which forced
      the factor to converge to ``sqrt(true)`` (real NHL: 1.0699 fitted vs
      1.15 true) and inflated expected totals by ~0.2 goals/game. The factor
      is now identified by the HOME:AWAY ratio of observed vs factor-free
      expectation, so it converges to the true advantage (1.1502 on the same
      synthetic check) and totals stay level.
    * TRUE damped Gamma-Poisson MAP: the team's own rating is divided out of
      the exposure BEFORE the ratio. The old ratio kept the rating inside the
      exposure, which pinned the fixed point at ``sqrt(true_rating)`` —
      noiseless data with true attacks 1.40/0.60 converged to 1.1368/0.8632
      (stationary, but wrong). With the exposure divided out, the same damped
      step converges to the true rating (3x lower recovery error).
    * ``league_rate`` is re-centred each iteration by
      ``(total_goals / total_expected) ** 0.5`` so expected totals match the
      observed scoring level despite the symmetric factor.
    """
    rows = [row for row in results if row.day < reference_day]
    teams: set[str] = set()
    for row in rows:
        teams.update((row.home_id, row.away_id))
    if not rows:
        return {
            "league_rate": 0.0,
            "home_factor": home_advantage_prior,
            "attack": {},
            "defence": {},
            "n": 0,
        }
    weights = [decay_weight(row.day, reference_day, half_life_days) for row in rows]
    total_weight = sum(weights)
    league_rate = (
        sum(w * (row.home_score + row.away_score) / 2.0 for w, row in zip(weights, rows))
        / total_weight
    )
    league_rate = max(league_rate, 0.2)
    attack = {team: 1.0 for team in teams}
    defence = {team: 1.0 for team in teams}
    home_factor = home_advantage_prior
    # prior_weight pseudo-games at the league's own scoring rate, in goals.
    prior_goals = prior_weight * league_rate
    for _ in range(iterations):
        scale = math.sqrt(home_factor)
        scored: Dict[str, float] = {team: 0.0 for team in teams}
        conceded: Dict[str, float] = {team: 0.0 for team in teams}
        expected_scored: Dict[str, float] = {team: 0.0 for team in teams}
        expected_conceded: Dict[str, float] = {team: 0.0 for team in teams}
        home_goals = 0.0
        away_goals = 0.0
        home_expected_base = 0.0
        away_expected_base = 0.0
        total_goals = 0.0
        total_expected = 0.0
        for weight, row in zip(weights, rows):
            lam_home = league_rate * attack[row.home_id] * defence[row.away_id] * scale
            lam_away = league_rate * attack[row.away_id] * defence[row.home_id] / scale
            scored[row.home_id] += weight * row.home_score
            conceded[row.away_id] += weight * row.home_score
            scored[row.away_id] += weight * row.away_score
            conceded[row.home_id] += weight * row.away_score
            expected_scored[row.home_id] += weight * lam_home
            expected_scored[row.away_id] += weight * lam_away
            expected_conceded[row.away_id] += weight * lam_home
            expected_conceded[row.home_id] += weight * lam_away
            home_goals += weight * row.home_score
            away_goals += weight * row.away_score
            total_goals += weight * (row.home_score + row.away_score)
            total_expected += weight * (lam_home + lam_away)
            # Factor-free expectations: divide each side's own scale out.
            home_expected_base += weight * lam_home / scale
            away_expected_base += weight * lam_away * scale
        for team in teams:
            # TRUE damped MAP with hard bounds: divide the team's own rating
            # out of the exposure first, so the damped step
            # ``a * (t/a)**0.5 = sqrt(a*t)`` has its fixed point at the true
            # rating instead of its square root. The damping itself is kept:
            # full-weight IPF diverges on an imbalanced schedule, and a
            # diverged rating is worse than an uninformative one.
            exp_scored = expected_scored[team] / attack[team]
            exp_conceded = expected_conceded[team] / defence[team]
            target_attack = (scored[team] + prior_goals) / (exp_scored + prior_goals)
            target_defence = (conceded[team] + prior_goals) / (exp_conceded + prior_goals)
            attack[team] = _clamp(
                attack[team] * max(target_attack / attack[team], 1e-9) ** 0.5, 0.50, 2.00
            )
            defence[team] = _clamp(
                defence[team] * max(target_defence / defence[team], 1e-9) ** 0.5, 0.50, 2.00
            )
        # Re-centre so the league average rating is 1.0 in both directions.
        mean_attack = sum(attack.values()) / len(attack)
        mean_defence = sum(defence.values()) / len(defence)
        for team in teams:
            attack[team] /= mean_attack
            defence[team] /= mean_defence
        # Re-centre the scoring level so E[total] matches the observed total
        # despite the (s + 1/s) asymmetry of the symmetric home factor.
        if total_expected > 0:
            league_rate *= (total_goals / total_expected) ** 0.5
        # Home factor from the home:away ratio of observed vs factor-free
        # expectation — identified by the ratio, so it converges to the true
        # home advantage instead of its square root.
        if home_expected_base > 0 and away_expected_base > 0:
            raw_hf = (home_goals / home_expected_base) / (away_goals / away_expected_base)
            home_factor = _clamp(
                home_factor * max(raw_hf / home_factor, 1e-9) ** 0.5, 1.0, 1.35
            )
    return {
        "league_rate": league_rate,
        "home_factor": home_factor,
        "attack": attack,
        "defence": defence,
        "n": len(rows),
    }


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def rates_for(
    ratings: Dict[str, object], home_id: str, away_id: str, *, neutral: bool = False
) -> Tuple[float, float]:
    """Home/away scoring rates under the symmetric home-advantage model.

    The home factor is applied as ``sqrt(hf)`` to the home side and divided
    out of the away side, mirroring exactly how ``fit_ratings`` saw the
    training data. A neutral site uses a factor of 1.0.
    """
    league_rate = float(ratings.get("league_rate") or 0.0)
    attack = ratings.get("attack") or {}
    defence = ratings.get("defence") or {}
    home_factor = float(ratings.get("home_factor") or 1.0)
    scale = 1.0 if neutral else math.sqrt(max(home_factor, 1e-6))
    lam_home = league_rate * attack.get(home_id, 1.0) * defence.get(away_id, 1.0) * scale
    lam_away = league_rate * attack.get(away_id, 1.0) * defence.get(home_id, 1.0) / scale
    return max(lam_home, 0.05), max(lam_away, 0.05)


def rest_adjustment(
    results: Sequence[Result], team_id: str, game_day: str, *, penalty: float = 0.975
) -> float:
    """Back-to-back fatigue multiplier from the team's most recent prior game."""
    previous = [
        row.day
        for row in results
        if team_id in (row.home_id, row.away_id) and row.day < game_day
    ]
    if not previous:
        return 1.0
    gap = total_days_between(max(previous), game_day)
    return penalty if gap <= 1 else 1.0


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def normal_ppf(probability: float) -> float:
    """Inverse standard-normal CDF via bisection (no scipy dependency)."""
    p = min(max(probability, 1e-6), 1.0 - 1e-6)
    low, high = -8.0, 8.0
    for _ in range(80):
        mid = (low + high) / 2.0
        if normal_cdf(mid) < p:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def residual_sigmas(
    results: Sequence[Result],
    ratings: dict,
    *,
    reference_day: str,
    half_life_days: float = 120.0,
) -> Tuple[float, float]:
    """Weighted residual std of margin and total, from fitted results.

    Shared by the football and basketball engines: both compare a normal
    approximation against fitted scoring means, so the sigma estimate has to be
    the same object. ``half_life_days`` stays a parameter so a lane with a
    faster schedule cycle can measure its own dispersion without forking the
    estimator; the football default keeps that lane byte-identical.
    """
    rows = [row for row in results if row.day < reference_day]
    if len(rows) < 20:
        return 0.0, 0.0
    margin_residuals: List[float] = []
    total_residuals: List[float] = []
    weights: List[float] = []
    for row in rows:
        home_rate, away_rate = rates_for(
            ratings, row.home_id, row.away_id
        )
        weights.append(decay_weight(row.day, reference_day, half_life_days))
        margin_residuals.append(row.margin - (home_rate - away_rate))
        total_residuals.append(row.total - (home_rate + away_rate))
    weight_sum = sum(weights)
    mean_margin = sum(w * r for w, r in zip(weights, margin_residuals)) / weight_sum
    mean_total = sum(w * r for w, r in zip(weights, total_residuals)) / weight_sum
    var_margin = sum(w * (r - mean_margin) ** 2 for w, r in zip(weights, margin_residuals)) / weight_sum
    var_total = sum(w * (r - mean_total) ** 2 for w, r in zip(weights, total_residuals)) / weight_sum
    return math.sqrt(max(var_margin, 1.0)), math.sqrt(max(var_total, 1.0))


def normal_line_probs(mean: float, sigma: float, line: float, side: str) -> Tuple[float, float]:
    """(conditional win probability, push probability) for a quoted line.

    A real-valued outcome has zero probability of landing exactly on a number,
    but integer lines are in fact pushes whenever the realised integer equals
    the line, so the normal approximation is discretised around the line. The
    conditional probability is what EV and ticket ranking consume; the push
    mass is reported separately and never folded into a win probability.
    """
    sigma = max(sigma, 0.5)
    if abs(line - round(line)) > 1e-9:
        over = 1.0 - normal_cdf((line - mean) / sigma)
        return (over if side == "over" else 1.0 - over), 0.0
    above = 1.0 - normal_cdf((line + 0.5 - mean) / sigma)
    below = normal_cdf((line - 0.5 - mean) / sigma)
    push = max(1.0 - above - below, 0.0)
    win = above if side == "over" else below
    return win / max(1.0 - push, 1e-9), push


def conditional_win(win: float, push: float) -> float:
    decisive = 1.0 - push
    if decisive <= 0:
        return 0.5
    return min(max(win / decisive, 0.0), 1.0)
