"""Shared, dependency-free numeric helpers for the three sport engines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Sequence, Tuple

MAX_GOALS = 10


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
    prior_weight: float = 2.0,
) -> Dict[str, object]:
    """Regularised attack/defence ratings via weighted iterative fitting.

    Returns ``{"league_rate", "home_factor", "attack", "defence", "n"}`` where
    ``league_rate`` is the average score per team per game. Ratings are pulled
    toward 1.0 by ``prior_weight`` pseudo-games, so a team with two games does
    not get a 3.0 attack rating.
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
    for _ in range(iterations):
        scored: Dict[str, float] = {team: 0.0 for team in teams}
        conceded: Dict[str, float] = {team: 0.0 for team in teams}
        expected_scored: Dict[str, float] = {team: 0.0 for team in teams}
        expected_conceded: Dict[str, float] = {team: 0.0 for team in teams}
        home_goals = 0.0
        home_expected = 0.0
        for weight, row in zip(weights, rows):
            lam_home = league_rate * attack[row.home_id] * defence[row.away_id] * home_factor
            lam_away = league_rate * attack[row.away_id] * defence[row.home_id]
            scored[row.home_id] += weight * row.home_score
            conceded[row.away_id] += weight * row.home_score
            scored[row.away_id] += weight * row.away_score
            conceded[row.home_id] += weight * row.away_score
            expected_scored[row.home_id] += weight * lam_home
            expected_scored[row.away_id] += weight * lam_away
            expected_conceded[row.away_id] += weight * lam_home
            expected_conceded[row.home_id] += weight * lam_away
            home_goals += weight * row.home_score
            home_expected += weight * lam_home
        for team in teams:
            # Damped multiplicative updates with hard bounds. Full-weight IPF
            # diverges on an imbalanced schedule (a team that only ever plays
            # one side of the fixture list is not identifiable), and a diverged
            # rating is worse than an uninformative one.
            target_attack = (scored[team] + prior_weight) / (
                expected_scored[team] + prior_weight
            )
            target_defence = (conceded[team] + prior_weight) / (
                expected_conceded[team] + prior_weight
            )
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
        if home_expected > 0:
            home_factor = max(1.0, min(1.25, home_goals / home_expected))
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
    """Home/away scoring rates, using a neutral-site home factor of 1.0."""
    league_rate = float(ratings.get("league_rate") or 0.0)
    attack = ratings.get("attack") or {}
    defence = ratings.get("defence") or {}
    home_factor = float(ratings.get("home_factor") or 1.0)
    if neutral:
        home_factor = 1.0
    lam_home = league_rate * attack.get(home_id, 1.0) * defence.get(away_id, 1.0) * home_factor
    lam_away = league_rate * attack.get(away_id, 1.0) * defence.get(home_id, 1.0)
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
