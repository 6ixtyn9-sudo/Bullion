"""Soccer engine: time-decayed Dixon-Coles bivariate Poisson (Top-6 lanes only).

Dixon-Coles adds a low-score dependence correction (rho) to two independent
Poisson counts. That correction is the entire reason this engine is not a
generic Poisson: draws and 0-0/1-0/0-1/1-1 scorelines are the market, and an
independence assumption systematically misprices them.

rho is estimated by maximum likelihood over a fixed grid on the *weighted*
historical sample (time decay, 240-day half-life by default). Nothing about
any specific league's live odds coverage is assumed here: the ticket layer
keeps a lane in ``research_only`` unless the live probe shows at least 50%
pre-kickoff named-book coverage.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from ..models import BookQuote, Forecast, Game
from . import (
    MAX_GOALS,
    Result,
    conditional_win,
    decay_weight,
    fit_ratings,
    poisson_pmf,
    quoted_lines,
    rates_for,
)

SOURCE = "dixon_coles_v2"  # v2: fit_ratings pseudo-game prior + home-factor fix
RHO_GRID_MIN = -0.18
RHO_GRID_MAX = 0.18
RHO_GRID_STEP = 0.01
RHO_PRIOR = 0.0


def _tau(home_goals: int, away_goals: int, home_rate: float, away_rate: float, rho: float) -> float:
    """Dixon-Coles low-score correction factor (identity outside 0/1 goals)."""
    if home_goals == 0 and away_goals == 0:
        return 1.0 - home_rate * away_rate * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + home_rate * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + away_rate * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def fit_rho(
    results: Sequence[Result],
    ratings: dict,
    *,
    reference_day: str,
    half_life_days: float = 240.0,
) -> float:
    """Grid-search rho by weighted log-likelihood; falls back to the prior."""
    rows = [row for row in results if row.day < reference_day]
    if len(rows) < 30:
        return RHO_PRIOR
    rates: List[Tuple[float, float, float, int, int]] = []
    for row in rows:
        home_rate, away_rate = rates_for(ratings, row.home_id, row.away_id)
        weight = decay_weight(row.day, reference_day, half_life_days)
        rates.append((home_rate, away_rate, weight, int(row.home_score), int(row.away_score)))
    best_rho, best_ll = RHO_PRIOR, float("-inf")
    steps = int(round((RHO_GRID_MAX - RHO_GRID_MIN) / RHO_GRID_STEP))
    for index in range(steps + 1):
        rho = RHO_GRID_MIN + index * RHO_GRID_STEP
        log_likelihood = 0.0
        for home_rate, away_rate, weight, home_goals, away_goals in rates:
            tau = _tau(home_goals, away_goals, home_rate, away_rate, rho)
            if tau <= 0:
                log_likelihood = float("-inf")
                break
            log_likelihood += weight * (
                math.log(tau)
                + math.log(max(poisson_pmf(home_rate, home_goals), 1e-12))
                + math.log(max(poisson_pmf(away_rate, away_goals), 1e-12))
            )
        if log_likelihood > best_ll:
            best_rho, best_ll = rho, log_likelihood
    return round(best_rho, 4)


def score_matrix(home_rate: float, away_rate: float, rho: float) -> List[List[float]]:
    matrix = []
    for home_goals in range(MAX_GOALS + 1):
        row = []
        for away_goals in range(MAX_GOALS + 1):
            value = (
                poisson_pmf(home_rate, home_goals)
                * poisson_pmf(away_rate, away_goals)
                * _tau(home_goals, away_goals, home_rate, away_rate, rho)
            )
            row.append(max(value, 0.0))
        matrix.append(row)
    total = sum(sum(row) for row in matrix)
    if total <= 0:
        raise ValueError("score matrix has no probability mass")
    return [[value / total for value in row] for row in matrix]


def _one_x_two(matrix: List[List[float]]) -> Tuple[float, float, float]:
    home = sum(
        matrix[h][a] for h in range(len(matrix)) for a in range(len(matrix[h])) if h > a
    )
    draw = sum(matrix[k][k] for k in range(len(matrix)))
    away = max(1.0 - home - draw, 0.0)
    return home, draw, away


def _line_probs(matrix: List[List[float]], line: float) -> Tuple[float, float, float]:
    """(over, under, push) for a goals line."""
    over = under = push = 0.0
    for home_goals, row in enumerate(matrix):
        for away_goals, probability in enumerate(row):
            total = home_goals + away_goals
            if abs(total - line) < 1e-9:
                push += probability
            elif total > line:
                over += probability
            else:
                under += probability
    return over, under, push


def _handicap_probs(matrix: List[List[float]], line: float) -> Tuple[float, float, float]:
    """(home covers, away covers, push) for a home-oriented handicap."""
    home = away = push = 0.0
    for home_goals, row in enumerate(matrix):
        for away_goals, probability in enumerate(row):
            margin = home_goals - away_goals + line
            if abs(margin) < 1e-9:
                push += probability
            elif margin > 0:
                home += probability
            else:
                away += probability
    return home, away, push


def forecast(
    game: Game,
    quotes: Sequence[BookQuote],
    *,
    results: Sequence[Result],
    reference_day: str,
    half_life_days: float = 240.0,
) -> List[Forecast]:
    """Model probabilities for every quoted soccer line (1X2, totals, spreads)."""
    ratings = fit_ratings(results, reference_day=reference_day, half_life_days=half_life_days)
    if not ratings.get("n"):
        return []
    home_rate, away_rate = rates_for(
        ratings, game.home_team_id, game.away_team_id, neutral=game.neutral_site
    )
    rho = fit_rho(results, ratings, reference_day=reference_day, half_life_days=half_life_days)
    matrix = score_matrix(home_rate, away_rate, rho)
    home, draw, away = _one_x_two(matrix)
    expected_total = sum(
        (h + a) * matrix[h][a] for h in range(len(matrix)) for a in range(len(matrix[h]))
    )
    expected_margin = sum(
        (h - a) * matrix[h][a] for h in range(len(matrix)) for a in range(len(matrix[h]))
    )
    probabilities = {"home": home, "draw": draw, "away": away}

    forecasts: List[Forecast] = [
        Forecast(
            game_id=game.game_id,
            league=game.league,
            scheduled_utc=game.scheduled_utc,
            home_name=game.home_name,
            away_name=game.away_name,
            market="moneyline",
            selection=selection,
            raw_probability=probability,
            alternative_probabilities=probabilities,
            source=SOURCE,
            expected_total=round(expected_total, 4),
            expected_margin=round(expected_margin, 4),
        )
        for selection, probability in probabilities.items()
    ]

    for line in quoted_lines(quotes, "total"):
        over, under, push = _line_probs(matrix, line)
        over_probability = conditional_win(over, push)
        under_probability = conditional_win(under, push)
        for selection, probability in (("over", over_probability), ("under", under_probability)):
            forecasts.append(
                Forecast(
                    game_id=game.game_id,
                    league=game.league,
                    scheduled_utc=game.scheduled_utc,
                    home_name=game.home_name,
                    away_name=game.away_name,
                    market="total",
                    selection=selection,
                    line=line,
                    raw_probability=probability,
                    push_probability=push,
                    alternative_probabilities={"over": over_probability, "under": under_probability},
                    source=SOURCE,
                    expected_total=round(expected_total, 4),
                )
            )

    for line in quoted_lines(quotes, "spread"):
        home_cover, away_cover, push = _handicap_probs(matrix, line)
        home_probability = conditional_win(home_cover, push)
        away_probability = conditional_win(away_cover, push)
        for selection, probability in (("home", home_probability), ("away", away_probability)):
            forecasts.append(
                Forecast(
                    game_id=game.game_id,
                    league=game.league,
                    scheduled_utc=game.scheduled_utc,
                    home_name=game.home_name,
                    away_name=game.away_name,
                    market="spread",
                    selection=selection,
                    line=line,
                    raw_probability=probability,
                    push_probability=push,
                    alternative_probabilities={"home": home_probability, "away": away_probability},
                    source=SOURCE,
                    expected_margin=round(expected_margin, 4),
                )
            )
    return forecasts
