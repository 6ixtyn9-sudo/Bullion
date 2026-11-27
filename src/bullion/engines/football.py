"""NFL / college football engine: ESPN FPI surface vs empirical score surface.

Two independent surfaces are built for every fixture:

* ``espn_fpi`` — the provider's own win probability and expected score from
  ``/predictor`` (never assumed to exist; the engine emits nothing without it);
* ``empirical_score`` — team scoring rates fitted on settled finals, with
  home-field advantage.

Neither is declared "better" in advance. The daily runner resolves the surface
per league by walk-forward Brier skill, so the choice is a measurement. Spreads
and totals use a normal approximation on margin/total with integer push mass,
because football scores are close enough to a normal margin for a 44.5/48 line
decision; the empirical sigma is estimated from the fitted residuals whenever
there is enough history.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from ..espn import PredictorEvidence
from ..models import BookQuote, Forecast, Game
from . import (
    DEFAULT_PRIOR_WEIGHT,
    Result,
    decay_weight,
    fit_ratings,
    normal_cdf,
    normal_line_probs,
    quoted_lines,
    rates_for,
    rest_adjustment,
)

#: Both surfaces ride the shared ``fit_ratings`` (symmetric home factor, MAP).
FPI_SOURCE = "espn_fpi"
EMPIRICAL_SOURCE = "empirical_score"
#: Per-lane shrinkage priors (pseudo-games) from the train-split sweep. CFB's
#: blowout-heavy, high-signal scoring wants light shrinkage (2 pseudo-games;
#: train Brier 0.1618 vs 0.1706 at the default 4, holdout 0.1521 vs 0.1586);
#: every other football lane stays on ``DEFAULT_PRIOR_WEIGHT``.
PRIOR_WEIGHTS = {"cfb": 2.0}
SIGMA_PRIORS = {"nfl": (13.5, 13.5), "cfb": (16.0, 14.5)}


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
    results: Sequence[Result], ratings: dict, *, reference_day: str
) -> Tuple[float, float]:
    """Weighted residual std of margin and total, from fitted results."""
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
        weights.append(decay_weight(row.day, reference_day, 120.0))
        margin_residuals.append(row.margin - (home_rate - away_rate))
        total_residuals.append(row.total - (home_rate + away_rate))
    weight_sum = sum(weights)
    mean_margin = sum(w * r for w, r in zip(weights, margin_residuals)) / weight_sum
    mean_total = sum(w * r for w, r in zip(weights, total_residuals)) / weight_sum
    var_margin = sum(w * (r - mean_margin) ** 2 for w, r in zip(weights, margin_residuals)) / weight_sum
    var_total = sum(w * (r - mean_total) ** 2 for w, r in zip(weights, total_residuals)) / weight_sum
    return math.sqrt(max(var_margin, 1.0)), math.sqrt(max(var_total, 1.0))


def _sigmas(league: str, results: Sequence[Result], ratings: dict, reference_day: str) -> Tuple[float, float]:
    prior_margin, prior_total = SIGMA_PRIORS.get(league, (14.0, 14.0))
    margin, total = residual_sigmas(results, ratings, reference_day=reference_day)
    return (margin or prior_margin, total or prior_total)


def forecast(
    game: Game,
    quotes: Sequence[BookQuote],
    *,
    results: Sequence[Result],
    reference_day: str,
    surface: str = EMPIRICAL_SOURCE,
    predictor: Optional[PredictorEvidence] = None,
) -> List[Forecast]:
    """Model probabilities for every quoted football line on one surface."""
    if surface == FPI_SOURCE:
        if predictor is None:
            return []
        prob_home = predictor.prob_home
        expected_home = predictor.expected_home
        expected_away = predictor.expected_away
        if expected_home is None or expected_away is None:
            # LIVE_VERIFIED 2026-11-26: ESPN FPI ships ``teamPredPtDiff`` — a
            # projected margin — while ``teamExpectedPts``/``oppExpectedPts``
            # are absent. Throwing that away left every FPI spread unpriced on
            # the one number the provider actually publishes. The total stays
            # unknown because no expected score exists to sum.
            expected_margin = predictor.predicted_margin
            expected_total = None
        else:
            expected_margin = expected_home - expected_away
            expected_total = expected_home + expected_away
        source = FPI_SOURCE
        if not results:
            sigma_margin, sigma_total = SIGMA_PRIORS.get(game.league, (14.0, 14.0))
        else:
            ratings = fit_ratings(
                results,
                reference_day=reference_day,
                prior_weight=PRIOR_WEIGHTS.get(game.league, DEFAULT_PRIOR_WEIGHT),
            )
            sigma_margin, sigma_total = _sigmas(
                game.league, results, ratings, reference_day
            )
    else:
        if not results:
            return []
        ratings = fit_ratings(
            results,
            reference_day=reference_day,
            prior_weight=PRIOR_WEIGHTS.get(game.league, DEFAULT_PRIOR_WEIGHT),
        )
        if not ratings.get("n"):
            return []
        home_rate, away_rate = rates_for(
            ratings, game.home_team_id, game.away_team_id, neutral=game.neutral_site
        )
        home_rate *= rest_adjustment(results, game.home_team_id, reference_day, penalty=0.99)
        away_rate *= rest_adjustment(results, game.away_team_id, reference_day, penalty=0.99)
        expected_margin = home_rate - away_rate
        expected_total = home_rate + away_rate
        sigma_margin, sigma_total = _sigmas(game.league, results, ratings, reference_day)
        prob_home = 1.0 - normal_cdf(
            (0.0 - expected_margin) / max(sigma_margin, 0.5)
        )
        source = EMPIRICAL_SOURCE

    forecasts: List[Forecast] = [
        Forecast(
            game_id=game.game_id,
            league=game.league,
            scheduled_utc=game.scheduled_utc,
            home_name=game.home_name,
            away_name=game.away_name,
            market="moneyline",
            selection="home",
            raw_probability=prob_home,
            alternative_probabilities={"home": prob_home, "away": 1.0 - prob_home},
            source=source,
            expected_margin=None if expected_margin is None else round(expected_margin, 4),
            expected_total=None if expected_total is None else round(expected_total, 4),
        ),
        Forecast(
            game_id=game.game_id,
            league=game.league,
            scheduled_utc=game.scheduled_utc,
            home_name=game.home_name,
            away_name=game.away_name,
            market="moneyline",
            selection="away",
            raw_probability=1.0 - prob_home,
            alternative_probabilities={"home": prob_home, "away": 1.0 - prob_home},
            source=source,
            expected_margin=None if expected_margin is None else round(expected_margin, 4),
            expected_total=None if expected_total is None else round(expected_total, 4),
        ),
    ]

    if expected_margin is None and surface == FPI_SOURCE:
        expected_margin = sigma_margin * normal_ppf(prob_home)

    for line in quoted_lines(quotes, "total"):
        if expected_total is None:
            continue
        over_probability, over_push = normal_line_probs(
            expected_total, sigma_total, line, "over"
        )
        under_probability, _ = normal_line_probs(
            expected_total, sigma_total, line, "under"
        )
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
                    push_probability=over_push,
                    alternative_probabilities={"over": over_probability, "under": under_probability},
                    source=source,
                    expected_total=round(expected_total, 4),
                )
            )

    for line in quoted_lines(quotes, "spread"):
        # A home handicap of -3 means the home side must win by more than 3.
        threshold = -line
        home_probability, push = normal_line_probs(
            expected_margin, sigma_margin, threshold, "over"
        )
        away_probability, _ = normal_line_probs(
            expected_margin, sigma_margin, threshold, "under"
        )
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
                    alternative_probabilities={
                        "home": home_probability,
                        "away": away_probability,
                    },
                    source=source,
                    expected_margin=round(expected_margin, 4),
                )
            )
    return forecasts


def tiebreak_surface(brier_by_source: Dict[str, Optional[float]]) -> Optional[str]:
    """Deterministic surface when the Brier gap is inside ``min_gap``.

    ``resolve_surface`` refuses to call a coin flip a measurement, which is
    right for reporting — but a ticket still has to come from ONE surface.
    Prefer the lower moneyline Brier, break exact ties toward the empirical
    surface, and fall back to whichever surface has evidence.
    """
    fpi = brier_by_source.get(FPI_SOURCE)
    empirical = brier_by_source.get(EMPIRICAL_SOURCE)
    if fpi is None and empirical is None:
        return None
    if fpi is None:
        return EMPIRICAL_SOURCE
    if empirical is None:
        return FPI_SOURCE
    if empirical <= fpi:
        return EMPIRICAL_SOURCE
    return FPI_SOURCE


def resolve_surface(
    brier_by_source: Dict[str, Optional[float]], *, min_gap: float = 0.002
) -> Optional[str]:
    """Pick the surface with the better walk-forward Brier score.

    Returns ``None`` when either side has no evidence or the difference is
    inside ``min_gap`` (a coin flip dressed as a model choice is not a
    decision).
    """
    fpi = brier_by_source.get(FPI_SOURCE)
    empirical = brier_by_source.get(EMPIRICAL_SOURCE)
    if fpi is None or empirical is None:
        return None
    if abs(fpi - empirical) < min_gap:
        return None
    return FPI_SOURCE if fpi < empirical else EMPIRICAL_SOURCE
