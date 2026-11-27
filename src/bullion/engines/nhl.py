"""NHL engine: regularised Poisson goals + home ice + rest, priced over OT/SO.

Why not a Gaussian: hockey is a low-count scoring sport, so a normal margin
model badly misstates ties, pucklines, and 5.5 totals. Regulation goals are
Poisson; the ~22% of games that reach overtime are resolved by an explicit
OT/SO split, and the extra overtime goal is carried into the final-total
distribution because that is what the official final score contains.

Model surfaces are priors where no public sample exists (OT share, OT goal
distribution). They are constants, not discoveries: ``scripts/backfill.py``
plus settled history can replace them later, and the ticket gate refuses to
price anything until 30 settled observations exist per segment.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from ..models import BookQuote, Forecast, Game
from . import (
    MAX_GOALS,
    Result,
    conditional_win,
    fit_ratings,
    poisson_pmf,
    quoted_lines,
    rates_for,
    rest_adjustment,
)

#: Symmetric home factor + damped Gamma-Poisson MAP fit (see ``fit_ratings``).
SOURCE = "nhl_poisson"
#: Shrinkage prior (pseudo-games) for the NHL fit, chosen by the {2,4,8,12,16}
#: sweep on the NHL train split only and confirmed on the untouched holdout
#: (Brier 0.2495 at 16 vs 0.2546 at the global default 4). Hockey's weak
#: per-game signal needs much heavier shrinkage than the other lanes.
PRIOR_WEIGHT = 16.0
OT_HOME_WIN_SHARE = 0.52  # 3-on-3 + shootout edge for the home side (prior)
#: Extra goals beyond regulation, for the games that reach OT/SO. NHL overtime is
#: sudden death and a shootout winner is credited exactly one goal, so **every**
#: regulation draw ends 1-0 in the extra period: the final total is always
#: ``regulation_total + 1``. The previous prior put 72% of OT games at +0 goals,
#: i.e. it settled drawn games as regulation draws that never happened.
OT_GOAL_DISTRIBUTION: Dict[int, float] = {1: 1.0}


def _final_distributions(
    home_rate: float, away_rate: float
) -> Tuple[Dict[int, float], Dict[int, float], float]:
    """(final margin, final total, regulation draw probability) distributions."""
    margin: Dict[int, float] = {}
    total: Dict[int, float] = {}
    draw = 0.0
    for home_goals in range(MAX_GOALS + 1):
        for away_goals in range(MAX_GOALS + 1):
            probability = poisson_pmf(home_rate, home_goals) * poisson_pmf(away_rate, away_goals)
            if home_goals == away_goals:
                draw += probability
                for ot_goals, ot_share in OT_GOAL_DISTRIBUTION.items():
                    weighted = probability * ot_share
                    margin[1] = margin.get(1, 0.0) + weighted * OT_HOME_WIN_SHARE
                    margin[-1] = margin.get(-1, 0.0) + weighted * (1.0 - OT_HOME_WIN_SHARE)
                    total_value = home_goals + away_goals + ot_goals
                    total[total_value] = total.get(total_value, 0.0) + weighted
            else:
                margin[home_goals - away_goals] = (
                    margin.get(home_goals - away_goals, 0.0) + probability
                )
                total[home_goals + away_goals] = (
                    total.get(home_goals + away_goals, 0.0) + probability
                )
    # The Poisson grid is truncated at MAX_GOALS; renorm so the tail mass is
    # redistributed instead of leaving a systematically under-confident gap.
    margin_sum = sum(margin.values())
    total_sum = sum(total.values())
    if margin_sum > 0:
        margin = {key: value / margin_sum for key, value in margin.items()}
    if total_sum > 0:
        total = {key: value / total_sum for key, value in total.items()}
    return margin, total, draw


def _market_probs(distribution: Dict[int, float], line: float) -> Tuple[float, float]:
    """(win, push) for an over/under or home-handicap line over a discrete dist."""
    win = 0.0
    push = 0.0
    for value, probability in distribution.items():
        if abs(value - line) < 1e-9:
            push += probability
        elif value > line:
            win += probability
    return win, push


def forecast(
    game: Game,
    quotes: Sequence[BookQuote],
    *,
    results: Sequence[Result],
    reference_day: str,
) -> List[Forecast]:
    """Model probabilities for every quoted NHL line. Prices are attached later."""
    ratings = fit_ratings(results, reference_day=reference_day, prior_weight=PRIOR_WEIGHT)
    if not ratings.get("n"):
        return []
    home_rate, away_rate = rates_for(ratings, game.home_team_id, game.away_team_id,
                                     neutral=game.neutral_site)
    game_day = reference_day
    home_rate *= rest_adjustment(results, game.home_team_id, game_day)
    away_rate *= rest_adjustment(results, game.away_team_id, game_day)
    margin, total, _draw = _final_distributions(home_rate, away_rate)
    prob_home = sum(p for m, p in margin.items() if m > 0)
    prob_away = sum(p for m, p in margin.items() if m < 0)
    expected_margin = sum(m * p for m, p in margin.items())
    expected_total = sum(t * p for t, p in total.items())

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
            alternative_probabilities={"home": prob_home, "away": prob_away},
            source=SOURCE,
            expected_margin=round(expected_margin, 4),
            expected_total=round(expected_total, 4),
        ),
        Forecast(
            game_id=game.game_id,
            league=game.league,
            scheduled_utc=game.scheduled_utc,
            home_name=game.home_name,
            away_name=game.away_name,
            market="moneyline",
            selection="away",
            raw_probability=prob_away,
            alternative_probabilities={"home": prob_home, "away": prob_away},
            source=SOURCE,
            expected_margin=round(expected_margin, 4),
            expected_total=round(expected_total, 4),
        ),
    ]

    for line in quoted_lines(quotes, "total"):
        over_win, over_push = _market_probs(total, line)
        over_probability = conditional_win(over_win, over_push)
        under_probability = conditional_win(1.0 - over_win - over_push, over_push)
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
                    source=SOURCE,
                    expected_total=round(expected_total, 4),
                )
            )

    for line in quoted_lines(quotes, "spread"):
        home_win, push = _market_probs(margin, -line)
        home_probability = conditional_win(home_win, push)
        away_probability = conditional_win(1.0 - home_win - push, push)
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
                    source=SOURCE,
                    expected_margin=round(expected_margin, 4),
                )
            )
    return forecasts
