"""NBA / NCAA / WNBA engine: ESPN BPI surface vs empirical score surface.

Two independent surfaces are built for every fixture, exactly as the football
engine does it:

* ``espn_bpi`` — the provider's own win probability and projected scores from
  ``/predictor`` (never assumed to exist; the engine emits nothing without it);
* ``empirical_score`` — team scoring rates fitted on settled finals, with
  symmetric home advantage, per-lane shrinkage and a schedule-fatigue multiplier.

Neither is declared "better" in advance: the daily runner resolves the surface
per league by walk-forward Brier skill, so the choice is a measurement.

Spread and total probabilities use a normal approximation on margin/total with
integer push mass, because a basketball margin is well described by a normal
around the fitted mean. Sigma comes from the fitted residuals when there is
enough history and from a per-lane prior otherwise.

**Posted-line anchoring (college only).** NCAA spreads and totals are anchored
to the single pre-tip main line: ``w * model + (1 - w) * market`` with
``w = 0.20`` for ``ncaa_m``/``ncaa_w`` and no anchoring at all for the NBA and
WNBA. The anchor is applied **once**, to the mean, and every quoted line
(main and alternate) is then evaluated against that same mean. Re-anchoring
inside the line loop would make each line its own model and destroy the
monotonicity that makes an alternate-line ladder readable.

**No leakage.** Only live pre-tip quotes are read (``BookQuote.require_pre_tip``
is enforced on every quote handed to :func:`forecast`, and ``quoted_lines``
filters on ``pre_match`` anyway). Calibration evaluates the shrunk forecast
out-of-sample against settled finals, so ``POSTED_LINE_MODEL_WEIGHTS`` is not a
claim — it is tested by walk-forward Brier like every other surface choice.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from ..espn import PredictorEvidence
from ..models import BookQuote, Forecast, Game, parse_utc
from . import (
    DEFAULT_PRIOR_WEIGHT,
    Result,
    anchored_residual_sigmas,
    fit_ratings,
    normal_cdf,
    normal_line_probs,
    normal_ppf,
    quoted_lines,
    rates_for,
    residual_sigmas,
    total_days_between,
)

#: Both surfaces ride the shared ``fit_ratings`` (symmetric home factor, MAP).
BPI_SOURCE = "espn_bpi"
#: Version tag for the empirical surface. Bumped when the fit's arithmetic
#: changes materially, so the seeder re-seeds history under the new tag instead
#: of mixing two models inside one calibration segment.
EMPIRICAL_SOURCE = "empirical_score_v2"

#: Per-lane shrinkage priors (pseudo-games). A 20-game NBA season and a 30-game
#: NCAA season are different amounts of evidence, and a 360-team D1 pool needs
#: more pull toward 1.0 than a 30-team league does.
PRIOR_WEIGHTS = {"nba": 4.0, "wnba": 4.0, "ncaa_m": 6.0, "ncaa_w": 6.0}
#: Rating half-life in days: the NBA/WNBA play a compact schedule, college
#: rosters and form move faster within a season.
HALF_LIFE_DAYS = {"nba": 90.0, "wnba": 90.0, "ncaa_m": 60.0, "ncaa_w": 60.0}

#: Walk-forward refit cadence for the historical seeder, in days. A basketball
#: ratings fit covers 30 NBA or ~360 D1 teams, so refitting per historical day
#: across a 400-day backfill dominates the job. Freezing the fit for a week at a
#: time costs a few days of staleness on 6 of every 7 seeded matchdays and is
#: the only reason a full seeding pass fits the workflow budget. Basketball
#: only: every other lane keeps its per-day cadence.
SEED_REFIT_EVERY_DAYS = 7
#: (margin, total) sigma priors for the normal approximation, per lane. An NCAA
#: margin is wider than an NBA one because the talent spread is wider.
SIGMA_PRIORS = {
    "nba": (11.5, 17.0),
    "wnba": (11.0, 16.0),
    "ncaa_m": (12.5, 18.5),
    "ncaa_w": (14.0, 17.0),
}
#: Model weight when anchoring college spread/total distributions to the single
#: pre-tip main line: ``w * model + (1 - w) * anchor``. ``0.20`` means the market
#: carries 80% of the mean, which is the point: a single fitted scoring rate is
#: a weak estimate against a priced main line, and the anchored forecast still
#: has to beat ``Brier < 0.25`` and its own base rate to be allowed anywhere
#: near a ticket. Applied ONLY to spread/total, NEVER moneyline.
POSTED_LINE_MODEL_WEIGHTS: Dict[str, Optional[float]] = {
    "ncaa_m": 0.20,
    "ncaa_w": 0.20,
    "nba": None,
    "wnba": None,
}
#: Market-line residual sigma priors for the anchored estimator, per college lane.
#: (spread, total). These are the dispersion floors of the MAIN market line itself.
ANCHOR_MARKET_SIGMA_PRIORS = {"ncaa_m": (10.7, 12.5), "ncaa_w": (11.5, 13.0)}
#: Back-to-back (played yesterday) scoring penalty, applied to the team's rate.
REST_B2B_PENALTY = 0.975
#: Third game in four nights, multiplicative on top of the B2B penalty.
REST_3IN4_PENALTY = 0.99


def fatigue_multiplier(
    results: Sequence[Result],
    team_id: str,
    game_day: str,
    *,
    prior_days: Optional[Dict[str, List[str]]] = None,
    game_scheduled_utc: str = "",
    prior_times: Optional[Dict[str, List[str]]] = None,
) -> float:
    """Schedule-fatigue multiplier from the team's own strictly-past games.

    Exact UTC-hour path when timestamps exist: a back-to-back is a minimum rest
    of ``<= 34h`` (so every 9-30h back-to-back counts, and a ``>= 38h``
    one-day-rest game does not, whatever the calendar day says), and a
    third-in-four is two or more games inside ``84h``. Falls back to the
    calendar-day path otherwise, so legacy synthetic ``Result`` rows without
    ``scheduled_utc`` behave exactly as before.

    ``prior_days``/``prior_times`` are optional precomputed
    ``team -> [day|instant, ...]`` indexes so a walk-forward seeder does not
    rescan the whole result list for every game. Only strictly-past games are
    read, so a same-day or future final can never leak into a pre-tip forecast.
    """
    if game_scheduled_utc:
        times = prior_times.get(team_id) if prior_times is not None else None
        if times is None:
            times = [
                row.scheduled_utc
                for row in results
                if team_id in (row.home_id, row.away_id) and row.scheduled_utc
            ]
        prior_utc = [stamp for stamp in times if stamp and stamp < game_scheduled_utc]
        if prior_utc:
            game_dt = parse_utc(game_scheduled_utc)
            hours = [(game_dt - parse_utc(stamp)).total_seconds() / 3600.0 for stamp in prior_utc]
            multiplier = 1.0
            if min(hours) <= 34.0:
                multiplier *= REST_B2B_PENALTY
            if sum(1 for rest in hours if rest <= 84.0) >= 2:
                multiplier *= REST_3IN4_PENALTY
            return multiplier
        # No prior instant: fall through to the calendar-day path below.
    if prior_days is not None:
        days = [day for day in prior_days.get(team_id, []) if day < game_day]
    else:
        days = [
            row.day
            for row in results
            if team_id in (row.home_id, row.away_id) and row.day < game_day
        ]
    if not days:
        return 1.0
    multiplier = 1.0
    if total_days_between(max(days), game_day) <= 1:
        multiplier *= REST_B2B_PENALTY
    recent = [day for day in days if 1 <= total_days_between(day, game_day) <= 3]
    if len(recent) >= 2:
        multiplier *= REST_3IN4_PENALTY
    return multiplier


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _anchor_line(quotes: Sequence[BookQuote], market: str) -> Optional[float]:
    """The single main line a lane's spread/total mean is anchored to.

    Prices marked ``main_line`` win; if a payload carries no declaration at all
    the median of every quoted line for that market is used, which is a
    deterministic choice rather than the first line that happened to arrive.
    """
    applicable = [
        quote
        for quote in quotes
        if quote.market == market and quote.line is not None and quote.pre_match
    ]
    main = [quote for quote in applicable if quote.main_line]
    return _median([float(quote.line) for quote in (main or applicable)])


def _sigmas(
    league: str, results: Sequence[Result], ratings: dict, reference_day: str
) -> Tuple[float, float]:
    prior_margin, prior_total = SIGMA_PRIORS.get(league, (12.5, 17.0))
    margin, total = residual_sigmas(
        results,
        ratings,
        reference_day=reference_day,
        half_life_days=HALF_LIFE_DAYS.get(league, 120.0),
    )
    return (margin or prior_margin, total or prior_total)


def fit_lane_ratings(league: str, results: Sequence[Result], *, reference_day: str) -> dict:
    """The empirical surface's rating fit with this lane's shrinkage and decay.

    Public so the historical seeder fits exactly what a live tick fits. A seeder
    that reproduced the arithmetic with its own constants would silently drift
    from production the first time a prior moved.
    """
    return fit_ratings(
        results,
        reference_day=reference_day,
        half_life_days=HALF_LIFE_DAYS.get(league, 120.0),
        prior_weight=PRIOR_WEIGHTS.get(league, DEFAULT_PRIOR_WEIGHT),
    )


def lane_sigmas(
    league: str, results: Sequence[Result], ratings: dict, reference_day: str
) -> Tuple[float, float]:
    """``(sigma_margin, sigma_total)`` with per-lane priors — public for seeder reuse."""
    return _sigmas(league, results, ratings, reference_day)


def prior_days_index(results: Sequence[Result]) -> Dict[str, List[str]]:
    """``team -> [day, ...]`` so fatigue does not rescan every result per game."""
    index: Dict[str, List[str]] = {}
    for row in results:
        index.setdefault(row.home_id, []).append(row.day)
        index.setdefault(row.away_id, []).append(row.day)
    return index


def prior_times_index(results: Sequence[Result]) -> Dict[str, List[str]]:
    """``team -> [scheduled_utc, ...]`` for the exact-hour fatigue path.

    The exact-hour path reads instants, so a caller that supplies a *different*
    population through ``prior_days`` (the seeder's in-window finals) must
    supply the matching instants too, or fatigue silently falls back to the
    frozen fit's narrower result list.
    """
    index: Dict[str, List[str]] = {}
    for row in results:
        if not row.scheduled_utc:
            continue
        index.setdefault(row.home_id, []).append(row.scheduled_utc)
        index.setdefault(row.away_id, []).append(row.scheduled_utc)
    return index


def empirical_expectations(
    league: str,
    results: Sequence[Result],
    ratings: dict,
    *,
    home_id: str,
    away_id: str,
    reference_day: str,
    neutral_site: bool = False,
    sigmas: Optional[Tuple[float, float]] = None,
    prior_days: Optional[Dict[str, List[str]]] = None,
    fatigue_day: Optional[str] = None,
    game_scheduled_utc: str = "",
    prior_times: Optional[Dict[str, List[str]]] = None,
) -> Tuple[float, float, float, float, float]:
    """``(margin, total, sigma_margin, sigma_total, prob_home)`` for one fixture.

    One function for the whole empirical surface so the moneyline, the spread
    ladder and the seeded history can never disagree about the same fixture.

    ``sigmas``/``prior_days``/``fatigue_day`` exist for the walk-forward seeder,
    which prices a whole refit window from one frozen fit: the residual sigmas
    are ``O(results)`` and must not be recomputed per fixture, and the fatigue
    question is "did this team play yesterday **before this matchday**", which
    is a different day from the frozen fit's reference day whenever the seeder
    is mid-window. All three are optional, so a live tick keeps the simple call.
    """
    fatigue_reference = fatigue_day or reference_day
    home_rate, away_rate = rates_for(ratings, home_id, away_id, neutral=neutral_site)
    home_rate *= fatigue_multiplier(
        results,
        home_id,
        fatigue_reference,
        prior_days=prior_days,
        game_scheduled_utc=game_scheduled_utc,
        prior_times=prior_times,
    )
    away_rate *= fatigue_multiplier(
        results,
        away_id,
        fatigue_reference,
        prior_days=prior_days,
        game_scheduled_utc=game_scheduled_utc,
        prior_times=prior_times,
    )
    expected_margin = home_rate - away_rate
    expected_total = home_rate + away_rate
    if sigmas is None:
        sigma_margin, sigma_total = _sigmas(league, results, ratings, reference_day)
    else:
        sigma_margin, sigma_total = sigmas
    # ``prob_home`` comes from the same margin the spread uses, so the moneyline
    # and the spread ladder cannot disagree about who is favoured.
    prob_home = 1.0 - normal_cdf((0.0 - expected_margin) / max(sigma_margin, 0.5))
    return expected_margin, expected_total, sigma_margin, sigma_total, prob_home


def _anchored(model: float, anchor: Optional[float], weight: Optional[float]) -> float:
    if weight is None or anchor is None:
        return model
    return weight * model + (1.0 - weight) * anchor


def forecast(
    game: Game,
    quotes: Sequence[BookQuote],
    *,
    results: Sequence[Result],
    reference_day: str,
    surface: str = EMPIRICAL_SOURCE,
    predictor: Optional[PredictorEvidence] = None,
) -> List[Forecast]:
    """Model probabilities for every quoted basketball line on one surface.

    Every quote is checked for pre-tip provenance before it can contribute a
    price or a line; a post-tip observation raises rather than being used.
    """
    for quote in quotes:
        quote.require_pre_tip()

    model_weight = POSTED_LINE_MODEL_WEIGHTS.get(game.league)
    ratings: dict = {}
    if surface == BPI_SOURCE:
        if predictor is None:
            return []
        prob_home = predictor.prob_home
        expected_home = predictor.expected_home
        expected_away = predictor.expected_away
        if expected_home is None or expected_away is None:
            # Basketball BPI ships ``gameProjection`` plus, on some payloads,
            # ``teamPredPtDiff``; expected point totals are frequently absent.
            # A margin can still be priced from it, a total cannot: no imputed
            # total, no total market.
            expected_margin = predictor.predicted_margin
            expected_total = None
        else:
            expected_margin = expected_home - expected_away
            expected_total = expected_home + expected_away
        source = BPI_SOURCE
        if not results:
            sigma_margin, sigma_total = SIGMA_PRIORS.get(game.league, (12.5, 17.0))
        else:
            ratings = fit_ratings(
                results,
                reference_day=reference_day,
                half_life_days=HALF_LIFE_DAYS.get(game.league, 120.0),
                prior_weight=PRIOR_WEIGHTS.get(game.league, DEFAULT_PRIOR_WEIGHT),
            )
            sigma_margin, sigma_total = _sigmas(
                game.league, results, ratings, reference_day
            )
    else:
        if not results:
            return []
        ratings = fit_lane_ratings(game.league, results, reference_day=reference_day)
        if not ratings.get("n"):
            return []
        (
            expected_margin,
            expected_total,
            sigma_margin,
            sigma_total,
            prob_home,
        ) = empirical_expectations(
            game.league,
            results,
            ratings,
            home_id=game.home_team_id,
            away_id=game.away_team_id,
            reference_day=reference_day,
            neutral_site=game.neutral_site,
            game_scheduled_utc=game.scheduled_utc,
        )
        source = EMPIRICAL_SOURCE

    if expected_margin is None:
        # Last resort for a BPI payload that carries a probability but no
        # projected score: invert the win probability onto the margin axis.
        expected_margin = sigma_margin * normal_ppf(prob_home)

    forecasts: List[Forecast] = []
    for selection, probability in (("home", prob_home), ("away", 1.0 - prob_home)):
        forecasts.append(
            Forecast(
                game_id=game.game_id,
                league=game.league,
                scheduled_utc=game.scheduled_utc,
                home_name=game.home_name,
                away_name=game.away_name,
                market="moneyline",
                selection=selection,
                raw_probability=probability,
                alternative_probabilities={"home": prob_home, "away": 1.0 - prob_home},
                source=source,
                expected_margin=round(expected_margin, 4),
                expected_total=None if expected_total is None else round(expected_total, 4),
            )
        )

    # Anchor ONCE, at the mean. Every quoted line below is scored against this
    # single number, so an alternate-line ladder stays monotonic.
    anchor_spread = _anchor_line(quotes, "spread")
    margin = _anchored(
        expected_margin,
        None if anchor_spread is None else -anchor_spread,
        model_weight,
    )
    total = (
        None
        if expected_total is None
        else _anchored(expected_total, _anchor_line(quotes, "total"), model_weight)
    )

    # College spread/total ladders price against an 80%-market-anchored mean, so
    # their dispersion must be measured on anchored residuals (or the orthogonal
    # closed form), never on raw-rating residuals — a raw sigma overstates the
    # residual scale of the anchored mean and makes every far alt line look
    # overpriced. Moneyline stays on the unanchored margin/sigma pair.
    spread_sigma = sigma_margin
    total_sigma = sigma_total
    if model_weight is not None:
        spread_sigma, total_sigma = anchored_residual_sigmas(
            results,
            ratings,
            reference_day=reference_day,
            half_life_days=HALF_LIFE_DAYS.get(game.league, 120.0),
            model_weight=model_weight,
            sigma_raw=(sigma_margin, sigma_total),
            market_priors=ANCHOR_MARKET_SIGMA_PRIORS.get(game.league, (10.7, 12.5)),
        )

    for line in quoted_lines(quotes, "total"):
        if total is None:
            continue
        over_probability, over_push = normal_line_probs(total, total_sigma, line, "over")
        under_probability, _ = normal_line_probs(total, total_sigma, line, "under")
        for selection, probability in (
            ("over", over_probability),
            ("under", under_probability),
        ):
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
                    alternative_probabilities={
                        "over": over_probability,
                        "under": under_probability,
                    },
                    source=source,
                    expected_total=round(total, 4),
                )
            )

    for line in quoted_lines(quotes, "spread"):
        # A home handicap of -6.5 means the home side must win by more than
        # 6.5, i.e. the market implies a home margin of +6.5.
        threshold = -line
        home_probability, push = normal_line_probs(
            margin, spread_sigma, threshold, "over"
        )
        away_probability, _ = normal_line_probs(
            margin, spread_sigma, threshold, "under"
        )
        for selection, probability in (
            ("home", home_probability),
            ("away", away_probability),
        ):
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
                    expected_margin=round(margin, 4),
                )
            )
    return forecasts


def tiebreak_surface(brier_by_source: Dict[str, Optional[float]]) -> Optional[str]:
    """Deterministic surface when the Brier gap is inside ``min_gap``.

    ``resolve_surface`` refuses to call a coin flip a measurement, which is
    right for reporting — but a ticket still has to come from ONE surface.
    Prefer the lower moneyline Brier and break exact ties toward the empirical
    surface, which is the one with no provider dependency.
    """
    bpi = brier_by_source.get(BPI_SOURCE)
    empirical = brier_by_source.get(EMPIRICAL_SOURCE)
    if bpi is None and empirical is None:
        return None
    if bpi is None:
        return EMPIRICAL_SOURCE
    if empirical is None:
        return BPI_SOURCE
    if empirical <= bpi:
        return EMPIRICAL_SOURCE
    return BPI_SOURCE


def resolve_surface(
    brier_by_source: Dict[str, Optional[float]], *, min_gap: float = 0.002
) -> Optional[str]:
    """Pick the surface with the better walk-forward Brier score.

    Returns ``None`` when either side has no evidence or the difference is
    inside ``min_gap`` (a coin flip dressed as a model choice is not a
    decision).
    """
    bpi = brier_by_source.get(BPI_SOURCE)
    empirical = brier_by_source.get(EMPIRICAL_SOURCE)
    if bpi is None or empirical is None:
        return None
    if abs(bpi - empirical) < min_gap:
        return None
    return BPI_SOURCE if bpi < empirical else EMPIRICAL_SOURCE
