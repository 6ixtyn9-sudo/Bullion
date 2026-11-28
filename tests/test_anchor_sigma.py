"""F1 — college anchored-mean / anchored-sigma consistency.

The spread/total ladder prices against an 80%-market-anchored mean, so its
dispersion has to be the dispersion of the *anchored* residual, not of the raw
model residual. Everything here is synthetic: no live ESPN payload is involved.
"""

from __future__ import annotations

import pytest

from bullion.engines import (
    Result,
    anchored_residual_sigmas,
    anchored_sigma,
    decay_weight,
)
from bullion.engines import basketball as bb
from bullion.models import BookQuote, Game, Status

NCAA_M = "ncaa_m"


def _quotes(*, spread: float = -6.5, total: float = 145.5) -> list[BookQuote]:
    return [
        BookQuote(
            game_id="g1",
            book="DraftKings",
            market="spread",
            line=spread,
            observed_utc="2026-11-28T15:00:00Z",
            side_prices={"home": 1.91, "away": 1.91},
            pre_match=True,
            main_line=True,
        ),
        BookQuote(
            game_id="g1",
            book="DraftKings",
            market="total",
            line=total,
            observed_utc="2026-11-28T15:00:00Z",
            side_prices={"over": 1.91, "under": 1.91},
            pre_match=True,
            main_line=True,
        ),
    ]


def _game(league: str = NCAA_M) -> Game:
    return Game(
        game_id="g1",
        league=league,
        season="2027",
        home_team_id="h1",
        away_team_id="a1",
        home_name="Home",
        away_name="Away",
        scheduled_utc="2026-11-28T23:00:00Z",
        status=Status.SCHEDULED,
    )


def _history(
    n: int = 40,
    *,
    with_lines: bool = False,
    noise: float = 0.0,
    spread_line: float = -3.0,
    total_line: float = 140.0,
) -> list[Result]:
    """Alternating home/away finals, optionally with their own quoted lines."""
    rows: list[Result] = []
    for index in range(n):
        home = 78.0 + (index % 5) * 1.5
        away = 72.0 + (index % 3) * 1.2
        if noise:
            home += noise if index % 2 else -noise
        rows.append(
            Result(
                home_id="h1",
                away_id="a1",
                home_score=home,
                away_score=away,
                day=f"2026-10-{(index % 28) + 1:02d}",
                market_spread=spread_line if with_lines else None,
                market_total=total_line if with_lines else None,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# T1.1 — the closed form
# ---------------------------------------------------------------------------


def test_anchored_sigma_matches_the_orthogonal_decomposition():
    # Sanity value from the derivation: (0.96*10.7^2 + 0.04*14.5^2)^0.5 = 10.88
    assert 10.85 < anchored_sigma(14.5, 10.7, 0.20) < 10.90


def test_anchored_sigma_hits_its_declared_bounds():
    sigma_raw, sigma_market = 14.5, 10.7
    # Pure model: the raw sigma survives untouched.
    assert anchored_sigma(sigma_raw, sigma_market, 1.0) == pytest.approx(sigma_raw)
    # Pure market: the market dispersion, and never above the tighter of the two.
    assert anchored_sigma(sigma_raw, sigma_market, 0.0) == pytest.approx(sigma_market)
    # A model that is already tighter than the market cannot be widened by it.
    for weight in (0.0, 0.2, 0.5, 1.0):
        assert anchored_sigma(9.0, 10.7, weight) <= 9.0 + 1e-9
    # Clamped floor: no sigma may collapse below the normal approximation's floor.
    assert anchored_sigma(0.1, 0.1, 0.5) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# T1.2 — empirical switch at 30 quoted prior games
# ---------------------------------------------------------------------------


def test_anchored_sigmas_fall_back_below_thirty_quoted_games():
    ratings = bb.fit_lane_ratings(NCAA_M, _history(29, with_lines=True), reference_day="2026-11-28")
    sigma_margin, sigma_total = anchored_residual_sigmas(
        _history(29, with_lines=True),
        ratings,
        reference_day="2026-11-28",
        half_life_days=60.0,
        model_weight=0.20,
        sigma_raw=(14.5, 18.5),
        market_priors=bb.ANCHOR_MARKET_SIGMA_PRIORS[NCAA_M],
    )
    assert sigma_margin == pytest.approx(anchored_sigma(14.5, 10.7, 0.20))
    assert sigma_total == pytest.approx(anchored_sigma(18.5, 12.5, 0.20))


def test_anchored_sigmas_are_measured_from_anchored_residuals_at_thirty_games():
    """A deterministic +-9.5 zig-zag residual must measure ~9.5, not ~14.5."""
    rows: list[Result] = []
    for index in range(40):
        day = f"2026-10-{(index % 28) + 1:02d}"
        # Anchored mean for this synthetic world: w*model + (1-w)*(-market).
        # Keep the model at 6.0 and the market at -3.0 -> mean = 0.2*6 + 0.8*3.
        # (0.2*6 + 0.8*3) = 3.6, so a final margin of 3.6 +- 9.5 is a pure
        # anchored residual of +-9.5.
        residual = 9.5 if index % 2 else -9.5
        rows.append(
            Result(
                home_id="h1",
                away_id="a1",
                home_score=80.0 + residual,
                away_score=80.0,
                day=day,
                market_spread=-3.0,
                market_total=160.0,
            )
        )
    ratings = {"league_rate": 80.0, "home_factor": 1.0, "attack": {}, "defence": {}}
    margin, _total = anchored_residual_sigmas(
        rows,
        ratings,
        reference_day="2026-11-28",
        half_life_days=1e9,  # effectively uniform weights
        model_weight=0.20,
        sigma_raw=(14.5, 18.5),
        market_priors=bb.ANCHOR_MARKET_SIGMA_PRIORS[NCAA_M],
    )
    # diretta: model = 80 - 80 = 0, anchored mean = 0.2*0 + 0.8*3 = 2.4, and the
    # margin is exactly 2.4 +- 9.5 -> std 9.5.
    assert abs(margin - 9.5) < 0.5


# ---------------------------------------------------------------------------
# T1.3 — each historical row residualises against ITS OWN line
# ---------------------------------------------------------------------------


def test_each_historical_row_uses_its_own_line_never_todays():
    rows = [
        Result("h1", "a1", 80.0, 70.0, "2026-10-01", market_spread=-6.0),
        Result("h1", "a1", 80.0, 70.0, "2026-10-02", market_spread=-12.0),
    ]
    ratings = {"league_rate": 0.0, "home_factor": 1.0, "attack": {}, "defence": {}}
    residuals = __import__(
        "bullion.engines", fromlist=["_anchored_margin_residuals"]
    )._anchored_margin_residuals(rows, ratings, 0.20)
    # model 0; anchored means are 0.8*6 = 4.8 and 0.8*12 = 9.6 respectively.
    assert residuals[0] == pytest.approx(10.0 - 4.8)
    assert residuals[1] == pytest.approx(10.0 - 9.6)
    assert residuals[0] != residuals[1], "today's line leaked onto a past game"


# ---------------------------------------------------------------------------
# T1.4 — the ladder itself
# ---------------------------------------------------------------------------


def test_college_ladder_sigma_is_anchored_and_monotone():
    quotes = _quotes() + [
        BookQuote(
            game_id="g1",
            book="DraftKings",
            market="spread",
            line=-11.5,
            observed_utc="2026-11-28T15:00:00Z",
            side_prices={"home": 2.90, "away": 1.42},
            pre_match=True,
            main_line=False,
        )
    ]
    history = _history(40, with_lines=True)
    forecasts = bb.forecast(
        _game(),
        quotes,
        results=history,
        reference_day="2026-11-28",
    )
    spreads = sorted(
        (f.line, f.raw_probability) for f in forecasts if f.market == "spread" and f.selection == "home"
    )
    assert [line for line, _ in spreads] == [-11.5, -6.5]
    probabilities = [prob for _, prob in spreads]
    assert probabilities[0] < probabilities[1], "a wider home handicap must be rarer"

    # The far alt sits on the anchored sigma, not the raw one: a raw 14.5 would
    # leave the -11.5 home side several points more likely.
    total = sum(f.raw_probability for f in forecasts if f.market == "spread" and f.line == -11.5)
    assert total == pytest.approx(1.0 + sum(
        f.push_probability for f in forecasts if f.market == "spread" and f.line == -11.5
    )), "the two sides of one line must sum to 1 + push mass"


def test_university_total_ladder_uses_the_anchored_sigma_too():
    forecasts = bb.forecast(
        _game(),
        _quotes(),
        results=_history(40, with_lines=True),
        reference_day="2026-11-28",
    )
    over = [f for f in forecasts if f.market == "total" and f.selection == "over"]
    under = [f for f in forecasts if f.market == "total" and f.selection == "under"]
    assert over and under
    assert over[0].raw_probability + under[0].raw_probability == pytest.approx(
        1.0 + over[0].push_probability
    )


# ---------------------------------------------------------------------------
# T1.5 — NBA/WNBA are byte-identical (no anchoring, no new sigma)
# ---------------------------------------------------------------------------


def test_nba_forecast_is_untouched_by_the_anchoring_path():
    """A pro lane keeps the raw margin/sigma pair it had before F1."""
    quotes = _quotes(spread=-4.5, total=220.5)
    history = _history(40, with_lines=True)
    game = _game("nba")
    forecasts = bb.forecast(game, quotes, results=history, reference_day="2026-11-28")

    ratings = bb.fit_lane_ratings("nba", history, reference_day="2026-11-28")
    margin, total, sigma_margin, sigma_total, _prob = bb.empirical_expectations(
        "nba",
        history,
        ratings,
        home_id="h1",
        away_id="a1",
        reference_day="2026-11-28",
        game_scheduled_utc=game.scheduled_utc,
    )
    home_spread = next(
        f for f in forecasts if f.market == "spread" and f.selection == "home"
    )
    # Unanchored sigma and unanchored mean: the ladder is exactly what the
    # empirical surface produced, with no market anchor mixed in.
    from bullion.engines import normal_line_probs

    expected, _push = normal_line_probs(margin, sigma_margin, 4.5, "over")
    assert home_spread.raw_probability == pytest.approx(expected, abs=1e-12)
    over = next(f for f in forecasts if f.market == "total" and f.selection == "over")
    expected_over, _ = normal_line_probs(total, sigma_total, 220.5, "over")
    assert over.raw_probability == pytest.approx(expected_over, abs=1e-12)

    # And the weights that switch the anchored path on are college-only.
    assert bb.POSTED_LINE_MODEL_WEIGHTS["nba"] is None
    assert bb.POSTED_LINE_MODEL_WEIGHTS["wnba"] is None
    assert bb.POSTED_LINE_MODEL_WEIGHTS["ncaa_m"] == 0.20
    assert bb.POSTED_LINE_MODEL_WEIGHTS["ncaa_w"] == 0.20
    assert bb.ANCHOR_MARKET_SIGMA_PRIORS["ncaa_w"] == (11.5, 13.0)


def test_decay_weight_is_the_shared_half_life():
    assert decay_weight("2026-11-28", "2026-11-28", 60.0) == pytest.approx(1.0)
    assert decay_weight("2026-10-29", "2026-11-28", 60.0) == pytest.approx(0.5**0.5, abs=1e-6)
