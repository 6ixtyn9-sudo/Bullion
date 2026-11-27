"""End-to-end gate -> receipt -> settlement -> replay contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bullion.calibration import CalibrationBook
from bullion.models import LEAGUES, ActualResult, BookQuote, Forecast, Game, Status
from bullion.replay import (
    ReplaySpec,
    check_parity,
    compare_variants,
    load_evidence,
    merge_days,
    replay,
    select_tickets,
)
from bullion.tickets import (
    GatePolicy,
    build_accas,
    eligible_rows,
    expected_value,
    freeze_receipt,
    pick_row,
    render_card,
    roi_band,
    score_forecasts,
    settle_lane,
)

DAY = "2026-11-28"
SEGMENT = ("nhl", "moneyline", "full_game", "nhl_poisson")
SOURCE = "nhl_poisson"


def _calibrated_report(n: int = 40, probability: float = 0.62, wins: int = 25):
    book = CalibrationBook()
    for index in range(n):
        book.add(SEGMENT, probability, 1 if index < wins else 0, "2026-11-01", f"hist{index}")
    return book.report(SEGMENT, DAY)


def _game(index: int, *, league: str = "nhl", final: str = "win") -> Game:
    game = Game(
        game_id=f"g{league}{index}",
        league=league,
        season="2026",
        home_team_id=f"h{index}",
        away_team_id=f"a{index}",
        home_name=f"Home {index}",
        away_name=f"Away {index}",
        scheduled_utc=f"2026-11-28T{18 + index % 4}:00:00Z",
        status=Status.SCHEDULED,
    )
    if final == "win":
        game.actual = ActualResult(home_score=4, away_score=1)
    elif final == "loss":
        game.actual = ActualResult(home_score=1, away_score=4)
    elif final == "push":
        game.actual = ActualResult(home_score=2, away_score=2)
    if final:
        game.status = Status.FINAL
    return game


def _pool(
    tmp_path: Path,
    *,
    n: int = 4,
    price: float = 1.95,
    probability: float = 0.62,
    report=None,
    prior_roi=None,
    league_key: str = "nhl",
    source: str = SOURCE,
    probabilities=None,
):
    league = LEAGUES[league_key]
    probability_at = (
        (lambda index: probability)
        if probabilities is None
        else (lambda index: probabilities[index % len(probabilities)])
    )
    if report is None:
        report = _calibrated_report(probability=probability)
    reports = {(league_key, "moneyline", "full_game", source): report}
    forecasts = []
    quotes = []
    for index in range(n):
        game = _game(index, league=league_key)
        forecast = Forecast(
            game_id=game.game_id,
            league=league_key,
            scheduled_utc=game.scheduled_utc,
            home_name=game.home_name,
            away_name=game.away_name,
            market="moneyline",
            selection="home",
            raw_probability=probability_at(index),
            source=source,
        )
        forecasts.append(forecast)
        quotes.append(
            BookQuote(
                game_id=game.game_id,
                book="DraftKings",
                market="moneyline",
                observed_utc="2026-11-28T12:00:00Z",
                side_prices={"home": price, "away": 1.9},
                pre_match=True,
            )
        )
    if prior_roi is None:
        prior_roi = {"status": "positive", "n": 8, "roi": 0.05}
    scored = score_forecasts(
        forecasts, quotes, league=league, reports=reports, prior_roi=prior_roi
    )
    return league, [pick_row(forecast, DAY) for forecast in scored]


def _freeze(tmp_path: Path, rows, *, league, legs: int = 2, limit: int = 1):
    eligible = eligible_rows(rows, policy=GatePolicy())
    tickets = build_accas(eligible, legs=legs, limit=limit)
    policy = {
        "requested_legs": legs,
        "limit": limit,
        "min_leg_odds": 1.20,
        "max_leg_odds": 3.00,
        "price_weight": 0.15,
        "allow_provisional": False,
        "require_disjoint_legs": True,
        "forbid_conflicting_game_sides": True,
    }
    receipt = freeze_receipt(
        tmp_path,
        league=league,
        day=DAY,
        captured_utc="2026-11-28T15:00:00Z",
        candidate_pool=rows,
        accas=tickets,
        selection_policy=policy,
        decision_state={"status": "frozen", "planned_cutoff_utc": "2026-11-28T14:00:00Z"},
    )
    return receipt, tickets


def _three_market_board(games: int = 67, league_key: str = "nhl"):
    """A real-sized board: ``games`` fixtures, each with three priced markets.

    Priced the way a live board is priced — a short moneyline favourite, and
    spread/total legs near even money. Pairs of favourites sit *below* the acca
    floor, so they dominate any confidence-ranked walk and starve it.
    """
    league = LEAGUES[league_key]
    surfaces = (
        ("moneyline", "home", None, 1.50, 0.78, 32),
        ("spread", "home", -1.5, 1.92, 0.55, 22),
        ("total", "over", 5.5, 1.98, 0.53, 21),
    )
    reports = {}
    for market, _selection, _line, _price, probability, wins in surfaces:
        segment = (league_key, market, "full_game", SOURCE)
        book = CalibrationBook()
        for index in range(40):
            book.add(
                segment,
                probability,
                1 if index < wins else 0,
                "2026-11-01",
                f"{market}{index}",
            )
        reports[segment] = book.report(segment, DAY)

    forecasts, quotes = [], []
    for index in range(games):
        game = _game(index, league=league_key)
        for market, selection, line, price, probability, _wins in surfaces:
            forecasts.append(
                Forecast(
                    game_id=game.game_id,
                    league=league_key,
                    scheduled_utc=game.scheduled_utc,
                    home_name=game.home_name,
                    away_name=game.away_name,
                    market=market,
                    selection=selection,
                    line=line,
                    raw_probability=probability,
                    source=SOURCE,
                )
            )
            other = "away" if market in {"moneyline", "spread"} else "under"
            quotes.append(
                BookQuote(
                    game_id=game.game_id,
                    book="DraftKings",
                    market=market,
                    observed_utc="2026-11-28T12:00:00Z",
                    side_prices={selection: price, other: 2.05},
                    line=line,
                    pre_match=True,
                )
            )
    scored = score_forecasts(
        forecasts,
        quotes,
        league=league,
        reports=reports,
        prior_roi={"status": "positive", "n": 8, "roi": 0.05},
    )
    return league, [pick_row(forecast, DAY) for forecast in scored]


def test_a_sixty_seven_fixture_board_still_returns_three_acca_lines():
    """P0-N1: the acca walk starved to zero on every slate of 46+ fixtures.

    The cheapest pairs on a big board are two short favourites, whose combined
    price sits under the ``@3.00`` floor. A bounded best-first walk spent its
    whole budget inside that useless region; with two legs the space is small
    enough to enumerate, so the floor can no longer hide the in-band pairs.
    """
    league, rows = _three_market_board()
    assert len(rows) == 201
    eligible = eligible_rows(rows, policy=GatePolicy())
    tickets = build_accas(
        eligible,
        legs=2,
        limit=3,
        min_combined_odds=3.00,
        max_combined_odds=4.50,
    )
    assert len(tickets) == 3, "a 67-fixture slate must still fill the lane"
    for ticket in tickets:
        assert 3.00 <= ticket["combined_decimal_odds"] <= 4.50
        fixtures = {leg["game_id"] for leg in ticket["legs"]}
        assert len(fixtures) == 2, "two legs from one fixture are one bet"
    used = [leg["game_id"] for ticket in tickets for leg in ticket["legs"]]
    assert len(set(used)) == len(used), "frozen accas never share a fixture"


def test_gate_rejects_uncalibrated_negative_ev_and_out_of_band_prices():
    _, rows = _pool(None, report=_calibrated_report(n=10), prior_roi={"status": "positive", "n": 9, "roi": 0.02})
    reasons = set(rows[0]["eligibility"]["reasons"])
    assert "calibration_not_sufficient_for_accas" in reasons
    assert not rows[0]["eligibility"]["acca_eligible"]

    _, rows = _pool(None, price=1.50)
    assert "non_positive_expected_value" in rows[0]["eligibility"]["reasons"]

    _, rows = _pool(None, price=3.40)
    assert "price_outside_ticket_band" in rows[0]["eligibility"]["reasons"]

    _, rows = _pool(None, prior_roi={"status": "insufficient_sample", "n": 3, "roi": None})
    assert "prior_roi_insufficient_sample" in rows[0]["eligibility"]["reasons"]


def test_cold_start_is_only_admitted_when_the_lane_has_zero_settled_picks():
    _, rows = _pool(None, prior_roi={"status": "cold_start", "n": 0, "roi": None})
    assert rows[0]["eligibility"]["acca_eligible"]
    assert rows[0]["eligibility"]["cold_start_admitted"]


def test_expected_value_accounts_for_push_mass():
    assert expected_value(0.5, 0.0, 2.0) == pytest.approx(0.0)
    assert expected_value(0.5, 0.1, 2.0) > 0.0


def test_receipt_is_ma_golide_schema_compatible_and_write_once(tmp_path):
    league, rows = _pool(tmp_path)
    receipt, tickets = _freeze(tmp_path, rows, league=league)
    assert receipt is not None
    assert receipt["schema_version"] == "magolide.paper_ticket_receipt"
    assert receipt["status"] == "frozen"
    assert receipt["paper_only"] is True
    assert receipt["stake_units_per_acca"] == 1.0
    assert receipt["stake_units_per_position"] == 1.0
    for key in ("accas", "singles", "candidate_pool", "selection_policy", "ticket_decision"):
        assert key in receipt
    assert len(receipt["accas"]) == 1
    assert len(receipt["accas"][0]["legs"]) == 2
    assert receipt["accas"][0]["combined_decimal_odds"] == pytest.approx(1.95**2, rel=1e-6)

    # Fields Ma_Golide's replay loader reads directly.
    row = receipt["candidate_pool"][0]
    for key in (
        "forecast_id",
        "pick_id",
        "game_id",
        "league",
        "scheduled_utc",
        "home",
        "away",
        "market",
        "period",
        "selection",
        "raw_probability",
        "calibrated_probability",
        "book",
        "eligibility",
        "calibration",
    ):
        assert key in row
    for key in (
        "provider",
        "book",
        "observed_utc",
        "pre_match",
        "main_line",
        "frozen",
        "price_provenance",
        "selected_decimal_price",
        "side_prices",
    ):
        assert key in row["book"]
    assert row["period"] == "full_game"
    assert row["market"] == "moneyline"

    frozen_bytes = (tmp_path / "nhl" / f"auto_tickets_{DAY}_decision.json").read_bytes()
    again = freeze_receipt(
        tmp_path,
        league=league,
        day=DAY,
        captured_utc="2026-11-29T15:00:00Z",
        candidate_pool=[],
        accas=[],
    )
    assert again is not None
    assert (tmp_path / "nhl" / f"auto_tickets_{DAY}_decision.json").read_bytes() == frozen_bytes


def test_no_bet_lock_is_terminal(tmp_path):
    league, rows = _pool(tmp_path, report=_calibrated_report(n=5))
    receipt, _ = _freeze(tmp_path, rows, league=league)
    assert receipt is None
    assert (tmp_path / "nhl" / f"auto_tickets_{DAY}_no_bet.json").exists()
    assert freeze_receipt(
        tmp_path,
        league=league,
        day=DAY,
        captured_utc="2026-11-28T16:00:00Z",
        candidate_pool=rows,
        accas=[],
    ) is None


def test_settlement_grades_pool_legs_and_positions(tmp_path):
    league, rows = _pool(tmp_path, n=4)
    receipt, _ = _freeze(tmp_path, rows, league=league)
    games = {game.game_id: game for game in (_game(index, final="win") for index in range(4))}
    report = settle_lane(tmp_path, league, games)
    assert report is not None
    assert report["settled_positions"] == 1
    assert report["wins"] == 1
    assert report["profit_units"] == pytest.approx(1.95**2 - 1)
    assert report["low_sample_flag"] is True
    day = report["history"][0]
    assert set(day["candidate_pool_outcomes"].values()) == {"win"}
    assert len(day["candidate_pool_outcomes"]) == 4
    assert day["positions"][0]["leg_outcomes"] == ["win", "win"]

    performance = json.loads(
        (tmp_path / "nhl" / "auto_tickets_performance.json").read_text()
    )
    assert performance["schema_version"] == "magolide.paper_ticket_performance"


def test_losing_leg_loses_the_acca(tmp_path):
    league, rows = _pool(tmp_path, n=4)
    receipt, _ = _freeze(tmp_path, rows, league=league)
    first_leg = receipt["accas"][0]["legs"][0]["game_id"]
    games = {}
    for index in range(4):
        game = _game(index, final="win")
        if game.game_id == first_leg:
            game = _game(index, final="loss")
        games[game.game_id] = game
    report = settle_lane(tmp_path, league, games)
    assert report["losses"] == 1
    assert report["profit_units"] == pytest.approx(-1.0)


def test_parity_is_vacuously_true_with_no_frozen_receipt():
    parity = check_parity([])
    assert parity["parity"] is True
    assert parity["vacuous"] is True
    assert parity["checked"] == 0


def test_parity_fails_and_the_harness_exits_non_zero_on_a_tampered_receipt(tmp_path):
    import importlib.util

    league, rows = _pool(tmp_path, n=4)
    _freeze(tmp_path, rows, league=league)
    receipt_path = next(tmp_path.glob("*/auto_tickets_*_decision.json"))
    receipt = json.loads(receipt_path.read_text())
    receipt["accas"][0]["combined_decimal_odds"] = 99.0  # frozen price tampered with
    receipt["accas"][0]["legs"][0]["price_evidence"]["selected_decimal_price"] = 2.50
    receipt_path.write_text(json.dumps(receipt))

    days = load_evidence(tmp_path)
    parity = check_parity(days)
    assert parity["parity"] is False
    assert parity["mismatched"] == 1
    assert parity["vacuous"] is False

    spec = importlib.util.spec_from_file_location(
        "bullion_replay_harness", Path(__file__).resolve().parents[1] / "scripts" / "replay_harness.py"
    )
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    assert harness.main(["--parity", "--localdata", str(tmp_path), "--json"]) == 1


def test_replay_parity_and_settled_replay(tmp_path):
    league, rows = _pool(tmp_path, n=4)
    _freeze(tmp_path, rows, league=league)
    games = {game.game_id: game for game in (_game(index, final="win") for index in range(4))}
    settle_lane(tmp_path, league, games)

    days = load_evidence(tmp_path)
    assert len(days) == 1
    parity = check_parity(days)
    assert parity["parity"] is True
    assert parity["checked"] == 1 and parity["matched"] == 1

    result = replay(days, ReplaySpec())
    assert result["settled_days"] == 1
    assert result["tickets_settled"] == 1
    assert result["profit_units"] == pytest.approx(1.95**2 - 1, rel=1e-6)
    assert result["low_sample"] is True


def test_three_leg_variant_rebuilds_from_the_same_frozen_pool(tmp_path):
    league, rows = _pool(tmp_path, n=4)
    _freeze(tmp_path, rows, league=league, legs=3)
    games = {game.game_id: game for game in (_game(index, final="win") for index in range(4))}
    settle_lane(tmp_path, league, games)
    days = load_evidence(tmp_path)
    assert check_parity(days)["parity"] is True
    assert len(days[0].receipt["accas"][0]["legs"]) == 3


def test_paired_ab_refuses_a_no_op_and_reports_low_sample(tmp_path):
    league, rows = _pool(tmp_path, n=4)
    _freeze(tmp_path, rows, league=league)
    games = {game.game_id: game for game in (_game(index, final="win") for index in range(4))}
    settle_lane(tmp_path, league, games)
    days = load_evidence(tmp_path)
    no_op = compare_variants(days, ReplaySpec(), ReplaySpec())
    assert no_op["status"] == "no_op"

    real = compare_variants(days, ReplaySpec(legs=2), ReplaySpec(legs=3))
    assert real["status"] in {"ok", "insufficient", "no_op"}
    if real["status"] == "ok":
        assert real["low_sample"] is True


def test_cross_sport_merge_builds_a_cross_league_acca(tmp_path):
    nhl_league, nhl_rows = _pool(tmp_path, n=2, probabilities=[0.66, 0.60])
    _freeze(tmp_path, nhl_rows, league=nhl_league)
    nfl_league, nfl_rows = _pool(tmp_path, n=2, league_key="nfl", probabilities=[0.70, 0.58])
    _freeze(tmp_path, nfl_rows, league=nfl_league)

    days = load_evidence(tmp_path)
    assert {day.league for day in days} == {"NHL", "NFL"}
    merged = merge_days(days)
    assert len(merged) == 1
    tickets = select_tickets(merged[0], ReplaySpec(legs=2, limit=3))
    assert tickets, "a cross-sport pool must produce at least one acca"
    cross = [
        ticket
        for ticket in tickets
        if len({leg["league"] for leg in ticket["legs"]}) == 2
    ]
    assert cross, "at least one acca must span both sports"
    assert all(3.0 <= ticket["combined_decimal_odds"] <= 4.0 for ticket in cross)


def test_daily_cap_limits_tickets_without_filler_legs(tmp_path):
    league, rows = _pool(tmp_path, n=8)
    receipt, _ = _freeze(tmp_path, rows, league=league, limit=3)
    assert len(receipt["accas"]) == 3
    used_legs = [
        leg["forecast_id"] for acca in receipt["accas"] for leg in acca["legs"]
    ]
    assert len(used_legs) == len(set(used_legs)), "no leg may be reused across tickets"
    assert all(len(acca["legs"]) == 2 for acca in receipt["accas"])


def test_missing_legs_produce_no_ticket_and_no_flat_curve(tmp_path):
    league, rows = _pool(tmp_path, n=1)
    assert build_accas(eligible_rows(rows), legs=2) == []
    days = load_evidence(tmp_path)
    assert days == []
    empty = replay(days, ReplaySpec())
    assert empty["settled_days"] == 0
    assert empty["mean_daily_log_growth"] is None
    assert "zero settled bet-days" in empty["note"]


def test_roi_band_requires_samples_and_recency():
    assert roi_band([], as_of_day=DAY)["status"] == "cold_start"
    few = [("2026-11-01", 1.9, "win")] * 4
    assert roi_band(few, as_of_day=DAY)["status"] == "insufficient_sample"
    wins = [("2026-11-01", 1.9, "win")] * 8
    band = roi_band(wins, as_of_day=DAY)
    assert band["status"] == "positive"
    losses = [("2026-11-01", 1.9, "loss")] * 8
    assert roi_band(losses, as_of_day=DAY)["status"] == "non_positive"


def test_card_renders_sast_and_paper_only_language(tmp_path):
    league, rows = _pool(tmp_path)
    receipt, _ = _freeze(tmp_path, rows, league=league)
    card = render_card(receipt)
    assert "SAST" in card
    assert "paper only" in card
    assert "No profit, edge, or readiness claim" in card
    assert "named-book pre-tip only" in card
    assert "calibrated n>=30" in card


def _three_way_pool(
    n: int = 4,
    *,
    selection: str = "home",
    probability: float = 0.55,
    leg_price: float = 1.95,
):
    """A soccer lane priced by a real three-way quote (draw included).

    The calibration segment is pooled over the three outcomes, which is what a
    1X2 market actually is: a single-outcome draw segment at p~0.3 would score a
    Brier that says nothing about the home/away prices being booked.
    """
    league = LEAGUES["eng.1"]
    source = "dixon_coles"
    segment = (league.key, "moneyline", "full_game", source)
    book = CalibrationBook()
    for index in range(40):
        book.add(segment, 0.55, 1 if index < 22 else 0, "2026-11-01", f"hist{index}")
        book.add(segment, 0.26, 1 if index < 10 else 0, "2026-11-01", f"hist{index}")
        book.add(segment, 0.19, 1 if index < 8 else 0, "2026-11-01", f"hist{index}")
    reports = {segment: book.report(segment, DAY)}

    forecasts, quotes = [], []
    for index in range(n):
        game = _game(index, league=league.key)
        forecasts.append(
            Forecast(
                game_id=game.game_id,
                league=league.key,
                scheduled_utc=game.scheduled_utc,
                home_name=game.home_name,
                away_name=game.away_name,
                market="moneyline",
                selection=selection,
                raw_probability=probability,
                alternative_probabilities={"home": 0.55, "draw": 0.26, "away": 0.19},
                source=source,
            )
        )
        prices = {"home": leg_price, "draw": 3.40, "away": 4.20}
        prices[selection] = leg_price
        quotes.append(
            BookQuote(
                game_id=game.game_id,
                book="DraftKings",
                market="moneyline",
                observed_utc="2026-11-28T09:30:00Z",
                side_prices=prices,
                pre_match=True,
            )
        )
    scored = score_forecasts(
        forecasts,
        quotes,
        league=league,
        reports=reports,
        prior_roi={"status": "positive", "n": 8, "roi": 0.05},
    )
    return league, [pick_row(forecast, DAY) for forecast in scored]


def test_three_way_quote_prices_a_home_leg_and_freezes_within_the_band(tmp_path):
    league, rows = _three_way_pool()
    eligible = eligible_rows(rows, policy=GatePolicy())
    assert eligible, "calibrated 0.55 at 1.95 is +EV behind a named-book quote"
    assert all(row["book"]["side_prices"]["draw"] == pytest.approx(3.40) for row in eligible), (
        "the quote carries the draw price, so the moneyline is a genuine three-way price"
    )

    tickets = build_accas(eligible, legs=2, limit=1, min_combined_odds=3.00, max_combined_odds=4.50)
    assert len(tickets) == 1
    ticket = tickets[0]
    assert 3.00 <= ticket["combined_decimal_odds"] <= 4.50
    assert all(leg["selection"] == "home" for leg in ticket["legs"])
    assert all(
        leg["book"]["selected_decimal_price"] == pytest.approx(1.95) for leg in ticket["legs"]
    )

    receipt, _ = _freeze(tmp_path, rows, league=league)
    assert receipt["schema_version"] == "magolide.paper_ticket_receipt"
    assert [leg["selection"] for leg in receipt["accas"][0]["legs"]] == ["home", "home"]
    assert all(
        leg["price_evidence"]["selected_decimal_price"] == pytest.approx(1.95)
        for leg in receipt["accas"][0]["legs"]
    )

    finals = {}
    for row in rows:
        game = _game(0, league="eng.1")
        game.game_id = row["game_id"]
        game.actual = ActualResult(home_score=2, away_score=1)
        game.status = Status.FINAL
        finals[game.game_id] = game
    report = settle_lane(tmp_path, league, finals)
    assert report["settled_positions"] == 1
    assert report["wins"] == 1


def test_a_draw_leg_is_priced_but_the_leg_band_keeps_it_off_the_ticket(tmp_path):
    """A draw is quotable and gradable; it is the [1.20, 3.00] band that excludes it."""
    league, rows = _three_way_pool(selection="draw", probability=0.28, leg_price=3.40)
    row = rows[0]
    assert row["book"]["selected_decimal_price"] == pytest.approx(3.40)
    assert row["book"]["side_prices"]["draw"] == pytest.approx(3.40)
    assert row["eligibility"]["brier_cap"] == 0.22
    eligible = eligible_rows(rows, policy=GatePolicy())
    assert eligible == []
    assert "price_outside_ticket_band" in row["eligibility"]["reasons"]


def test_draw_selection_loses_when_the_game_is_not_level():
    from bullion.tickets import resolve_outcome

    assert resolve_outcome("moneyline", "draw", None, 2.0, 2.0) == "win"
    assert resolve_outcome("moneyline", "draw", None, 2.0, 1.0) == "loss"
    assert resolve_outcome("moneyline", "home", None, 2.0, 2.0) == "loss"


# ---------------------------------------------------------------------------
# Silent-stall regressions: the ROI band and the no-bet lock (2026-11-26 audit)
# ---------------------------------------------------------------------------


def _final_games(game_ids, *, home_score: int = 4, away_score: int = 1) -> dict:
    games = {}
    for game_id in game_ids:
        game = _game(0)
        game.game_id = game_id
        game.actual = ActualResult(home_score=home_score, away_score=away_score)
        game.status = Status.FINAL
        games[game_id] = game
    return games


def test_settled_pick_history_counts_only_the_model_side(tmp_path):
    """Bug 3: grading both sides made ROI identically the bookmaker's vig."""
    league, rows = _pool(tmp_path, n=1)  # home @1.95, calibrated p=0.62 (+EV)
    home = rows[0]
    away = json.loads(json.dumps(home))
    away.update(
        {
            "forecast_id": "fid-away",
            "pick_id": "fid-away",
            "selection": "away",
            "calibrated_probability": 0.35,
            "raw_probability": 0.35,
            "book": {**home["book"], "selected_decimal_price": 1.95},
        }
    )
    freeze_receipt(
        tmp_path,
        league=league,
        day=DAY,
        captured_utc="2026-11-28T15:00:00Z",
        candidate_pool=[home, away],
        accas=[],
        selection_policy={"min_leg_odds": 1.20, "max_leg_odds": 3.00},
        decision_state={"status": "frozen_no_bet"},
    )
    settle_lane(tmp_path, league, _final_games([home["game_id"]]))

    from bullion.tickets import settled_pick_history

    picks = settled_pick_history(tmp_path, league)
    assert picks == [(DAY, pytest.approx(1.95), "win")], (
        "the -EV away side must never enter the ROI band"
    )


def test_a_postponed_fixture_is_counted_once_by_prior_roi(tmp_path):
    """P2-N3: a refrozen postponement is the same wager, not a second pick.

    The fixture is pushed to a later day, refunded at the first freeze and
    frozen again on the new date at a new price. Two receipts, one fixture: the
    ROI band must see one settled pick, at the price that was actually staked.
    """
    from bullion.tickets import settled_pick_history

    league, rows = _pool(tmp_path, n=3)
    policy = {"min_leg_odds": 1.20, "max_leg_odds": 3.00}
    first = rows[0]
    freeze_receipt(
        tmp_path,
        league=league,
        day="2026-11-28",
        captured_utc="2026-11-28T15:00:00Z",
        candidate_pool=[first],
        accas=[],
        selection_policy=policy,
    )
    refrozen = json.loads(json.dumps(first))
    refrozen.update({"date": "2026-11-29", "forecast_id": "fid-refrozen", "pick_id": "fid-refrozen"})
    refrozen["book"].update(
        {"observed_utc": "2026-11-29T12:00:00Z", "selected_decimal_price": 2.05}
    )
    freeze_receipt(
        tmp_path,
        league=league,
        day="2026-11-29",
        captured_utc="2026-11-29T15:00:00Z",
        candidate_pool=[refrozen],
        accas=[],
        selection_policy=policy,
    )
    settle_lane(tmp_path, league, _final_games([first["game_id"]]))

    picks = settled_pick_history(tmp_path, league)
    assert picks == [("2026-11-28", pytest.approx(1.95), "win")], (
        "one fixture contributes one pick, at the price of its first freeze"
    )


def test_no_bet_lock_preserves_and_grades_its_pool(tmp_path):
    """Bug 4: a no-bet lock used to discard the pool, so nothing accrued."""
    league, rows = _pool(tmp_path, n=4, report=_calibrated_report(n=5))
    receipt, _ = _freeze(tmp_path, rows, league=league)
    assert receipt is None
    no_bet_path = tmp_path / "nhl" / f"auto_tickets_{DAY}_no_bet.json"
    no_bet = json.loads(no_bet_path.read_text())
    assert no_bet["schema_version"] == "magolide.no_bet_decision"
    assert len(no_bet["candidate_pool"]) == 4
    assert no_bet["selection_policy"]["requested_legs"] == 2
    assert no_bet["positions"] == []
    assert no_bet["decision_id"]

    report = settle_lane(tmp_path, league, _final_games([row["game_id"] for row in rows]))
    day = next(item for item in report["history"] if item["date"] == DAY)
    assert day["status"] == "frozen_no_bet"
    assert day["positions"] == []
    assert len(day["candidate_pool_outcomes"]) == 4
    assert set(day["candidate_pool_outcomes"].values()) == {"win"}
    # A no-bet day moves ticket P&L by exactly nothing.
    assert report["settled_positions"] == 0
    assert report["profit_units"] == 0.0


def test_no_bet_day_replays_as_a_zero_ticket_day_with_parity(tmp_path):
    league, rows = _pool(tmp_path, n=4, report=_calibrated_report(n=5))
    _freeze(tmp_path, rows, league=league)
    settle_lane(tmp_path, league, _final_games([row["game_id"] for row in rows]))

    days = load_evidence(tmp_path)
    assert len(days) == 1, "a no-bet lock with a frozen pool is replayable evidence"
    assert days[0].candidate_pool
    parity = check_parity(days)
    assert parity["parity"] is True, parity["mismatches"]
    assert parity["checked"] == 1 and parity["matched"] == 1
    result = replay(days, ReplaySpec())
    assert result["no_ticket_days"] == 1
    assert result["settled_days"] == 0

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "bullion_replay_harness",
        Path(__file__).resolve().parents[1] / "scripts" / "replay_harness.py",
    )
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    assert harness.main(["--parity", "--localdata", str(tmp_path), "--json"]) == 0


def test_prior_roi_escapes_insufficient_sample_through_no_bet_pools(tmp_path):
    """1 <= n < 5 must be escapable: graded no-bet pools keep adding picks.

    Day 1 settles three picks and freezes no-bet (blocked by
    ``prior_roi_insufficient_sample``). Without grading that day's pool, day 2
    would freeze no-bet for the same reason, add nothing, and deadlock forever.
    """
    from bullion.tickets import settled_pick_history

    league, rows = _pool(tmp_path, n=5)
    games = _final_games([row["game_id"] for row in rows])
    policy = {"min_leg_odds": 1.20, "max_leg_odds": 3.00}

    freeze_receipt(
        tmp_path,
        league=league,
        day="2026-11-26",
        captured_utc="2026-11-26T15:00:00Z",
        candidate_pool=rows[:3],
        accas=[],
        selection_policy=policy,
    )
    settle_lane(tmp_path, league, games)
    blocked = roi_band(settled_pick_history(tmp_path, league), as_of_day="2026-11-28")
    assert blocked["status"] == "insufficient_sample"
    assert blocked["n"] == 3

    freeze_receipt(
        tmp_path,
        league=league,
        day="2026-11-27",
        captured_utc="2026-11-27T15:00:00Z",
        candidate_pool=rows[3:],
        accas=[],
        selection_policy=policy,
    )
    settle_lane(tmp_path, league, games)
    escaped = roi_band(settled_pick_history(tmp_path, league), as_of_day="2026-11-29")
    assert escaped["n"] == 5
    assert escaped["status"] in {"positive", "non_positive"}, escaped


def test_harness_reports_no_bet_days_explicitly(tmp_path):
    import importlib.util

    league, rows = _pool(tmp_path, n=4, report=_calibrated_report(n=5))
    _freeze(tmp_path, rows, league=league)
    settle_lane(tmp_path, league, _final_games([row["game_id"] for row in rows]))
    spec = importlib.util.spec_from_file_location(
        "bullion_replay_harness",
        Path(__file__).resolve().parents[1] / "scripts" / "replay_harness.py",
    )
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    days = harness.load_evidence(tmp_path)
    assert len(days) == 1
    report_meta = harness.check_parity(days)
    assert report_meta["checked"] == 1 and report_meta["parity"] is True


def test_no_bet_pool_feeds_counterfactual_replay_at_parity(tmp_path):
    """A declined pool is still replayable evidence, not a dead end.

    Four eligible legs at 2.90 combine to 8.41 for two legs, outside the frozen
    [3.00, 4.50] acca band — a genuine no-bet at parity. Widening that band in
    the replay harness then has to be able to bet the same frozen pool.
    """
    league, rows = _pool(tmp_path, n=4, price=2.90)
    policy = {
        "requested_legs": 2,
        "limit": 1,
        "min_leg_odds": 1.20,
        "max_leg_odds": 3.00,
        "price_weight": 0.15,
        "min_combined_odds": 3.00,
        "max_combined_odds": 4.50,
    }
    eligible = eligible_rows(rows, policy=GatePolicy())
    assert len(eligible) == 4
    assert build_accas(
        eligible, legs=2, limit=1, min_combined_odds=3.00, max_combined_odds=4.50
    ) == []

    assert (
        freeze_receipt(
            tmp_path,
            league=league,
            day=DAY,
            captured_utc="2026-11-28T15:00:00Z",
            candidate_pool=rows,
            accas=[],
            selection_policy=policy,
            decision_state={"status": "frozen_no_bet", "planned_cutoff_utc": "2026-11-28T14:00:00Z"},
        )
        is None
    )
    settle_lane(tmp_path, league, _final_games([row["game_id"] for row in rows]))

    days = load_evidence(tmp_path)
    assert check_parity(days)["parity"] is True
    frozen = replay(days, ReplaySpec())
    assert frozen["no_ticket_days"] == 1 and frozen["settled_days"] == 0

    wide = replay(days, ReplaySpec(min_combined_odds=0.0, max_combined_odds=1_000_000.0))
    assert wide["settled_days"] == 1
    assert wide["tickets_settled"] == 1
    assert wide["profit_units"] > 0


def _level_game(game_id: str = "nfl-tie", *, league: str = "nfl", home=20.0, away=20.0) -> Game:
    game = Game(
        game_id=game_id,
        league=league,
        season="2026",
        home_team_id="h1",
        away_team_id="a1",
        home_name="Home Club",
        away_name="Away Club",
        scheduled_utc="2026-11-28T18:00:00Z",
        status=Status.FINAL,
        actual=ActualResult(home_score=home, away_score=away),
    )
    return game


def test_a_two_way_moneyline_tie_pushes_and_a_three_way_tie_is_a_draw():
    from bullion.tickets import resolve_outcome

    for selection in ("home", "away"):
        assert resolve_outcome("moneyline", selection, None, 20.0, 20.0, has_draw=False) == "push"
    assert resolve_outcome("moneyline", "draw", None, 20.0, 20.0, has_draw=True) == "win"
    assert resolve_outcome("moneyline", "home", None, 20.0, 20.0, has_draw=True) == "loss"
    assert resolve_outcome("moneyline", "home", None, 24.0, 20.0, has_draw=False) == "win"
    assert resolve_outcome("moneyline", "away", None, 24.0, 20.0, has_draw=False) == "loss"


def _nfl_leg(forecast_id: str, game_id: str, *, price: float = 1.95) -> dict:
    return {
        "forecast_id": forecast_id,
        "pick_id": forecast_id,
        "game_id": game_id,
        "league": "nfl",
        "market": "moneyline",
        "period": "full_game",
        "selection": "home",
        "line": None,
        "raw_probability": 0.6,
        "calibrated_probability": 0.62,
        "push_probability": 0.0,
        "eligibility": {"acca_eligible": True, "reasons": []},
        "book": {
            "book": "DraftKings",
            "selected_decimal_price": price,
            "side_prices": {"home": price, "away": 1.9},
        },
    }


def test_a_pushed_leg_leaves_the_rest_of_the_acca_live(tmp_path):
    """A two-way tie voids one leg; the other leg still settles at its own price."""
    league = LEAGUES["nfl"]
    legs = [_nfl_leg("fid-tie", "nfl-tie"), _nfl_leg("fid-win", "nfl-win")]
    freeze_receipt(
        tmp_path,
        league=league,
        day=DAY,
        captured_utc="2026-11-28T15:00:00Z",
        candidate_pool=[dict(leg) for leg in legs],
        accas=[
            {
                "acca_id": "acca-1",
                "legs": [dict(leg) for leg in legs],
                "leg_count": 2,
                "combined_decimal_odds": 1.95 * 1.95,
                "estimated_joint_win_probability": 0.38,
                "stake_units": 1.0,
            }
        ],
        selection_policy={"min_leg_odds": 1.20, "max_leg_odds": 3.00},
    )
    games = {
        "nfl-tie": _level_game("nfl-tie"),
        "nfl-win": _level_game("nfl-win", home=27.0, away=17.0),
    }
    report = settle_lane(tmp_path, league, games)

    position = report["history"][0]["positions"][0]
    assert position["leg_outcomes"] == ["push", "win"]
    assert position["push_legs"] == 1
    assert position["effective_decimal_odds"] == pytest.approx(1.95)
    assert position["profit_units"] == pytest.approx(0.95)
    assert report["profit_units"] == pytest.approx(0.95)

    all_push = {
        "nfl-tie": _level_game("nfl-tie"),
        "nfl-win": _level_game("nfl-win", home=20.0, away=20.0),
    }
    report = settle_lane(tmp_path, league, all_push)
    position = report["history"][0]["positions"][0]
    assert position["grade_status"] == "push"
    assert position["profit_units"] == 0.0


def _viable_row(
    game_id: str,
    market: str,
    price: float,
    probability: float,
    *,
    line: float | None = None,
    selection: str = "home",
) -> dict:
    return {
        "forecast_id": f"{game_id}-{market}-{selection}",
        "pick_id": f"{game_id}-{market}-{selection}",
        "game_id": game_id,
        "league": "nfl",
        "market": market,
        "period": "full_game",
        "selection": selection,
        "line": line,
        "raw_probability": probability,
        "calibrated_probability": probability,
        "push_probability": 0.0,
        "eligibility": {"acca_eligible": True, "reasons": []},
        "book": {
            "book": "DraftKings",
            "selected_decimal_price": price,
            "side_prices": {selection: price, "away": 1.9, "under": 1.9},
        },
    }


def test_eligible_rows_keeps_the_priced_markets_an_acca_needs():
    """One row per game discarded exactly the legs that reach the @3.00 floor."""
    rows = []
    for index in range(4):
        game = f"game{index}"
        rows.append(_viable_row(game, "moneyline", 1.30, 0.80))
        rows.append(_viable_row(game, "spread", 1.95, 0.55, line=-2.5))
    eligible = eligible_rows(rows, policy=GatePolicy())

    assert len(eligible) == 8, "both the favourite and the spread stay available"
    tickets = build_accas(
        eligible,
        legs=2,
        limit=1,
        min_combined_odds=3.00,
        max_combined_odds=4.50,
    )
    assert tickets, "two @1.95 legs combine to @3.80, inside the band"
    ticket = tickets[0]
    assert ticket["combined_decimal_odds"] == pytest.approx(1.95**2)
    assert {leg["market"] for leg in ticket["legs"]} == {"spread"}
    assert len({leg["game_id"] for leg in ticket["legs"]}) == 2


def test_acca_assembly_refuses_two_legs_from_one_fixture():
    """Same-game exclusivity is enforced where the combination is built."""
    rows = [
        _viable_row("one-game", "moneyline", 1.30, 0.80),
        _viable_row("one-game", "spread", 1.95, 0.55, line=-2.5),
    ]
    assert build_accas(rows, legs=2, min_leg_odds=1.20, max_leg_odds=3.00) == []


def test_roi_band_ignores_pushed_picks():
    wins = [("2026-11-01", 1.20, "win")] * 5
    losses = [("2026-11-01", 1.90, "loss")] * 2
    baseline = roi_band(wins + losses, as_of_day=DAY)
    with_pushes = roi_band(wins + losses + [("2026-11-01", 1.95, "push")] * 20, as_of_day=DAY)

    assert with_pushes == baseline, "a push returns its own stake; it is not evidence"
    assert baseline["n"] == 7


def test_replay_settles_a_ticket_even_when_a_same_day_leg_is_pending():
    from bullion.replay import DayEvidence

    pool = []
    for index in range(4):
        pool.append(_viable_row(f"game{index}", "moneyline", 1.95, 0.70))
    day = DayEvidence(
        date=DAY,
        lane="nfl",
        league="NFL",
        decision_id="d1",
        frozen_at="2026-11-28T15:00:00Z",
        receipt={"schema_version": "magolide.paper_ticket_receipt"},
        candidate_pool=pool,
        selection_policy={"requested_legs": 2, "limit": 2},
        outcomes_by_fid={},
    )
    rebuilt = select_tickets(day, ReplaySpec(legs=2, limit=2))
    assert len(rebuilt) == 2
    settled_ids = {leg["forecast_id"] for leg in rebuilt[0]["legs"]}
    day.outcomes_by_fid = {forecast_id: "win" for forecast_id in settled_ids}

    result = replay([day], ReplaySpec(legs=2, limit=2))
    assert result["tickets_settled"] == 1
    assert result["pending_days"] == 1
    assert result["settled_days"] == 1, "one settled ticket is a settled day"
    assert result["profit_units"] == pytest.approx(1.95**2 - 1.0, rel=1e-6)
