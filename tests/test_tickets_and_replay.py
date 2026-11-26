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
SEGMENT = ("nhl", "moneyline", "full_game", "nhl_poisson_v1")
SOURCE = "nhl_poisson_v1"


def _calibrated_report(n: int = 40, probability: float = 0.62, wins: int = 25):
    book = CalibrationBook()
    for index in range(n):
        book.add(SEGMENT, probability, 1 if index < wins else 0, "2026-11-01")
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
