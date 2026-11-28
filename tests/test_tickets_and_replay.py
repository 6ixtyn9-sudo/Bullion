"""End-to-end gate -> receipt -> settlement -> replay contract tests."""

from __future__ import annotations

import itertools
import json
import math
import random
from pathlib import Path

import pytest

from bullion.calibration import CalibrationBook
from bullion.models import LEAGUES, ActualResult, BookQuote, Forecast, Game, Status
from bullion.replay import (
    ReplaySpec,
    check_parity,
    policy_limits as replay_policy_limits,
    compare_variants,
    load_evidence,
    merge_days,
    replay,
    select_tickets,
)
from bullion.tickets import (
    EDGE_FLAG_THRESHOLD,
    MAX_CLAIMED_EDGE,
    PRICE_WEIGHT,
    RANKING_METRIC_BLEND,
    RANKING_METRIC_KELLY,
    GatePolicy,
    attach_quotes,
    build_accas,
    eligible_rows,
    expected_value,
    kelly_log_growth,
    kelly_stake_fraction,
    freeze_receipt,
    latest_quotes,
    pick_row,
    render_card,
    render_performance,
    roi_band,
    row_price,
    score_forecasts,
    grade_row,
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
    assert "paper only" in card.lower()
    assert "No profit, edge, or readiness claim" in card
    assert "named-book pre-tip only" in card
    assert "calibrated n>=30" in card
    # Nothing settled yet, so every leg and the ticket itself stay PENDING.
    assert "⏳ PENDING" in card
    assert "FINAL" not in card


def test_settled_card_moves_to_the_lane_root_and_shows_the_result(tmp_path):
    """The dated card is a lane-root view that refreshes; the receipt is frozen."""
    league, rows = _pool(tmp_path)
    _freeze(tmp_path, rows, league=league)
    lane = tmp_path / "nhl"
    receipt_path = lane / f"auto_tickets_{DAY}_decision.json"
    frozen_bytes = receipt_path.read_bytes()

    settle_lane(tmp_path, league, _final_games([row["game_id"] for row in rows]))

    assert not (lane / "cards").exists(), "the card no longer hides inside cards/"
    card = (lane / f"auto_tickets_{DAY}.txt").read_text()
    assert card.startswith(f"BULLION PAPER TICKETS — NHL {DAY}")
    assert "CURRENT PAPER BANK (all dates): 102." in card
    assert "✅ WIN" in card
    assert "FINAL 4-1" in card, "a settled leg carries its official score"
    assert "⏳ PENDING" not in card, "no frozen PENDING may outlive a result"
    assert "net +2." in card
    assert receipt_path.read_bytes() == frozen_bytes, "the receipt itself is immutable"


def _stub_leg(*, game_id: str, selection: str = "home", price: float = 1.95, outcome=None) -> dict:
    return {
        "game_id": game_id,
        "home": "Home 1",
        "away": "Away 1",
        "scheduled_utc": "2026-11-28T18:00:00Z",
        "market": "moneyline",
        "selection": selection,
        "line": None,
        "calibrated_probability": 0.62,
        "grade_status": outcome or "pending",
        "outcome": outcome,
        "book": {"book": "DraftKings", "selected_decimal_price": price},
    }


def _stub_position(*, status: str, profit, legs, odds: float = 1.90, stake: float = 1.0) -> dict:
    return {
        "ticket_type": "acca",
        "grade_status": status,
        "profit_units": profit,
        "stake_units": stake,
        "combined_decimal_odds": odds,
        "estimated_joint_win_probability": 0.5,
        "leg_count": len(legs),
        "legs": legs,
        "bookmakers": ["DraftKings"],
    }


def _stub_report() -> dict:
    settled_legs = [_stub_leg(game_id="g1", outcome="win"), _stub_leg(game_id="g2", outcome="win")]
    pending_legs = [_stub_leg(game_id="g3"), _stub_leg(game_id="g4")]
    return {
        "schema_version": "magolide.paper_ticket_performance",
        "league": "NHL",
        "settled_positions": 1,
        "pending_positions": 1,
        "wins": 1,
        "losses": 0,
        "pushes": 0,
        "hit_rate": 1.0,
        "settled_stake_units": 1.0,
        "profit_units": 0.9,
        "roi": 0.9,
        "low_sample_flag": True,
        "low_sample_note": "[LOW SAMPLE: n < 30] no profit, edge, or readiness claim is supported",
        "history": [
            {
                "date": "2026-11-26",
                "decision_id": "dec-nobet",
                "frozen_at": "2026-11-26T14:00:00Z",
                "status": "frozen_no_bet",
                "positions": [],
            },
            {
                "date": "2026-11-27",
                "decision_id": "dec-win",
                "frozen_at": "2026-11-27T14:00:00Z",
                "status": "frozen",
                "positions": [
                    _stub_position(status="win", profit=0.9, legs=settled_legs, odds=1.90)
                ],
            },
            {
                "date": "2026-11-28",
                "decision_id": "dec-held",
                "frozen_at": "2026-11-28T14:00:00Z",
                "status": "frozen",
                "positions": [
                    _stub_position(status="pending", profit=None, legs=pending_legs, odds=3.80)
                ],
            },
        ],
    }


def test_performance_render_tracks_bank_record_bet_days_and_held():
    report = _stub_report()
    text = render_performance(report)

    assert text.startswith("BULLION AUTO-TICKETS (ROLLING) — NHL paper performance")
    # 100-unit base, +0.900 banked, the held ticket's stake reserved at cost.
    assert "BANK     100.900 units   x1.009 of the 100-unit base" in text
    assert "CAPITAL  99.900 free / 1.000 reserved paper units" in text
    assert "RECORD   ✅ WIN 1W (100.0% hit) · ROI +90.0%" in text
    assert "OPEN     1 held paper positions · 2 awaiting legs in open tickets" in text
    assert "LEGS     2 settled legs across 1 settled positions" in text
    assert "[LOW SAMPLE: n < 30]" in text

    # BET-DAYS, oldest first: the no-bet day, the settled day, then the held one.
    assert "BET-DAYS, oldest first (3)" in text
    table = text.split("BET-DAYS, oldest first (3)")[1].split("\n\n")[0]
    rows = [line for line in table.splitlines() if line.strip()]
    assert [row.split()[0] for row in rows] == ["2026-11-26", "2026-11-27", "2026-11-28"]
    assert rows[0].split()[1] == "—", "a no-bet day has no record to show"
    assert "1W-0L" in rows[1] and "staked 1.000" in rows[1]
    assert "returned 1.900" in rows[1] and "net +0.900" in rows[1]
    assert "bank* 100.900" in rows[1]
    assert "pending 1.000" in rows[2] and "bank* 100.900" in rows[2]

    held = text.split("HELD (not yet settled; stake counted as committed)")[1]
    held = held.split("SETTLED (most recent")[0]
    assert "ACCA #1 @3.800 stake 1.000 · receipt dec-held" in held
    assert "TIP-OFF Sat 2026-11-28 20:00 SAST" in held
    assert "@1.950" in held and "DraftKings" in held and "⏳ PENDING" in held

    settled = text.split("SETTLED (most recent")[1]
    assert "✅ WIN · net +0.900u" in settled
    assert settled.count("✅ WIN") == 3  # the position and both legs

    assert render_performance(report) == text, "rendering is deterministic"


def test_performance_render_is_empty_before_any_receipt():
    text = render_performance({"league": "NHL", "history": []})
    assert "BANK     100.000 units" in text
    assert "RECORD   no settled paper positions yet" in text
    assert "BET-DAYS, oldest first (0)" in text
    assert "HELD (not yet settled; stake counted as committed)\n  None." in text
    assert "SETTLED (most recent 0 of 0 bet-days)\n  None." in text


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

    Four eligible legs at 2.20 combine to 4.84 for two legs, outside the frozen
    [3.00, 4.50] acca band — a genuine no-bet at parity. Widening that band in
    the replay harness then has to be able to bet the same frozen pool.

    The price is 2.20 rather than the 2.90 this test used to use: at a 0.62
    calibrated probability that was a +80% claimed edge, i.e. exactly the broken
    input ``MAX_CLAIMED_EDGE`` now rejects, so no leg would have been eligible.
    """
    league, rows = _pool(tmp_path, n=4, price=2.20)
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


# ---------------------------------------------------------------------------
# Option 1 (exact Kelly log-growth ranking) + Option A (two-tier edge tripwire)
# ---------------------------------------------------------------------------


def _leg_row(
    index: int,
    *,
    price: float,
    probability: float,
    market: str = "moneyline",
    selection: str = "home",
    line=None,
    game_id=None,
    period: str = "full_game",
):
    """A minimal row that ``build_accas`` accepts, priced like a real leg."""
    return {
        "forecast_id": f"f{index}-{market}-{selection}",
        "league": "nhl",
        "game_id": game_id if game_id is not None else f"g{index}",
        "market": market,
        "period": period,
        "selection": selection,
        "line": line,
        "scheduled_utc": "2026-11-28T18:00:00Z",
        "raw_probability": probability,
        "calibrated_probability": probability,
        "book": {"book": "TestBook", "selected_decimal_price": price},
    }


def _leg_keys(ticket):
    return [
        (leg["game_id"], leg["market"], leg["selection"], leg.get("line"))
        for leg in ticket["legs"]
    ]


def test_kelly_log_growth_is_exact_full_kelly():
    """``g*`` must be the real log growth of the full-Kelly stake, not a proxy."""
    growth = kelly_log_growth(1.05 / 1.30, 1.30)
    assert growth == pytest.approx(0.004346, abs=1e-6)
    assert kelly_log_growth(1.05 / 3.00, 3.00) == pytest.approx(0.000620, abs=1e-6)
    assert growth / kelly_log_growth(1.05 / 3.00, 3.00) == pytest.approx(7.01, abs=0.01)

    # Rebuild g* from the reported stake fraction: same number to float noise.
    # The win branch grows by f*(O-1), not by f; that is the whole point of
    # scoring the *stake*, not the price.
    for probability, price in ((0.60, 1.80), (0.45, 2.50), (0.30, 4.20)):
        fraction = kelly_stake_fraction(probability, price)
        assert 0.0 < fraction < 1.0
        rebuilt = probability * math.log(1.0 + fraction * (price - 1.0)) + (
            1.0 - probability
        ) * math.log(1.0 - fraction)
        assert kelly_log_growth(probability, price) == pytest.approx(rebuilt, abs=1e-12)

    # No edge, no growth: a bet that cannot win a unit is not a bet.
    assert kelly_log_growth(0.35, 2.00) == float("-inf")
    assert kelly_log_growth(0.50, 2.00) == float("-inf")

    # At a fixed edge the exact objective prefers the shorter price.
    short = kelly_log_growth(1.05 / 1.30, 1.30)
    mid = kelly_log_growth(1.05 / 2.00, 2.00)
    long = kelly_log_growth(1.05 / 3.00, 3.00)
    assert short > mid > long > 0.0


def test_kelly_stake_fraction_matches_closed_form():
    assert kelly_stake_fraction(0.45, 2.50) == pytest.approx(0.125 / 1.5, abs=1e-12)
    assert kelly_stake_fraction(0.60, 2.00) == pytest.approx(0.20, abs=1e-12)
    assert kelly_stake_fraction(0.40, 2.00) == 0.0
    assert kelly_stake_fraction(0.30, 2.00) == 0.0


def _wager_key(row):
    return (
        row.get("league"),
        row.get("game_id"),
        row.get("market"),
        row.get("period"),
        row.get("selection"),
        row.get("line"),
    )


def _best_combo(
    rows,
    *,
    min_legs,
    max_legs,
    min_combined_odds,
    max_combined_odds,
    min_leg_odds=1.20,
    max_leg_odds=3.00,
    max_claimed_edge=None,
):
    """Highest-growth in-band combination: enumerated, no pruning at all."""
    usable = []
    for row in rows:
        price = row_price(row)
        probability = float(row["calibrated_probability"])
        if price is None or not (min_leg_odds <= price <= max_leg_odds):
            continue
        if max_claimed_edge is not None and probability * price - 1.0 > max_claimed_edge:
            continue
        usable.append(row)
    usable.sort(
        key=lambda row: (row_price(row), str(row.get("game_id")), str(row.get("forecast_id")))
    )

    scored = []
    for size in range(min_legs, max_legs + 1):
        for combo in itertools.combinations(usable, size):
            if len({row["game_id"] for row in combo}) != size:
                continue
            price = math.prod(row_price(row) for row in combo)
            if not (min_combined_odds <= price <= max_combined_odds):
                continue
            probability = math.prod(
                float(row["calibrated_probability"]) for row in combo
            )
            growth = kelly_log_growth(probability, price)
            if growth == float("-inf"):
                continue
            scored.append(
                (
                    -growth,
                    -probability,
                    price,
                    tuple(str(row["forecast_id"]) for row in combo),
                    combo,
                )
            )
    if not scored:
        return None
    # ``_kelly`` builds its tie-break tuple in canonical (price, game, id)
    # order, so the reference has to as well or an exact tie reads as a bug.
    scored.sort(key=lambda item: item[:4])
    return scored[0][4]


def _brute_force_kelly(rows, *, min_legs, max_legs, limit, **bands):
    """The residual-pool argmax loop, with every combination enumerated.

    ``build_accas`` re-selects on the pool left over after each ticket, so the
    reference has to as well; this compares the *search*, not a different
    selection rule.
    """
    remaining, used_wagers, used_sides, output = list(rows), set(), set(), []
    while len(output) < limit:
        pool = [
            row
            for row in remaining
            if _wager_key(row) not in used_wagers
            and (row["game_id"], row["market"], row["selection"]) not in used_sides
        ]
        best = _best_combo(pool, min_legs=min_legs, max_legs=max_legs, **bands)
        if best is None:
            break
        output.append(best)
        for row in best:
            used_wagers.add(_wager_key(row))
            used_sides.add((row["game_id"], row["market"], row["selection"]))
    return output


@pytest.mark.parametrize("max_legs", [2, 3])
def test_exact_kelly_search_matches_brute_force(max_legs):
    """The price-sorted window walk must visit exactly the combinations it scores."""
    prices = [1.35, 1.55, 1.72, 1.90, 2.05, 2.30, 2.55, 2.80]
    rows = []
    for index, price in enumerate(prices):
        rows.append(
            _leg_row(index, price=price, probability=round(1.22 / price, 4))
        )
        # A second market on the same fixture: in the pool, never in one acca.
        rows.append(
            _leg_row(
                index,
                price=round(price * 0.92, 2),
                probability=round(1.18 / (price * 0.92), 4),
                market="total",
                selection="over",
                line=5.5,
            )
        )

    reference = _brute_force_kelly(
        rows,
        min_legs=2,
        max_legs=max_legs,
        limit=3,
        min_combined_odds=3.30,
        max_combined_odds=5.00,
    )
    tickets = build_accas(
        rows,
        min_legs=2,
        max_legs=max_legs,
        limit=3,
        ranking_metric=RANKING_METRIC_KELLY,
        min_combined_odds=3.30,
        max_combined_odds=5.00,
        forbid_alt_line_reuse=True,
    )
    assert reference, "the fixture must contain in-band combinations"
    assert [_leg_keys(ticket) for ticket in tickets] == [
        [(row["game_id"], row["market"], row["selection"], row.get("line")) for row in combo]
        for combo in reference
    ]
    growths = [ticket["ranking_score"] for ticket in tickets]
    assert growths == sorted(growths, reverse=True)
    for ticket in tickets:
        assert 3.30 - 1e-9 <= ticket["combined_decimal_odds"] <= 5.00 + 1e-9
        assert ticket["ranking_metric"] == RANKING_METRIC_KELLY
        assert ticket["kelly_fraction"] > 0.0
        assert ticket["leg_count"] == len(ticket["legs"])


def test_exact_kelly_matches_brute_force_on_a_real_board():
    """The last-leg frontier must lose no optimum on a realistic slate."""
    _league, rows = _three_market_board(games=15)
    eligible = eligible_rows(rows, policy=GatePolicy())
    assert len(eligible) == 45
    bands = {"min_combined_odds": 3.30, "max_combined_odds": 5.00}
    reference = _brute_force_kelly(
        eligible, min_legs=2, max_legs=3, limit=3, **bands
    )
    tickets = build_accas(
        eligible,
        min_legs=2,
        max_legs=3,
        limit=3,
        ranking_metric=RANKING_METRIC_KELLY,
        **bands,
    )
    assert reference
    assert [ticket["leg_count"] for ticket in tickets] == [len(combo) for combo in reference]
    for ticket, combo in zip(tickets, reference):
        assert ticket["ranking_score"] == pytest.approx(
            kelly_log_growth(
                math.prod(row["calibrated_probability"] for row in combo),
                math.prod(row_price(row) for row in combo),
            ),
            abs=1e-7,
        )
        assert ticket["combined_decimal_odds"] == pytest.approx(
            math.prod(row_price(row) for row in combo)
        )


def _multi_market_board(seed: int, games: int = 8, correlated: bool = True):
    """Fixtures with three priced markets, the shape that made the bug bite."""
    rng = random.Random(seed)
    rows = []
    for game in range(games):
        strength = rng.uniform(0.45, 0.80)
        for market, selection in (
            ("moneyline", "home"),
            ("spread", "home"),
            ("total", "over"),
        ):
            price = round(rng.uniform(1.30, 2.60), 2)
            if correlated:
                probability = min(
                    0.95, max(0.30, strength + rng.uniform(-0.06, 0.06))
                )
            else:
                probability = rng.uniform(0.45, 0.80)
            # inside the 40% per-leg edge cap, so both sides of the frontier legal
            probability = min(probability, 1.38 / price)
            rows.append(
                _leg_row(
                    0,
                    price=price,
                    probability=round(probability, 4),
                    market=market,
                    selection=selection,
                    game_id=f"G{game}",
                )
            )
    rows.sort(key=lambda row: (row_price(row), str(row["game_id"]), str(row["forecast_id"])))
    return rows


def test_frontier_never_lets_a_fixture_collision_mask_a_legal_leg():
    """The trap: a same-fixture leg must not raise the dominance bar.

    ``f3`` sits on ``G1`` — already in the ticket — at a higher price and a
    higher probability than ``f2``. If it is allowed to prune the scan (it
    cannot be chosen, so it must not dominate), ``(f0, f1, f2)`` disappears and
    the build silently settles for ``(f1, f2, f3)`` at a worse growth rate.
    """
    rows = [
        _leg_row(0, price=1.40, probability=0.78, game_id="G1"),
        _leg_row(1, price=1.50, probability=0.72, game_id="G2"),
        _leg_row(2, price=1.80, probability=0.60, game_id="G3"),
        _leg_row(3, price=1.85, probability=0.61, game_id="G1"),
    ]
    tickets = build_accas(
        rows,
        min_legs=2,
        max_legs=3,
        limit=3,
        ranking_metric=RANKING_METRIC_KELLY,
        min_combined_odds=3.30,
        max_combined_odds=5.00,
        forbid_alt_line_reuse=True,
    )
    assert tickets
    assert [leg["forecast_id"].split("-")[0] for leg in tickets[0]["legs"]] == [
        "f0",
        "f1",
        "f2",
    ]
    assert tickets[0]["combined_decimal_odds"] == pytest.approx(3.78)
    assert tickets[0]["ranking_score"] == pytest.approx(0.012800, abs=1e-6)
    reference = _brute_force_kelly(
        rows,
        min_legs=2,
        max_legs=3,
        limit=3,
        min_combined_odds=3.30,
        max_combined_odds=5.00,
    )
    assert [sorted(leg["forecast_id"] for leg in ticket["legs"]) for ticket in tickets] == [
        sorted(row["forecast_id"] for row in combo) for combo in reference
    ]


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
def test_kelly_matches_brute_force_on_multi_market_seeds(seed):
    """Multi-market boards, correlated and not, against exhaustive enumeration."""
    rows = _multi_market_board(seed, games=8, correlated=seed % 2 == 0)
    eligible = eligible_rows(rows, policy=GatePolicy(max_claimed_edge=None))
    reference = _brute_force_kelly(
        eligible,
        min_legs=2,
        max_legs=3,
        limit=3,
        min_combined_odds=3.30,
        max_combined_odds=5.00,
    )
    tickets = build_accas(
        eligible,
        min_legs=2,
        max_legs=3,
        limit=3,
        ranking_metric=RANKING_METRIC_KELLY,
        min_combined_odds=3.30,
        max_combined_odds=5.00,
        forbid_alt_line_reuse=True,
    )
    assert [sorted(leg["forecast_id"] for leg in ticket["legs"]) for ticket in tickets] == [
        sorted(row["forecast_id"] for row in combo) for combo in reference
    ]


def test_kelly_prefers_shorter_price_at_equal_edge():
    """Equal edge, different price: exact log growth takes the short ticket."""
    rows = [
        _leg_row(0, price=2.00, probability=1.20 / 2.00),
        _leg_row(1, price=2.00, probability=1.20 / 2.00),
        _leg_row(2, price=2.10, probability=1.20 / 2.10),
        _leg_row(3, price=2.10, probability=1.20 / 2.10),
    ]
    tickets = build_accas(
        rows,
        min_legs=2,
        max_legs=2,
        limit=3,
        ranking_metric=RANKING_METRIC_KELLY,
        min_combined_odds=3.30,
        max_combined_odds=5.00,
    )
    assert tickets[0]["combined_decimal_odds"] == pytest.approx(4.00)
    assert [ticket["combined_decimal_odds"] for ticket in tickets] == sorted(
        ticket["combined_decimal_odds"] for ticket in tickets
    )


def test_two_and_three_leg_tickets_compete_in_one_pool():
    """Short-priced legs make 3-leg tickets; mid-priced legs make 2-leg ones."""
    rows = [
        _leg_row(index, price=1.50, probability=1.30 / 1.50) for index in range(6)
    ] + [_leg_row(10 + index, price=1.90, probability=1.06 / 1.90) for index in range(4)]

    mixed = build_accas(
        rows,
        min_legs=2,
        max_legs=3,
        limit=3,
        ranking_metric=RANKING_METRIC_KELLY,
        min_combined_odds=3.30,
        max_combined_odds=5.00,
    )
    pairs_only = build_accas(
        rows,
        min_legs=2,
        max_legs=2,
        limit=3,
        ranking_metric=RANKING_METRIC_KELLY,
        min_combined_odds=3.30,
        max_combined_odds=5.00,
    )
    assert [ticket["leg_count"] for ticket in mixed] == [3, 3, 2]
    assert {ticket["leg_count"] for ticket in pairs_only} == {2}
    # A superset of candidates can never rank a worse number one.
    assert mixed[0]["ranking_score"] > pairs_only[0]["ranking_score"]

    pinned = build_accas(
        rows,
        legs=3,
        limit=2,
        ranking_metric=RANKING_METRIC_KELLY,
        min_combined_odds=3.30,
        max_combined_odds=5.00,
    )
    assert [ticket["leg_count"] for ticket in pinned] == [3, 3]
    assert [ticket["acca_id"] for ticket in pinned] == [
        ticket["acca_id"] for ticket in mixed[:2]
    ]


def test_pairs_are_not_pruned_by_a_three_leg_ceiling():
    """Regression: a 2-leg pool under ``max_legs=3`` must still find its pairs.

    The window walk used to bound the cheapest completion with *all* remaining
    legs, so four 1.90 legs (pair 3.61, triple 6.86) were pruned to nothing.
    """
    rows = [_leg_row(index, price=1.90, probability=1.06 / 1.90) for index in range(4)]
    tickets = build_accas(
        rows,
        min_legs=2,
        max_legs=3,
        limit=3,
        ranking_metric=RANKING_METRIC_KELLY,
        min_combined_odds=3.30,
        max_combined_odds=5.00,
    )
    assert [ticket["leg_count"] for ticket in tickets] == [2, 2]
    assert all(ticket["combined_decimal_odds"] == pytest.approx(3.61) for ticket in tickets)


def test_leg_range_validation_is_explicit():
    rows = [_leg_row(0, price=1.60, probability=0.70)]
    with pytest.raises(ValueError):
        build_accas(rows, min_legs=1, max_legs=3)
    with pytest.raises(ValueError):
        build_accas(rows, min_legs=3, max_legs=2)
    with pytest.raises(ValueError):
        build_accas(rows, min_legs=2, max_legs=7)
    with pytest.raises(ValueError):
        build_accas(rows, ranking_metric="not_a_metric")


def test_default_build_accas_contract_is_still_two_leg_blend():
    """``build_accas(rows)`` is the pre-Option-1 contract: two legs, blend."""
    rows = [
        _leg_row(0, price=1.95, probability=0.60),
        _leg_row(1, price=1.90, probability=0.61),
    ]
    tickets = build_accas(rows)
    assert tickets
    assert all(ticket["leg_count"] == 2 for ticket in tickets)
    assert all(ticket["ranking_metric"] == RANKING_METRIC_BLEND for ticket in tickets)
    assert all(ticket["ranking_price_weight"] == PRICE_WEIGHT for ticket in tickets)
    assert all(ticket["kelly_fraction"] > 0.0 for ticket in tickets)


def test_option_a_sanity_cap_blocks_broken_edges_only(tmp_path):
    """+12%/+24% trade (24% tagged); +55% is refused as a broken input.

    Calibration moves a raw 0.50 to 0.562 and a raw 0.61 to 0.618, so the
    claimed edges here are the calibrated ones — the tripwire reads the number
    the ticket is actually built on, not the model's raw output.
    """
    honest = _pool(tmp_path, n=4, price=2.00, probability=0.50)
    rows = honest[1]
    eligible = eligible_rows(rows, policy=GatePolicy())
    assert len(eligible) == 4
    assert all(
        row["eligibility"]["claimed_edge"] == pytest.approx(0.1247, abs=1e-4)
        for row in eligible
    )
    assert not any(row["eligibility"]["edge_implausible"] for row in eligible)

    tagged = _pool(tmp_path, n=4, price=2.00, probability=0.61)
    rows = tagged[1]
    eligible = eligible_rows(rows, policy=GatePolicy())
    assert len(eligible) == 4
    assert all(
        row["eligibility"]["claimed_edge"] == pytest.approx(0.2360, abs=1e-4)
        for row in eligible
    )
    assert all(row["eligibility"]["edge_implausible"] for row in eligible)

    broken = _pool(tmp_path, n=4, price=2.00, probability=0.775)
    rows = broken[1]
    assert eligible_rows(rows, policy=GatePolicy()) == []
    assert all(
        "edge_above_sanity_cap" in row["eligibility"]["reasons"] for row in rows
    )
    # The cap is a rejection, not a housekeeping filter: the row still records
    # what it claimed, so an audit can see why it was refused.
    assert all(
        row["eligibility"]["claimed_edge"] == pytest.approx(0.55, abs=1e-6)
        for row in rows
    )


def test_edge_cap_is_off_for_legacy_policy_and_on_by_default():
    assert GatePolicy().max_claimed_edge == MAX_CLAIMED_EDGE
    assert GatePolicy(max_claimed_edge=None).max_claimed_edge is None
    assert GatePolicy().to_policy_dict()["max_claimed_edge"] == MAX_CLAIMED_EDGE
    assert GatePolicy().to_policy_dict()["edge_flag_threshold"] == EDGE_FLAG_THRESHOLD

    rows = [
        _leg_row(0, price=2.00, probability=0.775),  # +55%
        _leg_row(1, price=2.00, probability=0.775),
    ]
    assert build_accas(rows, min_combined_odds=None, max_combined_odds=None) != []
    capped = build_accas(
        rows,
        min_combined_odds=None,
        max_combined_odds=None,
        max_claimed_edge=MAX_CLAIMED_EDGE,
    )
    assert capped == []


def test_ticket_and_card_carry_the_edge_audit_tag(tmp_path):
    league, rows = _pool(tmp_path, n=4, price=2.00, probability=0.61)
    eligible = eligible_rows(rows, policy=GatePolicy())
    tickets = build_accas(
        eligible,
        min_legs=2,
        max_legs=2,
        limit=1,
        ranking_metric=RANKING_METRIC_KELLY,
        min_combined_odds=3.30,
        max_combined_odds=5.00,
        edge_flag_threshold=EDGE_FLAG_THRESHOLD,
    )
    assert tickets
    ticket = tickets[0]
    assert ticket["edge_implausible"] is True
    assert ticket["edge_implausible_leg_count"] == 2
    assert all(leg["edge_implausible"] for leg in ticket["legs"])

    receipt = freeze_receipt(
        tmp_path,
        league=league,
        day=DAY,
        captured_utc="2026-11-28T15:00:00Z",
        candidate_pool=rows,
        accas=tickets,
        selection_policy={
            "requested_legs": 2,
            "min_legs": 2,
            "max_legs": 2,
            "limit": 1,
            "ranking_metric": RANKING_METRIC_KELLY,
            "edge_flag_threshold": EDGE_FLAG_THRESHOLD,
            "max_claimed_edge": MAX_CLAIMED_EDGE,
        },
        decision_state={"status": "frozen", "planned_cutoff_utc": "2026-11-28T14:00:00Z"},
    )
    card = render_card(receipt)
    assert card.count("[EDGE>15% AUDIT]") == 2
    assert "kelly_log_growth" in card


def test_replay_legacy_receipt_defaults_to_the_old_policy(tmp_path):
    """A pre-Option-1 receipt replays as blend / fixed legs / no edge cap."""
    league, rows = _pool(tmp_path, n=4, price=2.20)
    _freeze(tmp_path, rows, league=league, legs=2, limit=1)
    settle_lane(tmp_path, league, _final_games([row["game_id"] for row in rows]))
    day = load_evidence(tmp_path)[0]
    limits = replay_policy_limits(day, ReplaySpec())
    assert limits["ranking_metric"] == RANKING_METRIC_BLEND
    assert limits["min_legs"] == 2 and limits["max_legs"] == 2
    assert limits["max_claimed_edge"] is None
    assert limits["edge_flag_threshold"] is None
    assert limits["forbid_alt_line_reuse"] is False
    assert check_parity([day])["parity"] is True
    replayed = select_tickets(day, ReplaySpec())
    assert [ticket["leg_count"] for ticket in replayed] == [2]
    assert replayed[0]["ranking_metric"] == RANKING_METRIC_BLEND


def test_replay_spec_reads_option_1_keys_from_a_new_receipt(tmp_path):
    league, rows = _pool(tmp_path, n=6, price=2.20)
    policy = GatePolicy()
    eligible = eligible_rows(rows, policy=policy)
    tickets = build_accas(
        eligible,
        min_legs=2,
        max_legs=3,
        limit=2,
        ranking_metric=RANKING_METRIC_KELLY,
        min_combined_odds=3.30,
        max_combined_odds=5.00,
        max_claimed_edge=policy.max_claimed_edge,
        edge_flag_threshold=policy.edge_flag_threshold,
    )
    if not tickets:
        pytest.skip("pool has no in-band combination")
    freeze_receipt(
        tmp_path,
        league=league,
        day=DAY,
        captured_utc="2026-11-28T15:00:00Z",
        candidate_pool=rows,
        accas=tickets,
        selection_policy={
            "requested_legs": 3,
            "min_legs": 2,
            "max_legs": 3,
            "limit": 2,
            "min_leg_odds": 1.20,
            "max_leg_odds": 3.00,
            "price_weight": 0.15,
            "min_combined_odds": 3.30,
            "max_combined_odds": 5.00,
            "ranking_metric": RANKING_METRIC_KELLY,
            "forbid_alt_line_reuse": True,
            "max_claimed_edge": policy.max_claimed_edge,
            "edge_flag_threshold": policy.edge_flag_threshold,
        },
        decision_state={"status": "frozen", "planned_cutoff_utc": "2026-11-28T14:00:00Z"},
    )
    settle_lane(tmp_path, league, _final_games([row["game_id"] for row in rows]))
    day = load_evidence(tmp_path)[0]
    limits = replay_policy_limits(day, ReplaySpec())
    assert limits["ranking_metric"] == RANKING_METRIC_KELLY
    assert limits["min_legs"] == 2 and limits["max_legs"] == 3
    assert limits["max_claimed_edge"] == MAX_CLAIMED_EDGE
    assert limits["edge_flag_threshold"] == EDGE_FLAG_THRESHOLD
    assert limits["forbid_alt_line_reuse"] is True
    assert check_parity([day])["parity"] is True
    replayed = select_tickets(day, ReplaySpec())
    assert [ticket["acca_id"] for ticket in replayed] == [t["acca_id"] for t in tickets]


def test_harness_variant_parsing_accepts_option_1_keys():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "bullion_replay_harness_variants",
        Path(__file__).resolve().parents[1] / "scripts" / "replay_harness.py",
    )
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    parsed = harness.parse_variant(
        "min_odds=2.6,max_odds=9,min_legs=2,max_legs=3,"
        "ranking_metric=kelly_log_growth,max_claimed_edge=0.4,"
        "edge_flag_threshold=0.15,forbid_alt_line_reuse=false,legs=3",
        ReplaySpec(),
    )
    assert parsed.min_combined_odds == pytest.approx(2.6)
    assert parsed.max_combined_odds == pytest.approx(9.0)
    assert parsed.min_legs == 2 and parsed.max_legs == 3 and parsed.legs == 3
    assert parsed.ranking_metric == RANKING_METRIC_KELLY
    assert parsed.max_claimed_edge == pytest.approx(0.4)
    assert parsed.edge_flag_threshold == pytest.approx(0.15)
    assert parsed.forbid_alt_line_reuse is False

    with pytest.raises(SystemExit):
        harness.parse_variant("not_a_key=1", ReplaySpec())
    with pytest.raises(SystemExit):
        harness.parse_variant("ranking_metric=made_up", ReplaySpec())


# ---------------------------------------------------------------------------
# Multi-tick quote hygiene: the warehouse keeps every snapshot, pricing may not
# ---------------------------------------------------------------------------

PRE_TIP = "2026-11-28T23:00:00Z"


def _quote(
    *,
    book: str = "DraftKings",
    market: str = "moneyline",
    line=None,
    observed: str = "2026-11-28T08:15:00Z",
    prices=None,
    game_id: str = "q1",
) -> BookQuote:
    return BookQuote(
        game_id=game_id,
        book=book,
        market=market,
        observed_utc=observed,
        side_prices=dict(prices or {"home": 1.75, "away": 2.10}),
        line=line,
        pre_match=True,
    )


def _quote_forecast(
    *, market: str = "moneyline", line=None, selection: str = "home", game_id: str = "q1"
) -> Forecast:
    return Forecast(
        game_id=game_id,
        league="nhl",
        scheduled_utc=PRE_TIP,
        home_name="Home",
        away_name="Away",
        market=market,
        line=line,
        selection=selection,
        raw_probability=0.55,
        source=SOURCE,
    )


def test_latest_quotes_keeps_the_newest_price_not_the_best_historical_one():
    """A better 08:15Z price is dead once the book repriced at 16:15Z."""
    stale = _quote(observed="2026-11-28T08:15:00Z", prices={"home": 2.20, "away": 1.70})
    fresh = _quote(observed="2026-11-28T16:15:00Z", prices={"home": 1.75, "away": 2.10})
    assert latest_quotes([stale, fresh]) == [fresh]

    forecast = _quote_forecast()
    priced = attach_quotes([forecast], [stale, fresh])
    assert len(priced) == 1
    assert priced[0].book is fresh
    assert priced[0].book.price_for("home") == pytest.approx(1.75)


def test_latest_quotes_drops_the_dead_line_when_a_book_moves_its_spread():
    """``line`` is not part of the grouping key: -3.5 is superseded by -6.5."""
    dead = _quote(
        market="spread",
        line=-3.5,
        observed="2026-11-28T08:15:00Z",
        prices={"home": 2.05, "away": 1.80},
    )
    live = _quote(
        market="spread",
        line=-6.5,
        observed="2026-11-28T16:15:00Z",
        prices={"home": 1.91, "away": 1.91},
    )
    kept = latest_quotes([dead, live])
    assert kept == [live]
    assert {quote.line for quote in kept} == {-6.5}

    # The engine inputs are what actually matters: unfiltered history would
    # average the dead -3.5 into a -5.0 anchor nobody quoted.
    from bullion.engines import quoted_lines
    from bullion.engines.basketball import _anchor_line

    assert quoted_lines([dead, live], "spread") == [-6.5, -3.5]
    assert _anchor_line([dead, live], "spread") == pytest.approx(-5.0)
    assert quoted_lines(kept, "spread") == [-6.5]
    assert _anchor_line(kept, "spread") == pytest.approx(-6.5)

    # The dead -3.5 line cannot be priced any more, and it must not price a
    # forecast that was built on it.
    assert attach_quotes([_quote_forecast(market="spread", line=-3.5)], [dead, live]) == []
    priced = attach_quotes([_quote_forecast(market="spread", line=-6.5)], [dead, live])
    assert len(priced) == 1
    assert priced[0].book.line == -6.5


def test_latest_quotes_still_compares_distinct_books_side_by_side():
    """Two books are a live choice; one book's two ticks are not."""
    draftkings_stale = _quote(
        book="DraftKings", observed="2026-11-28T08:15:00Z", prices={"home": 2.40, "away": 1.55}
    )
    draftkings_live = _quote(
        book="DraftKings", observed="2026-11-28T16:15:00Z", prices={"home": 1.75, "away": 2.10}
    )
    bet365_live = _quote(
        book="Bet365", observed="2026-11-28T16:10:00Z", prices={"home": 1.95, "away": 1.95}
    )
    kept = latest_quotes([draftkings_stale, draftkings_live, bet365_live])
    assert kept == [draftkings_live, bet365_live]

    priced = attach_quotes([_quote_forecast()], kept)
    assert len(priced) == 1
    # Best *live* book wins: 1.95 at Bet365, never the dead 2.40 at DraftKings.
    assert priced[0].book.book == "Bet365"
    assert priced[0].book.price_for("home") == pytest.approx(1.95)


def test_latest_quotes_keeps_every_alt_line_from_one_live_snapshot():
    """One snapshot legitimately quotes several lines; they all survive."""
    main = _quote(market="total", line=6.5, observed="2026-11-28T16:15:00Z")
    alt = _quote(market="total", line=7.5, observed="2026-11-28T16:15:00Z")
    stale_alt = _quote(market="total", line=5.5, observed="2026-11-28T08:15:00Z")
    kept = latest_quotes([stale_alt, main, alt])
    assert kept == [main, alt]
    assert latest_quotes([]) == []


def test_attach_quotes_still_rejects_a_post_tip_snapshot_a_newer_one_covered():
    """Fail-closed survives the filter: lateness is never forgiven by freshness."""
    late = _quote(observed="2026-11-28T23:30:00Z", prices={"home": 1.90, "away": 1.90})
    late.pre_match = False
    fresh = _quote(observed="2026-11-28T23:40:00Z", prices={"home": 1.80, "away": 2.00})
    fresh.pre_match = False
    with pytest.raises(ValueError, match="post-tip"):
        attach_quotes([_quote_forecast()], [late, fresh])


# ---------------------------------------------------------------------------
# Off fixtures: a canceled leg voids, a postponed leg stays pending
# ---------------------------------------------------------------------------


def test_grade_row_voids_a_canceled_fixture_and_pends_a_postponed_one():
    row = {"market": "moneyline", "selection": "home", "line": None}
    canceled = Game(
        game_id="c1",
        league="nhl",
        season="2026",
        home_team_id="h",
        away_team_id="a",
        home_name="Home",
        away_name="Away",
        scheduled_utc="2026-11-28T23:00:00Z",
        status=Status.CANCELLED,
    )
    assert canceled.actual is None
    assert grade_row(row, canceled) == "push", "a fixture that is off cannot lose a stake"

    postponed = Game(
        game_id="p1",
        league="nhl",
        season="2026",
        home_team_id="h",
        away_team_id="a",
        home_name="Home",
        away_name="Away",
        scheduled_utc="2026-11-28T23:00:00Z",
        status=Status.POSTPONED,
    )
    assert grade_row(row, postponed) is None, "a postponed leg is pending, not void"


def test_settle_lane_voids_a_canceled_leg_inside_a_multileg_ticket(tmp_path):
    league, rows = _pool(tmp_path, n=4)
    receipt, _ = _freeze(tmp_path, rows, league=league)
    assert receipt is not None
    canceled_leg = receipt["accas"][0]["legs"][0]["game_id"]
    games = {}
    for index in range(4):
        game = _game(index, final="win")
        games[game.game_id] = game
    games[canceled_leg].status = Status.CANCELLED
    games[canceled_leg].actual = None
    report = settle_lane(tmp_path, league, games)
    assert report is not None
    position = report["history"][0]["positions"][0]
    legs = receipt["accas"][0]["leg_count"]
    assert position["leg_outcomes"].count("push") == 1
    assert position["grade_status"] == "win"
    # The voided leg contributes 1.0 to the product; the others still pay 1.95.
    assert position["effective_decimal_odds"] == pytest.approx(1.95 ** (legs - 1))
    assert position["profit_units"] == pytest.approx(1.95 ** (legs - 1) - 1)
