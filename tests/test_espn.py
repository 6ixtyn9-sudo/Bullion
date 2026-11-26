"""Parser safety tests on SYNTHETIC_FIXTURE payloads (no live ESPN data)."""

from __future__ import annotations

import pytest

from bullion.espn import (
    EspnIdentityError,
    EspnParseError,
    et_days_for_sast_day,
    event_id_from_ref,
    lane_permission,
    parse_pickcenter,
    parse_predictor,
    parse_scoreboard,
    parse_summary,
)
from bullion.models import LEAGUES, BookQuote, Forecast, Status
from bullion.tickets import attach_quotes

from .factories import SYNTHETIC, close, event, offer, predictor, summary

OBSERVED = "2026-11-28T09:00:00Z"
NHL = LEAGUES["nhl"]


def _scoreboard_payload():
    offers = [
        offer(
            moneyline={"home": close(150), "away": close(-170)},
            spread={
                "home": close(-155, "-1.5"),
                "away": close(135, "+1.5"),
            },
            total={
                "over": close(-110, "o6.5"),
                "under": close(-110, "u6.5"),
            },
        )
    ]
    return {"events": [event(offers=offers)]}


def test_scoreboard_parses_sides_from_markers_not_array_order():
    parsed = parse_scoreboard(_scoreboard_payload(), NHL, OBSERVED)
    assert SYNTHETIC  # payload is synthetic, never claimed as live
    assert len(parsed.games) == 1
    game = parsed.games[0]
    assert (game.home_team_id, game.away_team_id) == ("10", "20")
    assert game.scheduled_utc == "2026-11-28T23:00:00Z"
    assert game.status is Status.SCHEDULED
    markets = {quote.market for quote in parsed.quotes}
    assert markets == {"moneyline", "spread", "total"}
    prices = {quote.market: quote.side_prices for quote in parsed.quotes}
    assert prices["moneyline"] == {"home": 2.5, "away": 1.5882352941176472}
    assert all(quote.pre_match for quote in parsed.quotes)
    assert all(quote.provenance == "named_book" for quote in parsed.quotes)
    total = next(quote for quote in parsed.quotes if quote.market == "total")
    assert total.line == 6.5
    spread = next(quote for quote in parsed.quotes if quote.market == "spread")
    assert spread.line == -1.5


def test_missing_odds_records_gap_and_never_invents_prices():
    parsed = parse_scoreboard({"events": [event(offers=[])]}, NHL, OBSERVED)
    assert parsed.quotes == []
    assert any(gap.field == "pickcenter" for gap in parsed.gaps)


def test_placeholder_competitors_are_skipped_not_guessed():
    payload = {"events": [event()]}
    payload["events"][0]["competitions"][0]["competitors"][0]["team"]["displayName"] = "TBD"
    parsed = parse_scoreboard(payload, NHL, OBSERVED)
    assert parsed.games == []


def test_missing_home_away_markers_quarantines_event():
    payload = _scoreboard_payload()
    for competitor in payload["events"][0]["competitions"][0]["competitors"]:
        competitor.pop("homeAway")
    parsed = parse_scoreboard(payload, NHL, OBSERVED)
    assert parsed.games == []
    assert parsed.gaps


def test_soccer_draw_price_path_and_gap_when_absent():
    soccer = LEAGUES["eng.1"]
    with_draw = offer(
        moneyline={"home": close(120), "away": close(240), "draw": close(230)},
    )
    parsed = parse_scoreboard({"events": [event(offers=[with_draw])]}, soccer, OBSERVED)
    ml = next(quote for quote in parsed.quotes if quote.market == "moneyline")
    assert "draw" in ml.side_prices

    without_draw = offer(moneyline={"home": close(120), "away": close(240)})
    parsed = parse_scoreboard({"events": [event(offers=[without_draw])]}, soccer, OBSERVED)
    assert any(
        gap.field == "pickcenter.moneyline.draw" for gap in parsed.gaps
    ), "a three-way lane must disclose a missing draw price"


def test_post_tip_quote_raises_at_the_ticket_boundary():
    forecast = Forecast(
        game_id="401",
        league="nhl",
        scheduled_utc="2026-11-28T23:00:00Z",
        home_name="Home",
        away_name="Away",
        market="moneyline",
        selection="home",
        raw_probability=0.55,
        source="nhl_poisson_v1",
    )
    quote = BookQuote(
        game_id="401",
        book="DraftKings",
        market="moneyline",
        observed_utc="2026-11-28T23:30:00Z",
        side_prices={"home": 1.9, "away": 1.9},
        pre_match=True,
    )
    with pytest.raises(ValueError, match="post-tip"):
        attach_quotes([forecast], [quote])

    quote.pre_match = False
    with pytest.raises(ValueError, match="post-tip"):
        attach_quotes([forecast], [quote])


def test_predictor_orientation_follows_team_ref_not_array_position():
    game = parse_scoreboard(_scoreboard_payload(), NHL, OBSERVED).games[0]
    # Statistics attached to the away side, exactly the proven BPI variant.
    evidence = parse_predictor(predictor(stats_on="away", projection=40.0), game)
    assert evidence.stats_side == "away"
    assert evidence.prob_home == pytest.approx(0.60)
    assert evidence.expected_home == pytest.approx(2.6)
    assert evidence.expected_away == pytest.approx(3.2)

    home_side = parse_predictor(predictor(stats_on="home", projection=60.0), game)
    assert home_side.prob_home == pytest.approx(0.60)


def test_predictor_contradiction_raises_instead_of_swapping():
    game = parse_scoreboard(_scoreboard_payload(), NHL, OBSERVED).games[0]
    with pytest.raises(EspnIdentityError):
        parse_predictor(predictor(home_ref="99", away_ref="98"), game)
    with pytest.raises(EspnIdentityError):
        parse_predictor(predictor(home_ref="10", away_ref="10"), game)
    incomplete = predictor()
    incomplete["awayTeam"]["statistics"] = [{"name": "teamChanceLoss", "value": 45.0}]
    with pytest.raises(EspnParseError):
        parse_predictor(incomplete, game)


def test_summary_requires_post_state_and_matching_teams():
    game = parse_scoreboard(_scoreboard_payload(), NHL, OBSERVED).games[0]
    final_game, gaps = parse_summary(summary(), game, NHL)
    assert final_game.status is Status.FINAL
    assert final_game.actual is not None
    assert final_game.actual.home_score == 3.0
    assert final_game.actual.periods == ((1.0, 1.0), (1.0, 1.0), (1.0, 0.0))
    assert final_game.actual.overtime_periods == 0
    assert final_game.actual.regulation_scores() == (3.0, 2.0)
    assert gaps == []

    with pytest.raises(EspnIdentityError):
        parse_summary(summary(state="in"), game, NHL)
    with pytest.raises(EspnIdentityError):
        parse_summary(summary(home="77", away="78"), game, NHL)


def test_summary_without_linescores_keeps_final_but_discloses_gap():
    game = parse_scoreboard(_scoreboard_payload(), NHL, OBSERVED).games[0]
    payload = summary()
    for competitor in payload["header"]["competitions"][0]["competitors"]:
        competitor.pop("linescores")
    final_game, gaps = parse_summary(payload, game, NHL)
    assert final_game.actual.home_score == 3.0
    assert final_game.actual.periods == ()
    assert any(gap.field == "boxscore.linescores" for gap in gaps)


def test_overtime_periods_become_overtime_evidence():
    game = parse_scoreboard(_scoreboard_payload(), NHL, OBSERVED).games[0]
    payload = summary(
        home_score="4",
        away_score="3",
        home_lines=[{"value": 1}, {"value": 1}, {"value": 1}, {"value": 1}],
        away_lines=[{"value": 1}, {"value": 1}, {"value": 1}, {"value": 0}],
    )
    final_game, _ = parse_summary(payload, game, NHL)
    assert final_game.actual.overtime_periods == 1
    assert final_game.actual.regulation_scores() == (3.0, 3.0)


def _live_report(moneyline=0.8, total=0.8, sample=8, status="ok", draw=None):
    markets = {
        "moneyline": {"event_coverage": moneyline, "sample_size": sample, "status": status},
        "total": {"event_coverage": total, "sample_size": sample, "status": status},
    }
    if draw is not None:
        markets["moneyline_draw"] = {
            "event_coverage": draw,
            "sample_size": sample,
            "status": status,
        }
    return {
        "schema_version": "bullion.espn_probe",
        "mode": "live",
        "captured_utc": "2026-11-27T00:00:00Z",
        "leagues": {"nhl": {"coverage": {"named_book_pre_tip": markets}}},
    }


def test_lane_permission_is_fail_closed_without_live_probe():
    assert lane_permission(None, NHL).research_only
    fake = {"schema_version": "bullion.espn_probe", "mode": "synthetic", "leagues": {}}
    assert lane_permission(fake, NHL).research_only
    missing_lane = {"schema_version": "bullion.espn_probe", "mode": "live", "leagues": {}}
    assert lane_permission(missing_lane, NHL).reason == "espn_probe_league_missing"


def test_lane_permission_needs_a_sample_behind_every_coverage_number():
    thin = _live_report(moneyline=1.0, total=1.0, sample=2)
    permission = lane_permission(thin, NHL)
    assert permission.research_only
    assert permission.reason == "espn_probe_coverage_insufficient_sample"

    unmeasured = _live_report(moneyline=0.0, total=0.0, sample=0, status="no_pre_match_events")
    permission = lane_permission(unmeasured, NHL)
    assert permission.research_only
    assert permission.reason == "espn_probe_no_pre_match_events"

    measured_zero = _live_report(moneyline=0.0, total=0.0, sample=8)
    permission = lane_permission(measured_zero, NHL)
    assert permission.research_only
    assert permission.reason == "named_book_coverage_below_50pct"

    covered = _live_report(moneyline=0.8, total=0.2, sample=8)
    permission = lane_permission(covered, NHL)
    assert permission.allows("moneyline")
    assert not permission.allows("total"), "under 50% coverage stays research-only"
    assert permission.captured_utc == "2026-11-27T00:00:00Z"


def _soccer_report(draw_coverage):
    report = _live_report(moneyline=0.9, total=0.9, draw=draw_coverage)
    report["leagues"]["eng.1"] = report["leagues"].pop("nhl")
    return report


def test_soccer_lane_requires_a_draw_price_before_1x2_tickets():
    soccer = LEAGUES["eng.1"]
    assert not lane_permission(_soccer_report(0.1), soccer).allows("moneyline")
    assert lane_permission(_soccer_report(0.9), soccer).allows("moneyline")
    assert lane_permission(_soccer_report(0.9), soccer).allows("total")


def test_sast_day_maps_to_the_two_eastern_days_it_can_occupy():
    assert et_days_for_sast_day("2026-11-26") == ["20261125", "20261126"]
    assert et_days_for_sast_day("2026-07-04") == ["20260703", "20260704"]
    # A 22:30 SAST game is 12:30 ET the same SAST day; a 02:00 SAST game is the
    # previous ET evening. Both live inside the two requested ET days.
    assert et_days_for_sast_day("2026-12-31") == ["20261230", "20261231"]


def test_core_event_refs_strip_their_query_suffix():
    ref = (
        "http://sports.core.api.espn.com/v2/sports/football/leagues/"
        "college-football/events/401856879?lang=en&region=us"
    )
    assert event_id_from_ref(ref) == "401856879"
    assert event_id_from_ref("http://x/events/740780?lang=en") == "740780"
    assert event_id_from_ref("http://x/teams/10") is None
    assert event_id_from_ref(None) is None


def _pickcenter_payload(entries, state="pre"):
    return {
        "header": {"competitions": [{"status": {"type": {"state": state}}}]},
        "pickcenter": entries,
        "odds": [],
    }


def test_pickcenter_nested_shape_is_parsed_as_named_book_quotes():
    """SYNTHETIC_FIXTURE — nesting copied from the proven scoreboard offer paths."""
    game = parse_scoreboard(_scoreboard_payload(), NHL, OBSERVED).games[0]
    payload = _pickcenter_payload(
        [
            {
                "provider": {"displayName": "DraftKings"},
                "moneyline": {"home": close(-120), "away": close(110)},
                "pointSpread": {"home": close(-110, "-1.5"), "away": close(-110, "+1.5")},
                "total": {"over": close(-110, "o6.5"), "under": close(-110, "u6.5")},
            }
        ]
    )
    parsed = parse_pickcenter(payload, game, OBSERVED)
    assert {quote.market for quote in parsed.quotes} == {"moneyline", "spread", "total"}
    assert all(quote.book == "DraftKings" for quote in parsed.quotes)
    assert all(quote.pre_match for quote in parsed.quotes)
    assert all(quote.provenance == "named_book" for quote in parsed.quotes)
    assert parsed.gaps == []


def test_pickcenter_flat_variant_and_unknown_shape_fail_closed():
    game = parse_scoreboard(_scoreboard_payload(), NHL, OBSERVED).games[0]
    flat = _pickcenter_payload(
        [
            {
                "provider": {"name": "ESPN BET"},
                "homeTeamOdds": {"moneyLine": -135},
                "awayTeamOdds": {"moneyLine": 115},
                "spread": -1.5,
                "overUnder": 6.5,
            }
        ]
    )
    parsed = parse_pickcenter(flat, game, OBSERVED)
    assert [quote.market for quote in parsed.quotes] == ["moneyline"]
    assert parsed.quotes[0].side_prices["home"] == pytest.approx(1.7407407407407407)

    unknown = _pickcenter_payload([{"provider": {"displayName": "DraftKings"}, "weirdKey": 1}])
    parsed = parse_pickcenter(unknown, game, OBSERVED)
    assert parsed.quotes == []
    assert any(gap.field == "pickcenter.shape_unrecognised" for gap in parsed.gaps)
    assert "weirdKey" in parsed.gaps[0].reason

    absent = parse_pickcenter({"header": {"competitions": []}}, game, OBSERVED)
    assert any(gap.field == "pickcenter.absent" for gap in absent.gaps)


def test_pickcenter_quotes_flip_to_post_tip_when_the_game_has_started():
    game = parse_scoreboard(_scoreboard_payload(), NHL, OBSERVED).games[0]
    payload = _pickcenter_payload(
        [{"provider": {"displayName": "DraftKings"}, "moneyline": {"home": close(-120), "away": close(110)}}],
        state="in",
    )
    parsed = parse_pickcenter(payload, game, OBSERVED)
    assert parsed.quotes and not parsed.quotes[0].pre_match
