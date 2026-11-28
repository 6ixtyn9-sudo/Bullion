"""F13 — a surface that wins on Brier but cannot price the board is demoted.

``ncaa_w`` shipped zero tickets for eight days while the empirical surface had
no usable spread/total evidence: the surface choice was measured on the
moneyline Brier alone, so a surface that structurally lacks a permitted market
silently deleted it. The guard picks a surface by *coverage* when coverage is
what is missing, still exactly one per lane.
"""

from __future__ import annotations

import pytest

from bullion.engines import basketball as bb
from bullion.engines.basketball import BPI_SOURCE, EMPIRICAL_SOURCE, resolve_surface
from bullion.espn import EspnClient, PROBE_SCHEMA
from bullion.models import LEAGUES
from bullion.runner import run_lane
from bullion.store import Store

from .factories import close, event, offer, predictor
from .test_basketball_lanes import _basketball_offers, _seed_nba_calibration

DAY = "2026-11-28"
# Numeric team refs: the predictor payload carries ``$ref`` URLs and the parser
# resolves the trailing segment, so a lexical id would be unresolvable.
HOME = "101"
AWAY = "202"


def _board(*, with_total: bool = True) -> dict:
    return {
        "events": [
            event(
                event_id="e0",
                home=HOME,
                away=AWAY,
                date="2026-11-28T19:30:00Z",
                offers=_basketball_offers() if with_total else [_moneyline_only()],
            )
        ]
    }


def _moneyline_only():
    return offer(moneyline={"home": close(-110), "away": close(-110)})


def _client(*, with_predictor: bool = True) -> EspnClient:
    board = _board()

    def fetch(url: str):
        if "/predictor" in url:
            # No ``teamExpectedPts``: prob-only BPI, exactly the live ncaa
            # payload that has a moneyline and no spread/total at all.
            return predictor(home_ref=HOME, away_ref=AWAY, expected=False)
        if "/summary?event=" in url:
            from .factories import summary

            event_id = url.split("event=")[1].split("&")[0]
            return summary(
                event_id=event_id,
                home=HOME,
                away=AWAY,
                home_score="112",
                away_score="104",
                home_lines=[{"value": 56}, {"value": 56}],
                away_lines=[{"value": 52}, {"value": 52}],
            )
        return board

    return EspnClient(fetch=fetch)


def _probe(*markets: str, league: str = "nba") -> dict:
    market = {"event_coverage": 0.9, "sample_size": 24, "status": "ok"}
    return {
        "schema_version": PROBE_SCHEMA,
        "mode": "live",
        "captured_utc": "2026-11-28T06:00:00Z",
        "leagues": {
            league: {
                "captured_utc": "2026-11-28T06:00:00Z",
                "coverage": {
                    "named_book_pre_tip": {name: dict(market) for name in markets}
                },
            }
        },
    }


def _receipt(tmp_path, league: str = "nba") -> dict:
    import json

    for name in (f"auto_tickets_{DAY}_decision.json", f"auto_tickets_{DAY}_no_bet.json"):
        path = tmp_path / league / name
        if path.exists():
            return json.loads(path.read_text())
    raise AssertionError("nothing was frozen")


def _policy(tmp_path, league: str = "nba") -> dict:
    return _receipt(tmp_path, league)["selection_policy"]


def _run(tmp_path, *, probe, monkeypatch, prefer=BPI_SOURCE, league="nba", predictor_ok=True):
    monkeypatch.setattr(bb, "resolve_surface", lambda _brier, **_kw: prefer)
    with Store(tmp_path / "bullion.db") as store:
        _seed_nba_calibration(store)
        return run_lane(
            league=LEAGUES[league],
            day=DAY,
            store=store,
            localdata=tmp_path,
            client=_client(with_predictor=predictor_ok),
            as_of_utc="2026-11-28T19:00:00Z",
            probe_report=probe,
        )


# ---------------------------------------------------------------------------
# T7.1 — the Brier winner is demoted when it cannot price a permitted market
# ---------------------------------------------------------------------------


def test_a_brier_winner_missing_a_permitted_market_is_demoted(tmp_path, monkeypatch):
    """BPI wins the moneyline Brier but has no total market at all."""
    run = _run(tmp_path, probe=_probe("moneyline", "total"), monkeypatch=monkeypatch)
    gap = next(g for g in run.gaps if g.field == "surface.market_coverage")
    assert "['total']" in gap.reason
    policy = _policy(tmp_path)
    assert policy["surface_resolution"] == f"coverage_demoted_{EMPIRICAL_SOURCE}"
    assert policy["ticket_markets"] == ["moneyline", "total"]
    pool = _receipt(tmp_path)["candidate_pool"]
    assert pool, "the demoted surface must still price the board"
    assert {row["source"] for row in pool} == {EMPIRICAL_SOURCE}
    assert {row["market"] for row in pool} == {"moneyline", "total"}


def test_exactly_one_surface_survives_the_demotion(tmp_path, monkeypatch):
    """No mixed ticket: the demotion is a swap, not an addition."""
    run = _run(tmp_path, probe=_probe("moneyline", "total"), monkeypatch=monkeypatch)
    assert _policy(tmp_path)["surface_resolution"] == f"coverage_demoted_{EMPIRICAL_SOURCE}"
    assert {row["source"] for row in _receipt(tmp_path)["candidate_pool"]} == {EMPIRICAL_SOURCE}
    assert list(run.ticket_markets) == ["moneyline", "total"]
    # Every ticket leg that was built came from that one surface.
    for ticket in _receipt(tmp_path).get("accas") or []:
        for leg in ticket["legs"]:
            assert leg["source"] == EMPIRICAL_SOURCE


def test_a_covering_winner_is_kept(tmp_path, monkeypatch):
    """When the winner covers every permitted market, nothing is demoted."""
    run = _run(tmp_path, probe=_probe("moneyline"), monkeypatch=monkeypatch)
    assert not any(g.field == "surface.market_coverage" for g in run.gaps)
    policy = _policy(tmp_path)
    assert policy["surface_resolution"] == BPI_SOURCE
    assert {row["source"] for row in _receipt(tmp_path)["candidate_pool"]} == {BPI_SOURCE}


def test_the_gap_names_the_surface_that_won_on_coverage(tmp_path, monkeypatch):
    run = _run(tmp_path, probe=_probe("moneyline", "total"), monkeypatch=monkeypatch)
    gap = next(g for g in run.gaps if g.field == "surface.market_coverage")
    assert BPI_SOURCE in gap.reason and EMPIRICAL_SOURCE in gap.reason
    assert _policy(tmp_path)["surface_resolution"] == f"coverage_demoted_{EMPIRICAL_SOURCE}"


def test_an_empirical_winner_that_covers_the_board_is_never_demoted_to_bpi(
    tmp_path, monkeypatch
):
    """The guard protects coverage, not one vendor: a covering winner stays."""
    _run(
        tmp_path,
        probe=_probe("moneyline", "total"),
        monkeypatch=monkeypatch,
        prefer=EMPIRICAL_SOURCE,
    )
    assert _policy(tmp_path)["surface_resolution"] == EMPIRICAL_SOURCE


def test_demotion_never_happens_when_nothing_is_permitted(tmp_path, monkeypatch):
    """A research-only lane has no markets to cover, so there is nothing to fix."""
    monkeypatch.setattr(bb, "resolve_surface", lambda _brier, **_kw: BPI_SOURCE)
    with Store(tmp_path / "bullion.db") as store:
        _seed_nba_calibration(store)
        run = run_lane(
            league=LEAGUES["nba"],
            day=DAY,
            store=store,
            localdata=tmp_path,
            client=_client(),
            as_of_utc="2026-11-28T19:00:00Z",
        )
    assert not any(g.field == "surface.market_coverage" for g in run.gaps)
    assert list(run.ticket_markets) == []
    assert _policy(tmp_path)["surface_resolution"] in {
        "no_surface_evidence",
        BPI_SOURCE,
        EMPIRICAL_SOURCE,
    }


def test_every_permitted_market_is_priced_after_the_demotion(tmp_path, monkeypatch):
    """The permitted board is covered in full — no market is silently dropped."""
    run = _run(tmp_path, probe=_probe("moneyline", "total", "spread"), monkeypatch=monkeypatch)
    policy = _policy(tmp_path)
    assert {row["market"] for row in _receipt(tmp_path)["candidate_pool"]} == {
        "moneyline",
        "spread",
        "total",
    }
    assert policy["ticket_markets"] == ["moneyline", "spread", "total"]
    assert run.status in {"frozen_ticket", "frozen_no_bet"}


# ---------------------------------------------------------------------------
# T7.2 — resolve_surface itself is unchanged
# ---------------------------------------------------------------------------


def test_resolve_surface_still_needs_both_sides_and_a_real_gap():
    assert resolve_surface({BPI_SOURCE: 0.18, EMPIRICAL_SOURCE: 0.21}) == BPI_SOURCE
    assert resolve_surface({BPI_SOURCE: 0.21, EMPIRICAL_SOURCE: 0.18}) == EMPIRICAL_SOURCE
    assert resolve_surface({BPI_SOURCE: 0.20, EMPIRICAL_SOURCE: 0.201}) is None
    assert resolve_surface({BPI_SOURCE: 0.18}) is None
    assert resolve_surface({}) is None


# ---------------------------------------------------------------------------
# T7.3 — the anchored-sigma stamp lands on the receipt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("league,anchored", [("ncaa_m", True), ("nba", False)])
def test_the_sigma_model_stamp_is_written_for_anchored_lanes(tmp_path, monkeypatch, league, anchored):
    monkeypatch.setattr(bb, "resolve_surface", lambda _brier, **_kw: EMPIRICAL_SOURCE)
    with Store(tmp_path / "bullion.db") as store:
        _seed_nba_calibration(store)
        run_lane(
            league=LEAGUES[league],
            day=DAY,
            store=store,
            localdata=tmp_path,
            client=_client(),
            as_of_utc="2026-11-28T19:00:00Z",
            probe_report=_probe("moneyline", "spread", "total", league=league),
        )
    policy = _policy(tmp_path, league)
    if anchored:
        assert policy["sigma_model"] == "anchored_orthogonal"
    else:
        assert "sigma_model" not in policy
