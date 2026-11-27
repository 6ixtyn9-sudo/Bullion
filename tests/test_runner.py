"""Runner integration on synthetic ESPN payloads (no network, labelled synthetic)."""

from __future__ import annotations

import json

import pytest

from bullion.espn import EspnClient
from bullion.models import LEAGUES, ActualResult, Forecast, Game, Status, sast_day
from bullion.runner import run_lane
from bullion.store import Store

from .factories import close, event, offer, summary

DAY = "2026-11-28"
CUTOFF_SAST = "2026-11-28T14:00:00Z"


def _offers() -> list:
    return [
        offer(
            moneyline={"home": close(-110), "away": close(-110)},
            spread={"home": close(-155, "-1.5"), "away": close(135, "+1.5")},
            total={"over": close(-110, "o6.5"), "under": close(-110, "u6.5")},
        )
    ]


def _scoreboard() -> dict:
    events = [
        event(
            event_id=f"e{index}",
            home="h1",
            away="a1",
            home_name="Home Club",
            away_name="Away Club",
            date=f"2026-11-28T{18 + index}:00:00Z",
            offers=_offers(),
        )
        for index in range(4)
    ]
    events.append(
        event(
            event_id="final1",
            home="h1",
            away="a1",
            date="2026-11-28T08:00:00Z",
            state="post",
            home_score="4",
            away_score="1",
            offers=[],
        )
    )
    return {"events": events}


def _bare_scoreboard() -> dict:
    """Scoreboard with the odds array present but empty (the live 2026-11-26 case)."""
    board = _scoreboard()
    for item in board["events"]:
        item["competitions"][0]["odds"] = []
    return board


def _pre_summary(event_id: str) -> dict:
    return {
        "header": {
            "competitions": [
                {
                    "status": {"type": {"state": "pre"}},
                    "competitors": [
                        {
                            "homeAway": "home",
                            "id": "h1",
                            "team": {"id": "h1", "displayName": "Home Club"},
                        },
                        {
                            "homeAway": "away",
                            "id": "a1",
                            "team": {"id": "a1", "displayName": "Away Club"},
                        },
                    ],
                }
            ]
        },
        "pickcenter": [
            {
                "provider": {"displayName": "DraftKings"},
                "moneyline": {"home": close(-110), "away": close(-110)},
                "total": {
                    "over": close(-110, "o6.5"),
                    "under": close(-110, "u6.5"),
                },
            }
        ],
    }


class FakeEspn:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, url: str):
        self.calls.append(url)
        if "/summary?event=" in url:
            return summary(event_id="final1", home="h1", away="a1", home_score="4", away_score="1")
        return _scoreboard()


class PickcenterOnlyEspn(FakeEspn):
    """No scoreboard odds; quotes only exist in the summary pickcenter."""

    def __call__(self, url: str):
        self.calls.append(url)
        if "/summary?event=" in url:
            event_id = url.split("event=")[1].split("&")[0]
            if event_id == "final1":
                return summary(
                    event_id="final1", home="h1", away="a1", home_score="4", away_score="1"
                )
            return _pre_summary(event_id)
        return _bare_scoreboard()


def _seed_history(store: Store) -> None:
    """40 settled pre-tip forecasts so the calibration segment is sufficient."""
    for index in range(40):
        day = f"2026-10-{(index % 28) + 1:02d}"
        game = Game(
            game_id=f"past{index}",
            league="nhl",
            season="2026",
            home_team_id="h1",
            away_team_id="a1",
            home_name="Home Club",
            away_name="Away Club",
            scheduled_utc=f"{day}T18:00:00Z",
            status=Status.FINAL,
            actual=ActualResult(home_score=4, away_score=1),
        )
        store.upsert_game(game)
        forecast = Forecast(
            game_id=game.game_id,
            league="nhl",
            scheduled_utc=game.scheduled_utc,
            home_name=game.home_name,
            away_name=game.away_name,
            market="moneyline",
            selection="home",
            raw_probability=0.62,
            source="nhl_poisson",
        )
        store.record_forecasts([forecast], sast_day(game.scheduled_utc))


def _probe_report(coverage: float = 0.8) -> dict:
    return {
        "schema_version": "bullion.espn_probe",
        "mode": "live",
        "captured_utc": "2026-11-28T06:00:00Z",
        "leagues": {
            "nhl": {
                "coverage": {
                    "named_book_pre_tip": {
                        "moneyline": {
                            "event_coverage": coverage,
                            "sample_size": 8,
                            "status": "ok",
                        },
                        "total": {
                            "event_coverage": coverage,
                            "sample_size": 8,
                            "status": "ok",
                        },
                        "spread": {
                            "event_coverage": coverage,
                            "sample_size": 8,
                            "status": "ok",
                        },
                    }
                }
            }
        },
    }


def test_lane_without_probe_stays_research_only_and_freezes_no_ticket(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        _seed_history(store)
        run = run_lane(
            league=LEAGUES["nhl"],
            day=DAY,
            store=store,
            localdata=tmp_path,
            client=EspnClient(fetch=FakeEspn()),
            as_of_utc="2026-11-28T15:00:00Z",
        )
    assert run.status == "frozen_no_bet"
    assert "espn_probe_evidence_missing" in run.reason
    assert not (tmp_path / "nhl" / f"auto_tickets_{DAY}_decision.json").exists()
    coverage = json.loads((tmp_path / "_coverage" / f"nhl_{DAY}.json").read_text())
    assert coverage["run"]["ticket_markets"] == []
    assert coverage["run"]["games"] == 5


def _run_with_probe(tmp_path, **kwargs):
    with Store(tmp_path / "db.sqlite") as store:
        _seed_history(store)
        return run_lane(
            league=LEAGUES["nhl"],
            day=DAY,
            store=store,
            localdata=tmp_path,
            client=EspnClient(fetch=FakeEspn()),
            as_of_utc="2026-11-28T15:00:00Z",
            probe_report=_probe_report(),
            **kwargs,
        )


def _frozen_policy(tmp_path):
    path = tmp_path / "nhl" / f"auto_tickets_{DAY}_decision.json"
    if not path.exists():
        path = tmp_path / "nhl" / f"auto_tickets_{DAY}_no_bet.json"
    return json.loads(path.read_text())["selection_policy"]


def test_default_selection_policy_is_option_1(tmp_path):
    """Option 1 shape and Option A tripwire are the defaults a lane freezes."""
    _run_with_probe(tmp_path)
    policy = _frozen_policy(tmp_path)
    assert policy["ranking_metric"] == "kelly_log_growth"
    assert policy["min_legs"] == 2 and policy["max_legs"] == 3
    assert policy["requested_legs"] == 3
    assert policy["limit"] == 3
    assert policy["min_combined_odds"] == 3.30 and policy["max_combined_odds"] == 5.00
    assert policy["min_leg_odds"] == 1.20 and policy["max_leg_odds"] == 3.00
    assert policy["max_claimed_edge"] == 0.40
    assert policy["edge_flag_threshold"] == 0.15
    assert policy["forbid_alt_line_reuse"] is True
    assert policy["forbid_conflicting_game_sides"] is True
    assert policy["stake_units_per_acca"] == 1.0


def test_pinned_legs_and_blend_reproduce_the_legacy_contract(tmp_path):
    _run_with_probe(tmp_path, legs=2, ranking_metric="probability_price_blend")
    policy = _frozen_policy(tmp_path)
    assert policy["requested_legs"] == 2
    assert policy["min_legs"] == 2 and policy["max_legs"] == 2
    assert policy["ranking_metric"] == "probability_price_blend"


def test_lane_rejects_an_impossible_shape(tmp_path):
    with pytest.raises(ValueError):
        _run_with_probe(tmp_path, min_legs=1, max_legs=3)
    with pytest.raises(ValueError):
        _run_with_probe(tmp_path, min_legs=4, max_legs=2)
    with pytest.raises(ValueError):
        _run_with_probe(tmp_path, ranking_metric="made_up")


def test_pre_cutoff_scan_never_freezes(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        _seed_history(store)
        run = run_lane(
            league=LEAGUES["nhl"],
            day=DAY,
            store=store,
            localdata=tmp_path,
            client=EspnClient(fetch=FakeEspn()),
            as_of_utc="2026-11-28T10:00:00Z",
            probe_report=_probe_report(),
        )
    assert run.status == "pre_cutoff_scan"
    assert run.cutoff_utc == CUTOFF_SAST
    assert not (tmp_path / "nhl" / f"auto_tickets_{DAY}_decision.json").exists()
    assert not (tmp_path / "nhl" / f"auto_tickets_{DAY}_no_bet.json").exists()


def test_cutoff_scan_with_probe_freezes_a_receipt_or_no_bet_lock(tmp_path):
    client = FakeEspn()
    with Store(tmp_path / "db.sqlite") as store:
        _seed_history(store)
        run = run_lane(
            league=LEAGUES["nhl"],
            day=DAY,
            store=store,
            localdata=tmp_path,
            client=EspnClient(fetch=client),
            as_of_utc="2026-11-28T15:00:00Z",
            probe_report=_probe_report(),
        )
        finals = store.get_game("final1")
    assert run.status in {"frozen_ticket", "frozen_no_bet"}
    assert finals is not None and finals.status is Status.FINAL
    assert finals.actual.home_score == 4.0
    lane_dir = tmp_path / "nhl"
    receipt_path = lane_dir / f"auto_tickets_{DAY}_decision.json"
    if run.status == "frozen_ticket":
        receipt = json.loads(receipt_path.read_text())
        assert all(3.30 <= acca["combined_decimal_odds"] <= 5.00 for acca in receipt["accas"])
        assert all(2 <= acca["leg_count"] <= 3 for acca in receipt["accas"])
        assert receipt["selection_policy"]["ranking_metric"] == "kelly_log_growth"
        assert receipt["schema_version"] == "magolide.paper_ticket_receipt"
        assert receipt["candidate_pool"], "a frozen receipt must carry its pool"
        assert receipt["selection_policy"]["probe_report_used"] is True
        assert (lane_dir / f"auto_tickets_{DAY}.txt").exists()
        assert (lane_dir / "auto_tickets_performance.txt").exists()
        assert (lane_dir / "auto_tickets_performance.json").exists()
        from bullion.replay import check_parity, load_evidence

        parity = check_parity(load_evidence(tmp_path))
        assert parity["parity"] is True, parity["mismatches"]
    else:
        assert (lane_dir / f"auto_tickets_{DAY}_no_bet.json").exists()


def test_frozen_receipt_is_not_rotated_by_a_later_scan(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        _seed_history(store)
        first = run_lane(
            league=LEAGUES["nhl"],
            day=DAY,
            store=store,
            localdata=tmp_path,
            client=EspnClient(fetch=FakeEspn()),
            as_of_utc="2026-11-28T15:00:00Z",
            probe_report=_probe_report(),
        )
        path = tmp_path / "nhl" / f"auto_tickets_{DAY}_decision.json"
        before = path.read_bytes() if path.exists() else None
        second = run_lane(
            league=LEAGUES["nhl"],
            day=DAY,
            store=store,
            localdata=tmp_path,
            client=EspnClient(fetch=FakeEspn()),
            as_of_utc="2026-11-28T17:30:00Z",
            probe_report=_probe_report(),
        )
    assert second.status == first.status
    if before is not None:
        assert path.read_bytes() == before, "a frozen receipt must be write-once"


def test_summary_pickcenter_prices_a_game_when_scoreboard_odds_are_empty(tmp_path):
    client = PickcenterOnlyEspn()
    with Store(tmp_path / "db.sqlite") as store:
        _seed_history(store)
        run = run_lane(
            league=LEAGUES["nhl"],
            day=DAY,
            store=store,
            localdata=tmp_path,
            client=EspnClient(fetch=client),
            as_of_utc="2026-11-28T15:00:00Z",
            probe_report=_probe_report(),
        )
        quotes = store.quotes_for_game("e0")
    assert run.status in {"frozen_ticket", "frozen_no_bet"}
    assert any("/summary?event=e0" in url for url in client.calls)
    assert quotes, "summary pickcenter quotes must be stored for pricing"
    assert {quote.book for quote in quotes} == {"DraftKings"}
    assert all(quote.provenance == "named_book" for quote in quotes)
    coverage = json.loads((tmp_path / "_coverage" / f"nhl_{DAY}.json").read_text())
    gaps_for_e0 = [
        gap for gap in coverage["run"]["gaps"] if gap["game_id"] == "e0"
    ]
    # Discovery still records that the *scoreboard* array was empty; what must
    # not appear is the "unpriceable" gap, because the summary priced the game.
    assert all(gap["reason"] != "no pre-tip quote for forecast" for gap in gaps_for_e0)


def test_probe_report_without_sample_size_cannot_unlock_tickets(tmp_path):
    legacy = _probe_report()
    for entry in legacy["leagues"]["nhl"]["coverage"]["named_book_pre_tip"].values():
        entry.pop("sample_size")
        entry.pop("status")
    with Store(tmp_path / "db.sqlite") as store:
        _seed_history(store)
        run = run_lane(
            league=LEAGUES["nhl"],
            day=DAY,
            store=store,
            localdata=tmp_path,
            client=EspnClient(fetch=FakeEspn()),
            as_of_utc="2026-11-28T15:00:00Z",
            probe_report=legacy,
        )
    assert run.status == "frozen_no_bet"
    assert run.ticket_markets == []
    assert "espn_probe_coverage_insufficient_sample" in run.reason
    assert not (tmp_path / "nhl" / f"auto_tickets_{DAY}_decision.json").exists()


# ---------------------------------------------------------------------------
# Silent-stall regressions (found in the 2026-11-26 read-only audit, fixed
# before the first live slate on 2026-11-27)
# ---------------------------------------------------------------------------


class StaleFinalEspn:
    """Scoreboard for ``day`` only; a stale event's summary reports ``post``."""

    def __init__(self, board: dict | None = None) -> None:
        self.calls: list[str] = []
        self.board = board or {"events": []}

    def __call__(self, url: str):
        self.calls.append(url)
        if "/summary?event=" in url:
            event_id = url.split("event=")[1].split("&")[0]
            return summary(event_id=event_id, home="h1", away="a1", home_score="4", away_score="1")
        return self.board


STALE_ID = "y1"
STALE_DAY = "2026-11-27"


def _stale_game(store: Store, *, status: Status = Status.SCHEDULED) -> None:
    store.upsert_game(
        Game(
            game_id=STALE_ID,
            league="nhl",
            season="2026",
            home_team_id="h1",
            away_team_id="a1",
            home_name="Home Club",
            away_name="Away Club",
            scheduled_utc=f"{STALE_DAY}T23:00:00Z",
            status=status,
        )
    )


def _pending_receipt(tmp_path) -> None:
    """A yesterday receipt whose only leg is not yet graded."""
    leg = {
        "forecast_id": "fid-y1",
        "pick_id": "fid-y1",
        "game_id": STALE_ID,
        "market": "moneyline",
        "period": "full_game",
        "selection": "home",
        "line": None,
        "book": {
            "book": "DraftKings",
            "provider": "espn",
            "selected_decimal_price": 1.95,
            "side_prices": {"home": 1.95, "away": 1.90},
            "pre_match": True,
        },
        "price_evidence": {"book": "DraftKings", "selected_decimal_price": 1.95},
        "calibrated_probability": 0.55,
        "raw_probability": 0.55,
        "push_probability": 0.0,
        "eligibility": {"acca_eligible": True, "reasons": []},
    }
    receipt = {
        "schema_version": "magolide.paper_ticket_receipt",
        "date": STALE_DAY,
        "league": "NHL",
        "decision_id": "d-y1",
        "status": "frozen",
        "paper_only": True,
        "captured_at": f"{STALE_DAY}T15:00:00Z",
        "frozen_at": f"{STALE_DAY}T15:00:00Z",
        "accas": [
            {
                "ticket_id": "t-y1",
                "ticket_type": "acca",
                "leg_count": 1,
                "combined_decimal_odds": 1.95,
                "stake_units": 1.0,
                "estimated_joint_win_probability": 0.55,
                "legs": [dict(leg)],
            }
        ],
        "singles": [],
        "candidate_pool": [dict(leg)],
        "selection_policy": {"min_leg_odds": 1.20, "max_leg_odds": 3.00},
    }
    lane_dir = tmp_path / "nhl"
    lane_dir.mkdir(parents=True, exist_ok=True)
    (lane_dir / f"auto_tickets_{STALE_DAY}_decision.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )


def _pool_rows(tmp_path) -> list:
    """The frozen pool of a finished run, whatever the freeze decided."""
    lane_dir = tmp_path / "nhl"
    for name in (f"auto_tickets_{DAY}_decision.json", f"auto_tickets_{DAY}_no_bet.json"):
        path = lane_dir / name
        if path.exists():
            return json.loads(path.read_text()).get("candidate_pool") or []
    return []


def test_pickcenter_only_quotes_reach_scoring_and_price_the_pool(tmp_path):
    """Bug 1: ``score_forecasts`` was fed the scoreboard array only."""
    with Store(tmp_path / "db.sqlite") as store:
        _seed_history(store)
        run = run_lane(
            league=LEAGUES["nhl"],
            day=DAY,
            store=store,
            localdata=tmp_path,
            client=EspnClient(fetch=PickcenterOnlyEspn()),
            as_of_utc="2026-11-28T15:00:00Z",
            probe_report=_probe_report(),
        )
    pool = _pool_rows(tmp_path)
    assert pool, "a pickcenter-only slate must still score"
    priced = [row for row in pool if row["game_id"] == "e0"]
    assert priced, "e0 is priced only by its summary pickcenter"
    assert all(row["book"]["selected_decimal_price"] > 1.0 for row in priced)
    assert run.candidate_pool == len(pool)


def test_no_slate_run_refreshes_yesterdays_final_and_settles_its_receipt(tmp_path):
    """Bug 2: the 06:15Z grade cron returned on ``no_slate`` before settling."""
    _pending_receipt(tmp_path)
    with Store(tmp_path / "db.sqlite") as store:
        _stale_game(store)
        run = run_lane(
            league=LEAGUES["nhl"],
            day=DAY,
            store=store,
            localdata=tmp_path,
            client=EspnClient(fetch=StaleFinalEspn()),
            as_of_utc="2026-11-28T06:15:00Z",
            probe_report=_probe_report(),
        )
        refreshed = store.get_game(STALE_ID)
    assert run.status == "no_slate"
    assert run.refreshed_finals == 1
    assert refreshed is not None and refreshed.status is Status.FINAL
    assert refreshed.actual is not None and refreshed.actual.home_score == 4.0
    performance = json.loads((tmp_path / "nhl" / "auto_tickets_performance.json").read_text())
    assert performance["settled_positions"] == 1
    assert performance["wins"] == 1
    assert performance["profit_units"] == pytest.approx(0.95)


def test_pre_cutoff_scan_also_refreshes_and_settles(tmp_path):
    """The same stall applied to every lane run before its cutoff."""
    _pending_receipt(tmp_path)
    board = {"events": [event(event_id="e0", date="2026-11-28T18:00:00Z", offers=_offers())]}
    with Store(tmp_path / "db.sqlite") as store:
        _stale_game(store)
        run = run_lane(
            league=LEAGUES["nhl"],
            day=DAY,
            store=store,
            localdata=tmp_path,
            client=EspnClient(fetch=StaleFinalEspn(board)),
            as_of_utc="2026-11-28T10:00:00Z",
            probe_report=_probe_report(),
        )
        refreshed = store.get_game(STALE_ID)
    assert run.status == "pre_cutoff_scan"
    assert refreshed is not None and refreshed.status is Status.FINAL
    performance = json.loads((tmp_path / "nhl" / "auto_tickets_performance.json").read_text())
    assert performance["settled_positions"] == 1


def test_in_progress_game_is_not_forced_to_a_final(tmp_path):
    """A still-``in`` summary is skipped quietly: no gap, no invented score."""
    class InProgressEspn(StaleFinalEspn):
        def __call__(self, url: str):
            if "/summary?event=" in url:
                self.calls.append(url)
                return summary(state="in", home="h1", away="a1", home_score="2", away_score="1")
            return super().__call__(url)

    with Store(tmp_path / "db.sqlite") as store:
        _stale_game(store)
        run = run_lane(
            league=LEAGUES["nhl"],
            day=DAY,
            store=store,
            localdata=tmp_path,
            client=EspnClient(fetch=InProgressEspn()),
            as_of_utc="2026-11-28T06:15:00Z",
        )
        game = store.get_game(STALE_ID)
    assert game is not None and game.status is Status.SCHEDULED
    assert run.refreshed_finals == 0
    gaps = [gap for gap in run.gaps if gap.field == "summary.refresh"]
    assert gaps == []
