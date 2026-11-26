"""Runner integration on synthetic ESPN payloads (no network, labelled synthetic)."""

from __future__ import annotations

import json

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
            source="nhl_poisson_v1",
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
        assert all(3.0 <= acca["combined_decimal_odds"] <= 4.5 for acca in receipt["accas"])
        assert receipt["schema_version"] == "magolide.paper_ticket_receipt"
        assert receipt["candidate_pool"], "a frozen receipt must carry its pool"
        assert receipt["selection_policy"]["probe_report_used"] is True
        assert (lane_dir / "cards" / f"auto_tickets_{DAY}.txt").exists()
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
