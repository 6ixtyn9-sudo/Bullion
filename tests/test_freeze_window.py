"""F3 — missed freeze-window visibility and per-leg freeze age."""

from __future__ import annotations

import json

from bullion.espn import EspnClient
from bullion.models import LEAGUES
from bullion.runner import run_lane
from bullion.store import Store

from .factories import close, event, offer, summary
from .test_runner import _probe_report, _seed_history

DAY = "2026-11-28"


def _offers() -> list:
    return [
        offer(
            moneyline={"home": close(-110), "away": close(-110)},
            spread={"home": close(-155, "-1.5"), "away": close(135, "+1.5")},
            total={"over": close(-110, "o6.5"), "under": close(-110, "u6.5")},
        )
    ]


class Board:
    """Scoreboard for one day plus post summaries for its games."""

    def __init__(self, events: list) -> None:
        self.events = events
        self.calls: list[str] = []

    def __call__(self, url: str):
        self.calls.append(url)
        if "/summary?event=" in url:
            return summary(home="h1", away="a1", home_score="4", away_score="1")
        return {"events": self.events}


def _event(event_id: str, kickoff: str, *, state: str = "pre") -> dict:
    return event(
        event_id=event_id,
        home="h1",
        away="a1",
        date=kickoff,
        state=state,
        offers=_offers() if state == "pre" else [],
        home_score="4" if state == "post" else None,
        away_score="1" if state == "post" else None,
    )


def _run(tmp_path, board: Board, *, as_of: str):
    with Store(tmp_path / "db.sqlite") as store:
        _seed_history(store)
        return run_lane(
            league=LEAGUES["nhl"],
            day=DAY,
            store=store,
            localdata=tmp_path,
            client=EspnClient(fetch=board),
            as_of_utc=as_of,
            probe_report=_probe_report(),
        )


def test_a_fully_tipped_day_with_no_receipt_freezes_a_terminal_no_bet(tmp_path):
    """Every game already tipped and nothing was ever frozen: say so, once."""
    board = Board([_event("e0", "2026-11-28T09:00:00Z")])  # 07:00Z < as_of
    run = _run(tmp_path, board, as_of="2026-11-28T11:00:00Z")
    assert run.status == "frozen_no_bet", run.reason
    assert run.reason == "missed_freeze_window"
    gap = next(g for g in run.gaps if g.field == "slate.missed_freeze_window")
    assert "1 of 1 day games tipped before any scan reached the cutoff" in gap.reason

    lock = json.loads((tmp_path / "nhl" / f"auto_tickets_{DAY}_no_bet.json").read_text())
    assert lock["reason"] == "missed_freeze_window"
    assert lock["positions"] == []
    assert lock["candidate_pool"] == []


def test_a_missed_window_never_overwrites_an_existing_receipt(tmp_path):
    """Write-once holds: an existing decision file stays the authority."""
    lane = tmp_path / "nhl"
    lane.mkdir(parents=True, exist_ok=True)
    decision = lane / f"auto_tickets_{DAY}_decision.json"
    decision.write_text(json.dumps({"schema_version": "magolide.paper_ticket_receipt"}))

    board = Board([_event("e0", "2026-11-28T09:00:00Z")])
    run = _run(tmp_path, board, as_of="2026-11-28T11:00:00Z")
    assert run.status == "no_slate", "a frozen day is not a missed one"
    assert json.loads(decision.read_text())["schema_version"] == (
        "magolide.paper_ticket_receipt"
    )
    assert not (lane / f"auto_tickets_{DAY}_no_bet.json").exists()


def _frozen(tmp_path) -> dict:
    """The day's frozen payload — a ticket receipt or a terminal no-bet lock."""
    lane = tmp_path / "nhl"
    for name in (f"auto_tickets_{DAY}_decision.json", f"auto_tickets_{DAY}_no_bet.json"):
        path = lane / name
        if path.exists():
            return json.loads(path.read_text())
    raise AssertionError("nothing was frozen for the day")


def test_a_partial_tip_over_keeps_the_live_slate_and_reports_the_shortfall(tmp_path):
    """One game tipped, two still to come: freeze the two, disclose the one."""
    board = Board(
        [
            _event("e0", "2026-11-28T09:00:00Z"),  # tipped
            _event("e1", "2026-11-28T19:00:00Z"),  # live slate
            _event("e2", "2026-11-28T20:00:00Z"),  # live slate
        ]
    )
    run = _run(tmp_path, board, as_of="2026-11-28T17:00:00Z")
    assert run.status in {"frozen_ticket", "frozen_no_bet"}
    gap = next(g for g in run.gaps if g.field == "slate.tipped_before_freeze")
    assert gap.reason == "1 of 3 day games tipped before freeze"
    # The live slate was not thrown away: both upcoming games reached the pool.
    pool = _frozen(tmp_path)["candidate_pool"]
    assert pool, "the remaining slate must still be priced"
    assert {row["game_id"] for row in pool} <= {"e1", "e2"}


def test_freeze_age_is_stamped_on_every_pool_row_and_leg(tmp_path):
    board = Board([_event("e0", "2026-11-28T19:00:00Z")])
    run = _run(tmp_path, board, as_of="2026-11-28T17:00:00Z")
    receipt = _frozen(tmp_path)
    assert receipt["candidate_pool"]
    for row in receipt["candidate_pool"]:
        assert row["freeze_age_hours"] == 2.0, "19:00Z tip frozen at 17:00Z"
    for ticket in receipt.get("accas") or []:
        for leg in ticket["legs"]:
            assert leg["freeze_age_hours"] == 2.0
    assert run.status in {"frozen_ticket", "frozen_no_bet"}


def test_a_missed_window_still_settles_yesterdays_positions(tmp_path):
    """The terminal lock must not cost settlement: finish() still runs."""
    from .test_runner import STALE_ID, _pending_receipt, _stale_game

    _pending_receipt(tmp_path)
    with Store(tmp_path / "db.sqlite") as store:
        _stale_game(store)
        run = run_lane(
            league=LEAGUES["nhl"],
            day=DAY,
            store=store,
            localdata=tmp_path,
            client=EspnClient(fetch=Board([_event("e0", "2026-11-27T09:00:00Z")])),
            as_of_utc="2026-11-28T11:00:00Z",
            probe_report=_probe_report(),
        )
        refreshed = store.get_game(STALE_ID)
    assert refreshed is not None and refreshed.status.value == "final"
    performance = json.loads(
        (tmp_path / "nhl" / "auto_tickets_performance.json").read_text()
    )
    assert performance["settled_positions"] == 1
    assert run.status in {"frozen_no_bet", "no_slate"}
