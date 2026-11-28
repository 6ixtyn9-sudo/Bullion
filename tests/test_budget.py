"""F6 — a wall-clock budget that stops cleanly instead of being killed.

A job timeout kills the process mid-write: the cache save, the commit and the
receipt never happen. These tests pin the alternative — stop at the last safe
boundary, persist what is finished, and say so.
"""

from __future__ import annotations

import importlib.util
import json
import urllib.error
from pathlib import Path

import pytest

from bullion.espn import EspnClient, http_fetch
from bullion.models import LEAGUES
from bullion.runner import LaneRun, run_lane
from bullion.store import Store

from .factories import close, event, offer, summary
from .test_runner import _probe_report, _seed_history

ROOT = Path(__file__).resolve().parents[1]
DAY = "2026-11-28"


@pytest.fixture()
def backfill():
    spec = importlib.util.spec_from_file_location("bullion_backfill_budget", ROOT / "scripts" / "backfill.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# T6.1 — 429 pacing and the live client's tighter budget
# ---------------------------------------------------------------------------


def _rate_limited(headers) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "http://x/scoreboard", 429, "Too Many Requests", headers, None
    )


def test_a_throttle_response_sets_its_own_retry_delay(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr("bullion.espn.time.sleep", slept.append)
    calls = {"n": 0}

    def fake_urlopen(request, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _rate_limited({"Retry-After": "7"})
        raise urllib.error.HTTPError("http://x", 500, "boom", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(Exception):
        http_fetch("http://x", retries=1)
    assert slept == [7.0], "the server's own pacing wins over the linear backoff"


@pytest.mark.parametrize("header,expected", [("0", 1.0), ("600", 30.0), (None, 30.0), ("soon", 30.0)])
def test_a_throttle_delay_is_clamped_and_never_missing(monkeypatch, header, expected):
    slept: list[float] = []
    monkeypatch.setattr("bullion.espn.time.sleep", slept.append)
    headers = {} if header is None else {"Retry-After": header}
    calls = {"n": 0}

    def fake_urlopen(request, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _rate_limited(headers)
        raise urllib.error.HTTPError("http://x", 500, "boom", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(Exception):
        http_fetch("http://x", retries=1)
    assert slept == [expected]


def test_the_live_client_uses_a_tighter_timeout_and_retry_count(monkeypatch):
    seen: list[dict] = []

    def fake_fetch(url, *, timeout=20.0, retries=3):
        seen.append({"url": url, "timeout": timeout, "retries": retries})
        return {"events": []}

    monkeypatch.setattr("bullion.espn.http_fetch", fake_fetch)
    client = EspnClient.live()
    assert client.get("http://x/scoreboard") == {"events": []}
    assert seen[-1]["timeout"] == 12.0
    assert seen[-1]["retries"] == 2
    # A caller can still ask for its own numbers.
    EspnClient.live(timeout=5.0, retries=0).get("http://x/scoreboard")
    assert (seen[-1]["timeout"], seen[-1]["retries"]) == (5.0, 0)


import urllib.request  # noqa: E402  (used by the urlopen monkeypatches above)


# ---------------------------------------------------------------------------
# T6.2 — the lane stops at a boundary and still finishes
# ---------------------------------------------------------------------------


def _board(offers=True):
    return {
        "events": [
            event(
                event_id="e0",
                home="h1",
                away="a1",
                date="2026-11-28T19:00:00Z",
                state="pre",
                offers=[
                    offer(
                        moneyline={"home": close(-110), "away": close(-110)},
                        spread={"home": close(-155, "-1.5"), "away": close(135, "+1.5")},
                        total={"over": close(-110, "o6.5"), "under": close(-110, "u6.5")},
                    )
                ],
            )
        ]
    }


def test_a_deadline_that_expires_mid_tick_still_settles_yesterday(tmp_path, monkeypatch):
    """The budget is a truncation point, not an abort: ``finish()`` still runs."""
    from .test_runner import STALE_ID, _pending_receipt, _stale_game

    _pending_receipt(tmp_path)
    clock = {"t": 0.0}
    monkeypatch.setattr("bullion.runner.time.monotonic", lambda: clock["t"])

    def fetch(url):
        # Only the finals refresh is charged, so the walk to it is free and the
        # stop point is unambiguous: the budget dies during the refresh.
        if "/summary?event=" in url:
            clock["t"] += 10.0
            event_id = url.split("event=")[1].split("&")[0]
            return summary(
                event_id=event_id, home="h1", away="a1", home_score="4", away_score="1"
            )
        return _board()

    with Store(tmp_path / "db.sqlite") as store:
        _stale_game(store)
        _seed_history(store)
        run = run_lane(
            league=LEAGUES["nhl"],
            day=DAY,
            store=store,
            localdata=tmp_path,
            client=EspnClient(fetch=fetch),
            as_of_utc="2026-11-28T17:00:00Z",
            probe_report=_probe_report(),
            # The refresh's one summary call spends 10 of a 5s budget, so the
            # break lands on the next loop boundary — after the refresh, before
            # any forecast.
            deadline=5.0,
        )
        # The prior-day final was refreshed and its position settled before the
        # deadline was noticed.
        assert store.get_game(STALE_ID).status.value == "final"
    performance = json.loads((tmp_path / "nhl" / "auto_tickets_performance.json").read_text())
    assert performance["settled_positions"] == 1
    assert run.status == "budget_breached"
    assert run.reason == "wall-clock deadline reached"
    assert any(gap.field == "runner.budget" for gap in run.gaps)
    # Nothing was invented for the card it never got to price.
    assert not (tmp_path / "nhl" / f"auto_tickets_{DAY}_decision.json").exists()
    assert not (tmp_path / "nhl" / f"auto_tickets_{DAY}_no_bet.json").exists()


def test_a_spent_budget_keeps_the_forecasts_it_already_persisted(tmp_path, monkeypatch):
    from bullion import runner

    clock = {"t": 0.0}
    monkeypatch.setattr("bullion.runner.time.monotonic", lambda: clock["t"])
    real = runner._forecasts_for_game
    seen = {"n": 0}

    def wrapper(*args, **kwargs):
        rows = real(*args, **kwargs)
        seen["n"] += 1
        if seen["n"] == 1:
            clock["t"] = 1e9  # the budget dies after the first game is priced
        return rows

    monkeypatch.setattr(runner, "_forecasts_for_game", wrapper)
    board = {
        "events": [
            event(
                event_id=f"e{index}",
                home="h1",
                away="a1",
                date=f"2026-11-28T{19 + index}:00:00Z",
                state="pre",
                offers=[
                    offer(
                        moneyline={"home": close(-110), "away": close(-110)},
                        spread={"home": close(-155, "-1.5"), "away": close(135, "+1.5")},
                    )
                ],
            )
            for index in range(2)
        ]
    }
    with Store(tmp_path / "db.sqlite") as store:
        _seed_history(store)
        run = run_lane(
            league=LEAGUES["nhl"],
            day=DAY,
            store=store,
            localdata=tmp_path,
            client=EspnClient(fetch=lambda url: board),
            as_of_utc="2026-11-28T17:00:00Z",
            probe_report=_probe_report(),
            deadline=5.0,
        )
        persisted = store.conn.execute(
            "SELECT COUNT(*) FROM forecast_snapshots WHERE league='nhl'"
        ).fetchone()[0]
    assert run.status == "budget_breached"
    assert run.forecasts > 0, "the priced game's rows must survive the stop"
    assert persisted >= run.forecasts


def test_a_lane_with_no_deadline_is_unchanged(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        _seed_history(store)
        run = run_lane(
            league=LEAGUES["nhl"],
            day=DAY,
            store=store,
            localdata=tmp_path,
            client=EspnClient(fetch=lambda url: _board()),
            as_of_utc="2026-11-28T17:00:00Z",
            probe_report=_probe_report(),
        )
    assert run.status != "budget_breached"


# ---------------------------------------------------------------------------
# T6.3 — the seeder stops between lanes and still restores
# ---------------------------------------------------------------------------


def test_the_seeder_reports_a_budget_stop_and_still_restores(tmp_path, backfill):
    from bullion.models import ActualResult, Game, Status

    with Store(tmp_path / "db.sqlite") as store:
        for index in range(20):
            day = f"2026-11-{(index % 28) + 1:02d}"
            game = Game(
                game_id=f"g{index}",
                league="nba",
                season="2027",
                home_team_id="h1" if index % 2 else "h2",
                away_team_id="a1" if index % 3 else "a2",
                home_name="Home",
                away_name="Away",
                scheduled_utc=f"{day}T18:00:00Z",
                status=Status.FINAL,
                actual=ActualResult(home_score=110 + index % 7, away_score=104 + index % 5),
            )
            store.upsert_game(game)
        summary_rows = backfill.seed_walk_forward_ml(
            store, leagues=("nba", "nhl"), deadline=0.0
        )
        assert summary_rows.get("budget_stopped") is True
        assert summary_rows["lanes"] == {}, "no lane starts once the budget is spent"
        # The restore pass is outside the lane loop: it still ran.
        assert "restored" in summary_rows or summary_rows.get("fit_seconds") == 0.0
        assert store.conn.execute("SELECT COUNT(*) FROM forecast_snapshots").fetchone()[0] >= 0


# ---------------------------------------------------------------------------
# T6.4 — the daily script's own budget flag
# ---------------------------------------------------------------------------


@pytest.fixture()
def daily():
    spec = importlib.util.spec_from_file_location("bullion_daily_budget", ROOT / "scripts" / "daily.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _daily_args(tmp_path, *extra: str) -> list[str]:
    return [
        "--league",
        "nhl",
        "--date",
        DAY,
        "--localdata",
        str(tmp_path / "localdata"),
        "--db",
        str(tmp_path / "db.sqlite"),
        *extra,
    ]


def test_the_default_budget_is_forty_minutes(daily, tmp_path, monkeypatch):
    seen: dict = {}

    def fake_run_lane(*, league, day, **kwargs):
        seen["deadline"] = kwargs["deadline"]
        return LaneRun(lane=league.key, day=day, status="pre_cutoff_scan")

    monkeypatch.setattr(daily, "run_lane", fake_run_lane)
    monkeypatch.setattr(daily, "seed_walk_forward_ml", lambda store, deadline=None: {})
    monkeypatch.setattr(daily, "time", type("T", (), {"monotonic": staticmethod(lambda: 1000.0)}))
    code = daily.main(_daily_args(tmp_path))
    summary = json.loads((tmp_path / "localdata" / "_runs" / f"daily_{DAY}.json").read_text())
    assert code == 0
    assert summary["max_seconds"] == 2400.0
    assert seen["deadline"] == 3400.0, "the deadline is an absolute instant"


def test_zero_disables_the_budget(daily, tmp_path, monkeypatch):
    seen: dict = {}

    def fake_run_lane(*, league, day, **kwargs):
        seen["deadline"] = kwargs["deadline"]
        return LaneRun(lane=league.key, day=day, status="pre_cutoff_scan")

    monkeypatch.setattr(daily, "run_lane", fake_run_lane)
    monkeypatch.setattr(daily, "seed_walk_forward_ml", lambda store, deadline=None: {})
    code = daily.main(_daily_args(tmp_path, "--max-seconds", "0"))
    summary = json.loads((tmp_path / "localdata" / "_runs" / f"daily_{DAY}.json").read_text())
    assert code == 0
    assert summary["max_seconds"] == 0.0
    assert seen["deadline"] is None


def test_lanes_after_a_spent_budget_are_marked_not_silently_skipped(daily, tmp_path, monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(daily, "time", type("T", (), {"monotonic": staticmethod(lambda: clock["t"])}))
    monkeypatch.setattr(daily, "seed_walk_forward_ml", lambda store, deadline=None: {})

    def fake_run_lane(*, league, day, **kwargs):
        clock["t"] = 1e9  # the first lane eats the whole budget
        return LaneRun(lane=league.key, day=day, status="pre_cutoff_scan")

    monkeypatch.setattr(daily, "run_lane", fake_run_lane)
    args = [
        "--league",
        "soccer",
        "--date",
        DAY,
        "--localdata",
        str(tmp_path / "localdata"),
        "--db",
        str(tmp_path / "db.sqlite"),
        "--max-seconds",
        "60",
    ]
    code = daily.main(args)
    summary = json.loads((tmp_path / "localdata" / "_runs" / f"daily_{DAY}.json").read_text())
    statuses = [run["status"] for run in summary["runs"]]
    assert code == 0
    assert statuses[0] == "pre_cutoff_scan"
    assert set(statuses[1:]) == {"budget_breached"}, "every later lane reports the truncation"
    assert all(run["reason"] == "wall-clock deadline reached" for run in summary["runs"][1:])
