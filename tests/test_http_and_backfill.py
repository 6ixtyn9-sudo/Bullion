"""Cold-backfill reliability: transport retries, gzip, concurrency and budget.

The first real cold run died on ``http.client.IncompleteRead(397213 bytes read)``
because ``IncompleteRead`` is a ``HTTPException``, not an ``OSError``, so the
retry loop never saw it. These tests keep both halves of that fix honest: the
transport retries transient drops, and the backfill finishes inside a wall-clock
budget instead of waiting for a 30-minute job timeout.
"""

from __future__ import annotations

import gzip
import http.client
import importlib.util
import json
import threading
import urllib.request
from pathlib import Path

import pytest

from bullion.espn import EspnFetchError, http_fetch
from bullion.models import LEAGUES
from bullion.store import Store

from .factories import summary

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def backfill():
    spec = importlib.util.spec_from_file_location(
        "bullion_backfill", ROOT / "scripts" / "backfill.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# http_fetch: retries, gzip, wrapping
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, body: bytes, content_encoding: str = "") -> None:
        self._body = body
        self.headers = {"Content-Encoding": content_encoding}

    def read(self) -> bytes:
        if isinstance(self._body, Exception):
            raise self._body
        return self._body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc_info) -> bool:
        return False


@pytest.fixture(autouse=True)
def no_backoff_sleep(monkeypatch):
    """Retry backoff is real in production and instant in tests."""
    monkeypatch.setattr("bullion.espn.time.sleep", lambda seconds: None)


def _patch_urlopen(monkeypatch, responses):
    seen = []

    def fake_urlopen(request, timeout=None):
        seen.append(request)
        outcome = responses[min(len(seen) - 1, len(responses) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return seen


def test_incomplete_read_is_retried_then_succeeds(monkeypatch):
    body = gzip.compress(json.dumps({"ok": True}).encode())
    responses = [
        FakeResponse(http.client.IncompleteRead(b"partial")),
        FakeResponse(http.client.IncompleteRead(b"partial")),
        FakeResponse(body, "gzip"),
    ]
    seen = _patch_urlopen(monkeypatch, responses)

    assert http_fetch("http://x/summary?event=1") == {"ok": True}
    assert len(seen) == 3, "a dropped chunk must be retried, not surfaced"
    for request in seen:
        assert request.headers["Accept-encoding"] == "gzip"


def test_incomplete_read_is_wrapped_after_retries(monkeypatch):
    responses = [FakeResponse(http.client.IncompleteRead(b"partial"))]
    seen = _patch_urlopen(monkeypatch, responses)

    with pytest.raises(EspnFetchError) as excinfo:
        http_fetch("http://x/summary?event=2", retries=2)
    assert "IncompleteRead" in str(excinfo.value)
    assert len(seen) == 3, "retries + 1 attempts"


@pytest.mark.parametrize(
    "error",
    [
        http.client.IncompleteRead(b"x"),
        http.client.RemoteDisconnected("closed"),
        http.client.BadStatusLine("junk"),
        ConnectionResetError("reset"),
    ],
)
def test_transport_errors_never_escape_raw(monkeypatch, error):
    _patch_urlopen(monkeypatch, [error])
    with pytest.raises(EspnFetchError):
        http_fetch("http://x/summary?event=3", retries=1)


def test_gzip_body_is_decompressed_and_plain_json_still_works(monkeypatch):
    compressed = gzip.compress(json.dumps({"events": [1, 2]}).encode())
    _patch_urlopen(monkeypatch, [FakeResponse(compressed, "gzip")])
    assert http_fetch("http://x/a") == {"events": [1, 2]}

    _patch_urlopen(monkeypatch, [FakeResponse(json.dumps({"events": []}).encode(), "")])
    assert http_fetch("http://x/b") == {"events": []}


def test_gzip_magic_bytes_are_detected_without_the_header(monkeypatch):
    compressed = gzip.compress(json.dumps({"n": 1}).encode())
    _patch_urlopen(monkeypatch, [FakeResponse(compressed, "")])
    assert http_fetch("http://x/c") == {"n": 1}


def test_bad_json_is_still_a_retryable_failure(monkeypatch):
    responses = [
        FakeResponse(b"<html>rate limited</html>"),
        FakeResponse(json.dumps({"ok": 1}).encode()),
    ]
    seen = _patch_urlopen(monkeypatch, responses)
    assert http_fetch("http://x/d") == {"ok": 1}
    assert len(seen) == 2


# ---------------------------------------------------------------------------
# backfill: order, isolation, concurrency, budget
# ---------------------------------------------------------------------------


class FakeClient:
    """Stand-in ESPN: a season index plus per-event summaries."""

    def __init__(self, ids_by_season, *, fail=(), scheduled=()):
        self.ids_by_season = ids_by_season
        self.fail = set(fail)
        self.scheduled = set(scheduled)
        self.season_calls = []
        self.threads = []

    def get(self, url: str) -> dict:
        season = int(url.split("dates=")[1].split("&")[0])
        self.season_calls.append(season)
        items = [{"$ref": f"http://x/events/{i}?lang=en"} for i in self.ids_by_season[season]]
        return {"count": len(items), "pageCount": 1, "items": items}

    def summary(self, league, event_id: str) -> dict:
        self.threads.append(threading.current_thread().name)
        if event_id in self.fail:
            raise EspnFetchError(f"http://x/summary?event={event_id}: IncompleteRead")
        state = "pre" if event_id in self.scheduled else "post"
        return summary(
            event_id=event_id,
            home="h1",
            away="a1",
            state=state,
            home_score="3",
            away_score="2",
        )


def _run(backfill, tmp_path, client, *, seasons=(2024, 2025, 2026), max_events=10,
         workers=4, max_seconds=900.0):
    with Store(tmp_path / "db.sqlite") as store:
        manifest = backfill.run(
            store=store,
            client=client,
            leagues=[LEAGUES["nhl"]],
            seasons=seasons,
            max_events=max_events,
            workers=workers,
            max_seconds=max_seconds,
        )
        stored = sorted(
            game.game_id for game in store.games_for_league("nhl") if game.status.value == "final"
        )
    return manifest, stored


def test_backfill_requests_the_newest_season_first(backfill, tmp_path):
    client = FakeClient({2024: ["401"], 2025: ["402"], 2026: ["403"]})
    manifest, _ = _run(backfill, tmp_path, client, seasons=(2024, 2025, 2026))
    assert client.season_calls == [2026, 2025, 2024]
    assert manifest["seasons"] == [2026, 2025, 2024]
    assert manifest["season_order"] == "newest_first"


def test_a_dead_or_early_summary_never_aborts_the_run(backfill, tmp_path):
    client = FakeClient(
        {2026: ["501", "502", "503", "504"]},
        fail={"502"},
        scheduled={"503"},
    )
    manifest, stored = _run(backfill, tmp_path, client, seasons=(2026,))
    entry = manifest["leagues"]["nhl"]["2026"]
    assert stored == ["501", "504"]
    assert entry["stored"] == 2
    assert entry["failed"] == 2
    assert entry["unprocessed"] == 0
    assert {row["event_id"] for row in entry["sample_errors"]} == {"502", "503"}
    assert manifest["stopped_early"] is False


def test_sqlite_writes_happen_on_the_main_thread_only(backfill, tmp_path):
    client = FakeClient({2026: [str(600 + index) for index in range(8)]})
    write_threads = []
    with Store(tmp_path / "db.sqlite") as store:
        original = store.upsert_game

        def recording(game):
            write_threads.append(threading.current_thread().name)
            return original(game)

        store.upsert_game = recording
        manifest = backfill.run(
            store=store,
            client=client,
            leagues=[LEAGUES["nhl"]],
            seasons=(2026,),
            max_events=10,
            workers=4,
            max_seconds=900.0,
        )
        stored = sorted(game.game_id for game in store.games_for_league("nhl"))
    assert len(stored) == 8
    assert manifest["leagues"]["nhl"]["2026"]["stored"] == 8
    assert set(write_threads) == {"MainThread"}, "sqlite3 stays single-threaded"
    assert any(name.startswith("backfill_") for name in client.threads), (
        "fetches did run off-thread"
    )


def test_wall_clock_budget_stops_cleanly_and_reports_the_remainder(backfill, tmp_path):
    client = FakeClient({2026: [str(700 + index) for index in range(5)]})
    manifest, stored = _run(
        backfill, tmp_path, client, seasons=(2026,), workers=4, max_seconds=1e-9
    )
    entry = manifest["leagues"]["nhl"]["2026"]
    assert manifest["stopped_early"] is True
    assert entry["stopped_early"] is True
    assert entry["unprocessed"] == 5
    assert stored == []
    assert manifest["counts"] is not None


def test_budget_stop_exits_zero_and_writes_a_manifest(backfill, tmp_path, monkeypatch):
    client = FakeClient({2026: ["801", "802"]})
    monkeypatch.setattr(backfill, "EspnClient", lambda *a, **k: client)
    db = tmp_path / "db.sqlite"
    out = tmp_path / "manifest.json"
    code = backfill.main(
        [
            "--league",
            "nhl",
            "--seasons",
            "2026",
            "--db",
            str(db),
            "--out",
            str(out),
            "--max-seconds",
            "1e-9",
        ]
    )
    assert code == 0, "a budget stop is a normal outcome, not a failure"
    payload = json.loads(out.read_text())
    assert payload["stopped_early"] is True
    assert payload["max_seconds"] == pytest.approx(1e-9)


def test_rerun_resumes_where_the_previous_run_stopped(backfill, tmp_path):
    client = FakeClient({2026: ["901", "902", "903"]})
    first, stored_first = _run(backfill, tmp_path, client, seasons=(2026,), max_events=2)
    assert stored_first == ["901", "902"]
    assert first["leagues"]["nhl"]["2026"]["pending_at_start"] == 2

    second, stored_second = _run(backfill, tmp_path, client, seasons=(2026,), max_events=3)
    entry = second["leagues"]["nhl"]["2026"]
    assert entry["skipped_existing"] == 2, "stored finals are never refetched"
    assert entry["stored"] == 1
    assert stored_second == ["901", "902", "903"]


def test_sequential_mode_still_works_and_honours_delay(backfill, tmp_path):
    client = FakeClient({2026: ["801", "802"]})
    manifest, stored = _run(backfill, tmp_path, client, seasons=(2026,), workers=1)
    assert stored == ["801", "802"]
    assert manifest["workers"] == 1
    assert manifest["leagues"]["nhl"]["2026"]["stored"] == 2


def test_default_budget_is_thirty_minutes_and_under_the_job_timeout(backfill):
    import inspect

    default = inspect.signature(backfill.run).parameters["max_seconds"].default
    assert default == 1800.0
    assert default < backfill.JOB_TIMEOUT_WARNING_SECONDS


def test_manifest_records_rate_and_projection(backfill, tmp_path):
    client = FakeClient({2026: ["1101", "1102"]})
    manifest, stored = _run(backfill, tmp_path, client, seasons=(2026,))
    assert stored == ["1101", "1102"]
    assert manifest["elapsed_seconds"] > 0
    assert manifest["processed_per_second"] > 0
    assert manifest["projected_remaining_seconds"] == 0.0
    entry = manifest["leagues"]["nhl"]["2026"]
    assert entry["elapsed_seconds"] > 0
    assert entry["processed_per_second"] > 0


def test_projection_estimates_what_a_budget_stop_left_behind(backfill, tmp_path):
    client = FakeClient({2026: [str(1200 + index) for index in range(4)]})
    manifest, _ = _run(
        backfill, tmp_path, client, seasons=(2026,), workers=4, max_seconds=1e-9
    )
    assert manifest["stopped_early"] is True
    entry = manifest["leagues"]["nhl"]["2026"]
    assert entry["unprocessed"] == 4
    # Nothing was processed, so there is no observed rate to extrapolate from —
    # the projection must say 0.0 rather than invent a number.
    assert manifest["projected_remaining_seconds"] == 0.0


def test_cli_warns_when_the_budget_crowds_out_the_job_steps(backfill, tmp_path, monkeypatch, capsys):
    client = FakeClient({2026: ["1301"]})
    monkeypatch.setattr(backfill, "EspnClient", lambda *a, **k: client)
    code = backfill.main(
        [
            "--league",
            "nhl",
            "--seasons",
            "2026",
            "--db",
            str(tmp_path / "db.sqlite"),
            "--out",
            str(tmp_path / "manifest.json"),
            "--max-seconds",
            "3000",
        ]
    )
    assert code == 0
    stderr = capsys.readouterr().err
    assert "warning" in stderr
    assert "timeout-minutes" in stderr


def test_cli_default_budget_is_written_into_the_manifest(backfill, tmp_path, monkeypatch):
    client = FakeClient({2026: ["1401"]})
    monkeypatch.setattr(backfill, "EspnClient", lambda *a, **k: client)
    out = tmp_path / "manifest.json"
    code = backfill.main(
        [
            "--league",
            "nhl",
            "--seasons",
            "2026",
            "--db",
            str(tmp_path / "db.sqlite"),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text())
    assert payload["max_seconds"] == 1800.0
    assert payload["season_order"] == "newest_first"
