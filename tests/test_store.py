"""Warehouse schema durability: cached-database migration and forecast restore."""

from __future__ import annotations

import sqlite3

from bullion.models import ActualResult, Game, Status
from bullion.store import Store

DAY = "2026-11-28"


def _legacy_predictor_table(path) -> None:
    """A database created before ``predicted_margin`` existed."""
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE predictor_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            league TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            observed_utc TEXT NOT NULL,
            prob_home REAL NOT NULL,
            expected_home REAL,
            expected_away REAL,
            UNIQUE (game_id, provider, observed_utc)
        )
        """
    )
    conn.commit()
    conn.close()


def test_a_cached_database_gains_new_predictor_columns_on_open(tmp_path):
    path = tmp_path / "cached.sqlite"
    _legacy_predictor_table(path)

    with Store(path) as store:
        columns = {
            row["name"]
            for row in store.conn.execute("PRAGMA table_info(predictor_snapshots)").fetchall()
        }
        assert "predicted_margin" in columns
        store.record_predictor(
            game_id="401",
            league="nfl",
            provider="espn",
            model="espn_fpi",
            observed_utc="2026-11-28T12:00:00Z",
            prob_home=0.62,
            expected_home=None,
            expected_away=None,
            predicted_margin=-3.5,
        )
        stored = store.conn.execute(
            "SELECT predicted_margin FROM predictor_snapshots"
        ).fetchone()
        assert stored["predicted_margin"] == -3.5

    # Re-opening an already-migrated warehouse must be a no-op, not an error.
    with Store(path) as store:
        assert store.counts()["predictor_snapshots"] == 1


def _final_game(game_id: str = "401") -> Game:
    return Game(
        game_id=game_id,
        league="nhl",
        season="2026",
        home_team_id="10",
        away_team_id="20",
        home_name="Home Club",
        away_name="Away Club",
        scheduled_utc=f"{DAY}T18:00:00Z",
        status=Status.FINAL,
        actual=ActualResult(home_score=3, away_score=1),
    )


def _frozen_row(game_id: str = "401") -> dict:
    return {
        "forecast_id": "fid-restored",
        "game_id": game_id,
        "league": "nhl",
        "market": "moneyline",
        "period": "full_game",
        "source": "nhl_poisson",
        "selection": "home",
        "line": None,
        "raw_probability": 0.62,
        "push_probability": 0.0,
        "observed_day": DAY,
        "created_utc": f"{DAY}T15:00:00Z",
    }


def test_restoring_frozen_forecasts_is_idempotent_and_feeds_calibration(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        store.upsert_game(_final_game())
        row = _frozen_row()

        assert store.restore_forecasts([row]) == 1
        assert store.restore_forecasts([row]) == 0, "re-running a restore must not duplicate"
        assert store.counts()["forecast_snapshots"] == 1

        observations = store.calibration_observations("nhl", "2026-11-30T00:00:00Z")
        assert len(observations) == 1
        assert observations[0]["raw_probability"] == 0.62
        assert observations[0]["observed_day"] == DAY
