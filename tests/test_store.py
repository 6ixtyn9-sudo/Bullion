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


# ---------------------------------------------------------------------------
# Superseded basketball seed rows (the pre-F4 arithmetic cleanup)
# ---------------------------------------------------------------------------


def _seed_row(forecast_id: str, game_id: str, league: str, source: str) -> dict:
    return {
        "forecast_id": forecast_id,
        "game_id": game_id,
        "league": league,
        "market": "moneyline",
        "period": "full_game",
        "source": source,
        "selection": "home",
        "line": None,
        "raw_probability": 0.62,
        "push_probability": 0.0,
        "observed_day": DAY,
        "created_utc": f"{DAY}T15:00:00Z",
    }


def _forecast_ids(store: Store) -> set[str]:
    return {
        row["forecast_id"]
        for row in store.conn.execute("SELECT forecast_id FROM forecast_snapshots")
    }


def test_pre_f4_basketball_seed_rows_are_retired_once_and_live_rows_survive(tmp_path):
    """Both kept generations go, one open only, and nothing else is touched."""
    path = tmp_path / "db.sqlite"
    with Store(path) as store:
        # Pre-F4 arithmetic, seeded under the plain tag.
        store.restore_forecasts(
            [_seed_row("fid-old", "g-old", "nba", "empirical_score")],
            provenance="walk_forward_seed",
        )
        # The briefly-suffixed generation: the marker the guard looks for.
        store.restore_forecasts(
            [_seed_row("fid-v2", "g-v2", "nba", "empirical_score_v2")],
            provenance="walk_forward_seed",
        )
        # A live pre-tip snapshot in the same lane, on the same tag.
        store.restore_forecasts(
            [_seed_row("fid-live", "g-live", "nba", "empirical_score")], provenance="live"
        )
        # Another sport's seed rows are not this migration's business.
        store.restore_forecasts(
            [_seed_row("fid-cfb", "g-cfb", "cfb", "empirical_score")],
            provenance="walk_forward_seed",
        )
        assert store.counts()["forecast_snapshots"] == 4

    with Store(path) as store:  # opening the warehouse is what runs the migration
        assert _forecast_ids(store) == {"fid-live", "fid-cfb"}
        # The wager identity is free again, so the ~7s replay can actually land;
        # a relabel alone would have collided on the primary key here.
        assert (
            store.restore_forecasts(
                [_seed_row("fid-old", "g-old", "nba", "empirical_score")],
                provenance="walk_forward_seed",
            )
            == 1
        )
        # No suffixed seed row is left, so the guard can never re-fire.
        assert store._drop_superseded_seed_rows() == 0
        assert _forecast_ids(store) == {"fid-live", "fid-cfb", "fid-old"}

    with Store(path) as store:  # and a second open is a plain no-op
        assert store.counts()["forecast_snapshots"] == 3


def test_the_cleanup_needs_the_marker_and_never_guesses_from_age(tmp_path):
    """A warehouse that never saw the suffixed tag keeps its history."""
    path = tmp_path / "db.sqlite"
    with Store(path) as store:
        store.restore_forecasts(
            [_seed_row("fid-keep", "g-keep", "ncaa_m", "empirical_score")],
            provenance="walk_forward_seed",
        )
    with Store(path) as store:
        assert _forecast_ids(store) == {"fid-keep"}
        assert store._drop_superseded_seed_rows() == 0
        assert store.counts()["forecast_snapshots"] == 1
