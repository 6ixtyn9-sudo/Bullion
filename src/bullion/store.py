"""Lightweight SQLite warehouse (stdlib ``sqlite3``, no ORM, no migrations).

Holds official fixtures/finals, named-book snapshots, and ESPN predictor
snapshots. Everything is upserted by natural identity so a re-run of the same
slate cannot duplicate a quote or rotate a stored price.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .models import (
    ActualResult,
    BookQuote,
    Game,
    Status,
    now_utc,
    to_utc_z,
    utc_minus_hours,
)

#: A final is only visible to a walk-forward step once this delay has passed.
#: Prevents a forecast "as of" tip time from seeing a score that leaked in
#: early (e.g. an abandoned/rescheduled event).
RESULT_AVAILABLE_AFTER_HOURS = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    league TEXT NOT NULL,
    season TEXT NOT NULL DEFAULT '',
    home_team_id TEXT NOT NULL,
    away_team_id TEXT NOT NULL,
    home_name TEXT NOT NULL,
    away_name TEXT NOT NULL,
    scheduled_utc TEXT NOT NULL,
    status TEXT NOT NULL,
    neutral_site INTEGER NOT NULL DEFAULT 0,
    venue TEXT NOT NULL DEFAULT '',
    home_score REAL,
    away_score REAL,
    overtime_periods INTEGER NOT NULL DEFAULT 0,
    periods_json TEXT NOT NULL DEFAULT '[]',
    updated_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS games_league_time ON games (league, scheduled_utc);

CREATE TABLE IF NOT EXISTS book_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    league TEXT NOT NULL,
    market TEXT NOT NULL,
    line REAL,
    book TEXT NOT NULL,
    provider TEXT NOT NULL,
    prices_json TEXT NOT NULL,
    observed_utc TEXT NOT NULL,
    pre_match INTEGER NOT NULL,
    main_line INTEGER NOT NULL DEFAULT 1,
    provenance TEXT NOT NULL DEFAULT 'named_book',
    UNIQUE (game_id, book, market, line, observed_utc)
);
CREATE INDEX IF NOT EXISTS book_game ON book_snapshots (game_id, market);

CREATE TABLE IF NOT EXISTS predictor_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    league TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    observed_utc TEXT NOT NULL,
    prob_home REAL NOT NULL,
    expected_home REAL,
    expected_away REAL,
    predicted_margin REAL,
    UNIQUE (game_id, provider, observed_utc)
);
CREATE INDEX IF NOT EXISTS predictor_game ON predictor_snapshots (game_id);

CREATE TABLE IF NOT EXISTS forecast_snapshots (
    forecast_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    league TEXT NOT NULL,
    market TEXT NOT NULL,
    period TEXT NOT NULL,
    source TEXT NOT NULL,
    selection TEXT NOT NULL,
    line REAL,
    raw_probability REAL NOT NULL,
    push_probability REAL NOT NULL DEFAULT 0,
    observed_day TEXT NOT NULL,
    created_utc TEXT NOT NULL,
    provenance TEXT NOT NULL DEFAULT 'live'
);
CREATE INDEX IF NOT EXISTS forecast_league ON forecast_snapshots (league, observed_day);

-- Which (lane, ET day) scoreboard sweeps are complete. This lives HERE, not in
-- a committed JSON manifest, because it is resume state for THIS database: the
-- warehouse is gitignored and restored from an Actions cache, so a manifest in
-- Git would outlive the DB it describes. A cache eviction would then leave a
-- fresh empty DB next to a manifest claiming 400 days were already captured,
-- and the backfill would skip all of them forever while the lane sat at 0
-- finals. Keeping the two in one file makes them fail together, recoverably.
CREATE TABLE IF NOT EXISTS scoreboard_backfill_days (
    league TEXT NOT NULL,
    et_date TEXT NOT NULL,
    finals INTEGER NOT NULL DEFAULT 0,
    captured_utc TEXT NOT NULL,
    PRIMARY KEY (league, et_date)
);
"""


class Store:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """Add columns introduced after a cached database was first created.

        ``CREATE TABLE IF NOT EXISTS`` never alters an existing table, and the
        workflow restores ``bullion.db`` from a cache that predates any new
        column. Each entry is idempotent and additive only.
        """
        additions = {
            "predictor_snapshots": {
                "predicted_margin": "REAL",
            },
            "forecast_snapshots": {
                # Provenance of a calibration observation: 'live' (recorded by a
                # scheduled pre-tip tick or restored from a receipt) vs
                # 'walk_forward_seed' (replayed from backfilled finals by
                # scripts/backfill.py). Additive; the calibration reader
                # ignores it, the audit reporting groups by it.
                "provenance": "TEXT NOT NULL DEFAULT 'live'",
            },
        }
        for table, columns in additions.items():
            existing = {
                row["name"]
                for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for column, sql_type in columns.items():
                if column in existing:
                    continue
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- scoreboard backfill resume state ----------------------------------
    def record_scoreboard_day(self, league: str, et_date: str, finals: int) -> None:
        """Mark one ``(lane, ET day)`` scoreboard sweep complete.

        Captured ET days are immutable (a final does not un-settle), so this is
        only ever written for a day strictly before today; today's slate stays
        refetchable because it is still tipping.
        """
        self.conn.execute(
            """
            INSERT INTO scoreboard_backfill_days (league, et_date, finals, captured_utc)
            VALUES (?,?,?,?)
            ON CONFLICT(league, et_date) DO UPDATE SET
                finals=excluded.finals,
                captured_utc=excluded.captured_utc
            """,
            (league, et_date, int(finals), now_utc()),
        )
        self.conn.commit()

    def scoreboard_day_done(self, league: str, et_date: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM scoreboard_backfill_days WHERE league=? AND et_date=?",
            (league, et_date),
        ).fetchone()
        return row is not None

    def scoreboard_days_captured(self, league: str, *, before: str) -> int:
        """Completed ET days strictly before ``before`` (``YYYYMMDD``)."""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM scoreboard_backfill_days WHERE league=? AND et_date<?",
            (league, before),
        ).fetchone()
        return int(row[0]) if row else 0

    def scoreboard_finals_captured(self, league: str) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(finals), 0) FROM scoreboard_backfill_days WHERE league=?",
            (league,),
        ).fetchone()
        return int(row[0]) if row else 0

    # -- games -------------------------------------------------------------
    def upsert_game(self, game: Game) -> None:
        actual = game.actual
        self.conn.execute(
            """
            INSERT INTO games (game_id, league, season, home_team_id, away_team_id,
                home_name, away_name, scheduled_utc, status, neutral_site, venue,
                home_score, away_score, overtime_periods, periods_json, updated_utc)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(game_id) DO UPDATE SET
                league=excluded.league,
                season=excluded.season,
                home_team_id=excluded.home_team_id,
                away_team_id=excluded.away_team_id,
                home_name=excluded.home_name,
                away_name=excluded.away_name,
                scheduled_utc=excluded.scheduled_utc,
                status=excluded.status,
                neutral_site=excluded.neutral_site,
                venue=excluded.venue,
                home_score=COALESCE(excluded.home_score, games.home_score),
                away_score=COALESCE(excluded.away_score, games.away_score),
                overtime_periods=excluded.overtime_periods,
                periods_json=excluded.periods_json,
                updated_utc=excluded.updated_utc
            """,
            (
                game.game_id,
                game.league,
                game.season,
                game.home_team_id,
                game.away_team_id,
                game.home_name,
                game.away_name,
                to_utc_z(game.scheduled_utc),
                game.status.value,
                int(game.neutral_site),
                game.venue,
                actual.home_score if actual else None,
                actual.away_score if actual else None,
                actual.overtime_periods if actual else 0,
                json.dumps(list(actual.periods)) if actual else "[]",
                now_utc(),
            ),
        )
        self.conn.commit()

    def get_game(self, game_id: str) -> Optional[Game]:
        row = self.conn.execute("SELECT * FROM games WHERE game_id=?", (game_id,)).fetchone()
        return _game_from_row(row) if row else None

    def games_for_league(self, league: str, day: Optional[str] = None) -> List[Game]:
        """Games whose SAST day matches ``day`` (or all games for the league)."""
        rows = self.conn.execute(
            "SELECT * FROM games WHERE league=? ORDER BY scheduled_utc", (league,)
        ).fetchall()
        games = [_game_from_row(row) for row in rows]
        if day is None:
            return games
        from .models import sast_day

        return [game for game in games if sast_day(game.scheduled_utc) == day]

    def settled_results(self, league: str, as_of_utc: str) -> List[Game]:
        """Finals visible at ``as_of`` under the result-availability delay."""
        cutoff = utc_minus_hours(as_of_utc, RESULT_AVAILABLE_AFTER_HOURS)
        rows = self.conn.execute(
            """
            SELECT * FROM games
            WHERE league=? AND status='final' AND home_score IS NOT NULL
              AND scheduled_utc <= ?
            ORDER BY scheduled_utc
            """,
            (league, cutoff),
        ).fetchall()
        return [_game_from_row(row) for row in rows]

    # -- quotes ------------------------------------------------------------
    def record_quotes(self, quotes: Iterable[BookQuote], league: str) -> int:
        written = 0
        for quote in quotes:
            cursor = self.conn.execute(
                """
                INSERT OR IGNORE INTO book_snapshots
                    (game_id, league, market, line, book, provider, prices_json,
                     observed_utc, pre_match, main_line, provenance)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    quote.game_id,
                    league,
                    quote.market,
                    quote.line,
                    quote.book,
                    quote.provider,
                    json.dumps({k: v for k, v in sorted(quote.side_prices.items())}),
                    quote.observed_utc,
                    int(quote.pre_match),
                    int(quote.main_line),
                    quote.provenance,
                ),
            )
            written += cursor.rowcount
        self.conn.commit()
        return written

    def quotes_for_game(self, game_id: str) -> List[BookQuote]:
        rows = self.conn.execute(
            "SELECT * FROM book_snapshots WHERE game_id=? ORDER BY observed_utc",
            (game_id,),
        ).fetchall()
        return [_quote_from_row(row) for row in rows]

    def latest_pre_tip_main_lines(self, league: str) -> Dict[str, Dict[str, float]]:
        """``game_id -> {"spread": line, "total": line}`` from the newest pre-tip
        main-line snapshot per (game, market).

        ONE bulk query per league: ``_results`` runs on thousands of settled
        games, so this must never become a per-game quote query. Ties on
        ``observed_utc`` resolve to the alphabetically first book, which is
        deterministic.
        """
        rows = self.conn.execute(
            """
            SELECT bs.game_id AS game_id, bs.market AS market, bs.line AS line, bs.book AS book
            FROM book_snapshots bs
            JOIN (
                SELECT game_id, market, MAX(observed_utc) AS observed_utc
                FROM book_snapshots
                WHERE league = ? AND main_line = 1 AND pre_match = 1
                  AND market IN ('spread', 'total') AND line IS NOT NULL
                GROUP BY game_id, market
            ) latest
              ON latest.game_id = bs.game_id AND latest.market = bs.market
             AND latest.observed_utc = bs.observed_utc
            WHERE bs.league = ? AND bs.main_line = 1 AND bs.pre_match = 1
            ORDER BY bs.game_id, bs.market, bs.book
            """,
            (league, league),
        ).fetchall()
        lines: Dict[str, Dict[str, float]] = {}
        for row in rows:  # first book per (game, market) in book-ascending order wins
            slot = lines.setdefault(row["game_id"], {})
            slot.setdefault(row["market"], float(row["line"]))
        return lines

    # -- predictor ---------------------------------------------------------
    def record_predictor(
        self,
        *,
        game_id: str,
        league: str,
        provider: str,
        model: str,
        observed_utc: str,
        prob_home: float,
        expected_home: Optional[float],
        expected_away: Optional[float],
        predicted_margin: Optional[float] = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO predictor_snapshots
                (game_id, league, provider, model, observed_utc, prob_home,
                 expected_home, expected_away, predicted_margin)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                game_id,
                league,
                provider,
                model,
                observed_utc,
                float(prob_home),
                expected_home,
                expected_away,
                predicted_margin,
            ),
        )
        self.conn.commit()

    # -- forecasts (walk-forward calibration evidence) ---------------------
    def record_forecasts(self, forecasts: Iterable[Any], observed_day: str) -> int:
        """Persist pre-tip model probabilities for later prequential scoring."""
        written = 0
        for forecast in forecasts:
            cursor = self.conn.execute(
                """
                INSERT OR IGNORE INTO forecast_snapshots
                    (forecast_id, game_id, league, market, period, source, selection,
                     line, raw_probability, push_probability, observed_day, created_utc,
                     provenance)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    forecast.forecast_id,
                    forecast.game_id,
                    forecast.league,
                    forecast.market,
                    forecast.period,
                    forecast.source,
                    forecast.selection,
                    forecast.line,
                    float(forecast.raw_probability),
                    float(forecast.push_probability),
                    observed_day,
                    now_utc(),
                    "live",
                ),
            )
            written += cursor.rowcount
        self.conn.commit()
        return written

    def restore_forecasts(
        self, rows: Iterable[Dict[str, Any]], *, provenance: str = "live"
    ) -> int:
        """Re-insert frozen pre-tip forecasts from committed receipts.

        ``forecast_snapshots`` lives only in the gitignored SQLite cache, so one
        cache eviction reset every lane's walk-forward history to zero. The
        receipts in ``localdata/`` are the durable copy: they are write-once and
        carry the frozen probability and day. The id is *recomputed* by the
        caller (:func:`bullion.models.forecast_id_for`) rather than copied,
        because a receipt's id was frozen after a book was attached and the
        warehouse records the forecast before; copying it inserted a second row
        for an observation the cache already held. ``INSERT OR IGNORE`` keeps
        this idempotent next to live recording. ``provenance`` is ``'live'``
        for receipt restores; the walk-forward seeder passes
        ``'walk_forward_seed'`` so replayed history stays auditable.
        """
        restored = 0
        for row in rows:
            cursor = self.conn.execute(
                """
                INSERT OR IGNORE INTO forecast_snapshots
                    (forecast_id, game_id, league, market, period, source, selection,
                     line, raw_probability, push_probability, observed_day, created_utc,
                     provenance)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    row["forecast_id"],
                    row["game_id"],
                    row["league"],
                    row["market"],
                    row["period"],
                    row["source"],
                    row["selection"],
                    row.get("line"),
                    float(row["raw_probability"]),
                    float(row.get("push_probability") or 0.0),
                    row["observed_day"],
                    row["created_utc"],
                    provenance,
                ),
            )
            restored += cursor.rowcount
        self.conn.commit()
        return restored

    def calibration_observations(
        self, league: str, as_of_utc: str
    ) -> List[Dict[str, Any]]:
        """Settled forecast rows visible at ``as_of`` (result delay respected)."""
        cutoff = utc_minus_hours(as_of_utc, RESULT_AVAILABLE_AFTER_HOURS)
        rows = self.conn.execute(
            """
            SELECT f.*, g.scheduled_utc, g.home_score, g.away_score, g.status,
                   g.overtime_periods, g.periods_json
            FROM forecast_snapshots f JOIN games g ON g.game_id = f.game_id
            WHERE f.league = ? AND g.status = 'final' AND g.home_score IS NOT NULL
              AND g.scheduled_utc <= ?
            """,
            (league, cutoff),
        ).fetchall()
        return [dict(row) for row in rows]

    def counts(self) -> Dict[str, int]:
        return {
            table: int(self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in (
                "games",
                "book_snapshots",
                "predictor_snapshots",
                "forecast_snapshots",
            )
        }


def _game_from_row(row: sqlite3.Row) -> Game:
    actual = None
    if row["home_score"] is not None and row["away_score"] is not None:
        periods = tuple(tuple(pair) for pair in json.loads(row["periods_json"] or "[]"))
        actual = ActualResult(
            home_score=float(row["home_score"]),
            away_score=float(row["away_score"]),
            periods=periods,
            overtime_periods=int(row["overtime_periods"]),
        )
    return Game(
        game_id=row["game_id"],
        league=row["league"],
        season=row["season"],
        home_team_id=row["home_team_id"],
        away_team_id=row["away_team_id"],
        home_name=row["home_name"],
        away_name=row["away_name"],
        scheduled_utc=row["scheduled_utc"],
        status=Status(row["status"]),
        neutral_site=bool(row["neutral_site"]),
        venue=row["venue"],
        actual=actual,
    )


def _quote_from_row(row: sqlite3.Row) -> BookQuote:
    return BookQuote(
        game_id=row["game_id"],
        book=row["book"],
        market=row["market"],
        observed_utc=row["observed_utc"],
        side_prices={k: float(v) for k, v in json.loads(row["prices_json"]).items()},
        line=row["line"],
        provider=row["provider"],
        pre_match=bool(row["pre_match"]),
        main_line=bool(row["main_line"]),
        provenance=row["provenance"],
    )
