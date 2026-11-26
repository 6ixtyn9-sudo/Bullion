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
    created_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS forecast_league ON forecast_snapshots (league, observed_day);
"""


class Store:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

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
    ) -> None:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO predictor_snapshots
                (game_id, league, provider, model, observed_utc, prob_home,
                 expected_home, expected_away)
            VALUES (?,?,?,?,?,?,?,?)
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
                     line, raw_probability, push_probability, observed_day, created_utc)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
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
                ),
            )
            written += cursor.rowcount
        self.conn.commit()
        return written

    def calibration_observations(
        self, league: str, as_of_utc: str
    ) -> List[Dict[str, Any]]:
        """Settled forecast rows visible at ``as_of`` (result delay respected)."""
        cutoff = utc_minus_hours(as_of_utc, RESULT_AVAILABLE_AFTER_HOURS)
        rows = self.conn.execute(
            """
            SELECT f.*, g.scheduled_utc, g.home_score, g.away_score, g.status
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
