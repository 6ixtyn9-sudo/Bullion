"""Walk-forward moneyline seeding: leak firewall, provenance and idempotence.

The seeder turns backfilled finals into calibration evidence, so its tests pin
the three properties that make that honest: a seeded probability for day D can
only ever see finals with SAST day strictly before D; a live pre-tip snapshot
can never be displaced by a replayed row; and re-running the seeder on a warm
cache performs zero engine fits and inserts zero rows.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from bullion.models import ActualResult, Forecast, Game, Status
from bullion.store import Store

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def backfill():
    spec = importlib.util.spec_from_file_location(
        "bullion_backfill", ROOT / "scripts" / "backfill.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _final(
    game_id: str,
    day_utc: str,
    home: str,
    away: str,
    home_score: float,
    away_score: float,
    league: str = "cfb",
) -> Game:
    return Game(
        game_id=game_id,
        league=league,
        season="2026",
        home_team_id=home,
        away_team_id=away,
        home_name=f"Home {home}",
        away_name=f"Away {away}",
        scheduled_utc=day_utc,
        status=Status.FINAL,
        actual=ActualResult(home_score=home_score, away_score=away_score),
    )


def _seed_cfb_history(store: Store, *, extra_later_day: bool = False) -> list:
    """15 priors across three days + 3 finals on the scored day 2026-10-04."""
    games = []
    index = 0
    for day, hour in (("2026-10-01", "18"), ("2026-10-02", "18"), ("2026-10-03", "18")):
        for _ in range(5):
            games.append(
                _final(
                    f"g{index:03d}",
                    f"{day}T{hour}:00:00Z",
                    str(100 + index),
                    str(200 + index),
                    home_score=24 + (index % 4),
                    away_score=17 + (index % 3),
                )
            )
            index += 1
    for _ in range(3):
        games.append(
            _final(
                f"g{index:03d}",
                "2026-10-04T18:00:00Z",
                str(100 + index),
                str(200 + index),
                home_score=31,
                away_score=20,
            )
        )
        index += 1
    if extra_later_day:
        # Finals settled AFTER the scored day: the leak firewall must make
        # these irrelevant to any 2026-10-04 probability.
        for _ in range(3):
            games.append(
                _final(
                    f"g{index:03d}",
                    "2026-10-05T18:00:00Z",
                    str(100 + index),
                    str(200 + index),
                    home_score=10,
                    away_score=45,
                )
            )
            index += 1
    for game in games:
        store.upsert_game(game)
    return games


def _seeded_rows(store: Store):
    return store.conn.execute(
        "SELECT * FROM forecast_snapshots WHERE provenance='walk_forward_seed'"
        " ORDER BY game_id, selection"
    ).fetchall()


def test_seed_produces_provenanced_moneyline_rows_and_is_idempotent(backfill, tmp_path):
    with Store(tmp_path / "s.sqlite") as store:
        _seed_cfb_history(store)
        summary = backfill.seed_walk_forward_ml(store, leagues=("cfb",))

        lane = summary["lanes"]["cfb"]
        assert lane["matchdays_scored"] == 1, "only 2026-10-04 has enough priors"
        assert lane["inserted"] == 6, "3 games x 2 moneyline sides"

        rows = [dict(row) for row in _seeded_rows(store)]
        assert len(rows) == 6
        assert {row["market"] for row in rows} == {"moneyline"}
        assert {row["source"] for row in rows} == {"empirical_score_v2"}
        assert {row["observed_day"] for row in rows} == {"2026-10-04"}
        for row in rows:
            assert row["provenance"] == "walk_forward_seed"
            assert 0.0 < row["raw_probability"] < 1.0

        # Warm re-run: every final already has a moneyline snapshot, so the
        # per-game_id existence check scores zero matchdays and inserts nothing.
        warm = backfill.seed_walk_forward_ml(store, leagues=("cfb",))
        assert warm["lanes"]["cfb"]["inserted"] == 0
        assert warm["lanes"]["cfb"]["matchdays_scored"] == 0
        assert store.counts()["forecast_snapshots"] == 6


def test_seed_never_displaces_a_live_pre_tip_snapshot(backfill, tmp_path):
    with Store(tmp_path / "s.sqlite") as store:
        games = _seed_cfb_history(store)
        live_game = games[-1]  # one 2026-10-04 final
        # A live tick recorded this game's moneyline pre-tip, before the seed
        # pass runs (the daily.py / backfill.py ordering).
        store.record_forecasts(
            [
                Forecast(
                    game_id=live_game.game_id,
                    league="cfb",
                    scheduled_utc=live_game.scheduled_utc,
                    home_name=live_game.home_name,
                    away_name=live_game.away_name,
                    market="moneyline",
                    selection="home",
                    raw_probability=0.9,
                    source="empirical_score_v2",
                )
            ],
            observed_day="2026-10-04",
        )
        summary = backfill.seed_walk_forward_ml(store, leagues=("cfb",))
        assert summary["lanes"]["cfb"]["inserted"] == 4, "the live game's two sides are skipped"

        rows = {
            (row["game_id"], row["selection"]): dict(row)
            for row in store.conn.execute("SELECT * FROM forecast_snapshots")
        }
        live_row = rows[(live_game.game_id, "home")]
        assert live_row["provenance"] == "live"
        assert live_row["raw_probability"] == 0.9, "INSERT OR IGNORE kept the live row"
        assert (live_game.game_id, "away") not in rows, (
            "the seeder excludes a whole game once any moneyline snapshot exists "
            "for it — no half-seeded fixture"
        )
        seeded = [row for row in rows.values() if row["provenance"] == "walk_forward_seed"]
        assert len(seeded) == 4


def test_seed_is_isolated_from_finals_settled_later(backfill, tmp_path):
    """Leak pin: adding later finals must not move any earlier probability."""
    with Store(tmp_path / "s.sqlite") as store:
        _seed_cfb_history(store)
        backfill.seed_walk_forward_ml(store, leagues=("cfb",))
        before = {
            (row["game_id"], row["selection"]): row["raw_probability"]
            for row in _seeded_rows(store)
        }

        # A later cache loss (or review re-run) re-seeds after the 10-05 finals
        # landed. Day 2026-10-04's fit set is finals strictly before it, so the
        # probabilities must be bit-identical.
        late = [
            _final(
                f"late{i}",
                "2026-10-05T18:00:00Z",
                str(300 + i),
                str(400 + i),
                home_score=10,
                away_score=45,
            )
            for i in range(3)
        ]
        for game in late:
            store.upsert_game(game)
        store.conn.execute(
            "DELETE FROM forecast_snapshots WHERE observed_day='2026-10-04'"
            " AND provenance='walk_forward_seed'"
        )
        store.conn.commit()
        summary = backfill.seed_walk_forward_ml(store, leagues=("cfb",))
        # The re-seed scores BOTH unfinished days: 2026-10-04 (its rows were
        # deleted) and 2026-10-05 (those finals did not exist at first run).
        assert summary["lanes"]["cfb"]["inserted"] == 12

        before_d4 = {
            key: value for key, value in before.items() if key[0].startswith("g0")
        }
        after_d4 = {
            (row["game_id"], row["selection"]): row["raw_probability"]
            for row in _seeded_rows(store)
            if row["observed_day"] == "2026-10-04"
        }
        assert set(after_d4) == set(before_d4)
        for key, value in before_d4.items():
            assert after_d4[key] == value, f"{key} moved: later finals leaked into day D"


def test_seed_is_deterministic_across_caches(backfill, tmp_path):
    probabilities = []
    for name in ("a.sqlite", "b.sqlite"):
        with Store(tmp_path / name) as store:
            _seed_cfb_history(store)
            backfill.seed_walk_forward_ml(store, leagues=("cfb",))
            probabilities.append(
                sorted(
                    (row["game_id"], row["selection"], round(row["raw_probability"], 10))
                    for row in _seeded_rows(store)
                )
            )
    assert probabilities[0] == probabilities[1]


def test_seed_covers_nhl_uniformly_and_soccer_three_way(backfill, tmp_path):
    with Store(tmp_path / "s.sqlite") as store:
        index = 0
        # Four history days x 4 games per league = 16 priors per lane, then one
        # scored day with a single game per lane.
        for day in ("2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04"):
            for _ in range(4):
                for league in ("nhl", "eng.1"):
                    store.upsert_game(
                        _final(
                            f"n{index:03d}",
                            f"{day}T19:00:00Z",
                            str(10 + index),
                            str(20 + index),
                            home_score=3,
                            away_score=2,
                            league=league,
                        )
                    )
                    index += 1
        for league, suffix in (("nhl", "h"), ("eng.1", "e")):
            store.upsert_game(
                _final(
                    f"scored-{suffix}",
                    "2026-10-05T19:00:00Z",
                    "77",
                    "88",
                    home_score=3,
                    away_score=2,
                    league=league,
                )
            )
        summary = backfill.seed_walk_forward_ml(store, leagues=("nhl", "eng.1"))
        nhl_rows = {
            (row["game_id"], row["selection"]): dict(row)
            for row in _seeded_rows(store)
            if row["league"] == "nhl"
        }
        soccer_rows = {
            (row["game_id"], row["selection"]): dict(row)
            for row in _seeded_rows(store)
            if row["league"] == "eng.1"
        }
        assert nhl_rows, "NHL is seeded like every other lane — the gate, not the seeder, rejects it"
        assert {row["source"] for row in nhl_rows.values()} == {"nhl_poisson_v2"}
        assert {row["selection"] for row in nhl_rows.values()} == {"home", "away"}
        assert soccer_rows, "soccer rows exist"
        assert {row["source"] for row in soccer_rows.values()} == {"dixon_coles_v2"}
        assert {row["selection"] for row in soccer_rows.values()} == {"home", "draw", "away"}
        assert summary["lanes"]["nhl"]["inserted"] == 2
        assert summary["lanes"]["eng.1"]["inserted"] == 3


def test_seed_skips_matchdays_without_enough_history(backfill, tmp_path):
    with Store(tmp_path / "s.sqlite") as store:
        for index in range(10):  # 10 priors + 2 finals: below MIN_SEED_HISTORY
            store.upsert_game(
                _final(
                    f"t{index:03d}",
                    "2026-10-01T18:00:00Z" if index < 10 - 2 else "2026-10-02T18:00:00Z",
                    str(100 + index),
                    str(200 + index),
                    home_score=21,
                    away_score=14,
                )
            )
        summary = backfill.seed_walk_forward_ml(store, leagues=("cfb",))
        assert summary["lanes"]["cfb"]["matchdays_scored"] == 0
        assert summary["lanes"]["cfb"]["inserted"] == 0
        # Both matchdays (8 and 10 priors) sit below the 15-final floor.
        assert summary["lanes"]["cfb"]["matchdays_below_min_history"] == 2
        assert store.counts()["forecast_snapshots"] == 0


def test_source_bump_reseeds_instead_of_silently_seeding_zero(backfill, tmp_path):
    """Regression for the per-source idempotence check (audit bug 5).

    A SOURCE version bump must seed its own rows for history the old source
    already covered; the old rows stay as evidence under their old segment.
    """
    with Store(tmp_path / "s.sqlite") as store:
        _seed_cfb_history(store)
        first = backfill.seed_walk_forward_ml(store, leagues=("cfb",))
        assert first["lanes"]["cfb"]["inserted"] == 6

        # Warm re-run with the SAME source: pure no-op.
        warm = backfill.seed_walk_forward_ml(store, leagues=("cfb",))
        assert warm["lanes"]["cfb"]["inserted"] == 0, (
            "same source + same history must be a no-op"
        )
        # Now move the existing rows to an OLD tag, the way a bumped engine
        # leaves its predecessor's evidence behind: the current source no longer
        # covers this history, so it must re-seed rather than stay silent.
        store.conn.execute(
            "UPDATE forecast_snapshots SET source='empirical_score_v0',"
            " forecast_id='legacy-'||forecast_id WHERE provenance='walk_forward_seed'"
        )
        store.conn.commit()
        bumped = backfill.seed_walk_forward_ml(store, leagues=("cfb",))
        assert bumped["lanes"]["cfb"]["inserted"] == 6, (
            "a SOURCE bump must re-seed history under the new tag"
        )
        counts = store.counts()["forecast_snapshots"]
        assert counts == 12, "old-source evidence survives next to the new tag"
