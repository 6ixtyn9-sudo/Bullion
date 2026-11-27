#!/usr/bin/env python3
"""ESPN historical backfill for settled finals (2024-25 onward, bounded).

What this does: stores official finals and their native period linescores so
the NHL/Soccer/Football rating fits start warm on day one.

What this deliberately does NOT do: fabricate pre-tip quotes for old games.
Historical scoreboards are served today, so their prices are closing/after-the-
fact numbers. Using them as ticket evidence would be a look-ahead fantasy, and
using them to "warm calibration" would smuggle post-game information into a
signal. Spread, total, and provider:espn_predictor calibration therefore starts
cold and must earn 30 settled observations per segment exactly like the live
lane does — they need a book quote or a provider snapshot no honest history
can supply.

The one deliberate exception is the model's own moneyline surface:
``seed_walk_forward_ml`` replays ``empirical_score_v2`` / ``nhl_poisson_v2`` /
``dixon_coles_v2`` over the backfilled finals, fitting strictly on finals whose
SAST day precedes the forecast day, and persists those out-of-sample
probabilities as ``provenance='walk_forward_seed'`` rows. They carry no price,
no book, and no provider information — and every honesty gate
(``DEFAULT_MIN_SAMPLES``, the Brier caps, ``cap_vs_base_rate``, the ROI band)
re-measures over them exactly as over live rows and fails closed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bullion.espn import (  # noqa: E402
    EspnClient,
    EspnFetchError,
    event_id_from_ref,
    game_from_summary,
    parse_summary,
    season_events_url,
)
from bullion.engines import (  # noqa: E402
    DEFAULT_PRIOR_WEIGHT,
    Result,
    fit_ratings,
    normal_cdf,
    rates_for,
    rest_adjustment,
)
from bullion.engines import nhl as nhl_engine  # noqa: E402
from bullion.engines import soccer as soccer_engine  # noqa: E402
from bullion.engines import football as football_engine  # noqa: E402
from bullion.models import LEAGUES, Sport, forecast_id_for, now_utc, sast_day  # noqa: E402
from bullion.store import Store  # noqa: E402

REINDEX_PAGES = 25
#: Committed receipts are the durable copy of every frozen pre-tip forecast.
RECEIPT_GLOBS = ("auto_tickets_*_decision.json", "auto_tickets_*_no_bet.json")
#: A 45-minute job that spends its last seconds backfilling loses the cache save
#: and the receipt commit. Warn well before a budget can crowd those steps out.
JOB_TIMEOUT_WARNING_SECONDS = 2400.0


def select_leagues(selector: str):
    selector = (selector or "all").strip().lower()
    if selector == "all":
        return list(LEAGUES.values())
    if selector == "soccer":
        return [league for league in LEAGUES.values() if league.sport is Sport.SOCCER]
    if selector not in LEAGUES:
        raise ValueError(f"unknown league {selector!r}")
    return [LEAGUES[selector]]


def season_event_ids(client: EspnClient, league, season: int) -> List[str]:
    ids: List[str] = []
    for page in range(1, REINDEX_PAGES + 1):
        payload = client.get(season_events_url(league, season, page))
        for item in payload.get("items") or []:
            # LIVE_VERIFIED 2026-11-26: refs arrive as
            # ``.../events/401856879?lang=en&region=us`` — the query suffix must
            # be stripped, otherwise every ref fails ``isdigit()`` and the
            # backfill silently stores nothing.
            event_id = event_id_from_ref(item.get("$ref"))
            if event_id and event_id not in ids:
                ids.append(event_id)
        page_count = int(payload.get("pageCount") or 1)
        if page >= page_count:
            break
    return ids


def _fetch_final(
    client: EspnClient, league, event_id: str
) -> Tuple[str, Optional[Any], Optional[str]]:
    """Fetch and parse one final. Worker-thread safe: no store access here."""
    try:
        payload = client.summary(league, event_id)
        game = game_from_summary(payload, league, event_id)
        final_game, _gaps = parse_summary(payload, game, league)
        return event_id, final_game, None
    except (EspnFetchError, ValueError) as exc:
        # A scheduled future event, a non-post summary, or a dead request is a
        # skipped row, never a reason to abandon the other 6,000.
        return event_id, None, str(exc)


def _handle(
    result: Tuple[str, Optional[Any], Optional[str]],
    store: Store,
    counters: Dict[str, int],
    sample_errors: List[Dict[str, str]],
) -> None:
    event_id, final_game, error = result
    if final_game is None:
        counters["failed"] += 1
        if len(sample_errors) < 3:
            sample_errors.append({"event_id": event_id, "error": str(error)})
        return
    store.upsert_game(final_game)
    counters["stored"] += 1


def _sequential(
    client: EspnClient,
    league,
    pending: List[str],
    store: Store,
    counters: Dict[str, int],
    sample_errors: List[Dict[str, str]],
    delay_seconds: float,
    deadline: Optional[float],
) -> bool:
    """Fetch one at a time (keeps ``--delay`` meaningful). Returns stopped_early."""
    for event_id in pending:
        if deadline is not None and time.monotonic() >= deadline:
            return True
        _handle(_fetch_final(client, league, event_id), store, counters, sample_errors)
        if delay_seconds:
            time.sleep(delay_seconds)
    return False


def _concurrent(
    client: EspnClient,
    league,
    pending: List[str],
    store: Store,
    counters: Dict[str, int],
    sample_errors: List[Dict[str, str]],
    workers: int,
    deadline: Optional[float],
) -> bool:
    """Fetch with a bounded pool; every SQLite write stays on this thread.

    Returns ``True`` when the wall-clock budget stopped the run early. Outstanding
    requests are cancelled rather than waited on, so a hung request cannot push
    the job past its budget.
    """
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="backfill")
    futures = {}
    submitted_all = True
    try:
        for event_id in pending:
            if deadline is not None and time.monotonic() >= deadline:
                submitted_all = False
                break
            futures[pool.submit(_fetch_final, client, league, event_id)] = event_id
        remaining = set(futures)
        while remaining:
            wait_timeout = 5.0
            if deadline is not None:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                wait_timeout = min(wait_timeout, left)
            done, remaining = wait(
                remaining, timeout=wait_timeout, return_when=FIRST_COMPLETED
            )
            for future in done:
                _handle(future.result(), store, counters, sample_errors)
        stopped_early = bool(remaining) or not submitted_all
    finally:
        for future in futures:
            future.cancel()
        pool.shutdown(wait=not (deadline is not None and time.monotonic() >= deadline))
    return stopped_early


def rehydrate_forecasts(store: Store, localdata: Path) -> Dict[str, int]:
    """Restore ``forecast_snapshots`` from the receipts committed in ``localdata``.

    The warehouse is a gitignored cache; the receipts are not. Without this, a
    single cache miss silently resets every lane's walk-forward calibration to
    ``n = 0`` and the gate fails closed forever instead of reopening. Only the
    frozen fields are restored — no probability is recomputed and no historical
    price is introduced — so the restored rows are exactly the evidence that was
    recorded before each tip.
    """
    counters = {"files": 0, "rows": 0, "skipped": 0}
    if not localdata.exists():
        return counters
    rows = []
    for pattern in RECEIPT_GLOBS:
        for path in sorted(localdata.glob(f"*/{pattern}")):
            try:
                receipt = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                counters["skipped"] += 1
                continue
            if not isinstance(receipt, dict):
                counters["skipped"] += 1
                continue
            day = receipt.get("date") or receipt.get("target_date") or ""
            created = str(receipt.get("frozen_at") or receipt.get("captured_at") or "")
            pool = receipt.get("candidate_pool") or []
            if not day or not pool:
                # A pre-fix receipt that never stored its pool: nothing to restore.
                counters["files"] += 1
                continue
            counters["files"] += 1
            for row in pool:
                if not isinstance(row, dict):
                    counters["skipped"] += 1
                    continue
                source = row.get("source") or row.get("model")
                required = (
                    row.get("forecast_id"),
                    row.get("game_id"),
                    row.get("league"),
                    row.get("market"),
                    row.get("period"),
                    source,
                    row.get("selection"),
                    row.get("raw_probability"),
                )
                if any(value in (None, "") for value in required):
                    counters["skipped"] += 1
                    continue
                try:
                    float(row["raw_probability"])
                except (TypeError, ValueError):
                    counters["skipped"] += 1
                    continue
                # The receipt's id was frozen *after* a book was attached; the
                # warehouse records the same forecast before attachment with an
                # older id that included the book. Recompute the price-free
                # identity so a restore is a true no-op on a warm database.
                identity = forecast_id_for(
                    league=str(row["league"]),
                    game_id=str(row["game_id"]),
                    market=str(row["market"]),
                    period=str(row["period"]),
                    selection=str(row["selection"]),
                    line=row.get("line"),
                    source=str(source),
                )
                rows.append(
                    {
                        "forecast_id": identity,
                        "game_id": str(row["game_id"]),
                        "league": str(row["league"]),
                        "market": str(row["market"]),
                        "period": str(row["period"]),
                        "source": str(source),
                        "selection": str(row["selection"]),
                        "line": row.get("line"),
                        "raw_probability": float(row["raw_probability"]),
                        "push_probability": row.get("push_probability"),
                        "observed_day": str(day),
                        "created_utc": created,
                    }
                )
    counters["rows"] = len(rows)
    counters["restored"] = store.restore_forecasts(rows) if rows else 0
    return counters


# -- walk-forward moneyline seeding -------------------------------------------
# The module docstring above used to end at "calibration must earn 30 settled
# observations per segment exactly like the live lane does". That stays true for
# every market that needs a book quote or a provider snapshot — spread, total,
# and provider:espn_predictor remain 100% live-only, because no honest historical
# price for them exists. The model's own moneyline probabilities are different:
# empirical_score_v1, nhl_poisson_v1 and dixon_coles_v1 need nothing but official
# final scores, so the seeder can replay them leak-proof (fit strictly on finals
# with SAST day < the forecast day) and persist the result as calibration
# evidence marked provenance='walk_forward_seed'. The honesty gates are NOT
# relaxed by a single constant: DEFAULT_MIN_SAMPLES, the Brier caps,
# cap_vs_base_rate and the ROI band all re-measure over the seeded rows exactly
# as they measure over live rows, fail closed, and re-lock a lane whose engine
# loses its edge (the 2026-11-27 study caught nhl_poisson_v1 doing exactly that:
# transient pass on small-n noise, Brier 0.2507 > 0.25 at n=734).

SEED_LANES = ("nhl", "nfl", "cfb", "eng.1", "esp.1", "ger.1", "ita.1", "fra.1")
SOCCER_SEED_LANES = frozenset({"eng.1", "esp.1", "ger.1", "ita.1", "fra.1"})
#: Minimum settled finals a matchday must have behind it before the seeder
#: scores it. Mirrors the study's warmup floor; the production gates stay the
#: ones that actually decide when a lane unlocks.
MIN_SEED_HISTORY = 15


def _score_matchday_moneyline(
    lane: str,
    day: str,
    results: List[Result],
    games: List[Dict[str, Any]],
) -> List[Tuple[str, str, float]]:
    """Production-exact moneyline probabilities for one SAST matchday.

    Mirrors the daily runner's engine calls line for line (same fit, same rate
    lookup, same rest adjustment, same distribution code). No book quotes exist
    for these finals, so only ``market='moneyline'`` rows come out. The caller
    guarantees ``results`` holds only finals with SAST day strictly before
    ``day`` — that is the leak firewall.
    """
    scored: List[Tuple[str, str, float]] = []
    if lane in SOCCER_SEED_LANES:
        ratings = soccer_engine.fit_ratings(
            results, reference_day=day, half_life_days=240.0
        )
        if not ratings.get("n"):
            return scored
        rho = soccer_engine.fit_rho(
            results, ratings, reference_day=day, half_life_days=240.0
        )
        for game in games:
            home_rate, away_rate = rates_for(
                ratings,
                game["home_team_id"],
                game["away_team_id"],
                neutral=bool(game["neutral_site"]),
            )
            matrix = soccer_engine.score_matrix(home_rate, away_rate, rho)
            home, draw, away = soccer_engine._one_x_two(matrix)
            for selection, probability in (("home", home), ("draw", draw), ("away", away)):
                scored.append((game["game_id"], selection, probability))
        return scored

    if lane == "nhl":
        ratings = fit_ratings(
            results, reference_day=day, prior_weight=nhl_engine.PRIOR_WEIGHT
        )
        if not ratings.get("n"):
            return scored
        for game in games:
            home_rate, away_rate = rates_for(
                ratings,
                game["home_team_id"],
                game["away_team_id"],
                neutral=bool(game["neutral_site"]),
            )
            home_rate *= rest_adjustment(results, game["home_team_id"], day)
            away_rate *= rest_adjustment(results, game["away_team_id"], day)
            margin, _total, _draw = nhl_engine._final_distributions(home_rate, away_rate)
            prob_home = sum(p for m, p in margin.items() if m > 0)
            prob_away = sum(p for m, p in margin.items() if m < 0)
            scored.append((game["game_id"], "home", prob_home))
            scored.append((game["game_id"], "away", prob_away))
        return scored

    # nfl / cfb: empirical_score_v2 (the FPI surface needs a live predictor
    # snapshot, which a historical final can never have — it stays live-only).
    ratings = fit_ratings(
        results,
        reference_day=day,
        prior_weight=football_engine.PRIOR_WEIGHTS.get(lane, DEFAULT_PRIOR_WEIGHT),
    )
    if not ratings.get("n"):
        return scored
    from bullion.engines.football import SIGMA_PRIORS, residual_sigmas

    prior_margin, prior_total = SIGMA_PRIORS.get(lane, (14.0, 14.0))
    sigma_margin, _sigma_total = residual_sigmas(results, ratings, reference_day=day)
    sigma_margin = sigma_margin or prior_margin
    for game in games:
        home_rate, away_rate = rates_for(
            ratings,
            game["home_team_id"],
            game["away_team_id"],
            neutral=bool(game["neutral_site"]),
        )
        home_rate *= rest_adjustment(results, game["home_team_id"], day, penalty=0.99)
        away_rate *= rest_adjustment(results, game["away_team_id"], day, penalty=0.99)
        expected_margin = home_rate - away_rate
        prob_home = 1.0 - normal_cdf((0.0 - expected_margin) / max(sigma_margin, 0.5))
        scored.append((game["game_id"], "home", prob_home))
        scored.append((game["game_id"], "away", 1.0 - prob_home))
    return scored


def seed_walk_forward_ml(
    store: Store, *, leagues: Tuple[str, ...] = SEED_LANES
) -> Dict[str, Any]:
    """Replay each lane's moneyline model over its backfilled finals.

    For every historical SAST matchday ``D`` the production engine is fitted on
    finals with SAST day strictly before ``D`` and scores that day's finals;
    rows are persisted through :meth:`Store.restore_forecasts` with
    ``provenance='walk_forward_seed'`` and the same wager-only
    :func:`bullion.models.forecast_id` a live tick uses. ``INSERT OR IGNORE``
    therefore makes the whole pass idempotent and guarantees a live pre-tip
    snapshot (recorded first, in :meth:`daily` and :meth:`rehydrate_forecasts`)
    can never be displaced by a seeded row.

    Idempotence is checked per ``(game_id, source)``, not per day: warm ticks
    run one ``SELECT DISTINCT`` per lane, find every historical matchday the
    current source already covered, and pay for zero fits. A SOURCE version
    bump re-seeds history under the new tag instead of silently seeding zero
    rows. No gate constant is read or modified here —
    the seeder only supplies observations; calibration and the ticket gates
    re-measure them every run and fail closed.
    """
    summary: Dict[str, Any] = {"lanes": {}, "fit_seconds": 0.0}
    started = time.monotonic()
    for lane in leagues:
        lane_stats = {
            "matchdays_scored": 0,
            "rows": 0,
            "inserted": 0,
            "matchdays_below_min_history": 0,
        }
        summary["lanes"][lane] = lane_stats
        if lane not in LEAGUES:
            continue
        finals = store.conn.execute(
            """
            SELECT * FROM games
            WHERE league=? AND status='final' AND home_score IS NOT NULL
            ORDER BY scheduled_utc
            """,
            (lane,),
        ).fetchall()
        if not finals:
            continue
        # The lane's moneyline model tag: matches what the daily engines emit.
        source = (
            soccer_engine.SOURCE
            if lane in SOCCER_SEED_LANES
            else nhl_engine.SOURCE
            if lane == "nhl"
            else football_engine.EMPIRICAL_SOURCE
        )
        # Idempotence is per (game_id, source): a SOURCE version bump must be
        # able to seed its own rows for history the old source already covered,
        # while a live row for the CURRENT source still excludes the whole game
        # (INSERT OR IGNORE keeps the live row authoritative either way).
        have = {
            (row[0], row[1])
            for row in store.conn.execute(
                """
                SELECT DISTINCT game_id, source FROM forecast_snapshots
                WHERE league=? AND market='moneyline'
                """,
                (lane,),
            )
        }
        by_day: Dict[str, List[Dict[str, Any]]] = {}
        for row in finals:
            by_day.setdefault(sast_day(row["scheduled_utc"]), []).append(dict(row))
        prior_results: List[Result] = []
        seeded_rows: List[Dict[str, Any]] = []
        created = now_utc()
        for day in sorted(by_day):
            games = by_day[day]
            if len(prior_results) < MIN_SEED_HISTORY:
                lane_stats["matchdays_below_min_history"] += 1
            else:
                missing = [
                    game for game in games if (game["game_id"], source) not in have
                ]
                if missing:
                    fit_started = time.monotonic()
                    scored = _score_matchday_moneyline(lane, day, prior_results, missing)
                    summary["fit_seconds"] += time.monotonic() - fit_started
                    lane_stats["matchdays_scored"] += 1
                    for game_id, selection, probability in scored:
                        market = "moneyline"
                        period = "full_game"
                        seeded_rows.append(
                            {
                                "forecast_id": forecast_id_for(
                                    league=lane,
                                    game_id=game_id,
                                    market=market,
                                    period=period,
                                    selection=selection,
                                    line=None,
                                    source=source,
                                ),
                                "game_id": game_id,
                                "league": lane,
                                "market": market,
                                "period": period,
                                "source": source,
                                "selection": selection,
                                "line": None,
                                "raw_probability": probability,
                                "push_probability": 0.0,
                                "observed_day": day,
                                "created_utc": created,
                            }
                        )
                    lane_stats["rows"] = len(seeded_rows)
            for row in games:
                prior_results.append(
                    Result(
                        home_id=row["home_team_id"],
                        away_id=row["away_team_id"],
                        home_score=float(row["home_score"]),
                        away_score=float(row["away_score"]),
                        day=day,
                    )
                )
        if seeded_rows:
            lane_stats["inserted"] = store.restore_forecasts(
                seeded_rows, provenance="walk_forward_seed"
            )
    summary["seconds"] = round(time.monotonic() - started, 2)
    summary["fit_seconds"] = round(summary["fit_seconds"], 2)
    return summary


def run(
    *,
    store: Store,
    client: EspnClient,
    leagues: Iterable,
    seasons: Iterable[int],
    max_events: int,
    delay_seconds: float = 0.15,
    workers: int = 6,
    max_seconds: Optional[float] = 1800.0,
) -> dict:
    """Backfill finals inside a wall-clock budget, newest season first.

    The budget exists because the cold-cache path used to be a 30-minute job
    timeout waiting to happen: 9 leagues x 3 seasons x 250 events, sequential,
    0.15s apart. Stopping early is a normal outcome, not an error: whatever was
    stored stays stored, the manifest records the remainder, and the next run
    resumes from the same event index.
    """
    ordered_seasons = sorted({int(season) for season in seasons}, reverse=True)
    run_started = time.monotonic()
    deadline = (
        time.monotonic() + float(max_seconds)
        if isinstance(max_seconds, (int, float)) and max_seconds and max_seconds > 0
        else None
    )
    manifest: Dict[str, Any] = {
        "schema_version": "bullion.backfill_manifest",
        "generated_utc": now_utc(),
        "seasons": ordered_seasons,
        "season_order": "newest_first",
        "max_events_per_season": max_events,
        "workers": workers,
        "max_seconds": max_seconds,
        "stopped_early": False,
        "leagues": {},
        "note": (
            "Finals only. No historical prices are used as ticket or calibration "
            "evidence; calibration must accrue from live pre-tip forecasts."
        ),
    }
    for league in leagues:
        league_manifest: Dict[str, Any] = {}
        for season in ordered_seasons:
            counters = {"stored": 0, "failed": 0}
            season_started = time.monotonic()
            sample_errors: List[Dict[str, str]] = []
            already_stopped = manifest["stopped_early"]
            if already_stopped:
                league_manifest[str(season)] = {"status": "skipped_budget_exhausted"}
                continue
            try:
                ids = season_event_ids(client, league, season)[:max_events]
            except (EspnFetchError, ValueError) as exc:
                league_manifest[str(season)] = {"error": str(exc)}
                continue
            pending = [event_id for event_id in ids if store.get_game(event_id) is None]
            skipped = len(ids) - len(pending)
            if workers > 1:
                stopped = _concurrent(
                    client,
                    league,
                    pending,
                    store,
                    counters,
                    sample_errors,
                    workers,
                    deadline,
                )
            else:
                stopped = _sequential(
                    client,
                    league,
                    pending,
                    store,
                    counters,
                    sample_errors,
                    delay_seconds,
                    deadline,
                )
            processed = counters["stored"] + counters["failed"]
            elapsed = max(time.monotonic() - season_started, 1e-9)
            entry: Dict[str, Any] = {
                "indexed": len(ids),
                "pending_at_start": len(pending),
                "stored": counters["stored"],
                "skipped_existing": skipped,
                "failed": counters["failed"],
                "unprocessed": len(pending) - processed,
                "elapsed_seconds": round(elapsed, 3),
                "processed_per_second": round(processed / elapsed, 3),
                "stopped_early": stopped,
            }
            if sample_errors:
                entry["sample_errors"] = sample_errors
            league_manifest[str(season)] = entry
            print(
                f"[backfill] {league.key} {season}: stored={entry['stored']} "
                f"skipped={entry['skipped_existing']} failed={entry['failed']} "
                f"unprocessed={entry['unprocessed']} "
                f"in {entry['elapsed_seconds']:.1f}s "
                f"({entry['processed_per_second']:.2f}/s)",
                file=sys.stderr,
            )
            if stopped:
                manifest["stopped_early"] = True
        manifest["leagues"][league.key] = league_manifest
    total_processed = sum(
        item.get("stored", 0) + item.get("failed", 0)
        for league_manifest in manifest["leagues"].values()
        for item in league_manifest.values()
        if isinstance(item, dict)
    )
    total_unprocessed = sum(
        item.get("unprocessed", 0)
        for league_manifest in manifest["leagues"].values()
        for item in league_manifest.values()
        if isinstance(item, dict)
    )
    elapsed_total = max(time.monotonic() - run_started, 1e-9)
    rate = total_processed / elapsed_total
    manifest["elapsed_seconds"] = round(elapsed_total, 3)
    manifest["processed_per_second"] = round(rate, 3)
    # Evidence for the next budget decision instead of a guess: the observed rate
    # plus what is left says whether 30 minutes, more workers, or fewer events is
    # the right lever.
    manifest["projected_remaining_seconds"] = (
        round(total_unprocessed / rate, 1) if rate > 0 and total_unprocessed else 0.0
    )
    manifest["counts"] = store.counts()
    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", default="all")
    parser.add_argument("--seasons", default="2024,2025,2026")
    parser.add_argument("--max-events", type=int, default=400)
    parser.add_argument("--db", default="localdata/bullion.db")
    parser.add_argument(
        "--localdata",
        default="localdata",
        help="directory holding the committed receipts that restore forecast history",
    )
    parser.add_argument("--out", default="localdata/_backfill/manifest.json")
    parser.add_argument(
        "--delay", type=float, default=0.15, help="seconds between requests when --workers 1"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="bounded parallel summary fetches (1 = sequential, honours --delay)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=1800.0,
        help=(
            "wall-clock budget (default 1800 = 30 min); 0 disables. Must stay below "
            "the workflow's timeout-minutes minus the time daily/parity/commit need"
        ),
    )
    args = parser.parse_args(argv)
    if args.max_seconds and args.max_seconds >= JOB_TIMEOUT_WARNING_SECONDS:
        print(
            f"[backfill] warning: --max-seconds {args.max_seconds:.0f} is at or above "
            f"{JOB_TIMEOUT_WARNING_SECONDS:.0f}s. A budget stop exits 0 and saves the "
            "manifest, but a job timeout does not save the cache or commit the receipts; "
            "keep the budget under the job's timeout-minutes.",
            file=sys.stderr,
        )

    # Newest first: the current season's finals are what the live lanes need
    # ratings from, so a budget stop must never cost us those.
    seasons = sorted(
        {int(item) for item in args.seasons.split(",") if item.strip()}, reverse=True
    )
    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = ROOT / db_path
    with Store(db_path) as store:
        manifest = run(
            store=store,
            client=EspnClient(),
            leagues=select_leagues(args.league),
            seasons=seasons,
            max_events=args.max_events,
            delay_seconds=args.delay,
            workers=max(1, args.workers),
            max_seconds=args.max_seconds,
        )
        # Runs on every invocation, not just a cold cache: it is idempotent, and
        # a partially lost cache heals the same way a fully lost one does.
        # Order matters: receipts (provenance='live') claim their wager ids
        # before the walk-forward seeder runs, so INSERT OR IGNORE can never
        # displace a real pre-tip snapshot with a replayed one.
        localdata_dir = Path(args.localdata)
        if not localdata_dir.is_absolute():
            localdata_dir = ROOT / localdata_dir
        rehydrated = rehydrate_forecasts(store, localdata_dir)
        manifest["rehydrated"] = rehydrated
        manifest["seeded_ml"] = seed_walk_forward_ml(store)
        manifest["counts"] = store.counts()
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
