"""End-to-end lane runner: ingest -> forecast -> gate -> freeze -> grade -> card.

Cutoffs (SAST, pulled earlier for an early slate tip):

* NHL / NFL / CFB: 16:00 SAST, bounded by earliest tip - 120m;
* Top-6 soccer: 12:00 SAST, bounded by earliest tip - 120m.

Before the cutoff a scan is only candidate evidence and never freezes a
receipt. At/after the cutoff the first scan freezes either its pre-tip ticket
candidates or a terminal no-bet lock. Neither is ever rotated.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import engines
from .calibration import CalibrationBook
from .engines import basketball as basketball_engine
from .engines import football as football_engine
from .engines import nhl as nhl_engine
from .engines import soccer as soccer_engine
from .espn import (
    EspnClient,
    discover,
    EspnFetchError,
    EspnIdentityError,
    EspnParseError,
    LanePermission,
    lane_permission,
    parse_pickcenter,
    parse_predictor,
    parse_summary,
    summary_scheduled_utc,
    summary_status,
)
from .models import (
    LEAGUES,
    ActualResult,
    BookQuote,
    CoverageGap,
    Forecast,
    Game,
    League,
    Sport,
    Status,
    now_utc,
    parse_utc,
    sast_day,
    to_utc_z,
)
from .store import Store
from .tickets import (
    DEFAULT_ACCA_LIMIT,
    DEFAULT_MAX_COMBINED_ODDS,
    DEFAULT_MAX_LEGS,
    DEFAULT_MIN_COMBINED_ODDS,
    DEFAULT_MIN_LEGS,
    PRICE_WEIGHT,
    RANKING_METRIC_KELLY,
    VALID_RANKING_METRICS,
    GatePolicy,
    build_accas,
    decision_cutoff,
    eligible_rows,
    freeze_receipt,
    latest_quotes,
    pick_row,
    resolve_outcome,
    roi_band,
    score_forecasts,
    settle_lane,
    settled_pick_history,
)


@dataclass
class LaneRun:
    lane: str
    day: str
    status: str
    reason: str = ""
    games: int = 0
    forecasts: int = 0
    candidate_pool: int = 0
    eligible_legs: int = 0
    refreshed_finals: int = 0
    tickets: int = 0
    ticket_markets: List[str] = field(default_factory=list)
    cutoff_utc: Optional[str] = None
    receipt_path: Optional[str] = None
    gaps: List[CoverageGap] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = {key: value for key, value in self.__dict__.items() if key != "gaps"}
        payload["gaps"] = [gap.to_dict() for gap in self.gaps]
        return payload


def select_leagues(selector: str) -> List[League]:
    selector = (selector or "all").strip().lower()
    if selector == "all":
        return list(LEAGUES.values())
    if "," in selector:
        # A comma list is how the scheduler names the cold lanes it just
        # detected; expanding it here keeps every caller on one selector rule.
        selected: List[League] = []
        for item in selector.split(","):
            item = item.strip()
            if not item:
                continue
            for league in select_leagues(item):
                if league not in selected:
                    selected.append(league)
        if not selected:
            raise ValueError(f"unknown league selector {selector!r}")
        return selected
    if selector == "soccer":
        return [league for league in LEAGUES.values() if league.sport is Sport.SOCCER]
    if selector == "basketball":
        return [league for league in LEAGUES.values() if league.sport is Sport.BASKETBALL]
    if selector not in LEAGUES:
        raise ValueError(f"unknown league selector {selector!r}")
    return [LEAGUES[selector]]


def _results(store: Store, league: League, as_of_utc: str) -> List[engines.Result]:
    """Settled finals with their schedule day, instant and quoted main lines.

    ``day`` and the forecast's ``reference_day`` must use the SAME timezone
    (``League.schedule_day``), or a Friday-night ET final filed under ET day D
    reads as "same day" against a Saturday card whose reference day is SAST
    D+1 — and a night-game back-to-back silently escapes the fatigue penalty.
    The market lines make the anchored sigma measurable on historical anchored
    residuals instead of a closed form.
    """
    main_lines = store.latest_pre_tip_main_lines(league.key)
    results: List[engines.Result] = []
    for game in store.settled_results(league.key, as_of_utc):
        if game.actual is None:
            continue
        lines = main_lines.get(game.game_id) or {}
        results.append(
            engines.Result(
                home_id=game.home_team_id,
                away_id=game.away_team_id,
                home_score=game.actual.home_score,
                away_score=game.actual.away_score,
                day=league.schedule_day(game.scheduled_utc),
                scheduled_utc=to_utc_z(game.scheduled_utc),
                market_spread=lines.get("spread"),
                market_total=lines.get("total"),
            )
        )
    return results


def _calibration_book(store: Store, league: League, as_of_utc: str) -> CalibrationBook:
    book = CalibrationBook()
    for row in store.calibration_observations(league.key, as_of_utc):
        home_score = float(row["home_score"])
        away_score = float(row["away_score"])
        overtime_periods = int(row["overtime_periods"] or 0)
        if league.has_draw and overtime_periods > 0:
            # Calibration must grade a 3-way market on regulation, exactly as
            # settlement does; an ET-inclusive score would teach the model a
            # draw probability it never had the chance to price.
            periods = tuple(tuple(pair) for pair in json.loads(row["periods_json"] or "[]"))
            actual = ActualResult(
                home_score=home_score,
                away_score=away_score,
                periods=periods,
                overtime_periods=overtime_periods,
            )
            regulation = actual.regulation_scores()
            if regulation is None:
                continue  # fail closed: never calibrate on ET-inclusive scores
            home_score, away_score = regulation
        outcome = resolve_outcome(
            str(row["market"]),
            str(row["selection"]),
            row["line"],
            home_score,
            away_score,
            has_draw=league.has_draw,
        )
        if outcome is None:
            continue
        result_day = sast_day(row["scheduled_utc"])
        book.add(
            (str(row["league"]), str(row["market"]), str(row["period"]), str(row["source"])),
            float(row["raw_probability"]),
            None if outcome == "push" else (1 if outcome == "win" else 0),
            result_day,
            str(row["game_id"]),
            selection=str(row["selection"]),
        )
    return book


def _segments(forecasts: Sequence[Forecast]) -> List[Tuple[str, str, str, str]]:
    return sorted({(f.league, f.market, f.period, f.source) for f in forecasts})


def _reports(
    book: CalibrationBook,
    league: League,
    forecasts: Sequence[Forecast],
    day: str,
    policy: Optional[GatePolicy] = None,
) -> Dict[Tuple[str, str, str, str], Any]:
    """One walk-forward report per segment, capped by that market's cardinality.

    A soccer ``spread`` is a two-outcome market, so the 3-way capacity (0.22)
    was the wrong bar for it; only a 3-way ``moneyline`` uses it. Every segment
    is also measured against its own base-rate Brier, because a segment that
    only matches "always predict the base rate" has no skill to sell.
    """
    policy = policy or GatePolicy()
    reports = {}
    for segment in _segments(forecasts):
        reports[segment] = book.report(
            segment,
            day,
            brier_cap=policy.brier_cap(league, segment[1]),
            cap_vs_base_rate=True,
        )
    return reports


#: A game whose tip passed more than this long ago should already have a final.
#: The 06:15Z grade cron only discovers "today", so without a lookback a game
#: that finished after rollover stays ``scheduled`` in SQLite and starves both
#: calibration (``WHERE g.status='final'``) and settlement. The queue is bounded
#: at both ends and drained newest-first (see ``REFRESH_LOOKBACK_DAYS`` and
#: ``REFRESH_MAX_GAMES``) so the games that just finished are always graded and
#: the queue cannot grow an unfetchable tail.
REFRESH_AFTER_HOURS = 2.0
#: 120, not 40: an NCAA Saturday plus a Sunday rollover can leave 80+ finals
#: unseen by the scoreboard scan, and a cap of 40 turns that into a backlog that
#: never drains (each run re-checks the same 120 and reports the rest).
REFRESH_MAX_GAMES = 120
#: How far back the refresh queue may reach. Without a floor, a fixture ESPN
#: never resolves (placeholder bracket slot, abandoned event on an old payload)
#: would be re-fetched on every tick forever, and with a bounded cap it would
#: eventually crowd out the games that just finished. Two weeks covers every real
#: rollover gap many times over.
REFRESH_LOOKBACK_DAYS = 14


def _scoring_quotes(
    store: Store,
    league: League,
    scoreboard_quotes: Sequence[BookQuote],
    scheduled: Sequence[Game],
    gaps: List[CoverageGap],
) -> List[BookQuote]:
    """Every pre-tip quote the *scheduled* games have, not just this scan's array.

    ``_forecasts_for_game`` can price a fixture from a summary ``pickcenter``
    when the scoreboard odds array is empty, and it writes those quotes to the
    store. Scoring has to read them back: passing only ``scoreboard.quotes``
    left every pickcenter-only leg with no book, which fails closed as
    ``no_pre_tip_named_book_quote`` and silently empties the lane.
    """
    by_id = {game.game_id: game for game in scheduled}
    quotes: List[BookQuote] = []
    seen: set[Tuple[Any, ...]] = set()
    flagged: set[str] = set()

    def _add(quote: BookQuote, game: Game) -> None:
        key = (quote.game_id, quote.market, quote.line, quote.book, quote.observed_utc)
        if key in seen:
            return
        if not quote.pre_match or parse_utc(quote.observed_utc) >= parse_utc(
            game.scheduled_utc
        ):
            if quote.game_id not in flagged:
                flagged.add(quote.game_id)
                gaps.append(
                    CoverageGap(
                        league.key,
                        "quote.post_tip",
                        "quote observed at/after tip for a still-scheduled game; not used",
                        quote.game_id,
                    )
                )
            return
        seen.add(key)
        quotes.append(quote)

    for quote in scoreboard_quotes:
        game = by_id.get(quote.game_id)
        if game is not None:
            _add(quote, game)
    for game in scheduled:
        for quote in store.quotes_for_game(game.game_id):
            _add(quote, game)
    # Same rule as forecasting: a stale earlier-tick snapshot may not shadow the
    # book's live board for that game/market.
    return latest_quotes(quotes)


def _refresh_recent_finals(
    store: Store,
    client: EspnClient,
    league: League,
    as_of_utc: str,
    gaps: List[CoverageGap],
    deadline: Optional[float] = None,
) -> int:
    """Re-check recently tipped, still-unsettled games and store their outcome.

    Discovery only asks ESPN for the ET days that can hold *this* SAST day, so a
    game that tipped yesterday evening and finished after rollover is never
    revisited. Grading, settlement and calibration all read ``status='final'``,
    so that game would sit as ``scheduled`` forever.

    The queue is deliberately small in *kind*: it holds only games that can still
    produce a result. A summary that reports the fixture canceled or postponed
    persists that status and leaves the queue (it will never have a score), and a
    fixture ESPN has moved is re-dated so the entry clocks off its new tip. One
    unfetchable event therefore costs one summary call in total, not one per tick.
    """
    as_of = parse_utc(as_of_utc)
    newest = as_of - timedelta(hours=REFRESH_AFTER_HOURS)
    oldest = as_of - timedelta(days=REFRESH_LOOKBACK_DAYS)
    stale = [
        game
        for game in store.games_for_league(league.key)
        # A canceled or postponed fixture is terminal for the refresh queue: it
        # has no result to fetch, and leaving it in would burn a summary call on
        # every tick for the rest of the season. If ESPN reschedules one, the
        # scoreboard scan on its new day upserts it back to ``scheduled``.
        if game.status not in {Status.FINAL, Status.CANCELLED, Status.POSTPONED}
        and oldest <= parse_utc(game.scheduled_utc) <= newest
    ]
    # Newest first: yesterday's finals must drain before a month-old backlog,
    # otherwise a growing tail starves the games that actually need grading.
    stale.sort(key=lambda game: game.scheduled_utc, reverse=True)
    if len(stale) > REFRESH_MAX_GAMES:
        gaps.append(
            CoverageGap(
                league.key,
                "summary.refresh.backlog",
                f"{len(stale)} unsettled past-tip games; this run drains the newest "
                f"{REFRESH_MAX_GAMES}",
            )
        )
    refreshed = 0
    for game in stale[:REFRESH_MAX_GAMES]:
        if deadline is not None and time.monotonic() >= deadline:
            # The queue is newest-first and persists terminal states, so
            # stopping early only delays the tail; it never corrupts it.
            gaps.append(
                CoverageGap(
                    league.key,
                    "runner.budget",
                    "wall-clock deadline reached during finals refresh",
                )
            )
            break
        try:
            payload = client.summary(league, game.game_id)
        except (EspnFetchError, EspnParseError, ValueError) as exc:
            gaps.append(CoverageGap(league.key, "summary.refresh", str(exc), game.game_id))
            continue
        status = summary_status(payload)
        if status in {Status.CANCELLED, Status.POSTPONED}:
            # Terminal, but not a result: persist the status so the next tick
            # skips the fixture instead of re-fetching it forever. No gap — an
            # off fixture is a fact, not a parser failure.
            store.upsert_game(replace(game, status=status))
            continue
        if status is Status.SCHEDULED:
            # ESPN moved the fixture. Adopt the new tip time when it is later
            # than ours; the queue entry now clocks off the new date.
            rescheduled = summary_scheduled_utc(payload)
            if rescheduled and parse_utc(rescheduled) > parse_utc(game.scheduled_utc):
                store.upsert_game(replace(game, scheduled_utc=rescheduled))
            continue
        if status is not Status.FINAL:
            # In progress, or a payload that is neither pre nor post: the next
            # run re-checks it. Never forced to a final.
            continue
        try:
            final_game, summary_gaps = parse_summary(payload, game, league)
        except (EspnIdentityError, EspnParseError, ValueError) as exc:
            gaps.append(CoverageGap(league.key, "summary.refresh", str(exc), game.game_id))
            continue
        gaps.extend(summary_gaps)
        store.upsert_game(final_game)
        refreshed += 1
    return refreshed


def _forecasts_for_game(
    store: Store,
    client: EspnClient,
    league: League,
    game: Game,
    results: Sequence[engines.Result],
    observed_utc: str,
    gaps: List[CoverageGap],
) -> List[Forecast]:
    # Latest board per book only: the warehouse keeps every snapshot it has ever
    # seen, and pricing off the best of several ticks would quote a line the book
    # has already moved away from.
    quotes = [
        quote for quote in latest_quotes(store.quotes_for_game(game.game_id)) if quote.pre_match
    ]
    if not quotes:
        # LIVE_VERIFIED 2026-11-26: scoreboard odds arrays can be empty while the
        # summary carries a populated ``pickcenter``. Fetch it once per game
        # rather than declaring the fixture unpriceable on scoreboard evidence.
        try:
            payload = client.summary(league, game.game_id)
        except (EspnFetchError, EspnParseError, ValueError) as exc:
            gaps.append(CoverageGap(league.key, "pickcenter.summary", str(exc), game.game_id))
        else:
            parsed_pickcenter = parse_pickcenter(payload, game, observed_utc)
            gaps.extend(parsed_pickcenter.gaps)
            if parsed_pickcenter.quotes:
                store.record_quotes(parsed_pickcenter.quotes, league.key)
                quotes = [quote for quote in parsed_pickcenter.quotes if quote.pre_match]
    if not quotes:
        gaps.append(
            CoverageGap(league.key, "pickcenter", "no pre-tip quote for forecast", game.game_id)
        )
        return []
    # Aligned with ``Result.day`` (same timezone per sport), so walk-forward
    # filters and the fatigue question agree with the ratings fit.
    reference_day = league.schedule_day(game.scheduled_utc)
    if league.sport is Sport.HOCKEY:
        return nhl_engine.forecast(game, quotes, results=results, reference_day=reference_day)
    if league.sport is Sport.SOCCER:
        return soccer_engine.forecast(
            game, quotes, results=results, reference_day=reference_day
        )
    engine = basketball_engine if league.sport is Sport.BASKETBALL else football_engine
    predictor = None
    try:
        payload = client.predictor(league, game.game_id)
        predictor = parse_predictor(payload, game)
        store.record_predictor(
            game_id=game.game_id,
            league=league.key,
            provider="espn",
            model=engine.BPI_SOURCE if engine is basketball_engine else "espn_fpi",
            observed_utc=observed_utc,
            prob_home=predictor.prob_home,
            expected_home=predictor.expected_home,
            expected_away=predictor.expected_away,
            predicted_margin=predictor.predicted_margin,
        )
    except (EspnFetchError, EspnIdentityError, EspnParseError, ValueError) as exc:
        gaps.append(CoverageGap(league.key, "predictor", str(exc), game.game_id))
    forecasts = engine.forecast(
        game,
        quotes,
        results=results,
        reference_day=reference_day,
        surface=engine.EMPIRICAL_SOURCE,
    )
    if predictor is not None:
        forecasts.extend(
            engine.forecast(
                game,
                quotes,
                results=results,
                reference_day=reference_day,
                surface=engine.BPI_SOURCE if engine is basketball_engine else engine.FPI_SOURCE,
                predictor=predictor,
            )
        )
    return forecasts


def run_lane(
    *,
    league: League,
    day: str,
    store: Store,
    localdata: Path,
    client: EspnClient,
    as_of_utc: Optional[str] = None,
    probe_report: Optional[Dict[str, Any]] = None,
    legs: Optional[int] = None,
    min_legs: int = DEFAULT_MIN_LEGS,
    max_legs: int = DEFAULT_MAX_LEGS,
    limit: int = DEFAULT_ACCA_LIMIT,
    policy: Optional[GatePolicy] = None,
    min_combined_odds: Optional[float] = DEFAULT_MIN_COMBINED_ODDS,
    max_combined_odds: Optional[float] = DEFAULT_MAX_COMBINED_ODDS,
    ranking_metric: str = RANKING_METRIC_KELLY,
    forbid_alt_line_reuse: bool = True,
    deadline: Optional[float] = None,
) -> LaneRun:
    """Run one lane for one SAST day. Freezes only at/after the lane cutoff."""
    as_of = to_utc_z(as_of_utc or now_utc())
    policy = policy or GatePolicy()
    if ranking_metric not in VALID_RANKING_METRICS:
        raise ValueError(f"unknown ranking metric {ranking_metric!r}")
    # ``legs`` is the pre-Option-1 override: an explicit single leg count pins
    # both ends of the range, otherwise 2- and 3-leg accas compete together.
    if legs is not None:
        eff_min_legs = eff_max_legs = int(legs)
    else:
        eff_min_legs, eff_max_legs = int(min_legs), int(max_legs)
    if not (2 <= eff_min_legs <= eff_max_legs <= 6):
        raise ValueError("min_legs/max_legs must satisfy 2 <= min <= max <= 6")
    run = LaneRun(lane=league.key, day=day, status="started")
    gaps = run.gaps

    def _budget_ok() -> bool:
        """True while wall-clock budget remains (``None`` deadline = unlimited)."""
        return deadline is None or time.monotonic() < deadline

    permission: LanePermission = lane_permission(probe_report, league, as_of_utc=as_of)
    run.ticket_markets = sorted(permission.ticket_markets)

    def finish() -> LaneRun:
        """Settle every frozen day, then persist this run's coverage.

        Grading must not depend on whether today happened to have a slate or a
        cutoff: the 06:15Z cron exists to settle yesterday, and ``no_slate`` /
        ``pre_cutoff_scan`` used to return before ``settle_lane`` ever ran.
        """
        games_by_id = {game.game_id: game for game in store.games_for_league(league.key)}
        settle_lane(localdata, league, games_by_id)
        _write_coverage(localdata, league, day, run)
        return run

    try:
        scoreboard = discover(client, league, day, as_of)
    except (EspnFetchError, EspnParseError, ValueError) as exc:
        run.status = "ingest_failed"
        run.reason = str(exc)
        gaps.append(CoverageGap(league.key, "scoreboard", str(exc)))
        return finish()

    gaps.extend(scoreboard.gaps)
    for game in scoreboard.games:
        store.upsert_game(game)
    store.record_quotes(scoreboard.quotes, league.key)
    run.games = len(scoreboard.games)

    # Finals: fetch the summary so period evidence and official scores exist.
    for game in scoreboard.games:
        if game.status is not Status.FINAL or game.actual is not None:
            continue
        try:
            payload = client.summary(league, game.game_id)
            final_game, summary_gaps = parse_summary(payload, game, league)
        except (EspnFetchError, EspnIdentityError, EspnParseError, ValueError) as exc:
            gaps.append(CoverageGap(league.key, "summary", str(exc), game.game_id))
            continue
        gaps.extend(summary_gaps)
        store.upsert_game(final_game)

    # Yesterday's late finals, refreshed before anything reads ``status='final'``.
    run.refreshed_finals = _refresh_recent_finals(
        store, client, league, as_of, gaps, deadline=deadline
    )

    results = _results(store, league, as_of)
    scheduled = [
        game
        for game in store.games_for_league(league.key, day)
        if game.status is Status.SCHEDULED and parse_utc(game.scheduled_utc) > parse_utc(as_of)
    ]
    # Today's card as the day itself sees it (an off fixture is not a game that
    # "tipped"), so a missed freeze window can be told apart from an empty day.
    day_games = [
        game
        for game in store.games_for_league(league.key, day)
        if game.status not in {Status.CANCELLED, Status.POSTPONED}
    ]
    tipped = [game for game in day_games if parse_utc(game.scheduled_utc) <= parse_utc(as_of)]
    forecasts: List[Forecast] = []
    breached = False
    for game in scheduled:
        if not _budget_ok():
            breached = True
            break
        forecasts.extend(
            _forecasts_for_game(
                store,
                client,
                league,
                game,
                results,
                observed_utc=as_of,
                gaps=gaps,
            )
        )
    run.forecasts = len(forecasts)
    store.record_forecasts(forecasts, day)
    if breached:
        # Freeze nothing new: a half-forecasted pool must never reach the
        # cutoff freeze. Settlement still runs in ``finish()`` so yesterday's
        # positions grade and today's partial evidence is persisted.
        run.status = "budget_breached"
        run.reason = "wall-clock deadline reached"
        gaps.append(
            CoverageGap(
                league.key,
                "runner.budget",
                "wall-clock deadline reached during forecasting; freezing nothing this tick",
            )
        )
        return finish()

    permission_markets = permission.ticket_markets
    ticket_forecasts = [f for f in forecasts if f.market in permission_markets]
    book = _calibration_book(store, league, as_of)
    reports = _reports(book, league, ticket_forecasts, day, policy)
    surface_resolution = "single_surface"
    # Any lane with a provider predictor runs two surfaces and has to pick one
    # per lane: leaving both in the pool lets a single ticket mix an FPI leg and
    # an empirical leg on the same market, which is a selection nobody measured.
    if league.predictor:
        surface_mod = (
            basketball_engine if league.sport is Sport.BASKETBALL else football_engine
        )
        brier_by_source = {
            source: reports[(league.key, "moneyline", "full_game", source)].brier
            for source in {forecast.source for forecast in ticket_forecasts}
            if (league.key, "moneyline", "full_game", source) in reports
        }
        winner = surface_mod.resolve_surface(brier_by_source)
        if winner is None:
            # "Undecided" must still mean one surface. Leaving both in the pool
            # lets a ticket mix FPI and empirical legs on the same market, which
            # is a selection nobody measured; pick deterministically instead and
            # say so in the receipt.
            winner = surface_mod.tiebreak_surface(brier_by_source)
            surface_resolution = (
                f"brier_tiebreak_{winner}" if winner else "no_surface_evidence"
            )
        else:
            surface_resolution = winner
        if winner is not None:
            have = {f.market for f in ticket_forecasts if f.source == winner}
            missing = permission_markets - have
            if missing:
                # A surface that wins the ML Brier can still be structurally
                # unable to price a probe-permitted market (live ncaa_m: BPI
                # ships prob_home with no expected scores, so it has no total
                # market at all). Never delete a permitted market silently and
                # never mix surfaces: demote to the surface that covers more of
                # the permitted markets, deterministically.
                alt = (
                    surface_mod.EMPIRICAL_SOURCE
                    if winner != surface_mod.EMPIRICAL_SOURCE
                    else getattr(
                        surface_mod,
                        "BPI_SOURCE",
                        getattr(surface_mod, "FPI_SOURCE", None),
                    )
                )
                alt_have = (
                    {f.market for f in ticket_forecasts if f.source == alt} if alt else set()
                )
                if alt and len(alt_have & permission_markets) > len(have & permission_markets):
                    gaps.append(
                        CoverageGap(
                            league.key,
                            "surface.market_coverage",
                            f"{winner} cannot price permitted markets "
                            f"{sorted(missing)}; demoted to {alt}",
                        )
                    )
                    winner = alt
                    surface_resolution = f"coverage_demoted_{alt}"
            ticket_forecasts = [f for f in ticket_forecasts if f.source == winner]
    prior_roi = roi_band(settled_pick_history(localdata, league), as_of_day=day)
    scored = score_forecasts(
        ticket_forecasts,
        _scoring_quotes(store, league, scoreboard.quotes, scheduled, gaps),
        league=league,
        reports=reports,
        prior_roi=prior_roi,
        policy=policy,
    )
    pool_rows = [pick_row(forecast, day) for forecast in scored]
    run.candidate_pool = len(pool_rows)
    eligible = eligible_rows(pool_rows, policy=policy)
    run.eligible_legs = len(eligible)
    tickets = build_accas(
        eligible,
        min_legs=eff_min_legs,
        max_legs=eff_max_legs,
        limit=limit,
        price_weight=PRICE_WEIGHT,
        ranking_metric=ranking_metric,
        min_leg_odds=policy.min_leg_odds,
        max_leg_odds=policy.max_leg_odds,
        min_combined_odds=min_combined_odds,
        max_combined_odds=max_combined_odds,
        forbid_conflicting_game_sides=True,
        forbid_alt_line_reuse=forbid_alt_line_reuse,
        max_claimed_edge=policy.max_claimed_edge,
        edge_flag_threshold=policy.edge_flag_threshold,
    )
    run.tickets = len(tickets)

    tips = [forecast.scheduled_utc for forecast in forecasts] or [
        game.scheduled_utc for game in scheduled
    ]
    cutoff = decision_cutoff(league, day, tips)
    run.cutoff_utc = cutoff
    if cutoff is None:
        decision_path = localdata / league.lane / f"auto_tickets_{day}_decision.json"
        no_bet_path = localdata / league.lane / f"auto_tickets_{day}_no_bet.json"
        if day_games and tipped and not decision_path.exists() and not no_bet_path.exists():
            # The freeze window was missed entirely (a weekend tick gap). A
            # partial card must never freeze silently, and the day must not
            # vanish as "no_slate" either: freeze a terminal no-bet lock, which
            # still contributes its graded pool to prior_roi evidence.
            gaps.append(
                CoverageGap(
                    league.key,
                    "slate.missed_freeze_window",
                    f"{len(tipped)} of {len(day_games)} day games tipped before any "
                    "scan reached the cutoff; freezing a terminal no-bet lock",
                )
            )
            freeze_receipt(
                localdata,
                league=league,
                day=day,
                captured_utc=as_of,
                candidate_pool=[],
                accas=[],
                decision_state={
                    "status": "frozen_no_bet",
                    "captured_utc": as_of,
                    "reason": "missed_freeze_window",
                },
                reason="missed_freeze_window",
            )
            run.status = "frozen_no_bet"
            run.reason = "missed_freeze_window"
            return finish()
        run.status = "no_slate"
        run.reason = "no scheduled fixtures on the requested day"
        return finish()
    if parse_utc(as_of) < parse_utc(cutoff):
        run.status = "pre_cutoff_scan"
        run.reason = f"decision cutoff {cutoff} not reached; scan only"
        return finish()

    if scheduled and tipped:
        # Part of today's card tipped before this scan reached the cutoff, so
        # those games' lines are unusable post-tip. Do not nuke the rest of the
        # live slate — make the shortfall visible instead.
        gaps.append(
            CoverageGap(
                league.key,
                "slate.tipped_before_freeze",
                f"{len(tipped)} of {len(day_games)} day games tipped before freeze",
            )
        )

    selection_policy = {
        "requested_legs": eff_max_legs,
        "ranking_metric": ranking_metric,
        "min_legs": eff_min_legs,
        "max_legs": eff_max_legs,
        "forbid_alt_line_reuse": forbid_alt_line_reuse,
        "max_claimed_edge": policy.max_claimed_edge,
        "edge_flag_threshold": policy.edge_flag_threshold,
        "limit": limit,
        "allow_provisional": False,
        "require_disjoint_legs": True,
        "forbid_conflicting_game_sides": True,
        "min_leg_odds": policy.min_leg_odds,
        "max_leg_odds": policy.max_leg_odds,
        "min_combined_odds": min_combined_odds,
        "max_combined_odds": max_combined_odds,
        "price_weight": PRICE_WEIGHT,
        "min_samples": 30,
        "min_games": 30,
        "brier_cap": policy.brier_cap(league, "moneyline"),
        "brier_cap_two_way": policy.brier_cap_2way,
        "brier_cap_three_way": policy.brier_cap_3way,
        "cap_vs_base_rate": True,
        "ticket_markets": sorted(permission_markets),
        "lane_permission": permission.reason,
        "probe_captured_utc": permission.captured_utc,
        "probe_market_evidence": permission.market_evidence,
        "surface_resolution": surface_resolution,
        "probe_report_used": probe_report is not None,
        "paper_only": True,
        "stake_units_per_acca": 1.0,
    }
    if basketball_engine.POSTED_LINE_MODEL_WEIGHTS.get(league.key) is not None:
        # Recorded on the receipt so a later reader can tell which dispersion
        # estimator priced this ladder; parity ignores unknown policy keys.
        selection_policy["sigma_model"] = "anchored_orthogonal_v1"
    decision_state = {
        "status": "frozen" if tickets else "frozen_no_bet",
        "planned_cutoff_utc": cutoff,
        "captured_utc": as_of,
        "candidate_count": len(pool_rows),
        "eligible_count": len(eligible),
        "permission": permission.reason,
        "probe_captured_utc": permission.captured_utc,
    }
    receipt = freeze_receipt(
        localdata,
        league=league,
        day=day,
        captured_utc=as_of,
        candidate_pool=pool_rows,
        accas=tickets,
        selection_policy=selection_policy,
        decision_state=decision_state,
    )
    run.status = "frozen_ticket" if receipt else "frozen_no_bet"
    if receipt is not None:
        run.receipt_path = str(localdata / league.lane / f"auto_tickets_{day}_decision.json")
        # The human card is written by ``settle_lane`` in ``finish()``, which
        # runs after this freeze: it re-renders every dated card at the lane
        # root from the graded copy, so a frozen PENDING never outlives a result.
    else:
        run.reason = (
            "no qualifying pre-tip ticket at cutoff"
            if permission_markets
            else f"research_only: {permission.reason}"
        )

    return finish()


def _write_coverage(localdata: Path, league: League, day: str, run: LaneRun) -> None:
    target = localdata / "_coverage"
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "bullion.lane_run",
        "generated_utc": now_utc(),
        "run": run.to_dict(),
        "gate": GatePolicy().to_policy_dict(),
    }
    (target / f"{league.key}_{day}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
