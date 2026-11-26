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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import engines
from .calibration import (
    DEFAULT_BRIER_CAP_2WAY,
    DEFAULT_BRIER_CAP_3WAY,
    CalibrationBook,
)
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
    parse_predictor,
    parse_summary,
)
from .models import (
    LEAGUES,
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
    PRICE_WEIGHT,
    GatePolicy,
    build_accas,
    decision_cutoff,
    eligible_rows,
    freeze_receipt,
    pick_row,
    render_card,
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
    if selector == "soccer":
        return [league for league in LEAGUES.values() if league.sport is Sport.SOCCER]
    if selector not in LEAGUES:
        raise ValueError(f"unknown league selector {selector!r}")
    return [LEAGUES[selector]]


def _results(store: Store, league: League, as_of_utc: str) -> List[engines.Result]:
    results: List[engines.Result] = []
    for game in store.settled_results(league.key, as_of_utc):
        if game.actual is None:
            continue
        results.append(
            engines.Result(
                home_id=game.home_team_id,
                away_id=game.away_team_id,
                home_score=game.actual.home_score,
                away_score=game.actual.away_score,
                day=sast_day(game.scheduled_utc),
            )
        )
    return results


def _calibration_book(store: Store, league: League, as_of_utc: str) -> CalibrationBook:
    book = CalibrationBook()
    for row in store.calibration_observations(league.key, as_of_utc):
        outcome = resolve_outcome(
            str(row["market"]),
            str(row["selection"]),
            row["line"],
            float(row["home_score"]),
            float(row["away_score"]),
        )
        if outcome is None:
            continue
        result_day = sast_day(row["scheduled_utc"])
        book.add(
            (str(row["league"]), str(row["market"]), str(row["period"]), str(row["source"])),
            float(row["raw_probability"]),
            None if outcome == "push" else (1 if outcome == "win" else 0),
            result_day,
        )
    return book


def _segments(forecasts: Sequence[Forecast]) -> List[Tuple[str, str, str, str]]:
    return sorted({(f.league, f.market, f.period, f.source) for f in forecasts})


def _reports(
    book: CalibrationBook, league: League, forecasts: Sequence[Forecast], day: str
) -> Dict[Tuple[str, str, str, str], Any]:
    cap = DEFAULT_BRIER_CAP_3WAY if league.has_draw else DEFAULT_BRIER_CAP_2WAY
    reports = {}
    for segment in _segments(forecasts):
        reports[segment] = book.report(
            segment,
            day,
            brier_cap=cap,
            cap_vs_base_rate=league.has_draw,
        )
    return reports


def _forecasts_for_game(
    store: Store,
    client: EspnClient,
    league: League,
    game: Game,
    results: Sequence[engines.Result],
    reference_day: str,
    observed_utc: str,
    gaps: List[CoverageGap],
) -> List[Forecast]:
    quotes = [quote for quote in store.quotes_for_game(game.game_id) if quote.pre_match]
    if not quotes:
        gaps.append(
            CoverageGap(league.key, "pickcenter", "no pre-tip quote for forecast", game.game_id)
        )
        return []
    if league.sport is Sport.HOCKEY:
        return nhl_engine.forecast(game, quotes, results=results, reference_day=reference_day)
    if league.sport is Sport.SOCCER:
        return soccer_engine.forecast(
            game, quotes, results=results, reference_day=reference_day
        )
    predictor = None
    try:
        payload = client.predictor(league, game.game_id)
        predictor = parse_predictor(payload, game)
        store.record_predictor(
            game_id=game.game_id,
            league=league.key,
            provider="espn",
            model="espn_fpi",
            observed_utc=observed_utc,
            prob_home=predictor.prob_home,
            expected_home=predictor.expected_home,
            expected_away=predictor.expected_away,
        )
    except (EspnFetchError, EspnIdentityError, EspnParseError, ValueError) as exc:
        gaps.append(CoverageGap(league.key, "predictor", str(exc), game.game_id))
    forecasts = football_engine.forecast(
        game,
        quotes,
        results=results,
        reference_day=reference_day,
        surface=football_engine.EMPIRICAL_SOURCE,
    )
    if predictor is not None:
        forecasts.extend(
            football_engine.forecast(
                game,
                quotes,
                results=results,
                reference_day=reference_day,
                surface=football_engine.FPI_SOURCE,
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
    legs: int = 2,
    limit: int = 3,
    policy: Optional[GatePolicy] = None,
    min_combined_odds: Optional[float] = 3.00,
    max_combined_odds: Optional[float] = 4.50,
) -> LaneRun:
    """Run one lane for one SAST day. Freezes only at/after the lane cutoff."""
    as_of = to_utc_z(as_of_utc or now_utc())
    policy = policy or GatePolicy()
    run = LaneRun(lane=league.key, day=day, status="started")
    gaps = run.gaps
    permission: LanePermission = lane_permission(probe_report, league)
    run.ticket_markets = sorted(permission.ticket_markets)

    try:
        scoreboard = discover(client, league, day, as_of)
    except (EspnFetchError, EspnParseError, ValueError) as exc:
        run.status = "ingest_failed"
        run.reason = str(exc)
        gaps.append(CoverageGap(league.key, "scoreboard", str(exc)))
        return run

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

    results = _results(store, league, as_of)
    scheduled = [
        game
        for game in store.games_for_league(league.key, day)
        if game.status is Status.SCHEDULED and parse_utc(game.scheduled_utc) > parse_utc(as_of)
    ]
    forecasts: List[Forecast] = []
    for game in scheduled:
        forecasts.extend(
            _forecasts_for_game(
                store,
                client,
                league,
                game,
                results,
                reference_day=day,
                observed_utc=as_of,
                gaps=gaps,
            )
        )
    run.forecasts = len(forecasts)
    store.record_forecasts(forecasts, day)

    permission_markets = permission.ticket_markets
    ticket_forecasts = [f for f in forecasts if f.market in permission_markets]
    book = _calibration_book(store, league, as_of)
    reports = _reports(book, league, ticket_forecasts, day)
    surface_resolution = "single_surface"
    if league.sport is Sport.FOOTBALL:
        brier_by_source = {
            source: reports[(league.key, "moneyline", "full_game", source)].brier
            for source in {forecast.source for forecast in ticket_forecasts}
            if (league.key, "moneyline", "full_game", source) in reports
        }
        winner = football_engine.resolve_surface(brier_by_source)
        surface_resolution = winner or "undecided_both_surfaces_gated"
        if winner is not None:
            ticket_forecasts = [f for f in ticket_forecasts if f.source == winner]
    prior_roi = roi_band(settled_pick_history(localdata, league), as_of_day=day)
    scored = score_forecasts(
        ticket_forecasts,
        list(scoreboard.quotes),
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
        legs=legs,
        limit=limit,
        price_weight=PRICE_WEIGHT,
        min_leg_odds=policy.min_leg_odds,
        max_leg_odds=policy.max_leg_odds,
        min_combined_odds=min_combined_odds,
        max_combined_odds=max_combined_odds,
        forbid_conflicting_game_sides=True,
    )
    run.tickets = len(tickets)

    tips = [forecast.scheduled_utc for forecast in forecasts] or [
        game.scheduled_utc for game in scheduled
    ]
    cutoff = decision_cutoff(league, day, tips)
    run.cutoff_utc = cutoff
    if cutoff is None:
        run.status = "no_slate"
        run.reason = "no scheduled fixtures on the requested day"
        return run
    if parse_utc(as_of) < parse_utc(cutoff):
        run.status = "pre_cutoff_scan"
        run.reason = f"decision cutoff {cutoff} not reached; scan only"
        return run

    selection_policy = {
        "requested_legs": legs,
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
        "brier_cap": policy.brier_cap(league),
        "ticket_markets": sorted(permission_markets),
        "lane_permission": permission.reason,
        "surface_resolution": surface_resolution,
        "probe_report_used": probe_report is not None,
        "paper_only": True,
        "stake_units_per_acca": 1.0,
    }
    decision_state = {
        "status": "frozen" if tickets else "frozen_no_bet",
        "planned_cutoff_utc": cutoff,
        "captured_utc": as_of,
        "candidate_count": len(pool_rows),
        "eligible_count": len(eligible),
        "permission": permission.reason,
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
        card_dir = localdata / league.lane / "cards"
        card_dir.mkdir(parents=True, exist_ok=True)
        (card_dir / f"auto_tickets_{day}.txt").write_text(render_card(receipt), encoding="utf-8")
    else:
        run.reason = (
            "no qualifying pre-tip ticket at cutoff"
            if permission_markets
            else f"research_only: {permission.reason}"
        )

    games_by_id = {game.game_id: game for game in store.games_for_league(league.key)}
    settle_lane(localdata, league, games_by_id)
    _write_coverage(localdata, league, day, run)
    return run


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
