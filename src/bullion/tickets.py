"""Calibrated value-ticket lane, acca assembly, immutable receipts, settlement.

Everything in a receipt is written once. Later scans may grade results and
refresh the human card; they may never rotate a frozen selection or price.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .calibration import (
    DEFAULT_BRIER_CAP_2WAY,
    DEFAULT_BRIER_CAP_3WAY,
    CalibrationReport,
)
from .models import (
    BookQuote,
    Forecast,
    League,
    format_sast,
    parse_utc,
    sast_cutoff_utc,
    to_utc_z,
    utc_minus_hours,
)

TICKET_SCHEMA = "magolide.paper_ticket_receipt"
PERFORMANCE_SCHEMA = "magolide.paper_ticket_performance"
NO_BET_SCHEMA = "magolide.no_bet_decision"

MIN_LEG_ODDS = 1.20
MAX_LEG_ODDS = 3.00
PRICE_WEIGHT = 0.15
MIN_ROI_SAMPLES = 5
RECEIPT_RE = re.compile(r"^auto_tickets_(\d{4}-\d{2}-\d{2})_decision\.json$")
NO_BET_RE = re.compile(r"^auto_tickets_(\d{4}-\d{2}-\d{2})_no_bet\.json$")


# ---------------------------------------------------------------------------
# Small math helpers
# ---------------------------------------------------------------------------


def expected_value(probability: float, push_probability: float, price: float) -> float:
    """EV per unit: win pays price-1, loss costs 1, push returns the stake."""
    loss = max(1.0 - probability - push_probability, 0.0)
    return probability * (price - 1.0) - loss


def _rank_score(probability: float, price: float, weight: float = PRICE_WEIGHT) -> float:
    if not (0.0 < probability <= 1.0) or price <= 1.0:
        return float("-inf")
    return (1.0 - weight) * math.log(probability) + weight * math.log(price)


# ---------------------------------------------------------------------------
# Quote attachment
# ---------------------------------------------------------------------------


def attach_quotes(forecasts: Iterable[Forecast], quotes: Sequence[BookQuote]) -> List[Forecast]:
    """Attach the best pre-tip named-book quote per exact market/line/selection.

    Post-tip evidence raises ``ValueError`` — it is never silently dropped at
    this boundary, because a dropped price looks like "no edge" instead of
    "the pipeline was late".
    """
    by_key: Dict[Tuple[str, str, Optional[float]], List[BookQuote]] = {}
    for quote in quotes:
        line_key = None if quote.market == "moneyline" else round(float(quote.line or 0), 3)
        by_key.setdefault((quote.game_id, quote.market, line_key), []).append(quote)
    priced: List[Forecast] = []
    for forecast in forecasts:
        line_key = None if forecast.market == "moneyline" else round(float(forecast.line or 0), 3)
        options = [
            quote
            for quote in by_key.get((forecast.game_id, forecast.market, line_key), [])
            if quote.provenance == "named_book" and quote.price_for(forecast.selection)
        ]
        if not options:
            continue
        for quote in options:
            if not quote.pre_match or parse_utc(quote.observed_utc) >= parse_utc(
                forecast.scheduled_utc
            ):
                raise ValueError(
                    f"post-tip quote rejected for {forecast.game_id} {forecast.market} "
                    f"{forecast.selection}: observed {quote.observed_utc} "
                    f">= scheduled {forecast.scheduled_utc}"
                )
        best = max(options, key=lambda quote: quote.price_for(forecast.selection) or 0.0)
        forecast.book = best
        forecast.calibration = dict(forecast.calibration)
        priced.append(forecast)
    return priced


# ---------------------------------------------------------------------------
# Row serialisation (Ma_Golide-compatible candidate/pick rows)
# ---------------------------------------------------------------------------


def _book_row(forecast: Forecast) -> Optional[Dict[str, Any]]:
    book = forecast.book
    if book is None:
        return None
    return {
        "provider": book.provider,
        "book": book.book,
        "observed_utc": book.observed_utc,
        "source_modified_utc": None,
        "pre_match": book.pre_match,
        "main_line": book.main_line,
        "frozen": True,
        "price_provenance": "named_book",
        "source_permission": "unresolved",
        "operational_validation": "probation",
        "game_id": forecast.game_id,
        "market": book.market,
        "period": "full_game",
        "line": book.line,
        "selected_decimal_price": book.price_for(forecast.selection),
        "side_prices": {key: value for key, value in sorted(book.side_prices.items())},
    }


def pick_row(forecast: Forecast, day: str) -> Dict[str, Any]:
    """One frozen pick/leg row. Field set mirrors ``Forecast.to_dict`` + pick keys."""
    price = forecast.decimal_price
    return {
        "forecast_id": forecast.forecast_id,
        "pick_id": forecast.forecast_id,
        "date": day,
        "game_id": forecast.game_id,
        "league": forecast.league,
        "scheduled_utc": forecast.scheduled_utc,
        "home": forecast.home_name,
        "away": forecast.away_name,
        "market": forecast.market,
        "period": forecast.period,
        "selection": forecast.selection,
        "line": forecast.line,
        "role": forecast.role,
        "model": forecast.source,
        "source": forecast.source,
        "raw_probability": round(forecast.raw_probability, 6),
        "calibrated_probability": (
            None
            if forecast.calibrated_probability is None
            else round(forecast.calibrated_probability, 6)
        ),
        "push_probability": round(forecast.push_probability, 6),
        "alternative_probabilities": {
            key: round(value, 6)
            for key, value in sorted(forecast.alternative_probabilities.items())
        },
        "expected_margin": forecast.expected_margin,
        "expected_total": forecast.expected_total,
        "training_games": int(forecast.calibration.get("n") or 0),
        "residual_games": 0,
        "book": _book_row(forecast),
        "prediction": {
            "provider": "espn",
            "model": forecast.source,
            "observed_utc": forecast.book.observed_utc if forecast.book else None,
            "frozen": True,
        },
        "calibration": dict(forecast.calibration),
        "calibration_version": forecast.calibration.get("calibration_version"),
        "eligibility": dict(forecast.eligibility),
        "bucket": "OK",
        "prior_heavy": False,
        "price_evidence": {
            "book": forecast.book.book if forecast.book else None,
            "selected_decimal_price": price,
            "observed_utc": forecast.book.observed_utc if forecast.book else None,
        },
        "grade_status": "pending",
        "outcome": None,
        "paper_only": True,
    }


def row_price(row: Dict[str, Any]) -> Optional[float]:
    book = row.get("book") or {}
    price = book.get("selected_decimal_price")
    if isinstance(price, (int, float)) and not isinstance(price, bool) and price > 1.0:
        return float(price)
    side = (book.get("side_prices") or {}).get(str(row.get("selection") or ""))
    if isinstance(side, (int, float)) and not isinstance(side, bool) and side > 1.0:
        return float(side)
    return None


# ---------------------------------------------------------------------------
# Eligibility gate
# ---------------------------------------------------------------------------


@dataclass
class GatePolicy:
    min_leg_odds: float = MIN_LEG_ODDS
    max_leg_odds: float = MAX_LEG_ODDS
    brier_cap_2way: float = DEFAULT_BRIER_CAP_2WAY
    brier_cap_3way: float = DEFAULT_BRIER_CAP_3WAY
    min_roi_samples: int = MIN_ROI_SAMPLES
    require_named_book: bool = True

    def brier_cap(self, league: League, market: str = "moneyline") -> float:
        """Brier capacity for one market: 3-way only where a draw is priced.

        The cap follows the market's cardinality, not the sport's. A soccer
        ``spread`` or ``total`` is a two-outcome market, so the looser 0.25 cap
        applies; only a 3-way ``moneyline`` earns the stricter 0.22.
        """
        three_way = league.has_draw and market == "moneyline"
        return self.brier_cap_3way if three_way else self.brier_cap_2way

    def to_policy_dict(self) -> Dict[str, Any]:
        return {
            "min_leg_odds": self.min_leg_odds,
            "max_leg_odds": self.max_leg_odds,
            "brier_cap": self.brier_cap_2way,
            "brier_cap_three_way": self.brier_cap_3way,
            "min_roi_samples": self.min_roi_samples,
            "require_named_book": self.require_named_book,
        }


def gate_forecast(
    forecast: Forecast,
    *,
    report: CalibrationReport,
    league: League,
    prior_roi: Optional[Dict[str, Any]],
    policy: GatePolicy = GatePolicy(),
) -> Dict[str, Any]:
    """Stamp ``eligibility`` + ``calibration`` on a forecast; returns the reasons.

    A lane with no settled history at all may admit a strictly gated pick once
    (``cold_start_no_settled_history``); every other missing gate fails closed.
    """
    price = forecast.decimal_price
    probability = report.calibrate(forecast.raw_probability)
    forecast.calibrated_probability = probability
    forecast.calibration = report.to_dict()
    reasons: List[str] = []
    if forecast.book is None or (policy.require_named_book and forecast.book.provenance != "named_book"):
        reasons.append("no_pre_tip_named_book_quote")
    if price is None:
        reasons.append("no_price")
    else:
        if not (policy.min_leg_odds <= price <= policy.max_leg_odds):
            reasons.append("price_outside_ticket_band")
        if expected_value(probability, forecast.push_probability, price) <= 0.0:
            reasons.append("non_positive_expected_value")
    if not report.sufficient:
        reasons.append("calibration_not_sufficient_for_accas")
    capacity = policy.brier_cap(league, str(forecast.market))
    if report.brier is not None and report.brier > capacity:
        reasons.append("brier_above_lane_cap")
    roi_status = (prior_roi or {}).get("status")
    if roi_status == "insufficient_sample":
        reasons.append("prior_roi_insufficient_sample")
    elif roi_status == "cold_start":
        pass  # self-terminating admission; flagged in the row for the card
    elif roi_status is None or (prior_roi or {}).get("roi", 0.0) <= 0.0:
        reasons.append("prior_roi_not_positive")
    cold_start = roi_status == "cold_start"
    eligibility = {
        "acca_eligible": not reasons,
        "reasons": reasons,
        "evidence_state": "walk_forward_calibrated" if report.sufficient else "provisional",
        "cold_start_admitted": cold_start and not reasons,
        "brier_cap": capacity,
        "model_brier": report.brier,
    }
    forecast.eligibility = eligibility
    return eligibility


def score_forecasts(
    forecasts: Iterable[Forecast],
    quotes: Sequence[BookQuote],
    *,
    league: League,
    reports: Dict[Tuple[str, str, str, str], CalibrationReport],
    prior_roi: Optional[Dict[str, Any]],
    policy: GatePolicy = GatePolicy(),
) -> List[Forecast]:
    """Attach quotes, then stamp calibration + eligibility on every forecast.

    Only rows whose segment has a walk-forward report are returned; an
    uncalibrated segment is a coverage gap, not a reason to guess.
    """
    scored: List[Forecast] = []
    for forecast in attach_quotes(forecasts, quotes):
        segment = (forecast.league, forecast.market, forecast.period, forecast.source)
        report = reports.get(segment)
        if report is None:
            continue
        gate_forecast(
            forecast,
            report=report,
            league=league,
            prior_roi=prior_roi,
            policy=policy,
        )
        scored.append(forecast)
    return scored


# ---------------------------------------------------------------------------
# Prior ROI band from settled individual picks
# ---------------------------------------------------------------------------


def roi_band(
    settled: Sequence[Tuple[str, float, str]],
    *,
    as_of_day: str,
    min_samples: int = MIN_ROI_SAMPLES,
    half_life_days: float = 30.0,
) -> Dict[str, Any]:
    """Recency-weighted, shrunk ROI of prior settled individual picks.

    ``settled`` is ``(result_day, decimal_price, outcome)``. Only strictly
    earlier result days count. Zero settled picks in the whole lane is an
    explicit ``cold_start``; a positive-but-tiny sample is
    ``insufficient_sample`` and blocks. Pushes are dropped before weighting:
    a pushed stake returns exactly its own unit, so counting it as a
    zero-profit row drags ``shrunk`` ROI toward 0 and fakes a
    ``prior_roi_not_positive`` lockout.
    """
    rows = [
        (day, price, outcome)
        for day, price, outcome in settled
        if outcome in {"win", "loss"} and day < as_of_day
    ]
    if not rows:
        return {"status": "cold_start", "n": 0, "roi": None, "weight": 0.0}
    weighted_profit = 0.0
    weight_sum = 0.0
    for day, price, outcome in rows:
        age = max(
            (parse_utc(f"{as_of_day}T00:00:00Z") - parse_utc(f"{day}T00:00:00Z")).days, 0
        )
        weight = 0.5 ** (age / half_life_days) if half_life_days > 0 else 1.0
        if outcome == "win":
            weighted_profit += weight * (price - 1.0)
        else:
            weighted_profit -= weight
        weight_sum += weight
    if len(rows) < min_samples:
        return {
            "status": "insufficient_sample",
            "n": len(rows),
            "roi": None,
            "minimum_samples": min_samples,
            "weight": round(weight_sum, 6),
        }
    shrunk = weighted_profit / (weight_sum + 5.0)  # 5 pseudo-units of prior at ROI 0
    return {
        "status": "positive" if shrunk > 0 else "non_positive",
        "n": len(rows),
        "roi": round(shrunk, 6),
        "weight": round(weight_sum, 6),
        "method": "recency_weighted_shrunk_roi",
    }


# ---------------------------------------------------------------------------
# Acca assembly (k-best heap walk, Ma_Golide semantics)
# ---------------------------------------------------------------------------


def _row_probability(row: Dict[str, Any]) -> float:
    value = row.get("calibrated_probability")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _wager_key(row: Dict[str, Any]) -> tuple:
    return (
        row.get("league"),
        row.get("game_id"),
        row.get("market"),
        row.get("period"),
        row.get("selection"),
        row.get("line"),
    )


def _line_marker(value: Any) -> str:
    """A line's identity text. A malformed line must not abort the ROI read."""
    if value is None or isinstance(value, bool):
        return ""
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def _fixture_key(row: Dict[str, Any]) -> tuple:
    return (row.get("league"), row.get("game_id"))


def eligible_rows(
    rows: Sequence[Dict[str, Any]], *, policy: GatePolicy = GatePolicy()
) -> List[Dict[str, Any]]:
    """Every usable wager: eligibility passed, priced inside the band.

    Collapsing to one row per fixture here was wrong. The surviving row was
    almost always a heavy moneyline favourite (``@1.25-@1.45``), so two legs
    could not reach the ``@3.00`` acca floor and the lane starved — while the
    ``@1.90-@2.10`` spread/total legs that would have reached it were discarded
    as "the same game". Same-game exclusivity belongs in :func:`build_accas`,
    which assembles the combination and can enforce it per acca.
    """
    by_wager: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in rows:
        if not (row.get("eligibility") or {}).get("acca_eligible"):
            continue
        price = row_price(row)
        probability = _row_probability(row)
        if price is None or not (policy.min_leg_odds <= price <= policy.max_leg_odds):
            continue
        if not 0.0 < probability <= 1.0:
            continue
        key = _wager_key(row)
        current = by_wager.get(key)
        # A duplicate wager (same fixture/market/selection/line) can only come
        # from a second model surface. Keep the more confident row instead of
        # whichever happened to arrive first.
        if current is None or probability > _row_probability(current):
            by_wager[key] = row
    return sorted(
        by_wager.values(),
        key=lambda row: (
            -_row_probability(row),
            str(row.get("league")),
            str(row.get("game_id")),
            str(row.get("market")),
            str(row.get("line")),
            str(row.get("forecast_id")),
        ),
    )


def _neighbors(state: Tuple[int, ...], population: int) -> Iterable[Tuple[int, ...]]:
    for index in range(len(state)):
        if index == 0 or state[index - 1] > state[index] + 1:
            if state[index] + 1 < population:
                yield state[:index] + (state[index] + 1,) + state[index + 1 :]


def build_accas(
    rows: Sequence[Dict[str, Any]],
    *,
    legs: int = 2,
    limit: int = 3,
    price_weight: float = PRICE_WEIGHT,
    min_leg_odds: float = MIN_LEG_ODDS,
    max_leg_odds: float = MAX_LEG_ODDS,
    min_combined_odds: Optional[float] = None,
    max_combined_odds: Optional[float] = None,
    require_disjoint_legs: bool = True,
    forbid_conflicting_game_sides: bool = True,
) -> List[Dict[str, Any]]:
    """Return the best-ranked accas over already-eligible rows."""
    if legs < 2 or legs > 6:
        raise ValueError("acca legs must be between two and six")
    if limit < 1:
        raise ValueError("limit must be positive")

    def _build(pool: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        candidates = []
        for row in pool:
            price = row_price(row)
            probability = _row_probability(row)
            if price is None or probability is None:
                continue
            if not (min_leg_odds <= price <= max_leg_odds) or not 0.0 < probability <= 1.0:
                continue
            candidates.append(row)
        if len(candidates) < legs:
            return []
        candidates.sort(
            key=lambda row: (
                -_rank_score(_row_probability(row), float(row_price(row) or 0.0), price_weight),
                str(row.get("game_id")),
                str(row.get("forecast_id")),
            )
        )
        costs = [
            -_rank_score(_row_probability(row), float(row_price(row) or 0.0), price_weight)
            for row in candidates
        ]
        def _ticket(
            selected: Sequence[Dict[str, Any]], cost: float, price: float, rank: int
        ) -> Dict[str, Any]:
            probability = math.exp(sum(math.log(_row_probability(row)) for row in selected))
            leg_ids = [str(row.get("forecast_id")) for row in selected]
            return {
                "acca_id": hashlib.sha256("|".join(leg_ids).encode()).hexdigest()[:20],
                "rank": rank,
                "legs": [dict(row) for row in selected],
                "leg_count": legs,
                "estimated_joint_win_probability": round(probability, 8),
                "combined_decimal_odds": round(price, 4),
                "bookmaker_implied_probability": round(1.0 / price, 8),
                "estimated_ev_per_unit": round(probability * price - 1.0, 8),
                "ranking_metric": "probability_price_blend",
                "ranking_price_weight": price_weight,
                "ranking_score": round(-cost, 8),
                "ranking_status": "paper_only_strict_gates_passed",
                "provisional_sample_gate_bypassed": False,
                "provisional_leg_count": 0,
                "dependence": {
                    "same_game_legs": "prohibited",
                    "cross_game_model": "conditional_independence",
                    "limitation": (
                        "Shared league, weather, scheduling, or other latent "
                        "dependence is not estimated; no same-game parlay is built."
                    ),
                },
            }

        def _usable(selected: Sequence[Dict[str, Any]], price: float) -> bool:
            if not (distinct_fixtures_ok(selected)):
                return False
            if min_combined_odds is not None and price < min_combined_odds:
                return False
            if max_combined_odds is not None and price > max_combined_odds:
                return False
            return True

        def distinct_fixtures_ok(selected: Sequence[Dict[str, Any]]) -> bool:
            if not forbid_conflicting_game_sides:
                return True
            return len({_fixture_key(row) for row in selected}) == len(selected)

        def _pair_tickets() -> List[Dict[str, Any]]:
            """Every two-leg combination, ranked. Complete by construction.

            A best-first heap cannot be both complete and bounded here: on a
            67-game CFB board the below-band favourite pairs ahead of the first
            in-band spread/total pair number 11,000+, so the old 5,000-pop
            budget returned zero accas on exactly the slates the product was
            built for. With two legs the state space is quadratic and cheap
            (201 rows -> 20k pairs), so it is enumerated instead of searched.
            """
            ranked: List[Tuple[float, int, int, float]] = []
            for first in range(len(candidates) - 1):
                first_row = candidates[first]
                for second in range(first + 1, len(candidates)):
                    second_row = candidates[second]
                    price = float(row_price(first_row) or 1.0) * float(
                        row_price(second_row) or 1.0
                    )
                    if not _usable([first_row, second_row], price):
                        continue
                    ranked.append((costs[first] + costs[second], first, second, price))
            ranked.sort(
                key=lambda item: (
                    item[0],
                    str(candidates[item[1]].get("game_id")),
                    str(candidates[item[1]].get("forecast_id")),
                    str(candidates[item[2]].get("forecast_id")),
                )
            )
            return [
                _ticket([candidates[first], candidates[second]], cost, price, rank)
                for rank, (cost, first, second, price) in enumerate(ranked[:limit], start=1)
            ]

        if legs == 2:
            return _pair_tickets()

        initial = tuple(range(legs))
        heap: List[Tuple[float, Tuple[int, ...]]] = [
            (sum(costs[i] for i in initial), initial)
        ]
        visited = {initial}
        output: List[Dict[str, Any]] = []
        # Three or more legs cannot be enumerated, so the walk stays bounded —
        # but the budget has to scale with the pool, not with the pool the old
        # one-row-per-game filter produced.
        pop_budget = max(5000, 200 * len(candidates))
        popped = 0
        while heap and len(output) < limit:
            if popped >= pop_budget:
                break
            cost, state = heapq.heappop(heap)
            popped += 1
            selected = [candidates[index] for index in state]
            price = math.prod(float(row_price(row) or 1.0) for row in selected)
            # Two legs from one fixture are one bet, not a parlay: keep walking
            # the heap instead of emitting a correlated acca.
            if _usable(selected, price):
                output.append(_ticket(selected, cost, price, len(output) + 1))
            for neighbor in _neighbors(state, len(candidates)):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                heapq.heappush(heap, (sum(costs[i] for i in neighbor), neighbor))
        return output

    if not require_disjoint_legs:
        return _build(rows)

    remaining = list(rows)
    used_wagers: set[tuple] = set()
    used_sides: set[tuple] = set()
    output: List[Dict[str, Any]] = []
    while len(output) < limit:
        pool = [
            row
            for row in remaining
            if _wager_key(row) not in used_wagers
            and (
                not forbid_conflicting_game_sides
                or (row.get("game_id"), row.get("market"), row.get("selection")) not in used_sides
            )
        ]
        best = _build(pool)
        if not best:
            break
        ticket = best[0]
        ticket["rank"] = len(output) + 1
        ticket["ticket_leg_reuse_policy"] = (
            "no_reused_wager_selections_or_opposing_game_sides"
            if forbid_conflicting_game_sides
            else "no_reused_wager_selections"
        )
        output.append(ticket)
        for leg in ticket["legs"]:
            used_wagers.add(_wager_key(leg))
            if forbid_conflicting_game_sides:
                used_sides.add((leg.get("game_id"), leg.get("market"), leg.get("selection")))
    return output


# ---------------------------------------------------------------------------
# Decision freezing
# ---------------------------------------------------------------------------


def decision_cutoff(league: League, day: str, scheduled_utc: Iterable[str]) -> Optional[str]:
    """Lane cutoff: the SAST publication hour, pulled earlier for early tips."""
    tips = [to_utc_z(value) for value in scheduled_utc]
    if not tips:
        return None
    base = sast_cutoff_utc(day, league.cutoff_hour_sast)
    earliest_tip = min(tips)
    bounded = utc_minus_hours(earliest_tip, 2.0)
    return min(base, bounded)


def _write_once(path: Path, payload: Dict[str, Any]) -> bool:
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return True


def freeze_receipt(
    localdata: Path,
    *,
    league: League,
    day: str,
    captured_utc: str,
    candidate_pool: List[Dict[str, Any]],
    accas: List[Dict[str, Any]],
    singles: Optional[List[Dict[str, Any]]] = None,
    selection_policy: Optional[Dict[str, Any]] = None,
    decision_state: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Write the day's immutable receipt once; return it (existing or new).

    Returns ``None`` when a terminal no-bet lock already exists for the day.
    """
    lane_dir = localdata / league.lane
    lane_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = lane_dir / f"auto_tickets_{day}_decision.json"
    if receipt_path.exists():
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    no_bet_path = lane_dir / f"auto_tickets_{day}_no_bet.json"
    if no_bet_path.exists():
        return None
    decision_id = hashlib.sha256(
        f"{league.lane}|{day}|{captured_utc}".encode()
    ).hexdigest()[:20]
    if not accas and not singles:
        # The pool travels with the no-bet lock: it is the only evidence of what
        # the model saw on a day it declined to bet, and settling it is what
        # lets ``prior_roi`` accrue instead of deadlocking below five picks.
        _write_once(
            no_bet_path,
            {
                "schema_version": NO_BET_SCHEMA,
                "status": "frozen_no_bet",
                "league": league.receipt_league,
                "date": day,
                "target_date": day,
                "decision_id": decision_id,
                "captured_at": captured_utc,
                "frozen_at": captured_utc,
                "paper_only": True,
                "reason": "no_qualifying_pre_tip_ticket_at_decision_cutoff",
                "ticket_decision": decision_state or {},
                "candidate_pool": candidate_pool,
                "selection_policy": selection_policy or {},
                "positions": [],
            },
        )
        return None
    ticket_rows: Dict[str, List[Dict[str, Any]]] = {"acca": [], "single": []}
    for index, ticket in enumerate(
        [*accas, *(singles or [])], start=1
    ):
        ticket_type = "single" if ticket.get("leg_count") == 1 else "acca"
        legs = [dict(leg) for leg in ticket["legs"]]
        row = {
            "ticket_id": ticket.get("ticket_id") or ticket["acca_id"],
            "ticket_type": ticket_type,
            "publication_index": index,
            "leg_count": int(ticket.get("leg_count") or len(legs)),
            "combined_decimal_odds": ticket["combined_decimal_odds"],
            "stake_units": 1.0,
            "estimated_joint_win_probability": ticket[
                "estimated_joint_win_probability"
            ],
            "paper_only": True,
            "provisional_sample_gate_bypassed": False,
            "provisional_leg_count": 0,
            "ranking_status": ticket.get("ranking_status"),
            "bookmakers": sorted(
                {
                    (leg.get("book") or {}).get("book")
                    for leg in legs
                    if (leg.get("book") or {}).get("book")
                }
            ),
            "legs": legs,
        }
        if ticket_type == "single":
            row["estimated_win_probability"] = ticket["estimated_joint_win_probability"]
        ticket_rows[ticket_type].append(row)
    receipt = {
        "schema_version": TICKET_SCHEMA,
        "date": day,
        "league": league.receipt_league,
        "decision_id": decision_id,
        "status": "frozen",
        "paper_only": True,
        "captured_at": captured_utc,
        "frozen_at": captured_utc,
        "ticket_decision": decision_state
        or {
            "status": "frozen",
            "planned_cutoff_utc": None,
            "candidate_count": len(candidate_pool),
        },
        "stake_units_per_acca": 1.0,
        "stake_units_per_position": 1.0,
        "accas": ticket_rows["acca"],
        "singles": ticket_rows["single"],
        "candidate_pool": candidate_pool,
        "selection_policy": selection_policy or {},
    }
    _write_once(receipt_path, receipt)
    return json.loads(receipt_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Grading and settlement
# ---------------------------------------------------------------------------


def resolve_outcome(
    market: str,
    selection: str,
    line: Optional[float],
    home: float,
    away: float,
    *,
    has_draw: bool = True,
) -> Optional[str]:
    """Pure settlement rule for one selection against a final score.

    ``has_draw`` is the market's cardinality, not the sport's: a soccer 1X2
    moneyline grades a level game as a win for ``draw`` and a loss for both
    sides, while a two-way moneyline (NFL/CFB, where no draw price exists) voids
    to ``"push"``. Grading a 2-way tie as a loss would book a real stake loss on
    a market the book returns.
    """
    if market == "moneyline":
        if home == away and not has_draw:
            return "push"
        result = "home" if home > away else "away" if away > home else "draw"
        return "win" if selection == result else "loss"
    if not isinstance(line, (int, float)) or isinstance(line, bool):
        return None
    if market == "total":
        difference = (home + away) - float(line)
        decisive = "over" if difference > 0 else "under" if difference < 0 else "push"
        return "push" if decisive == "push" else ("win" if selection == decisive else "loss")
    if market == "spread":
        cover = home + float(line) - away
        decisive = "home" if cover > 0 else "away" if cover < 0 else "push"
        return "push" if decisive == "push" else ("win" if selection == decisive else "loss")
    return None


def grade_row(row: Dict[str, Any], game: Any, *, has_draw: bool = True) -> Optional[str]:
    """Grade one frozen row against a reconciled final. Never guesses."""
    actual = getattr(game, "actual", None)
    if actual is None or getattr(game, "status", None) is None:
        return None
    if getattr(game.status, "value", game.status) != "final":
        return None
    return resolve_outcome(
        str(row.get("market") or ""),
        str(row.get("selection") or ""),
        row.get("line"),
        float(actual.home_score),
        float(actual.away_score),
        has_draw=has_draw,
    )


def _positions(receipt: Dict[str, Any]) -> List[Dict[str, Any]]:
    positions: List[Dict[str, Any]] = []
    for collection, default_type in (("accas", "acca"), ("singles", "single")):
        for row in receipt.get(collection) or []:
            if isinstance(row, dict):
                row.setdefault("ticket_type", default_type)
                positions.append(row)
    return positions


def _settle_position(position: Dict[str, Any]) -> None:
    outcomes = [leg.get("outcome") for leg in position.get("legs") or []]
    position["leg_outcomes"] = outcomes
    position["push_legs"] = sum(1 for item in outcomes if item == "push")
    position["unresolved_legs"] = sum(1 for item in outcomes if item is None)
    effective_odds = 1.0
    for leg, outcome in zip(position.get("legs") or [], outcomes):
        if outcome == "push":
            continue
        price = row_price(leg)
        if price is None:
            position["grade_status"] = "pending"
            position["profit_units"] = None
            return
        effective_odds *= price
    position["effective_decimal_odds"] = round(effective_odds, 4)
    stake = float(position.get("stake_units") or 1.0)
    if any(item == "loss" for item in outcomes):
        position["grade_status"] = "loss"
        position["profit_units"] = -stake
    elif any(item is None for item in outcomes):
        position["grade_status"] = "pending"
        position["profit_units"] = None
    elif outcomes and all(item == "push" for item in outcomes):
        position["grade_status"] = "push"
        position["profit_units"] = 0.0
    else:
        position["grade_status"] = "win"
        position["profit_units"] = round(stake * (effective_odds - 1.0), 6)


def _grade_pool(
    receipt: Dict[str, Any], games_by_id: Dict[str, Any], *, has_draw: bool = True
) -> Dict[str, str]:
    """Grade a frozen candidate pool: ``forecast_id -> win|loss|push``."""
    pool_outcomes: Dict[str, str] = {}
    for row in receipt.get("candidate_pool") or []:
        if not isinstance(row, dict):
            continue
        game = games_by_id.get(str(row.get("game_id")))
        outcome = grade_row(row, game, has_draw=has_draw) if game is not None else None
        if outcome in {"win", "loss", "push"}:
            pool_outcomes[str(row.get("forecast_id"))] = outcome
    return pool_outcomes


def settle_lane(
    localdata: Path, league: League, games_by_id: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Regrade every frozen receipt (ticket *and* no-bet) in the lane."""
    lane_dir = localdata / league.lane
    if not lane_dir.exists():
        return None
    history: List[Dict[str, Any]] = []
    for path in sorted(lane_dir.iterdir()):
        receipt_match = RECEIPT_RE.match(path.name)
        no_bet_match = NO_BET_RE.match(path.name)
        if not (receipt_match or no_bet_match):
            continue
        receipt = json.loads(path.read_text(encoding="utf-8"))
        schema = receipt.get("schema_version")
        if no_bet_match and schema != NO_BET_SCHEMA:
            continue
        if receipt_match and schema != TICKET_SCHEMA:
            continue
        pool_outcomes = _grade_pool(receipt, games_by_id, has_draw=league.has_draw)
        positions: List[Dict[str, Any]] = []
        # A no-bet lock has no positions by definition, so its graded pool moves
        # ticket P&L by exactly nothing while still accruing ROI evidence.
        for position in _positions(receipt) if schema == TICKET_SCHEMA else []:
            for leg in position.get("legs") or []:
                game = games_by_id.get(str(leg.get("game_id")))
                outcome = (
                    grade_row(leg, game, has_draw=league.has_draw)
                    if game is not None
                    else None
                )
                leg["outcome"] = outcome
                leg["grade_status"] = outcome or "pending"
            _settle_position(position)
            positions.append(position)
        day = (
            receipt.get("date")
            or receipt.get("target_date")
            or (receipt_match or no_bet_match).group(1)
        )
        history.append(
            {
                "date": day,
                "league": receipt.get("league"),
                "decision_id": receipt.get("decision_id") or "",
                "frozen_at": receipt.get("frozen_at"),
                "status": "frozen_no_bet" if schema == NO_BET_SCHEMA else "frozen",
                "positions": positions,
                "candidate_pool_outcomes": pool_outcomes,
            }
        )
    settled = [pos for day in history for pos in day["positions"] if pos["profit_units"] is not None]
    pending = [pos for day in history for pos in day["positions"] if pos["profit_units"] is None]
    profit = sum(float(pos["profit_units"]) for pos in settled)
    staked = sum(float(pos["stake_units"]) for pos in settled)
    wins = sum(pos["grade_status"] == "win" for pos in settled)
    losses = sum(pos["grade_status"] == "loss" for pos in settled)
    report = {
        "schema_version": PERFORMANCE_SCHEMA,
        "league": league.receipt_league,
        "settled_positions": len(settled),
        "pending_positions": len(pending),
        "wins": wins,
        "losses": losses,
        "pushes": sum(pos["grade_status"] == "push" for pos in settled),
        "hit_rate": (wins / (wins + losses)) if (wins + losses) else None,
        "settled_stake_units": staked,
        "profit_units": round(profit, 6),
        "roi": (profit / staked) if staked else None,
        "stake_policy": "flat_one_paper_unit_per_position_no_refill_no_real_money",
        "low_sample_flag": len(settled) < 30,
        "low_sample_note": (
            "[LOW SAMPLE: n < 30] no profit, edge, or readiness claim is supported"
            if len(settled) < 30
            else "sample >= 30 settled positions"
        ),
        "history": history,
    }
    lane_dir.mkdir(parents=True, exist_ok=True)
    (lane_dir / "auto_tickets_performance.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _pool_band(policy: Dict[str, Any]) -> Tuple[float, float]:
    """The ticket odds band the day's pool was frozen under."""

    def _price(key: str, fallback: float) -> float:
        value = policy.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 1.0:
            return float(value)
        return fallback

    return _price("min_leg_odds", MIN_LEG_ODDS), _price("max_leg_odds", MAX_LEG_ODDS)


def settled_pick_history(localdata: Path, league: League) -> List[Tuple[str, float, str]]:
    """(result_day, price, outcome) for settled *model-favoured* pool picks.

    ``candidate_pool`` deliberately contains both sides of every market, so
    feeding the whole pool to :func:`roi_band` measures the bookmaker's vig
    (about -4.5%) and would trip ``prior_roi_not_positive`` forever. Only the
    side the model made +EV, priced inside the frozen ticket band, counts — and
    at most one row per ``(game, market)`` so opposing sides cannot cancel out.

    No-bet locks contribute their graded pool too: otherwise a small slate that
    settles fewer than ``MIN_ROI_SAMPLES`` picks blocks the next day, which
    freezes another no-bet, which settles nothing, forever.
    """
    lane_dir = localdata / league.lane
    perf_path = lane_dir / "auto_tickets_performance.json"
    if not perf_path.exists():
        return []
    report = json.loads(perf_path.read_text(encoding="utf-8"))
    receipts: Dict[str, Dict[str, Any]] = {}
    for pattern in ("auto_tickets_*_decision.json", "auto_tickets_*_no_bet.json"):
        for path in lane_dir.glob(pattern):
            match = RECEIPT_RE.match(path.name) or NO_BET_RE.match(path.name)
            if not match:
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") in {TICKET_SCHEMA, NO_BET_SCHEMA}:
                receipts[match.group(1)] = payload
    rows: List[Tuple[str, float, str]] = []
    # A postponed fixture can be frozen into two consecutive pools and settle
    # against the same final; counting both would weight one intended bet twice.
    seen: set[Tuple[Any, ...]] = set()
    history = sorted(
        (entry for entry in report.get("history") or [] if isinstance(entry, dict)),
        key=lambda entry: str(entry.get("date") or ""),
    )
    for day in history:
        date = str(day.get("date") or "")
        receipt = receipts.get(date)
        if receipt is None:
            continue
        low, high = _pool_band(dict(receipt.get("selection_policy") or {}))
        outcomes = {
            str(key): str(value)
            for key, value in (day.get("candidate_pool_outcomes") or {}).items()
        }
        favoured: Dict[Tuple[Any, ...], Tuple[float, float, str, Tuple[Any, ...]]] = {}
        for row in receipt.get("candidate_pool") or []:
            if not isinstance(row, dict):
                continue
            forecast_id = str(row.get("forecast_id"))
            outcome = outcomes.get(forecast_id)
            if outcome not in {"win", "loss", "push"}:
                continue
            price = row_price(row)
            if price is None or not (low <= price <= high):
                continue
            probability = row.get("calibrated_probability")
            if not isinstance(probability, (int, float)) or isinstance(probability, bool):
                probability = row.get("raw_probability")
            if not isinstance(probability, (int, float)) or isinstance(probability, bool):
                continue
            push = row.get("push_probability")
            push_probability = float(push) if isinstance(push, (int, float)) else 0.0
            value = expected_value(float(probability), push_probability, price)
            if value <= 0.0:
                continue
            marker = (
                str(row.get("game_id")),
                str(row.get("market")),
                str(row.get("period")),
                str(row.get("selection")),
                _line_marker(row.get("line")),
            )
            key = (str(row.get("game_id")), str(row.get("market")))
            best = favoured.get(key)
            if best is None or value > best[0]:
                favoured[key] = (value, price, outcome, marker)
        for value, price, outcome, marker in favoured.values():
            # Postponed fixtures get forecast and frozen again the next day and
            # settle against the same final. Keep the earliest receipt only, so
            # one intended bet cannot weight the ROI band twice.
            if marker in seen:
                continue
            seen.add(marker)
            rows.append((date, price, outcome))
    return rows


# ---------------------------------------------------------------------------
# Card rendering
# ---------------------------------------------------------------------------


def render_card(receipt: Dict[str, Any]) -> str:
    lines = [
        f"BULLION PAPER CARD — {receipt.get('league')} — {receipt.get('date')}",
        f"frozen {format_sast(receipt.get('frozen_at'))} · paper only · stake 1.00u/position",
        "",
    ]
    positions = _positions(receipt)
    if not positions:
        lines.append("No qualifying ticket was frozen at the decision cutoff.")
    for position in positions:
        lines.append(
            f"[{position.get('ticket_type','acca').upper()}] @{position['combined_decimal_odds']:.2f} "
            f"· joint p={position['estimated_joint_win_probability']:.3f} "
            f"· {position['leg_count']} legs"
        )
        for leg in position.get("legs") or []:
            price = row_price(leg) or 0.0
            book = (leg.get("book") or {}).get("book") or "?"
            probability = leg.get("calibrated_probability")
            lines.append(
                f"  - {format_sast(leg.get('scheduled_utc'))} | "
                f"{leg.get('away')} @ {leg.get('home')} | "
                f"{str(leg.get('market')).upper()} {leg.get('line') if leg.get('line') is not None else ''}"
                f" -> {str(leg.get('selection')).upper()} @{price:.2f} ({book})"
                f"{'' if probability is None else f' p={probability:.3f}'}"
            )
        lines.append("")
    policy = receipt.get("selection_policy") or {}
    lines.append(
        "gates: named-book pre-tip only · calibrated n>="
        f"{policy.get('min_samples', 30)} · brier<={policy.get('brier_cap')} · "
        f"legs {policy.get('min_leg_odds')}-{policy.get('max_leg_odds')} · limit {policy.get('limit')}"
    )
    lines.append("No profit, edge, or readiness claim is made from an unsettled paper card.")
    return "\n".join(lines) + "\n"
