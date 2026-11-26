"""Live-faithful replay and paired A/B over frozen decision-time pools.

The replay never re-implements live selection rules: it calls the same
``build_accas`` the daily runner calls, over the ``candidate_pool`` that was
frozen at the cutoff. ``--parity`` proves that by rebuilding every receipt and
diffing ticket signatures.

Honesty rules baked in:
* a receipt without a frozen pool is skipped (post-tip re-scoring must not
  masquerade as a live pool);
* every summary prints ``n`` settled bet-days and flags ``n < 30``;
* a paired A/B that would select identical tickets is a NO-OP and is refused;
* zero settled days is reported as zero, never as a flat equity curve.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .calibration import DEFAULT_BRIER_CAP_2WAY
from .tickets import (
    MAX_LEG_ODDS,
    MIN_LEG_ODDS,
    PRICE_WEIGHT,
    GatePolicy,
    build_accas,
    eligible_rows,
    row_price,
)

RECEIPT_RE = re.compile(r"^auto_tickets_(\d{4}-\d{2}-\d{2})_decision\.json$")
LOW_SAMPLE_DAYS = 30
STAKE_MODES = frozenset({"flat", "daily_pct", "kelly", "skim"})


@dataclass(frozen=True)
class ReplaySpec:
    legs: Optional[int] = None
    limit: Optional[int] = None
    min_leg_odds: Optional[float] = None
    max_leg_odds: Optional[float] = None
    price_weight: Optional[float] = None
    min_combined_odds: Optional[float] = None
    max_combined_odds: Optional[float] = None
    stake_mode: str = "flat"
    daily_pct: float = 0.25
    kelly_fraction: float = 0.25
    slots: int = 3

    def __post_init__(self) -> None:
        if self.stake_mode not in STAKE_MODES:
            raise ValueError(f"unknown stake mode {self.stake_mode!r}")
        if self.daily_pct <= 0 or self.daily_pct > 1:
            raise ValueError("daily_pct must be in (0, 1]")
        if self.kelly_fraction <= 0 or self.kelly_fraction > 1:
            raise ValueError("kelly_fraction must be in (0, 1]")


@dataclass
class DayEvidence:
    date: str
    lane: str
    league: str
    decision_id: str
    frozen_at: str
    receipt: Dict[str, Any]
    candidate_pool: List[Dict[str, Any]]
    selection_policy: Dict[str, Any]
    outcomes_by_fid: Dict[str, str]

    @property
    def has_pool(self) -> bool:
        return bool(self.candidate_pool)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def discovery_dirs(localdata: Path) -> List[Path]:
    if not localdata.exists():
        return []
    if any(RECEIPT_RE.match(path.name) for path in localdata.iterdir() if path.is_file()):
        return [localdata]
    return [
        child
        for child in sorted(localdata.iterdir())
        if child.is_dir()
        and any(RECEIPT_RE.match(path.name) for path in child.iterdir() if path.is_file())
    ]


def _outcomes(lane_dir: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    path = lane_dir / "auto_tickets_performance.json"
    if not path.exists():
        return {}
    try:
        report = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return {}
    by_key: Dict[Tuple[str, str], Dict[str, str]] = {}
    for day in report.get("history") or []:
        if not isinstance(day, dict):
            continue
        key = (str(day.get("date") or ""), str(day.get("decision_id") or ""))
        outcomes = by_key.setdefault(key, {})
        pool_outcomes = day.get("candidate_pool_outcomes")
        if isinstance(pool_outcomes, dict):
            for fid, outcome in pool_outcomes.items():
                if outcome in {"win", "loss", "push"}:
                    outcomes[str(fid)] = str(outcome)
        for position in day.get("positions") or []:
            for leg in position.get("legs") or []:
                fid = str(leg.get("forecast_id") or leg.get("pick_id") or "")
                outcome = leg.get("outcome") or leg.get("grade_status")
                if fid and outcome in {"win", "loss", "push"}:
                    outcomes[fid] = str(outcome)
    return by_key


def load_evidence(
    localdata: Path,
    *,
    lanes: Optional[Sequence[str]] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
) -> List[DayEvidence]:
    """Load immutable receipts across lanes; skip pool-less legacy receipts."""
    allowed = {item.strip().lower() for item in (lanes or ()) if item.strip()}
    evidence: List[DayEvidence] = []
    for lane_dir in discovery_dirs(localdata):
        lane = lane_dir.name.lower()
        if allowed and lane not in allowed:
            continue
        outcomes = _outcomes(lane_dir)
        for path in sorted(lane_dir.iterdir()):
            match = RECEIPT_RE.match(path.name)
            if not match:
                continue
            date = match.group(1)
            if since and date < since:
                continue
            if until and date > until:
                continue
            receipt = _read_json(path)
            if not isinstance(receipt, dict):
                continue
            if receipt.get("schema_version") != "magolide.paper_ticket_receipt":
                continue
            pool = [row for row in receipt.get("candidate_pool") or [] if isinstance(row, dict)]
            if not pool:
                continue
            decision_id = str(receipt.get("decision_id") or "")
            evidence.append(
                DayEvidence(
                    date=date,
                    lane=lane,
                    league=str(receipt.get("league") or lane.upper()),
                    decision_id=decision_id,
                    frozen_at=str(receipt.get("frozen_at") or ""),
                    receipt=receipt,
                    candidate_pool=pool,
                    selection_policy=dict(receipt.get("selection_policy") or {}),
                    outcomes_by_fid=dict(outcomes.get((date, decision_id), {})),
                )
            )
    evidence.sort(key=lambda day: (day.date, day.league, day.decision_id))
    return evidence


def _policy_limits(day: DayEvidence, spec: ReplaySpec) -> Dict[str, Any]:
    policy = day.selection_policy
    return {
        "legs": spec.legs or int(policy.get("requested_legs") or 2),
        "limit": spec.limit or int(policy.get("limit") or 3),
        "min_leg_odds": spec.min_leg_odds
        if spec.min_leg_odds is not None
        else float(policy.get("min_leg_odds") or MIN_LEG_ODDS),
        "max_leg_odds": spec.max_leg_odds
        if spec.max_leg_odds is not None
        else float(policy.get("max_leg_odds") or MAX_LEG_ODDS),
        "price_weight": spec.price_weight
        if spec.price_weight is not None
        else float(policy.get("price_weight") or PRICE_WEIGHT),
        "min_combined_odds": spec.min_combined_odds
        if spec.min_combined_odds is not None
        else policy.get("min_combined_odds"),
        "max_combined_odds": spec.max_combined_odds
        if spec.max_combined_odds is not None
        else policy.get("max_combined_odds"),
        "forbid_conflicting": bool(policy.get("forbid_conflicting_game_sides", True)),
    }


def select_tickets(day: DayEvidence, spec: ReplaySpec = ReplaySpec()) -> List[Dict[str, Any]]:
    limits = _policy_limits(day, spec)
    rows = eligible_rows(
        day.candidate_pool,
        policy=GatePolicy(
            min_leg_odds=limits["min_leg_odds"],
            max_leg_odds=limits["max_leg_odds"],
            brier_cap_2way=DEFAULT_BRIER_CAP_2WAY,
        ),
    )
    if len(rows) < limits["legs"]:
        return []
    return build_accas(
        rows,
        legs=limits["legs"],
        limit=limits["limit"],
        price_weight=limits["price_weight"],
        min_leg_odds=limits["min_leg_odds"],
        max_leg_odds=limits["max_leg_odds"],
        min_combined_odds=limits["min_combined_odds"],
        max_combined_odds=limits["max_combined_odds"],
        forbid_conflicting_game_sides=limits["forbid_conflicting"],
    )


def _signature(
    tickets: Sequence[Dict[str, Any]], *, with_id: bool = True
) -> List[Tuple[str, ...]]:
    signatures: List[Tuple[str, ...]] = []
    for ticket in tickets:
        legs = tuple(str(leg.get("forecast_id")) for leg in ticket.get("legs") or [])
        if with_id:
            signatures.append((str(ticket.get("acca_id") or ticket.get("ticket_id")), *legs))
        else:
            signatures.append(legs)
    return signatures


def _decimal_token(value: Any) -> str:
    """Stable price token: frozen and rebuilt prices must match exactly."""
    if value is None:
        return ""
    try:
        return f"{round(float(value), 6):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _receipt_signature(receipt: Dict[str, Any]) -> List[Tuple[str, ...]]:
    signatures: List[Tuple[str, ...]] = []
    for collection in ("accas", "singles"):
        for position in receipt.get(collection) or []:
            if not isinstance(position, dict):
                continue
            signatures.append(
                (
                    str(position.get("ticket_id") or ""),
                    _decimal_token(position.get("combined_decimal_odds")),
                    tuple(
                        f"{leg.get('forecast_id') or leg.get('pick_id') or ''}@"
                        f"{_decimal_token(leg.get('decimal_odds'))}"
                        for leg in position.get("legs") or []
                    ),
                )
            )
    return signatures


def check_parity(days: Sequence[DayEvidence]) -> Dict[str, Any]:
    checked = matched = 0
    mismatches: List[Dict[str, Any]] = []
    for day in days:
        checked += 1
        rebuilt = select_tickets(day, ReplaySpec())
        expected = _receipt_signature(day.receipt)
        actual = [
            (
                str(ticket.get("acca_id")),
                _decimal_token(ticket.get("combined_decimal_odds")),
                tuple(
                    f"{leg.get('forecast_id')}@{_decimal_token(leg.get('decimal_odds'))}"
                    for leg in ticket.get("legs") or []
                ),
            )
            for ticket in rebuilt
        ]
        if expected == actual:
            matched += 1
        else:
            mismatches.append(
                {"date": day.date, "lane": day.lane, "expected": expected, "rebuilt": actual}
            )
    return {
        "checked": checked,
        "matched": matched,
        "mismatched": len(mismatches),
        # Vacuous truth: with no frozen receipt there is nothing that can
        # disagree, so parity holds. ``vacuous`` says so out loud, because
        # "checked: 0, parity: true" must never read as "verified on real days".
        "parity": not mismatches,
        "vacuous": checked == 0,
        "mismatches": mismatches,
    }


def _position_result(ticket: Dict[str, Any], outcomes: Dict[str, str]) -> Optional[Dict[str, Any]]:
    leg_outcomes = [
        outcomes.get(str(leg.get("forecast_id"))) for leg in ticket.get("legs") or []
    ]
    if any(outcome is None for outcome in leg_outcomes):
        return None
    if any(outcome == "loss" for outcome in leg_outcomes):
        return {"status": "loss", "profit_odds": 0.0, "effective_odds": 0.0}
    effective = 1.0
    for leg, outcome in zip(ticket.get("legs") or [], leg_outcomes):
        if outcome == "push":
            continue
        price = row_price(leg)
        if price is None:
            return None
        effective *= price
    if all(outcome == "push" for outcome in leg_outcomes):
        return {"status": "push", "profit_odds": 1.0, "effective_odds": 1.0}
    return {"status": "win", "effective_odds": effective, "profit_odds": effective}


def _stake(
    spec: ReplaySpec, ticket: Dict[str, Any], bank: float, slots: int
) -> float:
    if spec.stake_mode == "flat":
        return 1.0
    if spec.stake_mode == "skim":
        return 0.1
    bank_slice = bank * spec.daily_pct / max(slots, 1)
    if spec.stake_mode == "daily_pct":
        return min(bank_slice, bank)
    probability = float(ticket.get("estimated_joint_win_probability") or 0.0)
    price = float(ticket.get("combined_decimal_odds") or 1.0)
    if probability <= 0 or price <= 1.0:
        return 0.0
    edge = probability * price - 1.0
    if edge <= 0:
        return 0.0
    kelly = edge / (price - 1.0)
    return min(bank * kelly * spec.kelly_fraction, bank_slice, bank)


def replay(
    days: Sequence[DayEvidence],
    spec: ReplaySpec = ReplaySpec(),
    *,
    starting_bank: float = 100.0,
) -> Dict[str, Any]:
    """Replay frozen pools through the live selector and size the tickets."""
    bank = starting_bank
    settled_days = 0
    pending_days = 0
    profit_total = 0.0
    daily_log_growth: List[float] = []
    tickets_rebuilt = 0
    tickets_settled = 0
    no_ticket_days = 0
    for day in days:
        tickets = select_tickets(day, spec)
        if not tickets:
            no_ticket_days += 1
            continue
        tickets_rebuilt += len(tickets)
        day_profit = 0.0
        settled_here = 0
        pending_here = 0
        for ticket in tickets:
            result = _position_result(ticket, day.outcomes_by_fid)
            if result is None:
                pending_here += 1
                continue
            stake = _stake(spec, ticket, bank, spec.slots)
            if stake <= 0:
                continue
            if result["status"] == "loss":
                day_profit -= stake
            elif result["status"] == "win":
                day_profit += stake * (float(result["effective_odds"]) - 1.0)
            settled_here += 1
            tickets_settled += 1
        if settled_here and not pending_here:
            settled_days += 1
            growth = math.log(max((bank + day_profit) / bank, 1e-9))
            daily_log_growth.append(growth)
            bank += day_profit
            profit_total += day_profit
        elif pending_here:
            pending_days += 1
    mean_growth = (
        sum(daily_log_growth) / len(daily_log_growth) if daily_log_growth else None
    )
    low_sample = settled_days < LOW_SAMPLE_DAYS
    return {
        "stake_mode": spec.stake_mode,
        "legs": spec.legs,
        "limit": spec.limit,
        "days_considered": len(days),
        "settled_days": settled_days,
        "pending_days": pending_days,
        "no_ticket_days": no_ticket_days,
        "tickets_rebuilt": tickets_rebuilt,
        "tickets_settled": tickets_settled,
        "profit_units": round(profit_total, 6),
        "final_bank": round(bank, 6),
        "mean_daily_log_growth": None if mean_growth is None else round(mean_growth, 8),
        "low_sample": low_sample,
        "low_sample_flag": "[LOW SAMPLE: n < 30]"
        if low_sample
        else "sample >= 30 settled bet-days",
        "note": (
            "zero settled bet-days: no profit, edge, or readiness claim is supported"
            if settled_days == 0
            else "paper replay only; not a forecast of future returns"
        ),
    }


def compare_variants(
    days: Sequence[DayEvidence],
    spec_a: ReplaySpec,
    spec_b: ReplaySpec,
    *,
    iterations: int = 2000,
    seed: int = 20261126,
) -> Dict[str, Any]:
    """Paired day-block bootstrap over the daily log-growth difference.

    Refuses to bootstrap a NO-OP: if both variants pick identical tickets for
    every day with a candidate pool, the comparison is meaningless.
    """
    diffs: List[float] = []
    identical = 0
    comparable = 0
    for day in days:
        tickets_a = select_tickets(day, spec_a)
        tickets_b = select_tickets(day, spec_b)
        if not tickets_a and not tickets_b:
            continue
        comparable += 1
        if _signature(tickets_a, with_id=False) == _signature(tickets_b, with_id=False):
            identical += 1
        result_a = _day_growth(tickets_a, day, spec_a)
        result_b = _day_growth(tickets_b, day, spec_b)
        if result_a is None or result_b is None:
            continue
        diffs.append(result_a - result_b)
    if comparable and identical == comparable:
        return {
            "status": "no_op",
            "reason": "variants select identical tickets on every comparable day",
            "comparable_days": comparable,
        }
    if not diffs:
        return {
            "status": "insufficient",
            "reason": "no day has settled tickets under both variants",
            "comparable_days": comparable,
        }
    rng = random.Random(seed)
    means: List[float] = []
    count = len(diffs)
    for _ in range(iterations):
        sample = [diffs[rng.randrange(count)] for _ in range(count)]
        means.append(sum(sample) / count)
    means.sort()
    observed = sum(diffs) / count
    return {
        "status": "ok",
        "comparable_days": comparable,
        "settled_days": len(diffs),
        "mean_daily_log_growth_diff": round(observed, 8),
        "ci95": [round(means[int(0.025 * iterations)], 8), round(means[int(0.975 * iterations) - 1], 8)],
        "p_a_better": round(sum(1 for value in means if value > 0) / iterations, 4),
        "low_sample": len(diffs) < LOW_SAMPLE_DAYS,
        "low_sample_flag": "[LOW SAMPLE: n < 30]"
        if len(diffs) < LOW_SAMPLE_DAYS
        else "sample >= 30 settled bet-days",
        "note": "paired day-block bootstrap; identical tickets are refused before resampling",
    }


def _day_growth(
    tickets: Sequence[Dict[str, Any]], day: DayEvidence, spec: ReplaySpec
) -> Optional[float]:
    if not tickets:
        return None
    bank = 100.0
    profit = 0.0
    settled = 0
    for ticket in tickets:
        result = _position_result(ticket, day.outcomes_by_fid)
        if result is None:
            return None
        stake = _stake(spec, ticket, bank, spec.slots)
        if result["status"] == "loss":
            profit -= stake
        elif result["status"] == "win":
            profit += stake * (float(result["effective_odds"]) - 1.0)
        settled += 1
    if settled != len(tickets):
        return None
    return math.log(max((bank + profit) / bank, 1e-9))


def merge_days(days: Sequence[DayEvidence], *, label: str = "CROSS") -> List[DayEvidence]:
    """Merge same-date lane pools into one cross-sport replay day.

    Used for ``--mode cross-league``: the acca builder already forbids two legs
    from one fixture, so pooling rows across lanes can only add diversification,
    never a same-game parlay.
    """
    grouped: Dict[str, List[DayEvidence]] = {}
    for day in days:
        grouped.setdefault(day.date, []).append(day)
    merged: List[DayEvidence] = []
    for date, lane_days in sorted(grouped.items()):
        pool: List[Dict[str, Any]] = []
        outcomes: Dict[str, str] = {}
        policy: Dict[str, Any] = {}
        for day in lane_days:
            pool.extend(day.candidate_pool)
            outcomes.update(day.outcomes_by_fid)
            if not policy and day.selection_policy:
                policy = dict(day.selection_policy)
        merged.append(
            DayEvidence(
                date=date,
                lane="cross",
                league=label,
                decision_id=f"cross:{date}",
                frozen_at=max(day.frozen_at for day in lane_days),
                receipt={},
                candidate_pool=pool,
                selection_policy=policy,
                outcomes_by_fid=outcomes,
            )
        )
    return merged
