"""Prequential walk-forward calibration: Platt scaling, Brier skill, honesty gates.

Only settled observations whose *result day* is strictly earlier than the
scoring day are ever visible to a report, so a forecast can never be calibrated
on its own result. Nothing here estimates a market edge from in-sample data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

CALIBRATION_VERSION = "platt-walkforward-v1"
DEFAULT_MIN_SAMPLES = 30
DEFAULT_BRIER_CAP_2WAY = 0.25
DEFAULT_BRIER_CAP_3WAY = 0.22
_EPS = 1e-6


def logit(probability: float) -> float:
    clipped = min(max(probability, _EPS), 1.0 - _EPS)
    return math.log(clipped / (1.0 - clipped))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def brier_score(pairs: Iterable[Tuple[float, int]]) -> Optional[float]:
    """Mean squared error of probabilities against 0/1 outcomes."""
    values = [(float(p), int(y)) for p, y in pairs]
    if not values:
        return None
    return sum((p - y) ** 2 for p, y in values) / len(values)


def base_rate_brier(pairs: Iterable[Tuple[float, int]]) -> Optional[float]:
    """Brier of always predicting the sample's own base rate (naive baseline)."""
    values = [(float(p), int(y)) for p, y in pairs]
    if not values:
        return None
    rate = sum(y for _, y in values) / len(values)
    return sum((rate - y) ** 2 for _, y in values) / len(values)


def fit_platt(
    pairs: Sequence[Tuple[float, int]],
    *,
    prior_strength: float = 10.0,
    iterations: int = 400,
    learning_rate: float = 0.08,
) -> Tuple[float, float]:
    """Fit ``sigmoid(a * logit(p) + b)`` with an L2 pull toward identity.

    Deterministic gradient descent keeps this dependency-free and reproducible
    (no scipy/numpy). ``prior_strength`` shrinks small samples toward the raw
    model, which is the honest default when a segment has just cleared 30.
    """
    if not pairs:
        return 1.0, 0.0
    a, b = 1.0, 0.0
    for _ in range(iterations):
        grad_a = prior_strength * (a - 1.0)
        grad_b = prior_strength * b
        for probability, outcome in pairs:
            x = logit(probability)
            error = sigmoid(a * x + b) - outcome
            grad_a += error * x
            grad_b += error
        a -= learning_rate * grad_a / (len(pairs) + prior_strength)
        b -= learning_rate * grad_b / (len(pairs) + prior_strength)
        a = min(max(a, 0.05), 5.0)
        b = min(max(b, -3.0), 3.0)
    return a, b


@dataclass
class Observation:
    segment: Tuple[str, str, str, str]
    probability: float
    outcome: Optional[int]  # 1 win, 0 loss, None push/void
    result_day: str  # SAST day the result became known


@dataclass
class CalibrationReport:
    segment: Tuple[str, str, str, str]
    n: int
    brier: Optional[float]
    baseline_brier: Optional[float]
    a: float = 1.0
    b: float = 0.0
    sufficient: bool = False
    brier_cap: float = DEFAULT_BRIER_CAP_2WAY
    min_samples: int = DEFAULT_MIN_SAMPLES
    reasons: List[str] = field(default_factory=list)

    @property
    def skill(self) -> Optional[float]:
        if self.brier is None or not self.baseline_brier:
            return None
        return 1.0 - self.brier / self.baseline_brier

    def calibrate(self, probability: float) -> float:
        if not self.sufficient:
            return probability
        return min(max(sigmoid(self.a * logit(probability) + self.b), _EPS), 1.0 - _EPS)

    def to_dict(self) -> Dict[str, object]:
        return {
            "calibration_version": CALIBRATION_VERSION,
            "segment": {
                "league": self.segment[0],
                "market": self.segment[1],
                "period": self.segment[2],
                "source": self.segment[3],
            },
            "status": "calibrated" if self.sufficient else "provisional",
            "n": self.n,
            "minimum_samples": self.min_samples,
            "sufficient_for_accas": self.sufficient,
            "brier_score": None if self.brier is None else round(self.brier, 6),
            "brier_cap": self.brier_cap,
            "base_rate_brier": (
                None if self.baseline_brier is None else round(self.baseline_brier, 6)
            ),
            "skill_vs_base_rate": None if self.skill is None else round(self.skill, 6),
            "platt": {"a": round(self.a, 6), "b": round(self.b, 6)},
            "reasons": list(self.reasons),
        }


class CalibrationBook:
    """Segment-keyed settled observations with strict day-boundary filtering."""

    def __init__(self) -> None:
        self._rows: Dict[Tuple[str, str, str, str], List[Observation]] = {}

    def add(
        self,
        segment: Tuple[str, str, str, str],
        probability: float,
        outcome: Optional[int],
        result_day: str,
    ) -> None:
        self._rows.setdefault(tuple(segment), []).append(
            Observation(tuple(segment), float(probability), outcome, str(result_day))
        )

    def add_many(self, observations: Iterable[Observation]) -> None:
        for observation in observations:
            self.add(
                observation.segment,
                observation.probability,
                observation.outcome,
                observation.result_day,
            )

    def settled(self, segment: Tuple[str, str, str, str], as_of_day: str) -> List[Observation]:
        """Observations strictly before ``as_of_day`` (never same-day leakage)."""
        return [
            row
            for row in self._rows.get(tuple(segment), [])
            if row.outcome is not None and row.result_day < as_of_day
        ]

    def report(
        self,
        segment: Tuple[str, str, str, str],
        as_of_day: str,
        *,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        brier_cap: Optional[float] = None,
        cap_vs_base_rate: bool = False,
    ) -> CalibrationReport:
        rows = self.settled(segment, as_of_day)
        pairs = [(row.probability, int(row.outcome)) for row in rows if row.outcome is not None]
        cap = DEFAULT_BRIER_CAP_2WAY if brier_cap is None else float(brier_cap)
        report = CalibrationReport(
            segment=tuple(segment),
            n=len(pairs),
            brier=brier_score(pairs),
            baseline_brier=base_rate_brier(pairs),
            brier_cap=cap,
            min_samples=min_samples,
        )
        reasons: List[str] = []
        if len(pairs) < min_samples:
            reasons.append("insufficient_walk_forward_samples")
        if report.brier is not None and report.brier > cap:
            reasons.append("brier_above_cap")
        if (
            cap_vs_base_rate
            and report.brier is not None
            and report.baseline_brier is not None
            and report.brier >= report.baseline_brier
        ):
            reasons.append("not_better_than_base_rate")
        report.reasons = reasons
        report.sufficient = not reasons
        if report.sufficient:
            report.a, report.b = fit_platt(pairs)
        return report

    def ledger(self) -> Dict[str, int]:
        return {"|".join(key): len(value) for key, value in self._rows.items()}
