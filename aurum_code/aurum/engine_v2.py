"""
engine_v2.py -- Project AURUM V2.3 draft batch stochastic engine.

Numerical NumPy/SciPy core (no Streamlit import), so it can be tested, run offline, and reused by any interface.

ACCOUNTING UNIT: one complete production batch c, with optional daily schedules.
Yield and revenue are per batch at GROUP level. Expense rates carry explicit units.
Fixed-timing inputs remain available for equation comparisons:

    F   = Y * q                         valuation mass (g); recovered mass for farmgate
    R   = F * P                         gross revenue (PHP)
    Pi  = s_m * R - C_m                 miner-group profit        (tribute_base = "gross")
    Pi  = s_m * (R - C_total)           miner-group profit        (tribute_base = "net")
    L   = sum of person-hours           = n_c * D_c * h  (or an explicit hours matrix)
    E   = Pi / L                        PHP per person-hour

THREE SEPARATE THRESHOLDS (never merged):
    Y_BE : Pi = 0              operating break-even yield (g)
    Y_W  : E  = W_opp          opportunity-wage yield (g)
    Y_HH : Pi = B_HH           household-adequacy yield (g)

YIELD MODEL: hurdle + ordinary + jackpot mixture
    Pr(Y=0) = p0   (fixed scenario value, or p0 ~ Beta(a+z, b+n-z) drawn PER ITERATION)
    Pr(J=1 | Y>0) = pi_J
    Y | ordinary ~ standard Beta-PERT(a, m, b)
    Y | jackpot  ~ truncated Pareto on [threshold, cap]  (inverse-CDF draw, no atom at the cap)

PRICE-PURITY: handled exactly once.
    price_basis = "farmgate"  -> price is the realized price, q forced to 1
    price_basis = "reference" -> price is the pure-gold reference price, q is applied

ASSUMPTION FLAGS: every input can carry a provenance tag in Inputs.provenance
("external", "assumed", "synthetic", "user") that is written to the run manifest.

Batch timing: whole active days plus nonnegative idle days; all crew attend daily schedules.
Expense-rate dependence uses a Gaussian copula; explicit exposure scaling also links total
costs to time. Batch yield is independent of timing and is never multiplied by shifts.
"""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
import os
import platform
import subprocess
import importlib.metadata
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy
from scipy import stats

ENGINE_VERSION = "2.3.0-draft"

LABELS = {
    "P_negative_operating_return": "Probability of Negative Operating Return",
    "P_labor_return_below_wage": "Probability of Labor Return Below Wage Benchmark",
    "P_yield_below_breakeven": "Probability of Yield Below Operating Break-Even",
    "P_fail_household_adequacy": "Probability of Economic Profit Below Household Requirement",
    "P_cash_before_reserve_below_household": "Probability of Cash Shortfall Before Reserve",
    "P_cash_after_reserve_below_household": "Probability of Cash Shortfall After Reserve",
    "P_reserve_underfunded": "Probability of Reserve Funding Shortfall",
    "P_household_and_reserve_met": "Probability of Meeting Household and Reserve Targets",
}
PROB_KEYS = tuple(LABELS.keys())


class InputError(ValueError):
    """Raised when inputs are impossible or ambiguous (expert report, Section 8)."""


def finite_number(value) -> bool:
    """Accept real numeric scalars only, never booleans, strings, NaN or infinity."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def require_finite(name: str, value, errs: List[str]) -> bool:
    """Append an error for `name` if `value` is not a finite real number; return whether it passed."""
    if not finite_number(value):
        errs.append(f"{name}: must be a finite real number (not a boolean).")
        return False
    return True


def require_integer(name: str, value, minimum: int, errs: List[str]) -> bool:
    """Append an error for `name` if `value` is not an int >= minimum; return whether it passed."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        errs.append(f"{name}: must be an integer >= {minimum} (not a boolean).")
        return False
    return True


def _raise_errors(errs: List[str]) -> None:
    """Raise InputError with the collected messages, if any."""
    if errs:
        raise InputError("\n".join(f"- {e}" for e in errs))


def _finite_array(name, value) -> np.ndarray:
    """Coerce `value` to a float ndarray, raising InputError if any element isn't finite."""
    try:
        a = np.asarray(value, dtype=object)
        if not all(finite_number(x) for x in a.flat):
            raise ValueError()
        a = a.astype(float)
        if not np.all(np.isfinite(a)):
            raise ValueError()
        return a
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputError(f"{name}: must contain only finite real numbers.") from exc


# ----------------------------------------------------------------------------------------
# Distribution building blocks
# ----------------------------------------------------------------------------------------
@dataclass
class PERT:
    """Standard Beta-PERT(a, m, b): Beta(1+4(m-a)/(b-a), 1+4(b-m)/(b-a)) scaled to [a, b]."""
    a: float
    m: float
    b: float

    @property
    def degenerate(self) -> bool:
        """True when min equals max, i.e. the distribution collapses to a constant."""
        return self.b == self.a

    @property
    def alpha(self) -> float:
        """Beta shape parameter alpha implied by the PERT bounds and mode."""
        if self.degenerate:              # a == b: constant, shape is irrelevant (sample/ppf return a)
            return 1.0
        return 1.0 + 4.0 * (self.m - self.a) / (self.b - self.a)

    @property
    def beta(self) -> float:
        """Beta shape parameter beta implied by the PERT bounds and mode."""
        if self.degenerate:
            return 1.0
        return 1.0 + 4.0 * (self.b - self.m) / (self.b - self.a)

    @property
    def mean(self) -> float:
        """PERT mean: (a + 4m + b) / 6."""
        return (self.a + 4.0 * self.m + self.b) / 6.0

    def check(self, name: str, errs: List[str], strictly_positive: bool = False,
              upper: Optional[float] = None) -> None:
        """Append validation errors for this PERT's parameters (ordering, sign, upper bound)."""
        valid = [require_finite(f"{name}.{key}", getattr(self, key), errs) for key in ("a", "m", "b")]
        if not all(valid):
            return
        if not (self.a <= self.m <= self.b):
            errs.append(f"{name}: PERT needs min <= mode <= max (got {self.a}, {self.m}, {self.b}).")
        if strictly_positive and self.a <= 0:
            errs.append(f"{name}: minimum must be > 0 (got {self.a}).")
        if not strictly_positive and self.a < 0:
            errs.append(f"{name}: minimum must be >= 0 (got {self.a}).")
        if upper is not None and self.b > upper:
            errs.append(f"{name}: maximum must be <= {upper} (got {self.b}).")

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Draw `size` i.i.d. samples from this PERT distribution."""
        if self.degenerate:
            return np.full(size, float(self.a))
        return self.a + (self.b - self.a) * rng.beta(self.alpha, self.beta, size=size)

    def ppf(self, u: np.ndarray) -> np.ndarray:
        """Inverse-CDF (quantile function) of this PERT distribution at probabilities `u`."""
        if self.degenerate:
            return np.full(np.shape(u), float(self.a))
        return self.a + (self.b - self.a) * stats.beta.ppf(u, self.alpha, self.beta)


@dataclass
class BatchTiming:
    """Daily crew schedules for one batch; all workers attend all scheduled shifts.

    Working days use floor(PERT + .5); idle days are continuous. Daily counts and
    conditional durations are independent across days and simulated batches.
    """
    work_days: PERT
    idle_days: PERT
    shift_counts: Tuple[int, int, int]
    hours_by_shifts: Dict[int, PERT]

    def check(self, errs: List[str]) -> None:
        """Append validation errors for this batch timing's days, shift counts and hours."""
        for name, p in (("work_days", self.work_days), ("idle_days", self.idle_days)):
            if not isinstance(p, PERT):
                errs.append(f"batch_timing.{name}: must be a PERT.")
                continue
            p.check(f"batch_timing.{name}", errs)
            if name == "work_days":
                for value in (p.a, p.m, p.b):
                    if finite_number(value) and (value < 1 or value > 3650 or value != math.floor(value)):
                        errs.append("work_days: bounds and mode must be whole days in [1, 3650].")
        if not isinstance(self.shift_counts, (tuple, list)) or len(self.shift_counts) != 3:
            errs.append("shift_counts: provide three counts for 1, 2, 3 shifts.")
        else:
            valid = [require_integer(f"shift_counts[{i}]", x, 0, errs)
                     for i, x in enumerate(self.shift_counts)]
            if all(valid) and sum(self.shift_counts) != 10:
                errs.append("shift_counts: counts must sum to ten working days.")
        if (not isinstance(self.hours_by_shifts, dict) or set(self.hours_by_shifts) != {1, 2, 3}
                or any(isinstance(k, bool) for k in self.hours_by_shifts)):
            errs.append("hours_by_shifts: supply PERT bounds keyed by 1, 2, 3 shifts.")
        else:
            for d, p in self.hours_by_shifts.items():
                if not isinstance(p, PERT):
                    errs.append(f"hours_by_shifts[{d}]: must be a PERT.")
                    continue
                p.check(f"hours_by_shifts[{d}]", errs, upper=min(16., 24. / d))
                if finite_number(p.a) and p.a < .5:
                    errs.append(f"hours_by_shifts[{d}]: minimum must be >= 0.5 hours.")


COST_BASES = ("per_batch", "per_shift", "per_operating_hour", "per_working_day", "per_calendar_day")
TIMING_KEYS = ("working_days", "idle_days", "calendar_days_per_cycle", "total_shifts",
               "operating_hours", "L_person_hours", "B_HH")


@dataclass
class Jackpot:
    """Truncated Pareto jackpot magnitude on [threshold, cap] (grams per cycle).

    threshold: smallest yield counted as a jackpot (elicited per completed batch)
    cap:       physical upper bound (e.g. equipment/ore capacity)
    shape:     Pareto tail index; ASSUMED -- treat as a sensitivity parameter
    """
    threshold: float
    cap: float
    shape: float = 1.5

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Draw `size` i.i.d. truncated-Pareto jackpot magnitudes on [threshold, cap]."""
        u = rng.random(size)
        t, c, a = self.threshold, self.cap, self.shape
        tail = 1.0 - (t / c) ** a            # Pr(Pareto <= cap)
        return t / (1.0 - u * tail) ** (1.0 / a)


@dataclass
class ZeroYield:
    """Zero-yield (waste cycle) probability.

    Use EITHER a fixed scenario value p0, OR counts (z zero cycles out of n recorded cycles) with a
    Beta(prior_a, prior_b) prior, giving the posterior Beta(prior_a + z, prior_b + n - z).
    The posterior value is drawn PER ITERATION so draws stay independent (valid MCSE).
    """
    p0: Optional[float] = None
    z: Optional[int] = None
    n: Optional[int] = None
    prior_a: float = 1.0
    prior_b: float = 1.0

    def draw_p0(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Draw `size` zero-yield probabilities: the fixed p0 if set, else the posterior Beta draw."""
        if self.p0 is not None:
            return np.full(size, float(self.p0))
        return rng.beta(self.prior_a + self.z, self.prior_b + self.n - self.z, size=size)


@dataclass
class CostItem:
    """One operating-cost category j for the cycle. Give EITHER a fixed amount OR a PERT.

    miner_share = a_mj: fraction of this cost borne by the miner group (1 = miners pay, 0 = financier pays).
    """
    name: str
    miner_share: float = 1.0
    fixed: Optional[float] = None
    pert: Optional[PERT] = None
    basis: str = "per_batch"                  # amount/rate for the whole operation
    components: Optional[List[str]] = None     # declared coverage; default is the item name


@dataclass
class EquipmentItem:
    """Equipment, purchased OR rented -- never one blended number.

    purchased: depreciation per cycle = (acquisition - salvage) / life_cycles * utilization
    rented:    rental cost per cycle = rental_rate_per_shift * D_c
    """
    name: str
    mode: str                              # "purchased" | "rented"
    miner_share: float = 1.0
    acquisition: float = 0.0
    salvage: float = 0.0
    life_cycles: float = 1.0               # replacement life, in cycles
    utilization: float = 1.0               # 0..1
    rental_rate_per_shift: float = 0.0

    def cost_per_cycle(self, D_c: int) -> float:
        """Depreciation (purchased) or rental cost (rented) for the cycle's D_c shifts."""
        if self.mode == "purchased":
            return (self.acquisition - self.salvage) / self.life_cycles * self.utilization
        if self.mode == "rented":
            return self.rental_rate_per_shift * D_c
        raise InputError(f"Equipment '{self.name}': mode must be 'purchased' or 'rented'.")

    def check(self, errs: List[str]) -> None:
        """Append validation errors for this equipment item's mode-specific fields."""
        n = f"Equipment '{self.name}'"
        if self.mode not in ("purchased", "rented"):
            errs.append(f"{n}: mode must be 'purchased' or 'rented'.")
            return
        fields = ("miner_share", "acquisition", "salvage", "life_cycles", "utilization") if self.mode == "purchased" else ("miner_share", "rental_rate_per_shift")
        valid = [require_finite(f"{n}.{key}", getattr(self, key), errs) for key in fields]
        if not all(valid):
            return
        if not (0 <= self.miner_share <= 1):
            errs.append(f"{n}: miner_share must be in [0, 1].")
        if self.mode == "purchased":
            if self.acquisition < 0 or self.salvage < 0:
                errs.append(f"{n}: acquisition and salvage must be >= 0.")
            if self.salvage > self.acquisition:
                errs.append(f"{n}: salvage cannot exceed acquisition cost (depreciation would be negative).")
            if self.life_cycles <= 0:
                errs.append(f"{n}: life_cycles must be > 0.")
            if not (0 <= self.utilization <= 1):
                errs.append(f"{n}: utilization must be in [0, 1].")
        else:
            if self.rental_rate_per_shift < 0:
                errs.append(f"{n}: rental rate must be >= 0.")


# ----------------------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------------------
@dataclass(kw_only=True)
class Inputs:
    # --- accounting unit / labour ---
    n_c: int                                  # active crew size
    batch_timing: Optional[BatchTiming] = None
    D_c: int = 1                              # shifts in the cycle
    calendar_days_per_cycle: Optional[float] = None  # legacy fallback: one day per shift
    hours_per_worker_shift: float = 8.0       # h
    hours_matrix: Optional[List[List[float]]] = None   # optional D_c x n_c matrix of hours worked

    # --- stakeholder accounting ---
    s_m: float                                # miner-group revenue share (0 < s_m <= 1)
    tribute_base: str = "gross"               # "gross" | "net"

    # --- yield ---
    yield_ord: PERT
    zero: ZeroYield
    pi_J: float = 0.0
    jackpot: Optional[Jackpot] = None

    # --- price & purity ---
    price_basis: str = "farmgate"             # "farmgate" (q forced to 1) | "reference"
    price: Union[PERT, float]
    purity: Union[PERT, float] = 1.0          # legacy reference physical purity
    physical_purity: Optional[Union[PERT, float]] = None  # optional farmgate assay
    metadata: Dict[str, object] = field(default_factory=dict)  # source/date/raw conversions
    dataset_id: Optional[str] = None
    dataset_sha256: Optional[str] = None

    # --- costs & dependence ---
    costs: List[CostItem] = field(default_factory=list)
    equipment: List[EquipmentItem] = field(default_factory=list)
    rho_spearman: Optional[List[List[float]]] = None  # over stochastic cost items, in order
    dependence: str = "copula"                # "copula" | "independent"

    # --- per-cycle replacement reserve (no accumulated balance or borrowing) ---
    reserve_mode: str = "none"                # none | match_depreciation | custom
    reserve_location: Optional[str] = None    # before_sharing | after_sharing
    reserve_priority: Optional[str] = None    # household_first | reserve_first
    reserve_custom_per_cycle: float = 0.0

    # --- benchmarks ---
    W_opp: float                              # PHP per person-hour (opportunity wage)
    W_opp_source: str = ""
    W_opp_date: str = ""
    hh_need_per_household_day: float = 0.0    # PHP/day
    n_households: Optional[int] = None        # default = n_c
    nonmining_income_per_household_day: float = 0.0

    # --- bookkeeping ---
    provenance: Dict[str, str] = field(default_factory=dict)

    # derived helpers -----------------------------------------------------------------
    def person_hours(self) -> float:
        """L: total person-hours for the fixed-timing cycle (raises for batch_timing inputs)."""
        if self.batch_timing is not None:
            raise InputError("Batch person-hours are sampled per draw; use result arrays.")
        if self.hours_matrix is not None:
            return float(np.sum(np.asarray(self.hours_matrix, dtype=float)))
        return float(self.n_c * self.D_c * self.hours_per_worker_shift)

    def calendar_days(self) -> float:
        """Calendar days covered by the cycle, falling back to one day per shift if unset."""
        if self.batch_timing is not None:
            raise InputError("Batch calendar days are sampled per draw; use result arrays.")
        return float(self.D_c if self.calendar_days_per_cycle is None else self.calendar_days_per_cycle)

    def household_requirement(self) -> float:
        """B_HH: cash the miners' households still need from mining for the cycle (>= 0)."""
        nh = self.n_c if self.n_households is None else self.n_households
        per_day = max(0.0, self.hh_need_per_household_day - self.nonmining_income_per_household_day)
        return float(nh * self.calendar_days() * per_day)

    def depreciation(self) -> Tuple[float, float]:
        """Total and directly allocated miner depreciation, separate from cash costs."""
        purchased = [eq for eq in self.equipment if eq.mode == "purchased"]
        return (sum(eq.cost_per_cycle(self.D_c) for eq in purchased),
                sum(eq.miner_share * eq.cost_per_cycle(self.D_c) for eq in purchased))

    def all_cost_items(self) -> List[CostItem]:
        """All cost items, including equipment depreciation/rental cast as CostItems."""
        items = list(self.costs)
        for eq in self.equipment:
            if self.batch_timing is not None and eq.mode == "rented":
                items.append(CostItem(name=f"equipment:{eq.name}", miner_share=eq.miner_share,
                                      fixed=eq.rental_rate_per_shift, basis="per_shift"))
            else:
                items.append(CostItem(name=f"equipment:{eq.name}", miner_share=eq.miner_share,
                                      fixed=eq.cost_per_cycle(self.D_c)))
        return items

    def stochastic_cost_items(self) -> List[CostItem]:
        """All cost items drawn from a PERT distribution (excludes fixed-amount items)."""
        return [c for c in self.all_cost_items() if c.pert is not None]


# ----------------------------------------------------------------------------------------
# Validation (expert report, Section 8)
# ----------------------------------------------------------------------------------------
def spearman_to_pearson(rho_s: np.ndarray) -> np.ndarray:
    """Gaussian-copula conversion: rho_pearson = 2 sin(pi * rho_spearman / 6)."""
    r = 2.0 * np.sin(np.pi * np.asarray(rho_s, dtype=float) / 6.0)
    np.fill_diagonal(r, 1.0)
    return r


def validate(inp: Inputs) -> List[str]:
    """Raise InputError listing every problem; return a list of non-fatal warnings."""
    errs: List[str] = []
    warns: List[str] = []

    # Check scalar types before ordering/arithmetic so malformed inputs raise InputError.
    for name in ("s_m", "pi_J", "W_opp", "hh_need_per_household_day", "nonmining_income_per_household_day"):
        require_finite(name, getattr(inp, name), errs)
    require_integer("n_c", inp.n_c, 1, errs)
    if inp.n_households is not None:
        require_integer("n_households", inp.n_households, 0, errs)
    if inp.batch_timing is not None:
        if not isinstance(inp.batch_timing, BatchTiming):
            errs.append("batch_timing: must be a BatchTiming configuration.")
        else:
            inp.batch_timing.check(errs)
        if inp.hours_matrix is not None or inp.calendar_days_per_cycle is not None:
            errs.append("batch_timing cannot be combined with fixed calendar days or an hours matrix.")
        if inp.D_c != 1 or inp.hours_per_worker_shift != 8.0:
            errs.append("batch_timing cannot override fixed shifts/hours; leave their defaults unchanged.")
        require_integer("D_c", inp.D_c, 1, errs)
        require_finite("hours_per_worker_shift", inp.hours_per_worker_shift, errs)
    else:
        require_integer("D_c", inp.D_c, 1, errs)
        if inp.hours_matrix is None:
            require_finite("hours_per_worker_shift", inp.hours_per_worker_shift, errs)
        else:
            try:
                hm = _finite_array("hours_matrix", inp.hours_matrix)
                if hm.shape != (inp.D_c, inp.n_c) or np.any(hm < 0):
                    errs.append("hours_matrix: must have shape (D_c, n_c) and nonnegative hours.")
            except InputError as exc:
                errs.append(str(exc))
        if inp.calendar_days_per_cycle is None:
            warns.append("calendar_days_per_cycle omitted: assuming one calendar day per shift.")
        elif require_finite("calendar_days_per_cycle", inp.calendar_days_per_cycle, errs):
            if inp.calendar_days_per_cycle <= 0:
                errs.append("calendar_days_per_cycle: must be > 0.")
    if not isinstance(inp.price, PERT):
        require_finite("price", inp.price, errs)
    if inp.price_basis == "reference" and not isinstance(inp.purity, PERT):
        require_finite("purity", inp.purity, errs)
    if inp.physical_purity is not None:
        if inp.price_basis != "farmgate":
            errs.append("physical_purity is the optional farmgate assay; use purity for reference pricing.")
        if isinstance(inp.physical_purity, PERT):
            inp.physical_purity.check("physical_purity", errs, strictly_positive=True, upper=1.)
        elif require_finite("physical_purity", inp.physical_purity, errs):
            if not 0 < inp.physical_purity <= 1:
                errs.append("physical_purity: must be in (0, 1].")
    if not isinstance(inp.metadata, dict):
        errs.append("metadata: must be a JSON object.")
    else:
        try:
            json.dumps(inp.metadata, allow_nan=False)
        except (TypeError, ValueError):
            errs.append("metadata: must contain finite JSON-serializable values.")
    if (inp.dataset_id is None) != (inp.dataset_sha256 is None):
        errs.append("dataset_id and dataset_sha256 must be provided together.")
    if inp.dataset_id is not None:
        if not isinstance(inp.dataset_id, str) or not inp.dataset_id.strip():
            errs.append("dataset_id: must be a nonempty string.")
        if (not isinstance(inp.dataset_sha256, str) or len(inp.dataset_sha256) != 64
                or any(c not in "0123456789abcdef" for c in inp.dataset_sha256)):
            errs.append("dataset_sha256: must be a lowercase SHA-256 hex digest.")
    for c in inp.costs:
        require_finite(f"Cost '{c.name}'.miner_share", c.miner_share, errs)
        if c.fixed is not None:
            require_finite(f"Cost '{c.name}'.fixed", c.fixed, errs)
    if finite_number(inp.pi_J) and inp.pi_J > 0 and inp.jackpot is not None:
        for name in ("threshold", "cap", "shape"):
            require_finite(f"jackpot.{name}", getattr(inp.jackpot, name), errs)
    if inp.zero.p0 is not None:
        require_finite("zero.p0", inp.zero.p0, errs)
    else:
        require_integer("zero.z", inp.zero.z, 0, errs)
        require_integer("zero.n", inp.zero.n, 0, errs)
        require_finite("zero.prior_a", inp.zero.prior_a, errs)
        require_finite("zero.prior_b", inp.zero.prior_b, errs)
    if inp.reserve_mode not in ("none", "match_depreciation", "custom"):
        errs.append("reserve_mode: choose none, match_depreciation or custom.")
    if inp.reserve_mode != "none":
        if inp.reserve_location not in ("before_sharing", "after_sharing"):
            errs.append("reserve_location: explicitly choose before_sharing or after_sharing.")
        if inp.reserve_priority not in ("household_first", "reserve_first"):
            errs.append("reserve_priority: explicitly choose household_first or reserve_first.")
    if inp.reserve_mode == "custom" and require_finite("reserve_custom_per_cycle", inp.reserve_custom_per_cycle, errs):
        if inp.reserve_custom_per_cycle < 0:
            errs.append("reserve_custom_per_cycle: must be >= 0.")
    _raise_errors(errs)

    if inp.batch_timing is None:
        # crew, labour
        if not isinstance(inp.n_c, (int, np.integer)) or inp.n_c < 1:
            errs.append("n_c: crew size must be an integer >= 1 (active contributors only).")
        if not isinstance(inp.D_c, (int, np.integer)) or inp.D_c < 1:
            errs.append("D_c: shifts per cycle must be an integer >= 1.")
        if inp.hours_matrix is not None:
            hm = np.asarray(inp.hours_matrix, dtype=float)
            if hm.ndim != 2 or np.any(hm < 0):
                errs.append("hours_matrix: must be a 2-D matrix of hours >= 0.")
        elif inp.hours_per_worker_shift <= 0:
            errs.append("hours_per_worker_shift must be > 0.")
        try:
            if not finite_number(inp.person_hours()) or inp.person_hours() <= 0:
                errs.append("Person-hours L must be > 0 (cannot be replaced by shift duration alone).")
        except Exception:
            errs.append("Person-hours L could not be computed from the labour inputs.")

    # shares
    if not (0 < inp.s_m <= 1):
        errs.append("s_m: miner revenue share must satisfy 0 < s_m <= 1. "
                    "A wage-paid worker (s_m = 0) is outside the share model: analyse them separately.")
    if inp.tribute_base not in ("gross", "net"):
        errs.append("tribute_base must be 'gross' or 'net'.")

    # yield model
    inp.yield_ord.check("Ordinary yield (g/cycle)", errs)
    if finite_number(inp.yield_ord.b) and inp.yield_ord.b == 0:
        errs.append("Ordinary yield cannot be concentrated at zero; use p0=1 with valid positive ordinary parameters.")
    if not (0 <= inp.pi_J <= 1):
        errs.append("pi_J must be in [0, 1].")
    if inp.pi_J > 0:
        if inp.jackpot is None:
            errs.append("pi_J > 0 requires a Jackpot definition.")
        else:
            j = inp.jackpot
            if not (0 < j.threshold < j.cap):
                errs.append("Jackpot: need 0 < threshold < cap.")
            if j.shape <= 0:
                errs.append("Jackpot: Pareto shape must be > 0.")
            warns.append("Jackpot shape and threshold are scenario parameters; sensitivity-test them.")
    z = inp.zero
    if z.p0 is not None:
        if not (0 <= z.p0 <= 1):
            errs.append("p0 must be in [0, 1].")
    elif z.z is None or z.n is None:
        errs.append("Zero-yield model needs either p0 or counts (z, n).")
    else:
        if z.n < 0 or z.z < 0 or z.z > z.n:
            errs.append("Zero-yield counts: need 0 <= z <= n (keep numerator AND denominator).")
        if z.prior_a <= 0 or z.prior_b <= 0:
            errs.append("Beta prior hyperparameters must be > 0.")

    # price / purity
    if inp.price_basis not in ("farmgate", "reference"):
        errs.append("price_basis must be 'farmgate' or 'reference'.")
    if isinstance(inp.price, PERT):
        inp.price.check("Price (PHP/g)", errs, strictly_positive=True)
    elif inp.price <= 0:
        errs.append("Price must be > 0 PHP/g.")
    if inp.price_basis == "reference":
        if isinstance(inp.purity, PERT):
            inp.purity.check("Purity q", errs, strictly_positive=True, upper=1.0)
        elif not (0 < inp.purity <= 1):
            errs.append("Purity q must satisfy 0 < q <= 1 (karat / 24 if given in karats).")

    # costs
    cost_items = list(inp.costs)
    for eq in inp.equipment:
        eq_errs = []
        eq.check(eq_errs)
        errs.extend(eq_errs)
        fixed = 0.0 if eq_errs else eq.cost_per_cycle(inp.D_c)
        cost_items.append(
            CostItem(
                name=f"equipment:{eq.name}",
                miner_share=eq.miner_share,
                fixed=fixed,
            )
        )

    names = set()
    coverage = {}
    for c in cost_items:
        if c.basis not in COST_BASES:
            errs.append(f"Cost '{c.name}': unknown expense basis.")
        if inp.batch_timing is None and c.basis == "per_working_day":
            errs.append(f"Cost '{c.name}': working-day expenses require batch_timing.")
        if inp.hours_matrix is not None and c.basis == "per_operating_hour":
            errs.append(f"Cost '{c.name}': operating exposure is ambiguous with an hours matrix.")
        if c.name in names:
            errs.append(f"Cost '{c.name}' appears twice (each cost must appear exactly once).")
        names.add(c.name)
        components = [c.name] if c.components is None else c.components
        if (not isinstance(components, (list, tuple)) or not components
                or any(not isinstance(x, str) or not x.strip() for x in components)):
            errs.append(f"Cost '{c.name}': components must be nonempty identifiers.")
        else:
            for component in components:
                key = component.strip().casefold()
                if key in coverage:
                    errs.append(f"Cost component '{component}' overlaps '{coverage[key]}' and '{c.name}'.")
                coverage[key] = c.name
        if not (0 <= c.miner_share <= 1):
            errs.append(f"Cost '{c.name}': miner_share must be in [0, 1].")
        if (c.fixed is None) == (c.pert is None):
            errs.append(f"Cost '{c.name}': give exactly one of fixed amount or PERT.")
        if c.fixed is not None and c.fixed < 0:
            errs.append(f"Cost '{c.name}': must be >= 0 PHP per cycle.")
        if c.pert is not None:
            c.pert.check(f"Cost '{c.name}'", errs)

    # dependence
    sto = [c for c in inp.costs if c.pert is not None]
    if inp.dependence not in ("copula", "independent"):
        errs.append("dependence must be 'copula' or 'independent'.")
    if inp.dependence == "copula" and len(sto) > 1:
        if inp.rho_spearman is None:
            errs.append("copula dependence needs rho_spearman for the stochastic costs "
                        "(or set dependence='independent').")
        else:
            r = _finite_array("rho_spearman", inp.rho_spearman)
            if r.shape != (len(sto), len(sto)):
                errs.append(f"rho_spearman must be {len(sto)}x{len(sto)} (one row per stochastic cost).")
            elif (not np.allclose(r, r.T)) or np.any(np.abs(r) > 1) or not np.allclose(np.diag(r), 1):
                errs.append("rho_spearman must be symmetric, have a unit diagonal and entries in [-1, 1].")
            else:
                if np.min(np.linalg.eigvalsh(spearman_to_pearson(r))) <= 1e-10:
                    errs.append("rho_spearman is not positive definite after conversion.")

    # benchmarks
    if inp.W_opp <= 0:
        errs.append("W_opp must be > 0 PHP per person-hour.")
    if not inp.W_opp_source or not inp.W_opp_date:
        warns.append("W_opp: document its source and effective date (expert report, Section 8).")
    if inp.hh_need_per_household_day < 0 or inp.nonmining_income_per_household_day < 0:
        errs.append("Household need and non-mining income must be >= 0 PHP/day.")
    if inp.n_households is not None and inp.n_households < 0:
        errs.append("n_households must be >= 0.")

    _raise_errors(errs)
    try:
        derived = (inp.household_requirement() if inp.batch_timing is None else 0., *inp.depreciation(),
                   *(eq.cost_per_cycle(inp.D_c) for eq in inp.equipment))
        if inp.batch_timing is not None:
            t = inp.batch_timing
            # Check maximum derived exposures before NumPy can overflow or create object arrays.
            calendar_max = t.work_days.b + t.idle_days.b
            derived += (calendar_max, inp.n_c * (24. * t.work_days.b),
                        (inp.n_c if inp.n_households is None else inp.n_households) * calendar_max * max(0., inp.hh_need_per_household_day - inp.nonmining_income_per_household_day))
        if not all(finite_number(x) for x in derived):
            raise ValueError()
        if inp.zero.p0 is None and not all(finite_number(x) for x in
                (inp.zero.prior_a + inp.zero.z, inp.zero.prior_b + inp.zero.n - inp.zero.z)):
            raise ValueError()
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise InputError("Derived timing, household, equipment or zero-yield parameters exceed finite numeric limits.") from exc
    return warns


# ----------------------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------------------
def _copula_chol(inp: Inputs) -> Optional[np.ndarray]:
    """Cholesky factor of the Gaussian copula's correlation matrix, or None if not needed."""
    sto = inp.stochastic_cost_items()
    if inp.dependence != "copula" or len(sto) < 2:
        return None
    return np.linalg.cholesky(spearman_to_pearson(np.asarray(inp.rho_spearman, dtype=float)))


def sample_timing(inp: Inputs, rng: np.random.Generator, size: int) -> Dict[str, np.ndarray]:
    """Draw per-cycle timing arrays (days, shifts, hours, L, B_HH) for `size` draws.

    Falls back to fixed-timing constants when batch_timing is unset; otherwise simulates
    daily work/idle days, shift counts and hours per the batch_timing schedule.
    """
    if inp.batch_timing is None:
        return {"calendar_days_per_cycle": np.full(size, inp.calendar_days()),
                "total_shifts": np.full(size, inp.D_c, dtype=int),
                "operating_hours": np.full(size, inp.D_c * inp.hours_per_worker_shift),
                "L_person_hours": np.full(size, inp.person_hours()),
                "B_HH": np.full(size, inp.household_requirement())}
    t = inp.batch_timing
    work = np.floor(t.work_days.sample(rng, size) + .5).astype(int)
    idle = t.idle_days.sample(rng, size)
    shifts = np.zeros(size, dtype=int)
    hours = np.zeros(size)
    for day in range(int(work.max())):
        active = np.flatnonzero(work > day)
        daily = rng.choice([1, 2, 3], size=len(active), p=np.asarray(t.shift_counts) / 10.)
        shifts[active] += daily
        for d in (1, 2, 3):
            ix = active[daily == d]
            hours[ix] += d * t.hours_by_shifts[d].sample(rng, len(ix))
    calendar = work + idle
    return {"working_days": work, "idle_days": idle, "calendar_days_per_cycle": calendar,
            "total_shifts": shifts, "operating_hours": hours, "L_person_hours": inp.n_c * hours,
            "B_HH": (inp.n_c if inp.n_households is None else inp.n_households) * calendar
                    * max(0., inp.hh_need_per_household_day - inp.nonmining_income_per_household_day)}


def expense_exposure(basis: str, timing: Dict[str, np.ndarray]):
    """Exposure array (or scalar 1.0 for per_batch) that a cost `basis` scales against."""
    keys = {"per_shift": "total_shifts", "per_operating_hour": "operating_hours",
            "per_working_day": "working_days", "per_calendar_day": "calendar_days_per_cycle"}
    return 1.0 if basis == "per_batch" else timing[keys[basis]]


def sample_costs(inp: Inputs, rng: np.random.Generator, size: int,
                 chol: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """Draw expense rates (or batch amounts); stochastic rates share a Gaussian copula."""
    items = inp.all_cost_items()
    sto = [c for c in items if c.pert is not None]
    out: Dict[str, np.ndarray] = {}
    if sto:
        if chol is not None:
            z = rng.standard_normal((size, len(sto))) @ chol.T
            u = stats.norm.cdf(z)
        else:
            u = rng.random((size, len(sto)))
        u = np.clip(u, 1e-12, 1 - 1e-12)
        for k, c in enumerate(sto):
            out[c.name] = c.pert.ppf(u[:, k])
    for c in items:
        if c.fixed is not None:
            out[c.name] = np.full(size, float(c.fixed))
    return out


def sample_batch(inp: Inputs, rng: np.random.Generator, size: int,
                 chol: Optional[np.ndarray], timing_rng: Optional[np.random.Generator] = None,
                 assay_rng: Optional[np.random.Generator] = None) -> Dict[str, np.ndarray]:
    """Draw one chunk of `size` simulated batches: yield, price/purity, costs and timing."""
    p0 = inp.zero.draw_p0(rng, size)
    positive = rng.random(size) >= p0                    # hurdle: True = productive cycle
    is_jack = rng.random(size) < inp.pi_J                # given productive
    y_ord = inp.yield_ord.sample(rng, size)
    y_jack = inp.jackpot.sample(rng, size) if (inp.jackpot is not None and inp.pi_J > 0) else y_ord
    Y = np.where(positive, np.where(is_jack, y_jack, y_ord), 0.0)

    if inp.price_basis == "farmgate":
        q = np.ones(size)
    elif isinstance(inp.purity, PERT):
        q = inp.purity.sample(rng, size)
    else:
        q = np.full(size, float(inp.purity))
    P = inp.price.sample(rng, size) if isinstance(inp.price, PERT) else np.full(size, float(inp.price))

    physical = {}
    if inp.price_basis == "reference":
        physical["physical_purity_q"] = q.copy()
    elif inp.physical_purity is not None:
        assay = inp.physical_purity
        if isinstance(assay, PERT):
            if assay_rng is None:
                raise InputError("Stochastic farmgate assay requires a separate assay_rng.")
            physical["physical_purity_q"] = assay.sample(assay_rng, size)
        else:
            physical["physical_purity_q"] = np.full(size, float(assay))
    rates = sample_costs(inp, rng, size, chol)
    timing = sample_timing(inp, rng if timing_rng is None else timing_rng, size)
    costs = {c.name: rates[c.name] * expense_exposure(c.basis, timing) for c in inp.all_cost_items()}
    miner_share = {c.name: c.miner_share for c in inp.all_cost_items()}
    C_tot = np.zeros(size)
    C_m = np.zeros(size)
    for name, arr in costs.items():
        C_tot += arr
        C_m += miner_share[name] * arr
    return {"Y": Y, "q": q, "P": P, "C_tot": C_tot, "C_m": C_m, "costs": costs, "rates": rates, "timing": timing, "p0": p0, **physical,
            "is_ordinary": positive & ~is_jack, "is_jackpot": positive & is_jack}


# ----------------------------------------------------------------------------------------
# Accounting kernel (stakeholder cash flow + three thresholds)
# ----------------------------------------------------------------------------------------
def accounting_kernel(Y, q, P, C_m, C_tot, *, s_m: float, L: Union[float, np.ndarray], W_opp: float,
                      B_HH: Union[float, np.ndarray], tribute_base: str = "gross",
                      depreciation_total: float = 0.0, depreciation_miner: float = 0.0,
                      reserve_mode: str = "none", reserve_location: Optional[str] = None,
                      reserve_priority: Optional[str] = None,
                      reserve_custom_per_cycle: float = 0.0) -> Dict[str, np.ndarray]:
    """Vectorised, unit-checked accounting. Inputs are arrays per cycle (group level)."""
    F = Y * q                                  # valuation mass (fine gold only in reference mode)
    R = F * P                                  # gross revenue (PHP)
    Rm = s_m * R                               # miner-group revenue
    C_eff = C_m if tribute_base == "gross" else s_m * C_tot   # costs the miner group effectively bears
    Pi = Rm - C_eff                            # miner-group profit (PHP)
    E = Pi / L                                 # PHP per person-hour
    unit_rev = q * P * s_m                     # PHP per gram of recovered gold, after the share
    # Economic costs above include depreciation; cash costs below do not.
    dep_eff = depreciation_miner if tribute_base == "gross" else s_m * depreciation_total
    cash_cost_total = C_tot - depreciation_total
    cash_cost_miner = C_m - depreciation_miner
    cash_cost_eff = cash_cost_miner if tribute_base == "gross" else s_m * cash_cost_total
    cash_before = Rm - cash_cost_eff
    target = 0.0
    if reserve_mode == "match_depreciation":
        target = depreciation_total if reserve_location == "before_sharing" else dep_eff
    elif reserve_mode == "custom":
        target = reserve_custom_per_cycle
    common = reserve_location == "before_sharing"
    burden = s_m if common else 1.0
    available = np.maximum(R - cash_cost_total if common else cash_before, 0.0)
    if reserve_priority == "household_first":
        available = np.minimum(available, np.maximum((cash_before - B_HH) / burden, 0.0))
    planned = np.full_like(R, target, dtype=float)
    funded = np.minimum(planned, available)
    miner_contribution = burden * funded
    cash_after = cash_before - miner_contribution
    if reserve_priority == "household_first":
        # Preserve the protected amount explicitly: subtracting nearly equal large
        # values can otherwise put cash a few ulps below B_HH and count a false failure.
        protected = np.minimum(cash_before, B_HH)
        surplus = np.maximum(cash_before - B_HH, 0.0)
        cash_after = np.where(funded > 0, protected + np.maximum(surplus - miner_contribution, 0.0), cash_before)
    return {
        "cash_cost_total": cash_cost_total, "cash_cost_miner": cash_cost_eff,
        "depreciation_total": np.full_like(R, depreciation_total, dtype=float),
        "depreciation_miner": np.full_like(R, dep_eff, dtype=float),
        "cash_before_reserve": cash_before, "cash_after_reserve": cash_after,
        "reserve_planned": planned, "reserve_funded": funded,
        "reserve_shortfall": planned - funded, "reserve_miner_contribution": miner_contribution,
        "F": F, "R": R, "Rm": Rm, "C_eff": C_eff, "Pi": Pi, "E": E,
        "Y_BE": C_eff / unit_rev,
        "Y_W": (C_eff + W_opp * L) / unit_rev,
        "Y_HH": (C_eff + B_HH) / unit_rev,
    }


# ----------------------------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------------------------
def mcse_prob(k: int, n: int) -> float:
    """Monte Carlo standard error of a probability, using (k+2)/(n+4) so it is never 0."""
    p = (k + 2.0) / (n + 4.0)
    return math.sqrt(p * (1.0 - p) / n)


def risk_band(p_meet_benchmark: float, low_cut: float = 0.67, mod_cut: float = 0.34) -> str:
    """Mutually exclusive bands (expert report 3.11). Cut-offs are ILLUSTRATIVE: justify or test them."""
    if p_meet_benchmark >= low_cut:
        return "low"
    if p_meet_benchmark >= mod_cut:
        return "moderate"
    return "high"


def _prob_counts(a: Dict[str, np.ndarray], W_opp: float, B_HH=None) -> Dict[str, int]:
    """Count, per risk probability key, how many simulated cycles trigger the event."""
    if B_HH is None:
        B_HH = a["B_HH"]
    events = {
        "P_negative_operating_return": a["Pi"] < 0,
        "P_labor_return_below_wage": a["E"] < W_opp,
        "P_yield_below_breakeven": a["Y"] < a["Y_BE"],
        "P_fail_household_adequacy": a["Pi"] < B_HH,
        "P_cash_before_reserve_below_household": a["cash_before_reserve"] < B_HH,
        "P_cash_after_reserve_below_household": a["cash_after_reserve"] < B_HH,
        "P_reserve_underfunded": a["reserve_shortfall"] > 0,
        "P_household_and_reserve_met": (a["cash_after_reserve"] >= B_HH) & (a["reserve_shortfall"] == 0),
    }
    return {key: int(np.count_nonzero(event)) for key, event in events.items()}


def expected_shortfall(values: np.ndarray, fraction: float = 0.05) -> float:
    """Mean of exactly the lowest fraction of empirical mass, including boundary ties."""
    a = _finite_array("expected_shortfall values", values)
    if a.ndim != 1 or a.size == 0 or not finite_number(fraction) or not 0 < fraction <= 1:
        raise InputError("expected_shortfall: need a nonempty 1-D sample and 0 < fraction <= 1.")
    a = np.sort(a)
    mass = fraction * a.size
    whole = int(math.floor(mass))
    remainder = mass - whole
    # Divide before summing to avoid overflowing sums of otherwise finite returns.
    value = float(np.sum(a[:whole] / mass))
    if remainder > 0:
        value += float(a[whole] * (remainder / mass))
    if not finite_number(value):
        raise InputError("expected_shortfall: arithmetic exceeded finite numeric limits.")
    return value


RETURN_KEYS = ("E_mean", "E_median", "E_P5", "E_P95", "E_expected_shortfall_5pct")


def _validate_return_tolerances(abs_tol, rel_tol):
    """Raise InputError unless both return tolerances are finite, nonnegative and at least one is positive."""
    errs = []
    for key, value in (("return_abs_tol", abs_tol), ("return_rel_tol", rel_tol)):
        if require_finite(key, value, errs) and value < 0:
            errs.append(f"{key}: must be nonnegative.")
    if not errs and abs_tol == rel_tol == 0:
        errs.append("At least one return tolerance must be positive.")
    _raise_errors(errs)


def return_diagnostics(values, *, abs_tol=1., rel_tol=.01):
    """Estimated pointwise numerical errors; not model or predictive uncertainty.

    Quantile bounds invert a Binomial(N,p) count. Unbounded order-statistic
    intervals for small N are unavailable rather than falsely clipped to extrema.
    ES uses min(E, empirical P5)/.05 as its winsorized influence values.
    """
    _validate_return_tolerances(abs_tol, rel_tol)
    if isinstance(values, np.ndarray) and values.dtype.kind in "fiu" and np.all(np.isfinite(values)):
        a = values.astype(float, copy=False)
    else:
        a = _finite_array("return diagnostics", values)
    if a.ndim != 1 or not a.size:
        raise InputError("return diagnostics need a nonempty 1-D sample.")
    ordered = np.sort(a)
    n = len(a)
    points = dict(zip(("E_P5", "E_median", "E_P95"), np.percentile(a, [5, 50, 95])))
    # Reuse the sorted sample; the exact fractional boundary matches expected_shortfall.
    mass = .05 * n
    whole = int(math.floor(mass))
    remainder = mass - whole
    tail_mean = float(np.sum(ordered[:whole] / mass))
    if remainder > 0:
        tail_mean += float(ordered[whole] * (remainder / mass))
    points.update(E_mean=float(a.mean()), E_expected_shortfall_5pct=tail_mean)
    errors = {"E_mean": float(2 * np.std(a, ddof=1) / math.sqrt(n)) if n > 1 else None}
    winsor = np.minimum(a, points["E_P5"])
    errors["E_expected_shortfall_5pct"] = float(2 * np.std(winsor, ddof=1) / math.sqrt(n) / .05) if n > 1 else None
    result = {key: float(value) for key, value in points.items()}
    for key, probability in (("E_P5", .05), ("E_median", .5), ("E_P95", .95)):
        lo = int(stats.binom.ppf(.025, n, probability))
        hi = int(stats.binom.ppf(.975, n, probability)) + 1
        available = lo >= 1 and hi <= n
        low, high = (float(ordered[lo - 1]), float(ordered[hi - 1])) if available else (None, None)
        result[f"lower_{key}"], result[f"upper_{key}"] = low, high
        errors[key] = max(abs(points[key] - low), abs(high - points[key])) if available else None
    for key in RETURN_KEYS:
        result[f"error_{key}"] = errors[key]
        result[f"tolerance_{key}"] = max(abs_tol, rel_tol * abs(points[key]))
    result["return_precision_met"] = all(errors[k] is not None and finite_number(errors[k])
                                         and errors[k] <= result[f"tolerance_{k}"] for k in RETURN_KEYS)
    return result


def convergence_check(inp, arrays, counts, n, *, tol, abs_tol, rel_tol, previous=None):
    """One completed-batch check, including productive-component support."""
    d = return_diagnostics(arrays["E"], abs_tol=abs_tol, rel_tol=rel_tol)
    d["N"] = n
    for key in PROB_KEYS:
        d[key] = counts[key] / n
        d[f"MCSE_{key}"] = mcse_prob(counts[key], n)
    d["probability_precision_met"] = all(d[f"MCSE_{k}"] <= tol for k in PROB_KEYS)
    productive_enabled = inp.zero.p0 != 1  # a positive Beta posterior always permits production
    reasons = []
    for name, enabled in (("ordinary", productive_enabled and inp.pi_J < 1),
                          ("jackpot", productive_enabled and inp.pi_J > 0)):
        observed = int(np.count_nonzero(arrays[f"is_{name}"]))
        d[f"observed_{name}_count"] = observed
        if enabled and observed < 30:
            reasons.append(f"{name} component has {observed}/30 required observations")
    d["mixture_support_met"] = not reasons
    for key in RETURN_KEYS:
        d[f"change_{key}"] = abs(d[key] - previous[key]) if previous is not None else None
    d["return_stability_met"] = previous is not None and all(
        d[f"change_{key}"] <= d[f"tolerance_{key}"] for key in RETURN_KEYS)
    if not d["probability_precision_met"]:
        reasons.append("probability MCSE tolerance not met")
    for key in RETURN_KEYS:
        error = d[f"error_{key}"]
        if error is None or not finite_number(error) or error > d[f"tolerance_{key}"]:
            reasons.append(f"{key} numerical error unavailable or above tolerance")
    d["check_precision_met"] = d["probability_precision_met"] and d["return_precision_met"] and d["mixture_support_met"]
    if not d["return_stability_met"]:
        reasons.append("return changes require another stable completed-batch check")
    d["convergence_reasons"] = "; ".join(reasons)
    return d


def _validate_run_options(seed, batch_size, n_min, n_max, tol, n_fixed, sens_rows):
    """Raise InputError if any run() configuration option has an invalid type or range."""
    errs = []
    for name, value, minimum in (("seed", seed, 0), ("batch_size", batch_size, 1),
                                  ("n_min", n_min, 1), ("n_max", n_max, 1), ("sens_rows", sens_rows, 2)):
        require_integer(name, value, minimum, errs)
    if n_fixed is not None:
        require_integer("n_fixed", n_fixed, 1, errs)
    if require_finite("tol", tol, errs) and not 0 < tol < 1:
        errs.append("tol: must satisfy 0 < tol < 1.")
    # n_max < n_min is allowed: return an explicitly non-converged capped run.
    _raise_errors(errs)


# ----------------------------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------------------------
@dataclass
class Result:
    summary: Dict[str, object]
    stage_table: pd.DataFrame
    rank_sensitivity: Dict[str, float]
    arrays: Dict[str, np.ndarray]
    manifest: Dict[str, object]
    warnings: List[str]


# ----------------------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------------------
STAGES = (1_000, 10_000, 50_000, 100_000, 200_000)


def run(inp: Inputs, *, seed: int = 42, batch_size: int = 5_000, n_min: int = 10_000,
        n_max: int = 200_000, tol: float = 0.005, n_fixed: Optional[int] = None,
        sens_rows: int = 50_000, return_abs_tol: float = 1., return_rel_tol: float = .01) -> Result:
    """Run the simulation.

    Full convergence requires probability and return precision at two consecutive
    completed-batch checks, stable returns and supported mixture components after n_min.
    n_fixed still runs exactly N draws. Numerical bounds do not measure model uncertainty.
    """
    _validate_run_options(seed, batch_size, n_min, n_max, tol, n_fixed, sens_rows)
    _validate_return_tolerances(return_abs_tol, return_rel_tol)
    warns = validate(inp)
    rng = np.random.default_rng(seed)
    chol = _copula_chol(inp)
    children = np.random.SeedSequence(seed).spawn(2)
    timing_rng = np.random.default_rng(children[0])
    assay_rng = np.random.default_rng(children[1])
    dep_total, dep_miner = inp.depreciation()

    chunks: List[Dict[str, np.ndarray]] = []
    counts = {k: 0 for k in PROB_KEYS}
    N = 0
    converged = False
    previous = None
    checks = {}
    final_check = None
    target = n_fixed if n_fixed is not None else n_max
    while N < target:
        b = min(batch_size, target - N)
        d = sample_batch(inp, rng, b, chol, timing_rng, assay_rng)
        k = accounting_kernel(d["Y"], d["q"], d["P"], d["C_m"], d["C_tot"], s_m=inp.s_m, L=d["timing"]["L_person_hours"],
                              W_opp=inp.W_opp, B_HH=d["timing"]["B_HH"], tribute_base=inp.tribute_base,
                              depreciation_total=dep_total, depreciation_miner=dep_miner,
                              reserve_mode=inp.reserve_mode, reserve_location=inp.reserve_location,
                              reserve_priority=inp.reserve_priority,
                              reserve_custom_per_cycle=inp.reserve_custom_per_cycle)
        chunk = {**{n: d[n] for n in ("Y", "q", "P", "C_tot", "C_m", "p0", "is_ordinary", "is_jackpot")}, **k, **d["timing"],
                 **({"physical_purity_q": d["physical_purity_q"]} if "physical_purity_q" in d else {}),
                 **{f"cost::{n}": a for n, a in d["costs"].items()},
                 **{f"rate::{n}": a for n, a in d["rates"].items()}}
        for name, values in chunk.items():
            if not np.all(np.isfinite(values)):
                raise InputError(f"Simulation produced non-finite {name}; check input magnitudes.")
        chunks.append(chunk)
        c = _prob_counts(chunk, inp.W_opp)
        for key in PROB_KEYS:
            counts[key] += c[key]
        N += b
        if N >= n_min:
            diagnostic_arrays = {name: np.concatenate([ch[name] for ch in chunks])
                                 for name in ("E", "is_ordinary", "is_jackpot")}
            final_check = convergence_check(inp, diagnostic_arrays, counts, N, tol=tol,
                                           abs_tol=return_abs_tol, rel_tol=return_rel_tol, previous=previous)
            converged = bool(previous is not None and previous["check_precision_met"]
                             and final_check["check_precision_met"] and final_check["return_stability_met"])
            if not converged and not final_check["convergence_reasons"]:
                final_check["convergence_reasons"] = "two consecutive precision checks required"
            final_check["converged"] = converged
            checks[N] = final_check
            previous = final_check
            if n_fixed is None and converged:
                break

    arr = {name: np.concatenate([ch[name] for ch in chunks]) for name in chunks[0]}
    E, Pi = arr["E"], arr["Pi"]
    if final_check is None:
        final_check = convergence_check(inp, arr, counts, N, tol=tol, abs_tol=return_abs_tol,
                                       rel_tol=return_rel_tol)
        final_check["convergence_reasons"] += "; n_min and two completed-batch checks required"
        final_check["converged"] = False
    if not converged:
        warns.append(f"Convergence NOT reached at N = {N:,}: {final_check['convergence_reasons']}.")

    # ---- summary ----
    probs = {key: counts[key] / N for key in PROB_KEYS}
    p5, p50, p95 = np.percentile(E, [5, 50, 95])
    summary: Dict[str, object] = {
        **final_check, "N": N, "converged": converged,
        **probs,
        **{f"MCSE_{key}": mcse_prob(counts[key], N) for key in PROB_KEYS},
        "E_mean": float(E.mean()), "E_median": float(p50), "E_P5": float(p5), "E_P95": float(p95),
        "E_expected_shortfall_5pct": expected_shortfall(E),
        "Pi_mean": float(Pi.mean()), "Pi_median": float(np.median(Pi)),
        "Y_BE_median": float(np.median(arr["Y_BE"])), "Y_W_median": float(np.median(arr["Y_W"])),
        "Y_HH_median": float(np.median(arr["Y_HH"])),
        "Y_BE_mean": float(arr["Y_BE"].mean()), "Y_W_mean": float(arr["Y_W"].mean()),
        "Y_HH_mean": float(arr["Y_HH"].mean()),
        "observed_zero_cycle_share": float(np.mean(arr["Y"] == 0)),
        "risk_band_vs_wage": risk_band(1.0 - probs["P_labor_return_below_wage"]),
    }

    if inp.batch_timing is None:
        summary.update(L_person_hours=inp.person_hours(), B_HH=inp.household_requirement(),
                       calendar_days_per_cycle=inp.calendar_days())
    for name in TIMING_KEYS:
        if name in arr:
            summary[f"{name}_mean"] = float(arr[name].mean())
            summary[f"{name}_median"] = float(np.median(arr[name]))
    for name in ("cash_before_reserve", "cash_after_reserve", "reserve_planned", "reserve_funded",
                 "reserve_shortfall", "reserve_miner_contribution", "depreciation_miner"):
        summary[f"{name}_mean"] = float(arr[name].mean())
        summary[f"{name}_median"] = float(np.median(arr[name]))
    if any(not finite_number(v) for v in summary.values() if v is not None and not isinstance(v, (str, bool))):
        raise InputError("Summary arithmetic exceeded finite numeric limits.")

    # ---- staged-N stability table ----
    rows = []
    for n_k in sorted(set([s for s in STAGES if s <= N] + list(checks) + [N])):
        if n_k in checks:
            row = checks[n_k]
        elif n_k == N:
            row = final_check
        else:
            prefix = {name: values[:n_k] for name, values in arr.items()}
            pk = _prob_counts(prefix, inp.W_opp)
            row = convergence_check(inp, prefix, pk, n_k, tol=tol, abs_tol=return_abs_tol,
                                    rel_tol=return_rel_tol)
            row["converged"] = False
            row["convergence_reasons"] += "; descriptive prefix, not a stopping check"
        rows.append(row)
    stage_table = pd.DataFrame(rows)

    # ---- rank-based global sensitivity (Spearman) ----
    m = min(sens_rows, N)
    sens: Dict[str, float] = {}
    candidates = {"yield_Y": arr["Y"][:m], "valuation_multiplier": arr["q"][:m], "price_P": arr["P"][:m]}
    for name in TIMING_KEYS:
        if name in arr:
            candidates[name] = arr[name][:m]
    for name, a in arr.items():
        if name.startswith("rate::"):
            candidates[name.replace("rate::", "rate_")] = a[:m]
        if name.startswith("cost::"):
            candidates[name.replace("cost::", "cost_")] = a[:m]
    for name, a in candidates.items():
        if m >= 2 and np.ptp(a) > 0 and np.ptp(E[:m]) > 0:
            sens[name] = float(stats.spearmanr(a, E[:m]).statistic)
    sens = dict(sorted(sens.items(), key=lambda kv: -abs(kv[1])))

    manifest = build_manifest(inp, seed=seed, batch_size=batch_size, n_min=n_min, n_max=n_max,
                              tol=tol, n_fixed=n_fixed, sens_rows=sens_rows, N=N, converged=converged,
                              return_abs_tol=return_abs_tol, return_rel_tol=return_rel_tol,
                              probability_precision_met=final_check["probability_precision_met"],
                              return_precision_met=final_check["return_precision_met"],
                              convergence_reasons=final_check["convergence_reasons"])
    manifest["numerical_diagnostics"] = _jsonable(final_check)
    manifest["stopping_reason"] = ("fixed_draw_count" if n_fixed is not None else
                                   "converged" if converged else "maximum_draw_cap")
    if inp.zero.p0 is None:
        lower, upper = stats.beta.ppf([.025, .975], inp.zero.prior_a + inp.zero.z,
                                      inp.zero.prior_b + inp.zero.n - inp.zero.z)
        summary.update(p0_posterior_lower_95=float(lower), p0_posterior_upper_95=float(upper))
    return Result(summary=summary, stage_table=stage_table, rank_sensitivity=sens,
                  arrays=arr, manifest=manifest, warnings=warns)


# ----------------------------------------------------------------------------------------
# Run manifest (expert report 3.10: preserve code version, input version, configuration)
# ----------------------------------------------------------------------------------------
def _jsonable(o):
    """Recursively convert dataclasses and NumPy scalars into plain JSON-serialisable types."""
    if dataclasses.is_dataclass(o):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(o).items()}
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def source_identity():
    """Hash all application sources and dependency declarations, even outside Git."""
    folder = Path(__file__).resolve().parent if globals().get("__file__") else None
    identity = {"source_sha256": {}, "git_revision": None, "git_dirty": None}
    if folder is None:
        return identity
    excluded = {"__pycache__", ".git", ".venv", "venv", "site-packages", "node_modules"}
    for directory, subdirs, files in os.walk(folder):
        subdirs[:] = sorted(name for name in subdirs if name not in excluded)
        for name in sorted(files):
            path = Path(directory) / name
            if path.is_file() and (path.suffix == ".py" or path.name.startswith("requirements")):
                identity["source_sha256"][str(path.relative_to(folder))] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        identity["git_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=folder, stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
        identity["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=folder, stderr=subprocess.DEVNULL, text=True, timeout=5).strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return identity


def _package_version(name: str) -> Optional[str]:
    """Describe optional UI packages without requiring them for engine-only use."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_manifest(inp: Inputs, **cfg) -> Dict[str, object]:
    """Build the reproducibility manifest: code/input hashes, resolved calendar days, accounting basis, run config."""
    inputs_json = json.dumps(_jsonable(inp), sort_keys=True, default=str)
    code_hash = "unavailable"            # stays so if the code was pasted into a notebook cell (no __file__)
    src = globals().get("__file__")
    if src:
        try:
            with open(src, "rb") as fh:
                code_hash = hashlib.sha256(fh.read()).hexdigest()
        except OSError:
            pass
    return {
        "engine_version": ENGINE_VERSION,
        "export_schema_version": 2,
        **source_identity(),
        "engine_sha256": code_hash,
        "inputs_sha256": hashlib.sha256(inputs_json.encode()).hexdigest(),
        "inputs": json.loads(inputs_json),
        "provenance": dict(inp.provenance),
        **({"resolved_calendar_days_per_cycle": inp.calendar_days(),
            "calendar_days_assumed_from_shifts": inp.calendar_days_per_cycle is None}
           if inp.batch_timing is None else {}),
        "timing_mode": "batch" if inp.batch_timing is not None else "fixed",
        "timing_rules": {"work_days": "floor(PERT + 0.5), whole days 1..3650",
                         "calendar_days": "working_days + independent nonnegative PERT idle_days",
                         "schedule": "independent daily categorical count; one conditional duration per day; full crew",
                         "timing_rng": "SeedSequence(seed).spawn(2)[0]",
                         "households": "independent count; daily need less nonmining income, clipped at zero"} if inp.batch_timing else {},
        "expense_bases": {c.name: c.basis for c in inp.all_cost_items()},
        "cost_components": {c.name: c.components if c.components is not None else [c.name]
                            for c in inp.all_cost_items()},
        "dataset": {"id": inp.dataset_id, "sha256": inp.dataset_sha256} if inp.dataset_id else None,
        "metadata": _jsonable(inp.metadata),
        "random_streams": {"economic": "default_rng(seed)",
                           "timing": "default_rng(SeedSequence(seed).spawn(2)[0])",
                           "farmgate_assay": "default_rng(SeedSequence(seed).spawn(2)[1])"},
        "runtime_versions": {name: _package_version(name) for name in
                             ("numpy", "scipy", "pandas", "matplotlib", "streamlit")},
        "numerical_error_scope": "estimated pointwise sampling errors, not model or predictive intervals",
        "convergence_rules": {"consecutive_checks": 2, "component_minimum": 30,
                              "return_metrics": list(RETURN_KEYS),
                              "mean_error": "2 * sample standard error (ddof=1)",
                              "quantile_interval": "pointwise 95% binomial order-statistic interval",
                              "expected_shortfall_error": "2 * SE(min(E, empirical P5)) / 0.05",
                              "return_tolerance": "max(abs_tol, rel_tol * abs(statistic))"},
        "supported_scope": {
            "attendance": "full active crew in batch mode; fixed mode may supply an hours matrix",
            "household_requirement": "households * calendar_days * max(daily_need - nonmining_income, 0)",
            "household_default": "None means one per active worker; explicit zero means zero",
            "sharing": "aggregate miner/non-miner gross or net contract; net deducts all operating costs before sharing",
            "excluded": ["reimbursements", "full stakeholder ledgers", "changing batch membership"],
        },
        "correlation_order": [c.name for c in inp.stochastic_cost_items()],
        "correlation_scope": "expense rates before exposure scaling",
        "assumptions": ["batch yield, valuation, and timing independent",
                        "batch: all active contributors attend every scheduled shift; no changing membership",
                        "worksite food is additional consumption excluded from household needs",
                        "operating hours equal summed scheduled shift durations"],
        "valuation_mass_label": "fine gold mass" if inp.price_basis == "reference" else "recovered gold mass",
        "accounting_basis": {"Pi": "economic_profit", "E": "economic_profit_per_person_hour",
                             "Y_HH": "economic_profit_household_threshold",
                             "reserve_amounts": "common_fund" if inp.reserve_location == "before_sharing" else "miner_fund",
                             "reserve_horizon": "single_cycle_no_balance_or_borrowing"},
        "run_config": _jsonable(cfg),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ----------------------------------------------------------------------------------------
# Scenario grid (parameter-uncertainty layer: p0, PERT mode, jackpot shape, crew size)
# ----------------------------------------------------------------------------------------
def scenario_table(base: Inputs, *, p0: Optional[Sequence[float]] = None,
                   mode_frac: Optional[Sequence[float]] = None,
                   jackpot_shape: Optional[Sequence[float]] = None,
                   n_c: Optional[Sequence[int]] = None,
                   seed: int = 42, **run_kwargs) -> pd.DataFrame:
    """Run the engine over a grid of ASSUMED parameters and tabulate the spread of key outputs.

    mode_frac sets the ordinary-yield PERT mode at min + frac * (max - min).
    Each scenario reuses the same seed, so differences come from the parameter, not the noise.
    """
    p0s = list(p0) if p0 is not None else [None]
    mfs = list(mode_frac) if mode_frac is not None else [None]
    shs = list(jackpot_shape) if jackpot_shape is not None else [None]
    ncs = list(n_c) if n_c is not None else [None]
    rows = []
    for a_p0, a_mf, a_sh, a_n in itertools.product(p0s, mfs, shs, ncs):
        inp = dataclasses.replace(base)
        if a_p0 is not None:
            inp = dataclasses.replace(inp, zero=ZeroYield(p0=a_p0))
        if a_mf is not None:
            y = inp.yield_ord
            inp = dataclasses.replace(inp, yield_ord=PERT(y.a, y.a + a_mf * (y.b - y.a), y.b))
        if a_sh is not None and inp.jackpot is not None:
            j = inp.jackpot
            inp = dataclasses.replace(inp, jackpot=Jackpot(j.threshold, j.cap, a_sh))
        if a_n is not None:
            inp = dataclasses.replace(inp, n_c=a_n, hours_matrix=None)
        r = run(inp, seed=seed, **run_kwargs)
        s = r.summary
        rows.append({
            "p0": a_p0, "mode_frac": a_mf, "jackpot_shape": a_sh, "n_c": a_n,
            "N": s["N"], "converged": s["converged"],
            **{k: s[k] for k in PROB_KEYS},
            "E_median": s["E_median"], "E_mean": s["E_mean"],
            **{f"{name}_mean": s[f"{name}_mean"] for name in
               ("cash_before_reserve", "cash_after_reserve", "reserve_planned", "reserve_funded", "reserve_shortfall")},
            "Y_BE_median": s["Y_BE_median"], "Y_W_median": s["Y_W_median"], "Y_HH_median": s["Y_HH_median"],
        })
    return pd.DataFrame(rows)
