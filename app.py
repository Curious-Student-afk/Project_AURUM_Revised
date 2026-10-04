"""
engine_v2.py -- Project AURUM V2.0 zero-aware stochastic engine.

Pure NumPy/SciPy (no Streamlit), so it can be tested, run offline, and reused by any interface.

ACCOUNTING UNIT: one processing cycle c (a cycle = D_c shifts). Every input is expressed
per cycle at GROUP level, then converted to a per-person-hour return:

    F   = Y * q                         fine-gold equivalent (g)
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
("pilot", "external", "assumed", "synthetic") that is written to the run manifest.

Scope limit: dependence is modelled among cost categories only (Gaussian copula built from
a Spearman matrix). Yield-cost dependence is NOT modelled (no data) -- state this as a limitation.
"""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import scipy
from scipy import stats

ENGINE_VERSION = "2.0.0-draft"

LABELS = {
    "P_negative_operating_return": "Probability of Negative Operating Return",
    "P_labor_return_below_wage": "Probability of Labor Return Below Wage Benchmark",
    "P_yield_below_breakeven": "Probability of Yield Below Operating Break-Even",
    "P_fail_household_adequacy": "Probability of Failing Household-Adequacy Threshold",
}
PROB_KEYS = tuple(LABELS.keys())


class InputError(ValueError):
    """Raised when inputs are impossible or ambiguous (expert report, Section 8)."""


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
        return self.b == self.a

    @property
    def alpha(self) -> float:
        return 1.0 + 4.0 * (self.m - self.a) / (self.b - self.a)

    @property
    def beta(self) -> float:
        return 1.0 + 4.0 * (self.b - self.m) / (self.b - self.a)

    @property
    def mean(self) -> float:
        return (self.a + 4.0 * self.m + self.b) / 6.0

    def check(self, name: str, errs: List[str], strictly_positive: bool = False,
              upper: Optional[float] = None) -> None:
        if not (self.a <= self.m <= self.b):
            errs.append(f"{name}: PERT needs min <= mode <= max (got {self.a}, {self.m}, {self.b}).")
        if strictly_positive and self.a <= 0:
            errs.append(f"{name}: minimum must be > 0 (got {self.a}).")
        if not strictly_positive and self.a < 0:
            errs.append(f"{name}: minimum must be >= 0 (got {self.a}).")
        if upper is not None and self.b > upper:
            errs.append(f"{name}: maximum must be <= {upper} (got {self.b}).")

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        if self.degenerate:
            return np.full(size, float(self.a))
        return self.a + (self.b - self.a) * rng.beta(self.alpha, self.beta, size=size)

    def ppf(self, u: np.ndarray) -> np.ndarray:
        if self.degenerate:
            return np.full(np.shape(u), float(self.a))
        return self.a + (self.b - self.a) * stats.beta.ppf(u, self.alpha, self.beta)


@dataclass
class Jackpot:
    """Truncated Pareto jackpot magnitude on [threshold, cap] (grams per cycle).

    threshold: smallest yield counted as a jackpot (default scenario: lowest reported jackpot)
    cap:       physical upper bound (e.g. equipment/ore capacity)
    shape:     Pareto tail index; ASSUMED -- treat as a sensitivity parameter
    """
    threshold: float
    cap: float
    shape: float = 1.5

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
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
        if self.mode == "purchased":
            return (self.acquisition - self.salvage) / self.life_cycles * self.utilization
        if self.mode == "rented":
            return self.rental_rate_per_shift * D_c
        raise InputError(f"Equipment '{self.name}': mode must be 'purchased' or 'rented'.")

    def check(self, errs: List[str]) -> None:
        n = f"Equipment '{self.name}'"
        if self.mode not in ("purchased", "rented"):
            errs.append(f"{n}: mode must be 'purchased' or 'rented'.")
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
    D_c: int = 1                              # shifts in the cycle
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
    purity: Union[PERT, float] = 1.0          # used only when price_basis == "reference"

    # --- costs & dependence ---
    costs: List[CostItem] = field(default_factory=list)
    equipment: List[EquipmentItem] = field(default_factory=list)
    rho_spearman: Optional[List[List[float]]] = None  # over stochastic cost items, in order
    dependence: str = "copula"                # "copula" | "independent"

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
        if self.hours_matrix is not None:
            return float(np.sum(np.asarray(self.hours_matrix, dtype=float)))
        return float(self.n_c * self.D_c * self.hours_per_worker_shift)

    def household_requirement(self) -> float:
        """B_HH: cash the miners' households still need from mining for the cycle (>= 0)."""
        nh = self.n_c if self.n_households is None else self.n_households
        per_day = max(0.0, self.hh_need_per_household_day - self.nonmining_income_per_household_day)
        return float(nh * self.D_c * per_day)

    def all_cost_items(self) -> List[CostItem]:
        items = list(self.costs)
        for eq in self.equipment:
            items.append(CostItem(name=f"equipment:{eq.name}", miner_share=eq.miner_share,
                                  fixed=eq.cost_per_cycle(self.D_c)))
        return items

    def stochastic_cost_items(self) -> List[CostItem]:
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
        if inp.person_hours() <= 0:
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
            warns.append("Jackpot shape/threshold rest on very few reported jackpots; "
                         "treat as a sensitivity parameter.")
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
    names = set()
    for c in inp.all_cost_items():
        if c.name in names:
            errs.append(f"Cost '{c.name}' appears twice (each cost must appear exactly once).")
        names.add(c.name)
        if not (0 <= c.miner_share <= 1):
            errs.append(f"Cost '{c.name}': miner_share must be in [0, 1].")
        if (c.fixed is None) == (c.pert is None):
            errs.append(f"Cost '{c.name}': give exactly one of fixed amount or PERT.")
        if c.fixed is not None and c.fixed < 0:
            errs.append(f"Cost '{c.name}': must be >= 0 PHP per cycle.")
        if c.pert is not None:
            c.pert.check(f"Cost '{c.name}'", errs)
    for eq in inp.equipment:
        eq.check(errs)

    # dependence
    sto = inp.stochastic_cost_items()
    if inp.dependence not in ("copula", "independent"):
        errs.append("dependence must be 'copula' or 'independent'.")
    if inp.dependence == "copula" and len(sto) > 1:
        if inp.rho_spearman is None:
            errs.append("copula dependence needs rho_spearman for the stochastic costs "
                        "(or set dependence='independent').")
        else:
            r = np.asarray(inp.rho_spearman, dtype=float)
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

    if errs:
        raise InputError("\n".join(f"- {e}" for e in errs))
    return warns


# ----------------------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------------------
def _copula_chol(inp: Inputs) -> Optional[np.ndarray]:
    sto = inp.stochastic_cost_items()
    if inp.dependence != "copula" or len(sto) < 2:
        return None
    return np.linalg.cholesky(spearman_to_pearson(np.asarray(inp.rho_spearman, dtype=float)))


def sample_costs(inp: Inputs, rng: np.random.Generator, size: int,
                 chol: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """Draw every cost category for `size` cycles. Stochastic costs share a Gaussian copula."""
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
                 chol: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
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

    costs = sample_costs(inp, rng, size, chol)
    miner_share = {c.name: c.miner_share for c in inp.all_cost_items()}
    C_tot = np.zeros(size)
    C_m = np.zeros(size)
    for name, arr in costs.items():
        C_tot += arr
        C_m += miner_share[name] * arr
    return {"Y": Y, "q": q, "P": P, "C_tot": C_tot, "C_m": C_m, "costs": costs, "p0": p0}


# ----------------------------------------------------------------------------------------
# Accounting kernel (stakeholder cash flow + three thresholds)
# ----------------------------------------------------------------------------------------
def accounting_kernel(Y, q, P, C_m, C_tot, *, s_m: float, L: float, W_opp: float,
                      B_HH: float, tribute_base: str = "gross") -> Dict[str, np.ndarray]:
    """Vectorised, unit-checked accounting. Inputs are arrays per cycle (group level)."""
    F = Y * q                                  # fine-gold equivalent (g)
    R = F * P                                  # gross revenue (PHP)
    Rm = s_m * R                               # miner-group revenue
    C_eff = C_m if tribute_base == "gross" else s_m * C_tot   # costs the miner group effectively bears
    Pi = Rm - C_eff                            # miner-group profit (PHP)
    E = Pi / L                                 # PHP per person-hour
    unit_rev = q * P * s_m                     # PHP per gram of recovered gold, after the share
    return {
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


def _prob_counts(pi, E, y, y_be, y_hh, W_opp, B_HH) -> Dict[str, int]:
    return {
        "P_negative_operating_return": int(np.count_nonzero(pi < 0)),
        "P_labor_return_below_wage": int(np.count_nonzero(E < W_opp)),
        "P_yield_below_breakeven": int(np.count_nonzero(y < y_be)),
        "P_fail_household_adequacy": int(np.count_nonzero(pi < B_HH)),
    }


# ----------------------------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------------------------
@dataclass
class Result:
    summary: Dict[str, float]
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
        sens_rows: int = 50_000) -> Result:
    """Run the simulation.

    Stopping rule: keep adding batches until MCSE <= tol for EVERY reported probability and
    N >= n_min, or until n_max (then converged = False). Pass n_fixed to run exactly N draws.
    """
    warns = validate(inp)
    rng = np.random.default_rng(seed)
    chol = _copula_chol(inp)
    L = inp.person_hours()
    B_HH = inp.household_requirement()

    chunks: List[Dict[str, np.ndarray]] = []
    counts = {k: 0 for k in PROB_KEYS}
    N = 0
    converged = False
    target = n_fixed if n_fixed is not None else n_max
    while N < target:
        b = min(batch_size, target - N)
        d = sample_batch(inp, rng, b, chol)
        k = accounting_kernel(d["Y"], d["q"], d["P"], d["C_m"], d["C_tot"], s_m=inp.s_m, L=L,
                              W_opp=inp.W_opp, B_HH=B_HH, tribute_base=inp.tribute_base)
        chunk = {**{n: d[n] for n in ("Y", "q", "P", "C_tot", "C_m", "p0")}, **k,
                 **{f"cost::{n}": a for n, a in d["costs"].items()}}
        chunks.append(chunk)
        c = _prob_counts(k["Pi"], k["E"], d["Y"], k["Y_BE"], k["Y_HH"], inp.W_opp, B_HH)
        for key in PROB_KEYS:
            counts[key] += c[key]
        N += b
        if n_fixed is None and N >= n_min:
            if all(mcse_prob(counts[key], N) <= tol for key in PROB_KEYS):
                converged = True
                break
    if n_fixed is not None:
        converged = all(mcse_prob(counts[key], N) <= tol for key in PROB_KEYS)
    if not converged:
        warns.append(f"Convergence NOT reached: some MCSE > {tol} at N = {N:,}. "
                     "Report this, or raise n_max / widen tol with justification.")

    arr = {name: np.concatenate([ch[name] for ch in chunks]) for name in chunks[0]}
    E, Pi = arr["E"], arr["Pi"]

    # ---- summary ----
    probs = {key: counts[key] / N for key in PROB_KEYS}
    p5, p50, p95 = np.percentile(E, [5, 50, 95])
    tail = E[E <= p5]
    summary: Dict[str, float] = {
        "N": N, "converged": converged, "L_person_hours": L, "B_HH": B_HH,
        **probs,
        **{f"MCSE_{key}": mcse_prob(counts[key], N) for key in PROB_KEYS},
        "E_mean": float(E.mean()), "E_median": float(p50), "E_P5": float(p5), "E_P95": float(p95),
        "E_expected_shortfall_5pct": float(tail.mean()),
        "Pi_mean": float(Pi.mean()), "Pi_median": float(np.median(Pi)),
        "Y_BE_median": float(np.median(arr["Y_BE"])), "Y_W_median": float(np.median(arr["Y_W"])),
        "Y_HH_median": float(np.median(arr["Y_HH"])),
        "Y_BE_mean": float(arr["Y_BE"].mean()), "Y_W_mean": float(arr["Y_W"].mean()),
        "Y_HH_mean": float(arr["Y_HH"].mean()),
        "observed_zero_cycle_share": float(np.mean(arr["Y"] == 0)),
        "risk_band_vs_wage": risk_band(1.0 - probs["P_labor_return_below_wage"]),
    }

    # ---- staged-N stability table ----
    rows = []
    for n_k in sorted(set([s for s in STAGES if s <= N] + [N])):
        pk = {key: _prob_counts(Pi[:n_k], E[:n_k], arr["Y"][:n_k], arr["Y_BE"][:n_k],
                                arr["Y_HH"][:n_k], inp.W_opp, B_HH)[key] for key in PROB_KEYS}
        row = {"N": n_k, "E_mean": float(E[:n_k].mean()), "E_median": float(np.median(E[:n_k]))}
        for key in PROB_KEYS:
            row[key] = pk[key] / n_k
            row[f"MCSE_{key}"] = mcse_prob(pk[key], n_k)
        rows.append(row)
    stage_table = pd.DataFrame(rows)

    # ---- rank-based global sensitivity (Spearman) ----
    m = min(sens_rows, N)
    sens: Dict[str, float] = {}
    candidates = {"yield_Y": arr["Y"][:m], "purity_q": arr["q"][:m], "price_P": arr["P"][:m]}
    for name, a in arr.items():
        if name.startswith("cost::"):
            candidates[name.replace("cost::", "cost_")] = a[:m]
    for name, a in candidates.items():
        if np.ptp(a) > 0:
            sens[name] = float(stats.spearmanr(a, E[:m]).statistic)
    sens = dict(sorted(sens.items(), key=lambda kv: -abs(kv[1])))

    manifest = build_manifest(inp, seed=seed, batch_size=batch_size, n_min=n_min, n_max=n_max,
                              tol=tol, n_fixed=n_fixed, N=N, converged=converged)
    return Result(summary=summary, stage_table=stage_table, rank_sensitivity=sens,
                  arrays=arr, manifest=manifest, warnings=warns)


# ----------------------------------------------------------------------------------------
# Run manifest (expert report 3.10: preserve code version, input version, configuration)
# ----------------------------------------------------------------------------------------
def _jsonable(o):
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


def build_manifest(inp: Inputs, **cfg) -> Dict[str, object]:
    inputs_json = json.dumps(_jsonable(inp), sort_keys=True, default=str)
    try:
        with open(__file__, "rb") as fh:
            code_hash = hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        code_hash = "unavailable"
    return {
        "engine_version": ENGINE_VERSION,
        "engine_sha256": code_hash,
        "inputs_sha256": hashlib.sha256(inputs_json.encode()).hexdigest(),
        "inputs": json.loads(inputs_json),
        "provenance": dict(inp.provenance),
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
            "Y_BE_median": s["Y_BE_median"], "Y_W_median": s["Y_W_median"], "Y_HH_median": s["Y_HH_median"],
        })
    return pd.DataFrame(rows)
