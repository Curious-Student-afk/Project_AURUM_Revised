"""
scenarios.py -- named input sets for engine_v2.

IMPORTANT: every value is tagged in `provenance`:
    "pilot"     directly from the 21-interview pilot (non-representative convenience sample)
    "external"  from a cited outside source (none yet -- add citations before using)
    "assumed"   a placeholder the pilot did NOT collect; must be replaced or sensitivity-tested
    "synthetic" invented for verification only

Results from these scenarios are SCENARIO results. They are not findings about Pantukan miners.
"""
from __future__ import annotations

import dataclasses
from typing import Dict

from engine_v2 import CostItem, Inputs, Jackpot, PERT, ZeroYield


def pilot_provisional(mode_frac: float = 0.40) -> Inputs:
    """Pilot-informed baseline. Crew size, costs, p0, pi_J, mode, dependence are ASSUMPTIONS."""
    a, b = 0.33, 1.45                                   # pilot: mean of typical min / typical max (provisional)
    return Inputs(
        n_c=5,                                          # assumed typical crew (not collected)
        D_c=1,                                          # cycle = one shift for the pilot data
        hours_per_worker_shift=7.73,                    # pilot mean (n=20)
        s_m=0.658,                                      # pilot non-miner share 34.2% -> miner group share
        tribute_base="gross",                           # assumed: revenue base was not asked
        yield_ord=PERT(a, a + mode_frac * (b - a), b),  # mode is assumed
        zero=ZeroYield(p0=0.10),                        # assumed (zero-yield counts not collected)
        pi_J=0.01,                                      # assumed
        jackpot=Jackpot(threshold=14.4, cap=24.0, shape=1.5),  # pilot range (n=3 respondents); shape assumed
        price_basis="farmgate",
        price=PERT(6050.0, 8400.0, 8400.0),             # pilot min / most frequent / max farmgate price
        costs=[
            CostItem("fixed_operating", miner_share=1.0, fixed=400.0),        # assumed (legacy default)
            CostItem("fuel", miner_share=1.0, pert=PERT(600.0, 800.0, 1100.0)),       # assumed
            CostItem("transport_hauling", miner_share=1.0, pert=PERT(100.0, 200.0, 400.0)),  # assumed
        ],
        rho_spearman=[[1.0, 0.5], [0.5, 1.0]],          # assumed fuel-transport dependence
        dependence="copula",
        W_opp=65.63,                                    # fill W_opp_source / W_opp_date before reporting
        hh_need_per_household_day=1075.0,               # pilot data table mean (manuscript says 1,105: reconcile)
        nonmining_income_per_household_day=0.0,         # assumed (not collected)
        provenance={
            "n_c": "assumed", "D_c": "pilot", "hours_per_worker_shift": "pilot", "s_m": "pilot",
            "tribute_base": "assumed", "yield_ord.a,b": "pilot (provisional)", "yield_ord.m": "assumed",
            "zero.p0": "assumed", "pi_J": "assumed", "jackpot.threshold,cap": "pilot (n=3)",
            "jackpot.shape": "assumed", "price": "pilot (mode = most frequent reported price)",
            "costs": "assumed", "rho_spearman": "assumed", "W_opp": "pilot/manuscript (source pending)",
            "hh_need_per_household_day": "pilot", "nonmining_income": "assumed",
        },
    )


def _scale_costs(inp: Inputs, factor: float) -> Inputs:
    new = []
    for c in inp.costs:
        if c.fixed is not None:
            new.append(dataclasses.replace(c, fixed=c.fixed * factor))
        else:
            p = c.pert
            new.append(dataclasses.replace(c, pert=PERT(p.a * factor, p.m * factor, p.b * factor)))
    return dataclasses.replace(inp, costs=new)


def synthetic_profiles() -> Dict[str, Inputs]:
    """Chapter 3.4 test bench: Baseline, High-Cost, Low-Purity (verification only)."""
    base = pilot_provisional()
    high_cost = _scale_costs(base, 1.5)
    high_cost.provenance = {**base.provenance, "costs": "synthetic (1.5x baseline)"}
    low_purity = dataclasses.replace(
        base,
        price_basis="reference",
        price=8400.0,                                   # quoted pure-gold reference price
        purity=PERT(0.75, 0.82, 0.90),                  # synthetic low-purity dore
        provenance={**base.provenance, "purity": "synthetic", "price": "synthetic (quoted reference)"},
    )
    return {"Baseline": base, "High-Cost": high_cost, "Low-Purity": low_purity}
