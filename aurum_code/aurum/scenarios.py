"""Independent synthetic demonstrations; no defaults represent measured mining conditions."""
from __future__ import annotations

import dataclasses
from typing import Dict
from engine_v2 import BatchTiming, CostItem, Inputs, Jackpot, PERT, ZeroYield


def default_timing() -> BatchTiming:
    return BatchTiming(PERT(2, 3, 5), PERT(0, 1, 3), (6, 3, 1),
                       {1: PERT(6, 8, 10), 2: PERT(3, 4, 6), 3: PERT(2, 3, 4)})


def fixed_synthetic(mode_frac: float = 1 / 3) -> Inputs:
    """Small fixed-timing example for compatibility and regression verification."""
    return Inputs(
        n_c=4, D_c=1, calendar_days_per_cycle=1., hours_per_worker_shift=8.,
        s_m=.5, yield_ord=PERT(5, 5 + 15 * mode_frac, 20), zero=ZeroYield(p0=.1),
        pi_J=.02, jackpot=Jackpot(30, 60, 1.5), price=PERT(5000, 5500, 6000),
        costs=[CostItem('fixed_operating', fixed=500.),
               CostItem('fuel', pert=PERT(600, 900, 1200)),
               CostItem('transport_hauling', pert=PERT(200, 300, 400))],
        rho_spearman=[[1., .5], [.5, 1.]], W_opp=100., hh_need_per_household_day=800.,
        provenance={key: 'synthetic' for key in ('n_c', 's_m', 'yield_ord', 'zero', 'pi_J', 'jackpot',
                    'price', 'costs', 'rho_spearman', 'W_opp', 'hh_need_per_household_day')})


def batch_synthetic() -> Inputs:
    inp = dataclasses.replace(fixed_synthetic(), batch_timing=default_timing(), calendar_days_per_cycle=None,
        costs=[CostItem('fuel', pert=PERT(50, 75, 100), basis='per_operating_hour'),
               CostItem('milling', pert=PERT(500, 750, 1000)),
               CostItem('explosives', pert=PERT(100, 150, 200), basis='per_working_day'),
               CostItem('transport', pert=PERT(200, 300, 400)),
               CostItem('food', pert=PERT(100, 150, 200), basis='per_calendar_day')],
        rho_spearman=[[float(i == j) for j in range(5)] for i in range(5)])
    inp.provenance = {**inp.provenance, 'batch_timing': 'synthetic'}
    return inp


def _scale_costs(inp: Inputs, factor: float) -> Inputs:
    costs = [dataclasses.replace(c, fixed=c.fixed * factor) if c.fixed is not None else
             dataclasses.replace(c, pert=PERT(c.pert.a * factor, c.pert.m * factor, c.pert.b * factor))
             for c in inp.costs]
    return dataclasses.replace(inp, costs=costs)


def synthetic_profiles() -> Dict[str, Inputs]:
    base = batch_synthetic()
    return {'Baseline': base, 'High-Cost': _scale_costs(base, 1.5),
            'Low-Purity': dataclasses.replace(base, price_basis='reference', purity=PERT(.75, .85, .95))}


def legacy_synthetic_values() -> dict:
    """Independent illustrative inputs for exercising the unchanged V1 equations."""
    return dict(H=8, T=.8, Cfamily=90., W=100., Cfixed=1200., delta=1.1,
                Cfuel=800., alpha=.02, Gmin=5., Gsub=20., Gjack=60., eta=.9,
                Pmin=5000., Pmax=6000., N=100_000)
