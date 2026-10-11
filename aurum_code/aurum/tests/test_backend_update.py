"""Regression and accounting checks for the focused V2.1 update."""
import dataclasses
import io
import json
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import engine_v2 as E
import dashboard_logic as DL
from scenarios import pilot_provisional


def scenario(**kw):
    """Return the pilot-provisional Inputs with the given fields overridden."""
    return dataclasses.replace(pilot_provisional(), **kw)


def deterministic(**kw):
    """Build a single-crew, zero-variance scenario with known cash and depreciation totals.

    R=1000, cash cost total=400 / miner=150, depreciation total=100 / miner=20.
    """
    return scenario(n_c=1, hours_per_worker_shift=10, s_m=0.6,
                    yield_ord=E.PERT(10, 10, 10), zero=E.ZeroYield(p0=0), pi_J=0,
                    price=100., costs=[E.CostItem('operations', 0.375, fixed=400.)],
                    equipment=[E.EquipmentItem('mill', 'purchased', miner_share=0.2,
                                               acquisition=1000., life_cycles=10.)],
                    rho_spearman=None, hh_need_per_household_day=300., **kw)


class TestValidationBoundary(unittest.TestCase):
    """Stricter validation: non-finite, non-numeric, and malformed inputs must raise InputError."""

    def test_nonfinite_and_nonnumeric_inputs_are_rejected(self):
        """NaN, infinity, booleans, strings and None are rejected for finite numeric fields."""
        for field in ('price', 'W_opp', 'hours_per_worker_shift', 'calendar_days_per_cycle',
                      'hh_need_per_household_day', 'nonmining_income_per_household_day'):
            for value in (float('nan'), float('inf'), -float('inf'), True, '10', None):
                if field == 'calendar_days_per_cycle' and value is None:
                    continue
                with self.subTest(field=field, value=value), self.assertRaises(E.InputError):
                    E.run(scenario(**{field: value}), n_fixed=10)

    def test_integer_counts_reject_fractions_and_bools(self):
        """Crew/shift/household counts and zero-yield counts must be true integers, not bools or fractions."""
        for field in ('n_c', 'D_c', 'n_households'):
            for value in (1.5, 2.0, True, np.bool_(True)):
                with self.subTest(field=field, value=value), self.assertRaises(E.InputError):
                    E.validate(scenario(**{field: value}))
        for z, n in ((1.5, 10), (1, 10.5), (True, 10)):
            with self.assertRaises(E.InputError):
                E.validate(scenario(zero=E.ZeroYield(z=z, n=n)))

    def test_distribution_and_equipment_numeric_guards(self):
        """Non-finite values nested in PERTs, costs, jackpots, zero-yield priors, equipment and rho are rejected."""
        cases = [dict(price=E.PERT(1, float('nan'), 3)),
                 dict(costs=[E.CostItem('x', fixed=float('nan'))]),
                 dict(costs=[E.CostItem('x', pert=E.PERT(0, 1, float('inf')))]),
                 dict(jackpot=E.Jackpot(1, 2, float('nan'))),
                 dict(zero=E.ZeroYield(z=1, n=2, prior_a=float('inf'))),
                 dict(equipment=[E.EquipmentItem('x', 'purchased', life_cycles=float('nan'))]),
                 dict(rho_spearman=[[1, float('nan')], [float('nan'), 1]])]
        for fields in cases:
            with self.subTest(fields=fields), self.assertRaises(E.InputError):
                E.validate(scenario(**fields))

    def test_hours_matrix_requires_exact_finite_shape_and_positive_total(self):
        """hours_matrix must match (D_c, n_c), contain only finite nonnegative hours, and sum > 0."""
        for matrix in ([[8]], [[8]*5], [[float('nan')]*5]*2,
                       [[True]*5]*2, [['8']*5]*2, [[0]*5]*2, [[8, 8], [8]]):
            with self.subTest(matrix=matrix), self.assertRaises(E.InputError):
                E.validate(scenario(n_c=5, D_c=2, hours_matrix=matrix))
        good = scenario(n_c=2, D_c=2, hours_matrix=[[8, 0], [4, 6]])
        E.validate(good)
        self.assertEqual(good.person_hours(), 18)

    def test_run_configuration_rejected_before_sampling(self):
        """Invalid run() configuration options raise InputError before any sampling occurs."""
        cases = [dict(batch_size=0), dict(batch_size=-1), dict(n_fixed=0), dict(n_max=0),
                 dict(n_min=0), dict(seed=True), dict(seed=-1), dict(sens_rows=1),
                 dict(n_fixed=2.5), dict(n_max=float('inf')), dict(tol=float('nan')), dict(tol=0)]
        with patch.object(E, 'sample_batch', side_effect=AssertionError('must not sample')):
            for kw in cases:
                with self.subTest(kw=kw), self.assertRaises(E.InputError):
                    E.run(scenario(), **kw)

    def test_arithmetic_overflow_cannot_become_zero_risk(self):
        """Overflowing arithmetic raises InputError rather than silently reporting zero risk."""
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            with self.assertRaises(E.InputError):
                E.run(scenario(price=1e308, yield_ord=E.PERT(100, 100, 100), pi_J=0), n_fixed=20)
        with self.assertRaises(E.InputError):
            E.validate(scenario(n_c=10, hours_per_worker_shift=1e308))

    def test_dashboard_does_not_truncate_counts_or_coerce_bad_numbers(self):
        """The dashboard layer forwards bad numbers to the engine instead of coercing them."""
        defaults = DL.preset_values(DL.PRESETS[0])
        for key, value in (('n_c', 5.4), ('D_c', 1.5), ('n_hh', True), ('p_min', '100'),
                           ('seed', 2.5), ('n_max', 100.5), ('tol', True)):
            with self.subTest(key=key), self.assertRaises(E.InputError):
                DL.run_v2({**defaults, key: value}, DL.PRESETS[0])
        with self.assertRaises(E.InputError):
            DL.build_inputs({**defaults, 'p0_mode': 'from counts', 'z': 0.5})


class TestCalendarAndTail(unittest.TestCase):
    """Calendar-duration accounting and the exact lower-tail expected-shortfall estimator."""

    def test_calendar_changes_household_not_labor_or_rental(self):
        """calendar_days_per_cycle scales the household requirement only, not labor or rentals."""
        a = scenario(D_c=2, calendar_days_per_cycle=1., n_households=1,
                     hh_need_per_household_day=100., nonmining_income_per_household_day=20.,
                     equipment=[E.EquipmentItem('pump', 'rented', rental_rate_per_shift=10)])
        b = dataclasses.replace(a, calendar_days_per_cycle=3.)
        self.assertEqual(a.household_requirement(), 80)
        self.assertEqual(b.household_requirement(), 240)
        self.assertEqual(a.person_hours(), b.person_hours())
        self.assertEqual(a.all_cost_items()[-1].fixed, 20)
        self.assertEqual(b.all_cost_items()[-1].fixed, 20)
        self.assertEqual(dataclasses.replace(a, calendar_days_per_cycle=0.5).household_requirement(), 40)

    def test_legacy_fallback_is_explicit_in_manifest(self):
        """Omitting calendar_days_per_cycle is recorded and warned about in the manifest."""
        a = scenario(D_c=2, calendar_days_per_cycle=None)
        r = E.run(a, n_fixed=10)
        self.assertEqual(r.manifest['resolved_calendar_days_per_cycle'], 2)
        self.assertTrue(r.manifest['calendar_days_assumed_from_shifts'])
        self.assertTrue(any('one calendar day per shift' in w for w in r.warnings))

    def test_exact_lower_tail(self):
        """expected_shortfall averages exactly the lowest fraction of mass, including boundary ties."""
        self.assertEqual(E.expected_shortfall(np.array([-100.] + [0.]*9 + [100.]*90)), -20)
        self.assertEqual(E.expected_shortfall(np.full(100, -7.)), -7)
        self.assertAlmostEqual(E.expected_shortfall(np.arange(110.)), (0+1+2+3+4+0.5*5)/5.5)
        self.assertEqual(E.expected_shortfall(np.array([4.])), 4)
        self.assertEqual(E.expected_shortfall(np.array([3., -2.])), -2)


class TestReserveAccounting(unittest.TestCase):
    """Per-cycle cash and replacement-reserve accounting across contract, fund and priority settings."""

    def test_cash_and_economic_profit_are_distinct(self):
        """With no reserve enabled, cash before/after reserve equal profit plus depreciation."""
        r = E.run(deterministic(), n_fixed=100)
        np.testing.assert_allclose(r.arrays['Pi'], 430)
        np.testing.assert_allclose(r.arrays['cash_before_reserve'], 450)
        np.testing.assert_allclose(r.arrays['cash_after_reserve'], 450)
        np.testing.assert_allclose(r.arrays['reserve_funded'], 0)
        np.testing.assert_allclose(r.arrays['cash_cost_total'], 400)
        self.assertEqual(r.summary['P_household_and_reserve_met'], 1)

    def test_depreciation_can_fail_economic_adequacy_while_cash_succeeds(self):
        """Depreciation can push economic profit below the household requirement while cash still covers it."""
        inp = dataclasses.replace(deterministic(), hh_need_per_household_day=440)
        r = E.run(inp, n_fixed=100)
        self.assertEqual(r.summary['P_fail_household_adequacy'], 1)
        self.assertEqual(r.summary['P_cash_before_reserve_below_household'], 0)

    def test_all_contract_location_priority_combinations(self):
        """Every tribute-base, fund-location and priority combination funds and conserves cash correctly."""
        # Expected (funded selected-fund contribution, miner cash after reserve).
        expected = {
            ('gross', 'before_sharing', 'reserve_first'): (500, 150),
            ('gross', 'before_sharing', 'household_first'): (250, 300),
            ('gross', 'after_sharing', 'reserve_first'): (450, 0),
            ('gross', 'after_sharing', 'household_first'): (150, 300),
            ('net', 'before_sharing', 'reserve_first'): (500, 60),
            ('net', 'before_sharing', 'household_first'): (100, 300),
            ('net', 'after_sharing', 'reserve_first'): (360, 0),
            ('net', 'after_sharing', 'household_first'): (60, 300),
        }
        for (base, location, priority), (funded, after) in expected.items():
            with self.subTest(base=base, location=location, priority=priority):
                r = E.run(deterministic(tribute_base=base, reserve_mode='custom',
                          reserve_location=location, reserve_priority=priority,
                          reserve_custom_per_cycle=500), n_fixed=100)
                np.testing.assert_allclose(r.arrays['reserve_funded'], funded)
                np.testing.assert_allclose(r.arrays['reserve_shortfall'], 500-funded)
                np.testing.assert_allclose(r.arrays['cash_after_reserve'], after)
                np.testing.assert_allclose(r.arrays['Pi'], 430 if base == 'gross' else 300)
                # Cash conservation across miners and all other stakeholders.
                a = r.arrays
                other_before = (1-.6)*a['R'] - (a['cash_cost_total']-(150 if base == 'gross' else 240))
                other_after = other_before - ((1-.6)*a['reserve_funded'] if location == 'before_sharing' else 0)
                np.testing.assert_allclose(a['cash_after_reserve']+other_after+a['reserve_funded'], a['R']-a['cash_cost_total'])
                self.assertEqual(r.summary['P_cash_after_reserve_below_household'], float(priority == 'reserve_first'))
                self.assertEqual(r.summary['P_household_and_reserve_met'], 0)

    def test_household_first_preserves_adequacy_at_floating_point_boundary(self):
        """household_first never drops cash below B_HH due to floating-point subtraction of large values."""
        revenue = np.random.default_rng(12).uniform(100, 1e6, 10000)
        for location in ('before_sharing', 'after_sharing'):
            a = E.accounting_kernel(revenue, np.ones(10000), np.ones(10000),
                    np.zeros(10000), np.zeros(10000), s_m=.658, L=40., W_opp=65., B_HH=1075.,
                    reserve_mode='custom', reserve_custom_per_cycle=1e6,
                    reserve_location=location, reserve_priority='household_first')
            np.testing.assert_array_equal(a['cash_before_reserve'] >= 1075., a['cash_after_reserve'] >= 1075.)

    def test_match_depreciation_respects_fund_and_contract(self):
        """match_depreciation targets the correct amount for the selected fund and tribute base."""
        for base in ('gross', 'net'):
            for location in ('before_sharing', 'after_sharing'):
                r = E.run(deterministic(tribute_base=base, reserve_mode='match_depreciation',
                          reserve_location=location, reserve_priority='reserve_first'), n_fixed=100)
                planned = 100 if location == 'before_sharing' else (20 if base == 'gross' else 60)
                np.testing.assert_allclose(r.arrays['reserve_planned'], planned)
                self.assertEqual(r.summary['P_reserve_underfunded'], 0)
                self.assertEqual(r.summary['P_household_and_reserve_met'], 1)

    def test_common_reserve_cannot_use_cash_owed_for_operating_costs(self):
        """A before-sharing reserve cannot be funded from cash still owed for operating costs."""
        inp = dataclasses.replace(deterministic(), yield_ord=E.PERT(3, 3, 3),
                                  reserve_mode='custom', reserve_custom_per_cycle=100,
                                  reserve_location='before_sharing', reserve_priority='reserve_first')
        r = E.run(inp, n_fixed=100)
        self.assertGreater(r.summary['cash_before_reserve_mean'], 0)  # miners' share still positive
        np.testing.assert_allclose(r.arrays['reserve_funded'], 0)      # operation cash is negative
        np.testing.assert_allclose(r.arrays['reserve_shortfall'], 100)

    def test_losses_never_borrow_to_fund_reserve(self):
        """When a cycle has no cash available, the reserve goes unfunded rather than borrowing."""
        for base in ('gross', 'net'):
            for location in ('before_sharing', 'after_sharing'):
                inp = dataclasses.replace(deterministic(), zero=E.ZeroYield(p0=1), tribute_base=base,
                          reserve_mode='custom', reserve_custom_per_cycle=100,
                          reserve_location=location, reserve_priority='household_first')
                r = E.run(inp, n_fixed=100)
                np.testing.assert_allclose(r.arrays['reserve_funded'], 0)
                np.testing.assert_allclose(r.arrays['cash_before_reserve'], r.arrays['cash_after_reserve'])
                self.assertEqual(r.summary['P_reserve_underfunded'], 1)

    def test_zero_depreciation_and_rentals_have_no_replacement_target(self):
        """Rented-only equipment has zero depreciation, so match_depreciation plans no reserve."""
        inp = dataclasses.replace(deterministic(), equipment=[E.EquipmentItem('pump', 'rented', rental_rate_per_shift=50)],
                    reserve_mode='match_depreciation', reserve_location='before_sharing', reserve_priority='reserve_first')
        r = E.run(inp, n_fixed=100)
        np.testing.assert_allclose(r.arrays['reserve_planned'], 0)
        np.testing.assert_allclose(r.arrays['cash_cost_total'], 450)
        np.testing.assert_allclose(r.arrays['Pi'], r.arrays['cash_before_reserve'])

    def test_enabled_reserve_requires_policy_and_valid_target(self):
        """Enabling a reserve without an explicit valid location, priority and target raises InputError."""
        for fields in ({'reserve_mode':'custom'}, {'reserve_mode':'bad'},
                       {'reserve_mode':'custom', 'reserve_location':'after_sharing', 'reserve_priority':'reserve_first', 'reserve_custom_per_cycle':float('nan')}):
            with self.assertRaises(E.InputError):
                E.validate(scenario(**fields))

    def test_cash_probabilities_participate_in_convergence_and_exports(self):
        """Cash/reserve probabilities converge like other risk keys and appear in exports and the scenario grid."""
        inp = scenario(reserve_mode='custom', reserve_custom_per_cycle=500,
                       reserve_location='after_sharing', reserve_priority='household_first')
        r = E.run(inp)
        for key in E.PROB_KEYS:
            self.assertIn(key, r.stage_table)
            self.assertLessEqual(r.summary['MCSE_'+key], .005)
            self.assertAlmostEqual(r.stage_table.iloc[-1][key], r.summary[key])
        self.assertEqual(r.manifest['inputs']['reserve_priority'], 'household_first')
        table = pd.read_csv(io.BytesIO(DL.export_csv(r, 10)))
        self.assertIn('cash_after_reserve_php', table)
        self.assertIn('reserve_shortfall_php', table)
        self.assertEqual(json.loads(DL.export_manifest(r))['accounting_basis']['Pi'], 'economic_profit')
        grid = E.scenario_table(inp, p0=[0, .1], n_fixed=100)
        self.assertIn('reserve_shortfall_mean', grid)
        self.assertIn('P_household_and_reserve_met', grid)


if __name__ == '__main__':
    unittest.main()
