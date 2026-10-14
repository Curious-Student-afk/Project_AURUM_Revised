"""Batch schedule, exposure, and accounting acceptance checks using synthetic fixtures."""
import dataclasses
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import stats

import engine_v2 as E
import dashboard_logic as DL
from scenarios import batch_synthetic, fixed_synthetic, default_timing


def constant(x):
    """Return a degenerate PERT(x, x, x) that always samples to x."""
    return E.PERT(x, x, x)


def deterministic(**kw):
    """Build a zero-variance batch scenario with known timing, costs and household need."""
    base = dataclasses.replace(batch_synthetic(), n_c=2,
        batch_timing=E.BatchTiming(constant(2), constant(1.5), (0, 10, 0),
                                  {1: constant(8), 2: constant(3), 3: constant(2)}),
        yield_ord=constant(10), zero=E.ZeroYield(p0=0), pi_J=0, price=100.,
        costs=[E.CostItem('batch', fixed=10), E.CostItem('shifts', fixed=20, basis='per_shift'),
               E.CostItem('hours', fixed=30, basis='per_operating_hour'),
               E.CostItem('work', fixed=40, basis='per_working_day'),
               E.CostItem('calendar', fixed=50, basis='per_calendar_day')],
        rho_spearman=None, dependence='independent', hh_need_per_household_day=20.)
    return dataclasses.replace(base, **kw)


class TestBatchTiming(unittest.TestCase):
    def test_rounding_boundary_is_half_up(self):
        """Working-day counts round floor(PERT + 0.5), so exact half-days round up."""
        inp = deterministic(batch_timing=E.BatchTiming(E.PERT(1, 2, 3), constant(0), (10, 0, 0),
                               {1: constant(8), 2: constant(4), 3: constant(2)}))
        with patch.object(inp.batch_timing.work_days, 'sample', return_value=np.array([1.49, 1.5, 2.49, 2.5, 3.])):
            a = E.sample_timing(inp, np.random.default_rng(9), 5)
        np.testing.assert_array_equal(a['working_days'], [1, 2, 2, 3, 3])
        np.testing.assert_array_equal(a['calendar_days_per_cycle'], a['working_days'])
        np.testing.assert_array_equal(a['L_person_hours'], [16, 32, 32, 48, 48])

    def test_each_day_gets_an_independent_count_and_conditional_hours(self):
        """Each working day draws its own shift count and that count's conditional hours."""
        inp = deterministic(batch_timing=E.BatchTiming(constant(2), constant(0), (5, 0, 5),
                                  {1: constant(2), 2: constant(4), 3: constant(1)}))
        a = E.sample_timing(inp, np.random.default_rng(14), 40_000)
        for shifts, probability in ((2, .25), (4, .5), (6, .25)):
            self.assertAlmostEqual(np.mean(a['total_shifts'] == shifts), probability, delta=.01)
        # Two days: each contributes (shifts+3)/2 hours with these conditional durations.
        np.testing.assert_array_equal(a['operating_hours'], (a['total_shifts'] + 6) / 2)

    def test_default_schedule_constraints_and_positive_labor(self):
        """The default synthetic schedule obeys its own day/shift/hour bounds and yields positive labor."""
        inp = batch_synthetic()
        a = E.sample_timing(inp, np.random.default_rng(8), 30_000)
        self.assertTrue(np.all(a['calendar_days_per_cycle'] >= a['working_days']))
        self.assertTrue(np.all(a['operating_hours'] <= 24 * a['working_days']))
        self.assertTrue(np.all(a['operating_hours'] > 0))
        self.assertTrue(np.all((a['total_shifts'] >= a['working_days']) &
                              (a['total_shifts'] <= 3 * a['working_days'])))
        np.testing.assert_array_equal(a['L_person_hours'], inp.n_c * a['operating_hours'])
        self.assertAlmostEqual(a['idle_days'].mean(), inp.batch_timing.idle_days.mean, delta=.02)

    def test_invalid_timing_and_conflicting_legacy_inputs_rejected(self):
        """Malformed BatchTiming configurations and batch/fixed-timing conflicts raise InputError."""
        t = default_timing()
        bad = [dataclasses.replace(t, work_days=E.PERT(0, 1, 3)),
               dataclasses.replace(t, work_days=E.PERT(1, 1.5, 3)),
               dataclasses.replace(t, work_days=constant(3651)),
               dataclasses.replace(t, idle_days=E.PERT(-1, 0, 1)),
               dataclasses.replace(t, idle_days=E.PERT(0, float('nan'), 1)),
               dataclasses.replace(t, shift_counts=(4, 4, 4)),
               dataclasses.replace(t, shift_counts=(True, 8, 1)),
               dataclasses.replace(t, shift_counts=(5., 4, 1)),
               dataclasses.replace(t, shift_counts=(9, 1)),
               dataclasses.replace(t, hours_by_shifts={1: constant(8)}),
               dataclasses.replace(t, hours_by_shifts={1: constant(.25), 2: constant(4), 3: constant(2)}),
               dataclasses.replace(t, hours_by_shifts={1: constant(8), 2: constant(4), 3: constant(9)}),
               dataclasses.replace(t, hours_by_shifts={1: constant(17), 2: constant(4), 3: constant(2)})]
        for timing in bad:
            with self.subTest(timing=timing), self.assertRaises(E.InputError):
                E.validate(dataclasses.replace(batch_synthetic(), batch_timing=timing))
        for kw in ({'calendar_days_per_cycle': 5}, {'hours_matrix': [[8] * 4]},
                   {'D_c': 2}, {'hours_per_worker_shift': 9}, {'batch_timing': 'invalid'},
                   {'n_c': 10**400}, {'hh_need_per_household_day': 1e308}):
            with self.subTest(kw=kw), self.assertRaises(E.InputError):
                E.validate(dataclasses.replace(batch_synthetic(), **kw))

    def test_timing_does_not_multiply_or_change_batch_yield_draws(self):
        """Changing batch timing leaves yield draws untouched and does not scale them by shifts."""
        a = batch_synthetic()
        b = dataclasses.replace(a, batch_timing=dataclasses.replace(a.batch_timing, work_days=constant(10)))
        x, y = E.run(a, n_fixed=2000), E.run(b, n_fixed=2000)
        np.testing.assert_array_equal(x.arrays['Y'], y.arrays['Y'])
        self.assertGreater(y.arrays['L_person_hours'].mean(), x.arrays['L_person_hours'].mean())
        z = E.run(a, n_fixed=2000)
        for key in x.arrays:
            np.testing.assert_array_equal(x.arrays[key], z.arrays[key])


class TestExpenseAndAccounting(unittest.TestCase):
    def test_every_expense_basis_and_equipment_are_counted_once(self):
        """Every expense basis and both equipment modes resolve to the expected cost exactly once."""
        inp = deterministic(equipment=[E.EquipmentItem('rental', 'rented', rental_rate_per_shift=7),
               E.EquipmentItem('owned', 'purchased', acquisition=1000, life_cycles=10)])
        a = E.run(inp, n_fixed=30).arrays
        expected = {'batch': 10, 'shifts': 80, 'hours': 360, 'work': 80, 'calendar': 175,
                    'equipment:rental': 28, 'equipment:owned': 100}
        for name, value in expected.items():
            np.testing.assert_array_equal(a[f'cost::{name}'], value)
        np.testing.assert_array_equal(a['C_tot'], 833)
        np.testing.assert_array_equal(a['cash_cost_total'], 733)
        np.testing.assert_array_equal(a['L_person_hours'], 24)
        np.testing.assert_array_equal(a['B_HH'], 140)
        np.testing.assert_array_equal(a['Pi'], -333)
        np.testing.assert_array_equal(a['cash_before_reserve'], -233)

    def test_idle_days_change_household_and_calendar_cost_only(self):
        """Longer idle days raise household need and calendar-day costs without affecting working-time costs."""
        a = deterministic()
        b = dataclasses.replace(a, batch_timing=dataclasses.replace(a.batch_timing, idle_days=constant(4.5)))
        x, y = E.run(a, n_fixed=20).arrays, E.run(b, n_fixed=20).arrays
        for key in ('Y', 'operating_hours', 'L_person_hours', 'cost::hours', 'cost::work', 'cost::shifts'):
            np.testing.assert_array_equal(x[key], y[key])
        np.testing.assert_array_equal(y['B_HH'] - x['B_HH'], 120)
        np.testing.assert_array_equal(y['cost::calendar'] - x['cost::calendar'], 150)

    def test_copula_targets_rates_and_keeps_rate_marginals(self):
        """The Gaussian copula achieves the targeted rate correlation while preserving each rate's marginal."""
        inp = dataclasses.replace(batch_synthetic(), costs=[
            E.CostItem('a', pert=E.PERT(10, 20, 30), basis='per_working_day'),
            E.CostItem('b', pert=E.PERT(5, 10, 15), basis='per_calendar_day')],
            rho_spearman=[[1, .6], [.6, 1]])
        a = E.run(inp, n_fixed=30_000).arrays
        self.assertAlmostEqual(stats.spearmanr(a['rate::a'], a['rate::b']).statistic, .6, delta=.025)
        self.assertAlmostEqual(a['rate::a'].mean(), 20, delta=.1)
        np.testing.assert_allclose(a['cost::a'], a['rate::a'] * a['working_days'])
        np.testing.assert_allclose(a['cost::b'], a['rate::b'] * a['calendar_days_per_cycle'])

    def test_staged_probabilities_thresholds_and_reserves_use_each_draw(self):
        """Staged-N probabilities, the three thresholds, and reserve accounting hold for every draw and policy combination."""
        for base in ('gross', 'net'):
            for location in ('before_sharing', 'after_sharing'):
                for priority in ('household_first', 'reserve_first'):
                    inp = dataclasses.replace(batch_synthetic(), tribute_base=base,
                        costs=[E.CostItem('operations', .3, pert=E.PERT(100, 200, 400), basis='per_operating_hour')],
                        rho_spearman=None, reserve_mode='custom', reserve_custom_per_cycle=5000,
                        reserve_location=location, reserve_priority=priority,
                        equipment=[E.EquipmentItem('tool', 'purchased', miner_share=.2, acquisition=10000, life_cycles=10)])
                    r = E.run(inp, n_fixed=1200, batch_size=400)
                    a = r.arrays
                    with self.subTest(base=base, location=location, priority=priority):
                        np.testing.assert_allclose(a['Pi'], a['E'] * a['L_person_hours'])
                        unit = a['q'] * a['P'] * inp.s_m
                        np.testing.assert_allclose(unit * a['Y_HH'] - a['C_eff'], a['B_HH'])
                        np.testing.assert_allclose(unit * a['Y_W'] - a['C_eff'], inp.W_opp * a['L_person_hours'])
                        other_cash = a['R'] - a['cash_cost_total'] - a['cash_before_reserve']
                        if location == 'before_sharing':
                            other_cash -= (1 - inp.s_m) * a['reserve_funded']
                        np.testing.assert_allclose(a['cash_after_reserve'] + other_cash + a['reserve_funded'],
                                                   a['R'] - a['cash_cost_total'], atol=1e-9)
                        for _, row in r.stage_table.iterrows():
                            n = int(row['N'])
                            events = a['Pi'][:n] < a['B_HH'][:n]
                            self.assertEqual(row['P_fail_household_adequacy'], events.mean())
                            joint = (a['cash_after_reserve'][:n] >= a['B_HH'][:n]) & (a['reserve_shortfall'][:n] == 0)
                            self.assertEqual(row['P_household_and_reserve_met'], joint.mean())
                        if priority == 'household_first':
                            np.testing.assert_array_equal(a['cash_before_reserve'] >= a['B_HH'],
                                                          a['cash_after_reserve'] >= a['B_HH'])
                        for key in E.PROB_KEYS:
                            self.assertEqual(r.summary[key], r.stage_table.iloc[-1][key])

    def test_fixed_api_scalar_summaries_and_new_batch_summaries(self):
        """Fixed-timing runs keep their scalar summary fields; batch-timing runs report mean/median arrays instead."""
        old = E.run(fixed_synthetic(), n_fixed=20)
        new = E.run(deterministic(), n_fixed=20)
        self.assertEqual(old.summary['L_person_hours'], 32)
        for key in ('L_person_hours', 'B_HH', 'calendar_days_per_cycle'):
            self.assertNotIn(key, new.summary)
            self.assertIn(key + '_mean', new.summary)
            self.assertIn(key + '_median', new.summary)
        self.assertNotIn('resolved_calendar_days_per_cycle', new.manifest)
        self.assertEqual(new.manifest['timing_mode'], 'batch')
        for method in ('person_hours', 'calendar_days', 'household_requirement'):
            with self.assertRaises(E.InputError):
                getattr(deterministic(), method)()

    def test_bad_cost_bases_and_unknown_fixed_working_exposure_rejected(self):
        """Unknown cost bases and per-working-day costs without batch_timing both raise InputError."""
        for inp in (dataclasses.replace(batch_synthetic(), costs=[E.CostItem('x', fixed=1, basis='unknown')]),
                    dataclasses.replace(fixed_synthetic(), costs=[E.CostItem('x', fixed=1, basis='per_working_day')])):
            with self.assertRaises(E.InputError):
                E.validate(inp)


class TestBatchDashboard(unittest.TestCase):
    def test_csv_matches_draws_and_manifest_names_rate_units(self):
        """The exported CSV matches the simulated arrays, and the manifest records each rate's expense basis."""
        r = E.run(batch_synthetic(), n_fixed=40)
        frame = pd.read_csv(io.BytesIO(DL.export_csv(r, 30)))
        for key in E.TIMING_KEYS:
            np.testing.assert_allclose(frame[key], r.arrays[key][:30])
        for name in ('fuel', 'milling', 'explosives', 'transport', 'food'):
            for prefix in ('cost::', 'rate::'):
                np.testing.assert_allclose(frame[prefix + name], r.arrays[prefix + name][:30])
        np.testing.assert_allclose(frame.miner_profit_php / frame.L_person_hours, frame.return_php_per_person_hour)
        manifest = json.loads(DL.export_manifest(r))
        self.assertEqual(manifest['expense_bases']['fuel'], 'per_operating_hour')
        self.assertEqual(manifest['correlation_order'], ['fuel', 'milling', 'explosives', 'transport', 'food'])
        self.assertEqual(manifest['valuation_mass_label'], 'recovered gold mass')

    def test_share_presets_and_named_matrix_map_exactly(self):
        """Share presets set s_m and its provenance correctly, and the named correlation matrix maps exactly."""
        v = dict(DL.preset_values(DL.PRESETS[0]))
        for label, share in (('50/50', .5), ('60/40', .6), ('65/35', .65), ('Custom', .73)):
            inp = DL.build_inputs({**v, 'share_preset': label, 's_m': .73, 'rho_fuel_food': .4}, DL.PRESETS[0])
            self.assertEqual(inp.provenance['s_m'], 'synthetic' if share == .5 else 'user')
            self.assertEqual(inp.s_m, share)
            self.assertEqual(inp.rho_spearman[0][4], .4)
            self.assertEqual(inp.rho_spearman[4][0], .4)
            E.validate(inp)

    def test_invalid_batch_widgets_are_not_coerced(self):
        """Invalid batch-timing widget values (wrong type, NaN, infinity) are rejected, not silently coerced."""
        v = dict(DL.preset_values(DL.PRESETS[0]))
        for key, value in (('work_mode', 2.5), ('ym', float('nan')), ('days_1', True), ('idle_max', float('nan')),
                           ('h3_max', 9), ('rho_fuel_food', float('inf'))):
            with self.subTest(key=key), self.assertRaises(E.InputError):
                E.run(DL.build_inputs({**v, key: value}), n_fixed=10)


if __name__ == '__main__':
    unittest.main()
