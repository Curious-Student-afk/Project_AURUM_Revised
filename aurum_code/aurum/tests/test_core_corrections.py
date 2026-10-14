"""Regression tests for V2.3 contracts, numerical diagnostics and export semantics."""
import dataclasses
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine_v2 as E
import dashboard_logic as D
import normalization as N
from scenarios import batch_synthetic, fixed_synthetic


def fixed(**kw):
    """Fixed-timing synthetic Inputs, with fields overridden by `kw`."""
    return dataclasses.replace(fixed_synthetic(), **kw)


def constant(**kw):
    """Fixed-timing Inputs with deterministic yield/zero/price and no costs, for exact-arithmetic checks."""
    values = dict(yield_ord=E.PERT(1, 1, 1), zero=E.ZeroYield(p0=0), pi_J=0,
                  price=100., costs=[], equipment=[], dependence='independent',
                  hh_need_per_household_day=0.)
    values.update(kw)
    return fixed(**values)


class TestInputs(unittest.TestCase):
    def test_zero_atom_and_positive_component(self):
        """A yield distribution concentrated at zero is rejected unless p0=1; otherwise the zero share is as specified."""
        with self.assertRaisesRegex(E.InputError, 'concentrated at zero'):
            E.run(fixed(yield_ord=E.PERT(0, 0, 0)), n_fixed=100)
        with self.assertRaises(E.InputError):
            E.run(constant(yield_ord=E.PERT(0, 0, 0), zero=E.ZeroYield(p0=1)), n_fixed=100)
        r = E.run(constant(yield_ord=E.PERT(0, .5, 1), zero=E.ZeroYield(p0=.2)), n_fixed=10000)
        self.assertLess(abs(r.summary['observed_zero_cycle_share']-.2), .02)
        r = E.run(constant(zero=E.ZeroYield(p0=1)), n_fixed=100)
        np.testing.assert_array_equal(r.arrays['Y'], 0)

    def test_farmgate_assay_has_separate_rng_and_no_economic_effect(self):
        """A farmgate physical_purity draw uses its own RNG stream and leaves every economic array unchanged."""
        original = E.run(fixed(), n_fixed=500, seed=9)
        changed = E.run(fixed(physical_purity=E.PERT(.7, .8, .9)), n_fixed=500, seed=9)
        self.assertNotIn('physical_purity_q', original.arrays)
        for key in original.arrays:
            np.testing.assert_array_equal(original.arrays[key], changed.arrays[key])
        self.assertTrue(np.all((changed.arrays['physical_purity_q'] >= .7) & (changed.arrays['physical_purity_q'] <= .9)))
        np.testing.assert_array_equal(changed.arrays['q'], 1)
        again = E.run(fixed(physical_purity=E.PERT(.7, .8, .9)), n_fixed=500, seed=9)
        np.testing.assert_array_equal(changed.arrays['physical_purity_q'], again.arrays['physical_purity_q'])

    def test_reference_assay_and_conflict(self):
        """Reference pricing uses `purity` as the physical assay; declaring physical_purity alongside it is rejected."""
        r = E.run(constant(price_basis='reference', purity=.8), n_fixed=100)
        np.testing.assert_array_equal(r.arrays['q'], .8)
        np.testing.assert_allclose(r.arrays['R'], r.arrays['Y'] * .8 * 100)
        np.testing.assert_array_equal(r.arrays['q'], r.arrays['physical_purity_q'])
        with self.assertRaisesRegex(E.InputError, 'use purity'):
            E.validate(fixed(price_basis='reference', physical_purity=.7))
        for assay in (0, 1.1, True, float('nan'), E.PERT(.1, .2, 1.1)):
            with self.subTest(assay=assay), self.assertRaises(E.InputError):
                E.validate(fixed(physical_purity=assay))

    def test_batch_households_independent_of_labor_and_economic_profit(self):
        """Household count and nonmining income change B_HH only, leaving labor, yield and profit arrays unchanged."""
        inp = batch_synthetic()
        a = E.run(inp, n_fixed=500)
        b = E.run(dataclasses.replace(inp, n_households=2, nonmining_income_per_household_day=300.), n_fixed=500)
        np.testing.assert_allclose(b.arrays['B_HH'], 2*b.arrays['calendar_days_per_cycle']*500)
        for key in ('L_person_hours', 'Y', 'Pi', 'E', 'C_tot'):
            np.testing.assert_array_equal(a.arrays[key], b.arrays[key])
        for kw in ({'n_households': 0}, {'nonmining_income_per_household_day': 1000}):
            r = E.run(dataclasses.replace(inp, **kw), n_fixed=100)
            np.testing.assert_array_equal(r.arrays['B_HH'], 0)

    def test_components_and_metadata(self):
        """Cost component identifiers must be nonempty and non-overlapping; metadata and dataset fields must be valid."""
        for components in (['fuel'], [' FUEL '], ['repair', 'repair'], [], ['']):
            with self.subTest(components=components), self.assertRaises(E.InputError):
                E.validate(fixed(costs=[E.CostItem('overhead', fixed=20., components=components), E.CostItem('fuel', fixed=10.)]))
        E.validate(fixed(costs=[E.CostItem('overhead', fixed=20., components=['repair', 'security']), E.CostItem('fuel', fixed=10.)]))
        with self.assertRaisesRegex(E.InputError, 'overlaps'):
            E.validate(fixed(costs=[E.CostItem('overhead', fixed=20., components=['equipment:main'])],
                             equipment=[E.EquipmentItem('main', 'rented', rental_rate_per_shift=5.)]))
        with self.assertRaises(E.InputError):
            E.validate(fixed(metadata={'bad': float('nan')}))
        with self.assertRaises(E.InputError):
            E.validate(fixed(dataset_id='pilot'))

    def test_deterministic_unit_conversions_and_failures(self):
        """normalization helpers convert units correctly and raise InputError on invalid arguments."""
        self.assertEqual(N.purity_fraction(18, 'karat'), .75)
        self.assertEqual(N.purity_fraction(75, 'percent'), .75)
        self.assertEqual(N.realized_price(1000, 2), 500)
        self.assertEqual(N.fuel_cost(3, 50, 'per_shift', 'per_shift'), 150)
        cases = [(N.purity_fraction, (25, 'karat')), (N.purity_fraction, (80, 'unknown')),
                 (N.purity_fraction, (True, 'fraction')), (N.realized_price, (100, 0)),
                 (N.realized_price, (100, 1, 'reference')), (N.fuel_cost, (2, 0)),
                 (N.fuel_cost, (2, 10, 'per_batch', 'per_shift'))]
        for fn, args in cases:
            with self.subTest(args=args), self.assertRaises(E.InputError):
                fn(*args)

    def test_dashboard_conversions_preserve_raw_metadata_and_explicit_zero(self):
        """build_inputs applies sale/fuel/assay conversions, keeps raw measurement metadata, and rejects bad inputs."""
        v = dict(D.preset_values(D.PRESETS[0]))
        v.update(assay_mode='measured value', assay_value=18., assay_unit='karat', price_input='sale',
                 sale_proceeds=2000., sale_mass=2., fuel_input='liters × PHP/liter',
                 fuel_liters=3., fuel_unit_price=50., households_mode='explicit', n_hh=0,
                 nonmining=100., measurement_source='synthetic fixture', measurement_date='2026-10-13')
        inp = D.build_inputs(v)
        self.assertEqual(inp.physical_purity, .75)
        self.assertEqual(inp.price.a, 1000)
        self.assertEqual(inp.costs[1].pert.a, 150)
        self.assertEqual(inp.n_households, 0)
        self.assertEqual(inp.metadata['assay']['unit'], 'karat')
        r = E.run(inp, n_fixed=100)
        self.assertEqual(r.manifest['inputs']['metadata'], inp.metadata)
        for change in ({'measurement_date': 'bad'}, {'measurement_date': '20261013'},
                       {'measurement_date': '2026-W41-2'}, {'measurement_date': '2026-02-30'},
                       {'fixed_cost': 100., 'fixed_components': ''},
                       {'fixed_cost': 100., 'fixed_components': 'fuel'}):
            with self.subTest(change=change), self.assertRaises(E.InputError):
                E.run(D.build_inputs({**v, **change}), n_fixed=100)
        self.assertEqual(D.build_inputs({**v, 'measurement_date': '2024-02-29'}).metadata['date'], '2024-02-29')


class TestNumericalDiagnostics(unittest.TestCase):
    def test_error_estimators_and_order_intervals(self):
        """return_diagnostics' mean/ES standard errors and binomial order-statistic intervals match closed forms."""
        a = np.arange(1000, dtype=float)
        d = E.return_diagnostics(a)
        self.assertAlmostEqual(d['error_E_mean'], 2*a.std(ddof=1)/np.sqrt(1000))
        winsor = np.minimum(a, np.percentile(a, 5))
        self.assertAlmostEqual(d['error_E_expected_shortfall_5pct'], 2*winsor.std(ddof=1)/np.sqrt(1000)/.05)
        for key, p in (('E_P5', .05), ('E_median', .5), ('E_P95', .95)):
            lo = int(stats.binom.ppf(.025, 1000, p))
            hi = int(stats.binom.ppf(.975, 1000, p))+1
            self.assertEqual(d['lower_'+key], a[lo-1])
            self.assertEqual(d['upper_'+key], a[hi-1])
            self.assertLessEqual(d['lower_'+key], d[key])
            self.assertGreaterEqual(d['upper_'+key], d[key])
        self.assertEqual(d['tolerance_E_mean'], .01*d['E_mean'])

    def test_constant_and_tied_tail_diagnostics(self):
        """return_diagnostics stays finite on constant/tied samples and flags unmet precision for a single value."""
        for a in (np.full(1000, -10.), np.repeat([-5., 0., 20.], [10, 800, 190])):
            d = E.return_diagnostics(a)
            self.assertEqual(d['E_expected_shortfall_5pct'], E.expected_shortfall(a))
            for key in E.RETURN_KEYS:
                self.assertTrue(np.isfinite(d['error_'+key]))
        d = E.return_diagnostics([5.])
        self.assertFalse(d['return_precision_met'])
        self.assertIsNone(d['error_E_P5'])

    def test_two_checks_and_exact_fixed_n(self):
        """Convergence requires two consecutive passing checks; n_fixed always runs exactly N draws."""
        first = E.run(constant(), n_min=100, n_max=100, batch_size=100, tol=.05)
        self.assertTrue(first.summary['probability_precision_met'])
        self.assertTrue(first.summary['return_precision_met'])
        self.assertFalse(first.summary['converged'])
        r = E.run(constant(), n_min=100, n_max=300, batch_size=100, tol=.05)
        self.assertTrue(r.summary['converged'])
        self.assertEqual(r.summary['N'], 200)
        r = E.run(constant(), n_min=100, n_fixed=321, batch_size=100, tol=.05)
        self.assertEqual(r.summary['N'], 321)
        self.assertTrue(r.summary['converged'])
        self.assertIn('error_E_P95', r.stage_table)

    def test_rare_jackpot_blocks_full_convergence(self):
        """An underobserved ordinary or jackpot component fails mixture support and blocks full convergence."""
        r = E.run(constant(pi_J=1e-8, jackpot=E.Jackpot(100, 10000)), n_fixed=1000,
                  n_min=100, batch_size=100, tol=.05)
        self.assertTrue(r.summary['probability_precision_met'])
        self.assertFalse(r.summary['converged'])
        self.assertIn('jackpot component', r.summary['convergence_reasons'])
        self.assertFalse(r.summary['mixture_support_met'])
        self.assertEqual(r.manifest['stopping_reason'], 'fixed_draw_count')
        r = E.run(constant(pi_J=1-1e-8, jackpot=E.Jackpot(100, 10000)), n_max=1000,
                  n_min=100, batch_size=100, tol=.05)
        self.assertFalse(r.summary['converged'])
        self.assertIn('ordinary component', r.summary['convergence_reasons'])
        self.assertEqual(r.manifest['stopping_reason'], 'maximum_draw_cap')

    def test_probability_precision_does_not_override_return_error(self):
        """Meeting probability precision alone does not satisfy convergence when return error exceeds its tolerance."""
        r = E.run(constant(yield_ord=E.PERT(1, 100, 1000)), n_fixed=1000, n_min=100,
                  batch_size=100, tol=.05, return_abs_tol=1e-9, return_rel_tol=0)
        self.assertTrue(r.summary['probability_precision_met'])
        self.assertFalse(r.summary['return_precision_met'])
        self.assertFalse(r.summary['converged'])
        self.assertIn('numerical error', r.summary['convergence_reasons'])

    def test_unstable_returns_block_convergence_and_disabled_components_do_not(self):
        """A large change in E_mean between checks blocks convergence, but a disabled (p0=1) component does not."""
        def diagnostics(a, **kw):
            """Fake return_diagnostics that reports a large E_mean swing between the two checks while claiming precision."""
            r = real(a, **kw)
            r['E_mean'] = 100000. if len(a) == 200 else 0.
            r['return_precision_met'] = True
            return r
        real = E.return_diagnostics
        with patch.object(E, 'return_diagnostics', side_effect=diagnostics):
            r = E.run(constant(), n_fixed=200, n_min=100, batch_size=100, tol=.05)
        self.assertFalse(r.summary['converged'])
        self.assertFalse(r.summary['return_stability_met'])
        r = E.run(constant(zero=E.ZeroYield(p0=1)), n_min=100, n_max=300, batch_size=100, tol=.05)
        self.assertTrue(r.summary['converged'])
        self.assertEqual(r.summary['observed_ordinary_count'], 0)

    def test_tolerance_validation_and_posterior_interval(self):
        """Invalid return tolerances raise InputError, and the zero-yield posterior interval matches the Beta CDF."""
        for kw in ({'return_abs_tol': -1}, {'return_rel_tol': True}, {'return_abs_tol': 0, 'return_rel_tol': 0}):
            with self.assertRaises(E.InputError):
                E.run(constant(), n_fixed=100, **kw)
        for abs_tol, rel_tol in ((-1, .01), (1, True), (0, 0), (float('nan'), .01)):
            with self.subTest(abs_tol=abs_tol, rel_tol=rel_tol), self.assertRaises(E.InputError):
                E.return_diagnostics([1., 2.], abs_tol=abs_tol, rel_tol=rel_tol)
        r = E.run(fixed(zero=E.ZeroYield(z=2, n=10)), n_fixed=100)
        np.testing.assert_allclose([r.summary['p0_posterior_lower_95'], r.summary['p0_posterior_upper_95']], stats.beta.ppf([.025, .975], 3, 9))


class TestExportsAndReproducibility(unittest.TestCase):
    def test_versioned_unknown_and_known_assay_exports(self):
        """Schema 1 exposes legacy purity_q while schema 2 exposes valuation_multiplier and physical_purity_q (blank when unknown)."""
        for inp in (constant(), constant(physical_purity=.75), constant(price_basis='reference', purity=.8)):
            r = E.run(inp, n_fixed=100)
            corrected = pd.read_csv(io.BytesIO(D.export_csv(r)))
            legacy = pd.read_csv(io.BytesIO(D.export_csv(r, schema_version=1)))
            self.assertNotIn('purity_q', corrected)
            self.assertNotIn('physical_purity_q', legacy)
            np.testing.assert_array_equal(legacy['purity_q'], r.arrays['q'])
            np.testing.assert_array_equal(corrected['valuation_multiplier'], r.arrays['q'])
            if inp.price_basis=='farmgate' and inp.physical_purity is None:
                self.assertTrue(corrected['physical_purity_q'].isna().all())
            else:
                np.testing.assert_allclose(corrected['physical_purity_q'], r.arrays['physical_purity_q'])
            self.assertEqual(json.loads(D.export_manifest(r))['export_schema_version'], 2)
            self.assertEqual(json.loads(D.export_manifest(r, schema_version=1))['export_schema_version'], 1)
            self.assertNotIn('NaN', D.export_manifest(r).decode())
        with self.assertRaises(E.InputError):
            D.export_csv(r, schema_version=3)

    def test_full_source_identity_and_dataset_only_when_supplied(self):
        """The manifest hashes every source file, reports git identity when available, and omits dataset fields unless set."""
        r = E.run(constant(), n_fixed=100)
        m = r.manifest
        root = Path(E.__file__).parent
        for name in ('engine_v2.py', 'dashboard_logic.py', 'app.py', 'normalization.py', 'requirements.lock'):
            self.assertEqual(m['source_sha256'][name], hashlib.sha256((root/name).read_bytes()).hexdigest())
        if m['git_revision'] is not None:
            self.assertIsInstance(m['git_revision'], str)
        if m['git_dirty'] is not None:
            self.assertIsInstance(m['git_dirty'], bool)
        with patch.object(E.subprocess, 'check_output', side_effect=FileNotFoundError):
            without_git = E.source_identity()
        self.assertIsNone(without_git['git_revision'])
        self.assertIsNone(without_git['git_dirty'])
        self.assertEqual(without_git['source_sha256'], m['source_sha256'])
        self.assertIsNone(m['dataset'])
        self.assertEqual(m['numerical_diagnostics']['error_E_mean'], r.summary['error_E_mean'])
        self.assertEqual(m['numerical_diagnostics']['convergence_reasons'], r.summary['convergence_reasons'])
        self.assertEqual(m['stopping_reason'], 'fixed_draw_count')
        self.assertIn('farmgate_assay', m['random_streams'])
        for name in ('numpy', 'scipy', 'pandas', 'matplotlib', 'streamlit'):
            self.assertIn(name, m['runtime_versions'])
        r = E.run(constant(dataset_id='synthetic fixture', dataset_sha256='a'*64), n_fixed=100)
        self.assertEqual(r.manifest['dataset'], {'id': 'synthetic fixture', 'sha256': 'a'*64})
        self.assertEqual(m['run_config']['return_abs_tol'], 1.)

    def test_manifest_allows_missing_optional_ui_packages(self):
        """Missing optional UI packages report None in runtime_versions instead of raising, while other errors propagate."""
        installed_version = E.importlib.metadata.version

        def optional_missing(name):
            """Simulate streamlit/matplotlib not being installed while other packages resolve normally."""
            if name in ('streamlit', 'matplotlib'):
                raise E.importlib.metadata.PackageNotFoundError(name)
            return installed_version(name)

        with patch.object(E.importlib.metadata, 'version', side_effect=optional_missing):
            r = E.run(constant(), n_fixed=100)
        versions = r.manifest['runtime_versions']
        self.assertIsNone(versions['streamlit'])
        self.assertIsNone(versions['matplotlib'])
        for name in ('numpy', 'scipy', 'pandas'):
            self.assertEqual(versions[name], installed_version(name))
        exported = json.loads(D.export_manifest(r))
        self.assertIsNone(exported['runtime_versions']['streamlit'])
        with patch.object(E.importlib.metadata, 'version', side_effect=RuntimeError('metadata failure')):
            with self.assertRaisesRegex(RuntimeError, 'metadata failure'):
                E._package_version('streamlit')

    def test_source_identity_excludes_dependency_and_vcs_trees(self):
        """source_identity hashes application and requirements files while skipping VCS and dependency directories."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            included = ('engine_v2.py', 'src/helper.py', 'requirements.lock')
            excluded = tuple(f'{name}/nested/vendor.py' for name in
                             ('__pycache__', '.git', '.venv', 'venv', 'site-packages', 'node_modules'))
            for name in included + excluded:
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('synthetic fixture')
            with patch.object(E, '__file__', str(root/'engine_v2.py')), \
                    patch.object(E.subprocess, 'check_output', side_effect=FileNotFoundError):
                identity = E.source_identity()
            self.assertEqual(set(identity['source_sha256']), set(included))
            for name in included:
                self.assertEqual(identity['source_sha256'][name], hashlib.sha256((root/name).read_bytes()).hexdigest())

    def test_benchmark_sensitivity_uses_same_draws(self):
        """benchmark_sensitivity reuses the existing simulation draws (never calls run again) and matches risk_band directly."""
        inp = fixed()
        r = E.run(inp, n_fixed=1000)
        copy = r.arrays['E'].copy()
        with patch.object(E, 'run', side_effect=AssertionError('must not resample')):
            frame = D.benchmark_sensitivity(r, inp)
        self.assertEqual(len(frame), 9)
        np.testing.assert_array_equal(copy, r.arrays['E'])
        for _, row in frame.iterrows():
            p = np.mean(copy < row['Wage (PHP/person-hour)'])
            self.assertEqual(row['P_below_wage'], p)
            self.assertEqual(row['Illustrative band'], E.risk_band(1-p, row['Low cutoff'], row['Moderate cutoff']))
        for low, moderate in ((.60, .30), (.67, .34), (.75, .40)):
            for p, expected in ((1., 'low'), (low, 'low'), (low-1e-8, 'moderate'),
                                (moderate, 'moderate'), (moderate-1e-8, 'high'), (0., 'high')):
                self.assertEqual(E.risk_band(p, low, moderate), expected)


if __name__ == '__main__':
    unittest.main()
