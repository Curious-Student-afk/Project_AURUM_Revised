"""
Acceptance tests for engine_v2 (stdlib unittest; also runs under pytest).

Run from the project folder:   python -m unittest discover -s tests -v

Each test names the expert-report section / validation domain it supports. A passing test is
evidence of VERIFICATION (the code does what the specification says). It is not empirical
VALIDATION against field data (Domain G) -- that needs Phase 2 data.
"""
import dataclasses
import os
import sys
import unittest

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine_v2 as E2                      # noqa: E402
import legacy_v1                            # noqa: E402
from engine_v2 import (CostItem, EquipmentItem, InputError, Jackpot, PERT, ZeroYield,  # noqa: E402
                       accounting_kernel, mcse_prob, risk_band, run, sample_costs,
                       scenario_table, spearman_to_pearson, validate)
from scenarios import pilot_provisional, synthetic_profiles  # noqa: E402


def base(**kw):
    inp = pilot_provisional()
    inp = dataclasses.replace(inp, W_opp_source="test", W_opp_date="test")
    return dataclasses.replace(inp, **kw) if kw else inp


class TestDistributions(unittest.TestCase):
    """Domain D (distributional identifiability) -- synthetic verification only."""

    def test_pert_moments_match_theory(self):
        p = PERT(0.3, 0.8, 1.5)
        x = p.sample(np.random.default_rng(1), 400_000)
        mu = (0.3 + 4 * 0.8 + 1.5) / 6
        var = (mu - 0.3) * (1.5 - mu) / 7          # exact variance of standard Beta-PERT
        self.assertAlmostEqual(x.mean(), mu, delta=0.003)
        self.assertAlmostEqual(x.var(), var, delta=0.003)
        self.assertTrue(x.min() >= 0.3 and x.max() <= 1.5)

    def test_pert_ppf_endpoints_and_monotone(self):
        p = PERT(10, 40, 100)
        u = np.linspace(0.001, 0.999, 500)
        q = p.ppf(u)
        self.assertTrue(np.all(np.diff(q) > 0))
        self.assertAlmostEqual(float(p.ppf(np.array([0.0]))[0]), 10)
        self.assertAlmostEqual(float(p.ppf(np.array([1.0]))[0]), 100)

    def test_pert_shape_properties_safe_when_degenerate(self):
        p = PERT(5, 5, 5)
        self.assertEqual((p.alpha, p.beta), (1.0, 1.0))          # inspecting must not raise ZeroDivisionError

    def test_pert_degenerate_is_constant(self):
        x = PERT(5, 5, 5).sample(np.random.default_rng(1), 100)
        self.assertTrue(np.all(x == 5))

    def test_jackpot_truncated_no_atom_at_cap(self):
        j = Jackpot(14.4, 24.0, 1.5)
        x = j.sample(np.random.default_rng(2), 300_000)
        self.assertTrue(x.min() >= 14.4 and x.max() <= 24.0)
        self.assertLess(np.mean(x >= 24.0 - 1e-9), 1e-3)       # no mass piled on the cap
        tail = 1 - (14.4 / 24.0) ** 1.5
        median = 14.4 / (1 - 0.5 * tail) ** (1 / 1.5)
        self.assertAlmostEqual(np.median(x), median, delta=0.05)

    def test_zero_share_matches_p0_within_mcse(self):
        inp = dataclasses.replace(base(), zero=ZeroYield(p0=0.2), pi_J=0.0)
        r = run(inp, n_fixed=100_000)
        k = int(round(r.summary["observed_zero_cycle_share"] * r.summary["N"]))
        self.assertLessEqual(abs(r.summary["observed_zero_cycle_share"] - 0.2), 4 * mcse_prob(k, r.summary["N"]))

    def test_beta_posterior_p0_drawn_per_iteration(self):
        z = ZeroYield(z=3, n=10)                               # posterior Beta(4, 8)
        x = z.draw_p0(np.random.default_rng(3), 200_000)
        self.assertAlmostEqual(x.mean(), 4 / 12, delta=0.003)
        self.assertGreater(x.std(), 0.1)                       # varies by draw, not one value per batch

    def test_zero_yield_never_causes_division_by_zero(self):
        r = run(dataclasses.replace(base(), zero=ZeroYield(p0=1.0), pi_J=0.0), n_fixed=10_000)
        self.assertTrue(np.all(np.isfinite(r.arrays["E"])))
        self.assertTrue(np.all(np.isfinite(r.arrays["Y_BE"])))
        self.assertEqual(r.summary["P_negative_operating_return"], 1.0)


class TestValidation(unittest.TestCase):
    """Domain F (computational robustness) -- expert report Section 8 rules."""

    def bad(self, **kw):
        with self.assertRaises(InputError, msg=str(kw)):
            validate(dataclasses.replace(base(), **kw))

    def test_rejects_impossible_inputs(self):
        self.bad(n_c=0)
        self.bad(D_c=0)
        self.bad(hours_per_worker_shift=0)
        self.bad(s_m=1.2)
        self.bad(W_opp=0)
        self.bad(price=PERT(0, 5, 10))
        self.bad(price=-5.0)
        self.bad(yield_ord=PERT(1.0, 0.5, 2.0))                # mode below minimum
        self.bad(zero=ZeroYield(p0=1.5))
        self.bad(zero=ZeroYield(z=5, n=3))
        self.bad(zero=ZeroYield())
        self.bad(pi_J=0.1, jackpot=None)
        self.bad(pi_J=2.0)
        self.bad(hh_need_per_household_day=-1)
        self.bad(tribute_base="after_tax")
        self.bad(price_basis="mystery")

    def test_wage_paid_worker_is_flagged_not_divided(self):
        with self.assertRaises(InputError) as cm:
            validate(dataclasses.replace(base(), s_m=0.0))
        self.assertIn("wage-paid", str(cm.exception))

    def test_reference_basis_rejects_zero_or_over_unity_purity(self):
        self.bad(price_basis="reference", price=8400.0, purity=0.0)
        self.bad(price_basis="reference", price=8400.0, purity=1.3)
        self.bad(price_basis="reference", price=8400.0, purity=PERT(0.8, 0.9, 1.2))

    def test_cost_rules(self):
        b = base()
        self.bad(costs=[CostItem("x", 1.5, fixed=10.0)])
        self.bad(costs=[CostItem("x", 1.0, fixed=-5.0)])
        self.bad(costs=[CostItem("x", 1.0)])                                  # neither fixed nor PERT
        self.bad(costs=[CostItem("x", 1.0, fixed=1.0, pert=PERT(1, 2, 3))])   # both
        self.bad(costs=[CostItem("x", 1.0, fixed=1.0), CostItem("x", 1.0, fixed=2.0)])  # duplicate
        self.assertIsNotNone(b)

    def test_copula_matrix_rules(self):
        b = base()
        self.bad(rho_spearman=[[1, 0.5, 0.1], [0.5, 1, 0.1], [0.1, 0.1, 1]])   # wrong size
        self.bad(rho_spearman=[[1, 0.5], [0.4, 1]])                              # not symmetric
        self.bad(rho_spearman=[[1, 1.2], [1.2, 1]])                              # out of range
        self.bad(rho_spearman=None)                                              # copula needs a matrix
        three = dataclasses.replace(b, costs=[CostItem(f"c{i}", 1.0, pert=PERT(1, 2, 3)) for i in range(3)])
        with self.assertRaises(InputError):                                      # not positive definite
            validate(dataclasses.replace(three, rho_spearman=[[1, .99, -.99], [.99, 1, .99], [-.99, .99, 1]]))

    def test_unsourced_wage_gives_warning_not_error(self):
        w = validate(pilot_provisional())
        self.assertTrue(any("W_opp" in x for x in w))


class TestEquipment(unittest.TestCase):
    """Required revision 4: purchased depreciation vs rental, never a blended markup."""

    def test_purchased_depreciation_formula(self):
        eq = EquipmentItem("mill", "purchased", acquisition=10_000, salvage=1_000, life_cycles=90, utilization=0.5)
        self.assertAlmostEqual(eq.cost_per_cycle(1), (10_000 - 1_000) / 90 * 0.5)

    def test_rented_has_no_depreciation(self):
        eq = EquipmentItem("pump", "rented", rental_rate_per_shift=120, acquisition=99_999, salvage=0, life_cycles=1)
        self.assertEqual(eq.cost_per_cycle(3), 360)

    def test_negative_depreciation_rejected(self):
        inp = dataclasses.replace(base(), equipment=[EquipmentItem("x", "purchased", acquisition=100, salvage=500)])
        with self.assertRaises(InputError):
            validate(inp)

    def test_equipment_enters_costs_once(self):
        b = base()
        b2 = dataclasses.replace(b, equipment=[EquipmentItem("pump", "rented", rental_rate_per_shift=100)])
        r1 = run(b, n_fixed=10_000)
        r2 = run(b2, n_fixed=10_000)
        np.testing.assert_allclose(r2.arrays["C_tot"] - r1.arrays["C_tot"], 100.0)


class TestAccounting(unittest.TestCase):
    """Domains B, C, E: one unit, stakeholder accounting, three separate thresholds."""

    def check_identities(self, inp):
        r = run(inp, n_fixed=20_000)
        a = r.arrays
        L, B = r.summary["L_person_hours"], r.summary["B_HH"]
        unit = a["q"] * a["P"] * inp.s_m
        np.testing.assert_allclose(unit * a["Y_BE"] - a["C_eff"], 0.0, atol=1e-6)               # Pi = 0 at Y_BE
        np.testing.assert_allclose((unit * a["Y_W"] - a["C_eff"]) / L, inp.W_opp, rtol=1e-9)      # E = W_opp at Y_W
        np.testing.assert_allclose(unit * a["Y_HH"] - a["C_eff"], B, rtol=1e-9)                 # Pi = B_HH at Y_HH
        np.testing.assert_allclose(a["E"] * L, a["Pi"], rtol=1e-12)                            # E x L = Pi
        # per-draw equivalences between yield-side and return-side events
        np.testing.assert_array_equal(a["Y"] < a["Y_BE"], a["Pi"] < 0)
        np.testing.assert_array_equal(a["Y"] < a["Y_W"], a["E"] < inp.W_opp)
        self.assertEqual(r.summary["P_negative_operating_return"], r.summary["P_yield_below_breakeven"])

    def test_threshold_identities_gross(self):
        self.check_identities(base(tribute_base="gross"))

    def test_threshold_identities_net(self):
        self.check_identities(base(tribute_base="net"))

    def test_net_base_breakeven_independent_of_share(self):
        a = run(base(tribute_base="net", s_m=0.5), n_fixed=10_000).arrays["Y_BE"]
        b = run(base(tribute_base="net", s_m=0.8), n_fixed=10_000).arrays["Y_BE"]
        np.testing.assert_allclose(a, b)

    def test_cost_allocation_separates_who_pays(self):
        free = dataclasses.replace(base(), costs=[CostItem("fuel", 0.0, fixed=500.0)], rho_spearman=None)
        paid = dataclasses.replace(base(), costs=[CostItem("fuel", 1.0, fixed=500.0)], rho_spearman=None)
        d = run(paid, n_fixed=10_000).arrays["Pi"] - run(free, n_fixed=10_000).arrays["Pi"]
        np.testing.assert_allclose(d, -500.0)

    def test_price_purity_counted_once(self):
        a = run(base(purity=0.5), n_fixed=10_000).arrays["E"]       # farmgate basis: purity must be ignored
        b = run(base(purity=1.0), n_fixed=10_000).arrays["E"]
        np.testing.assert_array_equal(a, b)
        ref = dataclasses.replace(base(), price_basis="reference", price=8400.0)
        lo = run(dataclasses.replace(ref, purity=0.8), n_fixed=10_000).arrays["R"]
        hi = run(dataclasses.replace(ref, purity=1.0), n_fixed=10_000).arrays["R"]
        np.testing.assert_allclose(lo, hi * 0.8)                     # reference basis: purity applied exactly once

    def test_group_level_scaling_with_crew_size(self):
        r5 = run(base(n_c=5), n_fixed=10_000)
        r10 = run(base(n_c=10), n_fixed=10_000)
        self.assertAlmostEqual(r10.summary["L_person_hours"], 2 * r5.summary["L_person_hours"])
        np.testing.assert_allclose(r10.arrays["E"] * 2, r5.arrays["E"])
        np.testing.assert_allclose(r10.arrays["Pi"], r5.arrays["Pi"])

    def test_hours_matrix_defines_person_hours(self):
        inp = base(hours_matrix=[[8, 8, 6], [7, 8, 9]], n_c=3, D_c=2)
        self.assertEqual(inp.person_hours(), 46)

    def test_household_requirement(self):
        b = base(n_c=4, D_c=2, hh_need_per_household_day=1000, nonmining_income_per_household_day=200)
        self.assertEqual(b.household_requirement(), 4 * 2 * 800)
        b2 = base(hh_need_per_household_day=100, nonmining_income_per_household_day=500)
        self.assertEqual(b2.household_requirement(), 0)           # floored at zero
        b3 = base(n_c=4, n_households=2, D_c=1, hh_need_per_household_day=1000)
        self.assertEqual(b3.household_requirement(), 2000)

    def test_runs_with_only_fixed_costs(self):
        inp = dataclasses.replace(base(), costs=[CostItem("fixed", 1.0, fixed=300.0)], rho_spearman=None)
        r = run(inp, n_fixed=10_000)
        self.assertTrue(np.all(r.arrays["C_tot"] == 300.0))


class TestDependence(unittest.TestCase):
    """Required revision 9 / report 3.9: rank dependence, with an independent comparison."""

    def test_pearson_conversion(self):
        r = spearman_to_pearson(np.array([[1, 0.5], [0.5, 1]]))
        self.assertAlmostEqual(r[0, 1], 2 * np.sin(np.pi * 0.5 / 6))
        self.assertAlmostEqual(spearman_to_pearson(np.array([[1, 1.0], [1.0, 1]]))[0, 1], 1.0)

    def test_realized_rank_correlation_matches_target(self):
        inp = dataclasses.replace(base(), rho_spearman=[[1, 0.6], [0.6, 1]])
        rng = np.random.default_rng(5)
        chol = E2._copula_chol(inp)
        c = sample_costs(inp, rng, 150_000, chol)
        rho = stats.spearmanr(c["fuel"], c["transport_hauling"]).statistic
        self.assertAlmostEqual(rho, 0.6, delta=0.02)

    def test_independent_mode_has_no_dependence(self):
        inp = dataclasses.replace(base(), dependence="independent")
        c = sample_costs(inp, np.random.default_rng(6), 150_000, E2._copula_chol(inp))
        self.assertLess(abs(stats.spearmanr(c["fuel"], c["transport_hauling"]).statistic), 0.02)

    def test_dependence_changes_spread_not_marginals(self):
        dep = run(base(), n_fixed=60_000, seed=9).arrays["C_tot"]
        ind = run(base(dependence="independent"), n_fixed=60_000, seed=9).arrays["C_tot"]
        self.assertAlmostEqual(dep.mean(), ind.mean(), delta=15)   # same marginal means
        self.assertGreater(dep.std(), ind.std())                    # positive dependence widens total cost


class TestConvergenceAndReproducibility(unittest.TestCase):
    """Required revision 10 / report 3.10."""

    def test_stops_when_every_mcse_within_tolerance(self):
        r = run(base())
        s = r.summary
        self.assertTrue(s["converged"])
        self.assertGreaterEqual(s["N"], 10_000)
        for k in E2.PROB_KEYS:
            self.assertLessEqual(s[f"MCSE_{k}"], 0.005)

    def test_reports_non_convergence_honestly(self):
        r = run(base(), n_max=5_000)
        self.assertFalse(r.summary["converged"])
        self.assertTrue(any("NOT reached" in w for w in r.warnings))
        r2 = run(base(), tol=1e-5, n_max=20_000)
        self.assertFalse(r2.summary["converged"])

    def test_same_seed_same_result_different_seed_differs(self):
        a, b = run(base(), n_fixed=20_000, seed=7), run(base(), n_fixed=20_000, seed=7)
        c = run(base(), n_fixed=20_000, seed=8)
        np.testing.assert_array_equal(a.arrays["E"], b.arrays["E"])
        self.assertFalse(np.array_equal(a.arrays["E"], c.arrays["E"]))
        self.assertEqual(a.manifest["inputs_sha256"], b.manifest["inputs_sha256"])

    def test_staged_table_is_consistent(self):
        r = run(base(), n_fixed=100_000)
        t = r.stage_table
        self.assertEqual(list(t["N"]), [1_000, 10_000, 50_000, 100_000])
        for k in E2.PROB_KEYS:
            self.assertTrue(((t[k] >= 0) & (t[k] <= 1)).all())
            self.assertTrue(t[f"MCSE_{k}"].is_monotonic_decreasing)
            a, b = t.iloc[2], t.iloc[3]                              # 50k vs 100k should agree
            self.assertLess(abs(a[k] - b[k]), 5 * a[f"MCSE_{k}"])

    def test_manifest_survives_missing_file_attribute(self):
        """Code pasted into a notebook cell has no __file__; hashing must not crash the run."""
        saved = E2.__dict__.pop("__file__")
        try:
            m = E2.build_manifest(base(), seed=1)
        finally:
            E2.__file__ = saved
        self.assertEqual(m["engine_sha256"], "unavailable")

    def test_manifest_records_provenance_and_config(self):
        m = run(base(), n_fixed=10_000, seed=11).manifest
        self.assertEqual(m["run_config"]["seed"], 11)
        self.assertIn("s_m", m["provenance"])
        for key in ("engine_version", "engine_sha256", "inputs_sha256", "numpy", "scipy", "python"):
            self.assertIn(key, m)


class TestOutputsAndLabels(unittest.TestCase):
    """Report 3.11-3.12: honest labels, exclusive bands, distributional output."""

    def test_expert_labels_and_no_poverty_trap_wording(self):
        self.assertEqual(E2.LABELS["P_labor_return_below_wage"], "Probability of Labor Return Below Wage Benchmark")
        s = run(base(), n_fixed=10_000).summary
        for k in list(s) + list(E2.LABELS.values()):
            self.assertNotIn("poverty", k.lower())
        for key in ("E_median", "E_mean", "E_P5", "E_P95", "E_expected_shortfall_5pct"):
            self.assertIn(key, s)
        self.assertLessEqual(s["E_expected_shortfall_5pct"], s["E_P5"])

    def test_risk_bands_are_mutually_exclusive_with_exact_edges(self):
        self.assertEqual(risk_band(0.67), "low")
        self.assertEqual(risk_band(0.6699), "moderate")
        self.assertEqual(risk_band(0.34), "moderate")
        self.assertEqual(risk_band(0.3399), "high")
        self.assertEqual({risk_band(p) for p in np.linspace(0, 1, 1001)}, {"low", "moderate", "high"})

    def test_rank_sensitivity_signs(self):
        s = run(base(), n_fixed=60_000).rank_sensitivity
        self.assertGreater(s["yield_Y"], 0)
        self.assertGreater(s["price_P"], 0)
        self.assertLess(s["cost_fuel"], 0)

    def test_scenario_table_p0_monotone(self):
        t = scenario_table(base(), p0=[0.0, 0.1, 0.3], n_fixed=30_000)
        self.assertEqual(len(t), 3)
        self.assertTrue(t["P_negative_operating_return"].is_monotonic_increasing)

    def test_synthetic_profiles_run_and_differ(self):
        out = {k: run(v, n_fixed=20_000).summary for k, v in synthetic_profiles().items()}
        self.assertGreater(out["High-Cost"]["P_negative_operating_return"],
                           out["Baseline"]["P_negative_operating_return"])
        self.assertGreater(out["Low-Purity"]["P_labor_return_below_wage"], 0)


class TestLegacyRegression(unittest.TestCase):
    """V2.0 accounting kernel must reproduce V1.0 equations on IDENTICAL draws.

    Benchmark = the screenshot run (H=8, T=0.85, Cfamily=44.80, Cfixed=80, delta=1.10, Cfuel=160,
    alpha=0.02, eta=0.95, price 7282.35-8900.65, yields 0.33/1.45/19.47) executed with the frozen
    V1.0 code. Mapping: n_c = 1, D_c = 1, h = H, s_m = 1 - T, costs = delta*Cfixed + (1+alpha)*Cfuel,
    household requirement = Cfamily * H.
    """

    def test_kernel_reproduces_v1(self):
        H, T, Cf, W, Cfx, d, Cfu, al, eta = 8, 0.85, 44.80, 65.63, 80.0, 1.10, 160.0, 0.02, 0.95
        v1 = legacy_v1.run_monte_carlo(H, T, Cf, W, Cfx, d, Cfu, al, 0.33, 1.45, 19.47, eta, 7282.35, 8900.65, 100_000)
        K = d * Cfx + (1 + al) * Cfu
        n = len(v1["Gf_array"])
        k = accounting_kernel(v1["Gf_array"], np.full(n, eta), v1["Pg_array"], np.full(n, K), np.full(n, K),
                              s_m=1 - min(T, 0.99), L=float(H), W_opp=W, B_HH=Cf * H, tribute_base="gross")
        np.testing.assert_allclose(k["E"], v1["hourly_efficiency_array"], rtol=1e-12)
        success_v2 = 100.0 * float(np.mean(k["E"] >= W))
        self.assertAlmostEqual(success_v2, v1["success_rate"], places=9)
        self.assertAlmostEqual(float(np.mean(k["Y_HH"])), v1["mean_stochastic_cyt"], places=9)

    def test_v1_family_cost_threshold_is_not_the_wage_threshold(self):
        """Documents the V1.0 flaw: its 'required yield' (family-inclusive) differs from the wage threshold."""
        H, T = 8, 0.85
        v1 = legacy_v1.run_monte_carlo(H, T, 44.80, 65.63, 80.0, 1.10, 160.0, 0.02, 0.33, 1.45, 19.47, 0.95,
                                       7282.35, 8900.65, 50_000)
        K = 1.10 * 80 + 1.02 * 160
        n = len(v1["Gf_array"])
        k = accounting_kernel(v1["Gf_array"], np.full(n, 0.95), v1["Pg_array"], np.full(n, K), np.full(n, K),
                              s_m=0.15, L=8.0, W_opp=65.63, B_HH=44.80 * 8)
        self.assertFalse(np.allclose(k["Y_HH"], k["Y_W"]))


if __name__ == "__main__":
    unittest.main()
