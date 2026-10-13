"""Tests for ablation.py: the switches must do exactly what they claim (verification of the study design)."""
import dataclasses
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ablation as AB                    # noqa: E402
import engine_v2 as E2                   # noqa: E402
import legacy_v1                         # noqa: E402

N = 60_000


class TestSwitches(unittest.TestCase):
    def test_s0_is_exactly_frozen_v1(self):
        """Configuration S0 (V1) reproduces the frozen legacy V1.0 draws and success rate exactly."""
        res = AB.simulate(AB.V1, N)
        v1 = AB.v1_draws(N)
        np.testing.assert_allclose(res["E"], v1["hourly_efficiency_array"], rtol=1e-12)
        self.assertAlmostEqual(1 - AB.metrics(res)["p_wage_below"], v1["success_rate"] / 100, places=12)

    def test_chain_ends_at_full_model(self):
        """Applying every fix in CHAIN_ORDER to V1 reaches exactly the FULL configuration."""
        cfg = AB.V1
        fx = {f[0]: f for f in AB.FIXES}
        for key in AB.CHAIN_ORDER:
            cfg = dataclasses.replace(cfg, **fx[key][3])
        self.assertEqual(dataclasses.replace(cfg, label=""), dataclasses.replace(AB.FULL, label=""))

    def test_inseparable_v1_jackpot_is_guarded(self):
        """Combining V1's heuristic jackpot with a PERT yield model raises, since the two cannot be separated."""
        with self.assertRaises(ValueError):
            AB.simulate(dataclasses.replace(AB.V1, yield_model="pert"), 1000)   # pert + v1 heuristic jackpot

    def test_explicit_labor_lowers_return_every_draw(self):
        """Switching to explicit group-labor accounting lowers the return on every draw."""
        a = AB.simulate(AB.V1, N)["E"]
        b = AB.simulate(dataclasses.replace(AB.V1, labor="explicit"), N)["E"]
        self.assertTrue(np.all(b < a))             # group share / crew is below the synthetic V1 share on every draw

    def test_purity_once_raises_return_every_draw(self):
        """Counting purity once instead of twice raises the return on every draw."""
        a = AB.simulate(AB.V1, N)["E"]
        b = AB.simulate(dataclasses.replace(AB.V1, eta="once"), N)["E"]
        self.assertTrue(np.all(b > a))

    def test_hurdle_share_and_common_random_numbers(self):
        """The zero-yield hurdle hits the target share, and positive draws match the no-hurdle run exactly (CRN)."""
        no0 = AB.simulate(dataclasses.replace(AB.FULL, p0=0.0), N)["Y"]
        with0 = AB.simulate(AB.FULL, N)["Y"]
        zero_share = np.mean(with0 == 0)
        self.assertLessEqual(abs(zero_share - AB.ASSUME["p0"]), 4 * E2.mcse_prob(int(zero_share * N), N))
        pos = with0 > 0
        np.testing.assert_array_equal(with0[pos], no0[pos])      # other components' draws undisturbed

    def test_jackpot_toggle_leaves_ordinary_draws_untouched(self):
        """Toggling the jackpot tail on only changes the rare cycles that actually hit the jackpot threshold."""
        off = AB.simulate(dataclasses.replace(AB.FULL, jackpot="off", p0=0.0), N)["Y"]
        on = AB.simulate(dataclasses.replace(AB.FULL, jackpot="tail", p0=0.0), N)["Y"]
        same = on == off
        self.assertGreater(same.mean(), 0.97)                   # only ~1% of cycles change
        self.assertTrue(np.all(on[~same] >= AB.ASSUME["jack_threshold"]))

    def test_copula_with_zero_rho_equals_independent(self):
        """A copula with rho = 0 produces exactly the same draws as independent costs."""
        a = AB.simulate(dataclasses.replace(AB.FULL, costs="independent"), N)["E"]
        b = AB.simulate(dataclasses.replace(AB.FULL, costs="copula"), N, A={"rho": 0.0})["E"]
        np.testing.assert_array_equal(a, b)

    def test_dependence_widens_total_cost_but_not_its_mean(self):
        """Positive cost dependence widens the spread of total cost while preserving its mean."""
        ind = AB.simulate(dataclasses.replace(AB.FULL, costs="independent"), 200_000)["C"]
        dep = AB.simulate(AB.FULL, 200_000)["C"]
        self.assertGreater(dep.std(), ind.std())
        self.assertAlmostEqual(dep.mean(), ind.mean(), delta=3.0)   # mean-preserving PERT spread

    def test_reproducible_and_seed_sensitive(self):
        """The same seed reproduces identical draws; a different seed produces different draws."""
        a, b = AB.simulate(AB.FULL, N, seed=3), AB.simulate(AB.FULL, N, seed=3)
        c = AB.simulate(AB.FULL, N, seed=4)
        np.testing.assert_array_equal(a["E"], b["E"])
        self.assertFalse(np.array_equal(a["E"], c["E"]))


class TestStudyDesign(unittest.TestCase):
    def test_leave_one_out_matches_forward_step_when_same_config(self):
        """Reverting 'Cost dependence' from the full model matches the forward chain's step before it was added."""
        fwd = AB.forward_chain(N, 42)
        loo = AB.leave_one_out(N, 42)
        row = loo[loo["fix_reverted"] == "Cost dependence"].iloc[0]
        self.assertEqual(row["p_wage_below"], fwd.loc[fwd["step"] == "S7", "p_wage_below"].iloc[0])

    def test_loo_sorted_by_effect_and_complete(self):
        """The leave-one-out table covers every fix and is sorted by decreasing absolute effect size."""
        loo = AB.leave_one_out(N, 42)
        self.assertEqual(len(loo), len(AB.FIXES))
        self.assertTrue(loo["d_full_pp"].abs().is_monotonic_decreasing)

    def test_solo_has_one_row_per_fix(self):
        """The solo table has exactly one row per fix."""
        self.assertEqual(len(AB.solo(N, 42)), len(AB.FIXES))

    def test_sensitivity_has_exactly_one_primary_per_parameter(self):
        """Each swept parameter in the sensitivity grid has exactly one row flagged as the primary assumption."""
        s = AB.sensitivity(20_000, 42)
        for name, grp in s.groupby("parameter"):
            self.assertEqual(int(grp["is_primary_assumption"].sum()), 1, msg=name)

    def test_mode_matching_value_reproduces_v1_mean_yield(self):
        """The mode fraction returned by mode_frac_matching_v1_mean() gives a PERT with V1.0's mean yield."""
        f = AB.mode_frac_matching_v1_mean(200_000)
        a, b = AB.BENCH["Gmin"], AB.BENCH["Gsub"]
        pert_mean = E2.PERT(a, a + f * (b - a), b).mean
        self.assertAlmostEqual(pert_mean, float(AB.v1_draws(200_000)["Gf_array"].mean()), delta=1e-6)

    def test_v1_literal_diagnostic_is_exactly_n_c_times_the_explicit_return(self):
        """Group numerator / H instead of / (n_c * H): every draw is overstated by exactly n_c."""
        d = AB.v1_literal_diagnostic(N)
        s1 = AB.simulate(dataclasses.replace(AB.V1, labor="explicit"), N)["E"]     # same draws, same share, same costs
        np.testing.assert_allclose(d["E"], AB.ASSUME["n_c"] * s1, rtol=1e-9)
        self.assertLess(d["p_wage_below"], AB.metrics({"E": s1, "Pi": s1, "Y": s1})["p_wage_below"])

    def test_full_ablation_model_agrees_with_engine_v2(self):
        """Two independent implementations of the full model must agree within Monte Carlo error."""
        A, b = AB.ASSUME, AB.BENCH
        fixed = b["delta"] * b["Cfixed"] * A["n_c"]
        fuel = (1 + b["alpha"]) * b["Cfuel"] * A["n_c"]
        lo, hi = b["Pmin"], b["Pmax"]
        a_, b_ = b["Gmin"], b["Gsub"]
        inp = E2.Inputs(
            n_c=A["n_c"], D_c=1, hours_per_worker_shift=b["H"], s_m=A["s_group"],
            yield_ord=E2.PERT(a_, a_ + A["mode_frac"] * (b_ - a_), b_), zero=E2.ZeroYield(p0=A["p0"]),
            pi_J=A["pi_J"], jackpot=E2.Jackpot(A["jack_threshold"], A["jack_cap"], A["jack_shape"]),
            price_basis="farmgate", price=E2.PERT(lo, lo + A["price_mode_frac"] * (hi - lo), hi),
            costs=[E2.CostItem("fixed", 1.0, pert=E2.PERT(A["cost_lo"] * fixed, fixed, A["cost_hi"] * fixed)),
                   E2.CostItem("fuel", 1.0, pert=E2.PERT(A["cost_lo"] * fuel, fuel, A["cost_hi"] * fuel))],
            rho_spearman=[[1, A["rho"]], [A["rho"], 1]], W_opp=b["W"], W_opp_source="t", W_opp_date="t")
        r = E2.run(inp, n_fixed=200_000, seed=5)
        m = AB.run_cfg(AB.FULL, 200_000, 5)
        se = float(np.hypot(m["mcse"], r.summary["MCSE_P_labor_return_below_wage"]))
        self.assertLessEqual(abs(m["p_wage_below"] - r.summary["P_labor_return_below_wage"]), 4 * se)
        self.assertLessEqual(abs(m["E_median"] - r.summary["E_median"]), 2.0)


class TestOutputs(unittest.TestCase):
    def test_main_writes_all_files_with_honest_framing(self):
        """main() writes every expected output file and the report uses honest, non-directive framing."""
        with tempfile.TemporaryDirectory() as d:
            AB.main(["--n", "20000", "--out", d])
            for f in ("ablation_forward.csv", "ablation_solo.csv", "ablation_leave_one_out.csv",
                      "ablation_seed_check.csv", "ablation_sensitivity.csv", "ablation_forward.png",
                      "ablation_leave_one_out.png", "ablation_report.md", "ablation_manifest.json"):
                self.assertTrue(os.path.exists(os.path.join(d, f)), f)
            with open(os.path.join(d, "ablation_report.md"), encoding="utf-8") as fh:
                txt = fh.read()
            self.assertIn("scenario result", txt.lower())
            self.assertIn("Detectable?", txt)
            self.assertIn(f"[{AB.ASSUME['jack_threshold']}, {AB.ASSUME['jack_cap']}] g", txt)
            self.assertNotIn("poverty", txt.lower())


if __name__ == "__main__":
    unittest.main()
