"""
Tests for dashboard_logic.py (pure logic) and an offline smoke test of app.py using a fake Streamlit.

The smoke test proves the app code runs end to end and respects common Streamlit argument rules.
It does NOT prove the real UI renders correctly: run `streamlit run app.py` and check the screen.
"""
import dataclasses
import io
import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import dashboard_logic as DL             # noqa: E402
import engine_v2 as E2                   # noqa: E402
import legacy_v1                         # noqa: E402
from fake_streamlit import run_app       # noqa: E402
from scenarios import batch_synthetic, fixed_synthetic  # noqa: E402

BASE = DL.PRESETS[0]
FORBIDDEN = ["debt trap", "immediate exit", "greenlight", "gambling", "real economic sustainability",
             "capital hold"]


def vals(**kw):
    return {**DL.preset_values(BASE), "n_max": 50_000, **kw}


class TestMapping(unittest.TestCase):
    def test_ui_values_round_trip_to_identical_simulation(self):
        a = E2.run(dataclasses.replace(batch_synthetic(), W_opp_source="", W_opp_date=""), n_fixed=20_000, seed=42)
        b = E2.run(DL.build_inputs(DL.preset_values(BASE), BASE), n_fixed=20_000, seed=42)
        np.testing.assert_array_equal(a.arrays["E"], b.arrays["E"])

    def test_fixed_transport_alias_preserves_rate_correlation(self):
        values = DL.values_from_inputs(fixed_synthetic())
        self.assertEqual(values["rho_fuel_trans"], .5)
        restored = DL.build_inputs(values)
        self.assertEqual(restored.rho_spearman[0][3], .5)
        self.assertEqual(restored.rho_spearman[3][0], .5)
        E2.validate(restored)

    def test_all_presets_build_and_run(self):
        for p in DL.PRESETS:
            inp, res = DL.run_v2({**DL.preset_values(p), "n_max": 50_000}, p)
            self.assertTrue(res.summary["converged"], p)
        self.assertEqual(DL.preset_values(DL.PRESETS[2])["price_basis"], "reference")

    def test_equipment_and_counts_paths(self):
        inp = DL.build_inputs(vals(eq_mode="purchased", eq_acq=9000.0, eq_salv=0.0, eq_life=90.0, p0_mode="from counts", z=2, n_obs=10))
        self.assertEqual(len(inp.equipment), 1)
        self.assertEqual(inp.zero.z, 2)
        E2.validate(inp)

    def test_jackpot_off_when_probability_zero(self):
        self.assertIsNone(DL.build_inputs(vals(pi_J=0.0)).jackpot)

    def test_reference_basis_uses_purity_farmgate_ignores_it(self):
        self.assertEqual(DL.build_inputs(vals(price_basis="farmgate")).purity, 1.0)
        self.assertIsInstance(DL.build_inputs(vals(price_basis="reference")).purity, E2.PERT)

    def test_invalid_entries_raise_readable_errors(self):
        with self.assertRaises(E2.InputError) as cm:
            DL.run_v2(vals(ya=2.0, yb=1.0), BASE)
        self.assertIn("PERT", str(cm.exception))
        with self.assertRaises(E2.InputError):
            DL.run_v2(vals(p0_mode="from counts", z=9, n_obs=3), BASE)
        with self.assertRaises(E2.InputError):
            DL.run_v2(vals(jack_thr=30.0, jack_cap=24.0), BASE)


class TestProvenance(unittest.TestCase):
    def test_default_values_keep_preset_tag_and_edits_become_user(self):
        self.assertEqual(DL.tag_for("n_c", 4, BASE), "synthetic")
        self.assertEqual(DL.tag_for("n_c", 6, BASE), "user")
        self.assertEqual(DL.tag_for("s_m", 0.50, BASE), "synthetic")
        self.assertEqual(DL.tag_for("s_m", 0.7, BASE), "user")
        self.assertEqual(DL.tag_for("tribute_base", "gross", BASE), "synthetic")

    def test_synthetic_presets_tag_their_synthetic_inputs(self):
        self.assertEqual(DL.tag_for("fuel_mode", DL.preset_values(DL.PRESETS[1])["fuel_mode"], DL.PRESETS[1]), "synthetic")
        self.assertEqual(DL.tag_for("q_mode", DL.preset_values(DL.PRESETS[2])["q_mode"], DL.PRESETS[2]), "synthetic")

    def test_effective_tags_flow_into_manifest(self):
        inp, res = DL.run_v2(vals(n_c=6), BASE)
        self.assertEqual(inp.provenance["n_c"], "user")
        self.assertEqual(inp.provenance["s_m"], "synthetic")
        m = json.loads(DL.export_manifest(res, {"preset": BASE}))
        self.assertEqual(m["provenance"]["n_c"], "user")
        self.assertEqual(m["preset"], BASE)
        self.assertIn("summary", m)

    def test_assumed_list_matches_tags(self):
        tags = DL.effective_tags(DL.preset_values(BASE), BASE)
        self.assertEqual(DL.assumed_inputs(tags), [])
        self.assertEqual(DL.assumed_inputs({"example": "assumed"}), ["example"])
        self.assertNotIn("s_m", DL.assumed_inputs(tags))


class TestOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inp, cls.res = DL.run_v2(vals(), BASE)

    def test_tables_use_expert_labels_and_three_thresholds(self):
        self.assertEqual(list(DL.risk_table(self.res)["Output"]), list(E2.LABELS.values()))
        self.assertEqual(len(DL.thresholds_table(self.res)), 3)
        self.assertEqual(DL.return_table(self.res).iloc[-1, 0], "Expected shortfall (mean of worst 5%)")

    def test_interpretation_is_neutral_and_reports_mcse(self):
        t = DL.interpretation_text(self.res, self.inp, 20).lower()
        self.assertIn("2 mcse", t)
        self.assertIn("not predictions", t)
        for bad in FORBIDDEN:
            self.assertNotIn(bad, t)

    def test_risk_band_text_flags_illustrative_cutoffs(self):
        self.assertIn("not validated", DL.risk_band_text(self.res))

    def test_figures_have_labelled_axes(self):
        figs = [DL.fig_return_density(self.res.arrays["E"], 100.0),
                DL.fig_yield_exceedance(self.res.arrays["Y"], {"BE": 0.5, "W": 0.8, "HH": 1.2}),
                DL.fig_stability(self.res.stage_table), DL.fig_rank_sensitivity(self.res.rank_sensitivity)]
        for f in figs:
            ax = f.axes[0]
            self.assertTrue(ax.get_xlabel())
            self.assertNotIn("poverty", ax.get_title().lower())

    def test_density_figure_survives_constant_returns(self):
        DL.fig_return_density(np.full(500, 70.0), 100.0)

    def test_csv_export_shape_and_columns(self):
        df = __import__("pandas").read_csv(io.BytesIO(DL.export_csv(self.res, 1000)))
        self.assertEqual(len(df), 1000)
        for c in ("yield_g", "return_php_per_person_hour", "Y_breakeven_g", "Y_wage_g", "Y_household_g"):
            self.assertIn(c, df.columns)

    def test_manifest_is_valid_json_with_versions(self):
        m = json.loads(DL.export_manifest(self.res))
        for k in ("dashboard_version", "engine_version", "inputs_sha256", "run_config", "warnings"):
            self.assertIn(k, m)

    def test_unsourced_wage_gives_visible_message(self):
        msgs = DL.messages(self.res, DL.effective_tags(DL.preset_values(BASE), BASE))
        self.assertTrue(any("W_opp" in t or "wage" in t.lower() for _, t in msgs))


class TestLegacy(unittest.TestCase):
    def test_run_legacy_matches_frozen_function(self):
        v = {**DL.LEGACY_DEFAULTS, "N": 20_000}
        a = DL.run_legacy(v)
        b = legacy_v1.run_monte_carlo(8, .8, 90., 100., 1200., 1.1, 800., .02, 5., 20., 60., .9,
                                      5000., 6000., 20_000)
        self.assertEqual(a["success_rate"], b["success_rate"])

    def test_legacy_validation_matches_v1_rules(self):
        e = DL.legacy_errors({**DL.LEGACY_DEFAULTS, "Gmin": 25.0})                   # min above typical max
        self.assertEqual(len(e), 1)
        self.assertIn("minimum yield", e[0])
        self.assertEqual(len(DL.legacy_errors({**DL.LEGACY_DEFAULTS, "Gjack": 1.0, "Pmin": 9000.0})), 2)
        self.assertEqual(DL.legacy_errors(DL.LEGACY_DEFAULTS), [])

    def test_no_network_imports(self):
        for f in ("app.py", "dashboard_logic.py"):
            with open(os.path.join(os.path.dirname(HERE), f), encoding="utf-8") as fh:
                src = fh.read()
            for bad in ("import requests", "urllib", "http.client", "socket"):
                self.assertNotIn(bad, src, f)


class TestAppSmoke(unittest.TestCase):
    """Runs app.main() against a fake Streamlit (no real UI)."""

    def test_first_load_shows_instructions_not_results(self):
        fake, stopped = run_app(submit=False)
        self.assertFalse(stopped)
        self.assertTrue(any("press 'run project aurum model'" in t.lower() for t in fake.texts()))
        self.assertEqual(fake.calls("metric"), [])

    def test_default_run_renders_all_tabs_and_expert_labels(self):
        fake, stopped = run_app(overrides={"n_max": 50_000}, submit=True)
        self.assertFalse(stopped)
        metrics = [a[0] for a, _ in fake.calls("metric")]
        self.assertEqual(metrics, list(E2.LABELS.values()))
        self.assertEqual(len(fake.calls("pyplot")), 4)
        self.assertEqual(len(fake.calls("download_button")), 3)
        self.assertEqual(len(fake.calls("tabs")[0][0]), 6)
        blob = " ".join(fake.texts()).lower()
        self.assertIn("scenario results", blob)
        for bad in FORBIDDEN:
            self.assertNotIn(bad, blob)
        self.assertFalse(any("poverty" in m.lower() for m in metrics))

    def test_invalid_inputs_show_error_and_stop(self):
        fake, stopped = run_app(overrides={"ya": 2.0, "yb": 1.0, "n_max": 50_000}, submit=True)
        self.assertTrue(stopped)
        errs = [str(a[0]) for a, _ in fake.calls("error")]
        self.assertTrue(errs and "not valid" in errs[0])
        self.assertEqual(fake.calls("metric"), [])

    def test_preset_change_asks_for_rerun(self):
        f1, _ = run_app(overrides={"n_max": 50_000}, submit=True)
        f2, _ = run_app(overrides={"preset": DL.PRESETS[1]}, submit=False, session_state=f1.session_state)
        self.assertTrue(any("preset changed" in t.lower() for t in f2.texts()))
        self.assertEqual(f2.calls("metric"), [])

    def test_scenario_grid_button_populates_tables(self):
        """Running the scenario grid populates the overview, convergence and grid dataframes."""
        f1, _ = run_app(overrides={"n_max": 50_000}, submit=True)
        f2, _ = run_app(submit=False, press={"Run scenario grid"}, session_state=f1.session_state)
        self.assertEqual(len(f2.calls("dataframe")), 4 + 2 + 1)      # overview 4, convergence 1, grid 2

    def test_fixed_timing_explains_unsupported_daily_rates_before_submit(self):
        fake, stopped = run_app(overrides={"timing_mode": "fixed"}, submit=False)
        self.assertFalse(stopped)
        warnings = " ".join(str(a[0]) for a, _ in fake.calls("warning"))
        self.assertIn("Explosives: fixed timing has no working-day count", warnings)
        self.assertIn("enter its corresponding rates", warnings)

    def test_legacy_mode_renders_with_warning_and_flaw_list(self):
        fake, stopped = run_app(overrides={"version": "V1.0 (legacy, frozen - comparison only)", "N": 10_000}, submit=True)
        self.assertFalse(stopped)
        blob = " ".join(fake.texts()).lower()
        self.assertIn("comparison only", blob)
        self.assertIn("known v1.0 flaws", blob)
        self.assertEqual(len(fake.calls("pyplot")), 1)
        self.assertNotIn("debt trap", blob)

    def test_legacy_invalid_ordering_is_rejected(self):
        fake, stopped = run_app(overrides={"version": "V1.0 (legacy, frozen - comparison only)", "Gmin": 25.0}, submit=True)
        self.assertTrue(stopped)
        self.assertTrue(fake.calls("error"))

    def test_user_edit_changes_tag_in_manifest(self):
        fake, _ = run_app(overrides={"n_c": 7, "n_max": 50_000}, submit=True)
        st_ = fake.session_state["v2"]
        self.assertEqual(st_["tags"]["n_c"], "user")
        self.assertEqual(st_["inp"].n_c, 7)

    def test_fake_enforces_mixed_type_rule(self):
        from fake_streamlit import _check_numeric
        with self.assertRaises(TypeError):
            _check_numeric("x", 1, 0.0, 10.0, 1, False)


if __name__ == "__main__":
    unittest.main()
