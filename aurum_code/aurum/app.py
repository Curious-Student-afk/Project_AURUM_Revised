"""
app.py -- Project AURUM dashboard (Streamlit).

    streamlit run app.py          (works offline; no network calls)

Thin layer: all numerics live in engine_v2.py and all non-widget logic in dashboard_logic.py.
Only long-standing Streamlit calls are used (set_page_config, sidebar, form, number_input, slider, selectbox,
radio, text_input, tabs, columns, metric, dataframe, pyplot, download_button, expander, session_state, stop).
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import dashboard_logic as DL
import engine_v2 as E2

VERSIONS = ["V2.3 (current)", "V1.0 (legacy, frozen - comparison only)"]
N_MAX_CHOICES = [50_000, 100_000, 200_000, 500_000]


def _slug(text: str) -> str:
    """Turn `text` into a widget-key-safe slug (alphanumerics kept, everything else becomes '_')."""
    return "".join(ch if ch.isalnum() else "_" for ch in text)


def _show_fig(fig) -> None:
    """Render a Matplotlib figure in Streamlit, then close it to release memory."""
    st.pyplot(fig)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------
# V2.3 sidebar form
# ------------------------------------------------------------------------------------------------
def v2_form(preset: str) -> tuple:
    """Draw the input form. Returns (values dict, submitted flag)."""
    ps = _slug(preset)
    defaults = DL.preset_values(preset)

    def label(name: str, text: str) -> str:
        """Widget label for `name`, suffixed with its current provenance tag."""
        cur = st.session_state.get(f"{ps}__{name}", defaults[name])
        return f"{text}  {DL.TAG_LABEL[DL.tag_for(name, cur, preset)]}"

    def num(name, text, help_text=None, **kw):
        """Draw a tagged number_input widget bound to `name`."""
        return st.number_input(label(name, text), key=f"{ps}__{name}", help=help_text, **kw)

    def sld(name, text, help_text=None, **kw):
        """Draw a tagged slider widget bound to `name`."""
        return st.slider(label(name, text), key=f"{ps}__{name}", help=help_text, **kw)

    def rad(name, text, options, help_text=None):
        """Draw a tagged radio widget bound to `name`, defaulting to the preset's value."""
        return st.radio(label(name, text), options, index=options.index(defaults[name]), key=f"{ps}__{name}",
                        help=help_text)

    d = defaults
    v = dict(d)
    for name, title, options in (("timing_mode", "Timing model", ["batch", "fixed"]),
                                  ("price_basis", "Gold price basis", ["farmgate", "reference"]),
                                  ("share_preset", "Miner/financier split", ["Custom", "50/50", "60/40", "65/35"])):
        v[name] = st.sidebar.radio(label(name, title), options, index=options.index(d[name]), key=f"{ps}__{name}")
    assay_options = ["PERT fraction", "measured value"] if v["price_basis"] == "reference" else ["unknown", "PERT fraction", "measured value"]
    assay_default = d["assay_mode"] if d["assay_mode"] in assay_options else assay_options[0]
    v["assay_mode"] = st.sidebar.radio("Physical assay input", assay_options, index=assay_options.index(assay_default),
                                      key=f"{ps}_{v['price_basis']}__assay_mode")
    with st.sidebar.form(key=f"form_{ps}"):
        st.markdown("**1. Batch timing and labour**")
        v["n_c"] = num("n_c", "Active crew size", "Every active contributor attends every scheduled shift. Partial attendance and changing crew membership are unsupported in batch mode.",
                       min_value=1, max_value=100, value=int(d["n_c"]), step=1)
        if v["timing_mode"] == "batch":
            st.caption("Working days: discretized Beta-PERT, rounded to the nearest whole day. Idle days: continuous Beta-PERT.")
            for prefix, title in (("work", "Active working days"), ("idle", "Idle days")):
                for suffix in ("min", "mode", "max"):
                    key = f"{prefix}_{suffix}"
                    if prefix == "work":
                        v[key] = num(key, f"{title}: {suffix}", min_value=1, max_value=3650, value=int(d[key]), step=1)
                    else:
                        v[key] = num(key, f"{title}: {suffix}", min_value=0., max_value=3650., value=float(d[key]), step=.25)
            st.caption("Out of ten working days, how many had 1, 2, or 3 shifts? Counts must total ten.")
            for count in (1, 2, 3):
                key = f"days_{count}"
                v[key] = num(key, f"Days with {count} shift(s)", min_value=0, max_value=10, value=int(d[key]), step=1)
                for suffix in ("min", "mode", "max"):
                    key = f"h{count}_{suffix}"
                    v[key] = num(key, f"Hours per shift on {count}-shift days: {suffix}",
                                 min_value=.5, max_value=min(16., 24. / count), value=float(d[key]), step=.25)
        else:
            v["D_c"] = num("D_c", "Fixed shifts per cycle", min_value=1, max_value=3650, value=int(d["D_c"]), step=1)
            v["calendar_days"] = num("calendar_days", "Fixed calendar days", min_value=.01, max_value=3650.,
                                     value=float(d["calendar_days"]), step=.25)
            v["h"] = num("h", "Fixed hours per worker per shift", min_value=.5, max_value=24., value=float(d["h"]), step=.25)

        st.markdown("**2. Stakeholder accounting**")
        v["s_m"] = num("s_m", "Miner-group revenue share s_m", "Group share of the revenue base (1 minus the aggregate non-miner share).",
                       min_value=0.01, max_value=1.0, value=float(d["s_m"]), step=0.01, format="%.4f", disabled=v["share_preset"] != "Custom")
        v["tribute_base"] = rad("tribute_base", "Share applies to", ["gross", "net"],
                                "gross: miners pay their cost share out of their revenue share. "
                                "net: all operation costs are deducted first, then the share is applied; individual cost shares do not alter this contract.")

        st.markdown("**3. Gold yield** (group total, recovered grams per batch)")
        v["ya"] = num("ya", "Typical minimum yield a (g)", "Minimum ordinary batch yield.",
                      min_value=0.0, max_value=1000.0, value=float(d["ya"]), step=0.05, format="%.3f")
        v["yb"] = num("yb", "Typical maximum yield b (g)", "Maximum ordinary batch yield.",
                      min_value=0.0, max_value=1000.0, value=float(d["yb"]), step=0.05, format="%.3f")
        v["ym"] = num("ym", "Most likely ordinary batch yield (g)",
                      min_value=0., max_value=1000., value=float(d["ym"]), step=.05, format="%.3f")
        v["p0_mode"] = rad("p0_mode", "Zero-yield probability from", ["fixed value", "from counts"],
                           "counts: p0 ~ Beta(1+z, 1+n-z), drawn for every simulated batch.")
        v["p0"] = sld("p0", "Zero-yield (waste cycle) probability p0", "Probability that a completed batch has no recovered gold.",
                      min_value=0.0, max_value=1.0, value=float(d["p0"]), step=0.01)
        v["z"] = num("z", "Zero-yield completed batches z", None, min_value=0, max_value=1000, value=int(d["z"]), step=1)
        v["n_obs"] = num("n_obs", "Completed batches recorded n", None, min_value=0, max_value=1000, value=int(d["n_obs"]), step=1)
        v["pi_J"] = num("pi_J", "Jackpot probability pi_J (given a productive batch)", "Assumed. Set 0 for no jackpot component.",
                        min_value=0.0, max_value=1.0, value=float(d["pi_J"]), step=0.005, format="%.4f")
        v["jack_thr"] = num("jack_thr", "Jackpot threshold (g)", "Lower bound of the truncated-Pareto batch jackpot.",
                            min_value=0.1, max_value=1000.0, value=float(d["jack_thr"]), step=0.5, format="%.2f")
        v["jack_cap"] = num("jack_cap", "Jackpot cap (g)", "Physical upper bound for one batch.",
                            min_value=0.1, max_value=1000.0, value=float(d["jack_cap"]), step=0.5, format="%.2f")
        v["jack_shape"] = num("jack_shape", "Jackpot tail shape (Pareto)", "Assumed; smaller = heavier tail.",
                              min_value=0.2, max_value=5.0, value=float(d["jack_shape"]), step=0.1, format="%.2f")

        st.markdown("**4. Price and purity** (handled exactly once)")
        st.caption("Farmgate pricing values recovered gold directly. Reference pricing applies purity to obtain fine gold mass.")
        v["price_input"] = rad("price_input", "Price entered as", ["direct", "sale"])
        st.caption("Sale conversion uses total PHP paid divided by recovered grams sold, and requires farmgate pricing.")
        v["sale_proceeds"] = num("sale_proceeds", "Sale proceeds (PHP)", min_value=.01, value=float(d["sale_proceeds"]), step=1.)
        v["sale_mass"] = num("sale_mass", "Recovered mass sold (g)", min_value=.001, value=float(d["sale_mass"]), step=.1)
        v["p_min"] = num("p_min", "Price minimum (PHP/g)", None, min_value=1.0, max_value=1_000_000.0,
                         value=float(d["p_min"]), step=50.0, format="%.2f")
        v["p_mode"] = num("p_mode", "Price most likely (PHP/g)", "Most likely price under this scenario.",
                          min_value=1.0, max_value=1_000_000.0, value=float(d["p_mode"]), step=50.0, format="%.2f")
        v["p_max"] = num("p_max", "Price maximum (PHP/g)", None, min_value=1.0, max_value=1_000_000.0,
                         value=float(d["p_max"]), step=50.0, format="%.2f")
        v["q_min"] = num("q_min", "Physical purity minimum (fraction)", None, min_value=0.01,
                         max_value=1.0, value=float(d["q_min"]), step=0.01, format="%.3f", disabled=v["assay_mode"] != "PERT fraction")
        v["q_mode"] = num("q_mode", "Physical purity most likely (fraction)", None, min_value=0.01, max_value=1.0,
                          value=float(d["q_mode"]), step=0.01, format="%.3f", disabled=v["assay_mode"] != "PERT fraction")
        v["q_max"] = num("q_max", "Physical purity maximum (fraction)", None, min_value=0.01, max_value=1.0,
                         value=float(d["q_max"]), step=0.01, format="%.3f", disabled=v["assay_mode"] != "PERT fraction")

        v["assay_unit"] = rad("assay_unit", "Measured assay unit", ["fraction", "percent", "karat"])
        v["assay_value"] = num("assay_value", "Measured assay value", min_value=.001, max_value=100.,
                               value=float(d["assay_value"]), step=.1, disabled=v["assay_mode"] != "measured value")
        v["measurement_source"] = st.text_input("Measurement source", value=str(d["measurement_source"]), key=f"{ps}__measurement_source")
        v["measurement_date"] = st.text_input("Measurement date (YYYY-MM-DD, optional)", value=str(d["measurement_date"]), key=f"{ps}__measurement_date")
        st.caption("Unknown farmgate assay stays blank in corrected exports. Physical assay never changes farmgate valuation.")

        st.markdown("**5. Expense rates for the whole operation**")
        st.caption("Food is additional worksite consumption excluded from household needs. Operating hours sum scheduled shift durations, without multiplying by crew size.")
        v["fixed_cost"] = num("fixed_cost", "Additional fixed overhead (PHP/batch)", min_value=0.,
                              max_value=10_000_000., value=float(d["fixed_cost"]), step=10.)
        v["fixed_share"] = num("fixed_share", "Miner fraction of overhead", min_value=0., max_value=1.,
                               value=float(d["fixed_share"]), step=.5)
        v["fixed_components"] = st.text_input("Overhead component identifiers (comma separated)", value=str(d["fixed_components"]),
                                               key=f"{ps}__fixed_components", help="Required for nonzero overhead; exclude fuel, milling and other separately listed components.")
        v["fuel_input"] = rad("fuel_input", "Fuel entered as", ["direct", "liters × PHP/liter"])
        v["fuel_liters"] = num("fuel_liters", "Fuel liters per selected expense basis", min_value=0., value=float(d["fuel_liters"]), step=1.)
        v["fuel_unit_price"] = num("fuel_unit_price", "Fuel unit price (PHP/liter)", min_value=.01, value=float(d["fuel_unit_price"]), step=1.)
        for prefix, name in DL.COST_FIELDS.items():
            key = f"{prefix}_basis"
            options = list(E2.COST_BASES)
            v[key] = rad(key, f"{name.title()} expense basis", options)
            if v["timing_mode"] == "fixed" and v[key] == "per_working_day":
                st.warning(f"{name.title()}: fixed timing has no working-day count. Choose a per-batch, "
                           "per-shift, operating-hour, or calendar-day basis and enter its corresponding "
                           "rates before running. The current daily rate is not converted automatically.")
            for suffix in ("min", "mode", "max"):
                key = f"{prefix}_{suffix}"
                v[key] = num(key, f"{name.title()} rate: {suffix}", "Units follow the selected expense basis; whole operation.",
                             min_value=0., max_value=10_000_000., value=float(d[key]), step=10.)
            key = f"{prefix}_share"
            v[key] = num(key, f"Miner fraction of {name}", "0 = financier, 0.5 = split, 1 = miners.",
                         min_value=0., max_value=1., value=float(d[key]), step=.5)
        v["dependence"] = rad("dependence", "Expense-rate dependence", ["copula", "independent"])
        st.caption("Spearman targets apply to rates before exposure scaling. The diagonal is one; entries below define both symmetric cells.")
        for i, j, key in DL.correlation_keys():
            names = list(DL.COST_FIELDS.values())
            v[key] = num(key, f"Rate correlation: {names[i]} / {names[j]}",
                         min_value=-.95, max_value=.95, value=float(d[key]), step=.05)

        st.markdown("**6. Equipment** (purchased OR rented, never blended)")
        v["eq_mode"] = rad("eq_mode", "Main equipment", ["none", "purchased", "rented"])
        v["eq_acq"] = num("eq_acq", "Acquisition cost (PHP)", None, min_value=0.0, max_value=100_000_000.0,
                          value=float(d["eq_acq"]), step=100.0, format="%.2f")
        v["eq_salv"] = num("eq_salv", "Salvage value (PHP)", None, min_value=0.0, max_value=100_000_000.0,
                           value=float(d["eq_salv"]), step=100.0, format="%.2f")
        v["eq_life"] = num("eq_life", "Replacement life (cycles)", None, min_value=1.0, max_value=1_000_000.0,
                           value=float(d["eq_life"]), step=10.0, format="%.1f")
        v["eq_util"] = num("eq_util", "Utilization (0-1)", None, min_value=0.0, max_value=1.0,
                           value=float(d["eq_util"]), step=0.05, format="%.2f")
        v["eq_rent"] = num("eq_rent", "Rental rate per shift (PHP)", None, min_value=0.0, max_value=1_000_000.0,
                           value=float(d["eq_rent"]), step=10.0, format="%.2f")
        v["eq_share"] = num("eq_share", "Share paid by miners (equipment)", None, min_value=0.0, max_value=1.0,
                            value=float(d["eq_share"]), step=0.05, format="%.2f")

        st.markdown("**Reserve fund** (per batch: replacement or custom emergency target)")
        v["reserve_mode"] = rad("reserve_mode", "Reserve target", ["none", "match_depreciation", "custom"],
                                "None is the default. Matching depreciation is a savings target, not another economic expense.")
        v["reserve_location"] = rad("reserve_location", "Reserve funded from",
                                    ["choose", "before_sharing", "after_sharing"],
                                    "Before: common fund from operation cash after cash costs. After: miner fund from allocated cash. Choose explicitly when enabled.")
        v["reserve_priority"] = rad("reserve_priority", "When cash is insufficient",
                                    ["choose", "household_first", "reserve_first"],
                                    "Household-first protects modeled miner household needs; unfunded reserves are reported.")
        v["reserve_custom"] = num("reserve_custom", "Custom reserve target (PHP/cycle, selected fund)",
                                  min_value=0.0, max_value=100_000_000.0, value=float(d["reserve_custom"]), step=100.0)

        st.markdown("**7. Benchmarks** (normative thresholds)")
        v["W_opp"] = num("W_opp", "Opportunity-wage benchmark (PHP/person-hour)", "Cite the wage order below.",
                         min_value=0.01, max_value=5000.0, value=float(d["W_opp"]), step=1.0, format="%.2f")
        v["W_src"] = st.text_input("Wage benchmark source", value=str(d["W_src"]), key=f"{ps}__W_src")
        v["W_date"] = st.text_input("Wage benchmark effective date", value=str(d["W_date"]), key=f"{ps}__W_date")
        v["hh_need"] = num("hh_need", "Household consumption W_daily (PHP/household/calendar day)", "Excludes additional worksite food. Household count and outside income are independent of active crew size.",
                           min_value=0.0, max_value=1_000_000.0, value=float(d["hh_need"]), step=25.0, format="%.2f")
        v["households_mode"] = rad("households_mode", "Household count", ["one per worker", "explicit"])
        v["n_hh"] = num("n_hh", "Households supported (explicit zero is allowed)", min_value=0, max_value=200,
                        value=int(d["n_hh"]), step=1)
        v["nonmining"] = num("nonmining", "Non-mining income (PHP/household/calendar day)", min_value=0.,
                             max_value=1_000_000., value=float(d["nonmining"]), step=25.)

        st.markdown("**8. Simulation precision**")
        v["seed"] = st.number_input("Random seed", key=f"{ps}__seed", help="Same seed + same inputs = same results.",
                                    min_value=0, max_value=2_000_000_000, value=int(d["seed"]), step=1)
        v["tol"] = st.number_input("MCSE tolerance for every probability", key=f"{ps}__tol", min_value=0.001,
                                   max_value=0.05, value=float(d["tol"]), step=0.001, format="%.3f")
        v["n_max"] = st.selectbox("Maximum cycles N", N_MAX_CHOICES, index=N_MAX_CHOICES.index(int(d["n_max"])),
                                  key=f"{ps}__n_max", help="The run stops when probability and return precision, stability and mixture support pass.")
        v["return_abs_tol"] = num("return_abs_tol", "Return error tolerance (PHP/person-hour)", min_value=.01,
                                  max_value=1000., value=float(d["return_abs_tol"]), step=.5)
        v["return_rel_tol"] = num("return_rel_tol", "Return relative error tolerance (fraction)", min_value=0.,
                                  max_value=1., value=float(d["return_rel_tol"]), step=.005)
        st.caption("Use the larger absolute/relative tolerance. Two precision checks, stable returns and mixture support are required.")
        submitted = st.form_submit_button("Run Project AURUM model")
    return v, submitted


# ------------------------------------------------------------------------------------------------
# V2.3 results
# ------------------------------------------------------------------------------------------------
def render_v2(state: dict) -> None:
    """Render the V2.3 results tabs (overview, distributions, convergence, scenario, export, about)."""
    inp, res, tags = state["inp"], state["res"], state["tags"]
    s = res.summary
    assumed = DL.assumed_inputs(tags)
    user = [k for k, t in tags.items() if t == "user"]

    tab_over, tab_dist, tab_conv, tab_scen, tab_export, tab_about = st.tabs(
        ["Overview", "Distributions", "Convergence and sensitivity", "Scenario analysis", "Export and reproducibility",
         "How to read this"])

    with tab_over:
        st.info("Scenario results under the stated inputs. Presets are synthetic demonstrations.")
        st.markdown(f"**{len(assumed)}** inputs are 🟠 assumed and **{len(user)}** were 🟣 changed by you.")
        with st.expander("Which inputs are assumptions?"):
            st.write(", ".join(assumed) if assumed else "None.")
            st.caption("; ".join(f"{DL.TAG_LABEL[k]}: {DL.TAG_HELP[k]}" for k in DL.TAG_LABEL))
        for level, text in DL.messages(res, tags):
            getattr(st, level)(text)

        st.subheader("Risk probabilities")
        for offset in range(0, len(E2.PROB_KEYS), 4):
            cols = st.columns(4)
            for col, key in zip(cols, E2.PROB_KEYS[offset:offset + 4]):
                with col:
                    st.metric(E2.LABELS[key], f"{s[key]:.1%}")
                    st.caption(f"± {2 * s['MCSE_' + key]:.2%} (2 MCSE)")
        st.markdown(DL.risk_band_text(res))
        st.caption(f"N = {s['N']:,} simulated batches - convergence {'reached' if s['converged'] else 'NOT reached'}")
        st.subheader("Timing and household requirements")
        st.dataframe(DL.timing_table(res))

        left, right = st.columns(2)
        with left:
            st.subheader("Economic return per person-hour")
            st.dataframe(DL.return_table(res))
        with right:
            st.subheader("Yield thresholds (kept separate)")
            st.dataframe(DL.thresholds_table(res))
        st.subheader("Cash and equipment reserve")
        st.caption("Amounts are per cycle. Reserve totals refer to the selected common or miner fund; "
                   "the miner contribution is shown separately. No borrowing is assumed.")
        st.dataframe(DL.cash_table(res))
        st.markdown(DL.interpretation_text(res, inp, len(assumed)))

    with tab_dist:
        st.subheader("Distributions of the simulated cycles")
        c1, c2 = st.columns(2)
        with c1:
            _show_fig(DL.fig_return_density(res.arrays["E"], inp.W_opp))
            st.caption("Red: cycles with a labor return below the wage benchmark. Green: at or above. The axis is trimmed; "
                       "rare jackpot cycles extend far to the right and raise the mean above the median.")
        with c2:
            medians = {name: float(np.median(res.arrays[key])) for name, key, _ in DL.THRESHOLD_ROWS}
            _show_fig(DL.fig_yield_exceedance(res.arrays["Y"], medians))
            st.caption("The curve starts below 100% when zero-yield cycles are possible. Thresholds vary by draw; medians shown.")

    with tab_conv:
        st.subheader("Stability as N grows")
        st.caption(f"Probability precision: {s['probability_precision_met']}; return precision: {s['return_precision_met']}; full convergence: {s['converged']}")
        st.caption("Numerical bounds estimate sampling error. They are not model-uncertainty or predictive intervals.")
        if "p0_posterior_lower_95" in s:
            st.info(f"Zero-probability posterior 95% interval: {s['p0_posterior_lower_95']:.1%}–{s['p0_posterior_upper_95']:.1%}. This is parameter uncertainty, separate from MCSE.")
        st.dataframe(res.stage_table.round(4))
        _show_fig(DL.fig_stability(res.stage_table))
        st.subheader("Rank-based sensitivity")
        if res.rank_sensitivity:
            _show_fig(DL.fig_rank_sensitivity(res.rank_sensitivity))
        st.caption("Spearman rank correlation of each simulated input with the labor return. It shows direction and "
                   "strength of association; Sobol variance shares are future work.")

    with tab_scen:
        st.subheader("Wage and illustrative cutoff sensitivity (same simulated draws)")
        st.dataframe(DL.benchmark_sensitivity(res, inp))
        st.caption("These comparisons do not establish policy validity for the wage or cutoffs.")
        st.subheader("Spread over the assumed parameters")
        st.write("Re-runs the model (20,000 cycles each) over assumed values of the zero-yield probability and the yield "
                 "mode, and over crew size. Use the range, not one number.")
        if st.button("Run scenario grid"):
            with st.spinner("Running scenarios..."):
                grid = E2.scenario_table(inp, p0=[0.0, 0.1, 0.2, 0.3], mode_frac=[0.25, 0.4, 0.6], n_fixed=20_000,
                                         seed=int(state["values"]["seed"]))
                crew = E2.scenario_table(inp, n_c=[3, 4, 5, 6, 8], n_fixed=20_000, seed=int(state["values"]["seed"]))
            st.session_state["v2_grid"] = {"grid": grid, "crew": crew, "for": state["run_id"]}
        g = st.session_state.get("v2_grid")
        if g and g["for"] == state["run_id"]:
            cols_show = ["p0", "mode_frac", "n_c", "P_negative_operating_return", "P_labor_return_below_wage",
                         "P_fail_household_adequacy", "E_median",
                         "P_cash_before_reserve_below_household", "P_cash_after_reserve_below_household",
                         "P_reserve_underfunded", "P_household_and_reserve_met",
                         "cash_after_reserve_mean", "reserve_shortfall_mean"]
            st.markdown("**Zero-yield probability x yield mode**")
            st.dataframe(g["grid"][[c for c in cols_show if c in g["grid"].columns and c != "n_c"]].round(3))
            st.markdown("**Crew size**")
            st.dataframe(g["crew"][[c for c in cols_show if c in g["crew"].columns and c not in ("p0", "mode_frac")]].round(3))

    with tab_export:
        st.subheader("Reproducibility")
        extras = {"preset": state["preset"], "provenance_tags": tags, "ui_values": state["values"]}
        st.write(f"Engine {E2.ENGINE_VERSION} - dashboard {DL.DASHBOARD_VERSION} - seed {state['values']['seed']} - "
                 f"inputs hash {res.manifest['inputs_sha256'][:12]}...")
        st.download_button("Download run manifest (JSON)", DL.export_manifest(res, extras), file_name="aurum_run_manifest.json",
                           mime="application/json")
        st.download_button("Download first 5,000 cycles (CSV schema 2)", DL.export_csv(res, 5_000),
                           file_name="aurum_cycles_5000.csv", mime="text/csv")
        st.download_button("Download first 50,000 cycles (CSV schema 2)", DL.export_csv(res, 50_000),
                           file_name="aurum_cycles_50000.csv", mime="text/csv")
        st.download_button("Download legacy CSV schema 1 (purity_q means valuation multiplier)", DL.export_csv(res, 5_000, schema_version=1),
                           file_name="aurum_cycles_legacy_schema1.csv", mime="text/csv")
        with st.expander("Manifest preview"):
            st.code(json.dumps({k: res.manifest[k] for k in ("engine_version", "engine_sha256", "inputs_sha256", "run_config")},
                               indent=2), language="json")

    with tab_about:
        st.markdown(DL.ABOUT_MD)


# ------------------------------------------------------------------------------------------------
# Legacy V1.0
# ------------------------------------------------------------------------------------------------
def legacy_form() -> tuple:
    """Draw the sidebar form for the legacy V1.0 per-person inputs. Returns (values dict, submitted flag)."""
    d = DL.LEGACY_DEFAULTS
    v = {}
    with st.sidebar.form(key="form_legacy"):
        st.markdown("**V1.0 inputs (per person, as in the original dashboard)**")
        v["H"] = st.number_input("Total hours worked (H)", key="legacy__H", min_value=1, max_value=16, value=int(d["H"]), step=1)
        v["T"] = st.number_input("Tribute split ratio (T)", key="legacy__T", min_value=0.0, max_value=1.0, value=float(d["T"]),
                                 step=0.05, format="%.4f")
        v["Cfamily"] = st.number_input("Hourly family need (PHP/hr)", key="legacy__Cfamily", min_value=0.0,
                                       value=float(d["Cfamily"]), step=1.0, format="%.2f")
        v["W"] = st.number_input("Opportunity wage floor (PHP/hr)", key="legacy__W", min_value=0.0, value=float(d["W"]),
                                 step=1.0, format="%.2f")
        v["Cfixed"] = st.number_input("Fixed costs (PHP)", key="legacy__Cfixed", min_value=0.0, value=float(d["Cfixed"]),
                                      step=10.0, format="%.2f")
        v["delta"] = st.number_input("Depreciation factor", key="legacy__delta", min_value=1.0, max_value=1.5,
                                     value=float(d["delta"]), step=0.05, format="%.2f")
        v["Cfuel"] = st.number_input("Fuel costs (PHP)", key="legacy__Cfuel", min_value=0.0, value=float(d["Cfuel"]),
                                     step=10.0, format="%.2f")
        v["alpha"] = st.number_input("Logistics premium", key="legacy__alpha", min_value=0.0, max_value=0.2,
                                     value=float(d["alpha"]), step=0.01, format="%.2f")
        v["Gmin"] = st.number_input("Typical minimum yield (g)", key="legacy__Gmin", min_value=0.001,
                                    value=float(d["Gmin"]), step=0.01, format="%.3f")
        v["Gsub"] = st.number_input("Typical maximum yield (g)", key="legacy__Gsub", min_value=0.001,
                                    value=float(d["Gsub"]), step=0.05, format="%.3f")
        v["Gjack"] = st.number_input("Jackpot maximum yield (g)", key="legacy__Gjack", min_value=0.001,
                                     value=float(d["Gjack"]), step=0.1, format="%.3f")
        v["eta"] = st.number_input("Purity coefficient (eta)", key="legacy__eta", min_value=0.5, max_value=1.0,
                                   value=float(d["eta"]), step=0.01, format="%.3f")
        v["Pmin"] = st.number_input("Minimum price (PHP/g)", key="legacy__Pmin", min_value=1.0, value=float(d["Pmin"]),
                                    step=10.0, format="%.2f")
        v["Pmax"] = st.number_input("Maximum price (PHP/g)", key="legacy__Pmax", min_value=1.0, value=float(d["Pmax"]),
                                    step=10.0, format="%.2f")
        choices = [10_000, 100_000, 1_000_000]
        v["N"] = st.selectbox("Simulated shifts (N)", choices, index=choices.index(int(d["N"])), key="legacy__N")
        submitted = st.form_submit_button("Run legacy V1.0")
    return v, submitted


def render_legacy(v: dict, out: dict) -> None:
    """Render the legacy V1.0 results (deterministic reference, simulation, and known flaws)."""
    st.warning("Legacy V1.0, shown for comparison only. It has known flaws (listed below) and its outputs should not be "
               "reported as results.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Deterministic reference")
        st.metric("Net hourly return at mean inputs (PHP/hr)", f"{out['E_det']:.2f}")
        st.metric("V1.0 'target break-even yield' (g, includes family cost)", f"{out['Y_det']:.4f}")
    with c2:
        st.subheader("Simulation")
        st.metric("Share of shifts with return at or above the wage floor (V1.0 label: 'Probability of Real Economic "
                  "Sustainability')", f"{out['success_rate']:.2f}%")
        st.metric("Mean of V1.0 'required yield' (g)", f"{out['mean_stochastic_cyt']:.4f}")
    _show_fig(DL.fig_return_density(out["hourly_efficiency_array"], v["W"], "V1.0 hourly return (per person)"))
    st.caption(f"Jackpot frequency used by V1.0: {out['jackpot_prob']:.2f}% (heuristic clamp, not data). N = {out['iterations']:,}.")
    with st.expander("Known V1.0 flaws", expanded=True):
        st.markdown(DL.LEGACY_FLAWS_MD)


# ------------------------------------------------------------------------------------------------
def main() -> None:
    """Streamlit app entry point: render the sidebar, dispatch to V2.3 or legacy V1.0, and show results."""
    st.set_page_config(page_title="Project AURUM", layout="wide")
    st.title("Project AURUM: Decision-Support Model")
    st.markdown("### Zero-aware stochastic risk analysis for small-scale gold mining")
    st.caption("Palma, Postrero, Salaritan - Pantukan National High School - engine "
               f"{E2.ENGINE_VERSION} - dashboard {DL.DASHBOARD_VERSION}")

    st.sidebar.markdown("## Inputs")
    version = st.sidebar.radio("Model version", VERSIONS, index=0, key="version")

    if version == VERSIONS[1]:
        v, submitted = legacy_form()
        if submitted:
            errs = DL.legacy_errors(v)
            if errs:
                for e in errs:
                    st.error(e)
                st.stop()
            with st.spinner("Running legacy V1.0..."):
                st.session_state["legacy"] = {"v": v, "out": DL.run_legacy(v)}
        if st.session_state.get("legacy"):
            render_legacy(st.session_state["legacy"]["v"], st.session_state["legacy"]["out"])
        else:
            st.info("Set the V1.0 inputs and press 'Run legacy V1.0'.")
        return

    preset = st.sidebar.selectbox("Scenario preset", DL.PRESETS, index=0, key="preset")
    st.sidebar.caption("🟠 assumed  🔵 external  ⚪ synthetic  🟣 changed by you")
    values, submitted = v2_form(preset)

    if submitted:
        try:
            with st.spinner("Running Monte Carlo simulation..."):
                inp, res = DL.run_v2(values, preset)
        except E2.InputError as exc:
            st.session_state.pop("v2", None)
            st.error("These inputs are not valid:\n\n" + str(exc))
            st.stop()
        st.session_state["v2"] = {"inp": inp, "res": res, "values": values, "preset": preset,
                                  "tags": DL.effective_tags(values, preset),
                                  "run_id": res.manifest["inputs_sha256"] + str(values["seed"])}

    state = st.session_state.get("v2")
    if state and state["preset"] == preset:
        render_v2(state)
    elif state:
        st.info("The preset changed. Press 'Run Project AURUM model' to run it.")
    else:
        st.info("Choose a preset, review the inputs (🟠 = assumed), and press 'Run Project AURUM model'.")


if __name__ == "__main__":
    main()
