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

VERSIONS = ["V2.0 (current)", "V1.0 (legacy, frozen - comparison only)"]
N_MAX_CHOICES = [50_000, 100_000, 200_000, 500_000]


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text)


def _show_fig(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------
# V2.0 sidebar form
# ------------------------------------------------------------------------------------------------
def v2_form(preset: str) -> tuple:
    """Draw the input form. Returns (values dict, submitted flag)."""
    ps = _slug(preset)
    defaults = DL.preset_values(preset)

    def label(name: str, text: str) -> str:
        cur = st.session_state.get(f"{ps}__{name}", defaults[name])
        return f"{text}  {DL.TAG_LABEL[DL.tag_for(name, cur, preset)]}"

    def num(name, text, help_text=None, **kw):
        return st.number_input(label(name, text), key=f"{ps}__{name}", help=help_text, **kw)

    def sld(name, text, help_text=None, **kw):
        return st.slider(label(name, text), key=f"{ps}__{name}", help=help_text, **kw)

    def rad(name, text, options, help_text=None):
        return st.radio(label(name, text), options, index=options.index(defaults[name]), key=f"{ps}__{name}",
                        help=help_text)

    d = defaults
    v = {}
    with st.sidebar.form(key=f"form_{ps}"):
        st.markdown("**1. Accounting unit and labour** (one production cycle, group level)")
        v["n_c"] = num("n_c", "Active crew size n_c", "Not collected in the pilot. Drives person-hours and the result strongly.",
                       min_value=1, max_value=100, value=int(d["n_c"]), step=1)
        v["D_c"] = num("D_c", "Shifts per cycle D_c", "The pilot data are per shift, so a cycle = 1 shift.",
                       min_value=1, max_value=60, value=int(d["D_c"]), step=1)
        v["h"] = num("h", "Hours per worker per shift", None, min_value=0.5, max_value=24.0,
                     value=float(d["h"]), step=0.25, format="%.2f")

        st.markdown("**2. Stakeholder accounting**")
        v["s_m"] = num("s_m", "Miner-group revenue share s_m", "Group share of the revenue base (1 minus the non-miner share).",
                       min_value=0.01, max_value=1.0, value=float(d["s_m"]), step=0.01, format="%.4f")
        v["tribute_base"] = rad("tribute_base", "Share applies to", ["gross", "net"],
                                "gross: miners pay their cost share out of their revenue share. "
                                "net: costs are deducted first, then the share is applied.")

        st.markdown("**3. Gold yield** (group total, grams of dore per cycle)")
        v["ya"] = num("ya", "Typical minimum yield a (g)", "Pilot mean of respondents' typical minimum (provisional).",
                      min_value=0.0, max_value=1000.0, value=float(d["ya"]), step=0.05, format="%.3f")
        v["yb"] = num("yb", "Typical maximum yield b (g)", "Pilot mean of respondents' typical maximum (provisional).",
                      min_value=0.0, max_value=1000.0, value=float(d["yb"]), step=0.05, format="%.3f")
        v["mode_pct"] = sld("mode_pct", "Most-likely yield (mode) as % of the a-to-b range",
                            "No 'usual' value was collected, so the mode is an assumption.",
                            min_value=0.0, max_value=100.0, value=float(d["mode_pct"]), step=1.0)
        v["p0_mode"] = rad("p0_mode", "Zero-yield probability from", ["fixed value", "from counts"],
                           "counts: p0 ~ Beta(1+z, 1+n-z), drawn for every simulated cycle.")
        v["p0"] = sld("p0", "Zero-yield (waste cycle) probability p0", "Assumed: zero-yield counts were not collected.",
                      min_value=0.0, max_value=1.0, value=float(d["p0"]), step=0.01)
        v["z"] = num("z", "Zero-yield cycles observed z", None, min_value=0, max_value=1000, value=int(d["z"]), step=1)
        v["n_obs"] = num("n_obs", "Cycles recorded n", None, min_value=0, max_value=1000, value=int(d["n_obs"]), step=1)
        v["pi_J"] = num("pi_J", "Jackpot probability pi_J (given a productive cycle)", "Assumed. Set 0 for no jackpot component.",
                        min_value=0.0, max_value=1.0, value=float(d["pi_J"]), step=0.005, format="%.4f")
        v["jack_thr"] = num("jack_thr", "Jackpot threshold (g)", "Lowest reported jackpot (3 respondents).",
                            min_value=0.1, max_value=1000.0, value=float(d["jack_thr"]), step=0.5, format="%.2f")
        v["jack_cap"] = num("jack_cap", "Jackpot cap (g)", "Highest reported jackpot / physical capacity.",
                            min_value=0.1, max_value=1000.0, value=float(d["jack_cap"]), step=0.5, format="%.2f")
        v["jack_shape"] = num("jack_shape", "Jackpot tail shape (Pareto)", "Assumed; smaller = heavier tail.",
                              min_value=0.2, max_value=5.0, value=float(d["jack_shape"]), step=0.1, format="%.2f")

        st.markdown("**4. Price and purity** (handled exactly once)")
        v["price_basis"] = rad("price_basis", "Price basis", ["farmgate", "reference"],
                               "farmgate: realized price already reflects grade, so purity is NOT applied. "
                               "reference: pure-gold price, purity applied.")
        v["p_min"] = num("p_min", "Price minimum (PHP/g)", None, min_value=1.0, max_value=1_000_000.0,
                         value=float(d["p_min"]), step=50.0, format="%.2f")
        v["p_mode"] = num("p_mode", "Price most likely (PHP/g)", "Pilot: the most frequently reported price.",
                          min_value=1.0, max_value=1_000_000.0, value=float(d["p_mode"]), step=50.0, format="%.2f")
        v["p_max"] = num("p_max", "Price maximum (PHP/g)", None, min_value=1.0, max_value=1_000_000.0,
                         value=float(d["p_max"]), step=50.0, format="%.2f")
        v["q_min"] = num("q_min", "Purity minimum q (used only on a reference basis)", None, min_value=0.01,
                         max_value=1.0, value=float(d["q_min"]), step=0.01, format="%.3f")
        v["q_mode"] = num("q_mode", "Purity most likely q", None, min_value=0.01, max_value=1.0,
                          value=float(d["q_mode"]), step=0.01, format="%.3f")
        v["q_max"] = num("q_max", "Purity maximum q", None, min_value=0.01, max_value=1.0,
                         value=float(d["q_max"]), step=0.01, format="%.3f")

        st.markdown("**5. Operating costs** (PHP per cycle, group level)")
        v["fixed_cost"] = num("fixed_cost", "Fixed operating cost", "Assumed: costs were not collected in the pilot.",
                              min_value=0.0, max_value=10_000_000.0, value=float(d["fixed_cost"]), step=10.0, format="%.2f")
        v["fixed_share"] = num("fixed_share", "Share paid by miners (fixed cost)", "1 = miners pay, 0 = financier pays.",
                               min_value=0.0, max_value=1.0, value=float(d["fixed_share"]), step=0.05, format="%.2f")
        v["fuel_min"] = num("fuel_min", "Fuel minimum", None, min_value=0.0, max_value=10_000_000.0,
                            value=float(d["fuel_min"]), step=10.0, format="%.2f")
        v["fuel_mode"] = num("fuel_mode", "Fuel most likely", None, min_value=0.0, max_value=10_000_000.0,
                             value=float(d["fuel_mode"]), step=10.0, format="%.2f")
        v["fuel_max"] = num("fuel_max", "Fuel maximum", None, min_value=0.0, max_value=10_000_000.0,
                            value=float(d["fuel_max"]), step=10.0, format="%.2f")
        v["fuel_share"] = num("fuel_share", "Share paid by miners (fuel)", None, min_value=0.0, max_value=1.0,
                              value=float(d["fuel_share"]), step=0.05, format="%.2f")
        v["trans_min"] = num("trans_min", "Transport/hauling minimum", None, min_value=0.0, max_value=10_000_000.0,
                             value=float(d["trans_min"]), step=10.0, format="%.2f")
        v["trans_mode"] = num("trans_mode", "Transport/hauling most likely", None, min_value=0.0,
                              max_value=10_000_000.0, value=float(d["trans_mode"]), step=10.0, format="%.2f")
        v["trans_max"] = num("trans_max", "Transport/hauling maximum", None, min_value=0.0, max_value=10_000_000.0,
                             value=float(d["trans_max"]), step=10.0, format="%.2f")
        v["trans_share"] = num("trans_share", "Share paid by miners (transport)", None, min_value=0.0, max_value=1.0,
                               value=float(d["trans_share"]), step=0.05, format="%.2f")
        v["dependence"] = rad("dependence", "Cost dependence", ["copula", "independent"],
                              "copula: fuel and transport costs move together (Gaussian copula on a Spearman correlation).")
        v["rho"] = num("rho", "Spearman correlation fuel-transport", "Assumed; no data on dependence.",
                       min_value=-0.95, max_value=0.95, value=float(d["rho"]), step=0.05, format="%.2f")

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

        st.markdown("**7. Benchmarks** (normative thresholds)")
        v["W_opp"] = num("W_opp", "Opportunity-wage benchmark (PHP/person-hour)", "Cite the wage order below.",
                         min_value=0.01, max_value=5000.0, value=float(d["W_opp"]), step=1.0, format="%.2f")
        v["W_src"] = st.text_input("Wage benchmark source", value=str(d["W_src"]), key=f"{ps}__W_src")
        v["W_date"] = st.text_input("Wage benchmark effective date", value=str(d["W_date"]), key=f"{ps}__W_date")
        v["hh_need"] = num("hh_need", "Household essential spending (PHP/household/day)", "Pilot mean; household layer only.",
                           min_value=0.0, max_value=1_000_000.0, value=float(d["hh_need"]), step=25.0, format="%.2f")
        v["n_hh"] = num("n_hh", "Households supported (0 = one per crew member)", None, min_value=0, max_value=200,
                        value=int(d["n_hh"]), step=1)
        v["nonmining"] = num("nonmining", "Non-mining income (PHP/household/day)", None, min_value=0.0,
                             max_value=1_000_000.0, value=float(d["nonmining"]), step=25.0, format="%.2f")

        st.markdown("**8. Simulation precision**")
        v["seed"] = st.number_input("Random seed", key=f"{ps}__seed", help="Same seed + same inputs = same results.",
                                    min_value=0, max_value=2_000_000_000, value=int(d["seed"]), step=1)
        v["tol"] = st.number_input("MCSE tolerance for every probability", key=f"{ps}__tol", min_value=0.001,
                                   max_value=0.05, value=float(d["tol"]), step=0.001, format="%.3f")
        v["n_max"] = st.selectbox("Maximum cycles N", N_MAX_CHOICES, index=N_MAX_CHOICES.index(int(d["n_max"])),
                                  key=f"{ps}__n_max", help="The run stops earlier once every probability's MCSE is within tolerance.")
        submitted = st.form_submit_button("Run Project AURUM model")
    return v, submitted


# ------------------------------------------------------------------------------------------------
# V2.0 results
# ------------------------------------------------------------------------------------------------
def render_v2(state: dict) -> None:
    inp, res, tags = state["inp"], state["res"], state["tags"]
    s = res.summary
    assumed = DL.assumed_inputs(tags)
    user = [k for k, t in tags.items() if t == "user"]

    tab_over, tab_dist, tab_conv, tab_scen, tab_export, tab_about = st.tabs(
        ["Overview", "Distributions", "Convergence and sensitivity", "Scenario analysis", "Export and reproducibility",
         "How to read this"])

    with tab_over:
        st.info("Scenario results under the inputs in the sidebar. They are not findings about Pantukan miners.")
        st.markdown(f"**{len(assumed)}** inputs are 🟠 assumed and **{len(user)}** were 🟣 changed by you.")
        with st.expander("Which inputs are assumptions?"):
            st.write(", ".join(assumed) if assumed else "None.")
            st.caption("; ".join(f"{DL.TAG_LABEL[k]}: {DL.TAG_HELP[k]}" for k in DL.TAG_LABEL))
        for level, text in DL.messages(res, tags):
            getattr(st, level)(text)

        st.subheader("Risk probabilities")
        cols = st.columns(4)
        for col, key in zip(cols, E2.PROB_KEYS):
            with col:
                st.metric(E2.LABELS[key], f"{s[key]:.1%}")
                st.caption(f"± {2 * s['MCSE_' + key]:.2%} (2 MCSE)")
        st.markdown(DL.risk_band_text(res))
        st.caption(f"N = {s['N']:,} simulated cycles - convergence {'reached' if s['converged'] else 'NOT reached'} - "
                   f"person-hours per cycle L = {s['L_person_hours']:.1f} - household requirement B_HH = {s['B_HH']:,.0f} PHP")

        left, right = st.columns(2)
        with left:
            st.subheader("Labor return per person-hour")
            st.dataframe(DL.return_table(res))
        with right:
            st.subheader("Yield thresholds (kept separate)")
            st.dataframe(DL.thresholds_table(res))
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
        st.dataframe(res.stage_table.round(4))
        _show_fig(DL.fig_stability(res.stage_table))
        st.subheader("Rank-based sensitivity")
        if res.rank_sensitivity:
            _show_fig(DL.fig_rank_sensitivity(res.rank_sensitivity))
        st.caption("Spearman rank correlation of each simulated input with the labor return. It shows direction and "
                   "strength of association; Sobol variance shares are future work.")

    with tab_scen:
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
                         "P_fail_household_adequacy", "E_median"]
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
        st.download_button("Download first 5,000 cycles (CSV)", DL.export_csv(res, 5_000),
                           file_name="aurum_cycles_5000.csv", mime="text/csv")
        st.download_button("Download first 50,000 cycles (CSV)", DL.export_csv(res, 50_000),
                           file_name="aurum_cycles_50000.csv", mime="text/csv")
        with st.expander("Manifest preview"):
            st.code(json.dumps({k: res.manifest[k] for k in ("engine_version", "engine_sha256", "inputs_sha256", "run_config")},
                               indent=2), language="json")

    with tab_about:
        st.markdown(DL.ABOUT_MD)


# ------------------------------------------------------------------------------------------------
# Legacy V1.0
# ------------------------------------------------------------------------------------------------
def legacy_form() -> tuple:
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
    st.sidebar.caption("🟢 pilot  🟠 assumed  🔵 external  ⚪ synthetic  🟣 changed by you")
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
