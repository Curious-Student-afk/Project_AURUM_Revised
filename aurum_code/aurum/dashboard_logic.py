"""
dashboard_logic.py -- everything the dashboard does EXCEPT drawing widgets.

Kept free of Streamlit so it can be unit-tested offline. app.py is a thin layer on top.

Design rules taken from the expert report:
  * every input carries a provenance tag (assumed / external / synthetic / user-entered)
  * three thresholds (break-even, opportunity-wage, household) are reported separately
  * risk outputs use the recommended labels, with MCSE, quantiles and expected shortfall
  * no "poverty trap" wording and no directive advice (no "exit" / "greenlight" messages)
  * synthetic demonstrations and externally sourced inputs are labeled explicitly
"""
from __future__ import annotations

import functools
import json
import math
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import engine_v2 as E2  # noqa: E402
import legacy_v1  # noqa: E402
import normalization as N  # noqa: E402
from scenarios import synthetic_profiles, default_timing, legacy_synthetic_values  # noqa: E402

DASHBOARD_VERSION = "2.3.0-draft"

# --------------------------------------------------------------------------------------
# Provenance tags
# --------------------------------------------------------------------------------------
TAG_LABEL = {"assumed": "🟠 assumed", "external": "🔵 external",
             "synthetic": "⚪ synthetic", "user": "🟣 user-entered"}
TAG_HELP = {"assumed": "A stated assumption to sensitivity-test.",
            "external": "From a cited outside source.", "synthetic": "Invented for demonstration only.",
            "user": "User-entered; its source is not verified."}
PRESETS = ["Synthetic: Baseline", "Synthetic: High-Cost", "Synthetic: Low-Purity"]
_PRESET_TO_PROFILE = dict(zip(PRESETS, ("Baseline", "High-Cost", "Low-Purity")))
COST_FIELDS = {"fuel": "fuel", "milling": "milling", "explosives": "explosives",
               "trans": "transport", "food": "food"}
BASIS_LABELS = {"per_batch": "PHP/batch", "per_shift": "PHP/shift",
                "per_operating_hour": "PHP/operating hour", "per_working_day": "PHP/working day",
                "per_calendar_day": "PHP/calendar day"}


def correlation_keys():
    """List (i, j, widget key) triples for every pairwise cost-rate correlation above the diagonal."""
    names = list(COST_FIELDS)
    return [(i, j, f"rho_{names[i]}_{names[j]}") for i in range(len(names)) for j in range(i + 1, len(names))]


def param_tags(preset: str) -> Dict[str, str]:
    """Default provenance tag ('synthetic') for every preset parameter except run-control fields."""
    return {k: "synthetic" for k in preset_values(preset)
            if k not in ("seed", "tol", "n_max", "W_src", "W_date")}


def values_from_inputs(inp: E2.Inputs) -> Dict[str, object]:
    """Map supported dashboard expense categories and one equipment item to widgets."""
    y, j = inp.yield_ord, inp.jackpot or E2.Jackpot(30, 60, 1.5)
    price = inp.price if isinstance(inp.price, E2.PERT) else E2.PERT(inp.price, inp.price, inp.price)
    pur = inp.purity if isinstance(inp.purity, E2.PERT) else E2.PERT(inp.purity, inp.purity, inp.purity)
    if inp.price_basis == "farmgate":
        assay = inp.physical_purity
        pur = assay if isinstance(assay, E2.PERT) else E2.PERT(assay or 1., assay or 1., assay or 1.)
    v = {
        "timing_mode": "batch" if inp.batch_timing else "fixed",
        "calendar_days": 1. if inp.batch_timing else inp.calendar_days(),
        "n_c": inp.n_c, "D_c": inp.D_c, "h": float(inp.hours_per_worker_shift),
        "s_m": float(inp.s_m), "share_preset": "Custom", "tribute_base": inp.tribute_base,
        "ya": float(y.a), "yb": float(y.b),
        "ym": float(y.m),
        "p0_mode": "fixed value" if inp.zero.p0 is not None else "from counts",
        "p0": float(inp.zero.p0 if inp.zero.p0 is not None else .1),
        "z": inp.zero.z if inp.zero.z is not None else 0,
        "n_obs": inp.zero.n if inp.zero.n is not None else 10,
        "pi_J": inp.pi_J, "jack_thr": float(j.threshold), "jack_cap": float(j.cap), "jack_shape": float(j.shape),
        "price_basis": inp.price_basis, "p_min": float(price.a), "p_mode": float(price.m), "p_max": float(price.b),
        "assay_mode": "unknown" if inp.price_basis == "farmgate" and inp.physical_purity is None else "PERT fraction",
        "assay_value": float(pur.m), "assay_unit": "fraction",
        "price_input": "direct", "sale_proceeds": 1., "sale_mass": 1.,
        "fuel_input": "direct", "fuel_liters": 0., "fuel_unit_price": 1.,
        "measurement_source": str(inp.metadata.get("source", "")),
        "measurement_date": str(inp.metadata.get("date", "")),
        "fixed_components": "", "households_mode": "one per worker" if inp.n_households is None else "explicit",
        "return_abs_tol": 1., "return_rel_tol": .01,
        "q_min": float(pur.a), "q_mode": float(pur.m), "q_max": float(pur.b),
        "fixed_cost": 0., "fixed_share": 1., "dependence": inp.dependence,
        "reserve_mode": inp.reserve_mode, "reserve_location": inp.reserve_location or "choose",
        "reserve_priority": inp.reserve_priority or "choose", "reserve_custom": inp.reserve_custom_per_cycle,
        "eq_mode": "none", "eq_acq": 0., "eq_salv": 0., "eq_life": 1000., "eq_util": 1., "eq_rent": 0., "eq_share": 1.,
        "W_opp": inp.W_opp, "W_src": inp.W_opp_source, "W_date": inp.W_opp_date,
        "hh_need": inp.hh_need_per_household_day, "n_hh": inp.n_c if inp.n_households is None else inp.n_households,
        "nonmining": inp.nonmining_income_per_household_day, "seed": 42, "tol": .005, "n_max": 200_000,
    }
    t = inp.batch_timing or default_timing()
    for prefix, pert in (("work", t.work_days), ("idle", t.idle_days)):
        for suffix, value in zip(("min", "mode", "max"), (pert.a, pert.m, pert.b)):
            v[f"{prefix}_{suffix}"] = int(value) if prefix == "work" else float(value)
    for d in (1, 2, 3):
        v[f"days_{d}"] = t.shift_counts[d - 1]
        for suffix, value in zip(("min", "mode", "max"), pert_values(t.hours_by_shifts[d])):
            v[f"h{d}_{suffix}"] = float(value)
    costs = {c.name: c for c in inp.costs}
    resolved_names = {}
    for prefix, name in COST_FIELDS.items():
        c = costs.get(name) or (costs.get("transport_hauling") if prefix == "trans" else None)
        c = c or E2.CostItem(name, fixed=0.)
        resolved_names[name] = c.name
        pert = c.pert or E2.PERT(c.fixed, c.fixed, c.fixed)
        for suffix, value in zip(("min", "mode", "max"), pert_values(pert)):
            v[f"{prefix}_{suffix}"] = float(value)
        v[f"{prefix}_share"], v[f"{prefix}_basis"] = float(c.miner_share), c.basis
    if "fixed_operating" in costs:
        c = costs["fixed_operating"]
        v.update(fixed_cost=float(c.fixed), fixed_share=float(c.miner_share),
                 fixed_components=", ".join(c.components or [c.name]))
    stochastic_names = [c.name for c in inp.stochastic_cost_items()]
    for i, j, key in correlation_keys():
        names = list(COST_FIELDS.values())
        left, right = resolved_names[names[i]], resolved_names[names[j]]
        v[key] = (float(inp.rho_spearman[stochastic_names.index(left)][stochastic_names.index(right)])
                  if inp.rho_spearman is not None and left in stochastic_names and right in stochastic_names else 0.)
    if inp.equipment:
        eq = inp.equipment[0]
        v.update(eq_mode=eq.mode, eq_acq=eq.acquisition, eq_salv=eq.salvage, eq_life=eq.life_cycles,
                 eq_util=eq.utilization, eq_rent=eq.rental_rate_per_shift, eq_share=eq.miner_share)
    return v


def pert_values(pert):
    """Return a PERT's (min, mode, max) tuple."""
    return pert.a, pert.m, pert.b


@functools.lru_cache(maxsize=None)
def preset_values(preset: str) -> Dict[str, object]:
    """Default UI values for a preset. Cached: treat the returned dict as READ-ONLY (copy it to edit)."""
    return values_from_inputs(synthetic_profiles()[_PRESET_TO_PROFILE[preset]])


def tag_for(name: str, value, preset: str) -> str:
    """Provenance tag shown next to an input: the preset's tag if unchanged, otherwise 'user'."""
    default = preset_values(preset).get(name)
    tags = param_tags(preset)
    if name not in tags:
        return "assumed"
    same = (math.isclose(float(value), float(default), rel_tol=1e-9, abs_tol=1e-12)
            if isinstance(value, (int, float)) and isinstance(default, (int, float))
            and not isinstance(value, bool) else value == default)
    return tags[name] if same else "user"


def effective_tags(values: Dict[str, object], preset: str) -> Dict[str, str]:
    """Provenance tags for the current form `values`, resolving the share preset first."""
    effective = dict(values)
    if values.get("share_preset") in ("50/50", "60/40", "65/35"):
        effective["s_m"] = {"50/50": .5, "60/40": .6, "65/35": .65}[values["share_preset"]]
    return {k: tag_for(k, effective[k], preset) for k in param_tags(preset) if k in effective}


def build_inputs(v: Dict[str, object], preset: Optional[str] = None) -> E2.Inputs:
    """Flat UI values -> engine Inputs (validation happens in engine_v2.validate / run)."""
    v = dict(v)
    metadata = {"source": v.get("measurement_source", ""), "date": v.get("measurement_date", "")}
    if metadata["date"]:
        from datetime import date
        try:
            parsed_date = date.fromisoformat(metadata["date"])
            if parsed_date.isoformat() != metadata["date"]:
                raise ValueError("Measurement date is not in canonical format.")
        except (TypeError, ValueError) as exc:
            raise E2.InputError("Measurement date must be YYYY-MM-DD.") from exc
    if v.get("price_input", "direct") not in ("direct", "sale"):
        raise E2.InputError("Choose direct price or sale conversion.")
    if v.get("price_input") == "sale":
        price = N.realized_price(v["sale_proceeds"], v["sale_mass"], v["price_basis"])
        v.update(p_min=price, p_mode=price, p_max=price)
        metadata["sale"] = {"proceeds_php": v["sale_proceeds"], "mass_g": v["sale_mass"]}
    if v.get("fuel_input", "direct") not in ("direct", "liters × PHP/liter"):
        raise E2.InputError("Choose direct fuel rates or liters × PHP/liter.")
    if v.get("fuel_input") == "liters × PHP/liter":
        fuel = N.fuel_cost(v["fuel_liters"], v["fuel_unit_price"], v["fuel_basis"])
        v.update(fuel_min=fuel, fuel_mode=fuel, fuel_max=fuel)
        metadata["fuel"] = {"liters": v["fuel_liters"], "php_per_liter": v["fuel_unit_price"], "basis": v["fuel_basis"]}
    assay_mode = v.get("assay_mode", "unknown")
    if assay_mode not in ("unknown", "PERT fraction", "measured value"):
        raise E2.InputError("Choose unknown, PERT fraction or measured value for assay.")
    if v["price_basis"] == "reference" and assay_mode == "unknown":
        raise E2.InputError("Reference pricing requires physical purity.")
    assay = None
    if assay_mode == "measured value":
        assay = N.purity_fraction(v["assay_value"], v["assay_unit"])
        metadata["assay"] = {"value": v["assay_value"], "unit": v["assay_unit"]}
    numeric = ["s_m", "ya", "yb", "ym", "pi_J", "p_min", "p_mode", "p_max",
               "fixed_cost", "fixed_share", "W_opp", "hh_need"]
    numeric += ["nonmining"]
    integers = [("n_c", 1), ("n_hh", 0)]
    if v.get("households_mode", "one per worker") not in ("one per worker", "explicit"):
        raise E2.InputError("Choose default or explicit household count.")
    batch = v["timing_mode"] == "batch"
    if v["timing_mode"] not in ("batch", "fixed"):
        raise E2.InputError("Choose batch or fixed timing.")
    if batch:
        integers += [(f"work_{s}", 1) for s in ("min", "mode", "max")]
        integers += [(f"days_{d}", 0) for d in (1, 2, 3)]
        numeric += [f"idle_{s}" for s in ("min", "mode", "max")]
        numeric += [f"h{d}_{s}" for d in (1, 2, 3) for s in ("min", "mode", "max")]
    else:
        integers += [("D_c", 1)]
        numeric += ["h", "calendar_days", "nonmining"]
    for prefix in COST_FIELDS:
        numeric += [f"{prefix}_{s}" for s in ("min", "mode", "max", "share")]
    if v["p0_mode"] == "fixed value":
        numeric.append("p0")
    elif v["p0_mode"] == "from counts":
        integers += [("z", 0), ("n_obs", 0)]
    else:
        raise E2.InputError("Choose a fixed zero probability or counts.")
    if E2.finite_number(v["pi_J"]) and v["pi_J"] > 0:
        numeric += ["jack_thr", "jack_cap", "jack_shape"]
    if assay_mode == "PERT fraction":
        numeric += ["q_min", "q_mode", "q_max"]
    if v["dependence"] == "copula":
        numeric += [key for _, _, key in correlation_keys()]
    if v["eq_mode"] == "purchased":
        numeric += ["eq_acq", "eq_salv", "eq_life", "eq_util", "eq_share"]
    elif v["eq_mode"] == "rented":
        numeric += ["eq_rent", "eq_share"]
    if v.get("reserve_mode") == "custom":
        numeric.append("reserve_custom")
    errs = []
    for name in numeric:
        E2.require_finite(name, v[name], errs)
    for name, minimum in integers:
        E2.require_integer(name, v[name], minimum, errs)
    if errs:
        raise E2.InputError("\n".join(errs))
    def pert(prefix):
        """Build a PERT from the `{prefix}_min/mode/max` widget values."""
        return E2.PERT(*(v[f"{prefix}_{s}"] for s in ("min", "mode", "max")))
    if assay_mode == "PERT fraction":
        assay = pert("q")
    timing = (E2.BatchTiming(pert("work"), pert("idle"), tuple(v[f"days_{d}"] for d in (1, 2, 3)),
                            {d: pert(f"h{d}") for d in (1, 2, 3)}) if batch else None)
    zero = E2.ZeroYield(p0=v["p0"]) if v["p0_mode"] == "fixed value" else E2.ZeroYield(z=v["z"], n=v["n_obs"])
    y = E2.PERT(v["ya"], v["ym"], v["yb"])
    jack = E2.Jackpot(v["jack_thr"], v["jack_cap"], v["jack_shape"]) if v["pi_J"] > 0 else None
    components = [x.strip() for x in v.get("fixed_components", "").split(",") if x.strip()]
    if v["fixed_cost"] > 0 and not components:
        raise E2.InputError("Declare the component identifiers included in fixed overhead.")
    costs = [E2.CostItem("fixed_operating", v["fixed_share"], fixed=v["fixed_cost"], components=components or None)]
    costs += [E2.CostItem(name, v[f"{prefix}_share"], pert=pert(prefix), basis=v[f"{prefix}_basis"])
              for prefix, name in COST_FIELDS.items()]
    rho = np.eye(len(COST_FIELDS))
    if v["dependence"] == "copula":
        for i, j, key in correlation_keys():
            rho[i, j] = rho[j, i] = v[key]
    equipment = []
    if v["eq_mode"] != "none":
        equipment = [E2.EquipmentItem("main_equipment", v["eq_mode"], miner_share=v["eq_share"],
                     acquisition=v["eq_acq"], salvage=v["eq_salv"], life_cycles=v["eq_life"],
                     utilization=v["eq_util"], rental_rate_per_shift=v["eq_rent"])]
    share = {"50/50": .5, "60/40": .6, "65/35": .65}.get(v.get("share_preset"), v["s_m"])
    return E2.Inputs(
        n_c=v["n_c"], batch_timing=timing, D_c=1 if batch else v["D_c"],
        hours_per_worker_shift=8. if batch else v["h"], calendar_days_per_cycle=None if batch else v["calendar_days"],
        s_m=share, tribute_base=v["tribute_base"], yield_ord=y, zero=zero, pi_J=v["pi_J"], jackpot=jack,
        price_basis=v["price_basis"], price=pert("p"), purity=assay if v["price_basis"] == "reference" else 1.,
        physical_purity=assay if v["price_basis"] == "farmgate" else None, metadata=metadata,
        costs=costs, equipment=equipment, dependence=v["dependence"], rho_spearman=rho.tolist(),
        reserve_mode=v.get("reserve_mode", "none"),
        reserve_location=None if v.get("reserve_location", "choose") == "choose" else v["reserve_location"],
        reserve_priority=None if v.get("reserve_priority", "choose") == "choose" else v["reserve_priority"],
        reserve_custom_per_cycle=v.get("reserve_custom", 0.),
        W_opp=v["W_opp"], W_opp_source=v.get("W_src", ""), W_opp_date=v.get("W_date", ""),
        hh_need_per_household_day=v["hh_need"], n_households=None if v.get("households_mode", "one per worker") == "one per worker" else v["n_hh"],
        nonmining_income_per_household_day=v["nonmining"],
        provenance=effective_tags(v, preset) if preset else {})


def run_v2(v: Dict[str, object], preset: Optional[str] = None) -> Tuple[E2.Inputs, E2.Result]:
    """Build engine Inputs from UI values and run the V2 simulation."""
    inp = build_inputs(v, preset)
    res = E2.run(inp, seed=v["seed"], tol=v["tol"], n_max=v["n_max"],
                 return_abs_tol=v.get("return_abs_tol", 1.), return_rel_tol=v.get("return_rel_tol", .01))
    return inp, res


# --------------------------------------------------------------------------------------
# Text: interpretation and messages (neutral, no directive advice)
# --------------------------------------------------------------------------------------
def interpretation_text(res: E2.Result, inp: E2.Inputs, n_assumed: int) -> str:
    """Build the plain-language interpretation paragraph for the overview tab."""
    s = res.summary
    p = s["P_labor_return_below_wage"]
    lines = [
        f"In {s['N']:,} simulated production cycles, the labor return per person-hour was below the "
        f"{inp.W_opp:g} PHP/hour wage benchmark in {p:.1%} of cycles (±{2 * s['MCSE_P_labor_return_below_wage']:.1%}, "
        f"2 MCSE). The median return was {s['E_median']:.1f} PHP/hour (5th percentile {s['E_P5']:.1f}, "
        f"95th percentile {s['E_P95']:.1f}).",
        f"The miner group had a negative operating return in {s['P_negative_operating_return']:.1%} of cycles.",
    ]
    if s["E_mean"] > s["E_median"] * 1.1:
        lines.append(f"The mean ({s['E_mean']:.1f}) is above the median ({s['E_median']:.1f}) because rare high-yield "
                     "cycles pull the average up; the median and percentiles describe a typical cycle better.")
    lines.append("Profit and hourly returns include equipment depreciation. Household cash probabilities "
                 "exclude depreciation and separately show the effect of funded replacement reserves.")
    if inp.reserve_mode != "none":
        lines.append(f"Reserve policy: {inp.reserve_location.replace('_', ' ')}, "
                     f"{inp.reserve_priority.replace('_', ' ')}. Planned and funded amounts refer to the "
                     f"{'common' if inp.reserve_location == 'before_sharing' else 'miner'} fund. "
                     "This is a single-cycle allocation, with no accumulated balance or borrowing.")
    lines.append(f"These figures describe the simulated scenario defined by the inputs, **{n_assumed} of which are "
                 "assumptions**. They are not predictions for any real mining group and do not establish a poverty trap, "
                 "which would require a dynamic (multi-period) model.")
    return "\n\n".join(lines)


def messages(res: E2.Result, tags: Dict[str, str]) -> List[Tuple[str, str]]:
    """Build (streamlit level, text) message pairs for warnings and convergence/benchmark notices."""
    out = [("warning", w) for w in res.warnings]
    if not res.summary["converged"]:
        out.append(("warning", "Monte Carlo convergence was not reached; do not quote these probabilities without "
                    "stating the MCSE shown."))
    if tags.get("W_opp") and tags["W_opp"] != "external":
        out.append(("info", "The wage benchmark is a normative threshold. Cite its source and effective date "
                    "(wage order) before reporting."))
    return out


def risk_band_text(res: E2.Result) -> str:
    """One-line illustrative risk-band summary against the wage benchmark."""
    band = res.summary["risk_band_vs_wage"]
    meet = 1.0 - res.summary["P_labor_return_below_wage"]
    return (f"Illustrative band: **{band}** (benchmark met in {meet:.1%} of cycles; cut-offs 67% / 34% are the expert "
            "report's example and are not validated).")


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------
THRESHOLD_ROWS = [
    ("Operating break-even yield", "Y_BE", "Yield at which the miner group's profit is zero"),
    ("Opportunity-wage yield", "Y_W", "Yield at which the return equals the wage benchmark per person-hour"),
    ("Household economic-adequacy yield threshold", "Y_HH", "Yield at which economic profit, including depreciation, meets household needs; not a cash threshold"),
]


def thresholds_table(res: E2.Result) -> pd.DataFrame:
    """Build the table of the three yield thresholds with medians and 5th-95th percentiles."""
    rows = []
    for name, key, meaning in THRESHOLD_ROWS:
        a = res.arrays[key]
        p5, p50, p95 = np.percentile(a, [5, 50, 95])
        rows.append({"Threshold": name, "Median (g/cycle)": round(float(p50), 4),
                     "5th–95th percentile (g)": f"{p5:.4f} – {p95:.4f}", "Meaning": meaning})
    return pd.DataFrame(rows)


def risk_table(res: E2.Result) -> pd.DataFrame:
    """Build the table of risk probabilities with their labels and ± 2 MCSE bounds."""
    s = res.summary
    return pd.DataFrame([{"Output": E2.LABELS[k], "Probability": f"{s[k]:.2%}",
                          "± 2 MCSE": f"{2 * s['MCSE_' + k]:.2%}"} for k in E2.PROB_KEYS])


def return_table(res: E2.Result) -> pd.DataFrame:
    """Build the table of economic-return-per-person-hour summary statistics."""
    s = res.summary
    return pd.DataFrame([
        {"Statistic (PHP per person-hour)": "Median", "Value": round(s["E_median"], 2)},
        {"Statistic (PHP per person-hour)": "Mean", "Value": round(s["E_mean"], 2)},
        {"Statistic (PHP per person-hour)": "5th percentile", "Value": round(s["E_P5"], 2)},
        {"Statistic (PHP per person-hour)": "95th percentile", "Value": round(s["E_P95"], 2)},
        {"Statistic (PHP per person-hour)": "Expected shortfall (mean of worst 5%)",
         "Value": round(s["E_expected_shortfall_5pct"], 2)},
    ])


def cash_table(res: E2.Result) -> pd.DataFrame:
    """Build the per-cycle cash and replacement-reserve summary table."""
    labels = {
        "cash_before_reserve": "Miner cash before reserve",
        "depreciation_miner": "Miner depreciation (economic cost only)",
        "reserve_planned": "Reserve planned (selected fund)",
        "reserve_funded": "Reserve funded (selected fund)",
        "reserve_shortfall": "Reserve shortfall (selected fund)",
        "reserve_miner_contribution": "Miner cash allocated to reserve",
        "cash_after_reserve": "Miner cash after reserve",
    }
    return pd.DataFrame([{"Measure (PHP/cycle)": label,
                          "Mean": res.summary[f"{key}_mean"], "Median": res.summary[f"{key}_median"]}
                         for key, label in labels.items()])


def timing_table(res: E2.Result) -> pd.DataFrame:
    """Build the table of batch timing and household-requirement means/medians."""
    labels = {"working_days": "Active working days", "idle_days": "Idle days",
              "calendar_days_per_cycle": "Calendar days", "total_shifts": "Total shifts",
              "operating_hours": "Operating hours", "L_person_hours": "Labor (person-hours)",
              "B_HH": "Household consumption requirement (PHP)"}
    return pd.DataFrame([{"Measure": label, "Mean": res.summary[f"{key}_mean"],
                          "Median": res.summary[f"{key}_median"]}
                         for key, label in labels.items() if f"{key}_mean" in res.summary])


def assumed_inputs(tags: Dict[str, str]) -> List[str]:
    """Sorted names of every input tagged 'assumed'."""
    return sorted(k for k, t in tags.items() if t == "assumed")


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def fig_return_density(E: np.ndarray, W: float, title: str = "Labor return per person-hour",
                       upper_pct: float = 97.0) -> plt.Figure:
    """Density of E. The x-axis is trimmed at `upper_pct` so a rare jackpot tail does not squash the main body."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    sample = E[: min(len(E), 40_000)]
    lo, hi = np.percentile(sample, [0.5, upper_pct])
    if hi - lo < 1e-9:
        hi = lo + 1.0
    x = np.linspace(lo, hi, 600)
    try:
        d = stats.gaussian_kde(sample)(x)
    except Exception:                                    # degenerate sample
        d = np.zeros_like(x)
    ax.plot(x, d, color="#212529", lw=1.8)
    ax.fill_between(x, 0, d, where=x < W, color="#c0392b", alpha=0.45, label="Below wage benchmark")
    ax.fill_between(x, 0, d, where=x >= W, color="#2e7d32", alpha=0.45, label="At or above benchmark")
    ax.axvline(W, color="#7b1e1e", ls="--", lw=1.6, label=f"Wage benchmark ({W:g})")
    ax.axvline(float(np.median(E)), color="#0d6efd", ls="-", lw=1.4, label=f"Median ({np.median(E):.1f})")
    ax.axvline(float(np.mean(E)), color="#6f42c1", ls=":", lw=1.6, label=f"Mean ({np.mean(E):.1f})")
    ax.set_xlabel(f"PHP per person-hour (axis trimmed at the {upper_pct:g}th percentile)")
    ax.set_ylabel("Density")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(lo, hi)
    ax.set_ylim(bottom=0)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


def fig_yield_exceedance(Y: np.ndarray, thresholds: Dict[str, float]) -> plt.Figure:
    """Plot the empirical exceedance curve of yield Y against the three threshold medians."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ys = np.sort(Y[: min(len(Y), 60_000)])
    p = 1.0 - np.arange(len(ys)) / len(ys)
    ax.step(ys, p, where="post", color="#2e7d32", lw=1.8, label="Share of cycles reaching this yield")
    colors = ["#e65100", "#1f4e79", "#6f42c1"]
    for (name, val), c in zip(thresholds.items(), colors):
        ax.axvline(val, color=c, ls="--", lw=1.4, label=f"{name} (median {val:.3f} g)")
    hi = float(np.percentile(ys, 97.0))
    ax.set_xlim(0, max(hi, max(thresholds.values()) * 1.3))
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Group gold yield per cycle (g; axis trimmed at the 97th percentile)")
    ax.set_ylabel("Share of simulated cycles at or above")
    ax.set_title("Yield exceedance with the three separate thresholds", fontsize=10)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


def fig_stability(stage: pd.DataFrame) -> plt.Figure:
    """Plot how the risk probabilities and their MCSE bounds stabilize as N grows."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for k, c in zip(E2.PROB_KEYS, plt.get_cmap("tab10").colors):
        ax.errorbar(stage["N"], 100 * stage[k], yerr=200 * stage["MCSE_" + k], marker="o", capsize=3, color=c,
                    label=E2.LABELS[k], lw=1.4)
    ax.set_xscale("log")
    ax.set_xlabel("Simulated cycles (N, log scale)")
    ax.set_ylabel("Probability (%) ± 2 MCSE")
    ax.set_title("Stability of the risk probabilities as N grows", fontsize=10)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(fontsize=6.5, loc="best")
    fig.tight_layout()
    return fig


def fig_rank_sensitivity(rank: Dict[str, float]) -> plt.Figure:
    """Plot a horizontal bar chart of Spearman rank correlations with the labor return."""
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    items = list(rank.items())[::-1]
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    ax.barh(names, vals, color=["#2e7d32" if v > 0 else "#c0392b" for v in vals])
    ax.axvline(0, color="black", lw=1)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman rank correlation with the labor return (E)")
    ax.set_title("Rank-based sensitivity (monotone association, not a variance share)", fontsize=10)
    ax.grid(axis="x", ls=":", alpha=0.4)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# Exports
# --------------------------------------------------------------------------------------
def export_csv(res: E2.Result, max_rows: int = 5_000, *, schema_version: int = 2) -> bytes:
    """Export per-cycle simulation arrays, including cash and reserve fields, as CSV bytes."""
    if isinstance(schema_version, bool) or schema_version not in (1, 2):
        raise E2.InputError("CSV schema_version must be 1 (legacy) or 2.")
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 0:
        raise E2.InputError("max_rows must be a nonnegative integer.")
    a = res.arrays
    m = min(max_rows, len(a["E"]))
    df = pd.DataFrame({
        "cycle_id": np.arange(1, m + 1),
        "yield_g": a["Y"][:m],
        **({"purity_q": a["q"][:m]} if schema_version == 1 else
           {"export_schema_version": 2, "valuation_multiplier": a["q"][:m],
            "physical_purity_q": a["physical_purity_q"][:m] if "physical_purity_q" in a else np.full(m, np.nan)}),
        "price_php_per_g": a["P"][:m],
        "miner_costs_php": a["C_eff"][:m], "miner_profit_php": a["Pi"][:m],
        **{key: a[key][:m] for key in E2.TIMING_KEYS if key in a},
        **{key: values[:m] for key, values in a.items() if key.startswith(("rate::", "cost::"))},
        "valuation_mass_g": a["F"][:m],
        "valuation_mass_label": res.manifest["valuation_mass_label"],
        **{f"{key}_php": a[key][:m] for key in ("cash_cost_total", "cash_cost_miner",
           "depreciation_total", "depreciation_miner", "cash_before_reserve", "cash_after_reserve",
           "reserve_planned", "reserve_funded", "reserve_shortfall", "reserve_miner_contribution")},
        "return_php_per_person_hour": a["E"][:m],
        "Y_breakeven_g": a["Y_BE"][:m], "Y_wage_g": a["Y_W"][:m], "Y_household_g": a["Y_HH"][:m],
    })
    return df.to_csv(index=False).encode("utf-8")


def export_manifest(res: E2.Result, extras: Optional[dict] = None, *, schema_version: int = 2) -> bytes:
    """Serialize the run manifest, summary and warnings (plus any `extras`) to JSON bytes."""
    if isinstance(schema_version, bool) or schema_version not in (1, 2):
        raise E2.InputError("Export schema must be 1 or 2.")
    doc = {"dashboard_version": DASHBOARD_VERSION, **res.manifest, "summary": E2._jsonable(res.summary),
           "warnings": list(res.warnings), **(extras or {}), "export_schema_version": schema_version,
           "purity_fields": {"physical_purity_q": "physical assay; blank when unknown", "valuation_multiplier": "revenue factor"}
           if schema_version == 2 else {"purity_q": "legacy valuation multiplier, not necessarily physical assay"}}
    return json.dumps(E2._jsonable(doc), indent=2, default=str, allow_nan=False).encode("utf-8")


# --------------------------------------------------------------------------------------
# Legacy V1.0 (frozen) -- kept for comparison, relabelled
# --------------------------------------------------------------------------------------
LEGACY_DEFAULTS = legacy_synthetic_values()


def legacy_errors(v: Dict[str, float]) -> List[str]:
    """Validate legacy V1.0 inputs for required orderings; return a list of error messages."""
    errs = []
    if not v["Gmin"] < v["Gsub"]:
        errs.append("Typical minimum yield must be strictly lower than typical maximum yield.")
    if not v["Gsub"] < v["Gjack"]:
        errs.append("Jackpot maximum must be strictly greater than the typical maximum.")
    if not v["Pmin"] < v["Pmax"]:
        errs.append("Minimum price must be strictly lower than maximum price.")
    return errs


def run_legacy(v: Dict[str, float]) -> dict:
    """Run the frozen legacy V1.0 Monte Carlo simulation from UI values `v`."""
    return legacy_v1.run_monte_carlo(
        H=v["H"], T=v["T"], Cfamily=v["Cfamily"], opportunity_wage=v["W"], Cfixed=v["Cfixed"], delta=v["delta"],
        Cfuel=v["Cfuel"], alpha=v["alpha"], Gf_min=v["Gmin"], Gf_subsistence_max=v["Gsub"],
        Gf_jackpot_max=v["Gjack"], eta=v["eta"], Pg_min=v["Pmin"], Pg_max=v["Pmax"], iterations=int(v["N"]))


LEGACY_FLAWS_MD = """
**Why V1.0 is kept only for comparison**
* The yield lognormal assumes the typical minimum is the 5th percentile and the maximum the 99th; these percentile assignments are model assumptions.
* It cannot produce a zero-yield cycle, and its "jackpot" is a lognormal anchored at the typical minimum, not a tail.
* The jackpot frequency is a heuristic clamp (0.2%–5%), not data.
* Purity (η) is multiplied onto a farmgate price that already reflects grade.
* The return is divided by shift hours; group inputs need person-hours. The per-person conversion was done by hand.
* Its "target break-even yield" includes family cost, while "sustainability" is measured against the wage benchmark.
"""

ABOUT_MD = """
### Model scope and assumptions
Every iteration represents a complete production batch. The active crew attends every scheduled shift; partial attendance and changing membership are unsupported in batch mode.
Working days follow discretized Beta-PERT; independent nonnegative idle days extend calendar time.
Daily shift counts use the ten-day frequency grid; one conditional duration applies to all shifts that day.
Batch yield is sampled directly and is independent of timing and valuation.

Expense rates describe the whole operation. The Gaussian copula links rates before multiplication
by operating exposure. Realized total-expense correlations can differ from the rate targets.
Worksite food is additional consumption excluded from the household benchmark.

Household counts are independent of active crew size; daily outside income reduces household needs,
clipped at zero. All active contributors attend every scheduled shift.
Hourly wage adequacy, household economic adequacy, and household cash adequacy are distinct.
Economic profit includes depreciation; cash excludes it and accounts for funded reserves.
The reserve is a single-batch allocation without accumulated balances or borrowing.

All presets are synthetic demonstrations. Tests verify equations and software behavior;
empirical validation requires independently obtained observations. External benchmarks need a source and date.
Empirical fitting/resampling, fitted-parameter risk intervals, observed jackpot calibration and broader
cross-input dependence remain deferred. Aggregate gross/net sharing has no reimbursements or full stakeholder ledger.
Numerical error bounds and assumed-parameter scenario ranges do not establish predictive intervals.
"""


def benchmark_sensitivity(res, inp):
    """Threshold-only comparisons reuse the same economic draws, never resample."""
    rows = []
    for scale in (.8, 1., 1.2):
        wage = inp.W_opp * scale
        below = float(np.mean(res.arrays["E"] < wage))
        for low, moderate in ((.60, .30), (.67, .34), (.75, .40)):
            rows.append({"Wage (PHP/person-hour)": wage, "Wage factor": scale,
                         "Low cutoff": low, "Moderate cutoff": moderate,
                         "P_below_wage": below, "Illustrative band": E2.risk_band(1. - below, low, moderate),
                         "Median wage yield (g)": float(np.median((res.arrays["C_eff"] + wage * res.arrays["L_person_hours"])
                                                                     / (res.arrays["q"] * res.arrays["P"] * inp.s_m)))})
    return pd.DataFrame(rows)
