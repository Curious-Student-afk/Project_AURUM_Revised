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
from scenarios import synthetic_profiles, default_timing, legacy_synthetic_values  # noqa: E402

DASHBOARD_VERSION = "2.2.0-draft"

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
    names = list(COST_FIELDS)
    return [(i, j, f"rho_{names[i]}_{names[j]}") for i in range(len(names)) for j in range(i + 1, len(names))]


def param_tags(preset: str) -> Dict[str, str]:
    return {k: "synthetic" for k in preset_values(preset)
            if k not in ("seed", "tol", "n_max", "W_src", "W_date")}


def values_from_inputs(inp: E2.Inputs) -> Dict[str, object]:
    """Map supported dashboard expense categories and one equipment item to widgets."""
    y, j = inp.yield_ord, inp.jackpot or E2.Jackpot(30, 60, 1.5)
    price = inp.price if isinstance(inp.price, E2.PERT) else E2.PERT(inp.price, inp.price, inp.price)
    pur = inp.purity if isinstance(inp.purity, E2.PERT) else E2.PERT(inp.purity, inp.purity, inp.purity)
    if inp.price_basis == "farmgate":
        pur = E2.PERT(.75, .85, .95)
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
        "q_min": float(pur.a), "q_mode": float(pur.m), "q_max": float(pur.b),
        "fixed_cost": 0., "fixed_share": 1., "dependence": inp.dependence,
        "reserve_mode": inp.reserve_mode, "reserve_location": inp.reserve_location or "choose",
        "reserve_priority": inp.reserve_priority or "choose", "reserve_custom": inp.reserve_custom_per_cycle,
        "eq_mode": "none", "eq_acq": 0., "eq_salv": 0., "eq_life": 1000., "eq_util": 1., "eq_rent": 0., "eq_share": 1.,
        "W_opp": inp.W_opp, "W_src": inp.W_opp_source, "W_date": inp.W_opp_date,
        "hh_need": inp.hh_need_per_household_day, "n_hh": inp.n_households or 0,
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
        v.update(fixed_cost=float(c.fixed), fixed_share=float(c.miner_share))
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
    effective = dict(values)
    if values.get("share_preset") in ("50/50", "60/40", "65/35"):
        effective["s_m"] = {"50/50": .5, "60/40": .6, "65/35": .65}[values["share_preset"]]
    return {k: tag_for(k, effective[k], preset) for k in param_tags(preset) if k in effective}


def build_inputs(v: Dict[str, object], preset: Optional[str] = None) -> E2.Inputs:
    """Flat UI values -> engine Inputs (validation happens in engine_v2.validate / run)."""
    numeric = ["s_m", "ya", "yb", "ym", "pi_J", "p_min", "p_mode", "p_max",
               "fixed_cost", "fixed_share", "W_opp", "hh_need"]
    integers = [("n_c", 1)]
    batch = v["timing_mode"] == "batch"
    if v["timing_mode"] not in ("batch", "fixed"):
        raise E2.InputError("Choose batch or fixed timing.")
    if batch:
        integers += [(f"work_{s}", 1) for s in ("min", "mode", "max")]
        integers += [(f"days_{d}", 0) for d in (1, 2, 3)]
        numeric += [f"idle_{s}" for s in ("min", "mode", "max")]
        numeric += [f"h{d}_{s}" for d in (1, 2, 3) for s in ("min", "mode", "max")]
    else:
        integers += [("D_c", 1), ("n_hh", 0)]
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
    if v["price_basis"] == "reference":
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
        return E2.PERT(*(v[f"{prefix}_{s}"] for s in ("min", "mode", "max")))
    timing = (E2.BatchTiming(pert("work"), pert("idle"), tuple(v[f"days_{d}"] for d in (1, 2, 3)),
                            {d: pert(f"h{d}") for d in (1, 2, 3)}) if batch else None)
    zero = E2.ZeroYield(p0=v["p0"]) if v["p0_mode"] == "fixed value" else E2.ZeroYield(z=v["z"], n=v["n_obs"])
    y = E2.PERT(v["ya"], v["ym"], v["yb"])
    jack = E2.Jackpot(v["jack_thr"], v["jack_cap"], v["jack_shape"]) if v["pi_J"] > 0 else None
    costs = [E2.CostItem("fixed_operating", v["fixed_share"], fixed=v["fixed_cost"])]
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
        price_basis=v["price_basis"], price=pert("p"), purity=pert("q") if v["price_basis"] == "reference" else 1.,
        costs=costs, equipment=equipment, dependence=v["dependence"], rho_spearman=rho.tolist(),
        reserve_mode=v.get("reserve_mode", "none"),
        reserve_location=None if v.get("reserve_location", "choose") == "choose" else v["reserve_location"],
        reserve_priority=None if v.get("reserve_priority", "choose") == "choose" else v["reserve_priority"],
        reserve_custom_per_cycle=v.get("reserve_custom", 0.),
        W_opp=v["W_opp"], W_opp_source=v.get("W_src", ""), W_opp_date=v.get("W_date", ""),
        hh_need_per_household_day=v["hh_need"], n_households=None if batch or v["n_hh"] == 0 else v["n_hh"],
        nonmining_income_per_household_day=0. if batch else v["nonmining"],
        provenance=effective_tags(v, preset) if preset else {})


def run_v2(v: Dict[str, object], preset: Optional[str] = None) -> Tuple[E2.Inputs, E2.Result]:
    """Build engine Inputs from UI values and run the V2 simulation."""
    inp = build_inputs(v, preset)
    res = E2.run(inp, seed=v["seed"], tol=v["tol"], n_max=v["n_max"])
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
    out = [("warning", w) for w in res.warnings]
    if not res.summary["converged"]:
        out.append(("warning", "Monte Carlo convergence was not reached; do not quote these probabilities without "
                    "stating the MCSE shown."))
    if tags.get("W_opp") and tags["W_opp"] != "external":
        out.append(("info", "The wage benchmark is a normative threshold. Cite its source and effective date "
                    "(wage order) before reporting."))
    return out


def risk_band_text(res: E2.Result) -> str:
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
    rows = []
    for name, key, meaning in THRESHOLD_ROWS:
        a = res.arrays[key]
        p5, p50, p95 = np.percentile(a, [5, 50, 95])
        rows.append({"Threshold": name, "Median (g/cycle)": round(float(p50), 4),
                     "5th–95th percentile (g)": f"{p5:.4f} – {p95:.4f}", "Meaning": meaning})
    return pd.DataFrame(rows)


def risk_table(res: E2.Result) -> pd.DataFrame:
    s = res.summary
    return pd.DataFrame([{"Output": E2.LABELS[k], "Probability": f"{s[k]:.2%}",
                          "± 2 MCSE": f"{2 * s['MCSE_' + k]:.2%}"} for k in E2.PROB_KEYS])


def return_table(res: E2.Result) -> pd.DataFrame:
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
    labels = {"working_days": "Active working days", "idle_days": "Idle days",
              "calendar_days_per_cycle": "Calendar days", "total_shifts": "Total shifts",
              "operating_hours": "Operating hours", "L_person_hours": "Labor (person-hours)",
              "B_HH": "Household consumption requirement (PHP)"}
    return pd.DataFrame([{"Measure": label, "Mean": res.summary[f"{key}_mean"],
                          "Median": res.summary[f"{key}_median"]}
                         for key, label in labels.items() if f"{key}_mean" in res.summary])


def assumed_inputs(tags: Dict[str, str]) -> List[str]:
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
def export_csv(res: E2.Result, max_rows: int = 5_000) -> bytes:
    """Export per-cycle simulation arrays, including cash and reserve fields, as CSV bytes."""
    a = res.arrays
    m = min(max_rows, len(a["E"]))
    df = pd.DataFrame({
        "cycle_id": np.arange(1, m + 1),
        "yield_g": a["Y"][:m], "purity_q": a["q"][:m], "price_php_per_g": a["P"][:m],
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


def export_manifest(res: E2.Result, extras: Optional[dict] = None) -> bytes:
    doc = {"dashboard_version": DASHBOARD_VERSION, **res.manifest, "summary": E2._jsonable(res.summary),
           "warnings": list(res.warnings), **(extras or {})}
    return json.dumps(E2._jsonable(doc), indent=2, default=str).encode("utf-8")


# --------------------------------------------------------------------------------------
# Legacy V1.0 (frozen) -- kept for comparison, relabelled
# --------------------------------------------------------------------------------------
LEGACY_DEFAULTS = legacy_synthetic_values()


def legacy_errors(v: Dict[str, float]) -> List[str]:
    errs = []
    if not v["Gmin"] < v["Gsub"]:
        errs.append("Typical minimum yield must be strictly lower than typical maximum yield.")
    if not v["Gsub"] < v["Gjack"]:
        errs.append("Jackpot maximum must be strictly greater than the typical maximum.")
    if not v["Pmin"] < v["Pmax"]:
        errs.append("Minimum price must be strictly lower than maximum price.")
    return errs


def run_legacy(v: Dict[str, float]) -> dict:
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
Every iteration represents a complete production batch. The crew attends every scheduled shift.
Working days follow discretized Beta-PERT; independent nonnegative idle days extend calendar time.
Daily shift counts use the ten-day frequency grid; one conditional duration applies to all shifts that day.
Batch yield is sampled directly and is independent of timing and valuation.

Expense rates describe the whole operation. The Gaussian copula links rates before multiplication
by operating exposure. Realized total-expense correlations can differ from the rate targets.
Worksite food is additional consumption excluded from the household benchmark.

Hourly wage adequacy, household economic adequacy, and household cash adequacy are distinct.
Economic profit includes depreciation; cash excludes it and accounts for funded reserves.
The reserve is a single-batch allocation without accumulated balances or borrowing.

All presets are synthetic demonstrations. Tests verify equations and software behavior;
empirical validation requires independently obtained observations. External benchmarks need a source and date.
"""
