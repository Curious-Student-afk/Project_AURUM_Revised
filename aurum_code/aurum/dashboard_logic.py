"""
dashboard_logic.py -- everything the dashboard does EXCEPT drawing widgets.

Kept free of Streamlit so it can be unit-tested offline. app.py is a thin layer on top.

Design rules taken from the expert report:
  * every input carries a provenance tag (pilot / assumed / external / synthetic / user-entered)
  * three thresholds (break-even, opportunity-wage, household) are reported separately
  * risk outputs use the recommended labels, with MCSE, quantiles and expected shortfall
  * no "poverty trap" wording and no directive advice (no "exit" / "greenlight" messages)
  * observed pilot data, simulated results, assumptions and normative thresholds are kept visibly apart
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
from scenarios import synthetic_profiles  # noqa: E402

DASHBOARD_VERSION = "2.0.0-draft"

# --------------------------------------------------------------------------------------
# Provenance tags
# --------------------------------------------------------------------------------------
TAG_LABEL = {"pilot": "🟢 pilot", "assumed": "🟠 assumed", "external": "🔵 external",
             "synthetic": "⚪ synthetic", "user": "🟣 user-entered"}
TAG_HELP = {
    "pilot": "From the 21-interview pilot (non-representative convenience sample; averages of respondents' answers).",
    "assumed": "NOT collected in the pilot. A placeholder that must be replaced or sensitivity-tested.",
    "external": "From a cited outside source.",
    "synthetic": "Invented for verification only.",
    "user": "You changed this from the preset value. Its source is not verified.",
}

PRESETS = ["Pilot-provisional baseline", "Synthetic: High-Cost", "Synthetic: Low-Purity"]
_PRESET_TO_PROFILE = {PRESETS[0]: "Baseline", PRESETS[1]: "High-Cost", PRESETS[2]: "Low-Purity"}

_BASE_TAGS = {
    "n_c": "assumed", "D_c": "pilot", "h": "pilot", "s_m": "pilot", "tribute_base": "assumed",
    "ya": "pilot", "yb": "pilot", "mode_pct": "assumed",
    "p0_mode": "assumed", "p0": "assumed", "z": "assumed", "n_obs": "assumed",
    "pi_J": "assumed", "jack_thr": "pilot", "jack_cap": "pilot", "jack_shape": "assumed",
    "price_basis": "assumed", "p_min": "pilot", "p_mode": "pilot", "p_max": "pilot",
    "q_min": "assumed", "q_mode": "assumed", "q_max": "assumed",
    "fixed_cost": "assumed", "fixed_share": "assumed",
    "fuel_min": "assumed", "fuel_mode": "assumed", "fuel_max": "assumed", "fuel_share": "assumed",
    "trans_min": "assumed", "trans_mode": "assumed", "trans_max": "assumed", "trans_share": "assumed",
    "rho": "assumed", "dependence": "assumed",
    "eq_mode": "assumed", "eq_acq": "assumed", "eq_salv": "assumed", "eq_life": "assumed",
    "eq_util": "assumed", "eq_rent": "assumed", "eq_share": "assumed",
    "W_opp": "pilot", "hh_need": "pilot", "n_hh": "assumed", "nonmining": "assumed",
}
_COST_KEYS = ["fixed_cost", "fixed_share", "fuel_min", "fuel_mode", "fuel_max", "fuel_share",
              "trans_min", "trans_mode", "trans_max", "trans_share"]


def param_tags(preset: str) -> Dict[str, str]:
    tags = dict(_BASE_TAGS)
    if preset == PRESETS[1]:
        tags.update({k: "synthetic" for k in _COST_KEYS if not k.endswith("share")})
    if preset == PRESETS[2]:
        tags.update({k: "synthetic" for k in ("price_basis", "p_min", "p_mode", "p_max", "q_min", "q_mode", "q_max")})
    return tags


# --------------------------------------------------------------------------------------
# Inputs <-> flat UI values
# --------------------------------------------------------------------------------------
def values_from_inputs(inp: E2.Inputs) -> Dict[str, object]:
    cost = {c.name: c for c in inp.costs}
    fx, fu, tr = cost["fixed_operating"], cost["fuel"], cost["transport_hauling"]
    y = inp.yield_ord
    j = inp.jackpot or E2.Jackpot(14.4, 24.0, 1.5)
    price = inp.price if isinstance(inp.price, E2.PERT) else E2.PERT(inp.price, inp.price, inp.price)
    pur = inp.purity if isinstance(inp.purity, E2.PERT) else E2.PERT(inp.purity, inp.purity, inp.purity)
    if inp.price_basis == "farmgate":
        pur = E2.PERT(0.80, 0.88, 0.93)               # unused on a farmgate basis; shown as a starting point
    rho = inp.rho_spearman[0][1] if inp.rho_spearman else 0.5
    return {
        "n_c": int(inp.n_c), "D_c": int(inp.D_c), "h": float(inp.hours_per_worker_shift),
        "s_m": float(inp.s_m), "tribute_base": inp.tribute_base,
        "ya": float(y.a), "yb": float(y.b),
        "mode_pct": float(100.0 * (y.m - y.a) / (y.b - y.a)) if y.b > y.a else 50.0,
        "p0_mode": "fixed value", "p0": float(inp.zero.p0 if inp.zero.p0 is not None else 0.1), "z": 0, "n_obs": 10,
        "pi_J": float(inp.pi_J), "jack_thr": float(j.threshold), "jack_cap": float(j.cap), "jack_shape": float(j.shape),
        "price_basis": inp.price_basis,
        "p_min": float(price.a), "p_mode": float(price.m), "p_max": float(price.b),
        "q_min": float(pur.a), "q_mode": float(pur.m), "q_max": float(pur.b),
        "fixed_cost": float(fx.fixed), "fixed_share": float(fx.miner_share),
        "fuel_min": float(fu.pert.a), "fuel_mode": float(fu.pert.m), "fuel_max": float(fu.pert.b),
        "fuel_share": float(fu.miner_share),
        "trans_min": float(tr.pert.a), "trans_mode": float(tr.pert.m), "trans_max": float(tr.pert.b),
        "trans_share": float(tr.miner_share),
        "rho": float(rho), "dependence": inp.dependence,
        "eq_mode": "none", "eq_acq": 0.0, "eq_salv": 0.0, "eq_life": 1000.0, "eq_util": 1.0, "eq_rent": 0.0,
        "eq_share": 1.0,
        "W_opp": float(inp.W_opp), "W_src": inp.W_opp_source, "W_date": inp.W_opp_date,
        "hh_need": float(inp.hh_need_per_household_day), "n_hh": 0 if inp.n_households is None else int(inp.n_households),
        "nonmining": float(inp.nonmining_income_per_household_day),
        "seed": 42, "tol": 0.005, "n_max": 200_000,
    }


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
    return {k: tag_for(k, values[k], preset) for k in param_tags(preset) if k in values}


def build_inputs(v: Dict[str, object], preset: Optional[str] = None) -> E2.Inputs:
    """Flat UI values -> engine Inputs (validation happens in engine_v2.validate / run)."""
    pert = E2.PERT
    zero = (E2.ZeroYield(p0=float(v["p0"])) if v["p0_mode"] == "fixed value"
            else E2.ZeroYield(z=int(v["z"]), n=int(v["n_obs"])))
    a, b = float(v["ya"]), float(v["yb"])
    y = pert(a, a + float(v["mode_pct"]) / 100.0 * (b - a), b)
    jack = (E2.Jackpot(float(v["jack_thr"]), float(v["jack_cap"]), float(v["jack_shape"]))
            if float(v["pi_J"]) > 0 else None)
    costs = [
        E2.CostItem("fixed_operating", float(v["fixed_share"]), fixed=float(v["fixed_cost"])),
        E2.CostItem("fuel", float(v["fuel_share"]), pert=pert(float(v["fuel_min"]), float(v["fuel_mode"]), float(v["fuel_max"]))),
        E2.CostItem("transport_hauling", float(v["trans_share"]),
                    pert=pert(float(v["trans_min"]), float(v["trans_mode"]), float(v["trans_max"]))),
    ]
    equipment: List[E2.EquipmentItem] = []
    if v["eq_mode"] != "none":
        equipment.append(E2.EquipmentItem(
            "main_equipment", str(v["eq_mode"]), miner_share=float(v["eq_share"]),
            acquisition=float(v["eq_acq"]), salvage=float(v["eq_salv"]), life_cycles=float(v["eq_life"]),
            utilization=float(v["eq_util"]), rental_rate_per_shift=float(v["eq_rent"])))
    rho = float(v["rho"])
    price_basis = str(v["price_basis"])
    price = pert(float(v["p_min"]), float(v["p_mode"]), float(v["p_max"]))
    purity = (pert(float(v["q_min"]), float(v["q_mode"]), float(v["q_max"])) if price_basis == "reference" else 1.0)
    n_hh = int(v["n_hh"])
    return E2.Inputs(
        n_c=int(v["n_c"]), D_c=int(v["D_c"]), hours_per_worker_shift=float(v["h"]),
        s_m=float(v["s_m"]), tribute_base=str(v["tribute_base"]),
        yield_ord=y, zero=zero, pi_J=float(v["pi_J"]), jackpot=jack,
        price_basis=price_basis, price=price, purity=purity,
        costs=costs, equipment=equipment, rho_spearman=[[1.0, rho], [rho, 1.0]],
        dependence=str(v["dependence"]),
        W_opp=float(v["W_opp"]), W_opp_source=str(v.get("W_src", "")), W_opp_date=str(v.get("W_date", "")),
        hh_need_per_household_day=float(v["hh_need"]), n_households=None if n_hh == 0 else n_hh,
        nonmining_income_per_household_day=float(v["nonmining"]),
        provenance=effective_tags(v, preset) if preset else {},
    )


def run_v2(v: Dict[str, object], preset: Optional[str] = None) -> Tuple[E2.Inputs, E2.Result]:
    inp = build_inputs(v, preset)
    res = E2.run(inp, seed=int(v["seed"]), tol=float(v["tol"]), n_max=int(v["n_max"]))
    return inp, res


# --------------------------------------------------------------------------------------
# Text: interpretation and messages (neutral, no directive advice)
# --------------------------------------------------------------------------------------
def interpretation_text(res: E2.Result, inp: E2.Inputs, n_assumed: int) -> str:
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
    ("Household-adequacy yield", "Y_HH", "Yield at which profit covers the households' cash requirement for the cycle"),
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
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for k, c in zip(E2.PROB_KEYS, ["#c0392b", "#1f4e79", "#e65100", "#6f42c1"]):
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
    a = res.arrays
    m = min(max_rows, len(a["E"]))
    df = pd.DataFrame({
        "cycle_id": np.arange(1, m + 1),
        "yield_g": a["Y"][:m], "purity_q": a["q"][:m], "price_php_per_g": a["P"][:m],
        "miner_costs_php": a["C_eff"][:m], "miner_profit_php": a["Pi"][:m],
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
LEGACY_DEFAULTS = dict(H=8, T=0.85, Cfamily=44.80, W=65.63, Cfixed=80.0, delta=1.10, Cfuel=160.0, alpha=0.02,
                       Gmin=0.33, Gsub=1.45, Gjack=19.47, eta=0.95, Pmin=7282.35, Pmax=8900.65, N=100_000)


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
* The yield lognormal assumes the typical minimum is the 5th percentile and the maximum the 99th; respondents never said that.
* It cannot produce a zero-yield cycle, and its "jackpot" is a lognormal anchored at the typical minimum, not a tail.
* The jackpot frequency is a heuristic clamp (0.2%–5%), not data.
* Purity (η) is multiplied onto a farmgate price that already reflects grade.
* The return is divided by shift hours; group inputs need person-hours. The per-person conversion was done by hand.
* Its "target break-even yield" includes family cost, while "sustainability" is measured against the wage benchmark.
"""

ABOUT_MD = """
**What this dashboard is**
A simulation tool that turns the inputs you enter into probabilities and yield thresholds for a single production cycle.

**What the results are**
Scenario results under the stated inputs. Inputs marked 🟠 *assumed* were not collected in the pilot (crew size, costs,
zero-yield and jackpot frequencies, yield mode, dependence).

**What the results are not**
* Not findings about Pantukan miners: the pilot was a non-representative convenience sample of 21 people.
* Not an empirical validation (Domain G needs Phase 2 field data); passing tests verify the code, not the model.
* Not a poverty-trap model: a one-cycle probability of earning below a benchmark is not persistence over time.
* Not advice to stop, continue or invest.

| Kind of content | Where it appears |
|---|---|
| Observed pilot data | inputs tagged 🟢 pilot |
| Simulated predictions | probabilities, percentiles and thresholds in the result tabs |
| Assumptions | inputs tagged 🟠 assumed or 🟣 user-entered |
| Normative thresholds | wage benchmark, illustrative risk-band cut-offs |
"""
