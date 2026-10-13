"""
ablation.py -- V1.0 -> V2.0 ablation study for Project AURUM.

QUESTION: which of V1.0's flaws changed the answer, and by how much?

Four complementary views (each fix is a switch, everything else held fixed):
  1. FORWARD CHAIN  : start from V1.0 synthetic reference, add fixes one at a time, cumulatively  (S0 ... S8)
  2. SOLO           : V1.0 plus ONE fix only                                          (order-free)
  3. LEAVE-ONE-OUT  : the full V2.0 model with ONE fix reverted/removed               (order-free)
  4. SENSITIVITY    : the full model under alternative values of the ASSUMED parameters
plus a seed-to-seed noise check, and a diagnostic showing what V1.0's equation gives if the group-level
inputs are not converted by hand.

Headline metric: P(E < W_opp)  = "probability of labor return below the wage benchmark"
(1 - this = V1.0's "Probability of Real Economic Sustainability").

HOW TO READ THE RESULTS
  * These are SCENARIO results from independently constructed synthetic inputs.
    The comparison exercises the fixed-timing equations; it does not evaluate batch timing.
  * Fixes are bundled where V1.0 bundles them (e.g. V1.0's lognormal and its heuristic "jackpot"
    cannot be separated); the step descriptions say so.
  * Random streams are separate per component (SeedSequence.spawn), so switching one component on or
    off does not disturb the random numbers of the others (common random numbers). Steps that keep V1.0's
    yield and price use V1.0's exact draws. Differences are reported with a conservative standard error.

Usage:   python ablation.py [--n 500000] [--seed 42] [--out ablation_out]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import engine_v2 as E2  # noqa: E402
from scenarios import legacy_synthetic_values
import legacy_v1  # noqa: E402

# --------------------------------------------------------------------------------------
# Synthetic per-person benchmark for the V1 equations.
# --------------------------------------------------------------------------------------
BENCH = {k: v for k, v in legacy_synthetic_values().items() if k != "N"}
ASSUME = dict(n_c=4, s_group=.5, p0=.1, pi_J=.02, jack_threshold=30., jack_cap=60.,
              jack_shape=1.5, mode_frac=.4, price_mode_frac=.5, cost_lo=.8, cost_hi=1.2, rho=.5)
PROVENANCE = {key: "synthetic" for key in ASSUME}

STREAMS = ("yield", "hurdle", "jackstate", "jackmag", "price", "costs")


@dataclass(frozen=True)
class Cfg:
    label: str = ""
    labor: str = "v1_manual"          # "v1_manual" (per-person inputs) | "explicit" (group inputs, n_c*H)
    eta: str = "v1_double"            # "v1_double" (eta on a farmgate price) | "once" (q = 1)
    yield_model: str = "v1_lognormal"  # "v1_lognormal" | "pert"
    price_model: str = "v1_lognormal"  # "v1_lognormal" | "pert"
    p0: float = 0.0                   # zero-yield hurdle (0 = off)
    jackpot: str = "v1_heuristic"     # "v1_heuristic" | "tail" | "off"
    costs: str = "deterministic"      # "deterministic" | "independent" | "copula"


V1 = Cfg(label="S0  V1.0 synthetic reference")
FULL = Cfg(label="Full fixed-timing model", labor="explicit", eta="once", yield_model="pert", price_model="pert",
           p0=ASSUME["p0"], jackpot="tail", costs="copula")

# --------------------------------------------------------------------------------------
# V1.0 draws (frozen code, hard-coded seed 42) -- cached per N
# --------------------------------------------------------------------------------------
_V1_CACHE: Dict[int, dict] = {}


def v1_draws(N: int) -> dict:
    """Return V1.0's Monte Carlo draws for `N` iterations, cached per N."""
    if N not in _V1_CACHE:
        b = BENCH
        _V1_CACHE[N] = legacy_v1.run_monte_carlo(
            H=b["H"], T=b["T"], Cfamily=b["Cfamily"], opportunity_wage=b["W"], Cfixed=b["Cfixed"],
            delta=b["delta"], Cfuel=b["Cfuel"], alpha=b["alpha"], Gf_min=b["Gmin"],
            Gf_subsistence_max=b["Gsub"], Gf_jackpot_max=b["Gjack"], eta=b["eta"],
            Pg_min=b["Pmin"], Pg_max=b["Pmax"], iterations=N)
    return _V1_CACHE[N]


def _lognormal_base(rng, n):
    """V1.0's hidden-percentile calibration (min = P5, max = P99), clipped at 1.1 x max -- no jackpot."""
    b = BENCH
    sigma = (np.log(b["Gsub"]) - np.log(b["Gmin"])) / (2.326 + 1.645)
    mu = np.log(b["Gmin"]) + 1.645 * sigma
    return np.clip(rng.lognormal(mu, sigma, n), 0.0, b["Gsub"] * 1.1)


# --------------------------------------------------------------------------------------
# Simulator with one switch per fix
# --------------------------------------------------------------------------------------
def simulate(cfg: Cfg, N: int, seed: int = 42, A: Optional[dict] = None) -> Dict[str, np.ndarray]:
    """Simulate `N` cycles under configuration `cfg`, switching each fix on/off per its flags."""
    A = {**ASSUME, **(A or {})}
    b = BENCH
    rs = {n: np.random.default_rng(s) for n, s in zip(STREAMS, np.random.SeedSequence(seed).spawn(len(STREAMS)))}
    if cfg.jackpot == "v1_heuristic" and cfg.yield_model != "v1_lognormal":
        raise ValueError("V1.0's heuristic jackpot is inseparable from V1.0's lognormal yield.")

    # ---- yield ----
    if cfg.jackpot == "v1_heuristic":
        Y = v1_draws(N)["Gf_array"].copy()                       # V1.0's own mixture
    else:
        if cfg.yield_model == "pert":
            a_, b_ = b["Gmin"], b["Gsub"]
            y_ord = E2.PERT(a_, a_ + A["mode_frac"] * (b_ - a_), b_).sample(rs["yield"], N)
        else:
            y_ord = _lognormal_base(rs["yield"], N)
        if cfg.jackpot == "tail":
            is_j = rs["jackstate"].random(N) < A["pi_J"]
            y_j = E2.Jackpot(A["jack_threshold"], A["jack_cap"], A["jack_shape"]).sample(rs["jackmag"], N)
            Y = np.where(is_j, y_j, y_ord)
        else:
            Y = y_ord
    if cfg.p0 > 0:
        Y = np.where(rs["hurdle"].random(N) >= cfg.p0, Y, 0.0)

    # ---- price & purity (counted once, or twice as in V1.0) ----
    if cfg.price_model == "v1_lognormal":
        P = v1_draws(N)["Pg_array"].copy()
    else:
        lo, hi = b["Pmin"], b["Pmax"]
        P = E2.PERT(lo, lo + A["price_mode_frac"] * (hi - lo), hi).sample(rs["price"], N)
    q = b["eta"] if cfg.eta == "v1_double" else 1.0

    # ---- costs, group level, miners bear 100% (V1.0 convention) ----
    fixed = b["delta"] * b["Cfixed"] * A["n_c"]                  # 440 PHP per cycle
    fuel = (1 + b["alpha"]) * b["Cfuel"] * A["n_c"]              # 816 PHP per cycle
    if cfg.costs == "deterministic":
        c1, c2 = np.full(N, fixed), np.full(N, fuel)
    else:
        rho_p = 2.0 * np.sin(np.pi * (A["rho"] if cfg.costs == "copula" else 0.0) / 6.0)
        z = rs["costs"].standard_normal((N, 2))
        z[:, 1] = rho_p * z[:, 0] + np.sqrt(1 - rho_p ** 2) * z[:, 1]
        u = np.clip(stats.norm.cdf(z), 1e-12, 1 - 1e-12)
        c1 = E2.PERT(A["cost_lo"] * fixed, fixed, A["cost_hi"] * fixed).ppf(u[:, 0])
        c2 = E2.PERT(A["cost_lo"] * fuel, fuel, A["cost_hi"] * fuel).ppf(u[:, 1])
    C = c1 + c2

    # ---- labour & share ----
    if cfg.labor == "v1_manual":      # Synthetic per-person V1 inputs
        s, L, C = 1.0 - min(b["T"], 0.99), b["H"], C / A["n_c"]
    else:                             # explicit group accounting
        s, L = A["s_group"], A["n_c"] * b["H"]
    k = E2.accounting_kernel(Y, np.full(N, q), P, C, C, s_m=s, L=float(L), W_opp=b["W"], B_HH=0.0,
                             tribute_base="gross")
    return {"E": k["E"], "Pi": k["Pi"], "Y": Y, "C": C}


def metrics(res: Dict[str, np.ndarray]) -> dict:
    """Summarise a simulate() result into headline probabilities and E/Y statistics."""
    E, Pi = res["E"], res["Pi"]
    n = len(E)
    k, kn = int(np.count_nonzero(E < BENCH["W"])), int(np.count_nonzero(Pi < 0))
    return {"N": n, "p_wage_below": k / n, "mcse": E2.mcse_prob(k, n), "p_negative": kn / n,
            "mcse_neg": E2.mcse_prob(kn, n), "E_mean": float(E.mean()), "E_median": float(np.median(E)),
            "E_P5": float(np.percentile(E, 5)), "mean_yield_g": float(res["Y"].mean())}


def run_cfg(cfg: Cfg, N: int, seed: int, A: Optional[dict] = None) -> dict:
    """Simulate configuration `cfg` and return its label plus summary metrics."""
    return {"label": cfg.label, **metrics(simulate(cfg, N, seed, A))}


def _delta(a: dict, b: dict):
    """Difference b - a in percentage points with a conservative SE (independent-stream formula)."""
    return 100 * (b["p_wage_below"] - a["p_wage_below"]), 100 * float(np.hypot(a["mcse"], b["mcse"]))


# --------------------------------------------------------------------------------------
# The four views
# --------------------------------------------------------------------------------------
FIXES = [  # (key, short name, description, how to turn it ON starting from V1.0 / how to turn it OFF from FULL)
    ("labor", "Person-hour accounting",
     "Explicit crew person-hours and group share versus the synthetic per-person V1 inputs",
     dict(labor="explicit"), dict(labor="v1_manual")),
    ("eta", "Purity counted once",
     "q = 1 on a farmgate price (V1.0 multiplied eta = 0.9 onto a price that already embeds grade)",
     dict(eta="once"), dict(eta="v1_double")),
    ("yield", "Beta-PERT yield",
     "Beta-PERT (same min/max, assumed mode) replaces the hidden-percentile lognormal; for V1.0 it also removes V1.0's "
     "heuristic 'jackpot' (the two are inseparable)",
     dict(yield_model="pert", jackpot="off"), dict(yield_model="v1_lognormal")),
    ("price", "Beta-PERT price",
     "Beta-PERT (same bounds, midpoint mode) replaces the hidden-percentile lognormal price",
     dict(price_model="pert"), dict(price_model="v1_lognormal")),
    ("zero", "Zero-yield hurdle", "Waste cycles with Y = 0 at probability p0 (assumed)",
     dict(p0=ASSUME["p0"]), dict(p0=0.0)),
    ("jackpot", "Jackpot tail",
     f"Truncated-Pareto jackpot on [{ASSUME['jack_threshold']}, {ASSUME['jack_cap']}] g at probability pi_J (synthetic)",
     dict(jackpot="tail"), dict(jackpot="off")),
    ("costs", "Stochastic costs",
     "Costs become mean-preserving PERT draws (spread assumed), independent",
     dict(costs="independent"), dict(costs="deterministic")),
    ("depend", "Cost dependence",
     "Gaussian copula with assumed Spearman rho between the two stochastic costs",
     dict(costs="copula"), dict(costs="independent")),
]
CHAIN_ORDER = ["labor", "eta", "yield", "price", "zero", "jackpot", "costs", "depend"]


def forward_chain(N, seed, A=None) -> pd.DataFrame:
    """Build the S0..S8 cumulative forward-chain table, adding one fix at a time from V1.0."""
    fx = {f[0]: f for f in FIXES}
    cfg, rows, prev, first = V1, [], None, None
    r = {"step": "S0", "change": "V1.0 synthetic reference (frozen code, benchmark inputs)", **run_cfg(cfg, N, seed, A)}
    first = prev = r
    r["d_prev_pp"] = r["d_prev_se"] = r["d_v1_pp"] = r["d_v1_se"] = np.nan
    rows.append(r)
    for i, key in enumerate(CHAIN_ORDER, start=1):
        cfg = dataclasses.replace(cfg, **fx[key][3])
        r = {"step": f"S{i}", "change": fx[key][2], **run_cfg(cfg, N, seed, A)}
        r["d_prev_pp"], r["d_prev_se"] = _delta(prev, r)
        r["d_v1_pp"], r["d_v1_se"] = _delta(first, r)
        rows.append(r)
        prev = r
    return pd.DataFrame(rows)


def solo(N, seed, A=None) -> pd.DataFrame:
    """Build the order-free table of each fix applied alone to V1.0."""
    base = run_cfg(V1, N, seed, A)
    rows = []
    for key, name, desc, on, _ in FIXES:
        cfg = dataclasses.replace(V1, **on)
        r = run_cfg(cfg, N, seed, A)
        d, se = _delta(base, r)
        rows.append({"fix": name, "description": desc, **r, "d_v1_pp": d, "d_v1_se": se})
    return pd.DataFrame(rows)


def leave_one_out(N, seed, A=None) -> pd.DataFrame:
    """Build the order-free table of each fix reverted/removed from the full model, ranked by effect size."""
    full = run_cfg(FULL, N, seed, A)
    rows = []
    for key, name, desc, _, off in FIXES:
        cfg = dataclasses.replace(FULL, **off)
        r = run_cfg(cfg, N, seed, A)
        d, se = _delta(full, r)
        rows.append({"fix_reverted": name, "description": desc, **r, "d_full_pp": d, "d_full_se": se})
    out = pd.DataFrame(rows)
    out["abs_d"] = out["d_full_pp"].abs()
    out = out.sort_values("abs_d", ascending=False).drop(columns="abs_d").reset_index(drop=True)
    out.attrs["full"] = full
    return out


def seed_check(N, seeds) -> pd.DataFrame:
    """Run the full model under each seed in `seeds` to measure seed-to-seed noise."""
    rows = [{"seed": s, **run_cfg(FULL, N, s)} for s in seeds]
    return pd.DataFrame(rows)


def sensitivity(N, seed) -> pd.DataFrame:
    """Run the full model over a grid of alternative values for each ASSUMED parameter."""
    grid = [("p0", [0.0, 0.05, 0.10, 0.20, 0.30]), ("pi_J", [0.0, 0.005, 0.01, 0.02, 0.05]),
            ("mode_frac", sorted({0.25, 0.40, 0.60, round(mode_frac_matching_v1_mean(), 3)})), ("n_c", [3, 4, 5, 6, 8]), ("rho", [0.0, 0.3, 0.5, 0.8]),
            ("cost_spread", [(0.9, 1.1), (0.8, 1.2), (0.7, 1.5)])]
    rows = []
    for name, values in grid:
        for v in values:
            if name == "p0":
                cfg, A = dataclasses.replace(FULL, p0=v), None
            elif name == "cost_spread":
                cfg, A = FULL, {"cost_lo": v[0], "cost_hi": v[1]}
            else:
                cfg, A = FULL, {name: v}
            r = run_cfg(cfg, N, seed, A)
            rows.append({"parameter": name, "value": str(v), "is_primary_assumption":
                         (ASSUME.get(name) == v) or (name == "p0" and v == ASSUME["p0"]) or
                         (name == "cost_spread" and v == (ASSUME["cost_lo"], ASSUME["cost_hi"])), **r})
    return pd.DataFrame(rows)


def mode_frac_matching_v1_mean(N: int = 200_000) -> float:
    """Mode position (as a fraction of the min-max range) at which Beta-PERT's mean equals V1.0's mean yield."""
    a_, b_ = BENCH["Gmin"], BENCH["Gsub"]
    target = float(v1_draws(N)["Gf_array"].mean())
    m = (6.0 * target - a_ - b_) / 4.0
    return float((m - a_) / (b_ - a_))


def v1_literal_diagnostic(N) -> dict:
    """What V1.0's equation gives if the GROUP-level inputs are fed in without the crew-size conversion."""
    b = BENCH
    v1 = legacy_v1.run_monte_carlo(
        H=b["H"], T=1 - ASSUME["s_group"], Cfamily=b["Cfamily"], opportunity_wage=b["W"],
        Cfixed=b["Cfixed"] * ASSUME["n_c"], delta=b["delta"], Cfuel=b["Cfuel"] * ASSUME["n_c"], alpha=b["alpha"],
        Gf_min=b["Gmin"], Gf_subsistence_max=b["Gsub"], Gf_jackpot_max=b["Gjack"], eta=b["eta"],
        Pg_min=b["Pmin"], Pg_max=b["Pmax"], iterations=N)
    E = v1["hourly_efficiency_array"]
    k = int(np.count_nonzero(E < b["W"]))
    return {"p_wage_below": k / N, "mcse": E2.mcse_prob(k, N), "E_mean": float(E.mean()),
            "E_median": float(np.median(E)), "E": E}


# --------------------------------------------------------------------------------------
# Figures and report
# --------------------------------------------------------------------------------------
def plot_forward(df: pd.DataFrame, path: str) -> None:
    """Save the forward-chain error-bar plot to `path`."""
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(df))
    y = 100 * df["p_wage_below"].to_numpy()
    err = 200 * df["mcse"].to_numpy()
    ax.errorbar(x, y, yerr=err, fmt="o-", color="#1f4e79", ecolor="#7f7f7f", capsize=3, lw=2)
    for xi, yi, d in zip(x, y, df["d_prev_pp"]):
        if not np.isnan(d):
            ax.annotate(f"{d:+.1f}", (xi, yi), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    ax.set_ylim(y.min() - 6, y.max() + 8)
    ax.set_xticks(x)
    ax.set_xticklabels(["S0\nV1.0"] + [f"S{i}\n{n}" for i, n in enumerate(
        ["person-hrs", "purity once", "PERT yield", "PERT price", "zero hurdle", "jackpot", "stoch. costs",
         "dependence"], start=1)], fontsize=8)
    ax.set_ylabel("P(labor return below wage benchmark), %")
    ax.set_title("Forward ablation: V1.0 -> V2.0, one fix added at a time (cumulative)\n"
                 "Scenario results under assumed parameters; labels = change vs previous step (pp)", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_loo(df: pd.DataFrame, path: str) -> None:
    """Save the leave-one-out horizontal bar plot to `path`."""
    d = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#c0392b" if v > 0 else "#2e7d32" for v in d["d_full_pp"]]
    ax.barh(d["fix_reverted"], d["d_full_pp"], xerr=2 * d["d_full_se"], color=colors, ecolor="#555555", capsize=3)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Change in P(labor return below wage benchmark) when the fix is reverted/removed (pp)")
    ax.set_title("Leave-one-out from the full V2.0 model\n(red = removing the fix raises the probability; "
                 "bars show +/- 2 SE)", fontsize=10)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def md_table(df: pd.DataFrame, fmt: Dict[str, str]) -> str:
    """Render `df` as a GitHub-flavoured Markdown table, formatting columns per `fmt`."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float) and np.isnan(v):
                cells.append("")
            elif c in fmt and not isinstance(v, str):
                cells.append(format(v, fmt[c]))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(fwd, sol, loo, seeds, sens, diag, N, seed) -> str:
    """Assemble the full Markdown ablation report from all computed tables."""
    pct = lambda s: 100 * s
    noise = 2 * pct(seeds["p_wage_below"].std(ddof=1))
    f = fwd.copy()
    f["P(E<W) %"], f["±2 SE"] = pct(f["p_wage_below"]), pct(2 * f["mcse"])
    f["P(Π<0) %"] = pct(f["p_negative"])
    f["Detectable?"] = np.where(f["d_prev_pp"].isna(), "", np.where(f["d_prev_pp"].abs() > noise, "yes", "within noise"))
    f = f[["step", "change", "P(E<W) %", "±2 SE", "d_prev_pp", "Detectable?", "d_v1_pp", "P(Π<0) %", "E_P5", "E_median", "E_mean"]]
    f.columns = ["Step", "Change added", "P(E<W) %", "±2 SE", "Δ vs prev (pp)", "Detectable?", "Δ vs V1.0 (pp)",
                 "P(Π<0) %", "E P5", "E median", "E mean"]
    s = sol.copy()
    s["P(E<W) %"] = pct(s["p_wage_below"])
    s = s[["fix", "P(E<W) %", "d_v1_pp", "d_v1_se"]]
    s.columns = ["Fix added alone to V1.0", "P(E<W) %", "Δ vs V1.0 (pp)", "SE (pp)"]
    l = loo.copy()
    l["P(E<W) %"] = pct(l["p_wage_below"])
    l = l[["fix_reverted", "P(E<W) %", "d_full_pp", "d_full_se"]]
    l.columns = ["Fix reverted/removed from full V2.0", "P(E<W) %", "Δ vs full (pp)", "SE (pp)"]
    full = loo.attrs["full"]
    sd = pct(seeds["p_wage_below"].std(ddof=1))
    sn = sens.copy()
    sn["P(E<W) %"] = pct(sn["p_wage_below"])
    sn["primary"] = np.where(sn["is_primary_assumption"], "yes", "")
    sn = sn[["parameter", "value", "primary", "P(E<W) %", "E_median"]]
    sn.columns = ["Parameter varied", "Value", "Primary", "P(E<W) %", "E median"]
    mm = mode_frac_matching_v1_mean()
    def _prov(k):
        """Provenance label for parameter `k`, matching slash-joined PROVENANCE keys by prefix."""
        if k in PROVENANCE:
            return PROVENANCE[k]
        return next((p for pk, p in PROVENANCE.items() if "/" in pk and k.startswith(pk.split("_")[0] + "_")), "")
    assume_rows = "\n".join(f"| {k} | {v} | {_prov(k)} |" for k, v in ASSUME.items())
    return f"""# Project AURUM — Ablation study (V1.0 → V2.0)

*Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · N = {N:,} draws per configuration · seed {seed} · engine {E2.ENGINE_VERSION}*

**Read this first.** Everything below is a **scenario result under the assumptions listed at the end**. It is not a finding
about any real mining group. All parameters are synthetic. This study compares fixed-timing
equations and does not evaluate the daily batch schedule extension. Headline metric: **P(E < W)**, the probability
that the labor return per person-hour is below the {BENCH['W']} PHP/hour wage benchmark (V1.0 reported 1 − this as
"Probability of Real Economic Sustainability").

## 1. Forward chain (cumulative; order matters)
{md_table(f, {"P(E<W) %": ".1f", "±2 SE": ".2f", "Δ vs prev (pp)": "+.1f", "Δ vs V1.0 (pp)": "+.1f", "P(Π<0) %": ".1f", "E P5": ".1f", "E median": ".1f", "E mean": ".1f"})}
"Detectable?" compares each step with the seed-to-seed noise (section 4). A step that is "within noise" does not move
**this** metric in **this** scenario; it may still matter for tails or other scenarios (see the P5 and P(Π<0) columns).

## 2. Each fix alone, added to V1.0 (order-free)
{md_table(s, {"P(E<W) %": ".1f", "Δ vs V1.0 (pp)": "+.1f", "SE (pp)": ".2f"})}

## 3. Leave-one-out from the full model (order-free, ranked by size of effect)
Full fixed-timing model: P(E<W) = {pct(full['p_wage_below']):.1f}%.
{md_table(l, {"P(E<W) %": ".1f", "Δ vs full (pp)": "+.1f", "SE (pp)": ".2f"})}

## 4. Noise check (full model, 5 seeds)
P(E<W) ranges {pct(seeds['p_wage_below'].min()):.1f}%–{pct(seeds['p_wage_below'].max()):.1f}% across seeds
(SD {sd:.2f} pp). Treat differences smaller than about {2 * sd:.1f} pp as noise.

## 5. Sensitivity of the full model to the ASSUMED parameters (one at a time)
{md_table(sn, {"P(E<W) %": ".1f", "E median": ".1f"})}

## 6. Diagnostic: V1.0's equation on un-converted group inputs
If the group-level inputs (group yield, group costs, miner share {ASSUME['s_group']}) had been fed to V1.0's equation
**without** the crew-size conversion, V1.0 would give P(E<W) = {pct(diag['p_wage_below']):.1f}%
(mean E = {diag['E_mean']:.0f} PHP/h) because H, not person-hours, would divide a group-level numerator. The manual
conversion changes the accounting unit. Both parameter sets here are synthetic examples.

## 7. What this does and does not show
* It shows which V1.0 flaws are **large** and which are **small** under these assumptions, and in which direction each moves the answer.
* **The size of the PERT-yield step is mostly the assumed mode, not a measured V1.0 error.** With the mode at {mm:.0%} of the
  min–max range, Beta-PERT has the same mean yield as V1.0, and that step shrinks toward zero (section 5, parameter mode_frac).
  The mode is an illustrative input; report its sensitivity range.
* Stochastic costs and dependence barely move P(E<W) here because costs are small next to revenue and the assumed spreads are
  modest; they remain required by the specification and are verified in the test suite.
* The absolute level of P(E<W) depends strongly on the assumed crew size (see section 5) and on synthetic yield assumptions.
* S3 bundles V1.0's lognormal with its heuristic jackpot; S4 changes the mean price as well as the shape (the lognormal with
  min = P5 and max = P99 has a lower mean than a mid-mode PERT on the same bounds). Both are real V1.0 effects of the hidden
  calibration, but they are not pure "shape only" changes.
* Fixes that cannot move this metric by design (separating the three thresholds, renaming outputs, input guards) are not ablated;
  they are verified in the test suite instead.
* No empirical validation is implied (Domain G needs Phase 2 field data).

## Assumed / synthetic parameters used
| Parameter | Value | Provenance |
|---|---|---|
{assume_rows}

Synthetic V1.0 benchmark: {json.dumps(BENCH)}
"""


# --------------------------------------------------------------------------------------
def main(argv=None) -> None:
    """CLI entry point: run all four ablation views and write CSVs, plots, report and manifest."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--n", type=int, default=500_000, help="draws per configuration")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="ablation_out")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    fwd = forward_chain(a.n, a.seed)
    sol = solo(a.n, a.seed)
    loo = leave_one_out(a.n, a.seed)
    seeds = seed_check(min(a.n, 200_000), range(1, 6))
    sens = sensitivity(min(a.n, 200_000), a.seed)
    diag = v1_literal_diagnostic(a.n)

    fwd.to_csv(f"{a.out}/ablation_forward.csv", index=False)
    sol.to_csv(f"{a.out}/ablation_solo.csv", index=False)
    loo.to_csv(f"{a.out}/ablation_leave_one_out.csv", index=False)
    seeds.to_csv(f"{a.out}/ablation_seed_check.csv", index=False)
    sens.to_csv(f"{a.out}/ablation_sensitivity.csv", index=False)
    plot_forward(fwd, f"{a.out}/ablation_forward.png")
    plot_loo(loo, f"{a.out}/ablation_leave_one_out.png")
    with open(f"{a.out}/ablation_report.md", "w", encoding="utf-8") as fh:
        fh.write(build_report(fwd, sol, loo, seeds, sens, diag, a.n, a.seed))
    with open(f"{a.out}/ablation_manifest.json", "w") as fh:
        json.dump({"engine_version": E2.ENGINE_VERSION, "n": a.n, "seed": a.seed, "benchmark": BENCH,
                   "assumptions": ASSUME, "provenance": PROVENANCE,
                   "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "numpy": np.__version__}, fh, indent=2)

    print(fwd[["step", "p_wage_below", "d_prev_pp", "d_v1_pp"]].round(4).to_string(index=False))
    print(f"\nFiles written to ./{a.out}/")


if __name__ == "__main__":
    main()
