"""
run_matrix.py -- Project AURUM results matrix runner (M0-M10, optional M11 ablation).

Imports the FROZEN engine files unchanged (engine_v2, scenarios, dashboard_logic, ablation) and never edits them.
Place this file in the same folder as engine_v2.py and run from that folder:

    python run_matrix.py --list                 # print the run plan only
    python run_matrix.py --quick                # smoke test: 20,000 draws per run
    python run_matrix.py                        # full matrix (engine convergence rules, N cap 200,000)
    python run_matrix.py --blocks M0,M2         # subset of blocks
    python run_matrix.py --resume               # skip runs whose result row already exists in --out
    python run_matrix.py --ablation             # also run ablation.py (M11)

Outputs (in --out, default matrix_out/):
    matrix_results.csv        one row per run: probabilities+MCSE, return statistics, three thresholds, cash/reserve,
                              convergence status and reasons, input and code hashes
    matrix_inventory.json     run inventory + code identity, data-freeze values, package versions, freeze check
    manifests/<run_id>.json   full engine manifest for each run (inputs, seed, hashes, diagnostics)
    rows/<run_id>.json        per-run result row (used for --resume)
    M0/                       thresholds, returns, cash, timing, stage, rank-sensitivity and benchmark-sensitivity
                              tables, plus the first 5,000 cycles (CSV schema 2)
    M2_heatmap.png / .csv     yield-scale x p0 tipping-point map (headline figure)

All results are SCENARIO results under the assumptions stated in the constants below. They are not
predictions for any real operation and not empirical validation.
"""
from __future__ import annotations

import argparse
import dataclasses as dc
import json
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import engine_v2 as E            # noqa: E402
import dashboard_logic as DL     # noqa: E402
from scenarios import batch_synthetic, _scale_costs   # noqa: E402

MATRIX_VERSION = "1.0"
EXPECTED_GIT = "d74c7ccbd0af1a69d122cf22b0fa7fc933357418"

# --------------------------------------------------------------------------------------
# Data-freeze values (edit here, not in the engine). Everything below is recorded in the inventory.
# --------------------------------------------------------------------------------------
DATA_FREEZE = "2026-10-19"
S_REF = 8460.94          # PHP/g 24K; midpoint of the user-supplied BSP range 8,451.88-8,470.00 (replace with the exact rate)
S_SOURCE = "BSP official gold buying rate, 24K, 2026-10-19 (range supplied by the project author; not independently verified)"
SPOT_LOW, SPOT_HIGH = 7832.35, 9951.45   # 6-month 24K spot range as of 2026-07-31 (goldpricedata.com); RE-PULL to the freeze date
F_PERT = (0.60, 0.70, 0.80)              # farmgate price as fraction of S (fineness x informal discount) -- ASSUMED
W_OPP = 67.5                             # PHP/person-hour: PHP 540/day non-agriculture / 8 h, Wage Order RB XI-24, eff. 2026-09-01
W_SOURCE = "Wage Order No. RB XI-24 (RTWPB XI): non-agriculture PHP 540/day / 8 h"
W_DATE = "2026-09-01"
HH_NEED = 481.0          # PHP/household/day: PSA 2025 poverty threshold PHP 14,634/month, family of five, / 30.4 days
FOOD = (204.8, 256.0, 307.2)   # PHP/calendar day for the crew: mode = 4 x PHP 64 (national food threshold), +/-20% (ASSUMED spread)
YIELD_SCALES = (1.0, 0.75, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15)
P0_GRID = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40)
RESERVE_TARGET = 1000.0  # PHP/batch, ILLUSTRATIVE ONLY

NUMERIC_ROW_KEYS = ("E_mean", "E_median", "E_P5", "E_P95", "E_expected_shortfall_5pct")


# --------------------------------------------------------------------------------------
# Input builders
# --------------------------------------------------------------------------------------
def base_inputs(S: float = S_REF, f=F_PERT) -> E.Inputs:
    """Baseline-External: the frozen Baseline preset with tier-A values and the agreed food rate."""
    b = batch_synthetic()
    costs = [dc.replace(c, pert=E.PERT(*FOOD)) if c.name == "food" else c for c in b.costs]
    provenance = {**b.provenance, "W_opp": "external", "hh_need_per_household_day": "external",
                  "price": "assumed", "food_rate": "assumed"}
    meta = {"matrix_version": MATRIX_VERSION, "data_freeze": DATA_FREEZE, "S_php_per_g": S, "S_source": S_SOURCE,
            "price_fraction_f_pert": list(f), "W_opp_basis": W_SOURCE,
            "household_basis": "PSA 2025 poverty threshold, family of five, per 30.4 days",
            "food_basis": "4 workers x PHP 64/day national food threshold, +/-20% assumed spread"}
    return dc.replace(b, costs=costs, price=E.PERT(f[0] * S, f[1] * S, f[2] * S), W_opp=W_OPP,
                      W_opp_source=W_SOURCE, W_opp_date=W_DATE, hh_need_per_household_day=HH_NEED,
                      provenance=provenance, metadata=meta)


def with_yield_scale(inp: E.Inputs, k: float) -> E.Inputs:
    """Scale the ordinary yield by k; jackpot threshold and cap stay at 1.5x and 3x the (scaled) ordinary maximum."""
    y = inp.yield_ord
    scaled = E.PERT(y.a * k, y.m * k, y.b * k)
    jack = E.Jackpot(1.5 * scaled.b, 3.0 * scaled.b, inp.jackpot.shape) if inp.jackpot is not None else None
    return dc.replace(inp, yield_ord=scaled, jackpot=jack)


def with_rho(inp: E.Inputs, rho) -> E.Inputs:
    """rho=None -> independent expense rates; otherwise Gaussian copula with every pairwise Spearman = rho."""
    if rho is None:
        return dc.replace(inp, dependence="independent")
    n = len(inp.stochastic_cost_items())
    m = np.full((n, n), float(rho))
    np.fill_diagonal(m, 1.0)
    return dc.replace(inp, dependence="copula", rho_spearman=m.tolist())


def mode_fractions(inp: E.Inputs):
    """Base fraction, the two sensitivity fractions, and the V1.0 mean-matching fraction (from ablation.py)."""
    y = inp.yield_ord
    base_frac = round((y.m - y.a) / (y.b - y.a), 4)
    fracs = [0.25, base_frac, 0.60]
    try:
        import ablation as AB
        fracs.append(round(AB.mode_frac_matching_v1_mean(), 4))
    except Exception as exc:        # ablation import failure should not block the other blocks
        warnings.warn(f"V1.0 mean-matching mode fraction unavailable: {exc}")
    return sorted(set(fracs))


# --------------------------------------------------------------------------------------
# Run plan
# --------------------------------------------------------------------------------------
def build_plan(spot_ref: float, spot_low: float, spot_high: float):
    """Return M0-M10 run specifications using reference, low, and high spot prices."""
    base = base_inputs(spot_ref)
    plan = []

    def add(block, rid, label, inp, **factors):
        """Append a run with a block-prefixed ID, inputs, and sensitivity factors."""
        plan.append({"block": block, "run_id": f"{block}_{rid}", "label": label, "inp": inp, "factors": factors})

    add("M0", "baseline", "Baseline-External, gross, s_m=0.50", base)

    for contract in ("gross", "net"):                                           # M1
        for sm in (0.50, 0.60, 0.65):
            add("M1", f"{contract}_sm{sm:.2f}", f"{contract} contract, s_m={sm}",
                dc.replace(base, tribute_base=contract, s_m=sm), contract=contract, s_m=sm)

    for k in YIELD_SCALES:                                                      # M2
        for p0 in P0_GRID:
            add("M2", f"k{k}_p0{p0:.2f}", f"yield scale {k}, p0={p0}",
                dc.replace(with_yield_scale(base, k), zero=E.ZeroYield(p0=p0)), yield_scale=k, p0=p0)

    for frac in mode_fractions(base):                                           # M3
        y = base.yield_ord
        add("M3", f"mode{frac}", f"ordinary-yield mode at {frac} of range",
            dc.replace(base, yield_ord=E.PERT(y.a, y.a + frac * (y.b - y.a), y.b)), mode_fraction=frac)

    for n_c in (3, 4, 5, 6, 8):                                                 # M4
        add("M4", f"nc{n_c}", f"crew size {n_c}", dc.replace(base, n_c=n_c), n_c=n_c)

    f_cases = {"f0.60": (0.60,) * 3, "fPERT": F_PERT, "f0.80": (0.80,) * 3}     # M5
    for s_label, S in (("Slow", spot_low), ("Sref", spot_ref), ("Shigh", spot_high)):
        for f_label, f in f_cases.items():
            add("M5", f"{s_label}_{f_label}", f"S={S:,.2f}, f={f_label}", base_inputs(S, f),
                S=S, S_case=s_label, f_case=f_label)

    for hh in (320.0, 481.0, 800.0):                                            # M6
        for nm in (0.0, 100.0, 200.0):
            add("M6", f"hh{int(hh)}_nm{int(nm)}", f"household need {hh}, outside income {nm}",
                dc.replace(base, hh_need_per_household_day=hh, nonmining_income_per_household_day=nm),
                hh_need=hh, nonmining_income=nm)

    for pj in (0.0, 0.01, 0.02, 0.05):                                          # M7
        for shape in (1.0, 1.5, 2.5):
            j = E.Jackpot(base.jackpot.threshold, base.jackpot.cap, shape)
            add("M7", f"pj{pj}_shape{shape}", f"jackpot prob {pj}, shape {shape}",
                dc.replace(base, pi_J=pj, jackpot=j), pi_J=pj, jackpot_shape=shape)

    for scale in (0.5, 1.0, 2.0, 3.0):                                          # M8
        for rho in (None, 0.3, 0.6):
            tag = "indep" if rho is None else f"rho{rho}"
            add("M8", f"cost{scale}_{tag}", f"cost scale {scale}, dependence {tag}",
                with_rho(_scale_costs(base, scale), rho), cost_scale=scale, rho=("independent" if rho is None else rho))

    t = base.batch_timing                                                       # M9
    works = {"w1-2-3": E.PERT(1, 2, 3), "w2-3-5": E.PERT(2, 3, 5), "w4-6-10": E.PERT(4, 6, 10)}
    for idle_scale in (0.0, 1.0, 2.0):
        for w_label, w in works.items():
            idle = E.PERT(t.idle_days.a * idle_scale, t.idle_days.m * idle_scale, t.idle_days.b * idle_scale)
            add("M9", f"idle{idle_scale}_{w_label}", f"idle scale {idle_scale}, work days {w_label}",
                dc.replace(base, batch_timing=dc.replace(t, work_days=w, idle_days=idle)),
                idle_scale=idle_scale, work_days=w_label)

    for contract in ("gross", "net"):                                           # M10
        add("M10", f"{contract}_none", f"{contract}, no reserve", dc.replace(base, tribute_base=contract),
            contract=contract, location="none", priority="none")
        for loc in ("before_sharing", "after_sharing"):
            for pri in ("household_first", "reserve_first"):
                add("M10", f"{contract}_{loc}_{pri}", f"{contract}, reserve {loc}, {pri}",
                    dc.replace(base, tribute_base=contract, reserve_mode="custom",
                               reserve_custom_per_cycle=RESERVE_TARGET, reserve_location=loc, reserve_priority=pri),
                    contract=contract, location=loc, priority=pri, reserve_target=RESERVE_TARGET)
    return plan


# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------
def result_row(res: E.Result, item: dict, wall: float) -> dict:
    """Return a JSON-compatible row of run factors, statistics, diagnostics, and elapsed seconds."""
    s, a, m = res.summary, res.arrays, res.manifest
    row = {"run_id": item["run_id"], "block": item["block"], "label": item["label"],
           **{f"factor_{k}": v for k, v in item["factors"].items()},
           "mean_ordinary_yield_g": item["inp"].yield_ord.mean, "N": s["N"], "converged": s["converged"],
           "probability_precision_met": s["probability_precision_met"], "return_precision_met": s["return_precision_met"],
           "mixture_support_met": s["mixture_support_met"], "convergence_reasons": s["convergence_reasons"],
           "stopping_reason": m["stopping_reason"], "observed_zero_cycle_share": s["observed_zero_cycle_share"]}
    for key in E.PROB_KEYS:
        row[key], row["MCSE_" + key] = s[key], s["MCSE_" + key]
    for key in NUMERIC_ROW_KEYS:
        row[key], row["error_" + key] = s[key], s.get("error_" + key)
    for key in ("Y_BE", "Y_W", "Y_HH"):
        p5, p50, p95 = np.percentile(a[key], [5, 50, 95])
        row[f"{key}_p5"], row[f"{key}_median"], row[f"{key}_p95"] = float(p5), float(p50), float(p95)
    for key in ("cash_before_reserve", "cash_after_reserve", "reserve_planned", "reserve_funded", "reserve_shortfall"):
        row[key + "_mean"] = s[key + "_mean"]
    row.update(inputs_sha256=m["inputs_sha256"], engine_sha256=m["engine_sha256"], git_revision=m["git_revision"],
               git_dirty=m["git_dirty"], seed=m["run_config"]["seed"], wall_seconds=round(wall, 2),
               engine_warnings=" | ".join(res.warnings))
    return E._jsonable(row)


def save_m0_extras(res: E.Result, inp: E.Inputs, out: Path) -> None:
    """Write baseline diagnostic tables and up to 5,000 cycles as CSVs under out/M0."""
    d = out / "M0"
    d.mkdir(exist_ok=True)
    DL.thresholds_table(res).to_csv(d / "thresholds.csv", index=False)
    DL.return_table(res).to_csv(d / "returns.csv", index=False)
    DL.cash_table(res).to_csv(d / "cash_reserve.csv", index=False)
    DL.timing_table(res).to_csv(d / "timing.csv", index=False)
    DL.risk_table(res).to_csv(d / "risk_probabilities.csv", index=False)
    DL.benchmark_sensitivity(res, inp).to_csv(d / "benchmark_sensitivity_postprocessed.csv", index=False)
    res.stage_table.to_csv(d / "stage_table.csv", index=False)
    pd.Series(res.rank_sensitivity, name="spearman_with_E").to_csv(d / "rank_sensitivity.csv", header=True)
    (d / "cycles_first_5000_schema2.csv").write_bytes(DL.export_csv(res, 5000))


def heatmap(df: pd.DataFrame, out: Path) -> None:
    """Save M2 probability grids as CSVs and a PNG heatmap; skip when no M2 rows exist."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m2 = df[df["block"] == "M2"]
    if m2.empty:
        return
    panels = (("P_labor_return_below_wage", "P(labor return below wage benchmark)"),
              ("P_fail_household_adequacy", "P(economic profit below household requirement)"))
    ks = sorted(m2["factor_yield_scale"].unique(), reverse=True)
    p0s = sorted(m2["factor_p0"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    for ax, (key, title) in zip(axes, panels):
        grid = m2.pivot(index="factor_yield_scale", columns="factor_p0", values=key).loc[ks, p0s]
        grid.to_csv(out / f"M2_heatmap_{key}.csv")
        im = ax.imshow(grid.values, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
        for i in range(len(ks)):
            for j in range(len(p0s)):
                v = grid.values[i, j]
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8, color="white" if v > .75 or v < .15 else "black")
        if grid.values.min() < .5 < grid.values.max():
            ax.contour(grid.values, levels=[0.5], colors="black", linewidths=1.6)
        means = m2.groupby("factor_yield_scale")["mean_ordinary_yield_g"].first()
        ax.set_yticks(range(len(ks)))
        ax.set_yticklabels([f"{k:g}  ({means[k]:.1f} g)" for k in ks])
        ax.set_xticks(range(len(p0s)))
        ax.set_xticklabels([f"{p:.0%}" for p in p0s])
        ax.set_xlabel("Zero-yield probability p0 (assumed)")
        ax.set_ylabel("Yield scale k  (mean ordinary yield, g/batch)")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle("Tipping-point map: scenario results under assumed inputs (black line = 50%)", fontsize=11)
    fig.savefig(out / "M2_heatmap.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Freeze check and inventory
# --------------------------------------------------------------------------------------
def freeze_check(expected: str, out_name: str) -> dict:
    """Report revision agreement and dirtiness, excluding runner, output, and bytecode paths.

    Verification succeeds only after both HEAD and status have been read.
    """
    info = {"expected_git": expected, "git_revision": None, "matches_expected": None,
            "other_dirty_files": None, "verified": False}
    try:
        git = lambda *a: subprocess.check_output(["git", *a], cwd=HERE, text=True, stderr=subprocess.DEVNULL, timeout=10)
        head = git("rev-parse", "HEAD").strip()
        status = [ln for ln in git("status", "--porcelain").splitlines()
                  if "run_matrix.py" not in ln and out_name not in ln and "__pycache__" not in ln]
        info.update(git_revision=head, matches_expected=(head == expected),
                    other_dirty_files=status, verified=True)
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def main(argv=None) -> None:
    """Parse CLI arguments to list or execute the matrix and export results and inventory.

    Read arguments from sys.argv when argv is None, and run M11 if requested.
    """
    ap = argparse.ArgumentParser(description="Project AURUM results matrix runner")
    ap.add_argument("--out", default="matrix_out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-cap", type=int, default=200_000, help="engine N cap (convergence rules still apply)")
    ap.add_argument("--quick", action="store_true", help="smoke test: exactly 20,000 draws per run, not for reporting")
    ap.add_argument("--blocks", default="", help="comma-separated block ids, e.g. M0,M2")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--ablation", action="store_true", help="also run ablation.py (M11)")
    ap.add_argument("--spot", type=float, default=S_REF, help="24K spot S in PHP/g on the freeze date")
    ap.add_argument("--spot-low", type=float, default=SPOT_LOW)
    ap.add_argument("--spot-high", type=float, default=SPOT_HIGH)
    ap.add_argument("--expect-git", default=EXPECTED_GIT)
    ap.add_argument("--force", action="store_true", help="continue even if the frozen-code check fails")
    a = ap.parse_args(argv)

    lo, hi = sorted((a.spot_low, a.spot_high))
    lo, hi = min(lo, a.spot), max(hi, a.spot)
    plan = build_plan(a.spot, lo, hi)
    wanted = {b.strip() for b in a.blocks.split(",") if b.strip()}
    unknown = wanted - {p["block"] for p in plan}
    if unknown:
        sys.exit(f"Unknown block id(s): {', '.join(sorted(unknown))}")
    if wanted:
        plan = [p for p in plan if p["block"] in wanted]
    if a.list:
        for p in plan:
            print(f"{p['run_id']:34s} {p['label']}")
        print(f"\n{len(plan)} runs")
        return

    out = Path(a.out)
    for sub in ("manifests", "rows"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    check = freeze_check(a.expect_git, out.name)
    print("Freeze check:", json.dumps(check))
    if (check["verified"] is not True
            or check["matches_expected"] is not True
            or check["other_dirty_files"]):
        msg = "Frozen-code check failed (Git state could not be verified, HEAD differs from expected, or files other than run_matrix.py are modified)."
        if not a.force:
            sys.exit(msg + " Re-run with --force to proceed anyway.")
        print("WARNING:", msg)
    if a.quick:
        print("QUICK MODE: 20,000 draws per run. Do not report these results.")

    run_settings = {
        "matrix_version": MATRIX_VERSION, "quick_mode": a.quick, "seed": a.seed, "n_cap": a.n_cap,
        "data_freeze": {"date": DATA_FREEZE, "S_php_per_g": a.spot, "S_source": S_SOURCE, "spot_low": lo, "spot_high": hi,
                        "spot_range_note": "6-month range as of 2026-07-31 unless overridden; re-pull to the freeze date",
                        "f_pert": list(F_PERT), "W_opp": W_OPP, "W_source": W_SOURCE, "W_date": W_DATE,
                        "household_need": HH_NEED, "food_pert": list(FOOD)},
    }
    run_kwargs = {"seed": a.seed, **({"n_fixed": 20_000} if a.quick else {"n_max": a.n_cap})}
    inventory_runs = []
    t_all = time.time()
    for i, item in enumerate(plan, 1):
        row_path, man_path = out / "rows" / f"{item['run_id']}.json", out / "manifests" / f"{item['run_id']}.json"
        if (
            a.resume
            and row_path.exists()
            and man_path.exists()
            and json.loads(row_path.read_text()).get("run_settings") == run_settings
        ):
            print(f"[{i}/{len(plan)}] {item['run_id']}: skipped (resume)")
            continue
        t0 = time.time()
        res = E.run(item["inp"], **run_kwargs)
        wall = time.time() - t0
        row = result_row(res, item, wall)
        row["run_settings"] = run_settings
        extras = {"run_id": item["run_id"], "block": item["block"], "factors": item["factors"], "matrix_version": MATRIX_VERSION,
                  "quick_mode": a.quick, "data_freeze": DATA_FREEZE}
        man_path.write_bytes(DL.export_manifest(res, extras))
        row_path.write_text(json.dumps(row, indent=1, allow_nan=False))
        if item["run_id"] == "M0_baseline":
            save_m0_extras(res, item["inp"], out)
        print(f"[{i}/{len(plan)}] {item['run_id']:32s} N={row['N']:>7,} converged={row['converged']!s:5s} "
              f"P(E<W)={row['P_labor_return_below_wage']:.3f}  {wall:.1f}s")

    rows = []
    for p in plan:
        rp = out / "rows" / f"{p['run_id']}.json"
        if rp.exists():
            saved_row = json.loads(rp.read_text())
            if saved_row.get("run_settings") != run_settings:
                continue

            rows.append(saved_row)
            inventory_runs.append({
                "run_id": p["run_id"],
                "block": p["block"],
                "label": p["label"],
                "manifest": f"manifests/{p['run_id']}.json",
                "inputs_sha256": saved_row["inputs_sha256"],
                "N": saved_row["N"],
                "converged": saved_row["converged"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(out / "matrix_results.csv", index=False)
    heatmap(df, out)
    unconverged = df[~df["converged"].astype(bool)][["run_id", "convergence_reasons"]] if len(df) else df
    unconverged.to_csv(out / "unconverged_runs.csv", index=False)

    inventory = {
        "matrix_version": MATRIX_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "quick_mode": a.quick, "seed": a.seed, "n_cap": a.n_cap, "blocks": sorted({p["block"] for p in plan}),
        "n_runs_planned": len(plan), "n_runs_with_results": len(rows), "n_unconverged": int(len(unconverged)),
        "data_freeze": run_settings["data_freeze"],
        "freeze_check": check, "code_identity": E.source_identity(),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                    "scipy": __import__("scipy").__version__},
        "runs": inventory_runs, "elapsed_seconds": round(time.time() - t_all, 1),
        "scope": "Scenario results under stated assumptions; computational verification, not empirical validation.",
    }
    (out / "matrix_inventory.json").write_text(json.dumps(E._jsonable(inventory), indent=1, default=str, allow_nan=False))
    print(f"\n{len(rows)} results -> {out}/matrix_results.csv ; unconverged: {len(unconverged)}")

    if a.ablation:
        import ablation as AB
        AB.main(["--n", "20000" if a.quick else "500000", "--seed", str(a.seed), "--out", str(out / "ablation")])


if __name__ == "__main__":
    main()
