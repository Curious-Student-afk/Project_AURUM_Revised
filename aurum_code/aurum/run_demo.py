"""
run_demo.py -- quick offline demonstration of engine_v2 on the synthetic test bench.

    python run_demo.py

Prints, for each profile: summary, staged-N stability table, rank-based sensitivity, and a
scenario grid over ASSUMED parameters (p0, ordinary-yield mode, crew size).
Writes one JSON run manifest per profile to ./runs/ (code version, inputs, seed, provenance).
All outputs are SCENARIO results under stated assumptions -- illustrative demonstrations only.
"""
import json
import os

import pandas as pd

import engine_v2 as E2
from scenarios import batch_synthetic, synthetic_profiles

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
os.makedirs("runs", exist_ok=True)

for name, inp in synthetic_profiles().items():
    r = E2.run(inp, seed=42)
    s = r.summary
    print(f"\n=== {name} ===  N = {s['N']:,}  converged = {s['converged']}")
    for key, label in E2.LABELS.items():
        print(f"  {label}: {s[key]:.3f}  (MCSE {s['MCSE_' + key]:.4f})")
    print(f"  E median / mean (PHP per person-hour): {s['E_median']:.1f} / {s['E_mean']:.1f}   "
          f"P5 {s['E_P5']:.1f}  P95 {s['E_P95']:.1f}  ES5% {s['E_expected_shortfall_5pct']:.1f}")
    print(f"  Median thresholds (g): break-even {s['Y_BE_median']:.3f} | wage {s['Y_W_median']:.3f} | "
          f"household {s['Y_HH_median']:.3f}")
    print("  Rank sensitivity (Spearman with E):", {k: round(v, 2) for k, v in r.rank_sensitivity.items()})
    for w in r.warnings:
        print("  WARNING:", w)
    with open(f"runs/manifest_{name.lower().replace(' ', '_')}.json", "w") as fh:
        json.dump(r.manifest, fh, indent=2)

print("\n--- Staged-N stability (Baseline) ---")
print(E2.run(batch_synthetic(), n_fixed=100_000).stage_table.round(4).to_string(index=False))

print("\n--- Scenario grid over ASSUMED parameters (Baseline) ---")
grid = E2.scenario_table(batch_synthetic(), p0=[0.0, 0.1, 0.3], mode_frac=[0.25, 0.4, 0.6], n_fixed=30_000)
print(grid.round(3).to_string(index=False))

print("\n--- Crew-size sensitivity (assumed n_c) ---")
print(E2.scenario_table(batch_synthetic(), n_c=[3, 4, 5, 6, 8], n_fixed=30_000)
      [["n_c", "P_negative_operating_return", "P_labor_return_below_wage", "P_fail_household_adequacy",
        "E_median"]].round(3).to_string(index=False))
