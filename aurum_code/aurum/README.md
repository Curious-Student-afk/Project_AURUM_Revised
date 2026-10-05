# Project AURUM — Engine V2.0 (draft)

| File | Role |
|---|---|
| `legacy_v1.py` | **Frozen** V1.0 simulation function (do not edit). Baseline for the regression test and Chapter 3.1. |
| `engine_v2.py` | V2.0 engine: zero-aware hurdle + PERT + truncated-Pareto jackpot, group-level cycle accounting, three separate thresholds, Spearman-copula costs, MCSE stopping rule, run manifest. No Streamlit. |
| `scenarios.py` | Named input sets (pilot-provisional baseline; synthetic Baseline / High-Cost / Low-Purity). Every value carries a provenance tag. |
| `tests/test_engine_v2.py` | 44 acceptance tests mapped to the expert report. |
| `tests/test_ablation.py` | 18 tests that the ablation switches do exactly what they claim (92 tests in total with the dashboard tests). |
| `ablation.py` | V1.0 -> V2.0 ablation study: forward chain, each fix alone, leave-one-out, seed-noise check, sensitivity to assumed parameters, V1.0 diagnostic, figures and a Markdown report. |
| `ablation_out/` | Example outputs (N = 500,000, seed 42). Re-run to reproduce. |
| `run_demo.py` | Offline demo; writes run manifests to `runs/`. |
| `app.py` | Streamlit dashboard (V2.0 default, frozen V1.0 for comparison). Thin layer over the modules below. |
| `dashboard_logic.py` | Everything the dashboard does except draw widgets: inputs mapping, provenance tags, tables, figures, exports, wording. No Streamlit import, so it is unit-tested. |
| `tests/test_dashboard.py`, `tests/fake_streamlit.py` | Dashboard logic tests plus an offline smoke test of `app.py` against a fake Streamlit. |
| `requirements.txt` | numpy, scipy, pandas, matplotlib, streamlit. |

## Run (offline, Python 3.11+, needs numpy, scipy, pandas, matplotlib, streamlit)
```
python -m unittest discover -s tests -v
python run_demo.py
python ablation.py --n 500000 --seed 42 --out ablation_out
streamlit run app.py
```

## Read before using any number
* Outputs are **scenario results under stated assumptions**, not findings about Pantukan miners.
* Not collected in the pilot (so ASSUMED): crew size, costs, p0, pi_J, PERT mode, dependence, tribute base.
* Pilot yield bounds (a, b) are means of respondents' typical min/max: provisional. Jackpot inputs rest on n = 3.
* Set `W_opp_source` / `W_opp_date` before reporting (the engine warns until you do).
* Dependence is modelled among cost categories only; yield–cost dependence is a stated limitation.
* Passing tests = **verification** (code matches specification). Empirical **validation** (Domain G) needs Phase 2 data.

## Reading the ablation (important)
* The size of the "Beta-PERT yield" step is mostly the **assumed mode** (no 'usual' value was collected), not a measured V1.0 error.
  With the mode at about 19% of the min-max range, PERT has the same mean yield as V1.0 and that step nearly vanishes.
* Crew size (assumed 5) moves the headline probability more than any single fix. Report the full range.
* Stochastic costs and cost dependence do not move P(E<W) in this scenario; they remain required by the specification.
* Change the assumed values in one place: the `ASSUME` dictionary at the top of `ablation.py`.

## Dashboard: what was and was not verified
* Verified offline: the input mapping (UI values -> engine inputs gives identical simulations), provenance tags, tables,
  figures, exports, wording rules (no "debt trap", no exit/greenlight advice), the legacy path, and that `app.py` runs end to end
  against a fake Streamlit that enforces common argument rules (mixed int/float, duplicate keys, out-of-range defaults).
* NOT verified: the real Streamlit rendering. Streamlit was not available where this was written. Run `streamlit run app.py`,
  and check each tab, the 🟠/🟢/🟣 tags on the sidebar labels, and that fractional inputs (e.g. s_m = 0.658) keep their
  decimals. Send the first error or a screenshot back and fix it before submission.
* Scenario results only: every screen says so; inputs tagged 🟠 are assumptions.
