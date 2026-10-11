# Project AURUM — Engine V2.1 (draft)

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
* Verified for V2.1: real Streamlit AppTest initial/default runs, fractional calendar duration,
  required reserve-policy errors, all eight sharing/location/priority combinations, purchased-equipment
  depreciation reserves, and invalid-yield clearing. A shared Chromium smoke check exercised the
  default run and a custom common reserve with household-first allocation and 2.5 calendar days.
* Stakeholder usability and exhaustive browser/device coverage remain unverified.
* Scenario results only: every screen says so; inputs tagged 🟠 are assumptions.

## V2.1 accounting update

`Pi` and `E` retain their economic-profit meanings, including depreciation. `Y_HH` and
`P_fail_household_adequacy` also retain their economic-profit basis and are labeled explicitly.
New cash outputs exclude purchased-equipment depreciation; equipment rentals remain cash expenses.
The frozen V1 code and the distributions/cost copula are unchanged.

### Calendar period

Set `calendar_days_per_cycle` to the positive elapsed household expense/income period covered by
this cycle, including idle time; fractional days are allowed. `D_c` remains the working-shift count
for person-hours and rentals. Old inputs omitting calendar duration use `D_c`; a warning and the
manifest's `calendar_days_assumed_from_shifts` flag disclose that fallback. Current presets specify
one calendar day explicitly. Household count remains independent of crew size when supplied.

### Replacement reserve (one cycle only)

`reserve_mode` is `none` (default), `match_depreciation`, or `custom`. When enabled, explicitly set
`reserve_location` to `before_sharing` or `after_sharing`, and `reserve_priority` to
`household_first` or `reserve_first`. A custom target is `reserve_custom_per_cycle` (PHP >= 0).
Matching depreciation targets total depreciation for a common fund, or effective miner depreciation
for a miner fund (direct allocation under gross sharing, proportional allocation under net sharing).

Let R be gross revenue, C the total cash operating cost, M the effective miner cash cost,
s the miner revenue share, B the household requirement, and Q the reserve target. Miner cash
before the reserve is X = s*R - M. Existing gross/net rules determine M and economic profit.

| Fund | Available funding A | Miner burden per peso funded b |
|---|---|---|
| Before sharing (common) | max(R - C, 0) | s |
| After sharing (miner) | max(X, 0) | 1 |

Reserve-first funds F = min(Q, A). Household-first funds
F = min(Q, A, max((X - B)/b, 0)). Miner cash after reserve is X - b*F;
shortfall is Q - F. Common funding reduces the revenue base available for sharing under either
contract; the cash-cost allocation itself is unchanged. Household-first protects only the modeled
miner households, not financier households. Losses remain visible and never finance reserve deposits.
Depreciation is never deducted again from cash. There is no reserve balance, borrowing, interest,
equipment purchase payment, or accumulated debt in this single-cycle model.

Arrays and summaries expose cash before/after reserve, planned/funded/shortfall reserve amounts,
and the miner contribution. Reserve amounts refer to the selected common/miner fund, while the
miner contribution reports its effect on miner cash. New probabilities show household cash shortfall
before/after reserve, reserve underfunding, and meeting both targets. With no reserve, the reserve
target is zero and the joint probability equals household cash adequacy. Every reported probability
participates in MCSE stopping and the staged convergence table. CSVs and manifests retain the old
columns/keys and add the new outputs and policy. The engine/dashboard version is `2.1.0-draft`.

### Validation and tail statistics

The engine rejects non-finite or nonnumeric active inputs, boolean/fractional counts, malformed
hours matrices, and invalid run settings. `n_max < n_min` remains an intentional capped run that
reports non-convergence; all limits must be positive integers. Unexpected non-finite arithmetic
raises `InputError` before event counting. MCSE is numerical simulation error, not model uncertainty.
Expected shortfall averages exactly the lowest 5% of empirical probability mass, fractionally
weighting the boundary observation rather than including every tie at the fifth percentile.

Checks: `python -m unittest discover -s tests -v` and `python tests/streamlit_smoke.py`.
