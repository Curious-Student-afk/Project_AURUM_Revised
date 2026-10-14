# Project AURUM — Engine V2.3 (draft)

One Monte Carlo iteration represents a complete production batch. All presets are independently
constructed **synthetic demonstrations**, not measured mining conditions. Tests verify software
behavior; empirical validation requires independently obtained observations.

## Draft release 2.3.0

This release corrects zero-yield validation and physical-assay exports, supports independent batch
households and outside income, and adds deterministic measurement conversions with provenance.
Declared cost coverage prevents overlapping expense components. Aggregate gross/net accounting
and full attendance by the active batch crew remain the supported economic model.

Convergence now checks return errors, stability and productive-component samples alongside
probability precision. Wage/cutoff comparisons reuse existing draws. CSV schema 2 is the default;
readers requiring the historical `purity_q` field must explicitly select legacy schema 1.
Strict precision settings may leave capped runs unconverged; their results include the reasons.

## Run locally

Python 3.11+ is required. From this directory:

```bash
python -m pip install -r requirements.lock
python -m unittest discover -s tests -v
python tests/streamlit_smoke.py
python run_demo.py
streamlit run app.py
```

`engine_v2.py` contains pure NumPy/SciPy sampling and accounting. `dashboard_logic.py` maps inputs,
formats results, and exports CSV/JSON without importing Streamlit. `app.py` draws the dashboard.
`scenarios.py` supplies synthetic examples. `legacy_v1.py` retains the V1 numerical equations for
comparison only. `ablation.py` compares fixed-timing equations using independent synthetic inputs;
it does not evaluate the daily batch schedule extension.

To generate a new synthetic ablation report:

```bash
python ablation.py --n 500000 --seed 42 --out ablation_out
```

Generated reports, run manifests, and bytecode are excluded from Git. Saved reports must be regenerated
when inputs or code change. requirements.lock pins the tested CPython 3.11/Linux environment, including transitive dependencies.
requirements.txt lists the direct dependencies; use the lock for reproducible installation. Each manifest records runtime versions.

## Timing and production

`Inputs.batch_timing` accepts `BatchTiming(work_days, idle_days, shift_counts, hours_by_shifts)`.

- Working days use standard Beta-PERT (shape weight 4), rounded by `floor(draw + 0.5)`.
  Bounds and mode are whole numbers from 1 to 3650. This computational limit bounds scheduling work.
- Idle days use independent continuous nonnegative Beta-PERT. Calendar days equal working plus idle days.
- `shift_counts` contains three nonnegative integer counts for 1, 2, and 3 shifts, totaling ten working days.
  Each working day independently draws a categorical shift count using these frequencies.
- `hours_by_shifts` maps 1, 2, and 3 to conditional PERTs. One duration is drawn per day and applies to
  every shift that day. All active contributors attend. Batch mode does not model partial attendance or changing membership. Bounds are at least 0.5 and at most
  `min(16, 24 / shift_count)` hours. Dashboard bounds use 0.25-hour increments; draws remain continuous.
- Total shifts are summed over days. Operating hours sum scheduled shift durations, without multiplying
  by crew size. Person-hours equal crew size times operating hours. No worker rotation is modeled.
- Yield, zero-yield counts, jackpot frequency and capacity are elicited per completed batch. Batch yield
  is never multiplied by shift count. Jackpot probability is conditional on a productive batch; its
  magnitude uses a truncated Pareto with a lower threshold, cap, and positive shape.
- Batch yield, valuation, and timing are independent in this model. Timing has a separate seeded random
  stream. Results are reproducible given inputs, seed, batching configuration, code, and runtime versions.

Farmgate pricing values recovered gold directly (`q=1`, a valuation convention). Reference pricing
uses physical purity (`Inputs.purity`) and reports fine gold mass. Both price and reference purity can use bounded PERT.
An optional farmgate assay uses `Inputs.physical_purity`; missing assay stays unknown. Supplying it never
changes farmgate revenue, costs, or timing draws. Stochastic assay has a separate seeded stream.
Reference mode rejects the farmgate-only `physical_purity` field; provide its physical assay through `purity`.

## Expenses and dependence

`CostItem.basis` specifies the units of its fixed amount or PERT rate:

| Basis | Exposure |
|---|---|
| `per_batch` (default) | 1 |
| `per_shift` | Total scheduled shifts |
| `per_operating_hour` | Sum of scheduled shift durations |
| `per_working_day` | Rounded active days |
| `per_calendar_day` | Active plus idle days |

All rates describe the whole operation/crew, not an individual worker. Each rate is sampled once per
batch, then multiplied by its exposure. Default categories are fuel/hour, milling/batch,
explosives/working day, transport/batch, and food/calendar day. Additional fixed overhead is optional.
`CostItem.components` declares covered components (default: the item name). Identifiers are compared
case-insensitively after trimming; duplicate coverage across or within items is rejected. Nonzero grouped
UI overhead requires identifiers. This detects overlap in declared coverage, not undisclosed reporting errors.
Gross sharing uses each category's miner allocation; net sharing deducts **all** operation costs before
sharing, so category miner allocations do not change that contract. Reimbursements are not modeled.
Worksite food is additional consumption excluded from the household benchmark.

The existing Gaussian copula applies to expense rates before exposure scaling. The Spearman matrix
is ordered by stochastic cost items; the dashboard labels each pair. It must be symmetric with a
unit diagonal, and its Gaussian conversion must be positive definite. Degenerate PERT rates remain
in that order, but have no measurable rank correlation. Realized expense correlations can differ
from the rate targets because expenses also share time drivers. No rank reordering is performed.

Rented equipment costs its per-shift rental rate times sampled shifts. Purchased equipment has
per-batch depreciation `(acquisition - salvage) / life_cycles * utilization`. Depreciation is an
economic cost and is excluded from cash costs. Equipment is included exactly once.

## Three adequacy indicators

Let `C_eff` be directly allocated miner costs under gross sharing, or the miner share of all costs
under net sharing. Economic profit includes depreciation:

- `Pi = s_m * Y * q * P - C_eff` and `E = Pi / L`.
- Hourly wage adequacy compares `E` with `W_opp` (PHP/person-hour).
- Household economic adequacy compares `Pi` with
  `B_HH = households * calendar_days * max(daily_need - nonmining_income, 0)` in both modes.
  `n_households=None` defaults to active crew size; explicit zero means zero households.
  Need and outside income are PHP per household/calendar day. Additional worksite food is excluded.
- Household cash adequacy compares cash before and after reserve with the same draw's `B_HH`.

The three yield thresholds remain distinct:

- Operating break-even: `Y_BE = C_eff / (q * P * s_m)`.
- Opportunity wage: `Y_W = (C_eff + W_opp * L) / (q * P * s_m)`.
- **Household economic-adequacy yield threshold**: `Y_HH = (C_eff + B_HH) / (q * P * s_m)`.

Benchmarks are normative inputs. Cite their source and effective date before using them in reporting.
Provenance distinguishes synthetic demonstrations, stated assumptions, external sources, and user entries.

## Single reserve fund

`reserve_mode` is `none`, `match_depreciation`, or `custom` (which can represent an emergency fund).
An enabled fund requires explicit `reserve_location` (`before_sharing` or `after_sharing`) and
`reserve_priority` (`household_first` or `reserve_first`). There is no second fund, accumulated
balance, borrowing, interest, equipment purchase payment, or accumulated debt.

Let `R` be revenue, `C` cash operating cost, `M` effective miner cash cost, `s` miner share,
`B` household need, and `Q` reserve target. Miner cash before reserve is `X = s*R - M`.

| Fund | Available funding A | Miner burden b per peso funded |
|---|---|---|
| Common, before sharing | `max(R-C, 0)` | `s` |
| Miner, after sharing | `max(X, 0)` | `1` |

Reserve-first funds `F=min(Q,A)`. Household-first funds `F=min(Q,A,max((X-B)/b,0))`.
Cash after reserve is `X-b*F`, and reserve shortfall is `Q-F`. Household-first protects modeled miner
households. Losses remain visible and cannot fund deposits. Depreciation is never subtracted again
from cash. Matching depreciation targets total depreciation for a common fund and effective miner
depreciation for a miner fund. With no reserve, the joint target probability equals household cash adequacy.

## Fixed-timing compatibility and outputs

Omitting `batch_timing` preserves the existing fixed shift/calendar/hours API, optional hours matrix,
independent household count, and nonmining-income offset. Missing fixed calendar days retain the
explicitly warned one-day-per-shift fallback. Fixed mode cannot price per working day because it
has no working-day schedule. An hours matrix cannot price per operating hour because it does not
define shared machine exposure. In fixed dashboard mode, select a supported expense basis for every category.

Batch mode rejects conflicting calendar or hours-matrix inputs and nondefault fixed shifts/hours.
It supports independent household counts and nonnegative nonmining income. Scalar timing helper
methods raise `InputError` in batch mode; consume result arrays instead.

Both modes export draw-specific calendar days, shifts, operating hours, person-hours, household
requirements, expense rates, expense totals, cash/reserves, returns, and thresholds. Batch mode
also exports active and idle days. Means and medians have explicit `_mean`/`_median` keys. Original
scalar `L_person_hours`, `B_HH`, and `calendar_days_per_cycle` summaries exist only in fixed mode.
For a fixed hours matrix, the compatibility `operating_hours` value uses fixed shifts × fixed hours;
it is not inferred from individual attendance and cannot drive operating-hour expenses.

Manifests include inputs and code hashes, version, units, matrix order, schedule rules, provenance,
assumptions, and run configuration. `F` remains the valuation-mass array; its physical label follows
price basis. Per-draw economic and cash probabilities participate in every convergence stage and
probability precision decision, alongside return error/stability and mixture support checks. MCSE is numerical sampling error, not model uncertainty. Expected shortfall
uses exactly the lowest 5% of empirical probability mass, including fractional boundary weight.


## V2.3 numerical precision

`run(..., return_abs_tol=1., return_rel_tol=.01)` evaluates all probability MCSEs and errors of
mean, median, P5, P95 and worst-5% expected shortfall at completed batches after `n_min`.
Each return tolerance is `max(return_abs_tol, return_rel_tol * abs(statistic))` in PHP/person-hour.
Mean error is twice its sample standard error. Quantiles use pointwise 95% binomial order-statistic
intervals; error is the larger distance from the estimate to either endpoint. Small-sample unbounded
intervals are unavailable, not clipped to a deceptively narrow range. Expected-shortfall error is
twice the standard error of `min(E, empirical_P5)/0.05` winsorized influence values.

These are estimated **numerical sampling** bounds, not model uncertainty, simultaneous/sequential
coverage guarantees, or empirical predictive intervals. Overall `converged` requires precision at two
consecutive completed-batch checks, return changes within the same tolerances, and at least 30 draws
from each enabled productive ordinary/jackpot component. Components with zero probability are exempt.
`probability_precision_met` and `return_precision_met` separately describe the final check.
Stage tables include all eligible batch checks and descriptive historical prefixes, errors, tolerances,
component counts, stability flags and failure reasons. `n_fixed` always executes exactly its requested N;
it does not guarantee convergence. Capped results remain usable with their explicit unmet criteria.
Stricter checks can require more than 50,000 draws even when probability MCSE already passes.

An ordinary `PERT(0,0,0)` is rejected even for `p0=1`. Supply valid positive-component parameters and
use `p0=1` for an all-zero scenario. A continuous PERT with lower bound zero and positive width is allowed.
The count-based zero-state posterior 95% interval is displayed separately from MCSE.

Wage sensitivity reuses returns at 80%, 100%, 120% of the entered wage; illustrative cutoff sensitivity
uses (low, moderate) pairs (0.60, 0.30), (0.67, 0.34), (0.75, 0.40). This is consequence analysis, not a
policy justification. No additional Monte Carlo run is needed for these comparisons.

## Conversions, metadata and exports

`normalization.purity_fraction(value, unit)` accepts explicit fraction, percent or karat.
`realized_price(proceeds_php, mass_g, price_basis="farmgate")` requires positive sale proceeds/mass.
`fuel_cost(liters, php_per_liter, basis, quantity_basis=None)` requires nonnegative liters, positive
PHP/liter and matching supported exposure bases. These conversions are deterministic; stochastic
rates continue to use direct PERT inputs. Dashboard conversion modes take precedence over the inactive
direct inputs and preserve raw measurements with optional source/date in `Inputs.metadata`.
Reference pricing cannot derive a realized sale price. Measurement dates use YYYY-MM-DD.

`export_csv(result, max_rows=5000, schema_version=2)` defaults to corrected schema 2: `physical_purity_q`
is blank for unknown farmgate assay, and `valuation_multiplier` is the actual revenue multiplier.
The ambiguous `purity_q` is absent. Explicit `schema_version=1` preserves the historical valuation
column for existing readers; the dashboard labels it legacy. `export_manifest` accepts the same schema
selection and records the column meanings. Internal result array `q` remains the valuation multiplier.

Manifests include source hashes for application/test Python files and dependency declarations (excluding
virtual environments, dependency trees, Git internals and bytecode caches), Git
revision/dirty state, all relevant dependency versions, export version, normalized inputs/metadata,
seed, random-stream definitions, batching, numerical bounds/tolerances/status/reasons, and the stopping
reason (convergence, cap, or exact fixed count). Installed code without Git records null Git
identity plus source hashes; notebook-pasted code records unavailable identity. Set both `dataset_id`
and `dataset_sha256` only when a dataset is used; the hash is a lowercase 64-character SHA-256 digest.
No dataset identity is invented for manual scenarios. Re-run after source changes; manifests identify
the code present when generated rather than carrying old reports forward.
Engine-only use does not require the optional UI packages; their runtime versions are null in the
manifest when absent. Missing Git metadata also remains null without preventing simulation.

## Deferred capabilities

Empirical distribution fitting, bootstrap/event resampling, fitted-parameter risk intervals,
observed jackpot calibration and broader cross-input dependence await independent observations and a
modeling specification. Full stakeholder ledgers, cost reimbursements, partial batch attendance and
changing membership are unsupported. The current aggregate gross/net model and full-attendance
assumption are explicit. Software tests establish computational verification, not completed empirical
validation or implementation of every critique recommendation.
