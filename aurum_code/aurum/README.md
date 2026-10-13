# Project AURUM — Engine V2.2 (draft)

One Monte Carlo iteration represents a complete production batch. All presets are independently
constructed **synthetic demonstrations**, not measured mining conditions. Tests verify software
behavior; empirical validation requires independently obtained observations.

## Run locally

Python 3.11+ is required. From this directory:

```bash
python -m pip install -r requirements.txt
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
when inputs or code change. Requirements are currently unpinned; each manifest records runtime versions.

## Timing and production

`Inputs.batch_timing` accepts `BatchTiming(work_days, idle_days, shift_counts, hours_by_shifts)`.

- Working days use standard Beta-PERT (shape weight 4), rounded by `floor(draw + 0.5)`.
  Bounds and mode are whole numbers from 1 to 3650. This computational limit bounds scheduling work.
- Idle days use independent continuous nonnegative Beta-PERT. Calendar days equal working plus idle days.
- `shift_counts` contains three nonnegative integer counts for 1, 2, and 3 shifts, totaling ten working days.
  Each working day independently draws a categorical shift count using these frequencies.
- `hours_by_shifts` maps 1, 2, and 3 to conditional PERTs. One duration is drawn per day and applies to
  every shift that day. All registered workers attend. Bounds are at least 0.5 and at most
  `min(16, 24 / shift_count)` hours. Dashboard bounds use 0.25-hour increments; draws remain continuous.
- Total shifts are summed over days. Operating hours sum scheduled shift durations, without multiplying
  by crew size. Person-hours equal crew size times operating hours. No worker rotation is modeled.
- Yield, zero-yield counts, jackpot frequency and capacity are elicited per completed batch. Batch yield
  is never multiplied by shift count. Jackpot probability is conditional on a productive batch; its
  magnitude uses a truncated Pareto with a lower threshold, cap, and positive shape.
- Batch yield, valuation, and timing are independent in this model. Timing has a separate seeded random
  stream. Results are reproducible given inputs, seed, batching configuration, code, and runtime versions.

Farmgate pricing values recovered gold directly (`q=1`, a valuation convention). Reference pricing
uses physical purity and reports fine gold mass. Both price and reference purity can use bounded PERT.

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
- Household economic adequacy compares `Pi` with `B_HH = n_c * calendar_days * W_daily`.
  `W_daily` is PHP per represented household per calendar day; batch mode assumes one household per
  registered worker and no nonmining-income offset. Additional worksite food is excluded from this benchmark.
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

Batch mode rejects conflicting calendar or hours-matrix inputs, nondefault fixed shifts/hours,
a household count differing from crew size, and nonzero nonmining income. Scalar timing helper
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
MCSE stopping decision. MCSE is numerical sampling error, not model uncertainty. Expected shortfall
uses exactly the lowest 5% of empirical probability mass, including fractional boundary weight.
