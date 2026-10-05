# Project AURUM — Ablation study (V1.0 → V2.0)

*Generated 2026-10-04 07:54 UTC · N = 500,000 draws per configuration · seed 42 · engine 2.0.0-draft*

**Read this first.** Everything below is a **scenario result under the assumptions listed at the end**. It is not a finding
about Pantukan miners. The pilot did not collect crew size, costs, zero-yield counts, jackpot frequency, a yield mode, or
dependence, so those fixes can only be switched on with assumed values. Headline metric: **P(E < W)**, the probability
that the labor return per person-hour is below the 65.63 PHP/hour wage benchmark (V1.0 reported 1 − this as
"Probability of Real Economic Sustainability").

## 1. Forward chain (cumulative; order matters)
| Step | Change added | P(E<W) % | ±2 SE | Δ vs prev (pp) | Detectable? | Δ vs V1.0 (pp) | P(Π<0) % | E P5 | E median | E mean |
|---|---|---|---|---|---|---|---|---|---|---|
| S0 | V1.0 as run (frozen code, benchmark inputs) | 62.6 | 0.14 |  |  |  | 0.4 | 14.8 | 54.5 | 61.3 |
| S1 | Explicit group accounting (n_c people, unrounded share 0.658) instead of manual /5 with the share rounded to 0.15 | 74.8 | 0.12 | +12.2 | yes | +12.2 | 1.0 | 9.2 | 44.0 | 49.9 |
| S2 | q = 1 on a farmgate price (V1.0 multiplied eta = 0.95 onto a price that already embeds grade) | 70.3 | 0.13 | -4.5 | yes | +7.6 | 0.7 | 11.3 | 47.9 | 54.2 |
| S3 | Beta-PERT (same min/max, assumed mode) replaces the hidden-percentile lognormal; for V1.0 it also removes V1.0's heuristic 0.2% 'jackpot' (the two are inseparable) | 39.8 | 0.14 | -30.5 | yes | -22.8 | 0.0 | 31.5 | 73.5 | 74.9 |
| S4 | Beta-PERT (same bounds, midpoint mode) replaces the hidden-percentile lognormal price | 37.0 | 0.14 | -2.7 | yes | -25.6 | 0.0 | 33.0 | 75.8 | 77.1 |
| S5 | Waste cycles with Y = 0 at probability p0 (assumed) | 43.3 | 0.14 | +6.2 | yes | -19.3 | 10.0 | -31.4 | 71.5 | 66.3 |
| S6 | Truncated-Pareto jackpot on [14.4, 24] g at probability pi_J (assumed) | 42.9 | 0.14 | -0.3 | yes | -19.7 | 10.0 | -31.4 | 71.9 | 87.4 |
| S7 | Costs become mean-preserving PERT draws (spread assumed), independent | 42.9 | 0.14 | -0.0 | within noise | -19.7 | 10.0 | -31.4 | 71.9 | 87.4 |
| S8 | Gaussian copula with assumed Spearman rho between the two stochastic costs | 43.0 | 0.14 | +0.0 | within noise | -19.7 | 10.0 | -31.4 | 71.9 | 87.4 |
"Detectable?" compares each step with the seed-to-seed noise (section 4). A step that is "within noise" does not move
**this** metric in **this** scenario; it may still matter for tails or other scenarios (see the P5 and P(Π<0) columns).

## 2. Each fix alone, added to V1.0 (order-free)
| Fix added alone to V1.0 | P(E<W) % | Δ vs V1.0 (pp) | SE (pp) |
|---|---|---|---|
| Person-hour accounting | 74.8 | +12.2 | 0.09 |
| Purity counted once | 57.4 | -5.2 | 0.10 |
| Beta-PERT yield | 30.5 | -32.1 | 0.09 |
| Beta-PERT price | 60.6 | -2.1 | 0.10 |
| Zero-yield hurdle | 66.4 | +3.8 | 0.10 |
| Jackpot tail | 62.0 | -0.6 | 0.10 |
| Stochastic costs | 62.6 | -0.0 | 0.10 |
| Cost dependence | 62.6 | -0.1 | 0.10 |

## 3. Leave-one-out from the full model (order-free, ranked by size of effect)
Full V2.0 model: P(E<W) = 43.0%.
| Fix reverted/removed from full V2.0 | P(E<W) % | Δ vs full (pp) | SE (pp) |
|---|---|---|---|
| Beta-PERT yield | 70.9 | +27.9 | 0.10 |
| Person-hour accounting | 30.4 | -12.6 | 0.10 |
| Zero-yield hurdle | 36.7 | -6.3 | 0.10 |
| Purity counted once | 48.7 | +5.7 | 0.10 |
| Beta-PERT price | 45.4 | +2.5 | 0.10 |
| Jackpot tail | 43.3 | +0.3 | 0.10 |
| Cost dependence | 42.9 | -0.0 | 0.10 |
| Stochastic costs | 42.9 | -0.0 | 0.10 |

## 4. Noise check (full model, 5 seeds)
P(E<W) ranges 42.9%–43.3% across seeds
(SD 0.16 pp). Treat differences smaller than about 0.3 pp as noise.

## 5. Sensitivity of the full model to the ASSUMED parameters (one at a time)
| Parameter varied | Value | Primary | P(E<W) % | E median |
|---|---|---|---|---|
| p0 | 0.0 |  | 36.5 | 76.3 |
| p0 | 0.05 |  | 39.7 | 74.3 |
| p0 | 0.1 | yes | 42.8 | 72.0 |
| p0 | 0.2 |  | 49.1 | 66.5 |
| p0 | 0.3 |  | 55.5 | 59.0 |
| pi_J | 0.0 |  | 43.2 | 71.6 |
| pi_J | 0.005 |  | 43.0 | 71.8 |
| pi_J | 0.01 | yes | 42.8 | 72.0 |
| pi_J | 0.02 |  | 42.5 | 72.3 |
| pi_J | 0.05 |  | 41.5 | 73.5 |
| mode_frac | 0.188 |  | 69.9 | 48.8 |
| mode_frac | 0.25 |  | 62.1 | 55.5 |
| mode_frac | 0.4 | yes | 42.8 | 72.0 |
| mode_frac | 0.6 |  | 23.6 | 93.8 |
| n_c | 3 |  | 11.9 | 140.9 |
| n_c | 4 |  | 23.4 | 97.8 |
| n_c | 5 | yes | 42.8 | 72.0 |
| n_c | 6 |  | 64.5 | 54.7 |
| n_c | 8 |  | 93.9 | 33.2 |
| rho | 0.0 |  | 42.8 | 72.0 |
| rho | 0.3 |  | 42.8 | 72.0 |
| rho | 0.5 | yes | 42.8 | 72.0 |
| rho | 0.8 |  | 42.9 | 71.9 |
| cost_spread | (0.9, 1.1) |  | 42.8 | 72.0 |
| cost_spread | (0.8, 1.2) | yes | 42.8 | 72.0 |
| cost_spread | (0.7, 1.5) |  | 44.0 | 70.9 |

## 6. Diagnostic: V1.0's equation on un-converted group inputs
If the group-level inputs (group yield, group costs, miner share 0.658) had been fed to V1.0's equation
**without** the manual ÷5 step, V1.0 would give P(E<W) = 8.1%
(mean E = 250 PHP/h) because H, not person-hours, would divide a group-level numerator. The manual
conversion avoided this, but it was undocumented and the share was rounded (0.1316 → 0.15).

## 7. What this does and does not show
* It shows which V1.0 flaws are **large** and which are **small** under these assumptions, and in which direction each moves the answer.
* **The size of the PERT-yield step is mostly the assumed mode, not a measured V1.0 error.** With the mode at 19% of the
  min–max range, Beta-PERT has the same mean yield as V1.0, and that step shrinks toward zero (section 5, parameter mode_frac).
  The pilot has no "usual" value, so which mode is right is unknown. Report the whole range, not one number.
* Stochastic costs and dependence barely move P(E<W) here because costs are small next to revenue and the assumed spreads are
  modest; they remain required by the specification and are verified in the test suite.
* The absolute level of P(E<W) depends strongly on the assumed crew size (see section 5) and on yield inputs that are provisional.
* S3 bundles V1.0's lognormal with its heuristic jackpot; S4 changes the mean price as well as the shape (the lognormal with
  min = P5 and max = P99 has a lower mean than a mid-mode PERT on the same bounds). Both are real V1.0 effects of the hidden
  calibration, but they are not pure "shape only" changes.
* Fixes that cannot move this metric by design (separating the three thresholds, renaming outputs, input guards) are not ablated;
  they are verified in the test suite instead.
* No empirical validation is implied (Domain G needs Phase 2 field data).

## Assumed / pilot parameters used
| Parameter | Value | Provenance |
|---|---|---|
| n_c | 5 | assumed |
| s_group | 0.658 | pilot |
| p0 | 0.1 | assumed |
| pi_J | 0.01 | assumed |
| jack_threshold | 14.4 |  |
| jack_cap | 24.0 |  |
| jack_shape | 1.5 | assumed |
| mode_frac | 0.4 | assumed |
| price_mode_frac | 0.5 | assumed |
| cost_lo | 0.8 |  |
| cost_hi | 1.2 |  |
| rho | 0.5 | assumed |

Benchmark (V1.0 inputs reproduced from the dashboard screenshots): {"H": 8.0, "T": 0.85, "Cfamily": 44.8, "W": 65.63, "Cfixed": 80.0, "delta": 1.1, "Cfuel": 160.0, "alpha": 0.02, "eta": 0.95, "Gmin": 0.33, "Gsub": 1.45, "Gjack": 19.47, "Pmin": 7282.35, "Pmax": 8900.65}
