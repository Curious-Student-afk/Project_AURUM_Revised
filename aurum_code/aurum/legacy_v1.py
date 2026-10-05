"""
legacy_v1.py -- FROZEN copy of the Project AURUM V1.0 simulation function.

Source: run_monte_carlo() from the original Streamlit dashboard script.
Only the Streamlit/UI code was removed; the numerical code below is unchanged.
DO NOT EDIT. It exists so that V2.0 can be regression-tested against V1.0 and so
that Chapter 3 can describe V1.0 exactly as it was implemented.

Known V1.0 limitations (documented for the manuscript, intentionally NOT fixed here):
  * lognormal calibration assumes min = 5th percentile and max = 99th percentile
    (constants 1.645 and 2.326) -- never elicited from respondents
  * no zero-yield state; the "jackpot" component is a lognormal anchored at Gf_min
  * jackpot frequency is a heuristic clamp(2*Gf_sub_max/Gf_jack_max, 0.2, 5.0) percent
  * H (shift duration) is the labour denominator, not person-hours
  * eta is applied on top of a farmgate price
  * success_rate uses the wage floor while the "required yield" includes family cost
  * hard-coded seed = 42; np.clip piles probability mass at the caps
"""
import numpy as np


def run_monte_carlo(
    H: float,
    T: float,
    Cfamily: float,
    opportunity_wage: float,
    Cfixed: float,
    delta: float,
    Cfuel: float,
    alpha: float,
    Gf_min: float,
    Gf_subsistence_max: float,
    Gf_jackpot_max: float,
    eta: float,
    Pg_min: float,
    Pg_max: float,
    iterations: int,
) -> dict:
    """Executes vectorized Monte Carlo simulation for small-scale mining profitability."""
    rng = np.random.default_rng(seed=42)

    # Jackpot probability coupling
    if Gf_jackpot_max > Gf_subsistence_max > 0:
        calculated_jackpot_prob = 2.0 * (Gf_subsistence_max / Gf_jackpot_max)
        jackpot_prob = float(np.clip(calculated_jackpot_prob, 0.2, 5.0))
    else:
        jackpot_prob = 2.0

    # Lognormal percentile calibrations
    if Gf_subsistence_max > Gf_min:
        sigma_yield_base = (np.log(Gf_subsistence_max) - np.log(Gf_min)) / (2.326 + 1.645)
        mu_yield_base = np.log(Gf_min) + 1.645 * sigma_yield_base
    else:
        mu_yield_base, sigma_yield_base = np.log(Gf_min), 0.05

    if Gf_jackpot_max > Gf_min:
        sigma_yield_jackpot = (np.log(Gf_jackpot_max) - np.log(Gf_min)) / (2.326 + 1.645)
        mu_yield_jackpot = np.log(Gf_min) + 1.645 * sigma_yield_jackpot
    else:
        mu_yield_jackpot, sigma_yield_jackpot = np.log(Gf_min), 0.05

    if Pg_max > Pg_min:
        sigma_price = (np.log(Pg_max) - np.log(Pg_min)) / (2.326 + 1.645)
        mu_price = np.log(Pg_min) + 1.645 * sigma_price
        Pg_mean = float(np.exp(mu_price + (sigma_price**2) / 2))
    else:
        mu_price, sigma_price, Pg_mean = np.log(Pg_min), 0.0, Pg_min

    # Calculate array counts
    jackpot_count = int(iterations * (jackpot_prob / 100.0))
    if jackpot_count < 1 and jackpot_prob > 0:
        jackpot_count = 1
    base_count = iterations - jackpot_count

    # Generate lognormal distributions
    Gf_base_array = rng.lognormal(mu_yield_base, sigma_yield_base, base_count)
    Gf_jackpot_array = rng.lognormal(mu_yield_jackpot, sigma_yield_jackpot, jackpot_count)

    # Clip extreme tails
    Gf_base_array = np.clip(Gf_base_array, 0.00, Gf_subsistence_max * 1.1)
    Gf_jackpot_array = np.clip(Gf_jackpot_array, 0.00, Gf_jackpot_max * 1.1)

    Gf_array = np.concatenate([Gf_base_array, Gf_jackpot_array])
    rng.shuffle(Gf_array)

    Pg_array = rng.lognormal(mu_price, sigma_price, iterations)
    Gf_mean = float(np.mean(Gf_array))

    # Division-by-zero protection for Tribute Split Ratio T
    eff_T = min(T, 0.99)

    # Deterministic reference view
    E_det = ((Gf_mean * eta * Pg_mean * (1 - eff_T)) - (delta * Cfixed) - ((1 + alpha) * Cfuel)) / H
    Y_det = ((delta * Cfixed) + ((1 + alpha) * Cfuel) + (Cfamily * H)) / (eta * Pg_mean * (1 - eff_T))

    # Vectorized stochastic financial pipeline
    hourly_efficiency_array = ((Gf_array * eta * Pg_array * (1 - eff_T)) - (delta * Cfixed) - ((1 + alpha) * Cfuel)) / H
    required_yield_array = ((delta * Cfixed) + ((1 + alpha) * Cfuel) + (Cfamily * H)) / (eta * Pg_array * (1 - eff_T))

    viability_mask = hourly_efficiency_array >= opportunity_wage
    success_rate = float((np.count_nonzero(viability_mask) / iterations) * 100)
    mean_stochastic_efficiency = float(np.mean(hourly_efficiency_array))
    mean_stochastic_cyt = float(np.mean(required_yield_array))

    return {
        "jackpot_prob": jackpot_prob,
        "E_det": E_det,
        "Y_det": Y_det,
        "success_rate": success_rate,
        "mean_stochastic_cyt": mean_stochastic_cyt,
        "hourly_efficiency_array": hourly_efficiency_array,
        "required_yield_array": required_yield_array,
        "opportunity_wage": opportunity_wage,
        "mean_stochastic_efficiency": mean_stochastic_efficiency,
        "Gf_array": Gf_array,
        "Pg_array": Pg_array,
        "iterations": iterations,
    }
