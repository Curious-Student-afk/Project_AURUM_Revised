"""Deterministic unit conversions for optional observed inputs; no empirical fitting."""
from engine_v2 import InputError, finite_number, COST_BASES


def _number(name, value, positive=False):
    if not finite_number(value) or (value <= 0 if positive else value < 0):
        raise InputError(f"{name}: must be finite and {'positive' if positive else 'nonnegative'}.")
    return float(value)


def purity_fraction(value, unit="fraction"):
    """Convert explicit fraction, percent or karat; never guess the unit."""
    divisors = {"fraction": 1., "percent": 100., "karat": 24.}
    if unit not in divisors:
        raise InputError("Purity unit must be fraction, percent or karat.")
    q = _number("purity", value, True) / divisors[unit]
    if q > 1:
        raise InputError("Physical purity cannot exceed 1.")
    return q


def realized_price(proceeds_php, mass_g, price_basis="farmgate"):
    """Positive paid proceeds / sold recovered grams, for realized pricing only."""
    if price_basis != "farmgate":
        raise InputError("Sale conversion requires farmgate pricing, not a reference price.")
    result = _number("sale proceeds", proceeds_php, True) / _number("sold mass", mass_g, True)
    if not finite_number(result) or result <= 0:
        raise InputError("Sale conversion exceeded positive finite numeric limits.")
    return result


def fuel_cost(liters, php_per_liter, basis="per_batch", quantity_basis=None):
    """Derive a rate for a declared exposure; quantity and monetary bases must agree."""
    if basis not in COST_BASES or (quantity_basis is not None and quantity_basis != basis):
        raise InputError("Fuel quantity and monetary expense must use the same supported basis.")
    result = _number("fuel liters", liters) * _number("PHP/liter", php_per_liter, True)
    if not finite_number(result):
        raise InputError("Fuel conversion exceeded finite numeric limits.")
    return result
