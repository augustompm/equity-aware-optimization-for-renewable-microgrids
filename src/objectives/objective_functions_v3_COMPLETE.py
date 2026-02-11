import numpy as np
from typing import Dict, List, Tuple, Optional

DISCOUNT_RATE_DEFAULT = 0.03

PROJECT_LIFETIME_YEARS = 25

LIFETIME_PV = 25
LIFETIME_WIND = 20
LIFETIME_DIESEL = 20
LIFETIME_BATTERY = 5

SALVAGE_FRACTION_PV_NEW = 0.15
SALVAGE_FRACTION_WIND_NEW = 0.10
SALVAGE_FRACTION_DIESEL_NEW = 0.20
SALVAGE_FRACTION_BATTERY = 0.0

TOWER_COST_PER_KW = 245.0

TOWER_OM_PER_KW_YR = 41.0 * 0.14

DIESEL_BETA0_INT = 3.63719
DIESEL_BETA1_INT = 0.02031
DIESEL_BETA0_SLOPE = 0.25098
DIESEL_BETA1_SLOPE = -1.1827e-5

DIESEL_CO2_KG_PER_LITER = 2.6

LPSP_THRESHOLD = 0.01

BATTERY_SOH_THRESHOLD = 0.80

def present_value(
    future_value: float,
    discount_rate: float,
    years: int
) -> float:

    if years == 0:
        return future_value
    return future_value / ((1 + discount_rate) ** years)

def present_worth_factor(
    discount_rate: float,
    years: int
) -> float:

    if discount_rate == 0:
        return float(years)
    return (1 - (1 + discount_rate) ** (-years)) / discount_rate

def calculate_battery_replacement_years(
    project_years: int = PROJECT_LIFETIME_YEARS,
    battery_lifetime: int = LIFETIME_BATTERY
) -> List[int]:

    replacement_years = list(range(battery_lifetime, project_years, battery_lifetime))
    return replacement_years

def calculate_battery_replacements_pv(
    battery_capital_cost: float,
    project_years: int = PROJECT_LIFETIME_YEARS,
    battery_lifetime: int = LIFETIME_BATTERY,
    discount_rate: float = DISCOUNT_RATE_DEFAULT
) -> float:

    replacement_years = calculate_battery_replacement_years(project_years, battery_lifetime)

    pv_total = sum(
        present_value(battery_capital_cost, discount_rate, year)
        for year in replacement_years
    )

    return pv_total

def calculate_salvage_value_component(
    component_cost: float,
    age_at_project_end: int,
    component_lifetime: int,
    salvage_fraction_new: float
) -> float:

    if age_at_project_end >= component_lifetime:

        return component_cost * salvage_fraction_new
    else:

        remaining_life_fraction = (component_lifetime - age_at_project_end) / component_lifetime
        return component_cost * remaining_life_fraction

def objective_npc_v3(system: Dict) -> float:

    capital = system.get('capital_cost_usd', 0.0)
    fuel_annual = system.get('fuel_cost_annual_usd', 0.0)
    om_annual = system.get('om_cost_annual_usd', 0.0)
    discount_rate = system.get('discount_rate', DISCOUNT_RATE_DEFAULT)
    lifetime = system.get('lifetime_years', PROJECT_LIFETIME_YEARS)

    pwf = present_worth_factor(discount_rate, lifetime)

    pv_fuel = fuel_annual * pwf
    pv_om = om_annual * pwf

    battery_capital = system.get('battery_capital_usd', 0.0)
    battery_lifetime = system.get('battery_lifetime_years', LIFETIME_BATTERY)

    if battery_capital > 0:
        pv_battery_repl = calculate_battery_replacements_pv(
            battery_capital, lifetime, battery_lifetime, discount_rate
        )
    else:
        pv_battery_repl = 0.0

    n_wind_mw = system.get('n_wind_mw', 0.0)

    if n_wind_mw > 0:
        capital_tower = n_wind_mw * 1000 * TOWER_COST_PER_KW
        om_tower_annual = n_wind_mw * 1000 * TOWER_OM_PER_KW_YR
        pv_tower_om = om_tower_annual * pwf
    else:
        capital_tower = 0.0
        pv_tower_om = 0.0

    component_costs = system.get('component_costs_usd', {})
    component_ages = system.get('component_ages_at_end', {})
    component_lifetimes = system.get('component_lifetimes', {})

    salvage_year_25 = 0.0

    if component_costs:

        if 'pv' in component_costs:
            sv_pv = calculate_salvage_value_component(
                component_costs['pv'],
                component_ages.get('pv', 25),
                component_lifetimes.get('pv', LIFETIME_PV),
                SALVAGE_FRACTION_PV_NEW
            )
            salvage_year_25 += sv_pv

        if 'wind' in component_costs:
            sv_wind = calculate_salvage_value_component(
                component_costs['wind'],
                component_ages.get('wind', 5),
                component_lifetimes.get('wind', LIFETIME_WIND),
                SALVAGE_FRACTION_WIND_NEW
            )
            salvage_year_25 += sv_wind

        if 'diesel' in component_costs:
            sv_diesel = calculate_salvage_value_component(
                component_costs['diesel'],
                component_ages.get('diesel', 5),
                component_lifetimes.get('diesel', LIFETIME_DIESEL),
                SALVAGE_FRACTION_DIESEL_NEW
            )
            salvage_year_25 += sv_diesel

        if 'battery' in component_costs:
            sv_battery = calculate_salvage_value_component(
                component_costs['battery'],
                component_ages.get('battery', 5),
                component_lifetimes.get('battery', LIFETIME_BATTERY),
                SALVAGE_FRACTION_BATTERY
            )
            salvage_year_25 += sv_battery

    pv_salvage = present_value(salvage_year_25, discount_rate, lifetime)

    npc = (capital
           + capital_tower
           + pv_fuel
           + pv_om
           + pv_tower_om
           + pv_battery_repl
           - pv_salvage)

    return npc

def objective_lpsp_v3(dispatch_results: Dict) -> float:

    total_deficit = dispatch_results.get('total_deficit_mwh', 0.0)
    total_load = dispatch_results.get('total_load_mwh', 0.0)

    if total_load == 0:
        return 0.0

    lpsp = total_deficit / total_load

    return max(0.0, min(lpsp, 1.0))

def objective_co2_v3(dispatch_results: Dict, lifetime_years: int = PROJECT_LIFETIME_YEARS) -> float:

    fuel_diesel_liters = dispatch_results.get('fuel_diesel_liters_annual', 0.0)

    emission_factor_kg_per_liter = DIESEL_CO2_KG_PER_LITER

    co2_annual_kg = fuel_diesel_liters * emission_factor_kg_per_liter

    co2_lifetime_tonnes = (co2_annual_kg * lifetime_years) / 1000.0

    return co2_lifetime_tonnes

def disaggregate_load_to_households_simplified(
    aggregate_load_mwh_hourly: np.ndarray,
    n_households: int = 900,
    random_seed: int = 42
) -> np.ndarray:

    np.random.seed(random_seed)

    n_timesteps = len(aggregate_load_mwh_hourly)

    base_load_per_household = aggregate_load_mwh_hourly / n_households

    scaling_factors = np.random.uniform(0.5, 1.5, size=n_households)

    household_loads = np.outer(scaling_factors, base_load_per_household)

    total_allocated = household_loads.sum(axis=0)
    for t in range(n_timesteps):
        if total_allocated[t] > 0:
            household_loads[:, t] *= (aggregate_load_mwh_hourly[t] / total_allocated[t])

    return household_loads

def objective_gini_spatial_v3(renewable_fraction_per_household: np.ndarray) -> float:

    n = len(renewable_fraction_per_household)

    if n < 2:
        return 0.0

    total = np.sum(renewable_fraction_per_household)

    if total == 0:
        return 0.0

    sorted_fractions = np.sort(renewable_fraction_per_household)

    index = np.arange(1, n + 1)

    gini = ((2 * np.sum(index * sorted_fractions) - (n + 1) * total)
            / (n * total))

    return max(0.0, min(gini, 1.0))

if __name__ == "__main__":
    import doctest
    print("Running doctests for objective_functions_v3_COMPLETE.py...")
    results = doctest.testmod(verbose=True)
    print(f"\nDoctests: {results.attempted} tests, {results.failed} failures")
    if results.failed == 0:
        print("✅ ALL DOCTESTS PASSED")
    else:
        print(f"❌ {results.failed} DOCTESTS FAILED")
