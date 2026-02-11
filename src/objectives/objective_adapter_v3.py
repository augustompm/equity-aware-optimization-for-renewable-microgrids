import numpy as np
from typing import Dict, Tuple
from objectives.objective_functions_v3_COMPLETE import (
    objective_npc_v3,
    objective_lpsp_v3,
    objective_co2_v3,
    objective_gini_spatial_v3,
    disaggregate_load_to_households_simplified,

    LIFETIME_PV,
    LIFETIME_WIND,
    LIFETIME_DIESEL,
    LIFETIME_BATTERY,
)

DIESEL_LHV_MMBTU_PER_LITER = 0.1386

def adapt_npc_v1_to_v3(
    system_v1: Dict,
    components_info: Dict
) -> Dict:

    system_v3 = {
        'capital_cost_usd': system_v1['capital_cost_usd'],
        'fuel_cost_annual_usd': system_v1['fuel_cost_annual_usd'],
        'om_cost_annual_usd': system_v1['om_cost_annual_usd'],
        'discount_rate': system_v1['discount_rate'],
        'lifetime_years': system_v1['lifetime_years'],
    }

    system_v3['battery_capital_usd'] = components_info.get('battery_capital_usd', 0.0)
    system_v3['n_wind_mw'] = components_info.get('n_wind_mw', 0.0)
    system_v3['component_costs_usd'] = components_info.get('component_costs_usd', {})

    if 'component_ages_at_end' in components_info:
        system_v3['component_ages_at_end'] = components_info['component_ages_at_end']
    else:

        system_v3['component_ages_at_end'] = {
            'pv': 25,
            'wind': 5,
            'diesel': 5,
            'battery': 5
        }

    if 'component_lifetimes' in components_info:
        system_v3['component_lifetimes'] = components_info['component_lifetimes']
    else:
        system_v3['component_lifetimes'] = {
            'pv': LIFETIME_PV,
            'wind': LIFETIME_WIND,
            'diesel': LIFETIME_DIESEL,
            'battery': LIFETIME_BATTERY
        }

    return system_v3

def adapt_co2_v1_to_v3(dispatch_v1: Dict) -> Dict:

    fuel_diesel_mmbtu = dispatch_v1.get('fuel_diesel_mmbtu_annual', 0.0)

    fuel_diesel_liters = fuel_diesel_mmbtu / DIESEL_LHV_MMBTU_PER_LITER

    dispatch_v3 = {
        'fuel_diesel_liters_annual': fuel_diesel_liters
    }

    return dispatch_v3

def adapt_gini_v1_to_v3(
    hourly_costs: np.ndarray,
    aggregate_load_mwh: np.ndarray,
    renewable_production_mwh: np.ndarray,
    n_households: int = 1220,
    random_seed: int = 42
) -> np.ndarray:

    household_loads_mwh = disaggregate_load_to_households_simplified(
        aggregate_load_mwh,
        n_households=n_households,
        random_seed=random_seed
    )

    total_load_per_household = household_loads_mwh.sum(axis=1)

    total_renewable = renewable_production_mwh.sum()

    np.random.seed(random_seed)

    base_renewable_fraction = total_renewable / total_load_per_household.sum()

    n_low = int(n_households * 0.40)
    n_mid = int(n_households * 0.40)
    n_high = n_households - n_low - n_mid

    multipliers = np.concatenate([
        np.random.uniform(0.5, 1.0, n_low),
        np.random.uniform(0.8, 1.2, n_mid),
        np.random.uniform(1.2, 2.0, n_high)
    ])

    np.random.shuffle(multipliers)

    renewable_allocated_per_household = total_load_per_household * base_renewable_fraction * multipliers

    renewable_allocated_per_household *= (total_renewable / renewable_allocated_per_household.sum())

    renewable_fraction_per_household = renewable_allocated_per_household / total_load_per_household

    renewable_fraction_per_household = np.nan_to_num(renewable_fraction_per_household, nan=0.0)

    renewable_fraction_per_household = np.clip(renewable_fraction_per_household, 0.0, 1.0)

    return renewable_fraction_per_household

def objective_npc_adapted(system_v1: Dict, components_info: Dict) -> float:

    system_v3 = adapt_npc_v1_to_v3(system_v1, components_info)
    return objective_npc_v3(system_v3)

def objective_lpsp_adapted(dispatch_v1: Dict) -> float:

    return objective_lpsp_v3(dispatch_v1)

def objective_co2_adapted(dispatch_v1: Dict, lifetime_years: int = 25) -> float:

    dispatch_v3 = adapt_co2_v1_to_v3(dispatch_v1)
    return objective_co2_v3(dispatch_v3, lifetime_years)

def objective_gini_adapted(
    hourly_costs: np.ndarray,
    aggregate_load_mwh: np.ndarray,
    renewable_production_mwh: np.ndarray,
    n_households: int = 1220
) -> float:

    renewable_fraction_per_household = adapt_gini_v1_to_v3(
        hourly_costs,
        aggregate_load_mwh,
        renewable_production_mwh,
        n_households
    )
    return objective_gini_spatial_v3(renewable_fraction_per_household)

objective_npc = objective_npc_adapted
objective_lpsp = objective_lpsp_adapted
objective_co2 = objective_co2_adapted
objective_gini = objective_gini_adapted

if __name__ == "__main__":

    print("Testing objective_adapter_v3...")

    system_v1 = {
        'capital_cost_usd': 10_000_000,
        'fuel_cost_annual_usd': 0,
        'om_cost_annual_usd': 700_000,
        'discount_rate': 0.03,
        'lifetime_years': 25
    }
    components = {
        'battery_capital_usd': 4_500_000,
        'n_wind_mw': 19.1,
        'component_costs_usd': {
            'pv': 5_000_000,
            'wind': 8_000_000,
            'diesel': 2_000_000,
            'battery': 4_500_000
        }
    }
    npc = objective_npc_adapted(system_v1, components)
    print(f"NPC adapted: ${npc:,.0f}")
    assert 35_000_000 < npc < 42_000_000, f"NPC out of range: {npc}"

    dispatch_v1 = {'fuel_diesel_mmbtu_annual': 13_860}
    co2 = objective_co2_adapted(dispatch_v1, 25)
    print(f"CO2 adapted: {co2:,.0f} tonnes")
    assert 5_000 < co2 < 8_000, f"CO2 out of range: {co2}"

    np.random.seed(42)
    load = np.ones(8760) * 2.5
    renewable = np.ones(8760) * 1.0
    gini = objective_gini_adapted(np.ones(8760), load, renewable, n_households=100)
    print(f"Gini adapted: {gini:.4f}")
    assert 0.0 <= gini <= 1.0, f"Gini out of range: {gini}"

    print("\n✓ All adapter smoke tests passed!")
