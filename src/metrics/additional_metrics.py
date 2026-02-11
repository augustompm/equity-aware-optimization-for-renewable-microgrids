import numpy as np
from typing import Dict, Tuple

def calculate_re_penetration(
    p_pv_hourly: np.ndarray,
    p_wind_hourly: np.ndarray,
    p_load_hourly: np.ndarray
) -> float:

    re_annual = np.sum(p_pv_hourly + p_wind_hourly)
    load_annual = np.sum(p_load_hourly)

    if load_annual <= 0:
        return 0.0

    re_penetration = (re_annual / load_annual) * 100.0
    return re_penetration

def calculate_excess_power(
    p_pv_hourly: np.ndarray,
    p_wind_hourly: np.ndarray,
    p_load_hourly: np.ndarray,
    p_battery_charge_hourly: np.ndarray,
    p_diesel_hourly: np.ndarray
) -> Tuple[float, float]:

    re_available = p_pv_hourly + p_wind_hourly
    demand = p_load_hourly + p_battery_charge_hourly

    excess_hourly = np.maximum(0, re_available - demand)
    excess_annual = np.sum(excess_hourly)

    generation_annual = np.sum(p_pv_hourly + p_wind_hourly + p_diesel_hourly)

    if generation_annual <= 0:
        return 0.0, 0.0

    excess_pct = (excess_annual / generation_annual) * 100.0
    return excess_pct, excess_annual

def calculate_lcoe(
    npc: float,
    load_annual_kwh: float,
    lifetime_years: int = 25
) -> float:

    if load_annual_kwh <= 0 or lifetime_years <= 0:
        return 0.0

    total_energy = load_annual_kwh * lifetime_years
    lcoe = npc / total_energy
    return lcoe

def calculate_fuel_consumption(
    p_diesel_hourly: np.ndarray,
    diesel_efficiency: float = 0.30,
    fuel_energy_content_kwh_per_liter: float = 10.0
) -> Tuple[float, float]:

    if diesel_efficiency <= 0:
        return 0.0, 0.0

    diesel_annual_mwh = np.sum(p_diesel_hourly)
    diesel_annual_kwh = diesel_annual_mwh * 1000
    fuel_input_kwh = diesel_annual_kwh / diesel_efficiency
    fuel_liters = fuel_input_kwh / fuel_energy_content_kwh_per_liter

    diesel_density_kg_per_liter = 0.85
    fuel_kg = fuel_liters * diesel_density_kg_per_liter

    return fuel_liters, fuel_kg

def calculate_all_additional_metrics(
    simulation_results: Dict,
    npc: float,
    config: Dict
) -> Dict:

    p_pv = simulation_results['p_pv_hourly']
    p_wind = simulation_results['p_wind_hourly']
    p_load = simulation_results['p_load_hourly']
    p_battery_charge = simulation_results['p_battery_charge_hourly']
    p_diesel = simulation_results['p_diesel_hourly']

    load_annual_kwh = float(np.sum(p_load)) * 1000

    re_pct = calculate_re_penetration(p_pv, p_wind, p_load)

    excess_pct, excess_mwh = calculate_excess_power(
        p_pv, p_wind, p_load, p_battery_charge, p_diesel
    )

    lcoe = calculate_lcoe(
        npc, load_annual_kwh,
        config.get('lifetime_years', 25)
    )

    fuel_liters, fuel_kg = calculate_fuel_consumption(
        p_diesel,
        config.get('diesel_efficiency', 0.30)
    )

    return {
        're_penetration_pct': float(re_pct),
        'excess_power_pct': float(excess_pct),
        'excess_power_mwh': float(excess_mwh),
        'lcoe_cad_per_kwh': float(lcoe),
        'fuel_consumption_liters': float(fuel_liters),
        'fuel_consumption_kg': float(fuel_kg),
        'load_annual_kwh': float(load_annual_kwh)
    }
