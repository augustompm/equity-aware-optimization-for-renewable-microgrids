from typing import Dict, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from simulation.system_simulator import simulate_system
from resilience.scenarios import (
    apply_cold_snap_scenario,
    apply_fuel_disruption_scenario,
    apply_blizzard_scenario
)

def calculate_energy_not_served(dispatch_summary: Dict) -> Tuple[float, float]:

    if dispatch_summary['total_load_mwh'] <= 0:
        raise ValueError("total_load_mwh must be positive")

    ens_mwh = np.sum(dispatch_summary['deficit_mwh'])
    ens_fraction = ens_mwh / dispatch_summary['total_load_mwh']

    return ens_mwh, ens_fraction

def calculate_outage_hours(dispatch_summary: Dict) -> int:

    deficit = dispatch_summary['deficit_mwh']
    outage_hours = int(np.sum(deficit > 0))

    return outage_hours

def calculate_fuel_increase(dispatch_summary: Dict, baseline_fuel: float) -> float:

    if baseline_fuel <= 0:
        raise ValueError("baseline_fuel must be positive")

    scenario_fuel = dispatch_summary['total_diesel_fuel_mmbtu']
    fuel_increase_pct = (scenario_fuel / baseline_fuel - 1.0) * 100.0

    return fuel_increase_pct

def calculate_resilience_index(ens_fraction: float, outage_hours: int) -> float:

    energy_reliability = 1.0 - ens_fraction
    temporal_reliability = 1.0 - (outage_hours / 8760.0)

    resilience_index = energy_reliability * temporal_reliability

    return resilience_index

def calculate_resilience_metrics(
    dispatch_summary: Dict,
    baseline_fuel: float
) -> Dict[str, float]:

    ens_mwh, ens_fraction = calculate_energy_not_served(dispatch_summary)
    outage_hours = calculate_outage_hours(dispatch_summary)
    fuel_increase_pct = calculate_fuel_increase(dispatch_summary, baseline_fuel)
    resilience_index = calculate_resilience_index(ens_fraction, outage_hours)

    metrics = {
        'ens_mwh': ens_mwh,
        'ens_fraction': ens_fraction,
        'outage_hours': outage_hours,
        'fuel_increase_pct': fuel_increase_pct,
        'resilience_index': resilience_index
    }

    return metrics

def run_scenario_simulation(
    solution: Dict,
    scenario_name: str,
    start_day: int = 0,
    baseline_fuel: float = 333762.0
) -> Dict[str, float]:

    system_config = get_system_config()

    load_profile = pd.read_csv(system_config['load_profile_path'])['Load_MW'].values
    solar_cf = pd.read_csv(system_config['solar_cf_path'])['CF_pv'].values
    wind_cf = pd.read_csv(system_config['wind_cf_path'])['CF_wind'].values
    meteorology = pd.read_csv(system_config['meteorology_path'])
    temperature = meteorology['T_ambient_C'].values

    load_override = None
    solar_cf_override = None
    wind_cf_override = None
    temperature_override = None

    if scenario_name == 'A1_cold_snap':
        load_modified, temperature_modified = apply_cold_snap_scenario(
            load_profile, temperature, start_day, duration_days=7
        )
        load_override = load_modified
        temperature_override = temperature_modified

    elif scenario_name == 'A2_fuel_disruption':
        fuel_limit_per_hour = apply_fuel_disruption_scenario(
            fuel_reserves_mmbtu=5000.0, duration_days=14
        )
        pass

    elif scenario_name == 'A3_blizzard':
        pv_cf_modified, wind_cf_modified, load_modified = apply_blizzard_scenario(
            solar_cf, wind_cf, load_profile, start_day, duration_days=3
        )
        solar_cf_override = pv_cf_modified
        wind_cf_override = wind_cf_modified
        load_override = load_modified

    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    objectives, constraints, dispatch_summary = simulate_system(
        solution, system_config,
        load_override=load_override,
        solar_cf_override=solar_cf_override,
        wind_cf_override=wind_cf_override,
        temperature_override=temperature_override
    )

    metrics = calculate_resilience_metrics(dispatch_summary, baseline_fuel)

    return metrics

def get_system_config() -> Dict:

    return {
        'load_profile_path': project_root / 'data' / 'load-profile-8760h.csv',
        'solar_cf_path': project_root / 'data' / 'solar-capacity-factors.csv',
        'wind_cf_path': project_root / 'data' / 'wind-capacity-factors.csv',
        'meteorology_path': project_root / 'data' / 'meteorology-8760h.csv',
        'discount_rate': 0.03,
        'lifetime_years': 25,
        'diesel_efficiency': 0.30,
        'diesel_fuel_cost_per_mmbtu': 20.0,
        'diesel_min_load_fraction': 0.30,
        'pv_tilt_deg': 60,
        'wind_hub_height_m': 100,
        'battery_c_rate': 0.25,
        'battery_efficiency': 0.90,
        'battery_dod_max': 0.80,
        'area_available_m2': 100000.0,
        'renewable_fraction_max': 0.20,
        'reserve_fraction': 0.15,
        'lpsp_limit': 0.05,
        'grid_connected': False
    }
