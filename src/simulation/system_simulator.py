import pandas as pd
import numpy as np
from pathlib import Path

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from components.pv import SolarPV
from components.wind import WindTurbine
from components.battery import Battery
from components.generator import Generator
from components.load import LoadProfile

from objectives.objective_functions import (
    objective_npc,
    objective_lpsp,
    objective_co2,
    objective_gini,
    objective_gini_spatial,
    objective_gini_burden,
    objective_gini_theja
)

try:
    from objectives.objective_adapter_v3 import (
        objective_npc_adapted,
        objective_lpsp_adapted,
        objective_co2_adapted,
        objective_gini_adapted
    )
    V3_AVAILABLE = True
except ImportError:
    V3_AVAILABLE = False

from constraints.constraint_validator import validate_solution

def initialize_components(decision_vars, system_config):

    components = {}

    components['pv'] = SolarPV(
        capacity_kw=decision_vars['n_pv_kw'],
        tilt_deg=system_config['pv_tilt_deg'],
        temp_coeff=-0.004,
        name="PV-System"
    )

    components['wind'] = WindTurbine(
        capacity_mw=decision_vars['n_wind_mw'],
        hub_height_m=system_config['wind_hub_height_m'],
        cut_in_ms=3.0,
        rated_ms=12.0,
        cut_out_ms=25.0,
        name="Wind-Turbine"
    )

    components['battery'] = Battery(
        capacity_mwh=decision_vars['e_battery_mwh'],
        c_rate=system_config['battery_c_rate'],
        efficiency=system_config['battery_efficiency'],
        dod_max=system_config['battery_dod_max'],
        soc_initial=0.50,
        name="Battery-ESS"
    )

    components['diesel'] = Generator(
        capacity_mw=decision_vars['p_diesel_mw'],
        efficiency=system_config['diesel_efficiency'],
        fuel_cost_per_mmbtu=system_config['diesel_fuel_cost_per_mmbtu'],
        min_load_fraction=system_config['diesel_min_load_fraction'],
        startup_time_h=1.0,
        name="Diesel-Generator"
    )

    return components

def simulate_system(decision_vars, system_config,
                   load_override=None,
                   solar_cf_override=None,
                   wind_cf_override=None,
                   temperature_override=None,
                   use_v3=False):

    components = initialize_components(decision_vars, system_config)

    if load_override is not None:
        load_array = load_override
    else:
        load = LoadProfile(system_config['load_profile_path'])
        load_array = None

    if solar_cf_override is not None:
        solar_cf_array = solar_cf_override
    else:
        solar_df = pd.read_csv(system_config['solar_cf_path'])
        solar_cf_array = None

    if wind_cf_override is not None:
        wind_cf_array = wind_cf_override
    else:
        wind_df = pd.read_csv(system_config['wind_cf_path'])
        wind_cf_array = None

    if temperature_override is not None:
        temperature_array = temperature_override
    else:
        met_df = pd.read_csv(system_config['meteorology_path'])
        temperature_array = None

    hourly_results = []

    for t in range(8760):
        if load_array is not None:
            load_t = load_array[t]
        else:
            load_t = load.get_load(t)

        if solar_cf_array is not None:
            cf_pv = solar_cf_array[t]
        else:
            cf_pv = solar_df.loc[t, 'CF_pv']

        if wind_cf_array is not None:
            cf_wind = wind_cf_array[t]
        else:
            cf_wind = wind_df.loc[t, 'CF_wind']

        if temperature_array is not None:
            temp_c = temperature_array[t]
        else:
            temp_c = met_df.loc[t, 'T_ambient_C']

        p_pv = components['pv'].generate(cf_pv, temp_c)
        p_wind = components['wind'].generate(cf_wind)

        p_renewable = p_pv + p_wind

        p_battery_discharge = 0.0
        p_battery_charge = 0.0
        p_diesel = 0.0
        fuel_diesel_mmbtu = 0.0
        cost_diesel_usd = 0.0
        deficit = 0.0

        if p_renewable >= load_t:
            excess = p_renewable - load_t
            if components['battery'].capacity_mwh > 0:
                p_battery_charge_actual = components['battery'].charge(excess, 1.0)
                curtailment = excess - p_battery_charge_actual
                p_battery_charge = p_battery_charge_actual
            else:
                p_battery_charge = 0.0
                curtailment = excess

        else:
            shortfall = load_t - p_renewable

            if components['battery'].capacity_mwh > 0:
                available_battery = components['battery'].get_available_energy()
                p_battery_discharge_requested = min(shortfall, components['battery'].p_max_mw)
                p_battery_discharge_actual = components['battery'].discharge(p_battery_discharge_requested, 1.0)
                p_battery_discharge = p_battery_discharge_actual
            else:
                p_battery_discharge = 0.0

            remaining = shortfall - p_battery_discharge

            if remaining > 0:
                if components['diesel'].capacity_mw > 0:
                    p_diesel_requested = remaining

                    p_diesel_actual = min(p_diesel_requested, components['diesel'].capacity_mw)

                    if p_diesel_actual > 0:
                        result = components['diesel'].dispatch(p_diesel_actual, 1.0)
                        p_diesel = result['power_output_mw']
                        fuel_diesel_mmbtu = result['fuel_consumed_mmbtu']
                        cost_diesel_usd = result['operating_cost_usd']

                    deficit = remaining - p_diesel
                else:
                    deficit = remaining

        hourly_results.append({
            'load_mw': load_t,
            'pv_mw': p_pv,
            'wind_mw': p_wind,
            'battery_discharge_mw': p_battery_discharge,
            'battery_charge_mw': p_battery_charge,
            'diesel_mw': p_diesel,
            'diesel_fuel_mmbtu': fuel_diesel_mmbtu,
            'diesel_cost_usd': cost_diesel_usd,
            'deficit_mw': max(deficit, 0.0),
            'soc': components['battery'].soc
        })

    results_df = pd.DataFrame(hourly_results)

    total_load_mwh = results_df['load_mw'].sum()
    total_pv_mwh = results_df['pv_mw'].sum()
    total_wind_mwh = results_df['wind_mw'].sum()
    total_diesel_mwh = results_df['diesel_mw'].sum()
    total_diesel_fuel_mmbtu = results_df['diesel_fuel_mmbtu'].sum()
    total_deficit_mwh = results_df['deficit_mw'].sum()

    lpsp_value = total_deficit_mwh / total_load_mwh if total_load_mwh > 0 else 0.0

    capital_cost = (
        decision_vars['n_pv_kw'] * system_config['pv_capital_cost_per_kw'] +
        decision_vars['n_wind_mw'] * 1000 * system_config['wind_capital_cost_per_kw'] +
        decision_vars['e_battery_mwh'] * 1000 * system_config['battery_capital_cost_per_kwh'] +
        decision_vars['p_diesel_mw'] * 1000 * system_config['diesel_capital_cost_per_kw']
    )
    fuel_cost_annual = (total_diesel_fuel_mmbtu * system_config['diesel_fuel_cost_per_mmbtu'])

    om_cost_annual = (
        decision_vars['n_pv_kw'] * system_config['pv_om_cost_per_kw_yr'] +
        decision_vars['n_wind_mw'] * 1000 * system_config['wind_om_cost_per_kw_yr'] +
        decision_vars['e_battery_mwh'] * 1000 * system_config['battery_om_cost_per_kwh_yr']
    )

    replacement_cost = (
        decision_vars['e_battery_mwh'] * 1000 *
        system_config['battery_capital_cost_per_kwh'] *
        system_config['battery_replacement_fraction']
    )
    replacement_year = system_config['battery_replacement_years']

    if use_v3 and V3_AVAILABLE:

        system_for_npc = {
            'capital_cost_usd': capital_cost,
            'fuel_cost_annual_usd': fuel_cost_annual,
            'om_cost_annual_usd': om_cost_annual,
            'discount_rate': system_config['discount_rate'],
            'lifetime_years': system_config['lifetime_years']
        }

        components_info_for_npc = {
            'battery_capital_usd': components['battery'].capital_cost_usd if hasattr(components['battery'], 'capital_cost_usd') else capital_cost * 0.3,
            'n_wind_mw': decision_vars['n_wind_mw'],
            'component_costs_usd': {
                'pv': capital_cost * 0.4,
                'wind': capital_cost * 0.2,
                'diesel': capital_cost * 0.1,
                'battery': capital_cost * 0.3
            }
        }

        npc_value = objective_npc_adapted(system_for_npc, components_info_for_npc)

        dispatch_for_lpsp = {
            'total_deficit_mwh': total_deficit_mwh,
            'total_load_mwh': total_load_mwh
        }
        lpsp_value = objective_lpsp_adapted(dispatch_for_lpsp)

        dispatch_for_co2 = {
            'fuel_diesel_mmbtu_annual': total_diesel_fuel_mmbtu
        }
        co2_value = objective_co2_adapted(dispatch_for_co2, system_config['lifetime_years'])

        total_re_mwh = total_pv_mwh + total_wind_mwh

        gini_value = objective_gini_theja(
            total_re_mwh=total_re_mwh,
            total_load_mwh=total_load_mwh,
            n_households=1220
        )
    else:

        system_for_npc = {
            'capital_cost_usd': capital_cost,
            'fuel_cost_annual_usd': fuel_cost_annual,
            'om_cost_annual_usd': om_cost_annual,
            'replacement_cost_usd': replacement_cost,
            'replacement_year': replacement_year,
            'discount_rate': system_config['discount_rate'],
            'lifetime_years': system_config['lifetime_years']
        }

        npc_value = objective_npc(system_for_npc)

        dispatch_for_lpsp = {
            'total_deficit_mwh': total_deficit_mwh,
            'total_load_mwh': total_load_mwh
        }
        lpsp_value = objective_lpsp(dispatch_for_lpsp)

        dispatch_for_co2 = {
            'fuel_lng_mmbtu_annual': 0.0,
            'fuel_diesel_mmbtu_annual': total_diesel_fuel_mmbtu
        }
        co2_value = objective_co2(dispatch_for_co2, system_config['lifetime_years'])

        total_re_mwh = total_pv_mwh + total_wind_mwh

        gini_value = objective_gini_theja(
            total_re_mwh=total_re_mwh,
            total_load_mwh=total_load_mwh,
            n_households=1220
        )

    objectives = {
        'npc': npc_value,
        'lpsp': lpsp_value,
        'co2': co2_value,
        'gini': gini_value
    }

    x_for_constraints = {
        **decision_vars,
        'p_diesel_online_mw': decision_vars['p_diesel_mw'],
        'p_battery_discharge_mw': components['battery'].p_max_mw,
        'p_load_avg_mw': total_load_mwh / 8760.0,
        'p_pv_installed_kw': decision_vars['n_pv_kw'],
        'p_wind_installed_mw': decision_vars['n_wind_mw'],
        'p_grid_buy_mw': 0.0,
        'p_grid_sell_mw': 0.0,
        'grid_connected': system_config['grid_connected']
    }

    bounds = {
        'n_pv_kw': (0, 1000),
        'n_wind_mw': (0, 10),
        'e_battery_mwh': (0, 20),
        'p_diesel_mw': (0, 10)
    }

    if 'area_available_pv_m2' in system_config and 'area_available_wind_m2' in system_config:
        area_params = {
            'area_pv_per_kw': 2.0,
            'area_wind_per_mw': 186050.0,
            'area_battery_per_mwh': 10.0,
            'area_available_pv_m2': system_config['area_available_pv_m2'],
            'area_available_wind_m2': system_config['area_available_wind_m2']
        }
    else:
        area_params = {
            'area_pv_per_kw': 2.0,
            'area_wind_per_mw': 186050.0,
            'area_battery_per_mwh': 10.0,
            'area_available_m2': system_config['area_available_m2']
        }

    simulation_results = {
        'lpsp': lpsp_value
    }

    policy = {
        'renewable_fraction_max': system_config['renewable_fraction_max']
    }

    grid_limits = {
        'p_max_import_mw': 0.0,
        'p_max_export_mw': 0.0
    }

    is_feasible, total_cv, violations = validate_solution(
        x_for_constraints, bounds, area_params, simulation_results, policy,
        grid_limits, system_config['reserve_fraction'], system_config['lpsp_limit']
    )

    constraints = {
        'is_feasible': is_feasible,
        'total_violation': total_cv,
        **violations
    }

    dispatch_summary = {
        'total_load_mwh': total_load_mwh,
        'total_pv_generation_mwh': total_pv_mwh,
        'total_wind_generation_mwh': total_wind_mwh,
        'total_diesel_generation_mwh': total_diesel_mwh,
        'total_diesel_fuel_mmbtu': total_diesel_fuel_mmbtu,
        'total_deficit_mwh': total_deficit_mwh,
        'deficit_mwh': results_df['deficit_mw'].values,

        'p_load_hourly': results_df['load_mw'].values,
        'p_pv_hourly': results_df['pv_mw'].values,
        'p_wind_hourly': results_df['wind_mw'].values,
        'p_battery_charge_hourly': results_df['battery_charge_mw'].values,
        'p_diesel_hourly': results_df['diesel_mw'].values,
        'p_battery_discharge_hourly': results_df['battery_discharge_mw'].values,
        'soc_hourly': results_df['soc'].values
    }

    return objectives, constraints, dispatch_summary
