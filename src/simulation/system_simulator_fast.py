import numpy as np
from pathlib import Path

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from simulation.data_cache import get_data_cache
from objectives.objective_functions import (
    objective_npc,
    objective_lpsp,
    objective_co2,
    objective_gini_theja
)
from constraints.constraint_validator import validate_solution

def simulate_system_fast(decision_vars, system_config, data_cache=None):

    if data_cache is None:
        data_cache = get_data_cache()
        data_cache.initialize(system_config)

    load_mw, solar_cf, wind_cf, temperature_c = data_cache.get_arrays()

    pv_kw = decision_vars['n_pv_kw']
    wind_mw = decision_vars['n_wind_mw']
    battery_mwh = decision_vars['e_battery_mwh']
    diesel_mw = decision_vars['p_diesel_mw']

    battery_efficiency = system_config['battery_efficiency']
    battery_c_rate = system_config['battery_c_rate']
    battery_dod_max = system_config['battery_dod_max']
    diesel_efficiency = system_config['diesel_efficiency']
    diesel_min_load = system_config['diesel_min_load_fraction']
    fuel_cost_per_mmbtu = system_config['diesel_fuel_cost_per_mmbtu']

    temp_coeff = -0.004
    pv_gen_mw = (pv_kw / 1000.0) * solar_cf * (1.0 + temp_coeff * (temperature_c - 25.0))
    pv_gen_mw = np.maximum(pv_gen_mw, 0.0)

    wind_gen_mw = wind_mw * wind_cf

    renewable_mw = pv_gen_mw + wind_gen_mw

    if battery_mwh > 0:
        battery_p_max = battery_mwh * battery_c_rate
        battery_usable = battery_mwh * battery_dod_max
        soc_min = 1.0 - battery_dod_max
    else:
        battery_p_max = 0.0
        battery_usable = 0.0
        soc_min = 0.0

    n_hours = len(load_mw)
    diesel_gen_mw = np.zeros(n_hours)
    diesel_fuel_mmbtu = np.zeros(n_hours)
    battery_charge_mw = np.zeros(n_hours)
    battery_discharge_mw = np.zeros(n_hours)
    deficit_mw = np.zeros(n_hours)
    soc = np.zeros(n_hours)

    current_soc = 0.5

    for t in range(n_hours):
        load_t = load_mw[t]
        re_t = renewable_mw[t]

        if re_t >= load_t:

            excess = re_t - load_t
            if battery_mwh > 0 and current_soc < 1.0:
                charge_capacity = min(battery_p_max, (1.0 - current_soc) * battery_mwh / battery_efficiency)
                actual_charge = min(excess, charge_capacity)
                battery_charge_mw[t] = actual_charge
                current_soc += (actual_charge * battery_efficiency) / battery_mwh
                current_soc = min(current_soc, 1.0)
        else:

            shortfall = load_t - re_t

            if battery_mwh > 0 and current_soc > soc_min:
                available_energy = (current_soc - soc_min) * battery_mwh
                discharge_capacity = min(battery_p_max, available_energy)
                actual_discharge = min(shortfall, discharge_capacity)
                battery_discharge_mw[t] = actual_discharge
                current_soc -= actual_discharge / battery_mwh
                current_soc = max(current_soc, soc_min)
                shortfall -= actual_discharge

            if shortfall > 0 and diesel_mw > 0:
                diesel_output = min(shortfall, diesel_mw)
                diesel_gen_mw[t] = diesel_output

                heat_rate = 3.412 / diesel_efficiency
                fuel = diesel_output * heat_rate * 1.0
                diesel_fuel_mmbtu[t] = fuel

                shortfall -= diesel_output

            if shortfall > 0:
                deficit_mw[t] = shortfall

        soc[t] = current_soc

    total_load_mwh = load_mw.sum()
    total_pv_mwh = pv_gen_mw.sum()
    total_wind_mwh = wind_gen_mw.sum()
    total_diesel_mwh = diesel_gen_mw.sum()
    total_diesel_fuel = diesel_fuel_mmbtu.sum()
    total_deficit_mwh = deficit_mw.sum()

    capital_cost = (
        pv_kw * system_config['pv_capital_cost_per_kw'] +
        wind_mw * 1000 * system_config['wind_capital_cost_per_kw'] +
        battery_mwh * 1000 * system_config['battery_capital_cost_per_kwh'] +
        diesel_mw * 1000 * system_config['diesel_capital_cost_per_kw']
    )
    fuel_cost_annual = total_diesel_fuel * fuel_cost_per_mmbtu
    om_cost_annual = (
        pv_kw * system_config['pv_om_cost_per_kw_yr'] +
        wind_mw * 1000 * system_config['wind_om_cost_per_kw_yr'] +
        battery_mwh * 1000 * system_config['battery_om_cost_per_kwh_yr']
    )
    replacement_cost = (
        battery_mwh * 1000 *
        system_config['battery_capital_cost_per_kwh'] *
        system_config['battery_replacement_fraction']
    )

    system_for_npc = {
        'capital_cost_usd': capital_cost,
        'fuel_cost_annual_usd': fuel_cost_annual,
        'om_cost_annual_usd': om_cost_annual,
        'replacement_cost_usd': replacement_cost,
        'replacement_year': system_config['battery_replacement_years'],
        'discount_rate': system_config['discount_rate'],
        'lifetime_years': system_config['lifetime_years']
    }
    npc_value = objective_npc(system_for_npc)

    lpsp_value = total_deficit_mwh / total_load_mwh if total_load_mwh > 0 else 0.0

    dispatch_for_co2 = {
        'fuel_lng_mmbtu_annual': 0.0,
        'fuel_diesel_mmbtu_annual': total_diesel_fuel
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
        'p_diesel_online_mw': diesel_mw,
        'p_battery_discharge_mw': battery_p_max,
        'p_load_avg_mw': total_load_mwh / 8760.0,
        'p_pv_installed_kw': pv_kw,
        'p_wind_installed_mw': wind_mw,
        'p_grid_buy_mw': 0.0,
        'p_grid_sell_mw': 0.0,
        'grid_connected': system_config['grid_connected']
    }

    bounds = system_config.get('bounds', {
        'n_pv_kw': (0, 10000),
        'n_wind_mw': (0, 5),
        'e_battery_mwh': (0, 100),
        'p_diesel_mw': (0, 10)
    })

    area_params = {
        'area_pv_per_kw': 2.0,
        'area_wind_per_mw': 186050.0,
        'area_battery_per_mwh': 10.0,
        'area_available_pv_m2': system_config['area_available_pv_m2'],
        'area_available_wind_m2': system_config['area_available_wind_m2']
    }

    simulation_results = {'lpsp': lpsp_value}
    policy = {'renewable_fraction_max': system_config['renewable_fraction_max']}
    grid_limits = {'p_max_import_mw': 0.0, 'p_max_export_mw': 0.0}

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
        'total_diesel_fuel_mmbtu': total_diesel_fuel,
        'total_deficit_mwh': total_deficit_mwh,
        'deficit_mwh': deficit_mw,
        'p_load_hourly': load_mw,
        'p_pv_hourly': pv_gen_mw,
        'p_wind_hourly': wind_gen_mw,
        'p_battery_charge_hourly': battery_charge_mw,
        'p_diesel_hourly': diesel_gen_mw,
        'p_battery_discharge_hourly': battery_discharge_mw,
        'soc_hourly': soc
    }

    return objectives, constraints, dispatch_summary
