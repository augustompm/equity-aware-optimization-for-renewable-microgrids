def constraint_bounds(x, bounds):

    violation = 0.0

    for var_name, (lower, upper) in bounds.items():
        if var_name not in x:
            continue

        value = x[var_name]

        if value < lower:
            violation += abs(value - lower)

        if value > upper:
            violation += abs(value - upper)

    return violation

def constraint_area(x, area_params):

    area_pv = x.get('n_pv_kw', 0) * area_params['area_pv_per_kw']
    area_wind = x.get('n_wind_mw', 0) * area_params['area_wind_per_mw']
    area_battery = x.get('e_battery_mwh', 0) * area_params['area_battery_per_mwh']

    area_total = area_pv + area_wind + area_battery
    area_limit = area_params['area_available_m2']

    violation = max(area_total - area_limit, 0.0)

    return violation

def constraint_lpsp(simulation_results, lpsp_limit=0.05):

    lpsp = simulation_results.get('lpsp', 0.0)

    violation = max(lpsp - lpsp_limit, 0.0)

    return violation

def constraint_spinning_reserve(system, reserve_fraction=0.15):

    p_load_avg = system.get('p_load_avg_mw', 0.0)
    reserve_required = reserve_fraction * p_load_avg

    p_diesel_online = system.get('p_diesel_online_mw', 0.0)
    p_battery_discharge = system.get('p_battery_discharge_mw', 0.0)

    reserve_available = p_diesel_online + p_battery_discharge - p_load_avg

    violation = max(reserve_required - reserve_available, 0.0)

    return violation

def constraint_grid_limits(system, grid_limits):

    p_grid_buy = system.get('p_grid_buy_mw', 0.0)
    p_grid_sell = system.get('p_grid_sell_mw', 0.0)

    p_max_import = grid_limits['p_max_import_mw']
    p_max_export = grid_limits['p_max_export_mw']

    violation_import = max(p_grid_buy - p_max_import, 0.0)
    violation_export = max(p_grid_sell - p_max_export, 0.0)

    violation = violation_import + violation_export

    return violation

def constraint_renewable_cap(system, policy):

    p_pv_kw = system.get('p_pv_installed_kw', 0.0)
    p_wind_mw = system.get('p_wind_installed_mw', 0.0)
    p_load_avg = system.get('p_load_avg_mw', 0.0)

    renewable_total_mw = (p_pv_kw / 1000.0) + p_wind_mw

    renewable_cap_mw = policy['renewable_fraction_max'] * p_load_avg

    violation = max(renewable_total_mw - renewable_cap_mw, 0.0)

    return violation
