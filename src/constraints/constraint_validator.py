from .constraint_functions import (
    constraint_bounds,
    constraint_area,
    constraint_lpsp,
    constraint_spinning_reserve,
    constraint_grid_limits,
    constraint_renewable_cap
)

def validate_solution(x, bounds, area_params, simulation_results, policy,
                      grid_limits, reserve_fraction, lpsp_limit, tolerance=1e-6):
    violations = {}
    violations['bounds'] = constraint_bounds(x, bounds)
    violations['area'] = constraint_area(x, area_params)
    violations['lpsp'] = constraint_lpsp(simulation_results, lpsp_limit)

    system_for_reserve = {
        'p_diesel_online_mw': x.get('p_diesel_online_mw', 0.0),
        'p_battery_discharge_mw': x.get('p_battery_discharge_mw', 0.0),
        'p_load_avg_mw': x.get('p_load_avg_mw', 0.0)
    }
    violations['spinning_reserve'] = constraint_spinning_reserve(system_for_reserve, reserve_fraction)

    system_for_grid = {
        'p_grid_buy_mw': x.get('p_grid_buy_mw', 0.0),
        'p_grid_sell_mw': x.get('p_grid_sell_mw', 0.0),
        'grid_connected': x.get('grid_connected', False)
    }
    violations['grid_limits'] = constraint_grid_limits(system_for_grid, grid_limits)

    system_for_renewable = {
        'p_pv_installed_kw': x.get('p_pv_installed_kw', 0.0),
        'p_wind_installed_mw': x.get('p_wind_installed_mw', 0.0),
        'p_load_avg_mw': x.get('p_load_avg_mw', 0.0)
    }
    violations['renewable_cap'] = constraint_renewable_cap(system_for_renewable, policy)

    total_cv = sum(max(v, 0.0) for v in violations.values())
    is_feasible = (total_cv <= tolerance)
    return is_feasible, total_cv, violations
