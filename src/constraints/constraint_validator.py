from .constraint_functions_v6 import (
    constraint_bounds,
    constraint_area_v6 as constraint_area,
    constraint_lpsp,
    constraint_spinning_reserve,
    constraint_grid_limits,
    constraint_renewable_cap_v4 as constraint_renewable_cap
)

try:
    from .constraint_functions_v2 import constraint_area_two_bus
    HAS_TWO_BUS = True
except ImportError:
    HAS_TWO_BUS = False

def validate_solution(x, bounds, area_params, simulation_results, policy,
                      grid_limits, reserve_fraction, lpsp_limit, tolerance=1e-6):

    violations = {}

    violations['bounds'] = constraint_bounds(x, bounds)

    if HAS_TWO_BUS and 'area_wind_remote_m2' in area_params:
        violations['area'] = constraint_area_two_bus(x, area_params)
    else:
        violations['area'] = constraint_area(x, area_params)

    violations['lpsp'] = constraint_lpsp(simulation_results, lpsp_limit)

    system_for_reserve = {
        'p_diesel_online_mw': x.get('p_diesel_online_mw', 0.0),
        'p_battery_discharge_mw': x.get('p_battery_discharge_mw', 0.0),
        'p_load_avg_mw': x.get('p_load_avg_mw', 0.0)
    }
    violations['spinning_reserve'] = constraint_spinning_reserve(
        system_for_reserve, reserve_fraction
    )

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

    total_cv = calculate_total_violation(violations)

    is_feasible = (total_cv <= tolerance)

    return is_feasible, total_cv, violations

def calculate_total_violation(violations):

    total_cv = sum(max(v, 0.0) for v in violations.values())

    return total_cv

def get_violated_constraints(violations, tolerance=1e-6):

    violated = [name for name, value in violations.items() if value > tolerance]

    return violated
