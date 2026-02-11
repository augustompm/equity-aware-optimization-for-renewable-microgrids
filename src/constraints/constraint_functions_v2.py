from .constraint_functions import (
    constraint_bounds,
    constraint_lpsp,
    constraint_spinning_reserve,
    constraint_grid_limits,
    constraint_renewable_cap
)

def constraint_area_two_bus(x, area_params):

    area_pv = x.get('n_pv_kw', 0) * area_params['area_pv_per_kw']
    area_battery = x.get('e_battery_mwh', 0) * area_params['area_battery_per_mwh']
    area_wind = x.get('n_wind_mw', 0) * area_params['area_wind_per_mw']

    if 'area_wind_remote_m2' in area_params and area_params['area_wind_remote_m2'] > 0:
        area_urban_total = area_pv + area_battery
        area_urban_limit = area_params['area_available_m2']
        violation_urban = max(area_urban_total - area_urban_limit, 0.0)

        area_wind_limit = area_params['area_wind_remote_m2']
        violation_wind = max(area_wind - area_wind_limit, 0.0)

        total_violation = violation_urban + violation_wind

    else:
        area_total = area_pv + area_wind + area_battery
        area_limit = area_params['area_available_m2']
        total_violation = max(area_total - area_limit, 0.0)

    return total_violation
