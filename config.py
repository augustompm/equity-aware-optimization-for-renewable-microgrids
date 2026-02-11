from pathlib import Path

def get_v8_config():
    project_root = Path(__file__).parent
    return {
        'load_profile_path': project_root / 'data' / 'load-profile-8760h.csv',
        'solar_cf_path': project_root / 'data' / 'solar-capacity-factors.csv',
        'wind_cf_path': project_root / 'data' / 'wind-capacity-factors-highpoint-clean.csv',
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
        'area_available_pv_m2': 500000.0,
        'area_available_wind_m2': 3000000.0,
        'renewable_fraction_max': 1.00,
        'reserve_fraction': 0.15,
        'lpsp_limit': 0.05,
        'grid_connected': False,
        'bounds': {
            'pv_kw': (0, 10000),
            'wind_kw': (0, 5),
            'battery_kwh': (0, 100),
            'diesel_kw': (0, 10)
        },
        'calculate_re_penetration': True,
        'calculate_excess_power': True,
        'calculate_lcoe': True,
        'calculate_fuel_consumption': True,
        'community_name': 'Inuvik',
        'population': 3500,
        'latitude': 68.36,
        'benchmark_community': 'Sachs Harbour',
        'benchmark_population': 130,
        'benchmark_battery_kwh': 4500,
        'benchmark_re_pct': (71.64, 85.87),
        'pv_capital_cost_per_kw': 3250.0,
        'wind_capital_cost_per_kw': 5500.0,
        'battery_capital_cost_per_kwh': 500.0,
        'diesel_capital_cost_per_kw': 1000.0,
        'pv_om_cost_per_kw_yr': 10.0,
        'wind_om_cost_per_kw_yr': 75.0,
        'battery_om_cost_per_kwh_yr': 8.8,
        'battery_replacement_years': 10,
        'battery_replacement_fraction': 1.0,
    }

def get_v8_bounds():
    return {
        'pv_kw': (0, 10000),
        'wind_kw': (0, 5),
        'battery_kwh': (0, 100),
        'diesel_kw': (0, 10)
    }

def get_v8_nsga3_params():
    return {
        'population_size': 165,
        'reference_points_p': 8,
        'n_generations': 200,
        'crossover_prob': 0.9,
        'mutation_prob': 0.1
    }
