import numpy as np

def objective_npc_v2(system):

    capital = system['capital_cost_usd']
    fuel_annual = system['fuel_cost_annual_usd']
    om_annual = system['om_cost_annual_usd']
    discount_rate = system['discount_rate']
    lifetime = system['lifetime_years']

    component_costs = system.get('component_costs_usd', {})
    component_lifetimes = system.get('component_lifetimes', {
        'pv': 25, 'wind': 20, 'battery': 5, 'diesel': 20
    })

    n_wind_mw = system.get('n_wind_mw', 0.0)
    hub_height_m = system.get('hub_height_m', 100)

    if discount_rate == 0:
        pwf = lifetime
    else:
        pwf = (1 - (1 + discount_rate) ** (-lifetime)) / discount_rate

    pv_fuel = fuel_annual * pwf
    pv_om = om_annual * pwf

    capital_tower = 0.0
    om_tower_annual = 0.0

    if n_wind_mw > 0:
        turbine_capacity_mw = 1.91
        n_turbines = n_wind_mw / turbine_capacity_mw

        capital_tower = n_turbines * hub_height_m * 250
        om_tower_annual = n_turbines * hub_height_m * 2.5

    pv_om_tower = om_tower_annual * pwf

    pv_replacements = 0.0

    for component, cost in component_costs.items():
        lifetime_component = component_lifetimes.get(component, 25)

        if lifetime_component >= lifetime:
            continue

        replacement_years = range(lifetime_component, lifetime, lifetime_component)

        for year in replacement_years:
            pv_replacement = cost / ((1 + discount_rate) ** year)
            pv_replacements += pv_replacement

    salvage_value = 0.0

    for component, cost in component_costs.items():
        lifetime_component = component_lifetimes.get(component, 25)

        if component == 'battery':
            last_replacement_year = (lifetime // lifetime_component) * lifetime_component
            age_at_end = lifetime - last_replacement_year
        else:
            age_at_end = lifetime % lifetime_component if lifetime > lifetime_component else lifetime

        if age_at_end < lifetime_component:
            remaining_life_fraction = (lifetime_component - age_at_end) / lifetime_component
            component_salvage = cost * remaining_life_fraction

            pv_salvage = component_salvage / ((1 + discount_rate) ** lifetime)
            salvage_value += pv_salvage

    npc = (capital + capital_tower + pv_fuel + pv_om + pv_om_tower
           + pv_replacements - salvage_value)

    return npc

def objective_lpsp_v2(dispatch_results):

    total_deficit = dispatch_results['total_deficit_mwh']
    total_load = dispatch_results['total_load_mwh']

    if total_load == 0:
        return 0.0

    lpsp = total_deficit / total_load

    return lpsp

def objective_co2_v2(dispatch_results):

    fuel_diesel = dispatch_results.get('fuel_diesel_mmbtu_annual', 0.0)
    fuel_lng = dispatch_results.get('fuel_lng_mmbtu_annual', 0.0)
    lifetime_years = dispatch_results.get('lifetime_years', 25)

    emission_factor_diesel_hhv = 73.96
    emission_factor_lng_hhv = 53.06

    hhv_to_lhv_diesel = 1.06
    hhv_to_lhv_lng = 1.10

    emission_factor_diesel = emission_factor_diesel_hhv / hhv_to_lhv_diesel
    emission_factor_lng = emission_factor_lng_hhv / hhv_to_lhv_lng

    co2_annual_kg = (fuel_diesel * emission_factor_diesel +
                     fuel_lng * emission_factor_lng)

    co2_lifetime_tonnes = (co2_annual_kg * lifetime_years) / 1000.0

    return co2_lifetime_tonnes

def objective_gini_spatial_v2(household_renewable_fractions):

    n = len(household_renewable_fractions)
    total = np.sum(household_renewable_fractions)

    if total == 0 or n < 2:
        return 0.0

    sorted_fractions = np.sort(household_renewable_fractions)

    index = np.arange(1, n + 1)

    gini = ((2 * np.sum(index * sorted_fractions) - (n + 1) * total)
            / (n * total))

    return max(0.0, min(gini, 1.0))
