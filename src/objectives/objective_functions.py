import numpy as np

def objective_npc(system):

    capital = system['capital_cost_usd']
    fuel_annual = system['fuel_cost_annual_usd']
    om_annual = system['om_cost_annual_usd']
    replacement_cost = system.get('replacement_cost_usd', 0)
    replacement_year = system.get('replacement_year', None)
    discount_rate = system['discount_rate']
    lifetime = system['lifetime_years']

    if discount_rate == 0:
        pwf = lifetime
    else:
        pwf = (1 - (1 + discount_rate) ** (-lifetime)) / discount_rate

    pv_fuel = fuel_annual * pwf
    pv_om = om_annual * pwf

    if replacement_cost and replacement_year:
        pv_replacement = replacement_cost / ((1 + discount_rate) ** replacement_year)
    else:
        pv_replacement = 0.0

    npc = capital + pv_fuel + pv_om + pv_replacement

    return npc

def objective_lpsp(dispatch_results):

    total_deficit = dispatch_results['total_deficit_mwh']
    total_load = dispatch_results['total_load_mwh']

    if total_load == 0:
        return 0.0

    lpsp = total_deficit / total_load

    return lpsp

def objective_co2(dispatch_results, lifetime_years=25):

    fuel_lng = dispatch_results['fuel_lng_mmbtu_annual']
    fuel_diesel = dispatch_results['fuel_diesel_mmbtu_annual']

    emission_factor_lng = 53.06

    emission_factor_diesel = 72.22

    co2_annual_kg = (fuel_lng * emission_factor_lng +
                     fuel_diesel * emission_factor_diesel)

    co2_lifetime_tonnes = (co2_annual_kg * lifetime_years) / 1000.0

    return co2_lifetime_tonnes

def objective_gini(cost_per_hour):

    n = len(cost_per_hour)
    total_cost = np.sum(cost_per_hour)

    if total_cost == 0:
        return 0.0

    sorted_costs = np.sort(cost_per_hour)

    index = np.arange(1, n + 1)

    gini = (2 * np.sum(index * sorted_costs) - (n + 1) * total_cost) / (n * total_cost)

    return gini

def objective_gini_spatial(aggregate_load, renewable_production, n_households=1220, seed=42):

    np.random.seed(seed)

    n_low = int(n_households * 0.40)
    n_mid = int(n_households * 0.40)
    n_high = n_households - n_low - n_mid

    multipliers = np.concatenate([
        np.random.uniform(0.5, 1.0, n_low),
        np.random.uniform(0.8, 1.2, n_mid),
        np.random.uniform(1.2, 2.0, n_high)
    ])
    np.random.shuffle(multipliers)

    total_re = renewable_production.sum()
    total_load = aggregate_load.sum()
    base_re_frac = total_re / total_load if total_load > 0 else 0.0

    re_frac = multipliers * base_re_frac

    re_frac = np.clip(re_frac, 0.0, 1.0)

    if re_frac.sum() == 0:
        return 0.0

    n = len(re_frac)
    sorted_frac = np.sort(re_frac)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_frac) - (n + 1) * re_frac.sum()) / (n * re_frac.sum())

    return max(0.0, min(gini, 1.0))

def objective_gini_burden(
    fuel_cost_annual: float,
    capital_cost_annual: float,
    n_households: int = 1220,
    seed: int = 42,
    burden_cap: float = 0.15
) -> float:

    np.random.seed(seed)

    n_low = int(n_households * 0.40)
    n_mid = int(n_households * 0.40)
    n_high = n_households - n_low - n_mid

    incomes = np.concatenate([
        np.random.uniform(18000, 36000, n_low),
        np.random.uniform(40000, 70000, n_mid),
        np.random.uniform(70000, 110000, n_high)
    ])

    consumption = np.concatenate([
        np.random.uniform(0.7, 1.0, n_low),
        np.random.uniform(0.9, 1.1, n_mid),
        np.random.uniform(1.1, 1.4, n_high)
    ])
    consumption_shares = consumption / consumption.sum()

    fuel_per_hh = fuel_cost_annual / n_households
    capital_per_hh = capital_cost_annual * consumption_shares

    total_cost_per_hh = fuel_per_hh + capital_per_hh

    burden = total_cost_per_hh / incomes

    affordability = np.clip(1.0 - burden, 0.0, 1.0)

    n = len(affordability)
    if affordability.sum() == 0:
        return 1.0

    sorted_aff = np.sort(affordability)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_aff) - (n + 1) * affordability.sum()) / (n * affordability.sum())

    return max(0.0, min(gini, 1.0))

def objective_gini_theja(
    total_re_mwh: float,
    total_load_mwh: float,
    n_households: int = 1220,
    seed: int = 42
) -> float:

    np.random.seed(seed)

    re_ratio = total_re_mwh / total_load_mwh if total_load_mwh > 0 else 0.0

    n_low = int(n_households * 0.40)
    n_mid = int(n_households * 0.40)
    n_high = n_households - n_low - n_mid

    capture_low = np.random.uniform(0.3, 0.6, n_low)
    capture_mid = np.random.uniform(0.7, 1.1, n_mid)
    capture_high = np.random.uniform(1.2, 2.0, n_high)

    capture = np.concatenate([capture_low, capture_mid, capture_high])

    scarcity = np.clip(1.0 - re_ratio, 0.0, 1.0)

    effective_weight = 1.0 + scarcity * (capture - 1.0)

    allocation_shares = effective_weight / effective_weight.sum()

    benefit = n_households * re_ratio * allocation_shares

    benefit = np.clip(benefit, 0.0, 1.0)

    if benefit.sum() == 0:
        return 1.0

    n = len(benefit)
    sorted_benefit = np.sort(benefit)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_benefit) - (n + 1) * benefit.sum()) / (n * benefit.sum())

    return max(0.0, min(gini, 1.0))
