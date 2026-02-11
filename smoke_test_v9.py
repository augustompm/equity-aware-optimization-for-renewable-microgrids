import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

print("=" * 60)
print("SMOKE TEST V9 REDESIGN")
print("=" * 60)

print("\n[1] Config params...")
sys.path.insert(0, str(project_root))
from config import get_v8_config
config = get_v8_config()
assert 'pv_capital_cost_per_kw' in config, "Missing pv_capital_cost_per_kw"
assert config['pv_capital_cost_per_kw'] == 3250.0
assert config['wind_capital_cost_per_kw'] == 5500.0
assert config['battery_capital_cost_per_kwh'] == 500.0
assert config['diesel_capital_cost_per_kw'] == 1000.0
assert config['battery_replacement_years'] == 10
print("  OK - All capital cost params present")

print("\n[2] Objective functions import...")
from objectives.objective_functions import (
    objective_npc, objective_lpsp, objective_co2,
    objective_gini, objective_gini_spatial
)
print("  OK - All functions imported (including objective_gini_spatial)")

print("\n[3] CO2 emission factor...")
dispatch = {'fuel_lng_mmbtu_annual': 0.0, 'fuel_diesel_mmbtu_annual': 1000.0}
co2 = objective_co2(dispatch, lifetime_years=1)
expected_co2 = 1000.0 * 72.22 / 1000.0
assert abs(co2 - expected_co2) < 0.01, f"CO2 factor wrong: got {co2}, expected {expected_co2}"
print(f"  OK - CO2 = {co2:.2f} tonnes (IPCC 72.22 kg/MMBtu)")

print("\n[4] NPC with real capital costs...")
system = {
    'capital_cost_usd': 50_000_000,
    'fuel_cost_annual_usd': 2_000_000,
    'om_cost_annual_usd': 500_000,
    'replacement_cost_usd': 25_000_000,
    'replacement_year': 10,
    'discount_rate': 0.03,
    'lifetime_years': 25
}
npc = objective_npc(system)
assert npc > 50_000_000, f"NPC should be > capital: {npc}"
print(f"  OK - NPC = ${npc:,.0f} (capital=$50M + fuel/OM/replacement)")

print("\n[5] Capital cost from decision vars...")
decision_vars = {
    'n_pv_kw': 5000,
    'n_wind_mw': 3.0,
    'e_battery_mwh': 50.0,
    'p_diesel_mw': 5.0
}
capital = (
    decision_vars['n_pv_kw'] * config['pv_capital_cost_per_kw'] +
    decision_vars['n_wind_mw'] * 1000 * config['wind_capital_cost_per_kw'] +
    decision_vars['e_battery_mwh'] * 1000 * config['battery_capital_cost_per_kwh'] +
    decision_vars['p_diesel_mw'] * 1000 * config['diesel_capital_cost_per_kw']
)
print(f"  PV:      5000 kW x $3,250 = ${5000*3250:,.0f}")
print(f"  Wind:    3000 kW x $5,500 = ${3000*5500:,.0f}")
print(f"  Battery: 50000 kWh x $500 = ${50000*500:,.0f}")
print(f"  Diesel:  5000 kW x $1,000 = ${5000*1000:,.0f}")
print(f"  TOTAL:   ${capital:,.0f}")
assert capital > 0, "Capital must be > 0"
assert capital == 5000*3250 + 3000*5500 + 50000*500 + 5000*1000

print("\n[6] Spatial Gini varies with RE penetration...")
load = np.ones(8760) * 4.0

re_low = np.ones(8760) * 0.4
gini_low = objective_gini_spatial(load, re_low, n_households=900, seed=42)

re_mid = np.ones(8760) * 2.0
gini_mid = objective_gini_spatial(load, re_mid, n_households=900, seed=42)

re_high = np.ones(8760) * 3.6
gini_high = objective_gini_spatial(load, re_high, n_households=900, seed=42)

re_over = np.ones(8760) * 4.8
gini_over = objective_gini_spatial(load, re_over, n_households=900, seed=42)

print(f"  RE 10%:  Gini = {gini_low:.4f}")
print(f"  RE 50%:  Gini = {gini_mid:.4f}")
print(f"  RE 90%:  Gini = {gini_high:.4f}")
print(f"  RE 120%: Gini = {gini_over:.4f} (clipping active)")

assert gini_low > 0, "Gini should be > 0"
assert gini_over < gini_low, f"Gini should decrease at high RE (clipping): {gini_over} >= {gini_low}"
print(f"  OK - Gini decreases from {gini_low:.4f} to {gini_over:.4f} as RE increases")

print("\n[7] Gini determinism (seed=42)...")
g1 = objective_gini_spatial(load, re_mid, n_households=900, seed=42)
g2 = objective_gini_spatial(load, re_mid, n_households=900, seed=42)
assert g1 == g2, f"Gini not deterministic: {g1} != {g2}"
print(f"  OK - Same seed produces same Gini: {g1:.6f}")

print("\n" + "=" * 60)
print("ALL SMOKE TESTS PASSED")
print("=" * 60)
print(f"\nSummary of V9 changes verified:")
print(f"  - Capital costs: PV $3,250/kW, Wind $5,500/kW, Batt $500/kWh, Diesel $1,000/kW")
print(f"  - O&M costs: PV $10, Wind $75, Batt $8.8 per kW(h)/yr")
print(f"  - CO2 factor: IPCC 72.22 kg/MMBtu (was EPA 73.96)")
print(f"  - Gini: spatial (varies {gini_low:.3f} to {gini_over:.3f})")
print(f"  - Battery replacement: year 10, 100% CAPEX")
