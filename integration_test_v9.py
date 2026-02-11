import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from config import get_v8_config
from simulation.system_simulator import simulate_system

print("=" * 60)
print("INTEGRATION TEST V9 - FULL SIMULATION")
print("=" * 60)

config = get_v8_config()

decision_vars = {
    'n_pv_kw': 3000,
    'n_wind_mw': 2.0,
    'e_battery_mwh': 30.0,
    'p_diesel_mw': 6.0
}

print(f"\nDecision vars: PV={decision_vars['n_pv_kw']}kW, "
      f"Wind={decision_vars['n_wind_mw']}MW, "
      f"Batt={decision_vars['e_battery_mwh']}MWh, "
      f"Diesel={decision_vars['p_diesel_mw']}MW")

print("\nRunning 8760-hour simulation...")
objectives, constraints, dispatch = simulate_system(decision_vars, config)

print("\n--- OBJECTIVES ---")
print(f"  NPC:  ${objectives['npc']:,.0f}")
print(f"  LPSP: {objectives['lpsp']:.4f} ({objectives['lpsp']*100:.2f}%)")
print(f"  CO2:  {objectives['co2']:,.0f} tonnes")
print(f"  Gini: {objectives['gini']:.4f}")

print("\n--- CONSTRAINTS ---")
print(f"  Feasible: {constraints['is_feasible']}")
print(f"  Total violation: {constraints['total_violation']:.4f}")
for k, v in constraints.items():
    if k not in ('is_feasible', 'total_violation'):
        print(f"  {k}: {v:.4f}")

print("\n--- DISPATCH SUMMARY ---")
total_load = dispatch['total_load_mwh']
total_pv = dispatch['total_pv_generation_mwh']
total_wind = dispatch['total_wind_generation_mwh']
total_diesel = dispatch['total_diesel_generation_mwh']
total_deficit = dispatch['total_deficit_mwh']
re_pct = (total_pv + total_wind) / total_load * 100 if total_load > 0 else 0

print(f"  Total load:   {total_load:,.0f} MWh")
print(f"  PV gen:       {total_pv:,.0f} MWh")
print(f"  Wind gen:     {total_wind:,.0f} MWh")
print(f"  Diesel gen:   {total_diesel:,.0f} MWh")
print(f"  Deficit:      {total_deficit:,.0f} MWh")
print(f"  RE pct:       {re_pct:.1f}%")

capital_expected = (
    decision_vars['n_pv_kw'] * config['pv_capital_cost_per_kw'] +
    decision_vars['n_wind_mw'] * 1000 * config['wind_capital_cost_per_kw'] +
    decision_vars['e_battery_mwh'] * 1000 * config['battery_capital_cost_per_kwh'] +
    decision_vars['p_diesel_mw'] * 1000 * config['diesel_capital_cost_per_kw']
)
print(f"\n  Expected capital: ${capital_expected:,.0f}")
print(f"  NPC > capital?   {objectives['npc'] > capital_expected} "
      f"(NPC=${objectives['npc']:,.0f} vs capital=${capital_expected:,.0f})")

print("\n--- VALIDATIONS ---")
errors = []

if objectives['npc'] <= 0:
    errors.append("NPC <= 0")
if objectives['npc'] <= capital_expected:
    errors.append(f"NPC ({objectives['npc']:,.0f}) <= capital ({capital_expected:,.0f})")
if objectives['lpsp'] < 0 or objectives['lpsp'] > 1:
    errors.append(f"LPSP out of range: {objectives['lpsp']}")
if objectives['co2'] <= 0:
    errors.append("CO2 <= 0")
if objectives['gini'] <= 0 or objectives['gini'] >= 1:
    errors.append(f"Gini out of range: {objectives['gini']}")

print(f"  NPC/CO2 ratio: {objectives['npc']/objectives['co2']:,.2f}")
print(f"  (V8 had constant 188.35 - this should be different)")

if len(errors) == 0:
    print("\n  ALL VALIDATIONS PASSED")
else:
    print(f"\n  ERRORS: {errors}")

print("\n\n--- SECOND TEST: DIESEL-HEAVY ---")
dv2 = {
    'n_pv_kw': 100,
    'n_wind_mw': 0.1,
    'e_battery_mwh': 1.0,
    'p_diesel_mw': 8.0
}
obj2, con2, dis2 = simulate_system(dv2, config)
print(f"  NPC:  ${obj2['npc']:,.0f}")
print(f"  CO2:  {obj2['co2']:,.0f} tonnes")
print(f"  Gini: {obj2['gini']:.4f}")
print(f"  NPC/CO2: {obj2['npc']/obj2['co2']:,.2f}")

print(f"\n  Ratio difference: {abs(objectives['npc']/objectives['co2'] - obj2['npc']/obj2['co2']):,.2f}")
if abs(objectives['npc']/objectives['co2'] - obj2['npc']/obj2['co2']) > 1.0:
    print("  NPC-CO2 DECOUPLED (ratios differ)")
else:
    print("  WARNING: NPC-CO2 still may be coupled")

print("\n" + "=" * 60)
print("INTEGRATION TEST COMPLETE")
print("=" * 60)
