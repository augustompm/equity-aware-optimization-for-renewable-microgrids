import pandas as pd
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from simulation.system_simulator import simulate_system

def get_baseline_decision_vars():

    return {
        'n_pv_kw': 0,
        'n_wind_mw': 0,
        'e_battery_mwh': 0,
        'p_diesel_mw': 7.71
    }

def get_baseline_system_config():

    return {
        'load_profile_path': project_root / 'data' / 'load-profile-8760h.csv',
        'solar_cf_path': project_root / 'data' / 'solar-capacity-factors.csv',
        'wind_cf_path': project_root / 'data' / 'wind-capacity-factors.csv',
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
        'area_available_m2': 100000.0,
        'renewable_fraction_max': 0.20,
        'reserve_fraction': 0.15,
        'lpsp_limit': 0.05,
        'grid_connected': False
    }

def run_baseline():

    print("=" * 80)
    print("BASELINE SYSTEM EVALUATION")
    print("Arctic Microgrid Optimization - Diesel-Only Reference")
    print("=" * 80)
    print()

    decision_vars = get_baseline_decision_vars()
    system_config = get_baseline_system_config()

    print("Baseline Configuration:")
    print(f"  PV:      {decision_vars['n_pv_kw']:8.1f} kW   (no solar generation)")
    print(f"  Wind:    {decision_vars['n_wind_mw']:8.2f} MW   (no wind generation)")
    print(f"  Battery: {decision_vars['e_battery_mwh']:8.2f} MWh  (no energy storage)")
    print(f"  Diesel:  {decision_vars['p_diesel_mw']:8.2f} MW   (all LNG generators)")
    print()
    print("Running 8760-hour annual simulation...")
    print()

    objectives, constraints, dispatch_summary = simulate_system(
        decision_vars, system_config
    )

    print("=" * 80)
    print("BASELINE RESULTS")
    print("=" * 80)
    print()

    print("Objectives:")
    print(f"  NPC:  ${objectives['npc']:15,.2f}  (Net Present Cost)")
    print(f"  LPSP:  {objectives['lpsp']:15.4f}  (Loss of Power Supply Probability)")
    print(f"  CO2:   {objectives['co2']:15,.0f} kg  (Total Emissions)")
    print(f"  Gini:  {objectives['gini']:15.4f}  (Cost Equity)")
    print()

    print("Dispatch Summary:")
    print(f"  Total Load:             {dispatch_summary['total_load_mwh']:12,.1f} MWh")
    print(f"  Total Diesel Generation:{dispatch_summary['total_diesel_generation_mwh']:12,.1f} MWh")
    print(f"  Total Diesel Fuel:      {dispatch_summary['total_diesel_fuel_mmbtu']:12,.1f} MMBtu")
    print(f"  Total Unmet Load:       {dispatch_summary['total_deficit_mwh']:12,.1f} MWh")
    print()

    print("Renewable Generation:")
    print(f"  PV Generation:          {dispatch_summary['total_pv_generation_mwh']:12,.1f} MWh")
    print(f"  Wind Generation:        {dispatch_summary['total_wind_generation_mwh']:12,.1f} MWh")
    print()

    feasibility = "FEASIBLE" if constraints['is_feasible'] else "INFEASIBLE"
    print(f"Feasibility: {feasibility}")
    print(f"Total Constraint Violation: {constraints['total_violation']:.6f}")
    print()

    print("=" * 80)

    results = {
        'objectives': objectives,
        'constraints': constraints,
        'dispatch': dispatch_summary,
        'decision_vars': decision_vars
    }

    return results

def save_baseline_results(results, output_file=None):

    if output_file is None:
        results_dir = project_root / "results"
        results_dir.mkdir(exist_ok=True)
        output_file = results_dir / "baseline-results.csv"

    data = {
        'Configuration': ['Baseline Diesel-Only'],
        'n_pv_kw': [results['decision_vars']['n_pv_kw']],
        'n_wind_mw': [results['decision_vars']['n_wind_mw']],
        'e_battery_mwh': [results['decision_vars']['e_battery_mwh']],
        'p_diesel_mw': [results['decision_vars']['p_diesel_mw']],
        'NPC': [results['objectives']['npc']],
        'LPSP': [results['objectives']['lpsp']],
        'CO2': [results['objectives']['co2']],
        'Gini': [results['objectives']['gini']],
        'total_load_mwh': [results['dispatch']['total_load_mwh']],
        'total_diesel_mwh': [results['dispatch']['total_diesel_generation_mwh']],
        'total_diesel_fuel_mmbtu': [results['dispatch']['total_diesel_fuel_mmbtu']],
        'feasible': [results['constraints']['is_feasible']]
    }

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

    print(f"Baseline results saved to {output_file}")

def compare_with_pareto_front(baseline_results):

    pareto_csv = project_root / "results" / "pareto-front.csv"

    if not pareto_csv.exists():
        print()
        print("Note: No Pareto front results available for comparison.")
        print("Run NSGA-III optimization to generate Pareto solutions.")
        return

    df_pareto = pd.read_csv(pareto_csv)

    print()
    print("=" * 80)
    print("COMPARISON WITH OPTIMIZED SOLUTIONS")
    print("=" * 80)
    print()

    baseline_npc = baseline_results['objectives']['npc']
    baseline_co2 = baseline_results['objectives']['co2']
    baseline_fuel = baseline_results['dispatch']['total_diesel_fuel_mmbtu']

    best_npc_idx = df_pareto['NPC'].idxmin()
    best_npc_solution = df_pareto.loc[best_npc_idx]

    print("Best NPC Solution (from Pareto front):")
    print(f"  Configuration: {best_npc_solution['n_pv_kw']:.0f} kW PV, "
          f"{best_npc_solution['n_wind_mw']:.2f} MW wind, "
          f"{best_npc_solution['e_battery_mwh']:.2f} MWh battery")
    print(f"  NPC: ${best_npc_solution['NPC']:,.2f}")
    print()

    npc_savings = baseline_npc - best_npc_solution['NPC']
    npc_savings_pct = (npc_savings / baseline_npc) * 100

    print(f"NPC Savings vs Baseline: ${npc_savings:,.2f} ({npc_savings_pct:.1f}%)")
    print()

    best_co2_idx = df_pareto['CO2'].idxmin()
    best_co2_solution = df_pareto.loc[best_co2_idx]

    co2_reduction = baseline_co2 - best_co2_solution['CO2']
    co2_reduction_pct = (co2_reduction / baseline_co2) * 100

    print(f"CO2 Emissions Reduction: {co2_reduction:,.0f} kg ({co2_reduction_pct:.1f}%)")
    print()

    print("Summary:")
    print(f"  Baseline: {baseline_npc / 1e6:.1f}M USD NPC, {baseline_co2 / 1e3:.0f}k kg CO2")
    print(f"  Optimized: {best_npc_solution['NPC'] / 1e6:.1f}M USD NPC, "
          f"{best_co2_solution['CO2'] / 1e3:.0f}k kg CO2")
    print()
    print("=" * 80)

if __name__ == "__main__":

    baseline_results = run_baseline()

    save_baseline_results(baseline_results)

    compare_with_pareto_front(baseline_results)
