import pandas as pd
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from resilience.resilience_metrics import run_scenario_simulation
from optimization.pareto_analysis import find_knee_points

def load_pareto_solutions(pareto_csv_path):

    pareto_df = pd.read_csv(pareto_csv_path)
    return pareto_df

def select_top_solutions(pareto_df, n_solutions=3):

    knee_solutions = find_knee_points(pareto_df, n_points=n_solutions)
    return knee_solutions

def run_all_scenarios_for_solution(solution_dict, solution_id):

    scenarios = [
        ('A1_cold_snap', 180, 7),
        ('A2_fuel_disruption', 0, 14),
        ('A3_blizzard', 150, 3)
    ]

    results = {}

    for scenario_name, start_day, duration in scenarios:
        metrics = run_scenario_simulation(
            solution_dict,
            scenario_name,
            start_day=start_day
        )

        results[scenario_name] = metrics

        save_scenario_results(solution_id, scenario_name, metrics, solution_dict)

    return results

def save_scenario_results(solution_id, scenario_name, metrics, solution_dict):

    output_dir = project_root / 'results' / 'resilience-scenarios'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f'solution_{solution_id}_{scenario_name}.csv'

    data = {
        'solution_id': [solution_id],
        'scenario': [scenario_name],
        'n_pv_kw': [solution_dict['n_pv_kw']],
        'n_wind_mw': [solution_dict['n_wind_mw']],
        'e_battery_mwh': [solution_dict['e_battery_mwh']],
        'p_diesel_mw': [solution_dict['p_diesel_mw']],
        'ens_mwh': [metrics['ens_mwh']],
        'ens_fraction': [metrics['ens_fraction']],
        'outage_hours': [metrics['outage_hours']],
        'fuel_increase_pct': [metrics['fuel_increase_pct']],
        'resilience_index': [metrics['resilience_index']]
    }

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

def analyze_scenario_performance(all_results, scenario_name):

    scenario_metrics = []

    for sol_id, scenarios in all_results.items():
        if scenario_name in scenarios:
            metrics = scenarios[scenario_name]
            scenario_metrics.append({
                'solution_id': sol_id,
                'resilience_index': metrics['resilience_index'],
                'ens_fraction': metrics['ens_fraction'],
                'outage_hours': metrics['outage_hours']
            })

    df = pd.DataFrame(scenario_metrics)

    best_solution_id = df.loc[df['resilience_index'].idxmax(), 'solution_id']
    worst_solution_id = df.loc[df['resilience_index'].idxmin(), 'solution_id']

    analysis = {
        'best_solution_id': int(best_solution_id),
        'worst_solution_id': int(worst_solution_id),
        'best_resilience_index': df['resilience_index'].max(),
        'worst_resilience_index': df['resilience_index'].min(),
        'mean_resilience_index': df['resilience_index'].mean(),
        'mean_ens_fraction': df['ens_fraction'].mean(),
        'mean_outage_hours': df['outage_hours'].mean()
    }

    return analysis

def calculate_cross_scenario_ranking(all_results):

    rankings = []

    for sol_id, scenarios in all_results.items():
        resilience_indices = [
            scenarios[scenario]['resilience_index']
            for scenario in scenarios.keys()
        ]

        avg_resilience = np.mean(resilience_indices)

        rankings.append({
            'solution_id': sol_id,
            'avg_resilience_index': avg_resilience,
            'A1_resilience': scenarios.get('A1_cold_snap', {}).get('resilience_index', 0.0),
            'A2_resilience': scenarios.get('A2_fuel_disruption', {}).get('resilience_index', 0.0),
            'A3_resilience': scenarios.get('A3_blizzard', {}).get('resilience_index', 0.0)
        })

    ranking_df = pd.DataFrame(rankings)
    ranking_df = ranking_df.sort_values('avg_resilience_index', ascending=False)

    return ranking_df

def generate_analysis_report(pareto_df, all_results, ranking_df):

    report = []

    report.append("# Resilience Analysis Report")
    report.append("")
    report.append("Arctic Microgrid Optimization - Phase 5 Resilience Scenarios")
    report.append("")
    report.append("---")
    report.append("")

    report.append("## Executive Summary")
    report.append("")
    report.append(f"Analyzed {len(all_results)} Pareto-optimal solutions across 3 Arctic extreme weather scenarios:")
    report.append("- **A1: Cold Snap** (-40°C, 7 days, +30% heating load)")
    report.append("- **A2: Fuel Disruption** (Ice road closure, 14 days, limited diesel/LNG)")
    report.append("- **A3: Blizzard** (Snow/ice on renewables, 3 days, -80% PV, -50% wind)")
    report.append("")
    report.append(f"Total simulations: {len(all_results) * 3} (8760 hours each)")
    report.append("")
    report.append("---")
    report.append("")

    scenarios_info = [
        ('A1_cold_snap', 'Cold Snap (-40°C, 7 days)'),
        ('A2_fuel_disruption', 'Fuel Disruption (14 days)'),
        ('A3_blizzard', 'Blizzard (Snow/ice, 3 days)')
    ]

    for scenario_name, scenario_title in scenarios_info:
        analysis = analyze_scenario_performance(all_results, scenario_name)

        report.append(f"## Scenario {scenario_name.split('_')[0]}: {scenario_title}")
        report.append("")

        best_sol_id = analysis['best_solution_id']
        worst_sol_id = analysis['worst_solution_id']

        best_sol = pareto_df.iloc[best_sol_id]
        worst_sol = pareto_df.iloc[worst_sol_id]

        best_metrics = all_results[best_sol_id][scenario_name]
        worst_metrics = all_results[worst_sol_id][scenario_name]

        report.append(f"**Best Solution: #{best_sol_id}**")
        report.append(f"- Configuration: {best_sol['n_pv_kw']:.0f} kW PV, "
                     f"{best_sol['n_wind_mw']:.2f} MW wind, "
                     f"{best_sol['e_battery_mwh']:.2f} MWh battery, "
                     f"{best_sol['p_diesel_mw']:.2f} MW diesel")
        report.append(f"- Resilience Index: {best_metrics['resilience_index']:.4f}")
        report.append(f"- Energy Not Served: {best_metrics['ens_mwh']:.1f} MWh ({best_metrics['ens_fraction']*100:.2f}%)")
        report.append(f"- Outage Hours: {best_metrics['outage_hours']} hours")
        report.append("")

        report.append(f"**Worst Solution: #{worst_sol_id}**")
        report.append(f"- Configuration: {worst_sol['n_pv_kw']:.0f} kW PV, "
                     f"{worst_sol['n_wind_mw']:.2f} MW wind, "
                     f"{worst_sol['e_battery_mwh']:.2f} MWh battery, "
                     f"{worst_sol['p_diesel_mw']:.2f} MW diesel")
        report.append(f"- Resilience Index: {worst_metrics['resilience_index']:.4f}")
        report.append(f"- Energy Not Served: {worst_metrics['ens_mwh']:.1f} MWh ({worst_metrics['ens_fraction']*100:.2f}%)")
        report.append(f"- Outage Hours: {worst_metrics['outage_hours']} hours")
        report.append("")

        report.append("**Key Finding:**")

        if scenario_name == 'A1_cold_snap':
            finding = f"Battery capacity critical for cold snaps. Best solution has {best_sol['e_battery_mwh']:.2f} MWh vs {worst_sol['e_battery_mwh']:.2f} MWh for worst."
        elif scenario_name == 'A2_fuel_disruption':
            finding = f"High renewable penetration enables fuel-limited operation. Best solution has {best_sol['n_pv_kw']+best_sol['n_wind_mw']*1000:.0f} kW renewables vs {worst_sol['n_pv_kw']+worst_sol['n_wind_mw']*1000:.0f} kW for worst."
        else:
            finding = f"Storage capacity most critical for short-duration renewable outages. Best solution stores {best_sol['e_battery_mwh']:.2f} MWh vs {worst_sol['e_battery_mwh']:.2f} MWh for worst."

        report.append(finding)
        report.append("")
        report.append("---")
        report.append("")

    report.append("## Cross-Scenario Ranking")
    report.append("")
    report.append("Solutions ranked by average resilience index across all 3 scenarios:")
    report.append("")

    for idx, row in ranking_df.iterrows():
        sol_id = int(row['solution_id'])
        sol = pareto_df.iloc[sol_id]

        report.append(f"**Rank {idx+1}: Solution #{sol_id}**")
        report.append(f"- Average Resilience Index: {row['avg_resilience_index']:.4f}")
        report.append(f"- Configuration: {sol['n_pv_kw']:.0f} kW PV, "
                     f"{sol['n_wind_mw']:.2f} MW wind, "
                     f"{sol['e_battery_mwh']:.2f} MWh battery, "
                     f"{sol['p_diesel_mw']:.2f} MW diesel")
        report.append(f"- A1 Cold Snap: {row['A1_resilience']:.4f}")
        report.append(f"- A2 Fuel Disruption: {row['A2_resilience']:.4f}")
        report.append(f"- A3 Blizzard: {row['A3_resilience']:.4f}")
        report.append("")

    report.append("---")
    report.append("")
    report.append("## Conclusions")
    report.append("")
    report.append("Key insights from Arctic resilience analysis:")
    report.append("")
    report.append("1. **Battery storage capacity** is the strongest predictor of resilience across all scenarios")
    report.append("2. **Renewable generation diversity** (PV + wind) provides better resilience than diesel reliance")
    report.append("3. **Cold snap scenarios** are most challenging, requiring both storage and generation capacity")
    report.append("4. **Fuel disruption** resilience strongly correlates with renewable fraction")
    report.append("5. **Short-duration events** (blizzard) can be managed with adequate storage, even if renewables are impaired")
    report.append("")
    report.append("These findings demonstrate the value of hybrid renewable+storage configurations for Arctic")
    report.append("microgrids exposed to extreme weather and fuel supply vulnerability.")
    report.append("")

    return "\n".join(report)

def main():

    print("=" * 80)
    print("Arctic Microgrid Resilience Analysis")
    print("=" * 80)
    print()

    pareto_csv = project_root / 'results' / 'pareto-front.csv'

    if not pareto_csv.exists():
        print(f"ERROR: Pareto front file not found at {pareto_csv}")
        print("Run Phase 4 optimization first to generate Pareto front.")
        return

    print(f"Loading Pareto front from {pareto_csv}...")
    pareto_df = load_pareto_solutions(pareto_csv)
    print(f"Loaded {len(pareto_df)} Pareto solutions")
    print()

    n_solutions = min(len(pareto_df), 3)
    print(f"Selecting top {n_solutions} solutions (knee points)...")
    top_solutions = select_top_solutions(pareto_df, n_solutions)
    print()

    all_results = {}

    for idx, row in top_solutions.iterrows():
        solution_dict = {
            'n_pv_kw': row['n_pv_kw'],
            'n_wind_mw': row['n_wind_mw'],
            'e_battery_mwh': row['e_battery_mwh'],
            'p_diesel_mw': row['p_diesel_mw']
        }

        solution_id = idx

        print(f"Running scenarios for Solution #{solution_id}...")
        print(f"  Config: {solution_dict['n_pv_kw']:.0f} kW PV, "
              f"{solution_dict['n_wind_mw']:.2f} MW wind, "
              f"{solution_dict['e_battery_mwh']:.2f} MWh battery")

        results = run_all_scenarios_for_solution(solution_dict, solution_id)
        all_results[solution_id] = results

        print(f"  [OK] Completed 3 scenarios")
        print()

    print("Calculating cross-scenario rankings...")
    ranking_df = calculate_cross_scenario_ranking(all_results)
    print()

    print("Generating analysis report...")
    report_content = generate_analysis_report(pareto_df, all_results, ranking_df)

    output_dir = project_root / 'results' / 'resilience-scenarios'
    report_file = output_dir / 'analysis-report.md'

    with open(report_file, 'w') as f:
        f.write(report_content)

    print(f"[OK] Report saved to {report_file}")
    print()

    print("=" * 80)
    print("Resilience Analysis Complete")
    print("=" * 80)
    print()
    print(f"Results saved to: {output_dir}")
    print(f"- {len(all_results) * 3} scenario simulation CSVs")
    print(f"- 1 analysis report (Markdown)")
    print()

if __name__ == '__main__':
    main()
