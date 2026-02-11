import pandas as pd
import numpy as np
from pathlib import Path

def analyze_pareto_front(pareto_csv):

    df = pd.read_csv(pareto_csv)

    stats = {}
    for obj in ['NPC', 'LPSP', 'CO2', 'Gini']:
        stats[obj] = {
            'min': df[obj].min(),
            'mean': df[obj].mean(),
            'max': df[obj].max()
        }

    correlations = df[['NPC', 'LPSP', 'CO2', 'Gini']].corr()

    knee_solutions = find_knee_points(df, n_points=5)

    return stats, correlations, knee_solutions

def find_knee_points(df, n_points=5):

    objectives = df[['NPC', 'LPSP', 'CO2', 'Gini']].values

    obj_range = objectives.max(axis=0) - objectives.min(axis=0)
    obj_range[obj_range == 0] = 1.0

    normalized = (objectives - objectives.min(axis=0)) / obj_range

    distances = np.linalg.norm(normalized, axis=1)

    n_available = min(n_points, len(df))
    knee_indices = np.argsort(distances)[:n_available]

    return df.iloc[knee_indices].copy()

def calculate_hypervolume(objectives, ref_point=None):

    if ref_point is None:
        ref_point = objectives.max(axis=0) * 1.1

    normalized_objectives = objectives / ref_point

    dominated_volume = 0.0

    for i in range(len(objectives)):
        point = normalized_objectives[i]

        contribution = np.prod(1.0 - point)

        dominated_volume += contribution

    hypervolume = dominated_volume * np.prod(ref_point)

    return hypervolume

def get_extreme_solutions(df):

    extremes = {}

    for obj in ['NPC', 'LPSP', 'CO2', 'Gini']:
        min_idx = df[obj].idxmin()
        extremes[f'{obj}_min'] = df.loc[min_idx].to_dict()

    return extremes

def print_analysis_report(pareto_csv, output_file=None):

    stats, correlations, knee_solutions = analyze_pareto_front(pareto_csv)

    df = pd.read_csv(pareto_csv)
    extremes = get_extreme_solutions(df)

    objectives = df[['NPC', 'LPSP', 'CO2', 'Gini']].values
    hv = calculate_hypervolume(objectives)

    report_lines = []

    report_lines.append("=" * 80)
    report_lines.append("PARETO FRONT ANALYSIS REPORT")
    report_lines.append("Arctic Microgrid Optimization")
    report_lines.append("=" * 80)
    report_lines.append("")

    report_lines.append(f"Pareto Front Size: {len(df)} solutions")
    report_lines.append(f"Hypervolume Indicator: {hv:.6e}")
    report_lines.append("")

    report_lines.append("-" * 80)
    report_lines.append("OBJECTIVE STATISTICS")
    report_lines.append("-" * 80)
    report_lines.append("")

    for obj in ['NPC', 'LPSP', 'CO2', 'Gini']:
        report_lines.append(f"{obj}:")
        report_lines.append(f"  Min:  {stats[obj]['min']:15.2f}")
        report_lines.append(f"  Mean: {stats[obj]['mean']:15.2f}")
        report_lines.append(f"  Max:  {stats[obj]['max']:15.2f}")
        report_lines.append("")

    report_lines.append("-" * 80)
    report_lines.append("OBJECTIVE CORRELATIONS")
    report_lines.append("-" * 80)
    report_lines.append("")
    report_lines.append(correlations.to_string())
    report_lines.append("")

    report_lines.append("-" * 80)
    report_lines.append("KNEE POINT SOLUTIONS (Best Balanced)")
    report_lines.append("-" * 80)
    report_lines.append("")

    for i, (idx, row) in enumerate(knee_solutions.iterrows(), 1):
        report_lines.append(f"Knee Point {i}:")
        report_lines.append(f"  PV:      {row['n_pv_kw']:8.1f} kW")
        report_lines.append(f"  Wind:    {row['n_wind_mw']:8.2f} MW")
        report_lines.append(f"  Battery: {row['e_battery_mwh']:8.2f} MWh")
        report_lines.append(f"  Diesel:  {row['p_diesel_mw']:8.2f} MW")
        report_lines.append(f"  NPC:     {row['NPC']:15.0f} USD")
        report_lines.append(f"  LPSP:    {row['LPSP']:8.4f}")
        report_lines.append(f"  CO2:     {row['CO2']:15.0f} kg")
        report_lines.append(f"  Gini:    {row['Gini']:8.4f}")
        report_lines.append("")

    report_lines.append("-" * 80)
    report_lines.append("EXTREME SOLUTIONS (Single Objective Optima)")
    report_lines.append("-" * 80)
    report_lines.append("")

    for obj_name, solution in extremes.items():
        obj = obj_name.replace('_min', '')
        report_lines.append(f"Best {obj}:")
        report_lines.append(f"  {obj}: {solution[obj]:15.2f}")
        report_lines.append(f"  Configuration: {solution['n_pv_kw']:.0f} kW PV, "
                          f"{solution['n_wind_mw']:.2f} MW wind, "
                          f"{solution['e_battery_mwh']:.2f} MWh battery")
        report_lines.append("")

    report_lines.append("=" * 80)

    report_text = "\n".join(report_lines)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"Analysis report saved to {output_file}")
    else:
        print(report_text)

if __name__ == "__main__":

    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    pareto_csv = project_root / "results" / "pareto-front.csv"

    if not pareto_csv.exists():
        print(f"Error: Pareto front file not found at {pareto_csv}")
        print("Run NSGA-III optimization first to generate results.")
        sys.exit(1)

    output_file = project_root / "results" / "analysis-report.txt"

    print_analysis_report(pareto_csv, output_file)
