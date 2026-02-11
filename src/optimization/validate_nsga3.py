import numpy as np
import pandas as pd
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from optimization.nsga3_problem import MicrogridOptimizationProblem
from optimization.nsga3_runner import run_nsga3_optimization, get_system_config
from optimization.performance_metrics import (
    calculate_generation_metrics,
    analyze_convergence_history,
    get_reference_point
)

def run_validation():

    print("=" * 80)
    print("NSGA-III VALIDATION RUN")
    print("Arctic Microgrid Optimization - Short Test Configuration")
    print("=" * 80)
    print()
    print("Configuration:")
    print("  Population size: 20")
    print("  Generations: 10")
    print("  Total evaluations: 200")
    print("  Expected runtime: 3-5 minutes")
    print()

    system_config = get_system_config()
    problem = MicrogridOptimizationProblem(system_config)

    print("Starting optimization...")
    print()

    result = run_nsga3_optimization(problem, pop_size=20, n_gen=10, seed=42)

    print()
    print("=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print()

    if result.F is None or len(result.F) == 0:
        print("WARNING: No Pareto front found!")
        return

    results_dir = project_root / "results" / "validation"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Pareto front size: {len(result.F)} solutions")
    print()

    CV = np.sum(np.maximum(result.G, 0), axis=1)
    n_feasible = np.sum(CV <= 1e-6)
    feasibility_rate = n_feasible / len(result.F) * 100

    print(f"Feasible solutions: {n_feasible}/{len(result.F)} ({feasibility_rate:.1f}%)")
    print()

    pareto_df = pd.DataFrame(result.X, columns=['n_pv_kw', 'n_wind_mw', 'e_battery_mwh', 'p_diesel_mw'])
    pareto_df['NPC'] = result.F[:, 0]
    pareto_df['LPSP'] = result.F[:, 1]
    pareto_df['CO2'] = result.F[:, 2]
    pareto_df['Gini'] = result.F[:, 3]
    pareto_df['CV_total'] = CV
    pareto_df['feasible'] = CV <= 1e-6

    pareto_file = results_dir / "validation-pareto.csv"
    pareto_df.to_csv(pareto_file, index=False)

    print(f"Saved Pareto front to: {pareto_file}")
    print()

    print("Calculating performance metrics...")
    print()

    convergence_data = analyze_convergence_history(result.history)

    convergence_df = pd.DataFrame(convergence_data,
                                  columns=['generation', 'n_solutions', 'hypervolume',
                                           'spacing', 'diversity'])

    metrics_file = results_dir / "validation-metrics.csv"
    convergence_df.to_csv(metrics_file, index=False)

    print(f"Saved convergence metrics to: {metrics_file}")
    print()

    print("Convergence Summary:")
    print("-" * 80)
    print(f"{'Gen':>4}  {'N_Sol':>6}  {'Hypervolume':>15}  {'Spacing':>10}  {'Diversity':>12}")
    print("-" * 80)

    for idx, row in convergence_df.iterrows():
        gen = int(row['generation'])
        n_sol = int(row['n_solutions'])
        hv = row['hypervolume']
        sp = row['spacing']
        div = row['diversity']

        print(f"{gen:4d}  {n_sol:6d}  {hv:15.4e}  {sp:10.4f}  {div:12.4e}")

    print("-" * 80)
    print()

    first_gen_hv = convergence_df.iloc[0]['hypervolume']
    last_gen_hv = convergence_df.iloc[-1]['hypervolume']
    hv_improvement = (last_gen_hv / first_gen_hv - 1.0) * 100

    print(f"Hypervolume improvement: {hv_improvement:+.2f}%")
    print()

    print("Objective Statistics (Final Pareto Front):")
    print("-" * 80)
    print(f"{'Objective':12}  {'Min':>15}  {'Mean':>15}  {'Max':>15}")
    print("-" * 80)

    objectives = ['NPC', 'LPSP', 'CO2', 'Gini']
    for i, obj in enumerate(objectives):
        min_val = result.F[:, i].min()
        mean_val = result.F[:, i].mean()
        max_val = result.F[:, i].max()

        print(f"{obj:12}  {min_val:15.4e}  {mean_val:15.4e}  {max_val:15.4e}")

    print("-" * 80)
    print()

    report_content = generate_validation_report(
        pareto_df, convergence_df, feasibility_rate, hv_improvement
    )

    report_file = results_dir / "validation-report.md"
    with open(report_file, 'w') as f:
        f.write(report_content)

    print(f"Generated validation report: {report_file}")
    print()

    print("=" * 80)
    print("VALIDATION SUCCESSFUL")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  - Pareto front: {len(result.F)} solutions ({feasibility_rate:.1f}% feasible)")
    print(f"  - Hypervolume improvement: {hv_improvement:+.2f}%")
    print(f"  - Final HV: {last_gen_hv:.4e}")
    print()
    print("Next steps:")
    print("  1. Review validation-report.md for detailed analysis")
    print("  2. Check convergence plots in validation-metrics.csv")
    print("  3. If results look good, proceed with full optimization (100 pop × 200 gen)")
    print()

def generate_validation_report(pareto_df, convergence_df, feasibility_rate, hv_improvement):

    report = []

    report.append("# NSGA-III Validation Report")
    report.append("")
    report.append("**Arctic Microgrid Optimization - Short Test Run**")
    report.append("")
    report.append("---")
    report.append("")

    report.append("## Configuration")
    report.append("")
    report.append("- **Population size:** 20")
    report.append("- **Generations:** 10")
    report.append("- **Total evaluations:** 200")
    report.append("- **Random seed:** 42")
    report.append("")

    report.append("## Results Summary")
    report.append("")
    report.append(f"- **Pareto front size:** {len(pareto_df)} solutions")
    report.append(f"- **Feasible solutions:** {feasibility_rate:.1f}%")
    report.append(f"- **Hypervolume improvement:** {hv_improvement:+.2f}%")
    report.append("")

    report.append("## Convergence Metrics")
    report.append("")
    report.append("| Generation | Solutions | Hypervolume | Spacing | Diversity |")
    report.append("|------------|-----------|-------------|---------|-----------|")

    for _, row in convergence_df.iterrows():
        gen = int(row['generation'])
        n_sol = int(row['n_solutions'])
        hv = row['hypervolume']
        sp = row['spacing']
        div = row['diversity']

        report.append(f"| {gen} | {n_sol} | {hv:.4e} | {sp:.4f} | {div:.4e} |")

    report.append("")

    report.append("## Objective Statistics")
    report.append("")
    report.append("| Objective | Minimum | Mean | Maximum |")
    report.append("|-----------|---------|------|---------|")

    objectives = ['NPC', 'LPSP', 'CO2', 'Gini']
    for obj in objectives:
        min_val = pareto_df[obj].min()
        mean_val = pareto_df[obj].mean()
        max_val = pareto_df[obj].max()

        report.append(f"| {obj} | {min_val:.4e} | {mean_val:.4e} | {max_val:.4e} |")

    report.append("")

    report.append("## Analysis")
    report.append("")

    if hv_improvement > 10:
        report.append(f"**Convergence: GOOD** - Hypervolume improved by {hv_improvement:.1f}%, indicating effective optimization progress.")
    elif hv_improvement > 0:
        report.append(f"**Convergence: MODERATE** - Hypervolume improved by {hv_improvement:.1f}%. Consider more generations for full convergence.")
    else:
        report.append(f"**Convergence: POOR** - Hypervolume decreased or stagnated. Review algorithm parameters.")

    report.append("")

    if feasibility_rate > 80:
        report.append(f"**Feasibility: GOOD** - {feasibility_rate:.1f}% of solutions are feasible, indicating constraints are well-handled.")
    elif feasibility_rate > 50:
        report.append(f"**Feasibility: MODERATE** - {feasibility_rate:.1f}% feasible. Some constraint violations present.")
    else:
        report.append(f"**Feasibility: POOR** - Only {feasibility_rate:.1f}% feasible. Review constraint formulations.")

    report.append("")

    report.append("## Recommendations")
    report.append("")

    if hv_improvement > 10 and feasibility_rate > 80:
        report.append("1. **Proceed with full optimization** - Validation successful, ready for 100 pop × 200 gen")
        report.append("2. Monitor hypervolume convergence to identify stopping point")
        report.append("3. Expect similar or better performance with larger population")
    else:
        report.append("1. **Review validation results** - Check convergence and feasibility issues")
        report.append("2. Consider adjusting constraint parameters if feasibility is low")
        report.append("3. Increase generations if convergence is slow")

    report.append("")

    report.append("---")
    report.append("")
    report.append("**Generated:** 2025-11-04")
    report.append("")

    return "\n".join(report)

if __name__ == "__main__":
    run_validation()
