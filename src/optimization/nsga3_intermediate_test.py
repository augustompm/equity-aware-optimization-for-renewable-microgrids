from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.indicators.hv import HV
import pandas as pd
import numpy as np
import sys
import json
from pathlib import Path
from datetime import datetime
from scipy.spatial.distance import cdist

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from optimization.nsga3_problem import MicrogridOptimizationProblem

def calculate_metrics(F, ref_point):

    if len(F) == 0:
        return 0.0, 0.0, 0.0

    hv_indicator = HV(ref_point=ref_point)
    hv = hv_indicator(F)

    if len(F) < 2:
        return hv, 0.0, 0.0

    distances = cdist(F, F, metric='euclidean')
    np.fill_diagonal(distances, np.inf)
    min_distances = distances.min(axis=1)
    d_bar = min_distances.mean()
    sp = np.sqrt(((min_distances - d_bar) ** 2).sum() / (len(F) - 1))

    centroid = F.mean(axis=0)
    div = ((F - centroid) ** 2).sum()

    return hv, sp, div

def analyze_convergence_quality(metrics_df):

    results = {
        'passed': True,
        'checks': {}
    }

    hv_initial = metrics_df.iloc[0]['hypervolume']
    hv_final = metrics_df.iloc[-1]['hypervolume']
    hv_improvement = (hv_final - hv_initial) / hv_initial * 100 if hv_initial > 0 else 0

    results['checks']['hv_improvement'] = {
        'value': hv_improvement,
        'threshold': 200.0,
        'passed': hv_improvement > 200.0,
        'message': f"HV improved {hv_improvement:.1f}% (threshold: >200%)"
    }

    if not results['checks']['hv_improvement']['passed']:
        results['passed'] = False

    final_feas = metrics_df.iloc[-1]['n_feasible']
    final_total = metrics_df.iloc[-1]['n_pareto']
    feas_rate = final_feas / final_total if final_total > 0 else 0

    results['checks']['feasibility'] = {
        'value': feas_rate,
        'threshold': 0.95,
        'passed': feas_rate > 0.95,
        'message': f"Feasibility rate: {feas_rate*100:.1f}% (threshold: >95%)"
    }

    if not results['checks']['feasibility']['passed']:
        results['passed'] = False

    pareto_size = final_total

    results['checks']['pareto_size'] = {
        'value': pareto_size,
        'threshold': 10,
        'passed': pareto_size >= 10,
        'message': f"Pareto front: {pareto_size} solutions (threshold: >=10)"
    }

    if not results['checks']['pareto_size']['passed']:
        results['passed'] = False

    last_25_spacing = metrics_df.iloc[-25:]['spacing'].values
    spacing_std = last_25_spacing.std()
    spacing_mean = last_25_spacing.mean()
    spacing_cv = spacing_std / spacing_mean if spacing_mean > 0 else np.inf

    results['checks']['spacing_stability'] = {
        'value': spacing_cv,
        'threshold': 0.20,
        'passed': spacing_cv < 0.20,
        'message': f"Spacing CV: {spacing_cv:.2%} (threshold: <20%)"
    }

    if not results['checks']['spacing_stability']['passed']:
        results['passed'] = False

    final_objectives = {
        'min_NPC': metrics_df.iloc[-1]['min_NPC'],
        'min_LPSP': metrics_df.iloc[-1]['min_LPSP'],
        'min_CO2': metrics_df.iloc[-1]['min_CO2'],
        'min_Gini': metrics_df.iloc[-1]['min_Gini']
    }

    abnormal = []
    if final_objectives['min_NPC'] < 1e6 or final_objectives['min_NPC'] > 5e8:
        abnormal.append(f"NPC out of range: ${final_objectives['min_NPC']:.2e}")
    if final_objectives['min_LPSP'] < 0 or final_objectives['min_LPSP'] > 0.1:
        abnormal.append(f"LPSP out of range: {final_objectives['min_LPSP']:.2%}")
    if final_objectives['min_CO2'] < 0 or final_objectives['min_CO2'] > 1e8:
        abnormal.append(f"CO2 out of range: {final_objectives['min_CO2']:.2e} kg")
    if final_objectives['min_Gini'] < 0 or final_objectives['min_Gini'] > 1:
        abnormal.append(f"Gini out of range: {final_objectives['min_Gini']:.2f}")

    results['checks']['objective_ranges'] = {
        'abnormal': abnormal,
        'passed': len(abnormal) == 0,
        'message': "All objectives in expected ranges" if len(abnormal) == 0 else f"{len(abnormal)} abnormal values detected"
    }

    if not results['checks']['objective_ranges']['passed']:
        results['passed'] = False

    return results

def run_intermediate_test():

    print("=" * 80)
    print("NSGA-III INTERMEDIATE TEST - 1 HOUR VALIDATION")
    print("=" * 80)
    print()
    print("Purpose: Thorough validation before 12-hour full run")
    print()
    print("Configuration:")
    print("  Population: 84 (n_partitions=6)")
    print("  Generations: 75")
    print("  Total evaluations: 6,300")
    print("  Estimated time: 60-90 minutes")
    print()
    print("Success Criteria:")
    print("  1. Hypervolume improvement > 200%")
    print("  2. Final feasibility > 95%")
    print("  3. Pareto front size >= 10 solutions")
    print("  4. Spacing stabilized (CV < 20%)")
    print("  5. All objectives in realistic ranges")
    print()
    print("=" * 80)
    print()

    system_config = {
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

    problem = MicrogridOptimizationProblem(system_config)

    pop_size = 84
    n_gen = 75
    n_partitions = 6

    ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=n_partitions)

    print(f"Reference directions: {len(ref_dirs)} points")
    print(f"Population size: {pop_size}")
    print()

    algorithm = NSGA3(
        pop_size=pop_size,
        ref_dirs=ref_dirs,
        eliminate_duplicates=True
    )

    print("Starting optimization...")
    print()

    start_time = datetime.now()

    res = minimize(
        problem,
        algorithm,
        ('n_gen', n_gen),
        seed=42,
        verbose=True,
        save_history=True
    )

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    print()
    print("=" * 80)
    print(f"Optimization completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print("=" * 80)
    print()

    if res.F is None or len(res.F) == 0:
        print("[FAIL] No Pareto front found!")
        return False

    print(f"Pareto front size: {len(res.F)}")
    print()

    print("Calculating convergence metrics...")

    metrics_history = []
    ref_point = np.array([2e8, 0.2, 1e8, 1.0])

    for gen, entry in enumerate(res.history):
        F = entry.opt.get("F")
        CV = entry.opt.get("CV")

        if CV.ndim > 1:
            CV = CV.sum(axis=1)

        n_feasible = np.sum(CV <= 1e-6)
        n_pareto = len(F)

        feasible_mask = CV <= 1e-6
        F_feasible = F[feasible_mask] if n_feasible > 0 else F

        if len(F_feasible) > 0:
            hv, sp, div = calculate_metrics(F_feasible, ref_point)
        else:
            hv, sp, div = 0.0, 0.0, 0.0

        metrics_history.append({
            'generation': gen,
            'n_pareto': n_pareto,
            'n_feasible': n_feasible,
            'hypervolume': hv,
            'spacing': sp,
            'diversity': div,
            'min_NPC': F[:, 0].min(),
            'min_LPSP': F[:, 1].min(),
            'min_CO2': F[:, 2].min(),
            'min_Gini': F[:, 3].min()
        })

    df_metrics = pd.DataFrame(metrics_history)

    results_dir = project_root / "results"
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")

    metrics_file = results_dir / f"intermediate-test-metrics-{timestamp}.csv"
    df_metrics.to_csv(metrics_file, index=False)

    pareto_file = results_dir / f"intermediate-test-pareto-{timestamp}.csv"
    df_pareto = pd.DataFrame(res.X, columns=['n_pv_kw', 'n_wind_mw', 'e_battery_mwh', 'p_diesel_mw'])
    df_pareto['NPC'] = res.F[:, 0]
    df_pareto['LPSP'] = res.F[:, 1]
    df_pareto['CO2'] = res.F[:, 2]
    df_pareto['Gini'] = res.F[:, 3]
    df_pareto.to_csv(pareto_file, index=False)

    print(f"Saved: {metrics_file}")
    print(f"Saved: {pareto_file}")
    print()

    print("=" * 80)
    print("VALIDATION ANALYSIS")
    print("=" * 80)
    print()

    analysis = analyze_convergence_quality(df_metrics)

    for check_name, check_data in analysis['checks'].items():
        status = "[PASS]" if check_data['passed'] else "[FAIL]"
        print(f"{status} {check_data['message']}")

        if check_name == 'objective_ranges' and not check_data['passed']:
            for abnormal in check_data['abnormal']:
                print(f"      - {abnormal}")

    print()
    print("=" * 80)

    if analysis['passed']:
        print("[SUCCESS] ALL VALIDATION CRITERIA PASSED!")
        print("=" * 80)
        print()
        print("Recommendation: PROCEED WITH FULL OPTIMIZATION")
        print()
        print("Full run configuration:")
        print("  Population: 165 (n_partitions=8)")
        print("  Generations: 200")
        print("  Estimated time: ~12 hours")
        print()
        print("Command:")
        print("  python src/optimization/nsga3_runner_fixed.py")
    else:
        print("[FAIL] VALIDATION CRITERIA NOT MET!")
        print("=" * 80)
        print()
        print("DO NOT proceed with 12-hour run until issues are resolved!")
        print()
        print("Failed checks:")
        for check_name, check_data in analysis['checks'].items():
            if not check_data['passed']:
                print(f"  - {check_name}: {check_data['message']}")
        print()
        print("Recommended actions:")
        print("  1. Review failed checks above")
        print("  2. Investigate cause (constraints, data, algorithm settings)")
        print("  3. Fix issues")
        print("  4. Re-run this intermediate test")
        print("  5. Only proceed to full run after all checks pass")

    print()
    print("=" * 80)

    print()
    print("FINAL STATISTICS")
    print("=" * 80)
    print()
    print("Objective Ranges:")
    objectives = ['NPC ($)', 'LPSP', 'CO2 (kg)', 'Gini']
    for i, obj in enumerate(objectives):
        print(f"  {obj:12s}: [{res.F[:, i].min():12.2e}, {res.F[:, i].max():12.2e}]")
    print()

    print("Convergence Metrics:")
    print(f"  Initial HV:  {df_metrics.iloc[0]['hypervolume']:12.2e}")
    print(f"  Final HV:    {df_metrics.iloc[-1]['hypervolume']:12.2e}")
    print(f"  HV Improvement: {analysis['checks']['hv_improvement']['value']:+8.1f}%")
    print()
    print(f"  Final Spacing:   {df_metrics.iloc[-1]['spacing']:12.2e}")
    print(f"  Final Diversity: {df_metrics.iloc[-1]['diversity']:12.2e}")
    print()

    print("=" * 80)

    return analysis['passed']

if __name__ == "__main__":

    success = run_intermediate_test()
    sys.exit(0 if success else 1)
