from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.indicators.hv import HV
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import time
from scipy.spatial.distance import pdist, cdist

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from optimization.nsga3_problem import MicrogridOptimizationProblem

def calculate_hypervolume(F, ref_point):

    hv_indicator = HV(ref_point=ref_point)
    return hv_indicator(F)

def calculate_spacing(F):

    if len(F) < 2:
        return 0.0

    distances = cdist(F, F, metric='euclidean')
    np.fill_diagonal(distances, np.inf)
    min_distances = distances.min(axis=1)

    d_bar = min_distances.mean()
    sp = np.sqrt(((min_distances - d_bar) ** 2).sum() / (len(F) - 1))

    return sp

def calculate_diversity(F):

    if len(F) < 2:
        return 0.0

    centroid = F.mean(axis=0)

    diversity = ((F - centroid) ** 2).sum()

    return diversity

def run_validation_optimization(pop_size=50, n_gen=50, seed=42):

    print("=" * 80)
    print("NSGA-III VALIDATION WITH PROPER METRICS")
    print("=" * 80)
    print(f"Configuration: {pop_size} pop × {n_gen} gen = {pop_size*n_gen} evaluations")
    print(f"Estimated time: {pop_size*n_gen*1.5/60:.1f} minutes")
    print()
    print("Metrics tracked:")
    print("  - Hypervolume (HV): Should increase monotonically")
    print("  - Spacing (SP): Should stabilize (lower is better)")
    print("  - Diversity (ID): Should increase then stabilize")
    print("  - Pareto size: Should NOT decrease (elitist!)")
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

    if pop_size >= 100:
        n_partitions = 12
    elif pop_size >= 50:
        n_partitions = 6
    else:
        n_partitions = 3

    ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=n_partitions)

    print(f"Reference directions: {len(ref_dirs)} points (n_partitions={n_partitions})")
    print(f"Population size: {pop_size}")

    if pop_size < len(ref_dirs):
        print(f"WARNING: pop_size ({pop_size}) < ref_dirs ({len(ref_dirs)})")
        print("  This may affect diversity maintenance (Deb & Jain 2014 recommend N ≥ H)")
    print()

    algorithm = NSGA3(
        pop_size=pop_size,
        ref_dirs=ref_dirs,
        eliminate_duplicates=True
    )

    print("Starting optimization...")
    start_time = time.time()

    res = minimize(
        problem,
        algorithm,
        ('n_gen', n_gen),
        seed=seed,
        verbose=True,
        save_history=True
    )

    elapsed = time.time() - start_time
    print()
    print("=" * 80)
    print(f"Optimization completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print("=" * 80)
    print()

    print("Calculating convergence metrics...")
    print()

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
            hv = calculate_hypervolume(F_feasible, ref_point)
            sp = calculate_spacing(F_feasible)
            div = calculate_diversity(F_feasible)
        else:
            hv = sp = div = 0.0

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

    metrics_file = project_root / 'results' / 'validation-metrics.csv'
    df_metrics.to_csv(metrics_file, index=False)
    print(f"Saved: {metrics_file}")
    print()

    print("=" * 80)
    print("VALIDATION ANALYSIS")
    print("=" * 80)
    print()

    validation_passed = True

    print("[CHECK 1] Pareto Front Size Evolution (CRITICAL for elitist algorithm)")
    pareto_sizes = df_metrics['n_pareto'].values
    max_size = pareto_sizes.max()
    max_gen = pareto_sizes.argmax()
    final_size = pareto_sizes[-1]

    print(f"  Generation 0:   {pareto_sizes[0]} solutions")
    print(f"  Maximum (gen {max_gen}): {max_size} solutions")
    print(f"  Final (gen {n_gen-1}):   {final_size} solutions")

    if final_size < max_size:
        print(f"  [WARNING] Lost {max_size - final_size} solutions from peak!")
        print(f"  Possible causes:")
        print(f"    - Duplicate elimination removing similar solutions")
        print(f"    - Constraint violations making solutions infeasible")
        print(f"    - Numerical precision issues")
        validation_passed = False
    else:
        print(f"  [OK] Pareto front maintained or grew (elitist behavior)")
    print()

    print("[CHECK 2] Hypervolume Trend (should increase)")
    hv_values = df_metrics['hypervolume'].values
    hv_improvement = (hv_values[-1] - hv_values[0]) / hv_values[0] * 100 if hv_values[0] > 0 else 0

    print(f"  Initial HV: {hv_values[0]:.2e}")
    print(f"  Final HV:   {hv_values[-1]:.2e}")
    print(f"  Improvement: {hv_improvement:+.1f}%")

    hv_decreases = np.sum(np.diff(hv_values) < 0)
    print(f"  Generations with HV decrease: {hv_decreases}/{n_gen-1}")

    if hv_improvement > 10:
        print(f"  [OK] Significant hypervolume improvement")
    elif hv_improvement > 0:
        print(f"  [PARTIAL] Modest improvement, may need more generations")
    else:
        print(f"  [FAIL] Hypervolume did not improve!")
        validation_passed = False
    print()

    print("[CHECK 3] Feasibility Evolution")
    feas_initial = df_metrics.iloc[0]['n_feasible']
    feas_final = df_metrics.iloc[-1]['n_feasible']

    print(f"  Initial: {feas_initial}/{df_metrics.iloc[0]['n_pareto']} feasible")
    print(f"  Final:   {feas_final}/{df_metrics.iloc[-1]['n_pareto']} feasible")

    if feas_final == df_metrics.iloc[-1]['n_pareto']:
        print(f"  [OK] 100% feasible in final generation")
    elif feas_final > feas_initial:
        print(f"  [PARTIAL] Improved but not 100% feasible")
    else:
        print(f"  [FAIL] Feasibility did not improve")
        validation_passed = False
    print()

    print("[CHECK 4] Final Objective Value Ranges")
    if res.F is not None and len(res.F) > 0:
        objectives = ['NPC ($)', 'LPSP', 'CO2 (kg)', 'Gini']

        for i, obj_name in enumerate(objectives):
            min_val = res.F[:, i].min()
            max_val = res.F[:, i].max()
            range_val = max_val - min_val

            print(f"  {obj_name:12s}: [{min_val:12.2e}, {max_val:12.2e}] range={range_val:.2e}")
    print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if validation_passed:
        print("[SUCCESS] Algorithm is converging properly!")
        print()
        print("Observations:")
        print(f"  - Hypervolume improved {hv_improvement:.1f}%")
        print(f"  - {feas_final}/{final_size} feasible solutions")
        print(f"  - Pareto front size: {final_size} solutions")
        print()
        print("Recommendation for full run:")
        print(f"  - Population: 100 (currently {pop_size})")
        print(f"  - Generations: 200 (currently {n_gen})")
        print(f"  - Expected runtime: ~8 hours")
        print(f"  - Estimated hypervolume improvement: >{hv_improvement*2:.0f}%")
    else:
        print("[ISSUES DETECTED] Review warnings above!")
        print()
        print("Recommendations:")
        print("  1. Check if duplicate_elimination is too aggressive")
        print("  2. Review constraint formulation")
        print("  3. Consider increasing population size")
        print("  4. Try longer run to see if stabilizes")

    print("=" * 80)

    return {
        'passed': validation_passed,
        'metrics': df_metrics,
        'result': res
    }

if __name__ == "__main__":

    validation = run_validation_optimization(pop_size=50, n_gen=50, seed=42)

    sys.exit(0 if validation['passed'] else 1)
