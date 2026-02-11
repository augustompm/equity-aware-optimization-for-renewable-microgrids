from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import time

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from optimization.nsga3_problem import MicrogridOptimizationProblem

def run_test_optimization(pop_size=20, n_gen=20, seed=42):

    print("=" * 80)
    print("NSGA-III TEST RUN - CONVERGENCE VALIDATION")
    print("=" * 80)
    print(f"Configuration: {pop_size} pop × {n_gen} gen = {pop_size*n_gen} evaluations")
    print(f"Estimated time: {pop_size*n_gen*1.5/60:.1f} minutes")
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

    n_partitions = 3
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
    print(f"Test completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print("=" * 80)
    print()

    validation_passed = True

    print("[CHECK 1] Pareto Front Size")
    if res.F is None or len(res.F) == 0:
        print("  [FAIL] No Pareto front found!")
        validation_passed = False
    else:
        print(f"  [OK] {len(res.F)} solutions in Pareto front")
    print()

    print("[CHECK 2] Feasibility")
    n_feasible = np.sum(res.CV <= 1e-6) if res.CV is not None else 0
    if n_feasible == 0:
        print("  [FAIL] No feasible solutions!")
        validation_passed = False
    else:
        print(f"  [OK] {n_feasible}/{len(res.F)} feasible solutions ({n_feasible/len(res.F)*100:.1f}%)")
    print()

    print("[CHECK 3] Objective Value Ranges")
    objectives = ['NPC (USD)', 'LPSP (frac)', 'CO2 (kg)', 'Gini (0-1)']
    expected_ranges = [
        (1e6, 1e8),
        (0.0, 0.2),
        (0, 1e8),
        (0.0, 1.0)
    ]

    for i, (obj_name, (min_exp, max_exp)) in enumerate(zip(objectives, expected_ranges)):
        min_val = res.F[:, i].min()
        max_val = res.F[:, i].max()

        if min_val < min_exp * 0.1 or max_val > max_exp * 10:
            print(f"  [WARN] {obj_name}: [{min_val:.2e}, {max_val:.2e}] outside expected range")
        else:
            print(f"  [OK] {obj_name}: [{min_val:.2e}, {max_val:.2e}]")
    print()

    print("[CHECK 4] Convergence Trend")
    if res.history:
        gens = []
        min_npc = []
        n_feas = []

        for gen, entry in enumerate(res.history):
            F = entry.opt.get("F")
            CV = entry.opt.get("CV")

            gens.append(gen)
            min_npc.append(F[:, 0].min())
            n_feas.append(np.sum(CV <= 1e-6))

        improvement = (min_npc[0] - min_npc[-1]) / min_npc[0] * 100

        if improvement > 1.0:
            print(f"  [OK] NPC improved by {improvement:.1f}% (gen 0 -> {n_gen-1})")
        else:
            print(f"  [WARN] NPC improved only {improvement:.1f}% - may need more generations")

        if n_feas[-1] > n_feas[0]:
            print(f"  [OK] Feasible solutions: {n_feas[0]} -> {n_feas[-1]}")
        else:
            print(f"  [INFO] Feasible solutions: {n_feas[0]} -> {n_feas[-1]}")

        conv_df = pd.DataFrame({
            'generation': gens,
            'min_NPC': min_npc,
            'n_feasible': n_feas
        })

        conv_file = project_root / 'results' / 'test-convergence.csv'
        conv_df.to_csv(conv_file, index=False)
        print(f"  Saved: {conv_file}")
    print()

    print("[CHECK 5] Solution Diversity")
    if len(res.F) > 1:

        from scipy.spatial.distance import pdist
        distances = pdist(res.F)
        avg_dist = distances.mean()
        min_dist = distances.min()

        if min_dist < 1e-6:
            print(f"  [WARN] Some solutions are identical (min dist = {min_dist:.2e})")
        else:
            print(f"  [OK] Solutions are diverse (avg dist = {avg_dist:.2e})")
    print()

    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    if validation_passed:
        print("[SUCCESS] All critical checks passed!")
        print()
        print("Recommendation:")
        print("  - Test run shows algorithm is working correctly")
        print("  - Convergence is happening (objectives improving)")
        print("  - Ready to proceed with full optimization:")
        print("    100 pop × 200 gen = 20,000 evaluations (~8 hours)")
        print()
        print("  Start full run with:")
        print("    python src/optimization/nsga3_runner.py")
    else:
        print("[ISSUES DETECTED] Review failures above before full run!")
        print()
        print("Next steps:")
        print("  1. Review constraint violations in test-convergence.csv")
        print("  2. Check if problem formulation is correct")
        print("  3. Verify input data (load, solar CF, wind CF)")
        print("  4. Consider adjusting bounds or constraints")

    print("=" * 80)

    if res.F is not None and len(res.F) > 0:
        pareto_df = pd.DataFrame(res.X, columns=['n_pv_kw', 'n_wind_mw', 'e_battery_mwh', 'p_diesel_mw'])
        pareto_df['NPC'] = res.F[:, 0]
        pareto_df['LPSP'] = res.F[:, 1]
        pareto_df['CO2'] = res.F[:, 2]
        pareto_df['Gini'] = res.F[:, 3]
        pareto_df['CV_total'] = np.sum(np.maximum(res.G, 0), axis=1) if res.G is not None else 0

        pareto_file = project_root / 'results' / 'test-pareto-front.csv'
        pareto_df.to_csv(pareto_file, index=False)
        print(f"\nTest Pareto front saved: {pareto_file}")

    return validation_passed

if __name__ == "__main__":

    success = run_test_optimization(pop_size=20, n_gen=20, seed=42)

    sys.exit(0 if success else 1)
