from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
import pandas as pd
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from optimization.nsga3_problem import MicrogridOptimizationProblem

def run_nsga3_optimization(problem, pop_size=100, n_gen=200, seed=42):

    print("=" * 70)
    print("NSGA-III Multi-Objective Optimization")
    print("Arctic Microgrid Configuration Optimization")
    print("=" * 70)
    print()

    if pop_size >= 100:
        n_partitions = 12
    elif pop_size >= 50:
        n_partitions = 6
    else:
        n_partitions = 3

    ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=n_partitions)
    print(f"Reference directions: {len(ref_dirs)} points in 4D objective space (n_partitions={n_partitions})")

    algorithm = NSGA3(
        pop_size=pop_size,
        ref_dirs=ref_dirs,
        eliminate_duplicates=True
    )
    print(f"Population size: {pop_size}")
    print(f"Generations: {n_gen}")
    print(f"Total evaluations: {pop_size * n_gen}")
    print()

    estimated_time_min = (pop_size * n_gen * 1.0) / 60.0
    estimated_time_max = (pop_size * n_gen * 2.0) / 60.0
    print(f"Estimated runtime: {estimated_time_min:.1f} - {estimated_time_max:.1f} minutes")
    print()
    print("Starting optimization...")
    print()

    res = minimize(
        problem,
        algorithm,
        ('n_gen', n_gen),
        seed=seed,
        verbose=True,
        save_history=True
    )

    print()
    print("=" * 70)
    print("Optimization Complete!")
    print("=" * 70)

    if res.F is None or len(res.F) == 0:
        print("Warning: No Pareto front found. All solutions may be infeasible.")
        print(f"Algorithm status: {res.algorithm}")
        return res

    print(f"Pareto front size: {len(res.F)}")
    print(f"Feasible solutions: {np.sum(res.CV <= 1e-6)}")
    print()

    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    pareto_file = results_dir / "pareto-front.csv"
    convergence_file = results_dir / "convergence-history.csv"

    save_pareto_front(res.X, res.F, res.G, pareto_file)
    save_convergence_history(res.history, convergence_file)

    print(f"Results saved to:")
    print(f"  - {pareto_file}")
    print(f"  - {convergence_file}")
    print()

    print_summary_statistics(res.F)

    return res

def save_pareto_front(X, F, G, filename):

    df = pd.DataFrame(X, columns=['n_pv_kw', 'n_wind_mw', 'e_battery_mwh', 'p_diesel_mw'])

    df['NPC'] = F[:, 0]
    df['LPSP'] = F[:, 1]
    df['CO2'] = F[:, 2]
    df['Gini'] = F[:, 3]

    df['CV_total'] = np.sum(np.maximum(G, 0), axis=1)

    df.to_csv(filename, index=False)

def save_convergence_history(history, filename):

    convergence_data = []

    for gen, entry in enumerate(history):
        F = entry.opt.get("F")

        n_feasible = np.sum(entry.opt.get("CV") <= 1e-6)
        n_infeasible = len(F) - n_feasible

        convergence_data.append({
            'generation': gen,
            'n_feasible': n_feasible,
            'n_infeasible': n_infeasible,
            'min_NPC': F[:, 0].min(),
            'min_LPSP': F[:, 1].min(),
            'min_CO2': F[:, 2].min(),
            'min_Gini': F[:, 3].min()
        })

    df = pd.DataFrame(convergence_data)
    df.to_csv(filename, index=False)

def print_summary_statistics(F):

    print("Summary Statistics:")
    print("-" * 70)

    objectives = ['NPC', 'LPSP', 'CO2', 'Gini']
    for i, obj in enumerate(objectives):
        print(f"{obj:8s}: min={F[:, i].min():12.2f}  "
              f"mean={F[:, i].mean():12.2f}  max={F[:, i].max():12.2f}")

    print("=" * 70)

def get_system_config():

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

if __name__ == "__main__":

    system_config = get_system_config()

    problem = MicrogridOptimizationProblem(system_config)

    result = run_nsga3_optimization(problem, pop_size=100, n_gen=200, seed=42)
