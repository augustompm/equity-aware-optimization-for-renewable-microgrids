from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.indicators.hv import HV
from pymoo.termination import get_termination
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
from optimization.stopping_criteria import HypervolumeStagnation, EarlyStoppingCallback

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

def calculate_igd_plus(pareto_front, reference_front):

    if len(pareto_front) == 0 or len(reference_front) == 0:
        return np.inf

    n_ref = reference_front.shape[0]
    n_obtained = pareto_front.shape[0]

    min_modified_distances = np.zeros(n_ref)

    for i in range(n_ref):
        z = reference_front[i, :]
        modified_distances = np.zeros(n_obtained)

        for j in range(n_obtained):
            a = pareto_front[j, :]
            inferiority_vector = np.maximum(a - z, 0.0)
            d_plus = np.sqrt(np.sum(inferiority_vector ** 2))
            modified_distances[j] = d_plus

        min_modified_distances[i] = np.min(modified_distances)

    return np.mean(min_modified_distances)

def select_optimal_partitions(n_objectives=4):

    if n_objectives == 4:

        n_partitions = 8
        expected_ref_dirs = 165
    elif n_objectives == 3:
        n_partitions = 12
        expected_ref_dirs = 91
    else:

        n_partitions = max(3, 15 // n_objectives)
        ref_dirs = get_reference_directions("das-dennis", n_objectives, n_partitions=n_partitions)
        expected_ref_dirs = len(ref_dirs)

    return n_partitions, expected_ref_dirs

def run_nsga3_optimization(problem, pop_size=None, n_gen=200, seed=42, run_id=1):

    print("=" * 80)
    print(f"NSGA-III OPTIMIZATION - RUN {run_id}")
    print("Arctic Microgrid Configuration Optimization")
    print("=" * 80)
    print()

    n_partitions, expected_ref_dirs = select_optimal_partitions(n_objectives=4)

    ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=n_partitions)
    actual_ref_dirs = len(ref_dirs)

    print(f"Reference directions: {actual_ref_dirs} points (n_partitions={n_partitions})")

    if pop_size is None:
        pop_size = actual_ref_dirs
        print(f"Population size: {pop_size} (auto-set to match reference directions)")
    else:
        print(f"Population size: {pop_size}")
        if pop_size < actual_ref_dirs:
            print(f"WARNING: pop_size ({pop_size}) < ref_dirs ({actual_ref_dirs})")
            print(f"  Deb & Jain (2014) recommend N ≥ H for optimal diversity")

    print(f"Generations: {n_gen}")
    print(f"Total evaluations: {pop_size * n_gen}")
    print()

    estimated_time_min = (pop_size * n_gen * 1.0) / 60.0
    estimated_time_max = (pop_size * n_gen * 2.0) / 60.0
    print(f"Estimated runtime: {estimated_time_min:.1f} - {estimated_time_max:.1f} minutes")
    print(f"                  ({estimated_time_min/60:.1f} - {estimated_time_max/60:.1f} hours)")
    print()

    algorithm = NSGA3(
        pop_size=pop_size,
        ref_dirs=ref_dirs,
        eliminate_duplicates=True
    )

    print("Starting optimization...")
    print()

    start_time = datetime.now()

    ref_point = np.array([2e8, 0.2, 1e8, 1.0])
    callback = EarlyStoppingCallback(ref_point)

    from pymoo.core.termination import TerminationOr
    termination = TerminationOr(
        get_termination("n_gen", n_gen),
        HypervolumeStagnation(n_stagnant=20, tol=0.01)
    )

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        verbose=True,
        save_history=True,
        callback=callback
    )

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    print()
    print("=" * 80)
    print("Optimization Complete!")
    print("=" * 80)
    print(f"Runtime: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes, {elapsed/3600:.1f} hours)")
    print()

    if res.F is None or len(res.F) == 0:
        print("WARNING: No Pareto front found. All solutions may be infeasible.")
        return None

    print(f"Pareto front size: {len(res.F)}")

    if res.CV is not None:
        if res.CV.ndim > 1:
            CV_total = res.CV.sum(axis=1)
        else:
            CV_total = res.CV

        n_feasible = np.sum(CV_total <= 1e-6)
        print(f"Feasible solutions: {n_feasible}/{len(res.F)} ({n_feasible/len(res.F)*100:.1f}%)")
    print()

    print("Calculating convergence metrics...")

    metrics_history = []

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
            'mean_NPC': F[:, 0].mean(),
            'min_LPSP': F[:, 1].min(),
            'mean_LPSP': F[:, 1].mean(),
            'min_CO2': F[:, 2].min(),
            'mean_CO2': F[:, 2].mean(),
            'min_Gini': F[:, 3].min(),
            'mean_Gini': F[:, 3].mean()
        })

    df_metrics = pd.DataFrame(metrics_history)

    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = start_time.strftime("%Y%m%d_%H%M%S")

    pareto_file = results_dir / f"pareto-front-run{run_id}-{timestamp}.csv"
    save_pareto_front(res.X, res.F, res.G if hasattr(res, 'G') else None, pareto_file)

    metrics_file = results_dir / f"convergence-metrics-run{run_id}-{timestamp}.csv"
    df_metrics.to_csv(metrics_file, index=False)

    summary = {
        'run_id': run_id,
        'timestamp': timestamp,
        'configuration': {
            'population_size': pop_size,
            'n_partitions': n_partitions,
            'reference_directions': actual_ref_dirs,
            'max_generations': n_gen,
            'actual_generations': res.algorithm.n_gen,
            'total_evaluations': pop_size * res.algorithm.n_gen,
            'seed': seed,
            'early_stopping': {
                'enabled': True,
                'n_stagnant_threshold': 20,
                'tolerance': 0.01,
                'stopped_early': res.algorithm.n_gen < n_gen
            }
        },
        'runtime': {
            'seconds': elapsed,
            'minutes': elapsed / 60,
            'hours': elapsed / 3600
        },
        'final_results': {
            'pareto_size': len(res.F),
            'n_feasible': int(n_feasible),
            'feasibility_rate': float(n_feasible / len(res.F))
        },
        'final_metrics': {
            'hypervolume': float(df_metrics.iloc[-1]['hypervolume']),
            'spacing': float(df_metrics.iloc[-1]['spacing']),
            'diversity': float(df_metrics.iloc[-1]['diversity'])
        },
        'objectives': {
            'NPC': {
                'min': float(res.F[:, 0].min()),
                'mean': float(res.F[:, 0].mean()),
                'max': float(res.F[:, 0].max())
            },
            'LPSP': {
                'min': float(res.F[:, 1].min()),
                'mean': float(res.F[:, 1].mean()),
                'max': float(res.F[:, 1].max())
            },
            'CO2': {
                'min': float(res.F[:, 2].min()),
                'mean': float(res.F[:, 2].mean()),
                'max': float(res.F[:, 2].max())
            },
            'Gini': {
                'min': float(res.F[:, 3].min()),
                'mean': float(res.F[:, 3].mean()),
                'max': float(res.F[:, 3].max())
            }
        },
        'convergence': {
            'initial_hypervolume': float(df_metrics.iloc[0]['hypervolume']),
            'final_hypervolume': float(df_metrics.iloc[-1]['hypervolume']),
            'hv_improvement_pct': float((df_metrics.iloc[-1]['hypervolume'] - df_metrics.iloc[0]['hypervolume']) / df_metrics.iloc[0]['hypervolume'] * 100) if df_metrics.iloc[0]['hypervolume'] > 0 else 0,
            'igd_plus_vs_initial': float(calculate_igd_plus(res.F, algorithm.history[0].opt.get("F"))) if len(algorithm.history) > 0 else None
        }
    }

    summary_file = results_dir / f"summary-run{run_id}-{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved:")
    print(f"  - {pareto_file}")
    print(f"  - {metrics_file}")
    print(f"  - {summary_file}")
    print()

    print_summary_statistics(res.F, df_metrics)

    return {
        'result': res,
        'metrics': df_metrics,
        'summary': summary,
        'files': {
            'pareto': pareto_file,
            'metrics': metrics_file,
            'summary': summary_file
        }
    }

def save_pareto_front(X, F, G, filename):

    df = pd.DataFrame(X, columns=['n_pv_kw', 'n_wind_mw', 'e_battery_mwh', 'p_diesel_mw'])

    df['NPC'] = F[:, 0]
    df['LPSP'] = F[:, 1]
    df['CO2'] = F[:, 2]
    df['Gini'] = F[:, 3]

    if G is not None:
        df['CV_total'] = np.sum(np.maximum(G, 0), axis=1)

    df.to_csv(filename, index=False)

def print_summary_statistics(F, metrics_df):

    print("=" * 80)
    print("FINAL STATISTICS")
    print("=" * 80)
    print()

    print("Objective Value Ranges:")
    print("-" * 80)
    objectives = ['NPC ($)', 'LPSP', 'CO2 (kg)', 'Gini']
    for i, obj in enumerate(objectives):
        print(f"{obj:12s}: min={F[:, i].min():12.2e}  "
              f"mean={F[:, i].mean():12.2e}  max={F[:, i].max():12.2e}")
    print()

    print("Convergence Metrics:")
    print("-" * 80)
    print(f"Initial Hypervolume: {metrics_df.iloc[0]['hypervolume']:12.2e}")
    print(f"Final Hypervolume:   {metrics_df.iloc[-1]['hypervolume']:12.2e}")

    hv_improvement = (metrics_df.iloc[-1]['hypervolume'] - metrics_df.iloc[0]['hypervolume']) / metrics_df.iloc[0]['hypervolume'] * 100 if metrics_df.iloc[0]['hypervolume'] > 0 else 0
    print(f"HV Improvement:      {hv_improvement:+12.1f}%")
    print()

    print(f"Final Spacing:       {metrics_df.iloc[-1]['spacing']:12.2e} (lower is better)")
    print(f"Final Diversity:     {metrics_df.iloc[-1]['diversity']:12.2e} (higher is better)")
    print()

    print("=" * 80)

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

    result = run_nsga3_optimization(
        problem,
        pop_size=None,
        n_gen=200,
        seed=42,
        run_id=1
    )

    if result:
        print("\nOptimization successful!")
        print(f"Final Hypervolume: {result['summary']['final_metrics']['hypervolume']:.2e}")
        print(f"Pareto Front Size: {result['summary']['final_results']['pareto_size']}")
    else:
        print("\nOptimization failed!")
        sys.exit(1)
