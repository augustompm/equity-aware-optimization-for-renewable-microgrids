import sys
import os
import argparse
import importlib.util
import numpy as np
from pathlib import Path
from datetime import datetime

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from optimization.nsga3_problem_fast import MicrogridOptimizationProblemFast
from callbacks.nsga3_callback_fast import NSGA3ProgressCallbackFast, EarlyStopException
from simulation.data_cache import get_data_cache
from simulation.system_simulator_fast import simulate_system_fast

from metrics.solution_metrics import calculate_pareto_front_metrics
from visualization.plot_results import create_all_plots
from results.results_saver_v8 import save_v8_results

def load_reference_front(ref_front_path):

    if not ref_front_path.exists():
        print(f"[WARN] Reference front not found: {ref_front_path}")
        return None

    try:
        ref_front = np.loadtxt(ref_front_path, delimiter=',', skiprows=1)
        print(f"[OK] Reference front loaded: {ref_front.shape}")
        return ref_front
    except Exception as e:
        print(f"[WARN] Failed to load reference front: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Production Run V9 - FAST')
    parser.add_argument('--run_id', type=int, required=True, help='Run identifier')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--n_gen', type=int, default=200, help='Max generations')
    parser.add_argument('--results_dir', type=str, default='results', help='Results base directory')
    parser.add_argument('--n_jobs', type=int, default=1, help='Parallel jobs (-1=all cores, 1=sequential)')
    args = parser.parse_args()

    run_id = args.run_id
    seed = args.seed
    n_gen = args.n_gen
    results_base_dir = args.results_dir
    n_jobs = args.n_jobs

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"production-run-v9-{run_id}-{timestamp}.log"

    def log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_file, 'a') as f:
            f.write(line + '\n')

    log("=" * 80)
    log("PRODUCTION RUN V9 - FAST OPTIMIZED")
    log("=" * 80)
    log(f"Run ID: {run_id}")
    log(f"Seed: {seed}")
    log(f"Max Generations: {n_gen}")
    log(f"Parallel jobs: {n_jobs}")
    log(f"Timestamp: {timestamp}")
    log("")

    spec = importlib.util.spec_from_file_location(
        "config_v8",
        project_root / "config-v8-corrected-battery-bounds.py"
    )
    config_v8 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_v8)

    config = config_v8.get_v8_config()
    bounds = config_v8.get_v8_bounds()

    log("V9 Configuration (Fast):")
    log(f"  Community: {config['community_name']} (pop {config['population']})")
    log(f"  Wind CF: {config['wind_cf_path'].name} (CASES High Point 30%)")
    log(f"  Battery bounds: [0, {bounds['battery_kwh'][1]}] MWh")
    log("")

    log("Initializing data cache...")
    data_cache = get_data_cache()
    data_cache.initialize(config)
    log("")

    ref_point = np.array([1e9, 1.0, 1e7, 1.0])
    log(f"Reference point: {ref_point}")

    ref_front_path = project_root / "data" / "reference-front-v8.csv"
    reference_front = load_reference_front(ref_front_path)
    log("")

    log("Creating FAST Problem...")
    problem = MicrogridOptimizationProblemFast(config, n_jobs=n_jobs)
    log(f"  n_var: {problem.n_var}")
    log(f"  n_obj: {problem.n_obj}")
    log(f"  n_constr: {problem.n_ieq_constr}")
    log(f"  Parallel: {n_jobs} jobs")
    log("")

    log("Initializing NSGA-III...")
    ref_dirs = get_reference_directions("das-dennis", n_dim=4, n_partitions=8)
    log(f"  Reference directions (Das-Dennis p=8): {len(ref_dirs)}")

    algorithm = NSGA3(
        ref_dirs=ref_dirs,
        pop_size=len(ref_dirs),
        eliminate_duplicates=True
    )
    log(f"  Population size: {len(ref_dirs)}")
    log("")

    callback = NSGA3ProgressCallbackFast(
        ref_point=ref_point,
        reference_front=reference_front,
        log_every=5,
        stagnation_generations=20,
        stagnation_tolerance=0.001,
        log_file=log_file
    )
    log("FAST Callback configured:")
    log(f"  Log every: 5 generations")
    log(f"  Early stop: 20 gen stagnation (tol=0.001) - FIXED counting")
    log("")

    log("=" * 80)
    log("STARTING OPTIMIZATION")
    log("=" * 80)

    early_stopped = False
    n_gen_actual = n_gen
    start_time = datetime.now()

    try:
        res = minimize(
            problem,
            algorithm,
            ('n_gen', n_gen),
            seed=seed,
            callback=callback,
            verbose=False
        )

        F_final = res.F
        X_final = res.X
        G_final = res.G if hasattr(res, 'G') else None

    except EarlyStopException:
        log("")
        log("=" * 80)
        log("EARLY STOPPING TRIGGERED")
        log("=" * 80)
        early_stopped = True
        n_gen_actual = callback.hv_history[-1][0] if callback.hv_history else 0

        F_final, X_final = callback.get_last_solution()

        if F_final is None or X_final is None:
            log("[ERROR] Early stop but no solutions saved!")
            sys.exit(1)

        log(f"Using last saved solution: {len(F_final)} solutions at gen {n_gen_actual}")

        G_final = np.zeros((len(F_final), problem.n_ieq_constr))
        for i in range(len(F_final)):
            decision_vars = {
                'n_pv_kw': X_final[i, 0],
                'n_wind_mw': X_final[i, 1],
                'e_battery_mwh': X_final[i, 2],
                'p_diesel_mw': X_final[i, 3]
            }
            _, constraints, _ = simulate_system_fast(decision_vars, config, data_cache)

            G_final[i, 0] = constraints['bounds']
            G_final[i, 1] = constraints['area']
            G_final[i, 2] = constraints['lpsp']
            G_final[i, 3] = constraints['spinning_reserve']
            G_final[i, 4] = constraints['grid_limits']
            G_final[i, 5] = constraints['renewable_cap']

    elapsed = (datetime.now() - start_time).total_seconds()
    log("")
    log(f"Optimization time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    log("")

    log("=" * 80)
    log("CALCULATING ADDITIONAL METRICS")
    log("=" * 80)
    log(f"Pareto solutions: {len(F_final)}")
    log("")

    log("Calculating metrics for each solution...")
    pareto_metrics = calculate_pareto_front_metrics(
        F=F_final,
        X=X_final,
        G=G_final,
        system_config=config
    )
    log(f"  [OK] {len(pareto_metrics)} solutions processed")
    log("")

    n_feasible = sum(1 for s in pareto_metrics if s['is_feasible'])
    log(f"Feasibility: {n_feasible}/{len(pareto_metrics)} feasible")

    re_pcts = [s['re_penetration_pct'] for s in pareto_metrics]
    lcoes = [s['lcoe_cad_per_kwh'] for s in pareto_metrics]
    batteries = [s['battery_mwh'] for s in pareto_metrics]

    log(f"RE penetration: {min(re_pcts):.1f}% - {max(re_pcts):.1f}%")
    log(f"LCOE: {min(lcoes):.3f} - {max(lcoes):.3f} CAD/kWh")
    log(f"Battery: {min(batteries):.1f} - {max(batteries):.1f} MWh")
    log("")

    log("=" * 80)
    log("SAVING RESULTS")
    log("=" * 80)

    results_dir = save_v8_results(
        run_id=run_id,
        seed=seed,
        timestamp=timestamp,
        config=config,
        bounds=bounds,
        metrics_history=callback.get_metrics_history(),
        pareto_metrics=pareto_metrics,
        F=F_final,
        X=X_final,
        G=G_final,
        n_gen_actual=n_gen_actual,
        early_stopped=early_stopped,
        results_base_dir=results_base_dir
    )
    log(f"  Results directory: {results_dir}")
    log(f"  [OK] CSV files saved")
    log(f"  [OK] summary.json saved")
    log("")

    log("=" * 80)
    log("CREATING PLOTS")
    log("=" * 80)

    create_all_plots(
        metrics_history=callback.get_metrics_history(),
        pareto_data=pareto_metrics,
        output_dir=results_dir,
        formats=['png', 'pdf', 'svg'],
        dpi=300
    )
    log(f"  [OK] All plots created in {results_dir / 'figures'}")
    log("")

    log("=" * 80)
    log("RUN COMPLETED SUCCESSFULLY")
    log("=" * 80)
    log(f"Results: {results_dir}")
    log(f"Generations: {n_gen_actual}/{n_gen}")
    log(f"Early stopped: {early_stopped}")
    log(f"Pareto solutions: {len(F_final)}")
    log(f"Feasible solutions: {n_feasible}")
    log(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    log("=" * 80)

if __name__ == "__main__":
    main()
