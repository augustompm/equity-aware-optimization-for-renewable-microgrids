import sys
import os
import argparse
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

from config import get_v8_config, get_v8_bounds
from optimization.nsga3_problem_fast import MicrogridOptimizationProblemFast
from callbacks.nsga3_callback_fast import NSGA3ProgressCallbackFast, EarlyStopException
from simulation.data_cache import get_data_cache
from simulation.system_simulator_fast import simulate_system_fast
from metrics.solution_metrics import calculate_pareto_front_metrics
from visualization.plot_results import create_all_plots
from results.results_saver_v8 import save_v8_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--n_gen', type=int, default=200)
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--n_jobs', type=int, default=1)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = get_v8_config()
    bounds = get_v8_bounds()

    data_cache = get_data_cache()
    data_cache.initialize(config)

    ref_point = np.array([1e9, 1.0, 1e7, 1.0])
    problem = MicrogridOptimizationProblemFast(config, n_jobs=args.n_jobs)

    ref_dirs = get_reference_directions("das-dennis", n_dim=4, n_partitions=8)
    algorithm = NSGA3(ref_dirs=ref_dirs, pop_size=len(ref_dirs), eliminate_duplicates=True)

    callback = NSGA3ProgressCallbackFast(
        ref_point=ref_point,
        reference_front=None,
        log_every=5,
        stagnation_generations=20,
        stagnation_tolerance=0.001
    )

    early_stopped = False
    n_gen_actual = args.n_gen
    start_time = datetime.now()

    try:
        res = minimize(problem, algorithm, ('n_gen', args.n_gen), seed=args.seed, callback=callback, verbose=False)
        F_final, X_final = res.F, res.X
        G_final = res.G if hasattr(res, 'G') else None
    except EarlyStopException:
        early_stopped = True
        n_gen_actual = callback.hv_history[-1][0] if callback.hv_history else 0
        F_final, X_final = callback.get_last_solution()
        G_final = np.zeros((len(F_final), problem.n_ieq_constr))
        for i in range(len(F_final)):
            decision_vars = {'n_pv_kw': X_final[i, 0], 'n_wind_mw': X_final[i, 1],
                           'e_battery_mwh': X_final[i, 2], 'p_diesel_mw': X_final[i, 3]}
            _, constraints, _ = simulate_system_fast(decision_vars, config, data_cache)
            G_final[i] = [constraints['bounds'], constraints['area'], constraints['lpsp'],
                         constraints['spinning_reserve'], constraints['grid_limits'], constraints['renewable_cap']]

    elapsed = (datetime.now() - start_time).total_seconds()

    pareto_metrics = calculate_pareto_front_metrics(F=F_final, X=X_final, G=G_final, system_config=config)

    results_dir = save_v8_results(
        run_id=args.run_id, seed=args.seed, timestamp=timestamp, config=config, bounds=bounds,
        metrics_history=callback.get_metrics_history(), pareto_metrics=pareto_metrics,
        F=F_final, X=X_final, G=G_final, n_gen_actual=n_gen_actual, early_stopped=early_stopped,
        results_base_dir=args.results_dir
    )

    create_all_plots(
        metrics_history=callback.get_metrics_history(), pareto_data=pareto_metrics,
        output_dir=results_dir, formats=['png', 'pdf', 'svg'], dpi=300
    )

    print(f"Run {args.run_id} (seed {args.seed}): {len(F_final)} solutions, {n_gen_actual} gen, {elapsed:.0f}s")

if __name__ == "__main__":
    main()
