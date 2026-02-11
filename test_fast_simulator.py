import sys
import time
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from simulation.system_simulator import simulate_system
from simulation.system_simulator_fast import simulate_system_fast
from simulation.data_cache import get_data_cache

import importlib.util
spec = importlib.util.spec_from_file_location(
    "config_v8",
    project_root / "config-v8-corrected-battery-bounds.py"
)
config_v8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_v8)

def test_equivalence():

    print("=" * 80)
    print("TEST: Fast Simulator Equivalence")
    print("=" * 80)

    config = config_v8.get_v8_config()

    test_cases = [
        {'n_pv_kw': 1000, 'n_wind_mw': 2.0, 'e_battery_mwh': 20, 'p_diesel_mw': 5.0},
        {'n_pv_kw': 5000, 'n_wind_mw': 4.0, 'e_battery_mwh': 50, 'p_diesel_mw': 3.0},
        {'n_pv_kw': 100, 'n_wind_mw': 0.5, 'e_battery_mwh': 5, 'p_diesel_mw': 8.0},
        {'n_pv_kw': 0, 'n_wind_mw': 5.0, 'e_battery_mwh': 100, 'p_diesel_mw': 2.0},
    ]

    all_passed = True

    for i, decision_vars in enumerate(test_cases):
        print(f"\nTest case {i+1}: {decision_vars}")

        obj_orig, constr_orig, _ = simulate_system(decision_vars, config)

        obj_fast, constr_fast, _ = simulate_system_fast(decision_vars, config)

        for key in ['npc', 'lpsp', 'co2', 'gini']:
            orig_val = obj_orig[key]
            fast_val = obj_fast[key]
            rel_diff = abs(orig_val - fast_val) / (abs(orig_val) + 1e-10)

            if rel_diff > 0.001:
                print(f"  [FAIL] {key}: orig={orig_val:.6g}, fast={fast_val:.6g}, diff={rel_diff:.2%}")
                all_passed = False
            else:
                print(f"  [OK] {key}: {orig_val:.6g} (diff={rel_diff:.2%})")

    print()
    if all_passed:
        print("[PASS] All test cases passed!")
    else:
        print("[FAIL] Some test cases failed!")

    return all_passed

def test_performance():

    print()
    print("=" * 80)
    print("TEST: Performance Benchmark")
    print("=" * 80)

    config = config_v8.get_v8_config()
    decision_vars = {'n_pv_kw': 2000, 'n_wind_mw': 3.0, 'e_battery_mwh': 30, 'p_diesel_mw': 5.0}

    _ = simulate_system_fast(decision_vars, config)

    n_runs = 10

    print(f"\nOriginal simulator ({n_runs} runs)...")
    start = time.perf_counter()
    for _ in range(n_runs):
        simulate_system(decision_vars, config)
    orig_time = time.perf_counter() - start
    print(f"  Time: {orig_time:.3f}s ({orig_time/n_runs*1000:.1f}ms per run)")

    print(f"\nFast simulator ({n_runs} runs, cached)...")
    start = time.perf_counter()
    for _ in range(n_runs):
        simulate_system_fast(decision_vars, config)
    fast_time = time.perf_counter() - start
    print(f"  Time: {fast_time:.3f}s ({fast_time/n_runs*1000:.1f}ms per run)")

    speedup = orig_time / fast_time
    print(f"\nSpeedup: {speedup:.1f}x")

    return speedup

def test_parallel_problem():

    print()
    print("=" * 80)
    print("TEST: Parallel Problem Evaluation")
    print("=" * 80)

    from optimization.nsga3_problem import MicrogridOptimizationProblem
    from optimization.nsga3_problem_fast import MicrogridOptimizationProblemFast

    config = config_v8.get_v8_config()

    np.random.seed(42)
    n_pop = 20
    X = np.random.rand(n_pop, 4)
    X[:, 0] *= 10000
    X[:, 1] *= 5
    X[:, 2] *= 100
    X[:, 3] *= 10

    print(f"\nOriginal problem ({n_pop} evaluations)...")
    problem_orig = MicrogridOptimizationProblem(config)
    out_orig = {}
    start = time.perf_counter()
    problem_orig._evaluate(X, out_orig)
    orig_time = time.perf_counter() - start
    print(f"  Time: {orig_time:.3f}s")

    print(f"\nFast problem sequential ({n_pop} evaluations)...")
    problem_fast = MicrogridOptimizationProblemFast(config, n_jobs=1)
    out_fast = {}
    start = time.perf_counter()
    problem_fast._evaluate(X, out_fast)
    fast_seq_time = time.perf_counter() - start
    print(f"  Time: {fast_seq_time:.3f}s")

    print(f"\nFast problem parallel ({n_pop} evaluations)...")
    problem_fast_par = MicrogridOptimizationProblemFast(config, n_jobs=-1)
    out_fast_par = {}
    start = time.perf_counter()
    problem_fast_par._evaluate(X, out_fast_par)
    fast_par_time = time.perf_counter() - start
    print(f"  Time: {fast_par_time:.3f}s")

    print("\nResult comparison (first 3 solutions):")
    for i in range(min(3, n_pop)):
        print(f"  Solution {i}: orig={out_orig['F'][i][:2]}, fast={out_fast_par['F'][i][:2]}")

    max_diff = np.max(np.abs(out_orig['F'] - out_fast_par['F']))
    print(f"\nMax difference: {max_diff:.6g}")

    print(f"\nSpeedup (seq): {orig_time/fast_seq_time:.1f}x")
    print(f"Speedup (par): {orig_time/fast_par_time:.1f}x")

    return orig_time / fast_par_time

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("FAST SIMULATOR VALIDATION TESTS")
    print("=" * 80 + "\n")

    equiv_ok = test_equivalence()

    speedup_sim = test_performance()

    speedup_prob = test_parallel_problem()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Equivalence test: {'PASS' if equiv_ok else 'FAIL'}")
    print(f"Simulator speedup: {speedup_sim:.1f}x")
    print(f"Problem speedup: {speedup_prob:.1f}x")
    print("=" * 80)
