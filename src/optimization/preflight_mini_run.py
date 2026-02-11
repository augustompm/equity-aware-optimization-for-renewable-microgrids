from pathlib import Path
import sys
import time

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from optimization.nsga3_runner import run_nsga3_optimization, get_system_config
from optimization.nsga3_problem import MicrogridOptimizationProblem
import numpy as np

if __name__ == "__main__":
    print("=" * 70)
    print("PRE-FLIGHT CHECK 2: MINI RUN (EXTENDED)")
    print("Configuration: 10 population × 10 generations = 100 evaluations")
    print("Expected runtime: ~100-200 seconds")
    print("=" * 70)
    print()

    system_config = get_system_config()
    problem = MicrogridOptimizationProblem(system_config)

    start_time = time.time()

    try:
        result = run_nsga3_optimization(problem, pop_size=10, n_gen=10, seed=42)

        elapsed_time = time.time() - start_time
        time_per_eval = elapsed_time / 100.0

        print()
        print("=" * 70)
        print("MINI RUN COMPLETE")
        print("=" * 70)
        print()

        convergence_file = project_root / "results" / "convergence-history.csv"
        import pandas as pd
        convergence = pd.read_csv(convergence_file)

        cv_gen1 = convergence[convergence['generation'] == 0]['min_NPC'].iloc[0]
        cv_gen10 = convergence[convergence['generation'] == 9]['min_NPC'].iloc[0]

        n_feasible = result.opt.get("CV")
        n_feasible_count = np.sum(n_feasible <= 1e-6) if n_feasible is not None else 0

        print("Convergence Analysis:")
        print(f"  - Runtime: {elapsed_time:.1f} seconds")
        print(f"  - Time per evaluation: {time_per_eval:.2f} seconds")
        print(f"  - Pareto front size: {len(result.F) if result.F is not None else 0}")
        print(f"  - Feasible solutions: {n_feasible_count}")
        print(f"  - NPC gen 1: {cv_gen1:.2e}")
        print(f"  - NPC gen 10: {cv_gen10:.2e}")
        print()

        checks_passed = []
        checks_failed = []

        if time_per_eval < 2.0:
            checks_passed.append("Time per eval < 2s")
        else:
            checks_failed.append(f"Time per eval too high: {time_per_eval:.2f}s (target <2s)")

        if n_feasible_count >= 1:
            checks_passed.append("At least 1 feasible solution by gen 10")
        else:
            checks_failed.append("Zero feasible solutions - constraints may be too tight")

        if cv_gen10 < cv_gen1:
            checks_passed.append("NPC improved (convergence happening)")
        else:
            checks_failed.append("NPC did not improve - possible stagnation")

        print("Checks:")
        for check in checks_passed:
            print(f"  [PASS] {check}")
        for check in checks_failed:
            print(f"  [FAIL] {check}")
        print()

        if len(checks_failed) == 0:
            print("=" * 70)
            print("MINI RUN PASSED")
            print("=" * 70)
            print()
            print("All checks passed. Safe to proceed to CHECK 3.")
            print()
        else:
            print("=" * 70)
            print("MINI RUN PASSED WITH WARNINGS")
            print("=" * 70)
            print()
            print("Code executes but some metrics are concerning.")
            print("Review failures above before proceeding.")
            print()
            print("DECISION REQUIRED:")
            print("  - If 'Time per eval too high': Full run will take >10h")
            print("  - If 'Zero feasible': May need to relax constraints")
            print("  - If 'NPC did not improve': Possible local minimum")
            print()
            print("See PREFLIGHT-CHECKLIST.md section 'Cenário 2: Check 1-2 falham'")
            print("Communicate issues before proceeding (rules.md section 2)")
            print()

    except Exception as e:
        elapsed_time = time.time() - start_time
        print()
        print("=" * 70)
        print("MINI RUN FAILED")
        print("=" * 70)
        print()
        print(f"Runtime before crash: {elapsed_time:.1f} seconds")
        print(f"Error: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print()
        print("STOP - Debug required before proceeding")
        print("See PREFLIGHT-CHECKLIST.md section 'Cenário 2: Check 1-2 falham'")
        print()
        import traceback
        traceback.print_exc()
        sys.exit(1)
