from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from optimization.nsga3_runner import run_nsga3_optimization, get_system_config
from optimization.nsga3_problem import MicrogridOptimizationProblem

if __name__ == "__main__":
    print("=" * 70)
    print("PRE-FLIGHT CHECK 1: SMOKE TEST")
    print("Configuration: 1 population × 2 generations = 2 evaluations")
    print("Expected runtime: ~4 seconds")
    print("=" * 70)
    print()

    system_config = get_system_config()
    problem = MicrogridOptimizationProblem(system_config)

    try:
        result = run_nsga3_optimization(problem, pop_size=1, n_gen=2, seed=42)

        print()
        print("=" * 70)
        print("SMOKE TEST PASSED")
        print("=" * 70)
        print()
        print("Verification:")
        print(f"  - Result object created: {'Yes' if result is not None else 'No'}")
        print(f"  - Pareto front size: {len(result.F) if result.F is not None else 0}")
        print(f"  - No NaN values: {not hasattr(result.F, '__iter__') or not any([any([str(v) == 'nan' for v in row]) for row in result.F]  if result.F is not None else [])}")
        print()
        print("Code executes without crashes. Safe to proceed to CHECK 2.")
        print()

    except Exception as e:
        print()
        print("=" * 70)
        print("SMOKE TEST FAILED")
        print("=" * 70)
        print()
        print(f"Error: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print()
        print("STOP - Debug required before proceeding")
        print("See PREFLIGHT-CHECKLIST.md section 'Cenário 2: Check 1-2 falham'")
        print()
        import traceback
        traceback.print_exc()
        sys.exit(1)
