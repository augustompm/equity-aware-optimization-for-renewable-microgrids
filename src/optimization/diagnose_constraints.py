from pathlib import Path
import sys
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from optimization.nsga3_problem import MicrogridOptimizationProblem
from optimization.nsga3_runner import get_system_config

if __name__ == "__main__":
    print("=" * 70)
    print("CONSTRAINT DIAGNOSTICS")
    print("Analyzing constraint violations in sample solutions")
    print("=" * 70)
    print()

    system_config = get_system_config()
    problem = MicrogridOptimizationProblem(system_config)

    print("Decision Variable Bounds:")
    print(f"  PV (kW):       0 to {problem.xu[0]}")
    print(f"  Wind (MW):     0 to {problem.xu[1]}")
    print(f"  Battery (MWh): 0 to {problem.xu[2]}")
    print(f"  Diesel (MW):   0 to {problem.xu[3]}")
    print()

    test_solutions = [
        np.array([[0, 0, 0, 10]]),
        np.array([[100, 1, 5, 5]]),
        np.array([[500, 5, 10, 3]]),
        np.array([[50, 0.5, 2, 4]]),
        np.array([[200, 2, 8, 6]])
    ]

    labels = [
        "Diesel-only (baseline-like)",
        "Small hybrid",
        "Medium hybrid",
        "Conservative hybrid",
        "Balanced hybrid"
    ]

    constraint_names = [
        "Bounds",
        "Area",
        "LPSP",
        "Spinning Reserve",
        "Grid Limits",
        "Renewable Cap"
    ]

    for idx, (X, label) in enumerate(zip(test_solutions, labels)):
        print(f"Solution {idx+1}: {label}")
        print(f"  Config: PV={X[0,0]:.0f}kW, Wind={X[0,1]:.2f}MW, Batt={X[0,2]:.1f}MWh, Diesel={X[0,3]:.1f}MW")
        print()

        out = {}
        problem._evaluate(X, out)

        F = out['F'][0]
        G = out['G'][0]

        print(f"  Objectives:")
        print(f"    NPC:  ${F[0]:,.0f}")
        print(f"    LPSP: {F[1]:.4f} ({F[1]*100:.2f}%)")
        print(f"    CO2:  {F[2]:,.0f} kg")
        print(f"    Gini: {F[3]:.4f}")
        print()

        print(f"  Constraint Violations (positive = violation):")
        total_cv = 0.0
        for c_idx, (c_val, c_name) in enumerate(zip(G, constraint_names)):
            status = "VIOLATED" if c_val > 0 else "OK"
            print(f"    {c_name:20s}: {c_val:12.6f}  [{status}]")
            total_cv += max(0, c_val)

        print(f"  Total CV: {total_cv:.6f}")
        print(f"  Feasible: {'YES' if total_cv < 1e-6 else 'NO'}")
        print()
        print("-" * 70)
        print()

    print("=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)
    print()
    print("Analysis:")
    print("  - If ALL solutions violate same constraint: constraint too tight")
    print("  - If LPSP violations: need more renewables or battery")
    print("  - If Spinning Reserve violations: need more diesel capacity")
    print("  - If Renewable Cap violations: too much renewable capacity")
    print("  - If Area violations: footprint exceeds available land")
    print()
    print("Action:")
    print("  - If patterns unclear: extend mini run to 10 generations")
    print("  - If specific constraint always violated: may need adjustment")
    print()
