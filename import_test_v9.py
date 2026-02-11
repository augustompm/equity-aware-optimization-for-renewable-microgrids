import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

print("Testing imports...")

import importlib.util
spec = importlib.util.spec_from_file_location("config_v8", project_root / "config.py")
config_v8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_v8)
config = config_v8.get_v8_config()
print(f"  [OK] config.py (capital_cost keys: {sum(1 for k in config if 'capital' in k)})")

from components.pv import SolarPV
from components.wind import WindTurbine
from components.battery import Battery
from components.generator import Generator
from components.load import LoadProfile
print("  [OK] components (PV, Wind, Battery, Generator, Load)")

from objectives.objective_functions import (
    objective_npc, objective_lpsp, objective_co2,
    objective_gini, objective_gini_spatial
)
print("  [OK] objectives (NPC, LPSP, CO2, Gini, Gini_spatial)")

from constraints.constraint_validator import validate_solution
print("  [OK] constraints")

from simulation.system_simulator import simulate_system
print("  [OK] system_simulator")

from optimization.nsga3_problem import MicrogridOptimizationProblem
print("  [OK] nsga3_problem")

from callbacks.nsga3_callback import NSGA3ProgressCallback, EarlyStopException
print("  [OK] callbacks")

from metrics.solution_metrics import calculate_pareto_front_metrics
print("  [OK] metrics")

from visualization.plot_results import create_all_plots
print("  [OK] visualization")

from results.results_saver_v8 import save_v8_results
print("  [OK] results_saver_v8")

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
print("  [OK] pymoo (NSGA3, minimize, ref_dirs)")

print("\nAll imports OK - production-run-v8.py should work.")
