from pymoo.core.problem import Problem
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from simulation.system_simulator import simulate_system

class MicrogridOptimizationProblem(Problem):

    def __init__(self, system_config):

        self.system_config = system_config

        n_var = 4

        xl = np.array([0, 0, 0, 0])
        xu = np.array([1000, 10, 20, 10])

        n_obj = 4

        n_constr = 6

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr,
                         xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):

        n_pop = X.shape[0]

        F = np.zeros((n_pop, self.n_obj))
        G = np.zeros((n_pop, self.n_constr))

        for i in range(n_pop):
            decision_vars = {
                'n_pv_kw': X[i, 0],
                'n_wind_mw': X[i, 1],
                'e_battery_mwh': X[i, 2],
                'p_diesel_mw': X[i, 3]
            }

            objectives, constraints, dispatch_summary = simulate_system(
                decision_vars, self.system_config
            )

            F[i, 0] = objectives['npc']
            F[i, 1] = objectives['lpsp']
            F[i, 2] = objectives['co2']
            F[i, 3] = objectives['gini']

            G[i, 0] = constraints['bounds']
            G[i, 1] = constraints['area']
            G[i, 2] = constraints['lpsp']
            G[i, 3] = constraints['spinning_reserve']
            G[i, 4] = constraints['grid_limits']
            G[i, 5] = constraints['renewable_cap']

        out["F"] = F
        out["G"] = G
