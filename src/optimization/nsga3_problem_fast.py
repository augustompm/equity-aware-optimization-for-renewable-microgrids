from pymoo.core.problem import Problem
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from simulation.data_cache import get_data_cache
from simulation.system_simulator_fast import simulate_system_fast

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("[WARN] joblib not available, using sequential evaluation")

class MicrogridOptimizationProblemFast(Problem):

    def __init__(self, system_config, n_jobs=-1):

        self.system_config = system_config
        self.n_jobs = n_jobs

        self.data_cache = get_data_cache()
        self.data_cache.initialize(system_config)

        n_var = 4

        bounds = system_config.get('bounds', {
            'pv_kw': (0, 10000),
            'wind_kw': (0, 5),
            'battery_kwh': (0, 100),
            'diesel_kw': (0, 10)
        })

        xl = np.array([
            bounds['pv_kw'][0],
            bounds['wind_kw'][0],
            bounds['battery_kwh'][0],
            bounds['diesel_kw'][0]
        ])

        xu = np.array([
            bounds['pv_kw'][1],
            bounds['wind_kw'][1],
            bounds['battery_kwh'][1],
            bounds['diesel_kw'][1]
        ])

        n_obj = 4
        n_constr = 6

        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_constr,
                         xl=xl, xu=xu)

    def _evaluate_single(self, x):

        decision_vars = {
            'n_pv_kw': x[0],
            'n_wind_mw': x[1],
            'e_battery_mwh': x[2],
            'p_diesel_mw': x[3]
        }

        objectives, constraints, _ = simulate_system_fast(
            decision_vars, self.system_config, self.data_cache
        )

        f = np.array([
            objectives['npc'],
            objectives['lpsp'],
            objectives['co2'],
            objectives['gini']
        ])

        g = np.array([
            constraints['bounds'],
            constraints['area'],
            constraints['lpsp'],
            constraints['spinning_reserve'],
            constraints['grid_limits'],
            constraints['renewable_cap']
        ])

        return f, g

    def _evaluate(self, X, out, *args, **kwargs):

        n_pop = X.shape[0]

        if JOBLIB_AVAILABLE and self.n_jobs != 1:

            results = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(self._evaluate_single)(X[i]) for i in range(n_pop)
            )

            F = np.array([r[0] for r in results])
            G = np.array([r[1] for r in results])
        else:

            F = np.zeros((n_pop, self.n_obj))
            G = np.zeros((n_pop, self.n_ieq_constr))

            for i in range(n_pop):
                f, g = self._evaluate_single(X[i])
                F[i] = f
                G[i] = g

        out["F"] = F
        out["G"] = G
