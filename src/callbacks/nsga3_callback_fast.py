from pymoo.core.callback import Callback
from pymoo.indicators.hv import HV
from scipy.spatial.distance import cdist
import numpy as np
from datetime import datetime

class EarlyStopException(Exception):
    pass

class NSGA3ProgressCallbackFast(Callback):

    def __init__(
        self,
        ref_point,
        reference_front=None,
        log_every=5,
        stagnation_generations=20,
        stagnation_tolerance=0.001,
        log_file=None
    ):
        super().__init__()
        self.ref_point = ref_point
        self.reference_front = reference_front
        self.log_every = log_every
        self.stagnation_generations = stagnation_generations
        self.stagnation_tolerance = stagnation_tolerance
        self.log_file = log_file

        self.hv_indicator = HV(ref_point=ref_point)
        self.hv_history = []
        self.metrics_history = []
        self.last_F = None
        self.last_X = None
        self.best_hv = 0.0
        self.best_hv_gen = 0

    def notify(self, algorithm):
        gen = algorithm.n_gen

        opt = algorithm.opt
        if opt is None or len(opt) == 0:
            return

        F = opt.get("F")
        X = opt.get("X")

        self.last_F = F.copy()
        self.last_X = X.copy()

        hv = self._calculate_hv(F)
        self.hv_history.append((gen, hv))

        if hv > self.best_hv * (1 + self.stagnation_tolerance):
            self.best_hv = hv
            self.best_hv_gen = gen

        if gen % self.log_every == 0 or gen == 1:
            igd_plus = self._calculate_igd_plus(F)
            spacing = self._calculate_spacing(F)
            diversity = self._calculate_diversity(F)

            metrics = {
                'generation': gen,
                'n_solutions': len(F),
                'hypervolume': hv,
                'igd_plus': igd_plus,
                'spacing': spacing,
                'diversity': diversity
            }
            self.metrics_history.append(metrics)

            msg = (
                f"Gen {gen:3d}: "
                f"N={len(F):2d} | "
                f"HV={hv:.6e} | "
                f"IGD+={igd_plus:.4f} | "
                f"SP={spacing:.4e} | "
                f"DIV={diversity:.4e}"
            )
            self._log(msg)

        if gen - self.best_hv_gen >= self.stagnation_generations:
            self._log(f"Early stop: HV stagnant for {gen - self.best_hv_gen} gen (best at gen {self.best_hv_gen})")
            raise EarlyStopException()

    def _calculate_hv(self, F):
        if len(F) == 0:
            return 0.0
        try:
            return self.hv_indicator(F)
        except:
            return 0.0

    def _calculate_igd_plus(self, pareto_front):
        if self.reference_front is None or len(pareto_front) == 0:
            return np.inf

        if len(self.reference_front) == 0:
            return np.inf

        n_ref = self.reference_front.shape[0]
        n_obtained = pareto_front.shape[0]

        min_modified_distances = np.zeros(n_ref)
        for i in range(n_ref):
            z = self.reference_front[i, :]
            inferiority = np.maximum(pareto_front - z, 0.0)
            distances = np.linalg.norm(inferiority, axis=1)
            min_modified_distances[i] = np.min(distances)

        return np.mean(min_modified_distances)

    def _calculate_spacing(self, F):
        if len(F) < 2:
            return 0.0
        distances = cdist(F, F, metric='euclidean')
        np.fill_diagonal(distances, np.inf)
        min_distances = distances.min(axis=1)
        d_bar = min_distances.mean()
        sp = np.sqrt(((min_distances - d_bar) ** 2).sum() / (len(F) - 1))
        return sp

    def _calculate_diversity(self, F):
        if len(F) < 2:
            return 0.0
        centroid = F.mean(axis=0)
        diversity = ((F - centroid) ** 2).sum()
        return diversity

    def _log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line, flush=True)

        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(line + '\n')

    def get_last_solution(self):
        return self.last_F, self.last_X

    def get_metrics_history(self):
        return self.metrics_history
