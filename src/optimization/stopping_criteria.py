from pymoo.core.termination import Termination
import numpy as np

class HypervolumeStagnation(Termination):

    def __init__(self, n_stagnant=20, tol=0.01, **kwargs):
        super().__init__(**kwargs)
        self.n_stagnant = n_stagnant
        self.tol = tol
        self.hv_history = []
        self.best_hv = -np.inf
        self.generations_without_improvement = 0

    def _update(self, algorithm):

        if hasattr(algorithm, 'hv_history') and len(algorithm.hv_history) > 0:
            current_hv = algorithm.hv_history[-1]
        else:

            return 1.0

        self.hv_history.append(current_hv)

        relative_improvement = (current_hv - self.best_hv) / max(abs(self.best_hv), 1e-10)

        if relative_improvement > self.tol:

            self.best_hv = current_hv
            self.generations_without_improvement = 0
        else:

            self.generations_without_improvement += 1

        progress = self.generations_without_improvement / self.n_stagnant

        return min(progress, 1.0)

    def _do_continue(self, algorithm):

        return self.generations_without_improvement < self.n_stagnant

class MultiMetricStagnation(Termination):

    def __init__(self, n_stagnant=20, tol_hv=0.01, tol_sp=0.05, tol_div=0.05, **kwargs):
        super().__init__(**kwargs)
        self.n_stagnant = n_stagnant
        self.tol_hv = tol_hv
        self.tol_sp = tol_sp
        self.tol_div = tol_div

        self.hv_history = []
        self.sp_history = []
        self.div_history = []

        self.best_hv = -np.inf
        self.best_sp = np.inf
        self.best_div = -np.inf

        self.hv_stagnant = 0
        self.sp_stagnant = 0
        self.div_stagnant = 0

    def _update(self, algorithm):

        if not (hasattr(algorithm, 'hv_history') and
                hasattr(algorithm, 'sp_history') and
                hasattr(algorithm, 'div_history')):
            return 1.0

        if len(algorithm.hv_history) == 0:
            return 1.0

        current_hv = algorithm.hv_history[-1]
        current_sp = algorithm.sp_history[-1]
        current_div = algorithm.div_history[-1]

        self.hv_history.append(current_hv)
        self.sp_history.append(current_sp)
        self.div_history.append(current_div)

        hv_improvement = (current_hv - self.best_hv) / max(abs(self.best_hv), 1e-10)
        if hv_improvement > self.tol_hv:
            self.best_hv = current_hv
            self.hv_stagnant = 0
        else:
            self.hv_stagnant += 1

        sp_improvement = (self.best_sp - current_sp) / max(self.best_sp, 1e-10)
        if sp_improvement > self.tol_sp:
            self.best_sp = current_sp
            self.sp_stagnant = 0
        else:
            self.sp_stagnant += 1

        div_improvement = (current_div - self.best_div) / max(abs(self.best_div), 1e-10)
        if div_improvement > self.tol_div:
            self.best_div = current_div
            self.div_stagnant = 0
        else:
            self.div_stagnant += 1

        max_stagnant = max(self.hv_stagnant, self.sp_stagnant, self.div_stagnant)
        progress = max_stagnant / self.n_stagnant

        return min(progress, 1.0)

    def _do_continue(self, algorithm):

        all_stagnant = (
            self.hv_stagnant >= self.n_stagnant and
            self.sp_stagnant >= self.n_stagnant and
            self.div_stagnant >= self.n_stagnant
        )
        return not all_stagnant

class EarlyStoppingCallback:

    def __init__(self, ref_point):
        from pymoo.indicators.hv import HV
        from scipy.spatial.distance import cdist

        self.ref_point = ref_point
        self.hv_indicator = HV(ref_point=ref_point)

    def __call__(self, algorithm):

        F = algorithm.opt.get("F")

        if len(F) == 0:
            return

        hv = self._calculate_hypervolume(F)
        sp = self._calculate_spacing(F)
        div = self._calculate_diversity(F)

        if not hasattr(algorithm, 'hv_history'):
            algorithm.hv_history = []
            algorithm.sp_history = []
            algorithm.div_history = []

        algorithm.hv_history.append(hv)
        algorithm.sp_history.append(sp)
        algorithm.div_history.append(div)

    def _calculate_hypervolume(self, F):

        if len(F) == 0:
            return 0.0
        try:
            return self.hv_indicator(F)
        except:
            return 0.0

    def _calculate_spacing(self, F):

        if len(F) < 2:
            return 0.0

        from scipy.spatial.distance import cdist
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
