import numpy as np
from typing import Optional, Dict
from scipy.spatial.distance import cdist

def calculate_hypervolume(pareto_front: np.ndarray, reference_point: np.ndarray) -> float:

    from pymoo.indicators.hv import HV

    n_obj = pareto_front.shape[1]

    if pareto_front.shape[0] == 0:
        return 0.0

    if not np.all(pareto_front <= reference_point):
        raise ValueError("Reference point must dominate all Pareto front points")

    ind = HV(ref_point=reference_point)
    hv_value = ind(pareto_front)

    return hv_value

def calculate_igd(pareto_front: np.ndarray, reference_front: np.ndarray) -> float:

    if pareto_front.shape[0] == 0:
        return np.inf

    if reference_front.shape[0] == 0:
        return np.inf

    distances = cdist(reference_front, pareto_front, metric='euclidean')

    min_distances = np.min(distances, axis=1)

    igd_value = np.mean(min_distances)

    return igd_value

def calculate_igd_plus(pareto_front: np.ndarray, reference_front: np.ndarray) -> float:

    if pareto_front.shape[0] == 0:
        return np.inf

    if reference_front.shape[0] == 0:
        return np.inf

    n_ref = reference_front.shape[0]
    n_obtained = pareto_front.shape[0]
    n_obj = reference_front.shape[1]

    min_modified_distances = np.zeros(n_ref)

    for i in range(n_ref):
        z = reference_front[i, :]

        modified_distances = np.zeros(n_obtained)

        for j in range(n_obtained):
            a = pareto_front[j, :]

            inferiority_vector = np.maximum(a - z, 0.0)

            d_plus = np.sqrt(np.sum(inferiority_vector ** 2))

            modified_distances[j] = d_plus

        min_modified_distances[i] = np.min(modified_distances)

    igd_plus_value = np.mean(min_modified_distances)

    return igd_plus_value

def calculate_gd(pareto_front: np.ndarray, reference_front: np.ndarray) -> float:

    if pareto_front.shape[0] == 0:
        return np.inf

    if reference_front.shape[0] == 0:
        return np.inf

    distances = cdist(pareto_front, reference_front, metric='euclidean')

    min_distances = np.min(distances, axis=1)

    gd_value = np.mean(min_distances)

    return gd_value

def calculate_spacing(pareto_front: np.ndarray) -> float:

    if pareto_front.shape[0] <= 1:
        return 0.0

    n_solutions = pareto_front.shape[0]

    distances = cdist(pareto_front, pareto_front, metric='euclidean')

    np.fill_diagonal(distances, np.inf)

    min_distances = np.min(distances, axis=1)

    d_mean = np.mean(min_distances)

    spacing_value = np.sqrt(np.sum((min_distances - d_mean) ** 2) / (n_solutions - 1))

    return spacing_value

def calculate_diversity(pareto_front: np.ndarray) -> float:

    if pareto_front.shape[0] == 0:
        return 0.0

    centroid = np.mean(pareto_front, axis=0)

    squared_distances = np.sum((pareto_front - centroid) ** 2)

    diversity_value = squared_distances

    return diversity_value

def calculate_generation_metrics(pareto_front: np.ndarray,
                                reference_point: np.ndarray,
                                reference_front: Optional[np.ndarray] = None) -> Dict[str, float]:

    metrics = {}

    metrics['n_solutions'] = pareto_front.shape[0]

    if pareto_front.shape[0] > 0:
        metrics['hypervolume'] = calculate_hypervolume(pareto_front, reference_point)
        metrics['spacing'] = calculate_spacing(pareto_front)
        metrics['diversity'] = calculate_diversity(pareto_front)

        if reference_front is not None:
            metrics['igd'] = calculate_igd(pareto_front, reference_front)
            metrics['igd_plus'] = calculate_igd_plus(pareto_front, reference_front)
            metrics['gd'] = calculate_gd(pareto_front, reference_front)
    else:
        metrics['hypervolume'] = 0.0
        metrics['spacing'] = 0.0
        metrics['diversity'] = 0.0

        if reference_front is not None:
            metrics['igd'] = np.inf
            metrics['igd_plus'] = np.inf
            metrics['gd'] = np.inf

    return metrics

def analyze_convergence_history(history, reference_point: Optional[np.ndarray] = None,
                                reference_front: Optional[np.ndarray] = None) -> np.ndarray:

    if reference_point is None:
        all_F = []
        for entry in history:
            F = entry.opt.get("F")
            if F is not None and len(F) > 0:
                all_F.append(F)

        if len(all_F) > 0:
            combined_F = np.vstack(all_F)
            reference_point = get_reference_point(combined_F, offset=0.2)
        else:
            reference_point = np.array([1e10, 1.0, 1e10, 1.0])

    convergence_metrics = []

    for gen, entry in enumerate(history):
        F = entry.opt.get("F")

        metrics = calculate_generation_metrics(F, reference_point, reference_front)

        row = [
            gen,
            metrics['n_solutions'],
            metrics['hypervolume'],
            metrics['spacing'],
            metrics['diversity']
        ]

        if reference_front is not None:
            row.append(metrics.get('igd', np.nan))
            row.append(metrics.get('igd_plus', np.nan))
            row.append(metrics.get('gd', np.nan))

        convergence_metrics.append(row)

    convergence_data = np.array(convergence_metrics)

    return convergence_data

def get_reference_point(pareto_front: np.ndarray, offset: float = 0.1) -> np.ndarray:

    max_values = np.max(pareto_front, axis=0)

    reference_point = max_values * (1.0 + offset)

    return reference_point
