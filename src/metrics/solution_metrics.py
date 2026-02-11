import numpy as np
from simulation.system_simulator import simulate_system
from metrics.additional_metrics import calculate_all_additional_metrics

def calculate_solution_metrics(
    decision_vector: np.ndarray,
    objectives: np.ndarray,
    constraints: np.ndarray,
    system_config: dict,
    solution_index: int = 0
) -> dict:

    pv_kw = decision_vector[0]
    wind_mw = decision_vector[1]
    battery_mwh = decision_vector[2]
    diesel_mw = decision_vector[3]

    npc = objectives[0]
    lpsp = objectives[1]
    co2 = objectives[2]
    gini = objectives[3]

    if npc == 0.0:
        raise ValueError(
            f"Solution {solution_index}: NPC = 0.0 is INVALID. "
            "This indicates missing cost data or calculation error."
        )

    if not (0 <= lpsp <= 1):
        raise ValueError(
            f"Solution {solution_index}: LPSP = {lpsp} is OUT OF RANGE [0,1]. "
            "This indicates calculation error."
        )

    decision_vars_dict = {
        'n_pv_kw': pv_kw,
        'n_wind_mw': wind_mw,
        'e_battery_mwh': battery_mwh,
        'p_diesel_mw': diesel_mw
    }

    objectives_dict, constraints_dict, dispatch_summary = simulate_system(
        decision_vars=decision_vars_dict,
        system_config=system_config
    )

    additional_metrics = calculate_all_additional_metrics(
        simulation_results=dispatch_summary,
        npc=npc,
        config=system_config
    )

    is_feasible = np.all(constraints <= 0) if len(constraints) > 0 else True
    total_violation = np.sum(np.maximum(constraints, 0)) if len(constraints) > 0 else 0.0

    solution_dict = {
        'solution_index': solution_index,
        'pv_kw': float(pv_kw),
        'wind_mw': float(wind_mw),
        'battery_mwh': float(battery_mwh),
        'diesel_mw': float(diesel_mw),
        'npc_cad': float(npc),
        'lpsp': float(lpsp),
        'co2_kg': float(co2),
        'gini': float(gini),
        're_penetration_pct': additional_metrics['re_penetration_pct'],
        'excess_power_pct': additional_metrics['excess_power_pct'],
        'excess_power_mwh': additional_metrics['excess_power_mwh'],
        'lcoe_cad_per_kwh': additional_metrics['lcoe_cad_per_kwh'],
        'fuel_consumption_liters': additional_metrics['fuel_consumption_liters'],
        'fuel_consumption_kg': additional_metrics['fuel_consumption_kg'],
        'load_annual_kwh': additional_metrics['load_annual_kwh'],
        'is_feasible': bool(is_feasible),
        'total_violation': float(total_violation)
    }

    if not (0 <= solution_dict['re_penetration_pct'] <= 100):
        raise ValueError(
            f"Solution {solution_index}: RE% = {solution_dict['re_penetration_pct']} "
            "is OUT OF RANGE [0,100]. Calculation error."
        )

    if solution_dict['lcoe_cad_per_kwh'] <= 0:
        raise ValueError(
            f"Solution {solution_index}: LCOE = {solution_dict['lcoe_cad_per_kwh']} "
            "is NON-POSITIVE. Calculation error."
        )

    return solution_dict

def calculate_pareto_front_metrics(
    F: np.ndarray,
    X: np.ndarray,
    G: np.ndarray,
    system_config: dict
) -> list:

    n_solutions = F.shape[0]
    pareto_metrics = []

    for i in range(n_solutions):
        solution_metrics = calculate_solution_metrics(
            decision_vector=X[i, :],
            objectives=F[i, :],
            constraints=G[i, :] if G is not None and len(G) > 0 else np.array([]),
            system_config=system_config,
            solution_index=i
        )
        pareto_metrics.append(solution_metrics)

    return pareto_metrics
