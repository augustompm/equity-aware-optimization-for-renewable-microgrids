"""
Results saver for V8 production runs.

Saves all data in structured format per rules.md Section 9:
- Raw CSV files (convergence metrics, pareto solutions, decision variables)
- Summary JSON (all metrics + configuration)
- Figures (PNG, PDF, SVG)

Directory structure:
results/
  run{id}_seed{seed}_{timestamp}/
    convergence-metrics.csv
    pareto-front-solutions.csv
    summary.json
    figures/
      convergence-all.{png,pdf,svg}
      pareto-2d-*.{png,pdf,svg}
      parallel-coordinates.{png,pdf,svg}
      decision-variables-distribution.{png,pdf,svg}

References:
- rules.md Section 9: "NUNCA gerar gráficos sem preservar dados raw subjacentes"
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def save_v8_results(
    run_id,
    seed,
    timestamp,
    config,
    bounds,
    metrics_history,
    pareto_metrics,
    F,
    X,
    G,
    n_gen_actual,
    early_stopped,
    results_base_dir='results'
):
    """
    Save all V8 run results in structured format.

    Args:
        run_id: Run identifier
        seed: Random seed
        timestamp: Run timestamp string
        config: V8 configuration dict
        bounds: Decision variable bounds
        metrics_history: List of dicts from callback (HV, IGD+, SP, DIV per generation)
        pareto_metrics: List of dicts with all metrics per solution
        F: Objectives matrix (n_solutions, 4)
        X: Decision variables matrix (n_solutions, 4)
        G: Constraints matrix (n_solutions, 6)
        n_gen_actual: Actual generations run
        early_stopped: Boolean
        results_base_dir: Base directory for results

    Returns:
        Path to results directory

    Saves:
        - convergence-metrics.csv
        - pareto-front-solutions.csv
        - summary.json

    Raises:
        ValueError: If critical data is missing or invalid per rules.md
    """
    results_dir = Path(results_base_dir) / f"v8-run{run_id}-seed{seed}-{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    if len(metrics_history) == 0:
        raise ValueError("metrics_history is empty - optimization did not run")

    if len(pareto_metrics) == 0:
        raise ValueError("pareto_metrics is empty - no Pareto solutions found")

    df_convergence = pd.DataFrame(metrics_history)
    df_convergence.to_csv(results_dir / 'convergence-metrics.csv', index=False)

    df_pareto = pd.DataFrame(pareto_metrics)
    df_pareto.to_csv(results_dir / 'pareto-front-solutions.csv', index=False)

    initial_metrics = metrics_history[0]
    final_metrics = metrics_history[-1]

    hv_values = [m['hypervolume'] for m in metrics_history]
    igd_values = [m['igd_plus'] for m in metrics_history if m['igd_plus'] != np.inf]

    summary = {
        'run_info': {
            'run_id': run_id,
            'seed': seed,
            'timestamp': timestamp,
            'n_gen_configured': config.get('n_generations', None),
            'n_gen_actual': n_gen_actual,
            'early_stopped': early_stopped
        },
        'configuration': {
            'battery_bounds_mwh': bounds['battery_kwh'],
            'pv_bounds_kw': bounds['pv_kw'],
            'wind_bounds_mw': bounds['wind_kw'],
            'diesel_bounds_mw': bounds['diesel_kw'],
            'wind_cf_path': config['wind_cf_path'].name,
            'area_pv_m2': config['area_available_pv_m2'],
            'area_wind_m2': config['area_available_wind_m2'],
            'renewable_fraction_max': config['renewable_fraction_max'],
            'lpsp_limit': config['lpsp_limit'],
            'reserve_fraction': config['reserve_fraction'],
            'lifetime_years': config['lifetime_years'],
            'discount_rate': config['discount_rate']
        },
        'convergence': {
            'initial': {
                'generation': initial_metrics['generation'],
                'hypervolume': initial_metrics['hypervolume'],
                'igd_plus': initial_metrics['igd_plus'],
                'spacing': initial_metrics['spacing'],
                'diversity': initial_metrics['diversity'],
                'n_solutions': initial_metrics['n_solutions']
            },
            'final': {
                'generation': final_metrics['generation'],
                'hypervolume': final_metrics['hypervolume'],
                'igd_plus': final_metrics['igd_plus'],
                'spacing': final_metrics['spacing'],
                'diversity': final_metrics['diversity'],
                'n_solutions': final_metrics['n_solutions']
            },
            'hypervolume_improvement': final_metrics['hypervolume'] - initial_metrics['hypervolume'],
            'hypervolume_max': max(hv_values),
            'igd_plus_min': min(igd_values) if len(igd_values) > 0 else None
        },
        'pareto_front': {
            'n_solutions': len(pareto_metrics),
            'n_feasible': sum(1 for s in pareto_metrics if s['is_feasible']),
            'objectives': {
                'npc_cad': {
                    'min': float(np.min(F[:, 0])),
                    'max': float(np.max(F[:, 0])),
                    'mean': float(np.mean(F[:, 0])),
                    'std': float(np.std(F[:, 0]))
                },
                'lpsp': {
                    'min': float(np.min(F[:, 1])),
                    'max': float(np.max(F[:, 1])),
                    'mean': float(np.mean(F[:, 1])),
                    'std': float(np.std(F[:, 1]))
                },
                'co2_kg': {
                    'min': float(np.min(F[:, 2])),
                    'max': float(np.max(F[:, 2])),
                    'mean': float(np.mean(F[:, 2])),
                    'std': float(np.std(F[:, 2]))
                },
                'gini': {
                    'min': float(np.min(F[:, 3])),
                    'max': float(np.max(F[:, 3])),
                    'mean': float(np.mean(F[:, 3])),
                    'std': float(np.std(F[:, 3]))
                }
            },
            'decision_variables': {
                'pv_kw': {
                    'min': float(np.min(X[:, 0])),
                    'max': float(np.max(X[:, 0])),
                    'mean': float(np.mean(X[:, 0])),
                    'std': float(np.std(X[:, 0]))
                },
                'wind_mw': {
                    'min': float(np.min(X[:, 1])),
                    'max': float(np.max(X[:, 1])),
                    'mean': float(np.mean(X[:, 1])),
                    'std': float(np.std(X[:, 1]))
                },
                'battery_mwh': {
                    'min': float(np.min(X[:, 2])),
                    'max': float(np.max(X[:, 2])),
                    'mean': float(np.mean(X[:, 2])),
                    'std': float(np.std(X[:, 2]))
                },
                'diesel_mw': {
                    'min': float(np.min(X[:, 3])),
                    'max': float(np.max(X[:, 3])),
                    'mean': float(np.mean(X[:, 3])),
                    'std': float(np.std(X[:, 3]))
                }
            },
            'additional_metrics': {
                're_penetration_pct': {
                    'min': min(s['re_penetration_pct'] for s in pareto_metrics),
                    'max': max(s['re_penetration_pct'] for s in pareto_metrics),
                    'mean': np.mean([s['re_penetration_pct'] for s in pareto_metrics]),
                    'std': np.std([s['re_penetration_pct'] for s in pareto_metrics])
                },
                'excess_power_pct': {
                    'min': min(s['excess_power_pct'] for s in pareto_metrics),
                    'max': max(s['excess_power_pct'] for s in pareto_metrics),
                    'mean': np.mean([s['excess_power_pct'] for s in pareto_metrics]),
                    'std': np.std([s['excess_power_pct'] for s in pareto_metrics])
                },
                'lcoe_cad_per_kwh': {
                    'min': min(s['lcoe_cad_per_kwh'] for s in pareto_metrics),
                    'max': max(s['lcoe_cad_per_kwh'] for s in pareto_metrics),
                    'mean': np.mean([s['lcoe_cad_per_kwh'] for s in pareto_metrics]),
                    'std': np.std([s['lcoe_cad_per_kwh'] for s in pareto_metrics])
                },
                'fuel_consumption_liters': {
                    'min': min(s['fuel_consumption_liters'] for s in pareto_metrics),
                    'max': max(s['fuel_consumption_liters'] for s in pareto_metrics),
                    'mean': np.mean([s['fuel_consumption_liters'] for s in pareto_metrics]),
                    'std': np.std([s['fuel_consumption_liters'] for s in pareto_metrics])
                }
            }
        }
    }

    with open(results_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    return results_dir
