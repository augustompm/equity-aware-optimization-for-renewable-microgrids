from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.indicators.hv import HV
import pandas as pd
import numpy as np
import sys
import json
from pathlib import Path
from datetime import datetime
from scipy.spatial.distance import cdist

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from optimization.nsga3_problem import MicrogridOptimizationProblem

def calculate_hypervolume(F, ref_point):
    hv_indicator = HV(ref_point=ref_point)
    return hv_indicator(F)

def calculate_spacing(F):
    if len(F) < 2:
        return 0.0

    distances = cdist(F, F, metric='euclidean')
    np.fill_diagonal(distances, np.inf)
    min_distances = distances.min(axis=1)

    d_bar = min_distances.mean()
    sp = np.sqrt(((min_distances - d_bar) ** 2).sum() / (len(F) - 1))

    return sp

def calculate_diversity(F):
    if len(F) < 2:
        return 0.0

    centroid = F.mean(axis=0)
    diversity = ((F - centroid) ** 2).sum()

    return diversity

def create_variation_config(variation_name):

    base_config = {
        'load_profile_path': project_root / 'data' / 'load-profile-8760h.csv',
        'solar_cf_path': project_root / 'data' / 'solar-capacity-factors.csv',
        'wind_cf_path': project_root / 'data' / 'wind-capacity-factors.csv',
        'meteorology_path': project_root / 'data' / 'meteorology-8760h.csv',
        'discount_rate': 0.03,
        'lifetime_years': 25,
        'diesel_efficiency': 0.30,
        'diesel_fuel_cost_per_mmbtu': 20.0,
        'diesel_min_load_fraction': 0.30,
        'pv_tilt_deg': 60,
        'wind_hub_height_m': 100,
        'battery_c_rate': 0.25,
        'battery_efficiency': 0.90,
        'battery_dod_max': 0.80,
        'renewable_fraction_max': 0.20,
        'reserve_fraction': 0.15,
        'lpsp_limit': 0.05,
        'grid_connected': False
    }

    if variation_name == 'baseline':
        config = base_config.copy()
        config['area_available_m2'] = 100000.0
        config['bounds'] = {
            'n_pv_kw': (0, 1000),
            'n_wind_mw': (0, 10),
            'e_battery_mwh': (0, 20),
            'p_diesel_mw': (0, 10)
        }
        config['variation_description'] = 'Urban-only area constraint (100k m2), standard bounds'
        config['scientific_basis'] = 'Baseline for comparison with previous test 84x75'

    elif variation_name == 'two-bus':
        config = base_config.copy()
        config['area_available_m2'] = 100000.0
        config['area_wind_remote_m2'] = 2000000.0
        config['bounds'] = {
            'n_pv_kw': (0, 1000),
            'n_wind_mw': (0, 5),
            'e_battery_mwh': (0, 20),
            'p_diesel_mw': (0, 10)
        }
        config['variation_description'] = 'Two-bus model: Urban 100k m2 + Remote wind 2 km2'
        config['scientific_basis'] = 'CASES Inuvik 2020 lines 277-278: "2-5MW wind farm proposed, 5-10km from transmission"'

    elif variation_name == 'narrow':
        config = base_config.copy()
        config['area_available_m2'] = 100000.0
        config['bounds'] = {
            'n_pv_kw': (0, 700),
            'n_wind_mw': (0, 0.5),
            'e_battery_mwh': (5, 20),
            'p_diesel_mw': (3, 8)
        }
        config['variation_description'] = 'Narrow bounds sensitivity: PV 700kW, Wind 0.5MW, Battery 5-20MWh, Diesel 3-8MW'
        config['scientific_basis'] = 'CASES baseline 55kW PV + McKinley 2025 battery 4-16MWh + CASES diesel 5.73MW'

    else:
        raise ValueError(f"Unknown variation: {variation_name}")

    return config

def run_diagnostic_variation(variation_name, pop_size=84, n_gen=100, seed=42):

    print("=" * 80)
    print(f"DIAGNOSTIC VARIATION: {variation_name.upper()}")
    print("NSGA-III Multi-Objective Optimization - Arctic Microgrid")
    print("=" * 80)
    print()

    config = create_variation_config(variation_name)

    print(f"Configuration:")
    print(f"  {config['variation_description']}")
    print(f"  Scientific basis: {config['scientific_basis']}")
    print()

    print(f"Decision variable bounds:")
    for var, bounds in config['bounds'].items():
        print(f"  {var:20s}: [{bounds[0]:8.2f}, {bounds[1]:8.2f}]")
    print()

    if 'area_wind_remote_m2' in config:
        print(f"Area constraints:")
        print(f"  Urban area:       {config['area_available_m2']:10.0f} m2")
        print(f"  Remote wind area: {config['area_wind_remote_m2']:10.0f} m2")
        print()
    else:
        print(f"Area constraint: {config['area_available_m2']:.0f} m2 (urban-only)")
        print()

    n_partitions = 6
    ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=n_partitions)
    actual_ref_dirs = len(ref_dirs)

    print(f"NSGA-III Configuration:")
    print(f"  Reference directions: {actual_ref_dirs} (n_partitions={n_partitions})")
    print(f"  Population size: {pop_size}")
    print(f"  Generations: {n_gen}")
    print(f"  Total evaluations: {pop_size * n_gen}")
    print(f"  Seed: {seed}")
    print()

    estimated_time_min = (pop_size * n_gen * 1.0) / 60.0
    estimated_time_max = (pop_size * n_gen * 1.5) / 60.0
    print(f"Estimated runtime: {estimated_time_min:.0f} - {estimated_time_max:.0f} minutes")
    print(f"                  ({estimated_time_min/60:.1f} - {estimated_time_max/60:.1f} hours)")
    print()

    problem = MicrogridOptimizationProblem(config)

    algorithm = NSGA3(
        pop_size=pop_size,
        ref_dirs=ref_dirs,
        eliminate_duplicates=True
    )

    print("Starting optimization...")
    print()

    start_time = datetime.now()

    res = minimize(
        problem,
        algorithm,
        ('n_gen', n_gen),
        seed=seed,
        verbose=True,
        save_history=True
    )

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    print()
    print("=" * 80)
    print(f"VARIATION {variation_name.upper()} COMPLETE")
    print("=" * 80)
    print(f"Runtime: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes, {elapsed/3600:.2f} hours)")
    print()

    if res.F is None or len(res.F) == 0:
        print("WARNING: No Pareto front found. All solutions infeasible.")
        return None

    print(f"Pareto front size: {len(res.F)}")

    if res.CV is not None:
        if res.CV.ndim > 1:
            CV_total = res.CV.sum(axis=1)
        else:
            CV_total = res.CV

        n_feasible = np.sum(CV_total <= 1e-6)
        print(f"Feasible solutions: {n_feasible}/{len(res.F)} ({n_feasible/len(res.F)*100:.1f}%)")
    print()

    print("Calculating convergence metrics...")

    metrics_history = []
    ref_point = np.array([2e8, 0.2, 1e8, 1.0])

    for gen, entry in enumerate(res.history):
        F = entry.opt.get("F")
        CV = entry.opt.get("CV")

        if CV.ndim > 1:
            CV = CV.sum(axis=1)

        n_feasible = np.sum(CV <= 1e-6)
        n_pareto = len(F)

        feasible_mask = CV <= 1e-6
        F_feasible = F[feasible_mask] if n_feasible > 0 else F

        if len(F_feasible) > 0:
            hv = calculate_hypervolume(F_feasible, ref_point)
            sp = calculate_spacing(F_feasible)
            div = calculate_diversity(F_feasible)
        else:
            hv = sp = div = 0.0

        metrics_history.append({
            'generation': gen,
            'n_pareto': n_pareto,
            'n_feasible': n_feasible,
            'hypervolume': hv,
            'spacing': sp,
            'diversity': div,
            'min_NPC': F[:, 0].min(),
            'max_NPC': F[:, 0].max(),
            'mean_NPC': F[:, 0].mean(),
            'min_LPSP': F[:, 1].min(),
            'max_LPSP': F[:, 1].max(),
            'mean_LPSP': F[:, 1].mean(),
            'min_CO2': F[:, 2].min(),
            'max_CO2': F[:, 2].max(),
            'mean_CO2': F[:, 2].mean(),
            'min_Gini': F[:, 3].min(),
            'max_Gini': F[:, 3].max(),
            'mean_Gini': F[:, 3].mean()
        })

    df_metrics = pd.DataFrame(metrics_history)

    wind_utilization = np.sum(res.X[:, 1] > 0.1) / len(res.X)
    max_wind = res.X[:, 1].max()
    mean_wind_nonzero = res.X[res.X[:, 1] > 0.1, 1].mean() if np.sum(res.X[:, 1] > 0.1) > 0 else 0.0

    npc_co2_corr = np.corrcoef(res.F[:, 0], res.F[:, 2])[0, 1]

    results_dir = project_root / "results" / "diagnostic-v2"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = start_time.strftime("%Y%m%d_%H%M%S")

    pareto_file = results_dir / f"pareto-{variation_name}-{timestamp}.csv"
    save_pareto_front(res.X, res.F, res.G if hasattr(res, 'G') else None, pareto_file)

    metrics_file = results_dir / f"metrics-{variation_name}-{timestamp}.csv"
    df_metrics.to_csv(metrics_file, index=False)

    hv_initial = df_metrics.iloc[0]['hypervolume']
    hv_final = df_metrics.iloc[-1]['hypervolume']
    hv_improvement = (hv_final - hv_initial) / hv_initial * 100 if hv_initial > 0 else 0

    summary = {
        'variation': variation_name,
        'timestamp': timestamp,
        'configuration': {
            'population_size': pop_size,
            'generations': n_gen,
            'total_evaluations': pop_size * n_gen,
            'seed': seed,
            'n_partitions': n_partitions,
            'reference_directions': actual_ref_dirs,
            'bounds': config['bounds'],
            'area_urban_m2': config['area_available_m2'],
            'area_wind_remote_m2': config.get('area_wind_remote_m2', 0),
            'variation_description': config['variation_description'],
            'scientific_basis': config['scientific_basis']
        },
        'runtime': {
            'seconds': elapsed,
            'minutes': elapsed / 60,
            'hours': elapsed / 3600
        },
        'final_results': {
            'pareto_size': len(res.F),
            'n_feasible': int(n_feasible),
            'feasibility_rate': float(n_feasible / len(res.F))
        },
        'convergence_metrics': {
            'hypervolume_initial': float(hv_initial),
            'hypervolume_final': float(hv_final),
            'hv_improvement_pct': float(hv_improvement),
            'final_spacing': float(df_metrics.iloc[-1]['spacing']),
            'final_diversity': float(df_metrics.iloc[-1]['diversity'])
        },
        'objectives_ranges': {
            'NPC': {
                'min': float(res.F[:, 0].min()),
                'max': float(res.F[:, 0].max()),
                'mean': float(res.F[:, 0].mean()),
                'range_pct': float((res.F[:, 0].max() - res.F[:, 0].min()) / res.F[:, 0].min() * 100)
            },
            'LPSP': {
                'min': float(res.F[:, 1].min()),
                'max': float(res.F[:, 1].max()),
                'mean': float(res.F[:, 1].mean()),
                'range_abs': float(res.F[:, 1].max() - res.F[:, 1].min())
            },
            'CO2': {
                'min': float(res.F[:, 2].min()),
                'max': float(res.F[:, 2].max()),
                'mean': float(res.F[:, 2].mean()),
                'range_pct': float((res.F[:, 2].max() - res.F[:, 2].min()) / res.F[:, 2].min() * 100)
            },
            'Gini': {
                'min': float(res.F[:, 3].min()),
                'max': float(res.F[:, 3].max()),
                'mean': float(res.F[:, 3].mean()),
                'range_pct': float((res.F[:, 3].max() - res.F[:, 3].min()) / res.F[:, 3].min() * 100)
            }
        },
        'diagnostic_insights': {
            'wind_utilization_pct': float(wind_utilization * 100),
            'max_wind_mw': float(max_wind),
            'mean_wind_nonzero_mw': float(mean_wind_nonzero),
            'npc_co2_correlation': float(npc_co2_corr)
        }
    }

    summary_file = results_dir / f"summary-{variation_name}-{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved:")
    print(f"  Pareto front: {pareto_file.name}")
    print(f"  Metrics:      {metrics_file.name}")
    print(f"  Summary:      {summary_file.name}")
    print()

    print_variation_summary(summary)

    return {
        'variation': variation_name,
        'result': res,
        'metrics': df_metrics,
        'summary': summary,
        'files': {
            'pareto': pareto_file,
            'metrics': metrics_file,
            'summary': summary_file
        }
    }

def save_pareto_front(X, F, G, filename):
    df = pd.DataFrame(X, columns=['n_pv_kw', 'n_wind_mw', 'e_battery_mwh', 'p_diesel_mw'])

    df['NPC'] = F[:, 0]
    df['LPSP'] = F[:, 1]
    df['CO2'] = F[:, 2]
    df['Gini'] = F[:, 3]

    if G is not None:
        df['CV_total'] = np.sum(np.maximum(G, 0), axis=1)

    df.to_csv(filename, index=False)

def print_variation_summary(summary):
    print("=" * 80)
    print(f"VARIATION SUMMARY: {summary['variation'].upper()}")
    print("=" * 80)
    print()

    print("Convergence:")
    print(f"  HV initial:     {summary['convergence_metrics']['hypervolume_initial']:.4e}")
    print(f"  HV final:       {summary['convergence_metrics']['hypervolume_final']:.4e}")
    print(f"  HV improvement: {summary['convergence_metrics']['hv_improvement_pct']:+.2f}%")
    print(f"  Final spacing:  {summary['convergence_metrics']['final_spacing']:.4e}")
    print(f"  Final diversity:{summary['convergence_metrics']['final_diversity']:.4e}")
    print()

    print("Objective Ranges:")
    print(f"  NPC:  ${summary['objectives_ranges']['NPC']['min']:.2e} - ${summary['objectives_ranges']['NPC']['max']:.2e} ({summary['objectives_ranges']['NPC']['range_pct']:.1f}% range)")
    print(f"  LPSP: {summary['objectives_ranges']['LPSP']['min']:.4f} - {summary['objectives_ranges']['LPSP']['max']:.4f} (abs range {summary['objectives_ranges']['LPSP']['range_abs']:.4f})")
    print(f"  CO2:  {summary['objectives_ranges']['CO2']['min']:.2e} - {summary['objectives_ranges']['CO2']['max']:.2e} ({summary['objectives_ranges']['CO2']['range_pct']:.1f}% range)")
    print(f"  Gini: {summary['objectives_ranges']['Gini']['min']:.4f} - {summary['objectives_ranges']['Gini']['max']:.4f} ({summary['objectives_ranges']['Gini']['range_pct']:.1f}% range)")
    print()

    print("Diagnostic Insights:")
    print(f"  Wind utilization:   {summary['diagnostic_insights']['wind_utilization_pct']:.1f}% of solutions use Wind > 0.1 MW")
    print(f"  Max wind capacity:  {summary['diagnostic_insights']['max_wind_mw']:.3f} MW")
    print(f"  NPC-CO2 correlation:{summary['diagnostic_insights']['npc_co2_correlation']:.4f}")
    print()

    print("=" * 80)

if __name__ == "__main__":

    import sys

    if len(sys.argv) > 1:
        variation = sys.argv[1].lower()
        if variation not in ['baseline', 'two-bus', 'narrow']:
            print(f"Error: Unknown variation '{variation}'")
            print("Valid options: baseline, two-bus, narrow")
            sys.exit(1)

        variations = [variation]
    else:
        variations = ['baseline', 'two-bus', 'narrow']

    print("=" * 80)
    print("NSGA-III DIAGNOSTIC TEST V2")
    print("Arctic Microgrid Multi-Objective Optimization")
    print("=" * 80)
    print()
    print(f"Variations to run: {', '.join(variations)}")
    print(f"Total runtime estimate: {len(variations) * 3:.0f} hours")
    print()
    print("Scientific basis documented in:")
    print("  - DIAGNOSTIC-TEST-V2-SCIENTIFIC-BASIS.md")
    print("  - cases_energy_profile_inuvik.yaml lines 277-278")
    print("  - high_level_multi_objective_model_for_microgrid_design.yaml line 118")
    print()

    results_all = {}

    for variation in variations:
        result = run_diagnostic_variation(
            variation_name=variation,
            pop_size=84,
            n_gen=100,
            seed=42
        )

        if result:
            results_all[variation] = result
            print(f"\n✓ Variation '{variation}' completed successfully\n")
        else:
            print(f"\n✗ Variation '{variation}' FAILED\n")

    print("=" * 80)
    print("ALL VARIATIONS COMPLETE")
    print("=" * 80)
    print()
    print(f"Successful variations: {len(results_all)}/{len(variations)}")
    print()

    if len(results_all) == len(variations):
        print("Next step: Run diagnostic-plots-v2.py to generate comparative visualizations")
        print()
