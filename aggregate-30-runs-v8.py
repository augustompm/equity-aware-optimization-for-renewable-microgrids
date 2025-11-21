
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

def aggregate_pareto_fronts_30_runs():
    print("=" * 80)
    print("AGGREGATING 30 RUNS V8 - PARETO FRONTS")
    print("=" * 80)

    project_root = Path(__file__).parent

    local_results_dir = project_root / "results"
    server_results_dir = project_root / "results-servidor"

    all_solutions = []
    run_metadata = []

    print("\n1. Collecting Pareto fronts from 30 runs...")

    for seed in range(42, 52):
        pattern = f"v8-run*-seed{seed}-*"
        matching_dirs = list(local_results_dir.glob(pattern))

        if not matching_dirs:
            print(f"  WARNING: No directory found for seed {seed} in local results")
            continue

        run_dir = matching_dirs[0]
        pareto_file = run_dir / "pareto-front-solutions.csv"

        if not pareto_file.exists():
            print(f"  WARNING: {pareto_file} not found")
            continue

        df = pd.read_csv(pareto_file)
        df['source_run'] = run_dir.name
        df['source_seed'] = seed
        df['source_batch'] = 'local'

        all_solutions.append(df)
        run_metadata.append({
            'seed': seed,
            'batch': 'local',
            'n_solutions': len(df),
            'directory': str(run_dir)
        })
        print(f"  [OK] Seed {seed}: {len(df)} solutions (local)")

    for seed in range(52, 72):
        pattern = f"v8-run*-seed{seed}-*"
        matching_dirs = list(server_results_dir.glob(pattern))

        if not matching_dirs:
            print(f"  WARNING: No directory found for seed {seed} in server results")
            continue

        run_dir = matching_dirs[0]
        pareto_file = run_dir / "pareto-front-solutions.csv"

        if not pareto_file.exists():
            print(f"  WARNING: {pareto_file} not found")
            continue

        df = pd.read_csv(pareto_file)
        df['source_run'] = run_dir.name
        df['source_seed'] = seed
        df['source_batch'] = 'server'

        all_solutions.append(df)
        run_metadata.append({
            'seed': seed,
            'batch': 'server',
            'n_solutions': len(df),
            'directory': str(run_dir)
        })
        print(f"  [OK] Seed {seed}: {len(df)} solutions (server)")

    if len(all_solutions) == 0:
        raise ValueError("No Pareto fronts found! Check directory structure")

    print(f"\n2. Concatenating {len(all_solutions)} Pareto fronts...")
    all_solutions_df = pd.concat(all_solutions, ignore_index=True)
    print(f"  Total solutions before filtering: {len(all_solutions_df)}")

    objectives_cols = ['npc_cad', 'lpsp', 'co2_kg', 'gini']
    objectives = all_solutions_df[objectives_cols].values

    print("\n3. Filtering by Pareto dominance...")
    print("  (This may take a few minutes for ~1000 solutions)")

    is_dominated = np.zeros(len(objectives), dtype=bool)

    for i in range(len(objectives)):
        if i % 100 == 0:
            print(f"  Checking solution {i}/{len(objectives)}...")

        for j in range(len(objectives)):
            if i == j:
                continue

            dominates = all(objectives[j] <= objectives[i]) and any(objectives[j] < objectives[i])

            if dominates:
                is_dominated[i] = True
                break

    reference_front_df = all_solutions_df[~is_dominated].copy()
    print(f"  [OK] Reference front size: {len(reference_front_df)} non-dominated solutions")
    print(f"  (Filtered out {len(all_solutions_df) - len(reference_front_df)} dominated solutions)")

    output_dir = project_root / "results-aggregated"
    output_dir.mkdir(exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    reference_front_file = output_dir / f"reference-front-30runs-v8-{timestamp}.csv"
    reference_front_df.to_csv(reference_front_file, index=False)
    print(f"\n4. Reference front saved:")
    print(f"  {reference_front_file}")

    metadata = {
        'timestamp': timestamp,
        'total_runs': len(all_solutions),
        'total_solutions_before_filtering': len(all_solutions_df),
        'reference_front_size': len(reference_front_df),
        'runs': run_metadata
    }

    metadata_file = output_dir / f"reference-front-metadata-{timestamp}.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata: {metadata_file}")

    return reference_front_df, metadata


def aggregate_convergence_metrics_30_runs():
    print("\n" + "=" * 80)
    print("AGGREGATING 30 RUNS V8 - CONVERGENCE METRICS")
    print("=" * 80)

    project_root = Path(__file__).parent
    local_results_dir = project_root / "results"
    server_results_dir = project_root / "results-servidor"

    all_convergence_data = []

    print("\n1. Collecting convergence metrics from 30 runs...")

    for seed in range(42, 52):
        pattern = f"v8-run*-seed{seed}-*"
        matching_dirs = list(local_results_dir.glob(pattern))

        if not matching_dirs:
            continue

        run_dir = matching_dirs[0]
        conv_file = run_dir / "convergence-metrics.csv"

        if not conv_file.exists():
            print(f"  WARNING: {conv_file} not found")
            continue

        df = pd.read_csv(conv_file)
        df['seed'] = seed
        df['batch'] = 'local'
        all_convergence_data.append(df)
        print(f"  [OK] Seed {seed}: {len(df)} generations (local)")

    for seed in range(52, 72):
        pattern = f"v8-run*-seed{seed}-*"
        matching_dirs = list(server_results_dir.glob(pattern))

        if not matching_dirs:
            continue

        run_dir = matching_dirs[0]
        conv_file = run_dir / "convergence-metrics.csv"

        if not conv_file.exists():
            print(f"  WARNING: {conv_file} not found")
            continue

        df = pd.read_csv(conv_file)
        df['seed'] = seed
        df['batch'] = 'server'
        all_convergence_data.append(df)
        print(f"  [OK] Seed {seed}: {len(df)} generations (server)")

    if len(all_convergence_data) == 0:
        raise ValueError("No convergence data found!")

    all_conv_df = pd.concat(all_convergence_data, ignore_index=True)

    print("\n2. Computing statistics (mean ± std) per generation...")

    metrics = ['hypervolume', 'igd_plus', 'spacing', 'diversity', 'n_solutions']

    max_gen = all_conv_df['generation'].max()
    print(f"  Maximum generation found: {max_gen}")

    stats_data = []

    for gen in sorted(all_conv_df['generation'].unique()):
        gen_data = all_conv_df[all_conv_df['generation'] == gen]
        n_runs_at_gen = len(gen_data)

        stats_row = {'generation': gen, 'n_runs': n_runs_at_gen}

        for metric in metrics:
            values = gen_data[metric].values
            stats_row[f'{metric}_mean'] = np.mean(values)
            stats_row[f'{metric}_std'] = np.std(values, ddof=1) if len(values) > 1 else 0.0
            stats_row[f'{metric}_min'] = np.min(values)
            stats_row[f'{metric}_max'] = np.max(values)

        stats_data.append(stats_row)

    stats_df = pd.DataFrame(stats_data)

    output_dir = project_root / "results-aggregated"
    output_dir.mkdir(exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    stats_file = output_dir / f"convergence-statistics-30runs-v8-{timestamp}.csv"
    stats_df.to_csv(stats_file, index=False)
    print(f"\n3. Convergence statistics saved:")
    print(f"  {stats_file}")

    raw_file = output_dir / f"convergence-raw-30runs-v8-{timestamp}.csv"
    all_conv_df.to_csv(raw_file, index=False)
    print(f"  Raw data: {raw_file}")

    return stats_df, all_conv_df


def aggregate_final_metrics_30_runs():
    print("\n" + "=" * 80)
    print("AGGREGATING 30 RUNS V8 - FINAL METRICS")
    print("=" * 80)

    project_root = Path(__file__).parent
    local_results_dir = project_root / "results"
    server_results_dir = project_root / "results-servidor"

    all_final_metrics = []

    print("\n1. Collecting final metrics from summary.json files...")

    for seed in range(42, 52):
        pattern = f"v8-run*-seed{seed}-*"
        matching_dirs = list(local_results_dir.glob(pattern))

        if not matching_dirs:
            continue

        run_dir = matching_dirs[0]
        summary_file = run_dir / "summary.json"

        if not summary_file.exists():
            print(f"  WARNING: {summary_file} not found")
            continue

        with open(summary_file, 'r') as f:
            summary = json.load(f)

        final_metrics = {
            'seed': seed,
            'batch': 'local',
            'n_gen_actual': summary['run_info']['n_gen_actual'],
            'early_stopped': summary['run_info']['early_stopped'],
            'n_solutions': summary['convergence']['final']['n_solutions'],
            'hypervolume_final': summary['convergence']['final']['hypervolume'],
            'igd_plus_final': summary['convergence']['final']['igd_plus'],
            'spacing_final': summary['convergence']['final']['spacing'],
            'diversity_final': summary['convergence']['final']['diversity'],
            'hypervolume_improvement': summary['convergence']['hypervolume_improvement'],
        }

        for obj in ['npc_cad', 'lpsp', 'co2_kg', 'gini']:
            obj_data = summary['pareto_front']['objectives'][obj]
            final_metrics[f'{obj}_min'] = obj_data['min']
            final_metrics[f'{obj}_max'] = obj_data['max']
            final_metrics[f'{obj}_mean'] = obj_data['mean']
            final_metrics[f'{obj}_std'] = obj_data['std']

        for metric in ['re_penetration_pct', 'excess_power_pct', 'lcoe_cad_per_kwh', 'fuel_consumption_liters']:
            metric_data = summary['pareto_front']['additional_metrics'][metric]
            final_metrics[f'{metric}_min'] = metric_data['min']
            final_metrics[f'{metric}_max'] = metric_data['max']
            final_metrics[f'{metric}_mean'] = metric_data['mean']
            final_metrics[f'{metric}_std'] = metric_data['std']

        all_final_metrics.append(final_metrics)
        print(f"  [OK] Seed {seed}: Gen {final_metrics['n_gen_actual']}, HV={final_metrics['hypervolume_final']:.2e}")

    for seed in range(52, 72):
        pattern = f"v8-run*-seed{seed}-*"
        matching_dirs = list(server_results_dir.glob(pattern))

        if not matching_dirs:
            continue

        run_dir = matching_dirs[0]
        summary_file = run_dir / "summary.json"

        if not summary_file.exists():
            print(f"  WARNING: {summary_file} not found")
            continue

        with open(summary_file, 'r') as f:
            summary = json.load(f)

        final_metrics = {
            'seed': seed,
            'batch': 'server',
            'n_gen_actual': summary['run_info']['n_gen_actual'],
            'early_stopped': summary['run_info']['early_stopped'],
            'n_solutions': summary['convergence']['final']['n_solutions'],
            'hypervolume_final': summary['convergence']['final']['hypervolume'],
            'igd_plus_final': summary['convergence']['final']['igd_plus'],
            'spacing_final': summary['convergence']['final']['spacing'],
            'diversity_final': summary['convergence']['final']['diversity'],
            'hypervolume_improvement': summary['convergence']['hypervolume_improvement'],
        }

        for obj in ['npc_cad', 'lpsp', 'co2_kg', 'gini']:
            obj_data = summary['pareto_front']['objectives'][obj]
            final_metrics[f'{obj}_min'] = obj_data['min']
            final_metrics[f'{obj}_max'] = obj_data['max']
            final_metrics[f'{obj}_mean'] = obj_data['mean']
            final_metrics[f'{obj}_std'] = obj_data['std']

        for metric in ['re_penetration_pct', 'excess_power_pct', 'lcoe_cad_per_kwh', 'fuel_consumption_liters']:
            metric_data = summary['pareto_front']['additional_metrics'][metric]
            final_metrics[f'{metric}_min'] = metric_data['min']
            final_metrics[f'{metric}_max'] = metric_data['max']
            final_metrics[f'{metric}_mean'] = metric_data['mean']
            final_metrics[f'{metric}_std'] = metric_data['std']

        all_final_metrics.append(final_metrics)
        print(f"  [OK] Seed {seed}: Gen {final_metrics['n_gen_actual']}, HV={final_metrics['hypervolume_final']:.2e}")

    final_metrics_df = pd.DataFrame(all_final_metrics)

    output_dir = project_root / "results-aggregated"
    output_dir.mkdir(exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    final_file = output_dir / f"final-metrics-30runs-v8-{timestamp}.csv"
    final_metrics_df.to_csv(final_file, index=False)
    print(f"\n2. Final metrics saved:")
    print(f"  {final_file}")

    print("\n3. Overall statistics (30 runs):")
    print(f"  Hypervolume final: {final_metrics_df['hypervolume_final'].mean():.2e} ± {final_metrics_df['hypervolume_final'].std():.2e}")
    print(f"  IGD+ final: {final_metrics_df['igd_plus_final'].mean():.4f} ± {final_metrics_df['igd_plus_final'].std():.4f}")
    print(f"  N solutions: {final_metrics_df['n_solutions'].mean():.1f} ± {final_metrics_df['n_solutions'].std():.1f}")
    print(f"  Generations: {final_metrics_df['n_gen_actual'].mean():.1f} ± {final_metrics_df['n_gen_actual'].std():.1f}")
    print(f"  Early stopped: {final_metrics_df['early_stopped'].sum()}/30 runs")

    return final_metrics_df


def main():
    print("=" * 80)
    print("AGGREGATE 30 RUNS V8 - STATISTICAL ANALYSIS")
    print("Project P10 - SysCon 2026")
    print("=" * 80)

    try:
        reference_front_df, ref_metadata = aggregate_pareto_fronts_30_runs()

        conv_stats_df, conv_raw_df = aggregate_convergence_metrics_30_runs()

        final_metrics_df = aggregate_final_metrics_30_runs()

        print("\n" + "=" * 80)
        print("AGGREGATION COMPLETE")
        print("=" * 80)
        print("\nAll CSV files saved to: results-aggregated/")
        print("\nNext step: Run visualization script to generate publication figures")
        print("  python generate-publication-figures-v8.py")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
