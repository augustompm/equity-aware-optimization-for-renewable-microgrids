import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

def aggregate_30_runs(results_dir: Path):

    print("=" * 80)
    print("AGGREGATING 30 RUNS (seeds 42-71)")
    print("=" * 80)

    all_solutions = []

    for seed in range(42, 72):
        pattern = f"v8-run*-seed{seed}-20260210_*"
        matching_dirs = list(results_dir.glob(pattern))

        if not matching_dirs:
            print(f"  [WARN] No directory found for seed {seed}")
            continue

        run_dir = sorted(matching_dirs)[-1]
        pareto_file = run_dir / "pareto-front-solutions.csv"

        if not pareto_file.exists():
            print(f"  [WARN] {pareto_file} not found")
            continue

        df = pd.read_csv(pareto_file)
        df['source_seed'] = seed
        all_solutions.append(df)
        print(f"  [OK] Seed {seed}: {len(df)} solutions")

    print(f"\nTotal runs collected: {len(all_solutions)}")

    all_df = pd.concat(all_solutions, ignore_index=True)
    print(f"Total solutions before filtering: {len(all_df)}")

    objectives = all_df[['npc_cad', 'lpsp', 'co2_kg', 'gini']].values

    print("\nFiltering by Pareto dominance...")
    is_dominated = np.zeros(len(objectives), dtype=bool)

    for i in range(len(objectives)):
        if i % 200 == 0:
            print(f"  Checking solution {i}/{len(objectives)}...")

        for j in range(len(objectives)):
            if i == j:
                continue

            dominates = all(objectives[j] <= objectives[i]) and any(objectives[j] < objectives[i])

            if dominates:
                is_dominated[i] = True
                break

    reference_front = all_df[~is_dominated].copy()
    print(f"\nReference front size: {len(reference_front)} non-dominated solutions")
    print(f"Filtered out: {len(all_df) - len(reference_front)} dominated solutions")

    return reference_front, all_df

def plot_pareto_2d_3panels(df: pd.DataFrame, output_dir: Path, timestamp: str):

    print("\n" + "=" * 80)
    print("GENERATING FIG 2: 2D PARETO PROJECTIONS (3 panels)")
    print("=" * 80)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    ax = axes[0]
    x = df['lpsp'] * 100
    y = df['gini']

    sorted_idx = np.argsort(x)
    x_sorted = x.iloc[sorted_idx].values
    y_sorted = y.iloc[sorted_idx].values

    ax.scatter(x_sorted, y_sorted, s=25, c='steelblue', alpha=0.6, edgecolors='darkblue', linewidths=0.5)
    ax.set_xlabel('LPSP (%)')
    ax.set_ylabel('Gini Coefficient')
    ax.set_title('(a) Reliability vs Equity')
    ax.grid(True, alpha=0.3)

    corr = df['lpsp'].corr(df['gini'])
    ax.text(0.95, 0.95, f'r = {corr:+.2f}', transform=ax.transAxes,
            ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax = axes[1]
    x = df['npc_cad'] / 1e6
    y = df['gini']

    sorted_idx = np.argsort(x)
    x_sorted = x.iloc[sorted_idx].values
    y_sorted = y.iloc[sorted_idx].values

    ax.scatter(x_sorted, y_sorted, s=25, c='forestgreen', alpha=0.6, edgecolors='darkgreen', linewidths=0.5)
    ax.set_xlabel('NPC (M CAD)')
    ax.set_ylabel('Gini Coefficient')
    ax.set_title('(b) Cost vs Equity')
    ax.grid(True, alpha=0.3)

    corr = df['npc_cad'].corr(df['gini'])
    ax.text(0.95, 0.95, f'r = {corr:+.2f}', transform=ax.transAxes,
            ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax = axes[2]
    x = df['co2_kg'] / 1e3
    y = df['gini']

    sorted_idx = np.argsort(x)
    x_sorted = x.iloc[sorted_idx].values
    y_sorted = y.iloc[sorted_idx].values

    ax.scatter(x_sorted, y_sorted, s=25, c='coral', alpha=0.6, edgecolors='darkred', linewidths=0.5)
    ax.set_xlabel('CO$_2$ Emissions (kt/year)')
    ax.set_ylabel('Gini Coefficient')
    ax.set_title('(c) Emissions vs Equity')
    ax.grid(True, alpha=0.3)

    corr = df['co2_kg'].corr(df['gini'])
    ax.text(0.95, 0.95, f'r = {corr:+.2f}', transform=ax.transAxes,
            ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    for fmt in ['png', 'pdf', 'svg']:
        output_file = output_dir / f"fig2-pareto-fronts-2d-3panels-{timestamp}.{fmt}"
        plt.savefig(output_file, bbox_inches='tight', dpi=300)

    print(f"  Saved: fig2-pareto-fronts-2d-3panels-{timestamp}.[png/pdf/svg]")
    plt.close()

def plot_decision_variables(df: pd.DataFrame, output_dir: Path, timestamp: str):

    print("\n" + "=" * 80)
    print("GENERATING FIG 3: DECISION VARIABLE DISTRIBUTIONS")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    variables = [
        ('pv_kw', 'PV Capacity (kW)', 'skyblue', (0, 10000)),
        ('wind_mw', 'Wind Capacity (MW)', 'lightgreen', (0, 5)),
        ('battery_mwh', 'Battery Capacity (MWh)', 'gold', (0, 100)),
        ('diesel_mw', 'Diesel Capacity (MW)', 'salmon', (0, 10))
    ]

    for ax, (col, label, color, bounds) in zip(axes, variables):
        data = df[col]

        ax.hist(data, bins=30, color=color, edgecolor='black', alpha=0.7)
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.1f}')
        ax.axvline(data.median(), color='blue', linestyle=':', linewidth=2, label=f'Median: {data.median():.1f}')

        ax.set_xlabel(label)
        ax.set_ylabel('Frequency')
        ax.set_xlim(bounds)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        stats_text = f'Min: {data.min():.1f}\nMax: {data.max():.1f}\nStd: {data.std():.1f}'
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
                va='top', ha='left', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('Decision Variable Distributions Across Pareto Front', fontsize=12, y=1.02)
    plt.tight_layout()

    for fmt in ['png', 'pdf', 'svg']:
        output_file = output_dir / f"fig3-decision-variables-{timestamp}.{fmt}"
        plt.savefig(output_file, bbox_inches='tight', dpi=300)

    print(f"  Saved: fig3-decision-variables-{timestamp}.[png/pdf/svg]")
    plt.close()

def plot_parallel_coordinates(df: pd.DataFrame, output_dir: Path, timestamp: str):

    print("\n" + "=" * 80)
    print("GENERATING FIG 4: PARALLEL COORDINATES")
    print("=" * 80)

    fig, ax = plt.subplots(figsize=(10, 6))

    objectives = ['npc_cad', 'lpsp', 'co2_kg', 'gini']
    labels = ['NPC\n(M CAD)', 'LPSP\n(%)', 'CO$_2$\n(kt)', 'Gini']

    normalized = pd.DataFrame()
    for obj in objectives:
        normalized[obj] = (df[obj] - df[obj].min()) / (df[obj].max() - df[obj].min() + 1e-10)

    colors = plt.cm.RdYlGn_r(normalized['gini'])

    for idx in range(len(normalized)):
        ax.plot(range(4), normalized.iloc[idx],
               color=colors[idx], alpha=0.4, linewidth=1)

    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Normalized Value [0 = Best, 1 = Worst]')
    ax.set_title('Parallel Coordinates: Multi-Objective Trade-offs\n(Color: Red=High Gini, Green=Low Gini)')
    ax.grid(True, alpha=0.3, axis='y')

    for i, obj in enumerate(objectives):
        vmin, vmax = df[obj].min(), df[obj].max()
        if obj == 'npc_cad':
            ax.annotate(f'{vmin/1e6:.1f}', (i, 0), textcoords='offset points',
                       xytext=(0, -15), ha='center', fontsize=8)
            ax.annotate(f'{vmax/1e6:.1f}', (i, 1), textcoords='offset points',
                       xytext=(0, 10), ha='center', fontsize=8)
        elif obj == 'lpsp':
            ax.annotate(f'{vmin*100:.2f}%', (i, 0), textcoords='offset points',
                       xytext=(0, -15), ha='center', fontsize=8)
            ax.annotate(f'{vmax*100:.2f}%', (i, 1), textcoords='offset points',
                       xytext=(0, 10), ha='center', fontsize=8)
        elif obj == 'co2_kg':
            ax.annotate(f'{vmin/1e3:.0f}', (i, 0), textcoords='offset points',
                       xytext=(0, -15), ha='center', fontsize=8)
            ax.annotate(f'{vmax/1e3:.0f}', (i, 1), textcoords='offset points',
                       xytext=(0, 10), ha='center', fontsize=8)
        else:
            ax.annotate(f'{vmin:.3f}', (i, 0), textcoords='offset points',
                       xytext=(0, -15), ha='center', fontsize=8)
            ax.annotate(f'{vmax:.3f}', (i, 1), textcoords='offset points',
                       xytext=(0, 10), ha='center', fontsize=8)

    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Gini (normalized)', fontsize=10)

    plt.tight_layout()

    for fmt in ['png', 'pdf', 'svg']:
        output_file = output_dir / f"fig4-parallel-coordinates-{timestamp}.{fmt}"
        plt.savefig(output_file, bbox_inches='tight', dpi=300)

    print(f"  Saved: fig4-parallel-coordinates-{timestamp}.[png/pdf/svg]")
    plt.close()

def print_summary_statistics(df: pd.DataFrame):

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS FOR PAPER")
    print("=" * 80)

    print(f"\nTotal Pareto solutions: {len(df)}")

    print("\n### Objectives ###")
    print(f"NPC: {df['npc_cad'].min()/1e6:.1f} - {df['npc_cad'].max()/1e6:.1f} M CAD (mean: {df['npc_cad'].mean()/1e6:.1f})")
    print(f"LPSP: {df['lpsp'].min()*100:.2f} - {df['lpsp'].max()*100:.2f} % (mean: {df['lpsp'].mean()*100:.2f}%)")
    print(f"CO2: {df['co2_kg'].min()/1e3:.1f} - {df['co2_kg'].max()/1e3:.1f} kt (mean: {df['co2_kg'].mean()/1e3:.1f})")
    print(f"Gini: {df['gini'].min():.4f} - {df['gini'].max():.4f} (mean: {df['gini'].mean():.4f})")

    print("\n### Decision Variables ###")
    print(f"PV: {df['pv_kw'].min():.0f} - {df['pv_kw'].max():.0f} kW (mean: {df['pv_kw'].mean():.0f})")
    print(f"Wind: {df['wind_mw'].min():.2f} - {df['wind_mw'].max():.2f} MW (mean: {df['wind_mw'].mean():.2f})")
    print(f"Battery: {df['battery_mwh'].min():.1f} - {df['battery_mwh'].max():.1f} MWh (mean: {df['battery_mwh'].mean():.1f})")
    print(f"Diesel: {df['diesel_mw'].min():.2f} - {df['diesel_mw'].max():.2f} MW (mean: {df['diesel_mw'].mean():.2f})")

    print("\n### Additional Metrics ###")
    print(f"RE%: {df['re_penetration_pct'].min():.1f} - {df['re_penetration_pct'].max():.1f}% (mean: {df['re_penetration_pct'].mean():.1f}%)")
    print(f"LCOE: {df['lcoe_cad_per_kwh'].min():.3f} - {df['lcoe_cad_per_kwh'].max():.3f} CAD/kWh")

    print("\n### Correlations ###")
    print(f"LPSP-Gini: r = {df['lpsp'].corr(df['gini']):+.3f}")
    print(f"NPC-Gini: r = {df['npc_cad'].corr(df['gini']):+.3f}")
    print(f"CO2-Gini: r = {df['co2_kg'].corr(df['gini']):+.3f}")
    print(f"RE%-Gini: r = {df['re_penetration_pct'].corr(df['gini']):+.3f}")

def main():
    print("=" * 80)
    print("GENERATE PAPER FIGURES V9")
    print("SysCon 2026 - 30 Runs Aggregated")
    print("=" * 80)

    project_root = Path(__file__).parent
    results_dir = project_root / "results"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "paper-figures" / f"v9-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    reference_front, all_solutions = aggregate_30_runs(results_dir)

    ref_front_file = output_dir / "reference-front-30runs-v9.csv"
    reference_front.to_csv(ref_front_file, index=False)
    print(f"\nReference front saved: {ref_front_file}")

    print_summary_statistics(reference_front)

    plot_pareto_2d_3panels(reference_front, output_dir, timestamp)
    plot_decision_variables(reference_front, output_dir, timestamp)
    plot_parallel_coordinates(reference_front, output_dir, timestamp)

    print("\n" + "=" * 80)
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Files:")
    for f in sorted(output_dir.glob("*")):
        print(f"  - {f.name}")

if __name__ == "__main__":
    main()
