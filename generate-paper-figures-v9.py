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
    all_solutions = []
    for seed in range(42, 72):
        pattern = f"v8-run*-seed{seed}-20260210_*"
        matching_dirs = list(results_dir.glob(pattern))
        if not matching_dirs:
            continue
        run_dir = sorted(matching_dirs)[-1]
        pareto_file = run_dir / "pareto-front-solutions.csv"
        if not pareto_file.exists():
            continue
        df = pd.read_csv(pareto_file)
        df['source_seed'] = seed
        all_solutions.append(df)

    all_df = pd.concat(all_solutions, ignore_index=True)
    objectives = all_df[['npc_cad', 'lpsp', 'co2_kg', 'gini']].values
    is_dominated = np.zeros(len(objectives), dtype=bool)

    for i in range(len(objectives)):
        for j in range(len(objectives)):
            if i == j:
                continue
            dominates = all(objectives[j] <= objectives[i]) and any(objectives[j] < objectives[i])
            if dominates:
                is_dominated[i] = True
                break

    reference_front = all_df[~is_dominated].copy()
    return reference_front, all_df

def plot_pareto_2d_3panels(df: pd.DataFrame, output_dir: Path, timestamp: str):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    ax = axes[0]
    x = df['lpsp'] * 100
    y = df['gini']
    sorted_idx = np.argsort(x)
    ax.scatter(x.iloc[sorted_idx], y.iloc[sorted_idx], s=25, c='steelblue', alpha=0.6, edgecolors='darkblue', linewidths=0.5)
    ax.set_xlabel('LPSP (%)')
    ax.set_ylabel('Gini Coefficient')
    ax.set_title('(a) Reliability vs Equity')
    ax.grid(True, alpha=0.3)
    corr = df['lpsp'].corr(df['gini'])
    ax.text(0.95, 0.95, f'r = {corr:+.2f}', transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax = axes[1]
    x = df['npc_cad'] / 1e6
    y = df['gini']
    sorted_idx = np.argsort(x)
    ax.scatter(x.iloc[sorted_idx], y.iloc[sorted_idx], s=25, c='forestgreen', alpha=0.6, edgecolors='darkgreen', linewidths=0.5)
    ax.set_xlabel('NPC (M CAD)')
    ax.set_ylabel('Gini Coefficient')
    ax.set_title('(b) Cost vs Equity')
    ax.grid(True, alpha=0.3)
    corr = df['npc_cad'].corr(df['gini'])
    ax.text(0.95, 0.95, f'r = {corr:+.2f}', transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax = axes[2]
    x = df['co2_kg'] / 1e3
    y = df['gini']
    sorted_idx = np.argsort(x)
    ax.scatter(x.iloc[sorted_idx], y.iloc[sorted_idx], s=25, c='coral', alpha=0.6, edgecolors='darkred', linewidths=0.5)
    ax.set_xlabel('CO$_2$ Emissions (kt/year)')
    ax.set_ylabel('Gini Coefficient')
    ax.set_title('(c) Emissions vs Equity')
    ax.grid(True, alpha=0.3)
    corr = df['co2_kg'].corr(df['gini'])
    ax.text(0.95, 0.95, f'r = {corr:+.2f}', transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    for fmt in ['png', 'pdf', 'svg']:
        plt.savefig(output_dir / f"fig2-pareto-fronts-2d-3panels-{timestamp}.{fmt}", bbox_inches='tight', dpi=300)
    plt.close()

def plot_decision_variables(df: pd.DataFrame, output_dir: Path, timestamp: str):
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
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, va='top', ha='left', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('Decision Variable Distributions Across Pareto Front', fontsize=12, y=1.02)
    plt.tight_layout()
    for fmt in ['png', 'pdf', 'svg']:
        plt.savefig(output_dir / f"fig3-decision-variables-{timestamp}.{fmt}", bbox_inches='tight', dpi=300)
    plt.close()

def plot_parallel_coordinates(df: pd.DataFrame, output_dir: Path, timestamp: str):
    fig, ax = plt.subplots(figsize=(10, 6))

    objectives = ['npc_cad', 'lpsp', 'co2_kg', 'gini']
    labels = ['NPC\n(M CAD)', 'LPSP\n(%)', 'CO$_2$\n(kt)', 'Gini']

    normalized = pd.DataFrame()
    for obj in objectives:
        normalized[obj] = (df[obj] - df[obj].min()) / (df[obj].max() - df[obj].min() + 1e-10)

    colors = plt.cm.RdYlGn_r(normalized['gini'])
    for idx in range(len(normalized)):
        ax.plot(range(4), normalized.iloc[idx], color=colors[idx], alpha=0.4, linewidth=1)

    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Normalized Value [0 = Best, 1 = Worst]')
    ax.set_title('Parallel Coordinates: Multi-Objective Trade-offs')
    ax.grid(True, alpha=0.3, axis='y')

    for i, obj in enumerate(objectives):
        vmin, vmax = df[obj].min(), df[obj].max()
        if obj == 'npc_cad':
            ax.annotate(f'{vmin/1e6:.1f}', (i, 0), textcoords='offset points', xytext=(0, -15), ha='center', fontsize=8)
            ax.annotate(f'{vmax/1e6:.1f}', (i, 1), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=8)
        elif obj == 'lpsp':
            ax.annotate(f'{vmin*100:.2f}%', (i, 0), textcoords='offset points', xytext=(0, -15), ha='center', fontsize=8)
            ax.annotate(f'{vmax*100:.2f}%', (i, 1), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=8)
        elif obj == 'co2_kg':
            ax.annotate(f'{vmin/1e3:.0f}', (i, 0), textcoords='offset points', xytext=(0, -15), ha='center', fontsize=8)
            ax.annotate(f'{vmax/1e3:.0f}', (i, 1), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=8)
        else:
            ax.annotate(f'{vmin:.3f}', (i, 0), textcoords='offset points', xytext=(0, -15), ha='center', fontsize=8)
            ax.annotate(f'{vmax:.3f}', (i, 1), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=8)

    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Gini (normalized)', fontsize=10)

    plt.tight_layout()
    for fmt in ['png', 'pdf', 'svg']:
        plt.savefig(output_dir / f"fig4-parallel-coordinates-{timestamp}.{fmt}", bbox_inches='tight', dpi=300)
    plt.close()

def main():
    project_root = Path(__file__).parent
    results_dir = project_root / "results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "paper-figures" / f"v9-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_front, _ = aggregate_30_runs(results_dir)
    reference_front.to_csv(output_dir / "reference-front-30runs-v9.csv", index=False)

    plot_pareto_2d_3panels(reference_front, output_dir, timestamp)
    plot_decision_variables(reference_front, output_dir, timestamp)
    plot_parallel_coordinates(reference_front, output_dir, timestamp)

    print(f"Figures saved to {output_dir}")

if __name__ == "__main__":
    main()
