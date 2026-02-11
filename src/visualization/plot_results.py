import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations

def plot_convergence(
    metrics_history,
    output_dir,
    formats=['png', 'pdf', 'svg'],
    dpi=300
):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(metrics_history)
    df.to_csv(output_dir / 'convergence-metrics.csv', index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Convergence Metrics', fontsize=14, fontweight='bold')

    axes[0, 0].plot(df['generation'], df['hypervolume'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Hypervolume')
    axes[0, 0].set_title('Hypervolume (HV)')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(df['generation'], df['igd_plus'], 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Generation')
    axes[0, 1].set_ylabel('IGD+')
    axes[0, 1].set_title('Inverted Generational Distance Plus (IGD+)')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(df['generation'], df['spacing'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Generation')
    axes[1, 0].set_ylabel('Spacing')
    axes[1, 0].set_title('Spacing (SP)')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(df['generation'], df['diversity'], 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Generation')
    axes[1, 1].set_ylabel('Diversity')
    axes[1, 1].set_title('Diversity (DIV)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    for fmt in formats:
        plt.savefig(output_dir / f'convergence-all.{fmt}', dpi=dpi if fmt == 'png' else None)

    plt.close()

def plot_pareto_fronts_2d(
    pareto_data,
    output_dir,
    formats=['png', 'pdf', 'svg'],
    dpi=300
):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(pareto_data, list):
        df = pd.DataFrame(pareto_data)
    else:
        df = pareto_data.copy()

    df.to_csv(output_dir / 'pareto-front-solutions.csv', index=False)

    objectives = [
        ('npc_cad', 'NPC (CAD$)', 'M CAD', 1e6),
        ('lpsp', 'LPSP', '%', 100),
        ('co2_kg', 'CO2 (kg/yr)', 'ton/yr', 1e3),
        ('gini', 'Gini', '', 1)
    ]

    for (col1, label1, unit1, scale1), (col2, label2, unit2, scale2) in combinations(objectives, 2):
        fig, ax = plt.subplots(figsize=(8, 6))

        x = df[col1] / scale1
        y = df[col2] / scale2

        scatter = ax.scatter(x, y, c=range(len(df)), cmap='viridis', s=50, alpha=0.7, edgecolors='k', linewidth=0.5)

        ax.set_xlabel(f'{label1} ({unit1})' if unit1 else label1, fontsize=12)
        ax.set_ylabel(f'{label2} ({unit2})' if unit2 else label2, fontsize=12)
        ax.set_title(f'Pareto Front: {label1} vs {label2}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.colorbar(scatter, ax=ax, label='Solution Index')
        plt.tight_layout()

        safe_name1 = col1.replace('_', '-')
        safe_name2 = col2.replace('_', '-')

        for fmt in formats:
            plt.savefig(output_dir / f'pareto-2d-{safe_name1}-{safe_name2}.{fmt}', dpi=dpi if fmt == 'png' else None)

        plt.close()

def plot_parallel_coordinates(
    pareto_data,
    output_dir,
    formats=['png', 'pdf', 'svg'],
    dpi=300
):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(pareto_data, list):
        df = pd.DataFrame(pareto_data)
    else:
        df = pareto_data.copy()

    obj_cols = ['npc_cad', 'lpsp', 'co2_kg', 'gini']
    df_norm = df[obj_cols].copy()

    for col in obj_cols:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        if max_val > min_val:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        else:
            df_norm[col] = 0.5

    fig, ax = plt.subplots(figsize=(12, 6))

    for idx in range(len(df_norm)):
        ax.plot(range(len(obj_cols)), df_norm.iloc[idx], alpha=0.5, linewidth=1)

    ax.set_xticks(range(len(obj_cols)))
    ax.set_xticklabels(['NPC', 'LPSP', 'CO2', 'Gini'], fontsize=12)
    ax.set_ylabel('Normalized Value [0,1]', fontsize=12)
    ax.set_title('Parallel Coordinates: All Objectives', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    for fmt in formats:
        plt.savefig(output_dir / f'parallel-coordinates.{fmt}', dpi=dpi if fmt == 'png' else None)

    plt.close()

def plot_decision_variables_distribution(
    pareto_data,
    output_dir,
    formats=['png', 'pdf', 'svg'],
    dpi=300
):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(pareto_data, list):
        df = pd.DataFrame(pareto_data)
    else:
        df = pareto_data.copy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Decision Variables Distribution', fontsize=14, fontweight='bold')

    axes[0, 0].boxplot([df['pv_kw']], labels=['PV'])
    axes[0, 0].set_ylabel('Capacity (kW)')
    axes[0, 0].set_title('Solar PV')
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    axes[0, 1].boxplot([df['wind_mw']], labels=['Wind'])
    axes[0, 1].set_ylabel('Capacity (MW)')
    axes[0, 1].set_title('Wind Turbines')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    axes[1, 0].boxplot([df['battery_mwh']], labels=['Battery'])
    axes[1, 0].set_ylabel('Energy (MWh)')
    axes[1, 0].set_title('Battery Storage')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    axes[1, 1].boxplot([df['diesel_mw']], labels=['Diesel'])
    axes[1, 1].set_ylabel('Capacity (MW)')
    axes[1, 1].set_title('Diesel Generator')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    for fmt in formats:
        plt.savefig(output_dir / f'decision-variables-distribution.{fmt}', dpi=dpi if fmt == 'png' else None)

    plt.close()

def create_all_plots(
    metrics_history,
    pareto_data,
    output_dir,
    formats=['png', 'pdf', 'svg'],
    dpi=300
):

    output_dir = Path(output_dir)
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_convergence(metrics_history, figures_dir, formats, dpi)
    plot_pareto_fronts_2d(pareto_data, figures_dir, formats, dpi)
    plot_parallel_coordinates(pareto_data, figures_dir, formats, dpi)
    plot_decision_variables_distribution(pareto_data, figures_dir, formats, dpi)
