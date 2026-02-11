import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import json

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
    'font.serif': ['Times New Roman']
})

project_root = Path(__file__).parent.parent.parent
results_dir = project_root / 'results' / 'diagnostic-v2'
figures_dir = project_root / 'figures' / 'diagnostic-v2'
figures_dir.mkdir(parents=True, exist_ok=True)

def load_variation_data(variation_name):

    metrics_files = list(results_dir.glob(f'metrics-{variation_name}-*.csv'))
    pareto_files = list(results_dir.glob(f'pareto-{variation_name}-*.csv'))
    summary_files = list(results_dir.glob(f'summary-{variation_name}-*.json'))

    if not metrics_files or not pareto_files or not summary_files:
        print(f"Warning: Missing files for variation '{variation_name}'")
        return None

    metrics_file = sorted(metrics_files)[-1]
    pareto_file = sorted(pareto_files)[-1]
    summary_file = sorted(summary_files)[-1]

    df_metrics = pd.read_csv(metrics_file)
    df_pareto = pd.read_csv(pareto_file)

    with open(summary_file, 'r') as f:
        summary = json.load(f)

    return {
        'variation': variation_name,
        'metrics': df_metrics,
        'pareto': df_pareto,
        'summary': summary
    }

def plot_convergence_comparison(data_all):

    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(1, 3, figure=fig)

    variations = ['baseline', 'two-bus', 'narrow']
    labels = {
        'baseline': 'Baseline (Urban 100k m²)',
        'two-bus': 'Two-Bus (Remote Wind 2 km²)',
        'narrow': 'Narrow Bounds'
    }
    colors = {'baseline': '#1f77b4', 'two-bus': '#ff7f0e', 'narrow': '#2ca02c'}

    ax1 = fig.add_subplot(gs[0, 0])
    for var in variations:
        if var in data_all and data_all[var] is not None:
            df = data_all[var]['metrics']
            ax1.plot(df['generation'], df['hypervolume'], label=labels[var],
                    color=colors[var], linewidth=1.5)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Hypervolume')
    ax1.set_title('(a) Hypervolume Convergence')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    for var in variations:
        if var in data_all and data_all[var] is not None:
            df = data_all[var]['metrics']
            ax2.plot(df['generation'], df['spacing'], label=labels[var],
                    color=colors[var], linewidth=1.5)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Spacing (lower is better)')
    ax2.set_title('(b) Spacing Metric')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    for var in variations:
        if var in data_all and data_all[var] is not None:
            df = data_all[var]['metrics']
            ax3.plot(df['generation'], df['diversity'], label=labels[var],
                    color=colors[var], linewidth=1.5)
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Diversity')
    ax3.set_title('(c) Diversity Metric')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    filename = figures_dir / 'convergence-comparison.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename.name}")
    plt.close()

def plot_objective_ranges_boxplots(data_all):

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    variations = ['baseline', 'two-bus', 'narrow']
    labels_short = {'baseline': 'Baseline', 'two-bus': 'Two-Bus', 'narrow': 'Narrow'}

    objectives = [
        ('NPC', 'NPC ($M)', 1e6),
        ('LPSP', 'LPSP', 1.0),
        ('CO2', 'CO2 (kt)', 1e3),
        ('Gini', 'Gini Coefficient', 1.0)
    ]

    for idx, (obj_col, obj_label, scale) in enumerate(objectives):
        ax = axes[idx // 2, idx % 2]

        data_to_plot = []
        positions = []
        tick_labels = []

        for i, var in enumerate(variations):
            if var in data_all and data_all[var] is not None:
                values = data_all[var]['pareto'][obj_col].values / scale
                data_to_plot.append(values)
                positions.append(i + 1)
                tick_labels.append(labels_short[var])

        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                       patch_artist=True, showfliers=True)

        colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for patch, color in zip(bp['boxes'], colors_list[:len(data_to_plot)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, rotation=15, ha='right')
        ax.set_ylabel(obj_label)
        ax.set_title(f'({chr(97+idx)}) {obj_label} Range')
        ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    filename = figures_dir / 'objective-ranges-boxplots.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename.name}")
    plt.close()

def plot_wind_utilization_comparison(data_all):

    fig, ax = plt.subplots(figsize=(8, 5))

    variations = ['baseline', 'two-bus', 'narrow']
    labels = {
        'baseline': 'Baseline',
        'two-bus': 'Two-Bus',
        'narrow': 'Narrow'
    }
    colors = {'baseline': '#1f77b4', 'two-bus': '#ff7f0e', 'narrow': '#2ca02c'}

    bins = np.linspace(0, 5, 26)

    for var in variations:
        if var in data_all and data_all[var] is not None:
            wind_values = data_all[var]['pareto']['n_wind_mw'].values
            ax.hist(wind_values, bins=bins, alpha=0.5, label=labels[var],
                   color=colors[var], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Wind Capacity (MW)')
    ax.set_ylabel('Number of Solutions')
    ax.set_title('Wind Utilization Distribution Across Variations')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    textstr = '\n'.join([
        f"{labels[var]}: {data_all[var]['summary']['diagnostic_insights']['wind_utilization_pct']:.1f}% > 0.1 MW"
        for var in variations if var in data_all and data_all[var] is not None
    ])
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    filename = figures_dir / 'wind-utilization-comparison.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename.name}")
    plt.close()

def plot_npc_co2_correlation(data_all):

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    variations = ['baseline', 'two-bus', 'narrow']
    labels = {
        'baseline': 'Baseline',
        'two-bus': 'Two-Bus',
        'narrow': 'Narrow'
    }
    colors = {'baseline': '#1f77b4', 'two-bus': '#ff7f0e', 'narrow': '#2ca02c'}

    for idx, var in enumerate(variations):
        if var in data_all and data_all[var] is not None:
            df = data_all[var]['pareto']
            npc = df['NPC'].values / 1e6
            co2 = df['CO2'].values / 1e3

            corr = data_all[var]['summary']['diagnostic_insights']['npc_co2_correlation']

            axes[idx].scatter(npc, co2, alpha=0.6, s=50, color=colors[var],
                            edgecolors='black', linewidth=0.5)
            axes[idx].set_xlabel('NPC ($M)')
            if idx == 0:
                axes[idx].set_ylabel('CO2 (kt)')
            axes[idx].set_title(f'{labels[var]}\nr = {corr:.3f}')
            axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()

    filename = figures_dir / 'npc-co2-correlation.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename.name}")
    plt.close()

def plot_pareto_2d_comparison(data_all):

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    variations = ['baseline', 'two-bus', 'narrow']
    labels = {
        'baseline': 'Baseline',
        'two-bus': 'Two-Bus',
        'narrow': 'Narrow'
    }
    colors = {'baseline': '#1f77b4', 'two-bus': '#ff7f0e', 'narrow': '#2ca02c'}
    markers = {'baseline': 'o', 'two-bus': 's', 'narrow': '^'}

    plots = [
        ('NPC', 'LPSP', 'NPC ($M)', 'LPSP', 1e6, 1.0, '(a)'),
        ('NPC', 'CO2', 'NPC ($M)', 'CO2 (kt)', 1e6, 1e3, '(b)'),
        ('LPSP', 'CO2', 'LPSP', 'CO2 (kt)', 1.0, 1e3, '(c)'),
        ('CO2', 'Gini', 'CO2 (kt)', 'Gini', 1e3, 1.0, '(d)')
    ]

    for idx, (x_col, y_col, x_label, y_label, x_scale, y_scale, subplot_label) in enumerate(plots):
        ax = axes[idx // 2, idx % 2]

        for var in variations:
            if var in data_all and data_all[var] is not None:
                df = data_all[var]['pareto']
                x = df[x_col].values / x_scale
                y = df[y_col].values / y_scale

                ax.scatter(x, y, label=labels[var], alpha=0.6, s=40,
                          color=colors[var], marker=markers[var],
                          edgecolors='black', linewidth=0.5)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f'{subplot_label} {x_label} vs {y_label}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    filename = figures_dir / 'pareto-2d-comparison.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename.name}")
    plt.close()

def generate_diagnostic_report(data_all):

    report_lines = [
        "# Diagnostic Test V2 - Results Report",
        "",
        "**Generated**: Auto from diagnostic-plots-v2.py",
        "**Variations**: Baseline, Two-Bus, Narrow",
        "",
        "---",
        "",
        "## Summary Statistics",
        ""
    ]

    variations = ['baseline', 'two-bus', 'narrow']

    for var in variations:
        if var in data_all and data_all[var] is not None:
            summ = data_all[var]['summary']

            report_lines.extend([
                f"### {var.upper()}",
                "",
                f"**Configuration**: {summ['configuration']['variation_description']}",
                f"**Scientific Basis**: {summ['configuration']['scientific_basis']}",
                "",
                "**Convergence**:",
                f"- HV improvement: {summ['convergence_metrics']['hv_improvement_pct']:+.2f}%",
                f"- Final HV: {summ['convergence_metrics']['hypervolume_final']:.4e}",
                f"- Final Spacing: {summ['convergence_metrics']['final_spacing']:.4e}",
                "",
                "**Objective Ranges**:",
                f"- NPC: ${summ['objectives_ranges']['NPC']['min']:.2e} - ${summ['objectives_ranges']['NPC']['max']:.2e} ({summ['objectives_ranges']['NPC']['range_pct']:.1f}% range)",
                f"- LPSP: {summ['objectives_ranges']['LPSP']['min']:.4f} - {summ['objectives_ranges']['LPSP']['max']:.4f}",
                f"- CO2: {summ['objectives_ranges']['CO2']['min']:.2e} - {summ['objectives_ranges']['CO2']['max']:.2e} ({summ['objectives_ranges']['CO2']['range_pct']:.1f}% range)",
                f"- Gini: {summ['objectives_ranges']['Gini']['min']:.4f} - {summ['objectives_ranges']['Gini']['max']:.4f} ({summ['objectives_ranges']['Gini']['range_pct']:.1f}% range)",
                "",
                "**Diagnostic Insights**:",
                f"- Wind utilization: {summ['diagnostic_insights']['wind_utilization_pct']:.1f}% > 0.1 MW",
                f"- Max wind: {summ['diagnostic_insights']['max_wind_mw']:.3f} MW",
                f"- NPC-CO2 correlation: {summ['diagnostic_insights']['npc_co2_correlation']:.4f}",
                "",
                "---",
                ""
            ])

    report_lines.extend([
        "## Hypothesis Testing",
        "",
        "### H1: HV Drop Due to Initial Population",
        "",
        "Comparing HV_initial across variations:",
        ""
    ])

    for var in variations:
        if var in data_all and data_all[var] is not None:
            hv_init = data_all[var]['summary']['convergence_metrics']['hypervolume_initial']
            report_lines.append(f"- {var}: HV_0 = {hv_init:.4e}")

    report_lines.extend([
        "",
        "### H2: NPC-CO2 Strong Coupling",
        "",
        "Correlation coefficients:",
        ""
    ])

    for var in variations:
        if var in data_all and data_all[var] is not None:
            corr = data_all[var]['summary']['diagnostic_insights']['npc_co2_correlation']
            report_lines.append(f"- {var}: r = {corr:.4f}")

    report_lines.extend([
        "",
        "### H3: Wind Underutilization Due to Area Constraint",
        "",
        "Wind utilization rates:",
        ""
    ])

    for var in variations:
        if var in data_all and data_all[var] is not None:
            wind_util = data_all[var]['summary']['diagnostic_insights']['wind_utilization_pct']
            wind_max = data_all[var]['summary']['diagnostic_insights']['max_wind_mw']
            report_lines.append(f"- {var}: {wind_util:.1f}% (max {wind_max:.3f} MW)")

    report_lines.extend([
        "",
        "---",
        "",
        "## Figures Generated",
        "",
        "1. `convergence-comparison.pdf` - HV, Spacing, Diversity",
        "2. `objective-ranges-boxplots.pdf` - NPC, LPSP, CO2, Gini distributions",
        "3. `wind-utilization-comparison.pdf` - Histogram of wind capacities",
        "4. `npc-co2-correlation.pdf` - Scatter plots with correlation",
        "5. `pareto-2d-comparison.pdf` - Key trade-off visualizations",
        ""
    ])

    report_file = project_root / 'results' / 'diagnostic-v2' / 'DIAGNOSTIC-REPORT-V2-AUTO.md'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"\nSaved: {report_file.name}")

if __name__ == "__main__":
    print("=" * 80)
    print("DIAGNOSTIC PLOTS V2 - COMPARATIVE VISUALIZATION")
    print("=" * 80)
    print()

    variations = ['baseline', 'two-bus', 'narrow']

    print("Loading data...")
    data_all = {}
    for var in variations:
        print(f"  Loading {var}...")
        data = load_variation_data(var)
        if data:
            data_all[var] = data
            print(f"    ✓ {len(data['pareto'])} Pareto solutions, {len(data['metrics'])} generations")
        else:
            print(f"    ✗ Data not found")

    print()

    if len(data_all) == 0:
        print("ERROR: No variation data found!")
        print(f"Expected files in: {results_dir}")
        exit(1)

    print(f"Generating plots ({len(data_all)}/{len(variations)} variations loaded)...")
    print()

    plot_convergence_comparison(data_all)
    plot_objective_ranges_boxplots(data_all)
    plot_wind_utilization_comparison(data_all)
    plot_npc_co2_correlation(data_all)
    plot_pareto_2d_comparison(data_all)

    print()
    print("Generating diagnostic report...")
    generate_diagnostic_report(data_all)

    print()
    print("=" * 80)
    print("ALL PLOTS GENERATED")
    print("=" * 80)
    print(f"Output directory: {figures_dir}")
    print()
