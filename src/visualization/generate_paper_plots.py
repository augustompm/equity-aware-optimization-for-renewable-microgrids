import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12

class PaperPlotsGenerator:

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.data_dir = project_root / "data"
        self.results_dir = project_root / "results"
        self.figures_dir = project_root / "figures"
        self.figures_dir.mkdir(exist_ok=True)

        self.objectives = {
            'NPC': {'name': 'Net Present Cost', 'unit': '$', 'scale': 1e6, 'label': 'NPC (M$)'},
            'LPSP': {'name': 'Loss of Power Supply Probability', 'unit': '%', 'scale': 100, 'label': 'LPSP (%)'},
            'CO2': {'name': 'CO₂ Emissions', 'unit': 'kg', 'scale': 1e6, 'label': 'CO₂ (Mt)'},
            'Gini': {'name': 'Gini Coefficient', 'unit': '', 'scale': 1, 'label': 'Gini Coefficient'}
        }

    def plot_input_data(self, save_format: str = 'pdf'):

        print("\n[Figure 5] Generating input data plots...")

        wind_cf = pd.read_csv(self.data_dir / "wind-capacity-factors.csv")
        solar_cf = pd.read_csv(self.data_dir / "solar-capacity-factors.csv")
        meteo = pd.read_csv(self.data_dir / "meteorology-8760h.csv")
        load = pd.read_csv(self.data_dir / "load-profile-8760h.csv")

        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        hours = np.arange(len(wind_cf))

        axes[0].plot(hours, wind_cf.iloc[:, 0], linewidth=0.5, color='steelblue', alpha=0.7)
        axes[0].set_ylabel('Wind CF')
        axes[0].set_title('(a) Wind Capacity Factor')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(hours, solar_cf.iloc[:, 0], linewidth=0.5, color='orange', alpha=0.7)
        axes[1].set_ylabel('Solar CF')
        axes[1].set_title('(b) Solar Capacity Factor')
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(hours, meteo['T_ambient_C'], linewidth=0.5, color='red', alpha=0.7)
        axes[2].set_ylabel('Temperature (°C)')
        axes[2].set_title('(c) Ambient Temperature')
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(hours, load.iloc[:, 0], linewidth=0.5, color='green', alpha=0.7)
        axes[3].set_ylabel('Load (kW)')
        axes[3].set_xlabel('Hour of Year')
        axes[3].set_title('(d) Electrical Load Demand')
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = self.figures_dir / f"fig5_input_data.{save_format}"
        plt.savefig(output_file, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()

    def plot_convergence_metrics(self, metrics_file: Path, save_format: str = 'pdf'):

        print("\n[Convergence] Generating convergence plots...")

        df = pd.read_csv(metrics_file)

        fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

        axes[0].plot(df['generation'], df['hypervolume'], linewidth=2, color='blue', marker='o', markersize=3)
        axes[0].set_ylabel('Hypervolume')
        axes[0].set_title('(a) Hypervolume Indicator (HV)')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(df['generation'], df['spacing'], linewidth=2, color='red', marker='s', markersize=3)
        axes[1].set_ylabel('Spacing (SP)')
        axes[1].set_title('(b) Spacing Metric')
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(df['generation'], df['diversity'], linewidth=2, color='green', marker='^', markersize=3)
        axes[2].set_ylabel('Diversity (ID)')
        axes[2].set_xlabel('Generation')
        axes[2].set_title('(c) Diversity Indicator')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = self.figures_dir / f"convergence_metrics.{save_format}"
        plt.savefig(output_file, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()

    def plot_pareto_2d(self, pareto_file: Path, save_format: str = 'pdf'):

        print("\n[Figure 7] Generating 2D Pareto front plots...")

        df = pd.read_csv(pareto_file)

        combinations = [
            ('NPC', 'LPSP'),
            ('NPC', 'CO2'),
            ('NPC', 'Gini'),
            ('LPSP', 'CO2'),
            ('LPSP', 'Gini'),
            ('CO2', 'Gini')
        ]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, (obj1, obj2) in enumerate(combinations):
            ax = axes[idx]

            x = df[obj1] / self.objectives[obj1]['scale']
            y = df[obj2] / self.objectives[obj2]['scale']

            sorted_idx = np.argsort(x)
            x_sorted = x.iloc[sorted_idx]
            y_sorted = y.iloc[sorted_idx]

            ax.plot(x_sorted, y_sorted, 'o-', linewidth=2, markersize=6,
                   color='steelblue', markerfacecolor='lightblue', markeredgewidth=1.5)

            ax.set_xlabel(self.objectives[obj1]['label'])
            ax.set_ylabel(self.objectives[obj2]['label'])
            ax.set_title(f"({chr(97+idx)}) {obj1} vs {obj2}")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = self.figures_dir / f"fig7_pareto_2d.{save_format}"
        plt.savefig(output_file, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()

    def plot_boxplots(self, metrics_files: List[Path], save_format: str = 'pdf'):

        if len(metrics_files) < 2:
            print("\n[Figure 6] Skipping boxplots (requires ≥2 runs, have {})".format(len(metrics_files)))
            return

        print("\n[Figure 6] Generating boxplot comparison...")

        sp_values = []
        hv_values = []
        id_values = []

        for mfile in metrics_files:
            df = pd.read_csv(mfile)
            final = df.iloc[-1]
            sp_values.append(final['spacing'])
            hv_values.append(final['hypervolume'])
            id_values.append(final['diversity'])

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].boxplot([sp_values], labels=['NSGA-III'])
        axes[0].set_ylabel('Spacing (SP)')
        axes[0].set_title('(a) Spacing Metric')
        axes[0].grid(True, alpha=0.3, axis='y')

        axes[1].boxplot([hv_values], labels=['NSGA-III'])
        axes[1].set_ylabel('Hypervolume (HV)')
        axes[1].set_title('(b) Hypervolume Indicator')
        axes[1].grid(True, alpha=0.3, axis='y')

        axes[2].boxplot([id_values], labels=['NSGA-III'])
        axes[2].set_ylabel('Diversity (ID)')
        axes[2].set_title('(c) Diversity Indicator')
        axes[2].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_file = self.figures_dir / f"fig6_boxplots.{save_format}"
        plt.savefig(output_file, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()

    def plot_parallel_coordinates(self, pareto_file: Path, save_format: str = 'pdf'):

        print("\n[Parallel Coordinates] Generating plot...")

        df = pd.read_csv(pareto_file)

        normalized = pd.DataFrame()
        for obj in ['NPC', 'LPSP', 'CO2', 'Gini']:
            normalized[obj] = (df[obj] - df[obj].min()) / (df[obj].max() - df[obj].min())

        fig, ax = plt.subplots(figsize=(10, 6))

        for idx in range(len(normalized)):
            ax.plot(range(4), normalized.iloc[idx],
                   color='steelblue', alpha=0.5, linewidth=1.5)

        ax.set_xticks(range(4))
        ax.set_xticklabels(['NPC', 'LPSP', 'CO₂', 'Gini'])
        ax.set_ylabel('Normalized Value [0, 1]')
        ax.set_title('Parallel Coordinates: Multi-Objective Trade-offs')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_file = self.figures_dir / f"parallel_coordinates.{save_format}"
        plt.savefig(output_file, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()

    def generate_latex_table(self, metrics_files: List[Path], pareto_files: List[Path]) -> str:

        print("\n[Table 3] Generating statistical performance table...")

        if len(metrics_files) < 2:
            print("  Note: Single run, reporting final values only")

            df_metrics = pd.read_csv(metrics_files[0])
            final = df_metrics.iloc[-1]

            latex = r"""
\begin{table}[ht]
\centering
\caption{Performance Metrics (Single Run)}
\begin{tabular}{lccc}
\hline
Metric & Value & Unit & Status \\
\hline
Hypervolume (HV) & """ + f"{final['hypervolume']:.2e}" + r""" & - & Final \\
Spacing (SP) & """ + f"{final['spacing']:.2e}" + r""" & - & Final \\
Diversity (ID) & """ + f"{final['diversity']:.2e}" + r""" & - & Final \\
Pareto Size & """ + f"{int(final['n_pareto'])}" + r""" & solutions & Final \\
Feasibility & """ + f"{final['n_feasible']/final['n_pareto']*100:.1f}" + r"""\% & - & Final \\
\hline
\end{tabular}
\label{tab:performance_single}
\end{table}
\begin{table}[ht]
\centering
\caption{Statistical Performance Indices (""" + f"{len(metrics_files)} runs" + r""")}
\begin{tabular}{lcccccc}
\hline
Metric & Min & Median & Max & Mean & Std & Algorithm \\
\hline
SP & """ + f"{np.min(sp_vals):.2e}" + r""" & """ + f"{np.median(sp_vals):.2e}" + r""" & """ + f"{np.max(sp_vals):.2e}" + r""" & """ + f"{np.mean(sp_vals):.2e}" + r""" & """ + f"{np.std(sp_vals):.2e}" + r""" & NSGA-III \\
HV & """ + f"{np.min(hv_vals):.2e}" + r""" & """ + f"{np.median(hv_vals):.2e}" + r""" & """ + f"{np.max(hv_vals):.2e}" + r""" & """ + f"{np.mean(hv_vals):.2e}" + r""" & """ + f"{np.std(hv_vals):.2e}" + r""" & NSGA-III \\
ID & """ + f"{np.min(id_vals):.2e}" + r""" & """ + f"{np.median(id_vals):.2e}" + r""" & """ + f"{np.max(id_vals):.2e}" + r""" & """ + f"{np.mean(id_vals):.2e}" + r""" & """ + f"{np.std(id_vals):.2e}" + r""" & NSGA-III \\
\hline
\end{tabular}
\label{tab:performance_stats}
\end{table}
\begin{table}[ht]
\centering
\caption{Representative Solutions from Pareto Front}
\begin{tabular}{lcccccccc}
\hline
Solution & PV (kW) & Wind (MW) & Battery (MWh) & Diesel (MW) & NPC (M\$) & LPSP (\%) & CO$_2$ (Mt) & Gini \\
\hline
\end{tabular}
\label{tab:representative_solutions}
\end{table}
