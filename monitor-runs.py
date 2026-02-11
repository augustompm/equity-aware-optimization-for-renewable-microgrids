import sys
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

project_root = Path(__file__).parent
results_dir = project_root / "results"

def get_completed_runs():

    if not results_dir.exists():
        return []

    runs = []
    for d in sorted(results_dir.glob("v8-run*-seed*")):
        summary_file = d / "summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                runs.append({
                    'dir': d,
                    'name': d.name,
                    'summary': summary
                })
            except:
                pass
    return runs

def get_running_log():

    logs = sorted(project_root.glob("batch-v8-*.log"), reverse=True)
    if logs:
        return logs[0]
    return None

def parse_log_progress(log_file):

    if not log_file or not log_file.exists():
        return None

    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    current_run = 0
    current_gen = 0
    total_runs = 30
    last_hv = None
    status = "unknown"

    for line in reversed(lines):
        if "STARTING RUN" in line and "/" in line:
            parts = line.split("STARTING RUN")[1].split("/")
            current_run = int(parts[0].strip())
            total_runs = int(parts[1].split()[0])
            status = "running"
            break
        elif "[SUCCESS] Run" in line:
            parts = line.split("Run")[1].split()
            current_run = int(parts[0])
            status = "completed_run"
        elif "Gen" in line and "HV=" in line:
            try:
                gen_part = line.split("Gen")[1].split(":")[0].strip()
                current_gen = int(gen_part)
                hv_part = line.split("HV=")[1].split()[0]
                last_hv = float(hv_part)
            except:
                pass
        elif "BATCH COMPLETE" in line:
            status = "batch_complete"
            break

    return {
        'current_run': current_run,
        'total_runs': total_runs,
        'current_gen': current_gen,
        'last_hv': last_hv,
        'status': status,
        'log_file': log_file
    }

def show_status():

    print("=" * 80)
    print("BATCH OPTIMIZATION MONITOR")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    log_file = get_running_log()
    progress = parse_log_progress(log_file)

    if progress:
        print(f"Log file: {progress['log_file'].name}")
        print(f"Status: {progress['status']}")
        print(f"Current run: {progress['current_run']}/{progress['total_runs']}")
        if progress['current_gen'] > 0:
            print(f"Current generation: {progress['current_gen']}")
        if progress['last_hv']:
            print(f"Last hypervolume: {progress['last_hv']:.6e}")
        print()
    else:
        print("No active batch log found.")
        print()

    runs = get_completed_runs()
    print(f"Completed runs: {len(runs)}")
    print()

    if runs:
        print(f"{'Run':<8} {'Seed':<8} {'Gen':<8} {'Solutions':<12} {'HV':<15} {'Time (min)':<12}")
        print("-" * 70)

        for r in runs[-10:]:
            s = r['summary']
            run_id = s['run_info']['run_id']
            seed = s['run_info']['seed']
            n_gen = s['run_info']['n_gen_actual']
            n_sol = s['pareto_front']['n_solutions']
            hv = s['convergence']['final']['hypervolume']

            time_min = "N/A"

            print(f"{run_id:<8} {seed:<8} {n_gen:<8} {n_sol:<12} {hv:<15.6e} {time_min:<12}")

        if len(runs) > 10:
            print(f"... and {len(runs) - 10} more runs")

    print()
    print("=" * 80)
    print("Commands:")
    print("  python monitor-runs.py --watch    # Auto-refresh")
    print("  python monitor-runs.py --summary  # Preliminary results")
    print("  python monitor-runs.py --pareto   # Pareto analysis")
    print("=" * 80)

def show_summary():

    runs = get_completed_runs()

    if not runs:
        print("No completed runs found.")
        return

    print("=" * 80)
    print(f"PRELIMINARY RESULTS SUMMARY ({len(runs)} runs)")
    print("=" * 80)
    print()

    all_hv = []
    all_n_solutions = []
    all_n_gen = []
    all_gini_min = []
    all_gini_max = []
    all_npc_min = []
    all_npc_max = []
    all_re_min = []
    all_re_max = []

    for r in runs:
        s = r['summary']
        all_hv.append(s['convergence']['final']['hypervolume'])
        all_n_solutions.append(s['pareto_front']['n_solutions'])
        all_n_gen.append(s['run_info']['n_gen_actual'])

        pareto_file = r['dir'] / "pareto-front.csv"
        if pareto_file.exists():
            try:
                import pandas as pd
                df = pd.read_csv(pareto_file)
                if 'gini' in df.columns:
                    all_gini_min.append(df['gini'].min())
                    all_gini_max.append(df['gini'].max())
                if 'npc' in df.columns:
                    all_npc_min.append(df['npc'].min())
                    all_npc_max.append(df['npc'].max())
                if 're_penetration_pct' in df.columns:
                    all_re_min.append(df['re_penetration_pct'].min())
                    all_re_max.append(df['re_penetration_pct'].max())
            except Exception as e:
                pass

    print("CONVERGENCE METRICS:")
    print(f"  Hypervolume:  mean={np.mean(all_hv):.6e}, std={np.std(all_hv):.6e}")
    print(f"  Solutions:    mean={np.mean(all_n_solutions):.1f}, range=[{min(all_n_solutions)}, {max(all_n_solutions)}]")
    print(f"  Generations:  mean={np.mean(all_n_gen):.1f}, range=[{min(all_n_gen)}, {max(all_n_gen)}]")
    print()

    if all_gini_min:
        print("GINI COEFFICIENT (across all Pareto fronts):")
        print(f"  Min Gini:  {min(all_gini_min):.4f} (most equitable)")
        print(f"  Max Gini:  {max(all_gini_max):.4f} (least equitable)")
        print(f"  Range:     {max(all_gini_max) - min(all_gini_min):.4f}")
        print()

    if all_npc_min:
        print("NET PRESENT COST ($M):")
        print(f"  Min NPC:   ${min(all_npc_min)/1e6:.2f}M")
        print(f"  Max NPC:   ${max(all_npc_max)/1e6:.2f}M")
        print()

    if all_re_min:
        print("RE PENETRATION (%):")
        print(f"  Min RE%:   {min(all_re_min):.1f}%")
        print(f"  Max RE%:   {max(all_re_max):.1f}%")
        print()

    print("=" * 80)
    print("COMPARISON WITH PAPER CLAIMS:")
    print("=" * 80)
    if all_gini_min:
        print(f"  Paper Gini range:    0.169 - 0.506")
        print(f"  Actual Gini range:   {min(all_gini_min):.3f} - {max(all_gini_max):.3f}")

        if all_npc_min and len(all_npc_min) == len(all_gini_min):

            print()
            print("  Trade-off verification pending full Pareto analysis...")
    print()

def analyze_pareto():

    runs = get_completed_runs()

    if not runs:
        print("No completed runs found.")
        return

    print("=" * 80)
    print(f"PARETO FRONT ANALYSIS ({len(runs)} runs)")
    print("=" * 80)
    print()

    try:
        import pandas as pd
    except ImportError:
        print("pandas required for Pareto analysis")
        return

    all_solutions = []

    for r in runs:
        pareto_file = r['dir'] / "pareto-front.csv"
        if pareto_file.exists():
            try:
                df = pd.read_csv(pareto_file)
                df['run_id'] = r['summary']['run_info']['run_id']
                df['seed'] = r['summary']['run_info']['seed']
                all_solutions.append(df)
            except:
                pass

    if not all_solutions:
        print("No Pareto front CSV files found.")
        return

    combined = pd.concat(all_solutions, ignore_index=True)
    print(f"Total solutions across all runs: {len(combined)}")
    print()

    obj_cols = ['npc', 'lpsp', 'co2', 'gini']
    available_cols = [c for c in obj_cols if c in combined.columns]

    print("OBJECTIVE RANGES:")
    for col in available_cols:
        print(f"  {col:12s}: min={combined[col].min():.4g}, max={combined[col].max():.4g}")
    print()

    if len(available_cols) >= 2:
        print("CORRELATIONS (Spearman):")
        for i, c1 in enumerate(available_cols):
            for c2 in available_cols[i+1:]:
                corr = combined[c1].corr(combined[c2], method='spearman')
                print(f"  {c1} vs {c2}: {corr:.3f}")
        print()

    if 're_penetration_pct' in combined.columns and 'gini' in combined.columns:
        corr_re_gini = combined['re_penetration_pct'].corr(combined['gini'], method='spearman')
        print(f"KEY CORRELATION - RE% vs Gini: {corr_re_gini:.3f}")
        print(f"  Expected: -1.0 (more RE = less inequality)")
        print(f"  {'ALIGNED' if corr_re_gini < -0.5 else 'CHECK REQUIRED'}")
        print()

    if 'gini' in combined.columns and 'npc' in combined.columns:

        most_equitable = combined.loc[combined['gini'].idxmin()]
        least_equitable = combined.loc[combined['gini'].idxmax()]

        print("EQUITY PREMIUM:")
        print(f"  Most equitable:   Gini={most_equitable['gini']:.4f}, NPC=${most_equitable['npc']/1e6:.2f}M")
        print(f"  Least equitable:  Gini={least_equitable['gini']:.4f}, NPC=${least_equitable['npc']/1e6:.2f}M")

        premium_pct = (most_equitable['npc'] - least_equitable['npc']) / least_equitable['npc'] * 100
        print(f"  Premium:          {premium_pct:+.1f}%")
        print(f"  Paper claims:     10-15% higher capital for equity")
        print(f"  {'ALIGNED' if 5 <= premium_pct <= 20 else 'CHECK REQUIRED'}")
        print()

    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description='Monitor batch optimization runs')
    parser.add_argument('--watch', action='store_true', help='Auto-refresh every 30s')
    parser.add_argument('--summary', action='store_true', help='Show preliminary results')
    parser.add_argument('--pareto', action='store_true', help='Analyze Pareto fronts')
    args = parser.parse_args()

    if args.summary:
        show_summary()
    elif args.pareto:
        analyze_pareto()
    elif args.watch:
        try:
            while True:
                os.system('cls' if os.name == 'nt' else 'clear')
                show_status()
                print(f"\nRefreshing in 30s... (Ctrl+C to stop)")
                time.sleep(30)
        except KeyboardInterrupt:
            print("\nMonitor stopped.")
    else:
        show_status()

if __name__ == "__main__":
    main()
