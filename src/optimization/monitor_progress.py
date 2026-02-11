import pandas as pd
import time
import sys
from pathlib import Path
import argparse

project_root = Path(__file__).parent.parent.parent

def check_progress():

    convergence_file = project_root / "results" / "convergence-history.csv"

    if not convergence_file.exists():
        return {
            'status': 'NOT_STARTED',
            'message': 'Optimization not started yet (no convergence-history.csv)',
            'warnings': []
        }

    df = pd.read_csv(convergence_file)

    if len(df) == 0:
        return {
            'status': 'STARTING',
            'message': 'Optimization just started (0 generations)',
            'warnings': []
        }

    latest = df.iloc[-1]
    gen = int(latest['generation'])
    n_feasible = int(latest['n_feasible'])
    n_infeasible = int(latest['n_infeasible'])

    total_solutions = n_feasible + n_infeasible
    feasibility_rate = n_feasible / total_solutions if total_solutions > 0 else 0.0

    warnings = []
    status = 'RUNNING'

    if gen > 100 and n_feasible == 0:
        warnings.append("CRITICAL: No feasible solutions after 100 generations - constraints may be too tight")
        status = 'CONCERNING'

    if gen > 50 and feasibility_rate < 0.1:
        warnings.append(f"WARNING: Low feasibility rate ({feasibility_rate*100:.1f}%) after gen {gen}")
        status = 'CONCERNING'

    if gen >= 10 and gen < 200:
        first_10 = df.iloc[:10]
        last_10 = df.iloc[-10:]

        npc_first = first_10['min_NPC'].mean()
        npc_last = last_10['min_NPC'].mean()

        if abs(npc_last - npc_first) / npc_first < 0.01:
            warnings.append(f"WARNING: NPC stagnant (< 1% change in last {min(10, gen)} gens)")

    if gen > 150:
        last_50 = df.iloc[-50:]
        cv_avg_50 = (last_50['n_infeasible'] / (last_50['n_feasible'] + last_50['n_infeasible'])).mean()

        if cv_avg_50 > 0.5:
            warnings.append(f"WARNING: High infeasibility in last 50 gens ({cv_avg_50*100:.1f}%)")

    report = {
        'status': status,
        'generation': gen,
        'total_generations': 200,
        'progress_pct': (gen / 200) * 100,
        'n_feasible': n_feasible,
        'n_infeasible': n_infeasible,
        'feasibility_rate': feasibility_rate,
        'min_NPC': latest['min_NPC'],
        'min_LPSP': latest['min_LPSP'],
        'min_CO2': latest['min_CO2'],
        'min_Gini': latest['min_Gini'],
        'warnings': warnings
    }

    return report

def print_report(report):

    print("=" * 70)
    print("NSGA-III OPTIMIZATION PROGRESS MONITOR")
    print("=" * 70)
    print()

    if report['status'] == 'NOT_STARTED':
        print(report['message'])
        return

    if report['status'] == 'STARTING':
        print(report['message'])
        return

    gen = report['generation']
    total = report['total_generations']
    pct = report['progress_pct']

    print(f"Generation: {gen}/{total} ({pct:.1f}% complete)")
    print()

    bar_width = 50
    filled = int(bar_width * gen / total)
    bar = '#' * filled + '-' * (bar_width - filled)
    print(f"[{bar}]")
    print()

    print(f"Solutions:")
    print(f"  Feasible:   {report['n_feasible']}")
    print(f"  Infeasible: {report['n_infeasible']}")
    print(f"  Feasibility Rate: {report['feasibility_rate']*100:.1f}%")
    print()

    print(f"Best Objectives (minimization):")
    print(f"  NPC:  ${report['min_NPC']:,.0f}")
    print(f"  LPSP: {report['min_LPSP']:.4f} ({report['min_LPSP']*100:.2f}%)")
    print(f"  CO2:  {report['min_CO2']:,.0f} kg")
    print(f"  Gini: {report['min_Gini']:.4f}")
    print()

    if report['warnings']:
        print("WARNING:")
        for warning in report['warnings']:
            print(f"  - {warning}")
        print()
        print("ACTION: Review convergence-history.csv for details")
        print("If CRITICAL: May need to stop and adjust constraints")
    else:
        print("Status: Normal convergence")

    print()

    eta_hours = (total - gen) * 0.48 / 3600
    print(f"Estimated Time Remaining: {eta_hours:.1f} hours")
    print()
    print("=" * 70)

def monitor_continuous(interval_minutes=30):

    print("Starting continuous monitoring...")
    print(f"Checking every {interval_minutes} minutes")
    print("Press Ctrl+C to stop")
    print()

    try:
        while True:
            report = check_progress()
            print_report(report)

            if report['status'] in ['NOT_STARTED', 'STARTING']:
                print(f"Waiting {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)
                continue

            if report['generation'] >= 200:
                print("Optimization complete!")
                break

            if report['status'] == 'CONCERNING':
                print(f"Issues detected. Next check in {interval_minutes} minutes...")
            else:
                print(f"Next check in {interval_minutes} minutes...")

            time.sleep(interval_minutes * 60)

    except KeyboardInterrupt:
        print()
        print("Monitoring stopped by user")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Monitor NSGA-III optimization progress'
    )
    parser.add_argument('--check-once', action='store_true',
                        help='Check once and exit (default: continuous)')
    parser.add_argument('--interval', type=int, default=30,
                        help='Minutes between checks (default: 30)')

    args = parser.parse_args()

    if args.check_once:
        report = check_progress()
        print_report(report)
    else:
        monitor_continuous(args.interval)
