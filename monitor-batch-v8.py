import sys
import os
import time
import re
from pathlib import Path
from datetime import datetime, timedelta
import json

project_root = Path(__file__).parent

def find_latest_batch_log():

    log_files = sorted(project_root.glob("batch-v8-*.log"))
    if not log_files:
        return None
    return log_files[-1]

def parse_log_progress(log_file):

    if not log_file.exists():
        return None

    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    info = {
        'total_runs': 10,
        'current_run': 0,
        'completed_runs': 0,
        'current_generation': 0,
        'max_generations': 200,
        'elapsed_times': [],
        'early_stopped': [],
        'latest_metrics': {}
    }

    for line in lines:

        match = re.search(r'STARTING RUN (\d+)/(\d+)', line)
        if match:
            info['current_run'] = int(match.group(1))
            info['total_runs'] = int(match.group(2))

        match = re.search(r'\[SUCCESS\] Run (\d+) completed in ([\d.]+) min', line)
        if match:
            run_id = int(match.group(1))
            elapsed = float(match.group(2))
            info['completed_runs'] = max(info['completed_runs'], run_id)
            info['elapsed_times'].append(elapsed)

        match = re.search(r'Gen\s+(\d+):', line)
        if match:
            info['current_generation'] = int(match.group(1))

        match = re.search(r'HV=([\d.e+-]+)', line)
        if match:
            info['latest_metrics']['HV'] = match.group(1)

        match = re.search(r'IGD\+=([\d.e+-]+|inf|Infinity)', line)
        if match:
            info['latest_metrics']['IGD+'] = match.group(1)

        if 'Early stop:' in line or 'EARLY STOPPING TRIGGERED' in line:
            info['early_stopped'].append(info['current_run'])

    return info

def format_time_remaining(minutes):

    if minutes < 0:
        return "calculating..."

    hours = int(minutes // 60)
    mins = int(minutes % 60)

    if hours > 0:
        return f"{hours}h {mins}m"
    else:
        return f"{mins}m"

def display_progress(info):

    os.system('cls' if os.name == 'nt' else 'clear')

    print("="*80)
    print("BATCH V8 - PROGRESS MONITOR")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    completed = info['completed_runs']
    total = info['total_runs']
    current = info['current_run']

    progress_pct = (completed / total) * 100 if total > 0 else 0

    print(f"Overall Progress: {completed}/{total} runs completed ({progress_pct:.1f}%)")
    print(f"Current Run: {current}/{total}")
    print()

    if current > 0:
        gen = info['current_generation']
        max_gen = info['max_generations']
        gen_pct = (gen / max_gen) * 100 if max_gen > 0 else 0

        print(f"Current Generation: {gen}/{max_gen} ({gen_pct:.1f}%)")

        bar_width = 50
        filled = int(bar_width * gen / max_gen) if max_gen > 0 else 0
        bar = '█' * filled + '░' * (bar_width - filled)
        print(f"[{bar}]")
        print()

    if info['latest_metrics']:
        print("Latest Metrics:")
        for key, val in info['latest_metrics'].items():
            print(f"  {key}: {val}")
        print()

    if info['elapsed_times']:
        avg_time = sum(info['elapsed_times']) / len(info['elapsed_times'])
        remaining_runs = total - completed
        estimated_remaining = avg_time * remaining_runs

        print(f"Completed Runs: {len(info['elapsed_times'])}")
        print(f"Average Time per Run: {avg_time:.1f} min")
        print(f"Estimated Time Remaining: {format_time_remaining(estimated_remaining)}")
        print()

    if info['early_stopped']:
        print(f"Early Stopped Runs: {len(info['early_stopped'])} - {info['early_stopped']}")
        print()

    if info['elapsed_times']:
        print("Completed Run Times:")
        for i, t in enumerate(info['elapsed_times'], 1):
            status = " (early stopped)" if i in info['early_stopped'] else ""
            print(f"  Run {i}: {t:.1f} min{status}")
        print()

    print("="*80)
    print("Press Ctrl+C to exit monitor (batch will continue running)")
    print("="*80)

def main():
    print("Looking for batch log file...")

    log_file = find_latest_batch_log()

    if not log_file:
        print("[ERROR] No batch log file found")
        print("Expected pattern: batch-v8-*.log")
        sys.exit(1)

    print(f"Monitoring: {log_file}")
    print()
    time.sleep(2)

    try:
        while True:
            info = parse_log_progress(log_file)

            if info:
                display_progress(info)

                if info['completed_runs'] >= info['total_runs']:
                    print()
                    print("="*80)
                    print("BATCH COMPLETE!")
                    print("="*80)
                    break

            time.sleep(5)

    except KeyboardInterrupt:
        print()
        print()
        print("="*80)
        print("Monitor stopped (batch continues in background)")
        print("="*80)
        print(f"Log file: {log_file}")
        print()

if __name__ == "__main__":
    main()
