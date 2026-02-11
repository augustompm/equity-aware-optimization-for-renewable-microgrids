import sys
import os
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import json
import time

project_root = Path(__file__).parent

def log_batch(msg, batch_log_file):

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(batch_log_file, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def run_single_optimization(run_id, seed, n_gen, batch_log_file):

    log_batch(f"", batch_log_file)
    log_batch(f"{'='*80}", batch_log_file)
    log_batch(f"STARTING RUN {run_id}/10 (seed={seed})", batch_log_file)
    log_batch(f"{'='*80}", batch_log_file)

    start_time = time.time()

    cmd = [
        str(project_root / ".venv" / "Scripts" / "python.exe"),
        str(project_root / "production-run-v8.py"),
        f"--run_id={run_id}",
        f"--seed={seed}",
        f"--n_gen={n_gen}",
        "--results_dir=results"
    ]

    log_batch(f"Command: {' '.join(cmd)}", batch_log_file)

    try:

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=project_root
        )

        output_lines = []
        for line in process.stdout:
            line = line.rstrip()
            print(line, flush=True)
            output_lines.append(line)

            with open(batch_log_file, 'a', encoding='utf-8') as f:
                f.write(line + '\n')

        process.wait()

        elapsed = time.time() - start_time
        elapsed_min = elapsed / 60

        if process.returncode != 0:
            log_batch(f"", batch_log_file)
            log_batch(f"[ERROR] Run {run_id} FAILED with code {process.returncode}", batch_log_file)
            log_batch(f"Elapsed: {elapsed_min:.1f} min", batch_log_file)
            return None

        log_batch(f"", batch_log_file)
        log_batch(f"[SUCCESS] Run {run_id} completed in {elapsed_min:.1f} min", batch_log_file)

        results_base = project_root / "results"
        run_dirs = sorted(results_base.glob(f"v8-run{run_id}-seed{seed}-*"))

        if not run_dirs:
            log_batch(f"[WARN] No result directory found for run {run_id}", batch_log_file)
            return {
                'run_id': run_id,
                'seed': seed,
                'success': True,
                'elapsed_min': elapsed_min,
                'result_dir': None
            }

        result_dir = run_dirs[-1]

        summary_file = result_dir / "summary.json"
        summary = None
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)

        return {
            'run_id': run_id,
            'seed': seed,
            'success': True,
            'elapsed_min': elapsed_min,
            'result_dir': str(result_dir),
            'summary': summary
        }

    except Exception as e:
        elapsed = time.time() - start_time
        elapsed_min = elapsed / 60
        log_batch(f"", batch_log_file)
        log_batch(f"[ERROR] Run {run_id} exception: {e}", batch_log_file)
        log_batch(f"Elapsed: {elapsed_min:.1f} min", batch_log_file)
        return None

def create_batch_summary(results, batch_log_file, output_file):

    log_batch(f"", batch_log_file)
    log_batch(f"{'='*80}", batch_log_file)
    log_batch(f"CREATING BATCH SUMMARY", batch_log_file)
    log_batch(f"{'='*80}", batch_log_file)

    successful_runs = [r for r in results if r is not None and r['success']]
    failed_runs = [r for r in results if r is None or not r['success']]

    summary = {
        'batch_info': {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'total_runs': len(results),
            'successful_runs': len(successful_runs),
            'failed_runs': len(failed_runs)
        },
        'runs': []
    }

    total_time = 0

    for r in successful_runs:
        run_summary = {
            'run_id': r['run_id'],
            'seed': r['seed'],
            'elapsed_min': r['elapsed_min'],
            'result_dir': r['result_dir']
        }

        total_time += r['elapsed_min']

        if r.get('summary'):
            s = r['summary']
            run_summary['n_gen_actual'] = s['run_info']['n_gen_actual']
            run_summary['early_stopped'] = s['run_info']['early_stopped']
            run_summary['n_solutions'] = s['pareto_front']['n_solutions']
            run_summary['hypervolume_final'] = s['convergence']['final']['hypervolume']
            run_summary['hypervolume_improvement'] = s['convergence']['hypervolume_improvement']

        summary['runs'].append(run_summary)

    summary['batch_info']['total_time_min'] = total_time
    summary['batch_info']['total_time_hours'] = total_time / 60
    summary['batch_info']['avg_time_per_run_min'] = total_time / len(successful_runs) if successful_runs else 0

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    log_batch(f"Batch summary saved: {output_file}", batch_log_file)
    log_batch(f"", batch_log_file)
    log_batch(f"{'='*80}", batch_log_file)
    log_batch(f"BATCH COMPLETE", batch_log_file)
    log_batch(f"{'='*80}", batch_log_file)
    log_batch(f"Successful runs: {len(successful_runs)}/{len(results)}", batch_log_file)
    log_batch(f"Failed runs: {len(failed_runs)}", batch_log_file)
    log_batch(f"Total time: {total_time/60:.1f} hours", batch_log_file)
    log_batch(f"Average per run: {summary['batch_info']['avg_time_per_run_min']:.1f} min", batch_log_file)
    log_batch(f"{'='*80}", batch_log_file)

    return summary

def main():
    parser = argparse.ArgumentParser(description='Batch Runner V8')
    parser.add_argument('--n_gen', type=int, default=200, help='Generations per run')
    parser.add_argument('--start_seed', type=int, default=42, help='Starting seed (will use start_seed to start_seed+9)')
    parser.add_argument('--n_runs', type=int, default=10, help='Number of runs')
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_log_file = project_root / f"batch-v8-{timestamp}.log"

    log_batch("="*80, batch_log_file)
    log_batch("BATCH RUNNER V8 - PRODUCTION RUNS", batch_log_file)
    log_batch("="*80, batch_log_file)
    log_batch(f"Number of runs: {args.n_runs}", batch_log_file)
    log_batch(f"Generations per run: {args.n_gen}", batch_log_file)
    log_batch(f"Seeds: {args.start_seed} to {args.start_seed + args.n_runs - 1}", batch_log_file)
    log_batch(f"Batch log: {batch_log_file}", batch_log_file)
    log_batch("="*80, batch_log_file)

    results = []

    for i in range(args.n_runs):
        run_id = i + 1
        seed = args.start_seed + i

        result = run_single_optimization(run_id, seed, args.n_gen, batch_log_file)
        results.append(result)

        if i < args.n_runs - 1:
            log_batch(f"Pausing 5s before next run...", batch_log_file)
            time.sleep(5)

    summary_file = project_root / f"batch-summary-v8-{timestamp}.json"
    batch_summary = create_batch_summary(results, batch_log_file, summary_file)

    print("")
    print("="*80)
    print("BATCH RUNNER V8 - FINAL SUMMARY")
    print("="*80)
    print(f"Successful runs: {batch_summary['batch_info']['successful_runs']}/{batch_summary['batch_info']['total_runs']}")
    print(f"Total time: {batch_summary['batch_info']['total_time_hours']:.1f} hours")
    print(f"Average per run: {batch_summary['batch_info']['avg_time_per_run_min']:.1f} min")
    print(f"")
    print(f"Batch log: {batch_log_file}")
    print(f"Batch summary: {summary_file}")
    print("="*80)

    if batch_summary['batch_info']['failed_runs'] > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
