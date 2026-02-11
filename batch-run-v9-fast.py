import subprocess
import sys
import argparse
from pathlib import Path
from datetime import datetime
import time

def main():
    parser = argparse.ArgumentParser(description='Batch Run V9 - FAST')
    parser.add_argument('--start_seed', type=int, default=46, help='Starting seed')
    parser.add_argument('--end_seed', type=int, default=71, help='Ending seed (inclusive)')
    parser.add_argument('--n_gen', type=int, default=200, help='Max generations per run')
    parser.add_argument('--results_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--n_jobs', type=int, default=1, help='Parallel jobs per run')
    args = parser.parse_args()

    project_root = Path(__file__).parent
    python_exe = project_root / ".venv" / "Scripts" / "python.exe"
    runner_script = project_root / "production-run-v9-fast.py"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = project_root / f"batch-v9-{timestamp}.log"

    def log(msg):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_file, 'a') as f:
            f.write(line + '\n')

    seeds = list(range(args.start_seed, args.end_seed + 1))
    n_runs = len(seeds)

    log("=" * 80)
    log("BATCH RUN V9 - FAST OPTIMIZED")
    log("=" * 80)
    log(f"Seeds: {args.start_seed} - {args.end_seed} ({n_runs} runs)")
    log(f"Max generations: {args.n_gen}")
    log(f"Results dir: {args.results_dir}")
    log(f"Parallel jobs per run: {args.n_jobs}")
    log(f"Log file: {log_file}")
    log("")

    batch_start = time.time()
    completed = 0
    failed = 0
    run_times = []

    for i, seed in enumerate(seeds):
        run_id = i + 1 + (args.start_seed - 46)

        log("")
        log("=" * 80)
        log(f"STARTING RUN {i+1}/{n_runs} (seed={seed}, run_id={run_id})")
        log("=" * 80)

        cmd = [
            str(python_exe),
            str(runner_script),
            f"--run_id={run_id}",
            f"--seed={seed}",
            f"--n_gen={args.n_gen}",
            f"--results_dir={args.results_dir}",
            f"--n_jobs={args.n_jobs}"
        ]

        log(f"Command: {' '.join(cmd)}")

        run_start = time.time()

        try:
            result = subprocess.run(
                cmd,
                cwd=str(project_root),
                capture_output=False,
                text=True,
                timeout=3600
            )

            run_time = time.time() - run_start
            run_times.append(run_time)

            if result.returncode == 0:
                completed += 1
                log(f"")
                log(f"[SUCCESS] Run {i+1} completed in {run_time/60:.1f} min")
            else:
                failed += 1
                log(f"")
                log(f"[FAILED] Run {i+1} failed with code {result.returncode}")

        except subprocess.TimeoutExpired:
            failed += 1
            log(f"[TIMEOUT] Run {i+1} timed out after 1 hour")
        except Exception as e:
            failed += 1
            log(f"[ERROR] Run {i+1} exception: {e}")

        elapsed = time.time() - batch_start
        if run_times:
            avg_time = sum(run_times) / len(run_times)
            remaining = (n_runs - i - 1) * avg_time
            log(f"Progress: {i+1}/{n_runs} | Elapsed: {elapsed/60:.1f}min | ETA: {remaining/60:.1f}min")

        if i < n_runs - 1:
            log("Pausing 2s before next run...")
            time.sleep(2)

    total_time = time.time() - batch_start

    log("")
    log("=" * 80)
    log("BATCH COMPLETED")
    log("=" * 80)
    log(f"Total runs: {n_runs}")
    log(f"Completed: {completed}")
    log(f"Failed: {failed}")
    log(f"Total time: {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
    if run_times:
        log(f"Average time per run: {sum(run_times)/len(run_times)/60:.1f} min")
    log(f"Results in: {args.results_dir}")
    log("=" * 80)

if __name__ == "__main__":
    main()
