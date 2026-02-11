import subprocess
import sys
import argparse
from pathlib import Path
from datetime import datetime
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_seed', type=int, default=42)
    parser.add_argument('--end_seed', type=int, default=71)
    parser.add_argument('--n_gen', type=int, default=200)
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--n_jobs', type=int, default=1)
    args = parser.parse_args()

    project_root = Path(__file__).parent
    python_exe = sys.executable
    runner_script = project_root / "production-run-v9-fast.py"

    seeds = list(range(args.start_seed, args.end_seed + 1))
    n_runs = len(seeds)
    batch_start = time.time()

    print(f"Batch run: {n_runs} seeds ({args.start_seed}-{args.end_seed})")

    for i, seed in enumerate(seeds):
        run_id = i + 1
        cmd = [python_exe, str(runner_script),
               f"--run_id={run_id}", f"--seed={seed}",
               f"--n_gen={args.n_gen}", f"--results_dir={args.results_dir}",
               f"--n_jobs={args.n_jobs}"]

        run_start = time.time()
        result = subprocess.run(cmd, cwd=str(project_root))
        run_time = time.time() - run_start

        status = "ok" if result.returncode == 0 else "fail"
        print(f"[{i+1}/{n_runs}] seed {seed}: {status} ({run_time:.0f}s)")

        if i < n_runs - 1:
            time.sleep(2)

    total = time.time() - batch_start
    print(f"Batch complete: {total/60:.1f} min")

if __name__ == "__main__":
    main()
