import sys
import argparse
import numpy as np

def estimate_memory(pop_size, n_gen):

    bytes_per_float64 = 8

    bytes_per_solution = (
        4 * bytes_per_float64 +
        4 * bytes_per_float64 +
        6 * bytes_per_float64 +
        8760 * bytes_per_float64
    )

    population_bytes = pop_size * bytes_per_solution
    population_mb = population_bytes / (1024**2)

    history_bytes = pop_size * n_gen * bytes_per_solution
    history_mb = history_bytes / (1024**2)

    overhead_mb = population_mb * 2

    total_mb = population_mb + history_mb + overhead_mb

    return {
        'population_mb': population_mb,
        'history_mb': history_mb,
        'overhead_mb': overhead_mb,
        'total_mb': total_mb,
        'total_gb': total_mb / 1024
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Estimate memory requirements for NSGA-III optimization'
    )
    parser.add_argument('--pop', type=int, required=True,
                        help='Population size')
    parser.add_argument('--gen', type=int, required=True,
                        help='Number of generations')
    args = parser.parse_args()

    print("=" * 70)
    print("MEMORY ESTIMATION FOR NSGA-III OPTIMIZATION")
    print("=" * 70)
    print()

    mem = estimate_memory(args.pop, args.gen)

    print(f"Configuration:")
    print(f"  Population size: {args.pop}")
    print(f"  Generations: {args.gen}")
    print(f"  Total evaluations: {args.pop * args.gen:,}")
    print()

    print(f"Memory Breakdown:")
    print(f"  Population (current gen):  {mem['population_mb']:8.2f} MB")
    print(f"  History (all gens):        {mem['history_mb']:8.2f} MB")
    print(f"  pymoo overhead (est):      {mem['overhead_mb']:8.2f} MB")
    print(f"  " + "-" * 50)
    print(f"  Total estimated:           {mem['total_mb']:8.2f} MB")
    print(f"                            ({mem['total_gb']:8.2f} GB)")
    print()

    print("Assessment:")
    if mem['total_mb'] < 2000:
        print("  [OK] Memory usage < 2 GB - safe for standard machines")
        sys.exit(0)
    elif mem['total_mb'] < 4000:
        print("  [OK] Memory usage < 4 GB - acceptable for most systems")
        sys.exit(0)
    elif mem['total_mb'] < 8000:
        print("  [WARNING] Memory usage 4-8 GB - verify RAM available")
        print("  Recommendation: Close other applications before running")
        sys.exit(0)
    elif mem['total_mb'] < 16000:
        print("  [WARNING] Memory usage 8-16 GB - requires high-RAM machine")
        print("  Recommendation: Use workstation or cloud VM with 16+ GB RAM")
        sys.exit(0)
    else:
        print("  [ERROR] Memory usage > 16 GB - exceeds typical systems")
        print("  Recommendation: Reduce population or generations")
        print("  Alternative: Use HPC cluster or cloud instance with 32+ GB RAM")
        sys.exit(1)
