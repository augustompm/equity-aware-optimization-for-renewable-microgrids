# Equity-Aware Multi-Objective Optimization for Renewable Microgrids

Source code for IEEE SYSCON 2026 paper submission

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run optimization:

```bash
python production-run-v8.py --run_id 1 --seed 42 --n_gen 200
```

Aggregate results:

```bash
python aggregate-30-runs-v8.py
```

Generate figures:

```bash
python -c "from src.visualization.generate_paper_plots import generate_all_plots; generate_all_plots('results/aggregated')"
```

## Data

- `data/load-profile-8760h.csv` - Hourly load (Inuvik, NWT)
- `data/meteorology-8760h.csv` - Solar/wind capacity factors (CASES database)
- `data/reference-front-v8.csv` - Pareto front from 30 independent runs

## Structure

- `src/` - Core modules
- `data/` - Input datasets
- `results/` - Output directory (auto-created)
- `production-run-v8.py` - Main optimization script
- `aggregate-30-runs-v8.py` - Results aggregation

## License

MIT License
