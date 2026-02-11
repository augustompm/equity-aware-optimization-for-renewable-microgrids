# Equity-Aware Optimization for Renewable Microgrids

Multi-objective optimization framework for hybrid renewable energy systems with equity considerations.

## Overview

This framework uses NSGA-III to optimize microgrid configurations considering four objectives:
- Net Present Cost (NPC)
- Loss of Power Supply Probability (LPSP)
- CO2 Emissions
- Energy Equity (Gini coefficient)

## Case Study

Inuvik, Northwest Territories, Canada (68N latitude, population 3,137).

## Requirements

- Python 3.10+
- pymoo
- numpy
- pandas
- joblib

## Installation

```bash
pip install pymoo numpy pandas joblib matplotlib
```

## Usage

Run a single optimization:
```bash
python production-run-v9-fast.py
```

Run batch optimization (30 seeds):
```bash
python batch-run-v9-fast.py
```

Generate paper figures:
```bash
python generate-paper-figures-v9.py
```

## Project Structure

```
config.py                 - System configuration and parameters
production-run-v9-fast.py - Single run script
batch-run-v9-fast.py      - Batch runner for multiple seeds
data/                     - Input data (load profiles, capacity factors)
src/
  components/             - PV, wind, battery, diesel generator models
  objectives/             - Objective functions (NPC, LPSP, CO2, Gini)
  optimization/           - NSGA-III problem definition
  simulation/             - System simulator
  constraints/            - Constraint functions
```

## License

MIT License
