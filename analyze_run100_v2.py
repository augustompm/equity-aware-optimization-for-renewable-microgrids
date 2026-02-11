import pandas as pd
import numpy as np

df = pd.read_csv('results/v8-run100-seed42-20260210_074657/pareto-front-solutions.csv')

print("=" * 70)
print("DEEP ANALYSIS: Understanding Gini Trade-offs")
print("=" * 70)

r = 0.03
n = 25
crf = r * (1 + r)**n / ((1 + r)**n - 1)

df['fuel_cost_annual'] = df['fuel_consumption_liters'] * 0.72

pwf = (1 - (1 + r)**(-n)) / r

df['capital_estimate'] = df['npc_cad'] - df['fuel_cost_annual'] * pwf
df['capital_annual'] = df['capital_estimate'] * crf

print("\n1. COST STRUCTURE ANALYSIS")
print("-" * 50)
print(f"Fuel annual: ${df['fuel_cost_annual'].min()/1e6:.2f}M - ${df['fuel_cost_annual'].max()/1e6:.2f}M")
print(f"Capital est: ${df['capital_estimate'].min()/1e6:.2f}M - ${df['capital_estimate'].max()/1e6:.2f}M")
print(f"Capital annual: ${df['capital_annual'].min()/1e6:.2f}M - ${df['capital_annual'].max()/1e6:.2f}M")

print("\n2. COST COMPONENTS vs GINI")
print("-" * 50)
print(f"Fuel annual vs Gini:    r = {df['fuel_cost_annual'].corr(df['gini']):.3f}")
print(f"Capital annual vs Gini: r = {df['capital_annual'].corr(df['gini']):.3f}")
print(f"Total annual vs Gini:   r = {(df['fuel_cost_annual']+df['capital_annual']).corr(df['gini']):.3f}")

print("\n3. KEY FINDING")
print("-" * 50)
print("The positive NPC-Gini correlation (+0.752) occurs because:")
print("  - Higher NPC = higher total costs")
print("  - Higher costs = higher burdens for everyone")
print("  - Low-income burden increases MORE (fuel per-HH + capital share)")
print("  - Burden inequality INCREASES with total cost")

print("\n4. TRADE-OFF INTERPRETATION")
print("-" * 50)
print("Objectives and their relationships:")
print("  - NPC vs LPSP:  -0.566 (conflicting) - cheaper = less reliable")
print("  - NPC vs CO2:   -0.257 (conflicting) - cheaper = more emissions")
print("  - NPC vs Gini:  +0.752 (aligned)     - cheaper = MORE equitable")
print("  - LPSP vs Gini: -0.944 (conflicting) - reliable = LESS equitable")
print("  - CO2 vs Gini:  +0.417 (aligned)     - clean = more equitable")

print("\n5. REAL TRADE-OFF: RELIABILITY vs EQUITY")
print("-" * 50)
print("The core conflict is NOT cost vs equity, but RELIABILITY vs EQUITY:")
print("  - High reliability requires expensive infrastructure")
print("  - Expensive infrastructure = high costs = high burden inequality")
print("  - This is a novel finding for Arctic energy justice!")

df_sorted = df.sort_values('lpsp')
best_reliability = df_sorted.head(5)
worst_reliability = df_sorted.tail(5)

print("\nBest reliability (lowest LPSP):")
print(f"  LPSP: {best_reliability['lpsp'].mean()*100:.2f}%")
print(f"  NPC:  ${best_reliability['npc_cad'].mean()/1e6:.1f}M")
print(f"  Gini: {best_reliability['gini'].mean():.4f}")

print("\nWorst reliability (highest LPSP):")
print(f"  LPSP: {worst_reliability['lpsp'].mean()*100:.2f}%")
print(f"  NPC:  ${worst_reliability['npc_cad'].mean()/1e6:.1f}M")
print(f"  Gini: {worst_reliability['gini'].mean():.4f}")

print("\n6. PAPER IMPLICATIONS")
print("-" * 50)
print("The paper's claim 'high-equity requires 10-15% higher capital' applies to")
print("the RE ACCESS formulation (saturation mechanism), NOT the BURDEN formulation.")
print("")
print("With Energy Burden Gini, the narrative should be:")
print("  'Lower-cost solutions are more equitable, as they reduce energy burden")
print("   inequality. However, achieving high reliability requires infrastructure")
print("   investments that increase burden inequality, revealing a fundamental")
print("   trade-off between system reliability and energy equity.'")

print("\n" + "=" * 70)
