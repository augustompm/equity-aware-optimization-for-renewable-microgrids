import pandas as pd
import numpy as np

df = pd.read_csv('results/v8-run100-seed42-20260210_074657/pareto-front-solutions.csv')

print("=" * 70)
print("ANALYSIS: Run 100 (Energy Burden Gini)")
print("=" * 70)

print("\n1. OBJECTIVE RANGES")
print("-" * 50)
for col in ['npc_cad', 'lpsp', 'co2_kg', 'gini']:
    print(f"  {col:12s}: {df[col].min():.6f} - {df[col].max():.6f} (range: {df[col].max()-df[col].min():.6f})")

print("\n2. CORRELATION MATRIX")
print("-" * 50)
objectives = df[['npc_cad', 'lpsp', 'co2_kg', 'gini']]
corr = objectives.corr()
print(corr.round(3).to_string())

print("\n3. COMPARISON: V8 RUN 99 vs RUN 100")
print("-" * 50)
print("                      Run 99 (old)     Run 100 (new)")
print(f"  Gini range:         0.0000           {df['gini'].max()-df['gini'].min():.4f}")
print(f"  Gini std:           2.1e-16          {df['gini'].std():.6f}")
print(f"  NPC-Gini corr:      N/A (constant)   {corr.loc['npc_cad', 'gini']:.3f}")
print(f"  CO2-Gini corr:      N/A (constant)   {corr.loc['co2_kg', 'gini']:.3f}")

print("\n4. TRADE-OFF ANALYSIS")
print("-" * 50)

df_sorted = df.sort_values('npc_cad')
low_npc = df_sorted.head(5)
high_npc = df_sorted.tail(5)

print("Low NPC solutions (top 5):")
print(f"  NPC:  ${low_npc['npc_cad'].mean()/1e6:.1f}M")
print(f"  Gini: {low_npc['gini'].mean():.4f}")
print(f"  RE%:  {low_npc['re_penetration_pct'].mean():.1f}%")

print("\nHigh NPC solutions (top 5):")
print(f"  NPC:  ${high_npc['npc_cad'].mean()/1e6:.1f}M")
print(f"  Gini: {high_npc['gini'].mean():.4f}")
print(f"  RE%:  {high_npc['re_penetration_pct'].mean():.1f}%")

if low_npc['gini'].mean() > high_npc['gini'].mean():
    print("\n[ISSUE] Trade-off INVERTED: Low NPC has higher Gini")
else:
    print("\n[OK] Trade-off CORRECT: High NPC has higher Gini")

print("\n5. FUEL vs RE ANALYSIS")
print("-" * 50)
df['fuel_M'] = df['fuel_consumption_liters'] * 20 / 1e6 / 27.778

df['fuel_cost_M'] = df['fuel_consumption_liters'] * 0.72 / 1e6

print(f"Fuel cost range: ${df['fuel_cost_M'].min():.2f}M - ${df['fuel_cost_M'].max():.2f}M")
print(f"RE penetration range: {df['re_penetration_pct'].min():.1f}% - {df['re_penetration_pct'].max():.1f}%")

print(f"\nCorrelation RE% vs Gini: {df['re_penetration_pct'].corr(df['gini']):.3f}")
print(f"Correlation Fuel vs Gini: {df['fuel_cost_M'].corr(df['gini']):.3f}")

print("\n6. GINI BY RE PENETRATION QUARTILES")
print("-" * 50)
df['re_quartile'] = pd.qcut(df['re_penetration_pct'], 4, labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'])
for q in ['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']:
    subset = df[df['re_quartile'] == q]
    print(f"  {q}: RE% {subset['re_penetration_pct'].mean():.1f}%, Gini {subset['gini'].mean():.4f}")

print("\n" + "=" * 70)
