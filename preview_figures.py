import pandas as pd
import numpy as np

df = pd.read_csv('results/v8-run101-seed42-20260210_083848/pareto-front-solutions.csv')

print("=" * 80)
print("PREVIEW NUMERICO DOS GRAFICOS - RUN 101")
print("=" * 80)

print("\n" + "=" * 80)
print("FIG 2: 2D PARETO PROJECTIONS")
print("=" * 80)

print("\n### Panel (a): LPSP vs Gini ###")
print("-" * 50)

lpsp_bins = [(0, 0.01), (0.01, 0.02), (0.02, 0.03), (0.03, 0.04), (0.04, 0.05)]
print(f"{'LPSP Range':<20} {'N':<5} {'Gini Mean':<12} {'Gini Range'}")
for low, high in lpsp_bins:
    mask = (df['lpsp'] >= low) & (df['lpsp'] < high)
    n = mask.sum()
    if n > 0:
        gini_mean = df.loc[mask, 'gini'].mean()
        gini_min = df.loc[mask, 'gini'].min()
        gini_max = df.loc[mask, 'gini'].max()
        print(f"{low*100:.0f}-{high*100:.0f}%{'':<14} {n:<5} {gini_mean:.4f}       [{gini_min:.4f}, {gini_max:.4f}]")

print(f"\nCorrelacao LPSP-Gini: r = {df['lpsp'].corr(df['gini']):+.3f}")
print("Interpretacao: Quase independentes (fraca correlacao)")

print("\n### Panel (b): NPC vs Gini ###")
print("-" * 50)

npc_min, npc_max = df['npc_cad'].min()/1e6, df['npc_cad'].max()/1e6
npc_bins = [(105, 110), (110, 115), (115, 120), (120, 125), (125, 131)]
print(f"{'NPC Range (M$)':<20} {'N':<5} {'Gini Mean':<12} {'Gini Range'}")
for low, high in npc_bins:
    mask = (df['npc_cad']/1e6 >= low) & (df['npc_cad']/1e6 < high)
    n = mask.sum()
    if n > 0:
        gini_mean = df.loc[mask, 'gini'].mean()
        gini_min = df.loc[mask, 'gini'].min()
        gini_max = df.loc[mask, 'gini'].max()
        print(f"${low}-{high}M{'':<10} {n:<5} {gini_mean:.4f}       [{gini_min:.4f}, {gini_max:.4f}]")

print(f"\nCorrelacao NPC-Gini: r = {df['npc_cad'].corr(df['gini']):+.3f}")
print("Interpretacao: Maior NPC = Menor Gini (mais equity custa mais)")

print("\n### Panel (c): CO2 vs Gini ###")
print("-" * 50)

co2_bins = [(315, 330), (330, 350), (350, 370), (370, 390), (390, 410)]
print(f"{'CO2 Range (kt)':<20} {'N':<5} {'Gini Mean':<12} {'Gini Range'}")
for low, high in co2_bins:
    mask = (df['co2_kg']/1e3 >= low) & (df['co2_kg']/1e3 < high)
    n = mask.sum()
    if n > 0:
        gini_mean = df.loc[mask, 'gini'].mean()
        gini_min = df.loc[mask, 'gini'].min()
        gini_max = df.loc[mask, 'gini'].max()
        print(f"{low}-{high} kt{'':<10} {n:<5} {gini_mean:.4f}       [{gini_min:.4f}, {gini_max:.4f}]")

print(f"\nCorrelacao CO2-Gini: r = {df['co2_kg'].corr(df['gini']):+.3f}")
print("Interpretacao: Maior CO2 = Maior Gini (emissoes e inequidade andam juntos)")

print("\n" + "=" * 80)
print("FIG 3: DECISION VARIABLE DISTRIBUTIONS")
print("=" * 80)

vars_info = [
    ('pv_kw', 'PV (kW)', 0, 10000),
    ('wind_mw', 'Wind (MW)', 0, 5),
    ('battery_mwh', 'Battery (MWh)', 0, 100),
    ('diesel_mw', 'Diesel (MW)', 0, 10)
]

print(f"\n{'Variable':<20} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12} {'% Bound Used'}")
print("-" * 80)

for col, name, lb, ub in vars_info:
    vmin, vmax = df[col].min(), df[col].max()
    vmean, vstd = df[col].mean(), df[col].std()
    pct_used = (vmax - vmin) / (ub - lb) * 100
    print(f"{name:<20} {vmin:<12.2f} {vmax:<12.2f} {vmean:<12.2f} {vstd:<12.2f} {pct_used:.1f}%")

print("\n### Histograma Aproximado (10 bins) ###")
for col, name, lb, ub in vars_info:
    print(f"\n{name}:")
    hist, edges = np.histogram(df[col], bins=10)
    max_count = max(hist)
    for i, count in enumerate(hist):
        bar = '#' * int(count / max_count * 30) if max_count > 0 else ''
        print(f"  {edges[i]:8.1f}-{edges[i+1]:8.1f}: {bar} ({count})")

print("\n" + "=" * 80)
print("FIG 4: PARALLEL COORDINATES - ARCHETYPES")
print("=" * 80)

print("\n### Arquetipo 1: Diesel-dominated (High CO2, High Gini, Low NPC) ###")
archetype1 = df.nsmallest(5, 'npc_cad')
print(f"{'Metric':<20} {'Min':<12} {'Max':<12} {'Mean':<12}")
print("-" * 60)
for col in ['npc_cad', 'lpsp', 'co2_kg', 'gini', 're_penetration_pct']:
    unit = '/1e6' if col == 'npc_cad' else '/1e3' if col == 'co2_kg' else ''
    div = 1e6 if col == 'npc_cad' else 1e3 if col == 'co2_kg' else 1
    print(f"{col:<20} {archetype1[col].min()/div:<12.3f} {archetype1[col].max()/div:<12.3f} {archetype1[col].mean()/div:<12.3f}")

print("\n### Arquetipo 2: Renewable-dominated (Low CO2, Low Gini, High NPC) ###")
archetype2 = df.nsmallest(5, 'gini')
print(f"{'Metric':<20} {'Min':<12} {'Max':<12} {'Mean':<12}")
print("-" * 60)
for col in ['npc_cad', 'lpsp', 'co2_kg', 'gini', 're_penetration_pct']:
    div = 1e6 if col == 'npc_cad' else 1e3 if col == 'co2_kg' else 1
    print(f"{col:<20} {archetype2[col].min()/div:<12.3f} {archetype2[col].max()/div:<12.3f} {archetype2[col].mean()/div:<12.3f}")

print("\n### Arquetipo 3: Balanced (Medium everything) ###")

median_npc = df['npc_cad'].median()
archetype3 = df.iloc[(df['npc_cad'] - median_npc).abs().argsort()[:5]]
print(f"{'Metric':<20} {'Min':<12} {'Max':<12} {'Mean':<12}")
print("-" * 60)
for col in ['npc_cad', 'lpsp', 'co2_kg', 'gini', 're_penetration_pct']:
    div = 1e6 if col == 'npc_cad' else 1e3 if col == 'co2_kg' else 1
    print(f"{col:<20} {archetype3[col].min()/div:<12.3f} {archetype3[col].max()/div:<12.3f} {archetype3[col].mean()/div:<12.3f}")

print("\n" + "=" * 80)
print("COMPARACAO COM CLAIMS DO PAPER")
print("=" * 80)

print("\n### Paper L259: 'High-equity configurations (Gini below 0.25)' ###")
high_equity = df[df['gini'] < 0.25]
print(f"Solucoes com Gini < 0.25: {len(high_equity)} de {len(df)} ({len(high_equity)/len(df)*100:.0f}%)")
print(f"NOTA: Gini max = {df['gini'].max():.3f}, entao TODAS solucoes tem Gini < 0.25!")
print(f"Sugestao: Ajustar threshold para Gini < 0.15 ou usar 'lowest 20%'")

print("\n### Paper L259: 'require 10-15% higher capital' ###")
best_equity = df.nsmallest(10, 'gini')
worst_equity = df.nlargest(10, 'gini')
npc_diff = (best_equity['npc_cad'].mean() - worst_equity['npc_cad'].mean()) / worst_equity['npc_cad'].mean() * 100
print(f"NPC medio (10 mais equitativos): ${best_equity['npc_cad'].mean()/1e6:.2f}M")
print(f"NPC medio (10 menos equitativos): ${worst_equity['npc_cad'].mean()/1e6:.2f}M")
print(f"Diferenca: {npc_diff:+.1f}%")
print(f"Paper claim: 10-15% -> {'OK' if 10 <= npc_diff <= 15 else 'AJUSTAR'}")

print("\n### Paper L259: 'enable 35-48% renewable penetration' ###")
print(f"RE% range atual: {df['re_penetration_pct'].min():.1f}% - {df['re_penetration_pct'].max():.1f}%")
print(f"Paper claim: 35-48% -> {'OK' if df['re_penetration_pct'].min() >= 30 else 'AJUSTAR'}")

print("\n### Paper L261: '48% reduction in emissions' ###")
co2_min, co2_max = df['co2_kg'].min(), df['co2_kg'].max()
co2_reduction = (co2_max - co2_min) / co2_max * 100
print(f"CO2 range: {co2_min/1e3:.1f} - {co2_max/1e3:.1f} kt")
print(f"Reducao: {co2_reduction:.1f}%")
print(f"Paper claim: 48% -> {'OK' if co2_reduction >= 40 else 'AJUSTAR (atual: ' + str(round(co2_reduction)) + '%)'}")

print("\n### Paper L289: 'LCOE 0.082-0.158 CAD/kWh' ###")
print(f"LCOE range atual: {df['lcoe_cad_per_kwh'].min():.3f} - {df['lcoe_cad_per_kwh'].max():.3f} CAD/kWh")
print(f"Paper claim: 0.082-0.158 -> AJUSTAR")

print("\n### Paper L289: 'RE penetration 0-48.3% (mean 30.7%)' ###")
print(f"RE% range atual: {df['re_penetration_pct'].min():.1f}% - {df['re_penetration_pct'].max():.1f}%")
print(f"RE% mean atual: {df['re_penetration_pct'].mean():.1f}%")
print(f"Paper claim: 0-48.3% (mean 30.7%) -> AJUSTAR")

print("\n" + "=" * 80)
print("RESUMO: COERENCIA DOS GRAFICOS")
print("=" * 80)

print("""
FIG 2 - 2D Pareto Projections:
  (a) LPSP vs Gini: OK - correlacao fraca (r=+0.10), quase independentes
  (b) NPC vs Gini: OK - correlacao negativa (r=-0.45), trade-off visivel
  (c) CO2 vs Gini: OK - correlacao positiva (r=+0.78), alinhados

FIG 3 - Decision Variables:
  PV: Subutilizado (10% do bound) - Wind e mais economico
  Wind: No limite superior (5 MW) - Pode estar constrangendo
  Battery: Moderado (20% do bound) - Dentro do esperado
  Diesel: Bem distribuido (53% do bound) - OK

FIG 4 - Parallel Coordinates:
  Arquetipo Diesel: Alto CO2, Alto Gini, Baixo NPC
  Arquetipo RE: Baixo CO2, Baixo Gini, Alto NPC
  Arquetipo Balanced: Valores intermediarios

CLAIMS DO PAPER:
  [OK] Trade-off cost-equity: +11% NPC para melhor equity
  [OK] RE penetration: 31-48% (paper diz 35-48%)
  [AJUSTAR] Gini threshold: max=0.19, paper diz "below 0.25"
  [AJUSTAR] CO2 reduction: 21% (paper diz 48%)
  [AJUSTAR] LCOE range: 0.14-0.18 (paper diz 0.08-0.16)
  [AJUSTAR] NPC range: 105-131M (paper diz 60-116M)
