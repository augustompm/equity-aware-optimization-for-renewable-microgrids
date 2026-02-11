import numpy as np
import pandas as pd

def gini_coefficient(values):

    n = len(values)
    if n == 0 or values.sum() == 0:
        return 1.0
    sorted_vals = np.sort(values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_vals) - (n + 1) * values.sum()) / (n * values.sum())

def gini_paper_model(re_ratio, n_households=1220, seed=42):

    np.random.seed(seed)

    n_low = int(n_households * 0.40)
    n_mid = int(n_households * 0.40)
    n_high = n_households - n_low - n_mid

    m_low = np.random.uniform(0.5, 1.0, n_low)
    m_mid = np.random.uniform(0.8, 1.2, n_mid)
    m_high = np.random.uniform(1.2, 2.0, n_high)

    multipliers = np.concatenate([m_low, m_mid, m_high])
    np.random.shuffle(multipliers)

    re_i = multipliers * re_ratio

    re_i = np.clip(re_i, 0.0, 1.0)

    return gini_coefficient(re_i)

def gini_theja_scarcity(re_ratio, n_households=1220, seed=42):

    np.random.seed(seed)

    n_low = int(n_households * 0.40)
    n_mid = int(n_households * 0.40)
    n_high = n_households - n_low - n_mid

    capture_low = np.random.uniform(0.3, 0.6, n_low)
    capture_mid = np.random.uniform(0.7, 1.1, n_mid)
    capture_high = np.random.uniform(1.2, 2.0, n_high)

    capture = np.concatenate([capture_low, capture_mid, capture_high])

    scarcity = np.clip(1.0 - re_ratio, 0.0, 1.0)
    effective_weight = 1.0 + scarcity * (capture - 1.0)
    allocation_shares = effective_weight / effective_weight.sum()

    benefit = n_households * re_ratio * allocation_shares
    benefit = np.clip(benefit, 0.0, 1.0)

    if benefit.sum() == 0:
        return 1.0

    return gini_coefficient(benefit)

def gini_theja_with_paper_multipliers(re_ratio, n_households=1220, seed=42):

    np.random.seed(seed)

    n_low = int(n_households * 0.40)
    n_mid = int(n_households * 0.40)
    n_high = n_households - n_low - n_mid

    capture_low = np.random.uniform(0.5, 1.0, n_low)
    capture_mid = np.random.uniform(0.8, 1.2, n_mid)
    capture_high = np.random.uniform(1.2, 2.0, n_high)

    capture = np.concatenate([capture_low, capture_mid, capture_high])

    scarcity = np.clip(1.0 - re_ratio, 0.0, 1.0)
    effective_weight = 1.0 + scarcity * (capture - 1.0)
    allocation_shares = effective_weight / effective_weight.sum()

    benefit = n_households * re_ratio * allocation_shares
    benefit = np.clip(benefit, 0.0, 1.0)

    if benefit.sum() == 0:
        return 1.0

    return gini_coefficient(benefit)

def gini_enhanced_scarcity(re_ratio, n_households=1220, seed=42, scarcity_power=2.0):

    np.random.seed(seed)

    n_low = int(n_households * 0.40)
    n_mid = int(n_households * 0.40)
    n_high = n_households - n_low - n_mid

    capture_low = np.random.uniform(0.5, 1.0, n_low)
    capture_mid = np.random.uniform(0.8, 1.2, n_mid)
    capture_high = np.random.uniform(1.2, 2.0, n_high)

    capture = np.concatenate([capture_low, capture_mid, capture_high])

    scarcity = np.clip(1.0 - re_ratio, 0.0, 1.0) ** scarcity_power
    effective_weight = 1.0 + scarcity * (capture - 1.0)
    allocation_shares = effective_weight / effective_weight.sum()

    benefit = n_households * re_ratio * allocation_shares
    benefit = np.clip(benefit, 0.0, 1.0)

    if benefit.sum() == 0:
        return 1.0

    return gini_coefficient(benefit)

print("=" * 80)
print("GINI MODEL COMPARISON ACROSS RE% RANGE")
print("=" * 80)

re_ratios = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

results = []
for re in re_ratios:
    results.append({
        'RE%': re * 100,
        'Paper Model': gini_paper_model(re),
        'Theja Scarcity (current)': gini_theja_scarcity(re),
        'Theja + Paper Mult': gini_theja_with_paper_multipliers(re),
        'Enhanced (power=2)': gini_enhanced_scarcity(re, scarcity_power=2.0),
        'Enhanced (power=3)': gini_enhanced_scarcity(re, scarcity_power=3.0),
    })

df = pd.DataFrame(results)
print("\nGini values by RE% and model:")
print(df.to_string(index=False))

print("\n" + "=" * 80)
print("GINI RANGES (for RE 30-50%, typical optimization range)")
print("=" * 80)

re_30_50 = [0.30, 0.35, 0.40, 0.45, 0.50]
for model in ['Paper Model', 'Theja Scarcity (current)', 'Theja + Paper Mult', 'Enhanced (power=2)', 'Enhanced (power=3)']:
    vals = [results[i][model] for i, r in enumerate(re_ratios) if r in re_30_50]
    print(f"{model:30s}: {min(vals):.4f} - {max(vals):.4f} (range: {max(vals)-min(vals):.4f})")

print("\n" + "=" * 80)
print("COMPARISON WITH PAPER CLAIMS")
print("=" * 80)
print(f"Paper states Gini range: 0.169 - 0.506 (range: 0.337)")
print(f"Run 101 actual:          0.138 - 0.188 (range: 0.050)")
print()

print("For RE range 10-90% (wider than Run 101's 31-48%):")
re_10_90 = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
for model in ['Paper Model', 'Theja Scarcity (current)', 'Theja + Paper Mult', 'Enhanced (power=2)']:
    vals = [results[i][model] for i, r in enumerate(re_ratios) if r in re_10_90]
    print(f"{model:30s}: {min(vals):.4f} - {max(vals):.4f} (range: {max(vals)-min(vals):.4f})")

print("\n" + "=" * 80)
print("SENSITIVITY ANALYSIS: Tier Percentages (R1 Request)")
print("=" * 80)

tier_configs = [
    ("40/40/20 (Theja)", 0.40, 0.40, 0.20),
    ("33/33/33 (Equal)", 0.33, 0.34, 0.33),
    ("50/30/20 (More low)", 0.50, 0.30, 0.20),
    ("30/50/20 (More mid)", 0.30, 0.50, 0.20),
    ("20/60/20 (Concentrated mid)", 0.20, 0.60, 0.20),
]

def gini_with_tiers(re_ratio, pct_low, pct_mid, pct_high, n_households=1220, seed=42):
    np.random.seed(seed)
    n_low = int(n_households * pct_low)
    n_mid = int(n_households * pct_mid)
    n_high = n_households - n_low - n_mid

    m_low = np.random.uniform(0.5, 1.0, n_low)
    m_mid = np.random.uniform(0.8, 1.2, n_mid)
    m_high = np.random.uniform(1.2, 2.0, n_high)

    multipliers = np.concatenate([m_low, m_mid, m_high])
    re_i = np.clip(multipliers * re_ratio, 0.0, 1.0)
    return gini_coefficient(re_i)

print(f"\n{'Config':<25} RE=30%   RE=40%   RE=50%   Range(30-50%)")
print("-" * 70)
for name, pl, pm, ph in tier_configs:
    g30 = gini_with_tiers(0.30, pl, pm, ph)
    g40 = gini_with_tiers(0.40, pl, pm, ph)
    g50 = gini_with_tiers(0.50, pl, pm, ph)
    print(f"{name:<25} {g30:.4f}   {g40:.4f}   {g50:.4f}   {max(g30,g40,g50)-min(g30,g40,g50):.4f}")

print("\n" + "=" * 80)
print("SENSITIVITY ANALYSIS: Multiplier Ranges (R1 Request)")
print("=" * 80)

mult_configs = [
    ("Paper (0.5-1, 0.8-1.2, 1.2-2)", (0.5, 1.0), (0.8, 1.2), (1.2, 2.0)),
    ("Wider (0.3-1, 0.7-1.3, 1.3-2.5)", (0.3, 1.0), (0.7, 1.3), (1.3, 2.5)),
    ("Narrower (0.7-1, 0.9-1.1, 1.1-1.5)", (0.7, 1.0), (0.9, 1.1), (1.1, 1.5)),
    ("Extreme (0.2-0.8, 0.6-1.4, 1.5-3.0)", (0.2, 0.8), (0.6, 1.4), (1.5, 3.0)),
]

def gini_with_mult_ranges(re_ratio, range_low, range_mid, range_high, n_households=1220, seed=42):
    np.random.seed(seed)
    n_low = int(n_households * 0.40)
    n_mid = int(n_households * 0.40)
    n_high = n_households - n_low - n_mid

    m_low = np.random.uniform(range_low[0], range_low[1], n_low)
    m_mid = np.random.uniform(range_mid[0], range_mid[1], n_mid)
    m_high = np.random.uniform(range_high[0], range_high[1], n_high)

    multipliers = np.concatenate([m_low, m_mid, m_high])
    re_i = np.clip(multipliers * re_ratio, 0.0, 1.0)
    return gini_coefficient(re_i)

print(f"\n{'Config':<40} RE=30%   RE=40%   RE=50%   Range")
print("-" * 80)
for name, rl, rm, rh in mult_configs:
    g30 = gini_with_mult_ranges(0.30, rl, rm, rh)
    g40 = gini_with_mult_ranges(0.40, rl, rm, rh)
    g50 = gini_with_mult_ranges(0.50, rl, rm, rh)
    print(f"{name:<40} {g30:.4f}   {g40:.4f}   {g50:.4f}   {max(g30,g40,g50)-min(g30,g40,g50):.4f}")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print("""
FINDINGS:
1. Paper Model (direct multiplication + clip) produces CONSTANT Gini ~0.19
   for RE 30-50% because clipping only activates when m_i × RE > 1.0
   (i.e., when RE > 50% for max multiplier 2.0)

2. Theja Scarcity Model produces VARIABLE Gini but narrow range (0.14-0.19)
   because scarcity effect is moderate at RE 30-50%

3. To get paper's claimed range (0.17-0.51), we need EITHER:
   a) Much wider RE% range (10-90%) in the Pareto front
   b) More extreme scarcity effect
   c) Different model altogether

4. The paper's stated range 0.169-0.506 is INCONSISTENT with RE range 31-48%
   using any reasonable multiplier model.

RECOMMENDATION:
- Option A: Re-run optimization allowing wider RE% range (0-100%)
- Option B: Update paper to reflect actual Gini range from Theja model
- Option C: Use enhanced scarcity (power=2) for wider range

For R1 response (sensitivity analysis):
- Document that Gini varies primarily with RE%, not tier configuration
- Show that 40/40/20 is justified by Theja 2025 (income stratification)
- Note that multiplier ranges follow Theja Eq.22 clipping [0.1, 2.0]
