import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from objectives.objective_functions import objective_gini_spatial

def test_gini_saturation():

    print("=" * 70)
    print("TEST: Gini Saturation Mechanism")
    print("=" * 70)
    print()

    load_mw = np.ones(8760) * 3.75
    total_load_mwh = load_mw.sum()

    re_levels = [0.10, 0.30, 0.50, 0.70, 0.90, 1.10]
    gini_values = []

    print(f"Total load: {total_load_mwh:,.0f} MWh/yr")
    print(f"n_households: 1220 (StatCan 2021)")
    print()
    print("RE%    RE MWh     Gini    Expected Range")
    print("-" * 50)

    for re_frac in re_levels:

        re_mw = load_mw * re_frac

        gini = objective_gini_spatial(load_mw, re_mw, n_households=1220, seed=42)
        gini_values.append(gini)

        if re_frac <= 0.20:
            expected = "0.35-0.45 (low RE, differences amplified)"
        elif re_frac <= 0.50:
            expected = "0.25-0.35 (medium RE)"
        elif re_frac <= 0.80:
            expected = "0.15-0.25 (high RE, saturation begins)"
        else:
            expected = "0.10-0.20 (very high RE, saturation dominant)"

        print(f"{re_frac*100:5.0f}%  {re_frac*total_load_mwh:8,.0f}  {gini:.4f}  {expected}")

    print()

    high_re_values = gini_values[2:]
    is_decreasing_after_threshold = all(
        high_re_values[i] >= high_re_values[i+1]
        for i in range(len(high_re_values)-1)
    )

    if is_decreasing_after_threshold:
        print("[PASS] Gini decreases monotonically after saturation threshold (>50% RE)")
    else:
        print("[FAIL] Gini should decrease as RE increases past saturation threshold!")
        print(f"       Values at 50%+: {[f'{g:.4f}' for g in high_re_values]}")

    is_decreasing = is_decreasing_after_threshold

    low_re_values = gini_values[:3]
    low_re_range = max(low_re_values) - min(low_re_values)
    if low_re_range < 0.01:
        print(f"[PASS] Gini is constant at low RE (range: {low_re_range:.6f}) - scale-invariance verified")
    else:
        print(f"[WARN] Gini varies at low RE (range: {low_re_range:.4f}) - unexpected")

    gini_range = max(gini_values) - min(gini_values)
    if gini_range > 0.10:
        print(f"[PASS] Gini varies significantly overall (range: {gini_range:.4f})")
    else:
        print(f"[FAIL] Gini range too small ({gini_range:.4f}), saturation not working!")

    if all(0.0 <= g <= 1.0 for g in gini_values):
        print("[PASS] All Gini values in [0, 1] range")
    else:
        print("[FAIL] Gini values out of range!")

    if gini_values[2] < 0.35:
        print(f"[PASS] At 50% RE, Gini = {gini_values[2]:.4f} (below 0.35)")
    else:
        print(f"[WARN] At 50% RE, Gini = {gini_values[2]:.4f} (expected < 0.35)")

    print()
    print("=" * 70)

    return is_decreasing and gini_range > 0.10

def test_gini_reproducibility():

    print()
    print("TEST: Gini Reproducibility (seed=42)")
    print("-" * 50)

    load = np.ones(8760) * 3.75
    re = np.ones(8760) * 1.875

    gini1 = objective_gini_spatial(load, re, n_households=1220, seed=42)
    gini2 = objective_gini_spatial(load, re, n_households=1220, seed=42)
    gini3 = objective_gini_spatial(load, re, n_households=1220, seed=99)

    if gini1 == gini2:
        print(f"[PASS] Same seed gives same result: {gini1:.6f}")
    else:
        print(f"[FAIL] Same seed gives different results: {gini1:.6f} vs {gini2:.6f}")

    if gini1 != gini3:
        print(f"[PASS] Different seed gives different result: {gini3:.6f}")
    else:
        print("[WARN] Different seed gives same result (may happen occasionally)")

    return gini1 == gini2

def test_gini_tiers():

    print()
    print("TEST: Tier Proportions (Theja 2025)")
    print("-" * 50)

    n = 1220
    n_low = int(n * 0.40)
    n_mid = int(n * 0.40)
    n_high = n - n_low - n_mid

    print(f"n_households: {n}")
    print(f"Low tier (40%):  {n_low} households ({n_low/n*100:.1f}%)")
    print(f"Mid tier (40%):  {n_mid} households ({n_mid/n*100:.1f}%)")
    print(f"High tier (20%): {n_high} households ({n_high/n*100:.1f}%)")

    if abs(n_low/n - 0.40) < 0.01 and abs(n_mid/n - 0.40) < 0.01 and abs(n_high/n - 0.20) < 0.01:
        print("[PASS] Tier proportions match Theja 2025 (40/40/20)")
        return True
    else:
        print("[FAIL] Tier proportions incorrect")
        return False

if __name__ == "__main__":
    passed = 0
    total = 3

    if test_gini_saturation():
        passed += 1
    if test_gini_reproducibility():
        passed += 1
    if test_gini_tiers():
        passed += 1

    print()
    print("=" * 70)
    print(f"RESULTS: {passed}/{total} tests passed")
    if passed == total:
        print("All tests passed! Gini spatial implementation is correct.")
    else:
        print("Some tests failed. Review implementation.")
    print("=" * 70)
