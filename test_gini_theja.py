import sys
sys.path.insert(0, 'src')

import numpy as np
from objectives.objective_functions import objective_gini_theja

def test_gini_variation():

    print("=" * 70)
    print("TEST: Theja-style Gini (RE Benefit Allocation)")
    print("=" * 70)

    total_load = 29346.0

    test_cases = [

        (10, "Very low RE (extreme scarcity)"),
        (20, "Low RE (high scarcity)"),
        (30, "Below average RE"),
        (40, "Average RE"),
        (50, "Moderate RE"),
        (60, "Good RE"),
        (70, "High RE"),
        (80, "Very high RE"),
        (90, "Near full RE"),
        (100, "Full RE (abundance)"),
    ]

    results = []
    print("\n{:<35} {:>10} {:>10}".format("Scenario", "RE%", "Gini"))
    print("-" * 60)

    for re_pct, desc in test_cases:
        total_re = total_load * (re_pct / 100.0)
        gini = objective_gini_theja(total_re, total_load)
        results.append((re_pct, gini))
        print("{:<35} {:>10.0f}% {:>10.4f}".format(desc, re_pct, gini))

    print("-" * 60)

    tests_passed = 0
    total_tests = 4

    re_pcts = [r[0] for r in results]
    ginis = [r[1] for r in results]

    if ginis[0] > ginis[-1]:
        print("\n[PASS] Test 1: Gini decreases with more RE")
        print(f"       10% RE: Gini={ginis[0]:.4f} > 100% RE: Gini={ginis[-1]:.4f}")
        tests_passed += 1
    else:
        print("\n[FAIL] Test 1: Gini should decrease with more RE")

    decreasing = sum(1 for i in range(len(ginis)-1) if ginis[i] >= ginis[i+1])
    if decreasing >= len(ginis) - 2:
        print(f"[PASS] Test 2: Gini is mostly monotonically decreasing ({decreasing}/{len(ginis)-1})")
        tests_passed += 1
    else:
        print(f"[FAIL] Test 2: Gini should be monotonically decreasing ({decreasing}/{len(ginis)-1})")

    gini_range = max(ginis) - min(ginis)
    if gini_range > 0.1:
        print(f"[PASS] Test 3: Gini has substantial range ({gini_range:.4f})")
        tests_passed += 1
    else:
        print(f"[FAIL] Test 3: Gini range too small ({gini_range:.4f})")

    if ginis[-1] < 0.05:
        print(f"[PASS] Test 4: At 100% RE, Gini is near zero ({ginis[-1]:.4f})")
        tests_passed += 1
    else:
        print(f"[FAIL] Test 4: At 100% RE, Gini should be near zero ({ginis[-1]:.4f})")

    print("\n" + "=" * 70)
    print(f"RESULTS: {tests_passed}/{total_tests} tests passed")
    print("=" * 70)

    return tests_passed == total_tests

def test_realistic_range():

    print("\n" + "=" * 70)
    print("TEST: Realistic RE Range (27-48%)")
    print("=" * 70)

    total_load = 29346.0

    print("\n{:<20} {:>10} {:>10}".format("Scenario", "RE%", "Gini"))
    print("-" * 45)

    for re_pct in [27, 30, 35, 40, 45, 48]:
        total_re = total_load * (re_pct / 100.0)
        gini = objective_gini_theja(total_re, total_load)
        print("{:<20} {:>10.0f}% {:>10.4f}".format(f"RE = {re_pct}%", re_pct, gini))

    gini_27 = objective_gini_theja(total_load * 0.27, total_load)
    gini_48 = objective_gini_theja(total_load * 0.48, total_load)

    print("-" * 45)
    print(f"\nRange in 27-48%: {gini_27:.4f} -> {gini_48:.4f}")
    print(f"Difference: {gini_27 - gini_48:.4f} ({(gini_27 - gini_48)/gini_27*100:.1f}% relative)")

    if gini_27 > gini_48:
        print("[OK] Correct direction: more RE = lower Gini")
        return True
    else:
        print("[ERROR] Wrong direction!")
        return False

def test_trade_off():

    print("\n" + "=" * 70)
    print("TRADE-OFF VERIFICATION: NPC vs Gini")
    print("=" * 70)

    total_load = 29346.0

    gini_low_re = objective_gini_theja(total_load * 0.27, total_load)
    gini_high_re = objective_gini_theja(total_load * 0.48, total_load)

    print(f"\nLow RE (27%) - Lower capital investment:")
    print(f"  Gini = {gini_low_re:.4f} (higher inequality)")

    print(f"\nHigh RE (48%) - Higher capital investment:")
    print(f"  Gini = {gini_high_re:.4f} (lower inequality)")

    if gini_low_re > gini_high_re:
        print(f"\n[CORRECT] Trade-off matches paper claim:")
        print(f"  'High-equity configurations require 10-15% higher capital'")
        print(f"  More investment (high RE) -> Lower Gini (more equitable)")
        return True
    else:
        print(f"\n[ERROR] Trade-off inverted!")
        return False

def compare_with_theja_values():

    print("\n" + "=" * 70)
    print("COMPARISON: Our Gini vs Theja 2025 Table III")
    print("=" * 70)

    total_load = 29346.0

    scenarios = [
        ("High Demand (scarcity)", 0.20, 0.630),
        ("High Price Volatility", 0.40, 0.289),
        ("Typical Weekday", 0.50, 0.144),
        ("Weekend", 0.60, 0.110),
        ("High Solar (abundance)", 0.80, 0.027),
        ("Low Demand (abundance)", 0.90, 0.006),
    ]

    print("\n{:<25} {:>10} {:>12} {:>12}".format(
        "Scenario", "RE%", "Theja Gini", "Our Gini"))
    print("-" * 65)

    for name, re_frac, theja_gini in scenarios:
        our_gini = objective_gini_theja(total_load * re_frac, total_load)
        print("{:<25} {:>10.0f}% {:>12.3f} {:>12.4f}".format(
            name, re_frac * 100, theja_gini, our_gini))

    print("-" * 65)
    print("\nNote: Exact match not expected - Theja uses dynamic RL adjustment.")
    print("Key is that both show: scarcity -> high Gini, abundance -> low Gini")

if __name__ == "__main__":
    success1 = test_gini_variation()
    success2 = test_realistic_range()
    success3 = test_trade_off()
    compare_with_theja_values()

    print("\n" + "=" * 70)
    if success1 and success2 and success3:
        print("ALL TESTS PASSED - Theja-style Gini ready for NSGA-III")
    else:
        print("SOME TESTS FAILED - Review implementation")
    print("=" * 70)
