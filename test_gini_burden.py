import sys
sys.path.insert(0, 'src')

import numpy as np
from objectives.objective_functions import objective_gini_burden

def test_gini_variation():

    print("=" * 70)
    print("TEST: Energy Burden Gini Variation")
    print("=" * 70)

    test_cases = [

        (7_000_000, 2_000_000, "High diesel, low capital (27% RE)"),
        (5_500_000, 3_500_000, "Medium diesel, medium capital (40% RE)"),
        (4_000_000, 5_000_000, "Low diesel, high capital (50% RE)"),
        (3_000_000, 6_000_000, "Very low diesel, very high capital (65% RE)"),
    ]

    results = []
    print("\n{:<45} {:>12} {:>12} {:>10}".format(
        "Scenario", "Fuel ($M)", "Capital ($M)", "Gini"))
    print("-" * 70)

    for fuel, capital, desc in test_cases:
        gini = objective_gini_burden(fuel, capital)
        results.append((desc, fuel/1e6, capital/1e6, gini))
        print("{:<45} {:>12.2f} {:>12.2f} {:>10.4f}".format(
            desc, fuel/1e6, capital/1e6, gini))

    print("-" * 70)

    ginis = [r[3] for r in results]
    tests_passed = 0
    total_tests = 3

    if ginis[0] > ginis[-1]:
        print("\n[PASS] Test 1: Gini decreases with more RE (high diesel -> low diesel)")
        tests_passed += 1
    else:
        print("\n[FAIL] Test 1: Gini should decrease with more RE")
        print(f"       Got: {ginis[0]:.4f} -> {ginis[-1]:.4f}")

    gini_range = max(ginis) - min(ginis)
    relative_range = gini_range / np.mean(ginis) if np.mean(ginis) > 0 else 0
    if relative_range > 0.10:
        print(f"[PASS] Test 2: Gini varies meaningfully (range = {gini_range:.4f}, {relative_range*100:.1f}% relative)")
        tests_passed += 1
    else:
        print(f"[FAIL] Test 2: Gini should vary more (range = {gini_range:.4f}, {relative_range*100:.1f}% relative)")

    if all(0.05 < g < 0.6 for g in ginis):
        print(f"[PASS] Test 3: Gini in reasonable range (0.05-0.6)")
        tests_passed += 1
    else:
        print(f"[FAIL] Test 3: Gini out of expected range")
        print(f"       Got: {min(ginis):.4f} - {max(ginis):.4f}")

    print("\n" + "=" * 70)
    print(f"RESULTS: {tests_passed}/{total_tests} tests passed")
    print("=" * 70)

    return tests_passed == total_tests

def test_sensitivity():

    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS: Fuel vs Capital")
    print("=" * 70)

    base_fuel = 5_000_000
    base_capital = 4_000_000

    print("\n1. Varying Fuel (Capital fixed at $4M/yr):")
    print("-" * 50)
    for fuel in [3_000_000, 5_000_000, 7_000_000, 9_000_000]:
        gini = objective_gini_burden(fuel, base_capital)
        print(f"   Fuel ${fuel/1e6:.1f}M -> Gini = {gini:.4f}")

    print("\n2. Varying Capital (Fuel fixed at $5M/yr):")
    print("-" * 50)
    for capital in [2_000_000, 4_000_000, 6_000_000, 8_000_000]:
        gini = objective_gini_burden(base_fuel, capital)
        print(f"   Capital ${capital/1e6:.1f}M -> Gini = {gini:.4f}")

def test_trade_off():

    print("\n" + "=" * 70)
    print("TRADE-OFF VERIFICATION: NPC vs Gini")
    print("=" * 70)

    low_npc_fuel = 6_500_000
    low_npc_capital = 2_500_000

    high_npc_fuel = 4_000_000
    high_npc_capital = 5_500_000

    gini_low_npc = objective_gini_burden(low_npc_fuel, low_npc_capital)
    gini_high_npc = objective_gini_burden(high_npc_fuel, high_npc_capital)

    print(f"\nLow NPC scenario (high diesel):")
    print(f"  Fuel: ${low_npc_fuel/1e6:.1f}M, Capital: ${low_npc_capital/1e6:.1f}M")
    print(f"  Gini = {gini_low_npc:.4f}")

    print(f"\nHigh NPC scenario (high RE):")
    print(f"  Fuel: ${high_npc_fuel/1e6:.1f}M, Capital: ${high_npc_capital/1e6:.1f}M")
    print(f"  Gini = {gini_high_npc:.4f}")

    if gini_low_npc > gini_high_npc:
        print(f"\n[CORRECT] Trade-off confirmed:")
        print(f"  Low NPC -> High Gini ({gini_low_npc:.4f})")
        print(f"  High NPC -> Low Gini ({gini_high_npc:.4f})")
        print(f"  Matches paper L259: 'High-equity configs require 10-15% higher capital'")
        return True
    else:
        print(f"\n[ERROR] Trade-off inverted! Check formula.")
        return False

if __name__ == "__main__":
    success = test_gini_variation()
    test_sensitivity()
    trade_off_ok = test_trade_off()

    print("\n" + "=" * 70)
    if success and trade_off_ok:
        print("ALL TESTS PASSED - Energy Burden Gini ready for NSGA-III")
    else:
        print("SOME TESTS FAILED - Review implementation")
    print("=" * 70)
