import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from constraints.constraint_functions import (
    constraint_bounds,
    constraint_area,
    constraint_lpsp,
    constraint_spinning_reserve,
    constraint_grid_limits,
)

def constraint_renewable_cap_v4(system, policy, enable_renewable_cap=False):

    if not enable_renewable_cap:
        return 0.0

    p_pv_kw = system.get('p_pv_installed_kw', 0.0)
    p_wind_mw = system.get('p_wind_installed_mw', 0.0)
    p_load_avg = system.get('p_load_avg_mw', 0.0)

    renewable_total_mw = (p_pv_kw / 1000.0) + p_wind_mw

    renewable_cap_mw = policy['renewable_fraction_max'] * p_load_avg

    violation = max(renewable_total_mw - renewable_cap_mw, 0.0)

    return violation

constraint_renewable_cap = constraint_renewable_cap_v4

if __name__ == "__main__":

    print("Testing constraint_functions_v4...")
    print()

    system = {'p_pv_installed_kw': 1000, 'p_wind_installed_mw': 10.0, 'p_load_avg_mw': 3.35}
    policy = {'renewable_fraction_max': 0.20}

    violation_v4 = constraint_renewable_cap_v4(system, policy, enable_renewable_cap=False)
    print(f"V4 (disabled): {violation_v4:.2f} MW violation")
    assert violation_v4 == 0.0, "V4 should return 0 when disabled"
    print("  PASS: Constraint disabled, allows >20% renewable")
    print()

    violation_v3 = constraint_renewable_cap_v4(system, policy, enable_renewable_cap=True)
    print(f"V3 (enabled): {violation_v3:.2f} MW violation")
    assert violation_v3 > 9.0, "V3 should enforce 20% cap"
    print("  PASS: Constraint enabled, enforces 20% cap")
    print()

    system_low = {'p_pv_installed_kw': 60, 'p_wind_installed_mw': 0.5, 'p_load_avg_mw': 3.35}
    violation_low_v4 = constraint_renewable_cap_v4(system_low, policy, enable_renewable_cap=False)
    violation_low_v3 = constraint_renewable_cap_v4(system_low, policy, enable_renewable_cap=True)

    print(f"Low renewable (0.56 MW < 0.67 MW cap):")
    print(f"  V4: {violation_low_v4:.2f} MW")
    print(f"  V3: {violation_low_v3:.2f} MW")
    assert violation_low_v4 == 0.0, "V4 always 0"
    assert violation_low_v3 == 0.0, "V3 passes within limit"
    print("  PASS: Both modes OK for configurations within 20%")
    print()

    print("All constraint_functions_v4 smoke tests passed!")
