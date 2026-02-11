import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from constraints.constraint_functions_v4 import (
    constraint_bounds,
    constraint_lpsp,
    constraint_spinning_reserve,
    constraint_grid_limits,
    constraint_renewable_cap_v4,
)

def constraint_area_v6(x, area_params, use_separated_areas=True):

    area_pv = x.get('n_pv_kw', 0) * area_params['area_pv_per_kw']
    area_wind = x.get('n_wind_mw', 0) * area_params['area_wind_per_mw']
    area_battery = x.get('e_battery_mwh', 0) * area_params['area_battery_per_mwh']

    if use_separated_areas:

        area_pv_total = area_pv + area_battery
        area_pv_limit = area_params['area_available_pv_m2']
        violation_pv = max(area_pv_total - area_pv_limit, 0.0)

        area_wind_limit = area_params['area_available_wind_m2']
        violation_wind = max(area_wind - area_wind_limit, 0.0)

        violation = violation_pv + violation_wind

    else:

        area_total = area_pv + area_wind + area_battery
        area_limit = area_params.get('area_available_m2', 500000.0)
        violation = max(area_total - area_limit, 0.0)

    return violation

constraint_area = constraint_area_v6

def get_v6_constraint_list():

    return [
        constraint_bounds,
        constraint_area_v6,
        constraint_lpsp,
        constraint_spinning_reserve,
        constraint_grid_limits,
        constraint_renewable_cap_v4,
    ]

if __name__ == "__main__":
    print("=" * 80)
    print("CONSTRAINT FUNCTIONS V6: SEPARATED AREA CONSTRAINTS")
    print("=" * 80)
    print()
    print("KEY CHANGE: Area constraint now separates PV (urban) and Wind (remote)")
    print()
    print("GEOGRAPHIC JUSTIFICATION:")
    print("  - PV: Urban/near-town area (500,000 m²)")
    print("    Locations: Rooftops, parking lots, brownfield")
    print("    Reference: LBNL 2022 (13,090 m²/MW for Arctic fixed-tilt)")
    print()
    print("  - Wind: Remote monitoring site (3,000,000 m²)")
    print("    Location: CASES High Point, 7 km from town, elevated")
    print("    Reference: CASES 2020 (yaml/cases_energy_profile_inuvik.yaml:354)")
    print()
    print("  - Battery: Co-located with PV (urban area)")
    print("    Area: ~10 m²/MWh (minimal footprint)")
    print()
    print("=" * 80)
    print("IMPACT ON OPTIMIZATION:")
    print("=" * 80)
    print()
    print("V1-V5 (Shared Constraint):")
    print("  - Total area: 500,000 m²")
    print("  - PV 10 MW uses: 20,000 m²")
    print("  - Wind 5 MW uses: 930,250 m²")
    print("  - Total: 950,250 m² > 500k = VIOLATION 450k m²")
    print("  - Result: Wind blocked even with good CF")
    print()
    print("V6 (Separated Constraints):")
    print("  - PV area: 500,000 m²")
    print("    - PV 10 MW uses: 20,000 m² < 500k OK")
    print("  - Wind area: 3,000,000 m²")
    print("    - Wind 5 MW uses: 930,250 m² < 3M OK")
    print("  - Result: BOTH viable, no artificial competition")
    print()
    print("=" * 80)
    print("EXPECTED OUTCOMES:")
    print("=" * 80)
    print()
    print("With separated areas + corrected wind CF (30%):")
    print("  - Wind becomes economically competitive")
    print("  - Solutions match Quitoras (2020): Wind dominant or equal PV")
    print("  - RE penetration: 50-80% (vs 3.6% in V5)")
    print("  - CO2 reduction: 15-25% (vs 7.3% in V5)")
    print("  - Pareto size: 40-80 solutions (vs 22 in V5)")
    print()
    print("References:")
    print("  - yaml/cases_energy_profile_inuvik.yaml:354")
    print("  - yaml/exploring_electricity_generation_alternatives_for_canadian_arctic.yaml")
    print("  - CORRECTION-ARCTIC-LIMITATION-CLAIM.md")
    print("  - ACTION-PLAN-CORRECT-DATA.md")
    print()
    print("=" * 80)
