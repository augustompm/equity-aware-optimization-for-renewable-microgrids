import json
from pathlib import Path

def extract_baseline_system():
    project_root = Path(__file__).parent.parent.parent

    baseline_system = {
        "location": {
            "name": "Inuvik",
            "province": "Northwest Territories",
            "country": "Canada",
            "latitude": 68.36,
            "longitude": -133.72,
            "timezone": "America/Edmonton",
            "population_approx": 3243,
            "classification": {
                "governance": "Decentralized CEC",
                "isolation": "Full Island (Physical + Political + Service)",
                "climate": "Arctic-Cold-Fuel",
                "business": "Cost-reduction + Community-investing"
            }
        },
        "baseline_2020": {
            "lng_generators": {
                "count": 3,
                "capacity_mw_each": 2.57,
                "total_capacity_mw": 7.71,
                "efficiency": 0.35,
                "fuel_cost_per_mmbtu": 12.5,
                "startup_time_h": 1.0,
                "min_load_fraction": 0.30,
                "commissioned_year": 1999,
                "fuel_source": "LNG trucked from southern Canada",
                "notes": "Converted from local Ikhil gas (1999-2012) to imported LNG (2013-present)"
            },
            "diesel_generators": {
                "count": 3,
                "capacity_mw_each": 1.91,
                "total_capacity_mw": 5.73,
                "efficiency": 0.30,
                "fuel_cost_per_mmbtu": 15.0,
                "startup_time_h": 0.5,
                "min_load_fraction": 0.20,
                "role": "Backup generation during LNG supply disruptions",
                "notes": "N-1 criteria: backup units 110% of primary unit size"
            },
            "solar_pv_existing": {
                "total_capacity_kw": 55,
                "installations": [
                    {
                        "location": "Aurora Research Institute",
                        "capacity_kw": 10,
                        "year_installed": "pre-2020"
                    },
                    {
                        "location": "17-unit apartment complex (NWTHC)",
                        "capacity_kw": 20,
                        "year_installed": 2017
                    },
                    {
                        "location": "Department of Infrastructure building",
                        "capacity_kw": 25,
                        "year_installed": "pre-2020"
                    }
                ],
                "notes": "All installations are government-owned (GNWT), grid-connected"
            },
            "wind_existing": {
                "capacity_mw": 0,
                "notes": "No wind generation in 2020 baseline. 3.5 MW installed in 2023 (validation target)."
            },
            "battery_existing": {
                "capacity_mwh": 0,
                "notes": "No utility-scale battery storage in 2020 baseline."
            },
            "biomass_existing": {
                "installations": [
                    {
                        "location": "East Three School",
                        "capacity_kw_thermal": 950,
                        "fuel_type": "Wood pellets",
                        "year_installed": 2018
                    },
                    {
                        "location": "Inuvik Regional Hospital",
                        "capacity_kw_thermal": 1250,
                        "fuel_type": "Wood pellets",
                        "year_installed": 2019
                    }
                ],
                "notes": "Biomass for heating only, not electricity generation"
            },
            "load": {
                "average_mw": 3.35,
                "peak_mw_estimated": 5.0,
                "annual_consumption_mwh": 29346,
                "notes": "Peak estimated from N-1 criteria and diesel backup capacity"
            },
            "wind_resource_potential": {
                "high_point_site": {
                    "average_speed_ms": 6.42,
                    "measurement_height_m": 50,
                    "measurement_period": "2015-2017",
                    "distance_from_town_km": 5,
                    "distance_from_transmission_km": 10,
                    "notes": "50m tower installed 2017. Good wind potential confirmed."
                }
            }
        },
        "constraints": {
            "renewable_penetration_cap_fraction": 0.20,
            "renewable_penetration_cap_mw": 0.67,
            "calculation": "0.20 × 3.35 MW average load = 0.67 MW",
            "area_available_m2": 10000,
            "grid_connected": False,
            "transmission_infrastructure": "None. Isolated island system.",
            "notes": "20% renewable cap enforced by NTPC to protect grid stability and avoid outages"
        },
        "policy": {
            "renewable_cap_source": "NTPC 2018 Policy",
            "renewable_cap_reason": "Grid stability, avoid outages from inefficient generator cycling",
            "nwt_carbon_tax_per_tonne_co2": 20,
            "carbon_tax_increase_schedule": "Increases annually to $50/tonne by 2022",
            "energy_strategy": "2030 Energy Strategy (GNWT 2019)",
            "climate_framework": "2030 NWT Climate Change Strategic Framework"
        },
        "costs_2020": {
            "lng_fuel_cost_per_mmbtu": 12.5,
            "diesel_fuel_cost_per_mmbtu": 15.0,
            "electricity_rate_residential_per_kwh": "Not specified in CASES document",
            "notes": "Fuel costs do not include carbon tax rebates for utilities"
        },
        "data_sources": {
            "cases_profile": "yaml/cases_energy_profile_inuvik.yaml",
            "nrel_nsrdb": "dataset/raw/nrel-nsrdb-inuvik-2020.csv",
            "validation_target": "NTPC 3.5 MW wind installation (2023)",
            "baseline_year": 2020,
            "rationale": "2020 = normal year, complete data, pre-wind installation"
        },
        "metadata": {
            "extracted_date": "2025-11-03",
            "extractor_version": "1.0",
            "schema_version": "1.0"
        }
    }

    output_path = project_root / "data" / "baseline-system.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(baseline_system, f, indent=2, ensure_ascii=False)

    return output_path

if __name__ == "__main__":
    output_file = extract_baseline_system()
    print(f"Baseline system extracted successfully")
    print(f"Output: {output_file}")
    print(f"Size: {output_file.stat().st_size} bytes")
