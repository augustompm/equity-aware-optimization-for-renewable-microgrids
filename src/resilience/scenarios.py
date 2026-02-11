from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

@dataclass
class ScenarioParameters:

    name: str
    duration_days: int
    temperature_override: Optional[float] = None
    load_multiplier: float = 1.0
    pv_multiplier: float = 1.0
    wind_multiplier: float = 1.0
    fuel_limit_mmbtu: Optional[float] = None

    @classmethod
    def cold_snap(cls) -> 'ScenarioParameters':

        return cls(
            name='A1_cold_snap',
            duration_days=7,
            temperature_override=-40.0,
            load_multiplier=1.30
        )

    @classmethod
    def fuel_disruption(cls) -> 'ScenarioParameters':

        return cls(
            name='A2_fuel_disruption',
            duration_days=14,
            fuel_limit_mmbtu=5000.0
        )

    @classmethod
    def blizzard(cls) -> 'ScenarioParameters':

        return cls(
            name='A3_blizzard',
            duration_days=3,
            pv_multiplier=0.20,
            wind_multiplier=0.50,
            load_multiplier=1.20
        )

def apply_cold_snap_scenario(
    load_profile: np.ndarray,
    temperature_profile: np.ndarray,
    start_day: int,
    duration_days: int = 7
) -> Tuple[np.ndarray, np.ndarray]:

    if start_day < 0 or start_day > 364:
        raise ValueError(f"start_day must be 0-364, got {start_day}")

    load_modified = load_profile.copy()
    temperature_modified = temperature_profile.copy()

    start_hour = start_day * 24
    end_hour = min(start_hour + (duration_days * 24), 8760)

    load_modified[start_hour:end_hour] *= 1.30
    temperature_modified[start_hour:end_hour] = -40.0

    return load_modified, temperature_modified

def apply_fuel_disruption_scenario(
    fuel_reserves_mmbtu: float,
    duration_days: int = 14
) -> float:

    if fuel_reserves_mmbtu <= 0:
        raise ValueError(f"fuel_reserves_mmbtu must be positive, got {fuel_reserves_mmbtu}")

    total_hours = duration_days * 24
    fuel_limit_per_hour = fuel_reserves_mmbtu / total_hours

    return fuel_limit_per_hour

def apply_blizzard_scenario(
    pv_cf: np.ndarray,
    wind_cf: np.ndarray,
    load_profile: np.ndarray,
    start_day: int,
    duration_days: int = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    if start_day < 0 or start_day > 364:
        raise ValueError(f"start_day must be 0-364, got {start_day}")

    pv_cf_modified = pv_cf.copy()
    wind_cf_modified = wind_cf.copy()
    load_modified = load_profile.copy()

    start_hour = start_day * 24
    end_hour = min(start_hour + (duration_days * 24), 8760)

    pv_cf_modified[start_hour:end_hour] *= 0.20
    wind_cf_modified[start_hour:end_hour] *= 0.50
    load_modified[start_hour:end_hour] *= 1.20

    return pv_cf_modified, wind_cf_modified, load_modified
