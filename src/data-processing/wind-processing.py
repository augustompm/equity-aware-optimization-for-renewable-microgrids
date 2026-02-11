import pandas as pd
import numpy as np
from pathlib import Path

def extrapolate_wind_speed(v_ref, h_ref, h_hub, alpha=0.17):
    v_hub = v_ref * (h_hub / h_ref) ** alpha
    return v_hub

def turbine_power_curve(v_hub, p_rated=3.5, v_cutin=3.0, v_rated=12.0, v_cutout=25.0):
    if v_hub < v_cutin:
        return 0.0
    elif v_hub < v_rated:
        power = p_rated * ((v_hub - v_cutin) / (v_rated - v_cutin)) ** 3
        return power
    elif v_hub < v_cutout:
        return p_rated
    else:
        return 0.0

def calculate_wind_capacity_factors():
    project_root = Path(__file__).parent.parent.parent

    nsrdb_path = project_root / "dataset" / "raw" / "nrel-nsrdb-inuvik-2020.csv"

    df = pd.read_csv(nsrdb_path, skiprows=2)

    wind_data = []

    h_ref = 10.0
    h_hub = 100.0
    alpha = 0.17
    p_rated = 3.5

    for idx in range(len(df)):
        year = int(df.loc[idx, 'Year'])
        month = int(df.loc[idx, 'Month'])
        day = int(df.loc[idx, 'Day'])
        hour = int(df.loc[idx, 'Hour'])

        v_10m = df.loc[idx, 'Wind Speed']

        v_100m = extrapolate_wind_speed(v_10m, h_ref, h_hub, alpha)

        power_mw = turbine_power_curve(v_100m, p_rated)

        cf_wind = power_mw / p_rated

        wind_data.append({
            'Year': year,
            'Month': month,
            'Day': day,
            'Hour': hour,
            'v_10m_ms': v_10m,
            'v_100m_ms': v_100m,
            'Power_MW': power_mw,
            'CF_wind': cf_wind
        })

    wind_df = pd.DataFrame(wind_data)

    output_path = project_root / "data" / "wind-capacity-factors.csv"
    wind_df.to_csv(output_path, index=False, float_format='%.4f')

    return output_path, wind_df

if __name__ == "__main__":
    output_file, wind_df = calculate_wind_capacity_factors()

    print(f"Wind capacity factors processed successfully")
    print(f"Output: {output_file}")
    print(f"\nStatistics:")
    print(f"  Rows: {len(wind_df)}")
    print(f"  v_10m mean: {wind_df['v_10m_ms'].mean():.2f} m/s")
    print(f"  v_100m mean: {wind_df['v_100m_ms'].mean():.2f} m/s")
    print(f"  v_10m max: {wind_df['v_10m_ms'].max():.2f} m/s")
    print(f"  v_100m max: {wind_df['v_100m_ms'].max():.2f} m/s")
    print(f"  CF_wind mean: {wind_df['CF_wind'].mean():.4f}")
    print(f"  CF_wind max: {wind_df['CF_wind'].max():.4f}")

    winter_months = wind_df[wind_df['Month'].isin([12, 1, 2])]
    summer_months = wind_df[wind_df['Month'].isin([6, 7, 8])]

    winter_cf = winter_months['CF_wind'].mean()
    summer_cf = summer_months['CF_wind'].mean()

    print(f"\nSeasonal Variation (CF_wind):")
    print(f"  Winter (Dec-Feb) avg: {winter_cf:.4f}")
    print(f"  Summer (Jun-Aug) avg: {summer_cf:.4f}")

    v_50m = wind_df['v_10m_ms'].mean() * (50.0 / 10.0) ** 0.17
    print(f"\nValidation vs CASES:")
    print(f"  Extrapolated to 50m: {v_50m:.2f} m/s")
    print(f"  CASES High Point @ 50m: 6.42 m/s (2015-2017)")
    print(f"  Difference: {abs(v_50m - 6.42):.2f} m/s")

    cut_in_hours = (wind_df['v_100m_ms'] >= 3.0).sum()
    rated_hours = (wind_df['v_100m_ms'] >= 12.0).sum()
    cut_out_hours = (wind_df['v_100m_ms'] >= 25.0).sum()

    print(f"\nTurbine Operating Hours:")
    print(f"  Above cut-in (3.0 m/s): {cut_in_hours} hours ({cut_in_hours/8760*100:.1f}%)")
    print(f"  At rated (>=12.0 m/s): {rated_hours} hours ({rated_hours/8760*100:.1f}%)")
    print(f"  Cut-out (>=25.0 m/s): {cut_out_hours} hours ({cut_out_hours/8760*100:.1f}%)")
