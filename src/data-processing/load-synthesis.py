import pandas as pd
import numpy as np
import json
from pathlib import Path

def get_daily_shape_multiplier(hour):
    if 6 <= hour < 9:
        return 1.15
    elif 9 <= hour < 17:
        return 1.0
    elif 17 <= hour < 22:
        return 1.25
    else:
        return 0.75

def get_seasonal_multiplier(month):
    if month in [11, 12, 1, 2, 3]:
        return 1.20
    elif month in [6, 7, 8]:
        return 0.85
    else:
        return 1.0

def synthesize_load_profile(target_average_mw=3.35, random_seed=42):
    np.random.seed(random_seed)

    project_root = Path(__file__).parent.parent.parent

    baseline_path = project_root / "data" / "baseline-system.json"
    with open(baseline_path, 'r', encoding='utf-8') as f:
        baseline = json.load(f)

    nsrdb_path = project_root / "dataset" / "raw" / "nrel-nsrdb-inuvik-2020.csv"
    nsrdb_df = pd.read_csv(nsrdb_path, skiprows=2)

    load_data = []

    day_variability = np.random.normal(1.0, 0.10, 366)

    for idx in range(8760):
        year = 2020
        month = nsrdb_df.loc[idx, 'Month']
        day = nsrdb_df.loc[idx, 'Day']
        hour = nsrdb_df.loc[idx, 'Hour']
        temperature = nsrdb_df.loc[idx, 'Temperature']

        day_of_year = idx // 24

        base_load = 1.0

        daily_shape = get_daily_shape_multiplier(hour)

        seasonal = get_seasonal_multiplier(month)

        heating_threshold = 15.0
        heating_sensitivity = 0.01
        if temperature < heating_threshold:
            heating_load = (heating_threshold - temperature) * heating_sensitivity
        else:
            heating_load = 0.0

        day_var = day_variability[day_of_year]

        timestep_var = np.random.normal(1.0, 0.05)

        load_raw = base_load * daily_shape * seasonal * (1 + heating_load) * day_var * timestep_var

        load_data.append({
            'Year': year,
            'Month': month,
            'Day': day,
            'Hour': hour,
            'Load_MW_raw': load_raw
        })

    load_df = pd.DataFrame(load_data)

    current_mean = load_df['Load_MW_raw'].mean()
    scale_factor = target_average_mw / current_mean

    load_df['Load_MW'] = load_df['Load_MW_raw'] * scale_factor

    load_df = load_df.drop(columns=['Load_MW_raw'])

    output_path = project_root / "data" / "load-profile-8760h.csv"
    load_df.to_csv(output_path, index=False, float_format='%.4f')

    return output_path, load_df

if __name__ == "__main__":
    output_file, load_df = synthesize_load_profile()

    print(f"Load profile synthesized successfully")
    print(f"Output: {output_file}")
    print(f"\nStatistics:")
    print(f"  Rows: {len(load_df)}")
    print(f"  Mean: {load_df['Load_MW'].mean():.4f} MW")
    print(f"  Std Dev: {load_df['Load_MW'].std():.4f} MW")
    print(f"  Min: {load_df['Load_MW'].min():.4f} MW")
    print(f"  Max: {load_df['Load_MW'].max():.4f} MW")
    print(f"  Peak/Average: {load_df['Load_MW'].max() / load_df['Load_MW'].mean():.2f}")

    winter_months = load_df[load_df['Month'].isin([12, 1, 2])]['Load_MW'].mean()
    summer_months = load_df[load_df['Month'].isin([6, 7, 8])]['Load_MW'].mean()
    print(f"\nSeasonal Variation:")
    print(f"  Winter (Dec-Feb) avg: {winter_months:.4f} MW")
    print(f"  Summer (Jun-Aug) avg: {summer_months:.4f} MW")
    print(f"  Winter/Summer ratio: {winter_months / summer_months:.2f}")
