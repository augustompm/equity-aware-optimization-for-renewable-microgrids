import pandas as pd
import numpy as np
from pathlib import Path

def calculate_solar_capacity_factors():
    project_root = Path(__file__).parent.parent.parent

    nsrdb_path = project_root / "dataset" / "raw" / "nrel-nsrdb-inuvik-2020.csv"

    df = pd.read_csv(nsrdb_path, skiprows=2)

    solar_data = []

    for idx in range(len(df)):
        year = int(df.loc[idx, 'Year'])
        month = int(df.loc[idx, 'Month'])
        day = int(df.loc[idx, 'Day'])
        hour = int(df.loc[idx, 'Hour'])

        ghi = df.loc[idx, 'GHI']
        dni = df.loc[idx, 'DNI']
        dhi = df.loc[idx, 'DHI']
        temperature = df.loc[idx, 'Temperature']

        poa_wm2 = ghi

        cf_pv = poa_wm2 / 1000.0
        cf_pv = min(cf_pv, 1.0)
        cf_pv = max(cf_pv, 0.0)

        solar_data.append({
            'Year': year,
            'Month': month,
            'Day': day,
            'Hour': hour,
            'GHI_Wm2': ghi,
            'DNI_Wm2': dni,
            'DHI_Wm2': dhi,
            'POA_Wm2': poa_wm2,
            'CF_pv': cf_pv,
            'T_ambient_C': temperature
        })

    solar_df = pd.DataFrame(solar_data)

    output_path = project_root / "data" / "solar-capacity-factors.csv"
    solar_df.to_csv(output_path, index=False, float_format='%.4f')

    return output_path, solar_df

if __name__ == "__main__":
    output_file, solar_df = calculate_solar_capacity_factors()

    print(f"Solar capacity factors processed successfully")
    print(f"Output: {output_file}")
    print(f"\nStatistics:")
    print(f"  Rows: {len(solar_df)}")
    print(f"  GHI mean: {solar_df['GHI_Wm2'].mean():.2f} W/m²")
    print(f"  GHI max: {solar_df['GHI_Wm2'].max():.2f} W/m²")
    print(f"  CF_pv mean: {solar_df['CF_pv'].mean():.4f}")
    print(f"  CF_pv max: {solar_df['CF_pv'].max():.4f}")

    winter_months = solar_df[solar_df['Month'].isin([12, 1, 2])]
    summer_months = solar_df[solar_df['Month'].isin([6, 7, 8])]

    winter_ghi = winter_months['GHI_Wm2'].mean()
    summer_ghi = summer_months['GHI_Wm2'].mean()

    winter_cf = winter_months['CF_pv'].mean()
    summer_cf = summer_months['CF_pv'].mean()

    print(f"\nSeasonal Variation (GHI):")
    print(f"  Winter (Dec-Feb) avg: {winter_ghi:.2f} W/m²")
    print(f"  Summer (Jun-Aug) avg: {summer_ghi:.2f} W/m²")
    if winter_ghi > 0:
        print(f"  Summer/Winter ratio: {summer_ghi / winter_ghi:.1f}×")

    print(f"\nSeasonal Variation (CF_pv):")
    print(f"  Winter (Dec-Feb) avg: {winter_cf:.4f}")
    print(f"  Summer (Jun-Aug) avg: {summer_cf:.4f}")
    if winter_cf > 0:
        print(f"  Summer/Winter ratio: {summer_cf / winter_cf:.1f}×")

    print(f"\nTemperature Range:")
    print(f"  Min: {solar_df['T_ambient_C'].min():.1f}°C")
    print(f"  Max: {solar_df['T_ambient_C'].max():.1f}°C")
