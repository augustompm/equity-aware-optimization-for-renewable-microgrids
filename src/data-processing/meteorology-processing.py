import pandas as pd
from pathlib import Path

def extract_meteorology():
    project_root = Path(__file__).parent.parent.parent

    nsrdb_path = project_root / "dataset" / "raw" / "nrel-nsrdb-inuvik-2020.csv"

    df = pd.read_csv(nsrdb_path, skiprows=2)

    meteorology_data = []

    for idx in range(len(df)):
        year = int(df.loc[idx, 'Year'])
        month = int(df.loc[idx, 'Month'])
        day = int(df.loc[idx, 'Day'])
        hour = int(df.loc[idx, 'Hour'])

        temperature_c = df.loc[idx, 'Temperature']
        pressure_mbar = df.loc[idx, 'Pressure']

        meteorology_data.append({
            'Year': year,
            'Month': month,
            'Day': day,
            'Hour': hour,
            'T_ambient_C': temperature_c,
            'Pressure_mbar': pressure_mbar
        })

    met_df = pd.DataFrame(meteorology_data)

    output_path = project_root / "data" / "meteorology-8760h.csv"
    met_df.to_csv(output_path, index=False, float_format='%.2f')

    return output_path, met_df

if __name__ == "__main__":
    output_file, met_df = extract_meteorology()

    print(f"Meteorology data extracted successfully")
    print(f"Output: {output_file}")
    print(f"\nStatistics:")
    print(f"  Rows: {len(met_df)}")

    print(f"\nTemperature (°C):")
    print(f"  Mean: {met_df['T_ambient_C'].mean():.2f}°C")
    print(f"  Std Dev: {met_df['T_ambient_C'].std():.2f}°C")
    print(f"  Min: {met_df['T_ambient_C'].min():.1f}°C")
    print(f"  Max: {met_df['T_ambient_C'].max():.1f}°C")

    winter_months = met_df[met_df['Month'].isin([12, 1, 2])]
    summer_months = met_df[met_df['Month'].isin([6, 7, 8])]

    winter_temp = winter_months['T_ambient_C'].mean()
    summer_temp = summer_months['T_ambient_C'].mean()

    print(f"\nSeasonal Temperature:")
    print(f"  Winter (Dec-Feb) avg: {winter_temp:.1f}°C")
    print(f"  Summer (Jun-Aug) avg: {summer_temp:.1f}°C")
    print(f"  Seasonal range: {summer_temp - winter_temp:.1f}°C")

    print(f"\nPressure (mbar):")
    print(f"  Mean: {met_df['Pressure_mbar'].mean():.1f} mbar")
    print(f"  Std Dev: {met_df['Pressure_mbar'].std():.1f} mbar")
    print(f"  Min: {met_df['Pressure_mbar'].min():.1f} mbar")
    print(f"  Max: {met_df['Pressure_mbar'].max():.1f} mbar")

    R_air = 287.05
    T_kelvin_mean = met_df['T_ambient_C'].mean() + 273.15
    P_pascal_mean = met_df['Pressure_mbar'].mean() * 100

    rho_air = P_pascal_mean / (R_air * T_kelvin_mean)

    T_cold = winter_temp + 273.15
    P_cold = met_df['Pressure_mbar'].mean() * 100
    rho_cold = P_cold / (R_air * T_cold)

    density_increase = (rho_cold - rho_air) / rho_air * 100

    print(f"\nAir Density (for wind power correction):")
    print(f"  Mean density: {rho_air:.3f} kg/m³")
    print(f"  Winter density: {rho_cold:.3f} kg/m³")
    print(f"  Cold weather increase: {density_increase:.1f}%")

    below_freezing_hours = (met_df['T_ambient_C'] < 0).sum()
    below_minus_20_hours = (met_df['T_ambient_C'] < -20).sum()

    print(f"\nArctic Characteristics:")
    print(f"  Below freezing (0°C): {below_freezing_hours} hours ({below_freezing_hours/8760*100:.1f}%)")
    print(f"  Below -20°C: {below_minus_20_hours} hours ({below_minus_20_hours/8760*100:.1f}%)")
