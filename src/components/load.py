import pandas as pd
import numpy as np

class LoadProfile:

    def __init__(self, csv_path):

        self.csv_path = csv_path

        df = pd.read_csv(csv_path)

        self.load_mw = df['Load_MW'].values

    def get_load(self, timestep):

        if timestep < 0 or timestep >= 8760:
            raise IndexError(f"Timestep {timestep} out of range [0, 8759]")

        return float(self.load_mw[timestep])

    def get_total_annual_energy(self):

        return float(self.load_mw.sum())

    def get_statistics(self):

        return {
            'mean': float(self.load_mw.mean()),
            'min': float(self.load_mw.min()),
            'max': float(self.load_mw.max()),
            'std': float(self.load_mw.std())
        }

    def __repr__(self):

        stats = self.get_statistics()
        return (f"LoadProfile(timesteps=8760, mean={stats['mean']:.2f}MW, "
                f"range=[{stats['min']:.2f}, {stats['max']:.2f}]MW)")
