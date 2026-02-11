import numpy as np
import pandas as pd
from pathlib import Path

class DataCache:

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if DataCache._initialized:
            return

        self.load_mw = None
        self.solar_cf = None
        self.wind_cf = None
        self.temperature_c = None
        self._config_hash = None
        DataCache._initialized = True

    def initialize(self, system_config):

        config_hash = hash(str(system_config.get('load_profile_path', '')))

        if self._config_hash == config_hash and self.load_mw is not None:
            return

        load_df = pd.read_csv(system_config['load_profile_path'])
        self.load_mw = load_df['Load_MW'].values.astype(np.float64)

        solar_df = pd.read_csv(system_config['solar_cf_path'])
        self.solar_cf = solar_df['CF_pv'].values.astype(np.float64)
        self.temperature_c = solar_df['T_ambient_C'].values.astype(np.float64)

        wind_df = pd.read_csv(system_config['wind_cf_path'])
        self.wind_cf = wind_df['CF_wind'].values.astype(np.float64)

        self._config_hash = config_hash

        print(f"[DataCache] Loaded: load={len(self.load_mw)}h, solar={len(self.solar_cf)}h, wind={len(self.wind_cf)}h")

    def get_arrays(self):

        return self.load_mw, self.solar_cf, self.wind_cf, self.temperature_c

_cache = DataCache()

def get_data_cache():

    return _cache
