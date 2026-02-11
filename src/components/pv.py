class SolarPV:

    def __init__(self, capacity_kw, tilt_deg=60, temp_coeff=-0.004, name="SolarPV"):

        self.capacity_kw = capacity_kw
        self.tilt_deg = tilt_deg
        self.temp_coeff = temp_coeff
        self.name = name

    def generate(self, cf_pv, temperature_c):

        p_stc_kw = self.capacity_kw * cf_pv

        temperature_reference_c = 25.0
        derating_factor = 1.0 + self.temp_coeff * (temperature_c - temperature_reference_c)

        p_derated_kw = p_stc_kw * derating_factor

        p_output_mw = max(p_derated_kw, 0.0) / 1000.0

        return p_output_mw

    def __repr__(self):

        return (f"SolarPV(name='{self.name}', capacity={self.capacity_kw}kW, "
                f"tilt={self.tilt_deg}°, temp_coeff={self.temp_coeff}/°C)")
