class Battery:

    def __init__(self, capacity_mwh, c_rate=0.25, efficiency=0.90,
                 dod_max=0.80, soc_initial=0.50, name="Battery"):

        self.capacity_mwh = capacity_mwh
        self.c_rate = c_rate
        self.p_max_mw = capacity_mwh * c_rate
        self.efficiency = efficiency
        self.dod_max = dod_max
        self.soc = soc_initial
        self.name = name

    def charge(self, power_mw, duration_h):

        power_actual = min(power_mw, self.p_max_mw)

        energy_in_mwh = power_actual * duration_h

        energy_stored_mwh = energy_in_mwh * self.efficiency

        soc_increase = energy_stored_mwh / self.capacity_mwh
        self.soc = self.soc + soc_increase

        self.soc = min(self.soc, 1.0)

        return power_actual

    def discharge(self, power_mw, duration_h):

        power_requested = min(power_mw, self.p_max_mw)

        energy_out_mwh = power_requested * duration_h

        energy_from_battery_mwh = energy_out_mwh / self.efficiency

        available_energy = self.get_available_energy()

        if energy_from_battery_mwh > available_energy:

            energy_from_battery_mwh = available_energy

            energy_out_mwh = energy_from_battery_mwh * self.efficiency

        soc_decrease = energy_from_battery_mwh / self.capacity_mwh
        self.soc = self.soc - soc_decrease

        min_soc = 1 - self.dod_max
        self.soc = max(self.soc, min_soc)

        if duration_h > 0:
            power_actual = energy_out_mwh / duration_h
        else:
            power_actual = 0.0

        return power_actual

    def get_available_energy(self):

        min_soc = 1 - self.dod_max
        available_energy_mwh = self.capacity_mwh * (self.soc - min_soc)

        return max(available_energy_mwh, 0.0)

    def __repr__(self):

        return (f"Battery(name='{self.name}', capacity={self.capacity_mwh}MWh, "
                f"c_rate={self.c_rate}, p_max={self.p_max_mw}MW, "
                f"efficiency={self.efficiency}, dod_max={self.dod_max}, "
                f"soc={self.soc:.2f})")
