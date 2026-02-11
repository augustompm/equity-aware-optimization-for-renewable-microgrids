class Generator:

    def __init__(self, capacity_mw, efficiency, fuel_cost_per_mmbtu,
                 min_load_fraction, startup_time_h, name="Generator"):

        self.capacity_mw = capacity_mw
        self.efficiency = efficiency
        self.fuel_cost_per_mmbtu = fuel_cost_per_mmbtu
        self.min_load_fraction = min_load_fraction
        self.startup_time_h = startup_time_h
        self.name = name

        self.min_load_mw = capacity_mw * min_load_fraction

        self.is_online = False
        self.hours_since_startup = 0

    def get_heat_rate_mmbtu_per_mwh(self):

        return 3.412 / self.efficiency

    def fuel_consumption_mmbtu_h(self, power_output_mw):

        if power_output_mw < 0:
            raise ValueError("Power output cannot be negative")

        heat_rate = self.get_heat_rate_mmbtu_per_mwh()
        return power_output_mw * heat_rate

    def operating_cost(self, power_output_mw, duration_h):

        if duration_h < 0:
            raise ValueError("Duration cannot be negative")

        fuel_consumed_mmbtu = self.fuel_consumption_mmbtu_h(power_output_mw)
        return fuel_consumed_mmbtu * self.fuel_cost_per_mmbtu * duration_h

    def can_dispatch(self, power_demand_mw):

        if power_demand_mw > self.capacity_mw:
            return False

        if self.is_online and power_demand_mw < self.min_load_mw:
            return False

        return True

    def dispatch(self, power_demand_mw, timestep_h):

        self.is_online = True

        power_output_mw = min(power_demand_mw, self.capacity_mw)

        self.hours_since_startup += timestep_h

        fuel_consumed_mmbtu = self.fuel_consumption_mmbtu_h(power_output_mw) * timestep_h
        operating_cost_usd = self.operating_cost(power_output_mw, timestep_h)

        return {
            'power_output_mw': power_output_mw,
            'fuel_consumed_mmbtu': fuel_consumed_mmbtu,
            'operating_cost_usd': operating_cost_usd
        }
