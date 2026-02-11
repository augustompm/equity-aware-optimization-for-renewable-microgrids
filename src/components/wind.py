class WindTurbine:

    def __init__(self, capacity_mw, hub_height_m=100,
                 cut_in_ms=3.0, rated_ms=12.0, cut_out_ms=25.0,
                 name="WindTurbine"):

        self.capacity_mw = capacity_mw
        self.hub_height_m = hub_height_m
        self.cut_in_ms = cut_in_ms
        self.rated_ms = rated_ms
        self.cut_out_ms = cut_out_ms
        self.name = name

    def power_curve(self, wind_speed_ms):

        if wind_speed_ms < 0:
            return 0.0

        if wind_speed_ms < self.cut_in_ms:
            return 0.0

        if wind_speed_ms < self.rated_ms:

            normalized_speed = ((wind_speed_ms - self.cut_in_ms) /
                               (self.rated_ms - self.cut_in_ms))

            power_mw = self.capacity_mw * (normalized_speed ** 3)
            return power_mw

        if wind_speed_ms < self.cut_out_ms:
            return self.capacity_mw

        return 0.0

    def generate(self, cf_wind):

        power_mw = self.capacity_mw * cf_wind

        return max(power_mw, 0.0)

    def __repr__(self):

        return (f"WindTurbine(name='{self.name}', capacity={self.capacity_mw}MW, "
                f"hub_height={self.hub_height_m}m, "
                f"cut_in={self.cut_in_ms}m/s, rated={self.rated_ms}m/s, "
                f"cut_out={self.cut_out_ms}m/s)")
