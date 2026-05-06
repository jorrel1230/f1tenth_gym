"""
Battery / MGU-K model layered on top of the F110 vehicle for F1 2026 energy
management
"""

from dataclasses import dataclass


@dataclass
class EnergyParams:
    P_mgu_k: float = 200.0      # MGU-K peak power [W]
    P_regen: float = 130.0      # Regen peak power [W]
    E_batt: float = 400.0       # Battery capacity [J]
    eta_regen: float = 0.7      # Regen efficiency
    v_floor: float = 0.5        # Velocity floor for division safety [m/s]
    soc_init: float = 1.0       # Initial state of charge
    m: float = 3.74             # Vehicle mass [kg] (matches default F110 params)
    a_max_base: float = 9.51    # Baseline ICE a_max [m/s^2]
    v_switch: float = 7.319     # Power-limit velocity [m/s]

    @property
    def a_max_boost(self) -> float:
        # At v = v_switch, the env's accl cap is a_max; raising a_max by this
        # amount adds (a_max_boost * m * v_switch) = P_mgu_k watts at v_switch.
        return self.P_mgu_k / (self.m * self.v_switch)


class EnergyManagedCar:
    def __init__(self, params: EnergyParams = None):
        self.params = params if params is not None else EnergyParams()
        self.soc = self.params.soc_init
        self.last_step: dict = {}

    def reset(self) -> None:
        self.soc = self.params.soc_init
        self.last_step = {}

    def effective_deploy(self, deploy: float) -> float:
        d = max(-1.0, min(1.0, float(deploy)))
        if d > 0.0 and self.soc <= 0.0:
            return 0.0
        if d < 0.0 and self.soc >= 1.0:
            return 0.0
        return d

    def a_max_effective(self, deploy: float) -> float:
        d = self.effective_deploy(deploy)
        if d > 0.0:
            return self.params.a_max_base + d * self.params.a_max_boost
        return self.params.a_max_base

    def step(self, deploy: float, a_actual: float, v: float, dt: float, a_drag: float = 0.0) -> dict:
        """Advance the battery / MGU-K model by one sim tick.

        a_drag is the signed drag accel imposed by the inner sim
        (-0.5*rho*Cd_A*v*|v|/m, negative when v>0). The energy model uses
        a_engine = a_actual - a_drag to attribute mechanical work to the
        engine rather than to drag. Both deploy and harvest branches use
        a_engine: deploy gates on engine output above the ICE cap; harvest
        credits only the brake-driven portion of decel beyond drag.
        """
        d = self.effective_deploy(deploy)
        p = self.params

        a_engine = a_actual - a_drag

        P_out = 0.0
        P_in = 0.0

        # Commanded-ERS model: deploy drain is proportional to commanded
        # power, not to ICE-cap headroom. Real F1 ERS spends energy when
        # commanded; whether the drivetrain converts that into extra accel
        # is a separate question. Gated on motion via v_floor so a parked
        # car can't grief SOC by holding deploy.
        if d > 0.0 and abs(v) >= p.v_floor:
            P_out = d * p.P_mgu_k
        elif d < 0.0 and a_engine < 0.0:
            braking_power = abs(a_engine) * p.m * max(abs(v), p.v_floor)
            available = abs(d) * p.P_regen
            P_in = p.eta_regen * min(available, braking_power)

        dSOC = (-P_out + P_in) * dt / p.E_batt
        self.soc = max(0.0, min(1.0, self.soc + dSOC))

        self.last_step = {
            'soc': self.soc,
            'deploy_effective': d,
            'P_out': P_out,
            'P_in': P_in,
            'dSOC': dSOC,
        }
        return self.last_step
