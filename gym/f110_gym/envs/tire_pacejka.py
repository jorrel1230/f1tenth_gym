"""Pacejka 89 magic formula tire model + friction ellipse combined slip.

Gonultas et al. 2023 F1Tenth coefficients (B=10, C=1.9,
D=1, E=0.97): 1:10 scale.
"""

import math
from numba import njit


@njit(cache=True)
def pacejka_long(kappa, Fz, mu, B, C, D_scale, E):
    """Longitudinal tire force from slip ratio kappa.

    Args:
        kappa:   slip ratio = (omega*r - v_x_w) / max(|v_x_w|, eps); dimensionless.
        Fz:      vertical load on the tire [N], non-negative.
        mu:      surface friction multiplier (dimensionless).
        B,C,E:   Pacejka shape parameters.
        D_scale: peak friction coefficient at mu=1 (dimensionless).

    Returns:
        Fx [N], signed (sign of kappa).
    """
    if Fz <= 0.0:
        return 0.0
    D = mu * D_scale * Fz
    Bk = B * kappa
    inner = Bk - E * (Bk - math.atan(Bk))
    return D * math.sin(C * math.atan(inner))


@njit(cache=True)
def pacejka_lat(alpha, Fz, mu, B, C, D_scale, E):
    """Lateral tire force from slip angle alpha [rad].

    Same magic formula shape as the longitudinal axis. Camber is not modeled
    (gamma = 0 implicit); F1 has small static camber and active camber
    correction is out of scope.
    """
    if Fz <= 0.0:
        return 0.0
    D = mu * D_scale * Fz
    Ba = B * alpha
    inner = Ba - E * (Ba - math.atan(Ba))
    return D * math.sin(C * math.atan(inner))


@njit(cache=True)
def friction_ellipse(Fx, Fy, Fz, mu):
    """Cap (Fx, Fy) inside a circle of radius mu*Fz, preserving direction.

    If sqrt(Fx^2 + Fy^2) <= mu*Fz, returns (Fx, Fy) unchanged.
    Otherwise scales both uniformly so the magnitude == mu*Fz.
    """
    if Fz <= 0.0:
        return 0.0, 0.0
    mag2 = Fx * Fx + Fy * Fy
    cap = mu * Fz
    cap2 = cap * cap
    if mag2 <= cap2:
        return Fx, Fy
    s = cap / math.sqrt(mag2)
    return Fx * s, Fy * s
