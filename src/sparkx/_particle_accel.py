"""
Pure-Python fallback for sparkx._particle_accel.
Provides the same function signatures as the Cython-accelerated module.
"""

import math

EPS = 1e-10

def p_abs(px: float, py: float, pz: float) -> float:
    return math.sqrt(px * px + py * py + pz * pz)

def pT_abs(px: float, py: float) -> float:
    return math.sqrt(px * px + py * py)

def phi(px: float, py: float) -> float:
    if abs(px) < 1e-6 and abs(py) < 1e-6:
        return 0.0
    return math.atan2(py, px)

def theta(p_abs_val: float, pz: float) -> float:
    if p_abs_val == 0.0:
        return 0.0
    c = pz / p_abs_val
    if c > 1.0:
        c = 1.0
    elif c < -1.0:
        c = -1.0
    return math.acos(c)

def mT(E: float, pz: float) -> float:
    mtsq = E * E - pz * pz
    if mtsq >= 0.0:
        return math.sqrt(mtsq)
    elif -mtsq < 1e-16:
        return 0.0
    else:
        return float("nan")

def rapidity(E: float, pz: float, mT_val: float) -> float:
    if mT_val > 1e-16:
        x = pz / mT_val
        # asinh(x) available since Python 3.2
        return math.asinh(x)
    if math.isnan(E):
        return float("nan")
    numer = E + pz
    denom = E - pz
    if denom <= 0.0 or numer <= 0.0:
        return float("nan")
    return 0.5 * math.log(numer / denom)

def pseudorapidity(p_abs_val: float, pz: float) -> float:
    if math.isnan(p_abs_val) or math.isnan(pz):
        return float("nan")
    denom = p_abs_val - pz
    if abs(denom) < EPS:
        denom = denom + EPS if denom >= 0.0 else -(-denom + EPS)
    return 0.5 * math.log((p_abs_val + pz) / denom)

def spacetime_rapidity(t: float, z: float) -> float:
    if math.isnan(t) or math.isnan(z):
        return float("nan")
    if t > abs(z):
        return 0.5 * math.log((t + z) / (t - z))
    else:
        raise ValueError("|z| < t not fulfilled")

def proper_time(t: float, z: float) -> float:
    if math.isnan(t) or math.isnan(z):
        return float("nan")
    if t > abs(z):
        return math.sqrt(t * t - z * z)
    else:
        raise ValueError("|z| < t not fulfilled")
