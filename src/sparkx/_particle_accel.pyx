# cython: boundscheck=False, wraparound=False, cdivision=True
# Lightweight numeric helpers for sparkx.Particle hot paths
cimport libc.math as cm

DEF EPS = 1e-10

cpdef double p_abs(double px, double py, double pz):
    return cm.sqrt(px*px + py*py + pz*pz)

cpdef double pT_abs(double px, double py):
    return cm.sqrt(px*px + py*py)

cpdef double phi(double px, double py):
    if cm.fabs(px) < 1e-6 and cm.fabs(py) < 1e-6:
        return 0.0
    return cm.atan2(py, px)

cpdef double theta(double p_abs, double pz):
    cdef double c
    if p_abs == 0.0:
        return 0.0
    c = pz / p_abs
    if c > 1.0:
        c = 1.0
    elif c < -1.0:
        c = -1.0
    return cm.acos(c)

cpdef double mT(double E, double pz):
    cdef double mtsq = E*E - pz*pz
    if mtsq >= 0.0:
        return cm.sqrt(mtsq)
    elif -mtsq < 1e-16:
        return 0.0
    else:
        return float('nan')

cpdef double rapidity(double E, double pz, double mT):
    cdef double x
    cdef double numer
    cdef double denom
    if mT > 1e-16:
        # asinh(pz/mT) = log(pz/mT + sqrt((pz/mT)^2 + 1))
        x = pz / mT
        return cm.asinh(x)
    if E != E:
        return float('nan')
    numer = E + pz
    denom = E - pz
    if denom <= 0.0 or numer <= 0.0:
        return float('nan')
    return 0.5 * cm.log(numer/denom)

cpdef double pseudorapidity(double p_abs, double pz):
    cdef double denom
    if p_abs != p_abs or pz != pz:
        return float('nan')
    denom = p_abs - pz
    if cm.fabs(denom) < EPS:
        denom = denom + EPS if denom >= 0.0 else -(-denom + EPS)
    return 0.5 * cm.log((p_abs + pz) / denom)

cpdef double spacetime_rapidity(double t, double z) except *:
    if t != t or z != z:
        return float('nan')
    if t > cm.fabs(z):
        return 0.5 * cm.log((t + z) / (t - z))
    else:
        raise ValueError("|z| < t not fulfilled")

cpdef double proper_time(double t, double z) except *:
    if t != t or z != z:
        return float('nan')
    if t > cm.fabs(z):
        return cm.sqrt(t*t - z*z)
    else:
        raise ValueError("|z| < t not fulfilled")
