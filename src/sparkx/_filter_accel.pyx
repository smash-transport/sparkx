# cython: boundscheck=False, wraparound=False, cdivision=True

# Optional Cython-accelerated helpers for sparkx.Filter
# These operate on Python lists of Particle objects but implement the hot loops
# in C-level for speed. Public API is internal to sparkx; Filter.py delegates here.

import math
import numpy as np


cpdef list charged_particles(list particle_list):
    cdef list out = []
    cdef list kept
    cdef object elem
    cdef double q
    cdef Py_ssize_t i, n = len(particle_list)
    for i in range(n):
        kept = []
        for elem in particle_list[i]:
            q = elem.charge
            if (q != 0.0) and (not math.isnan(q)):
                kept.append(elem)
        out.append(kept)
    return out


cpdef list uncharged_particles(list particle_list):
    cdef list out = []
    cdef list kept
    cdef object elem
    cdef double q
    cdef Py_ssize_t i, n = len(particle_list)
    for i in range(n):
        kept = []
        for elem in particle_list[i]:
            q = elem.charge
            if (q == 0.0) and (not math.isnan(q)):
                kept.append(elem)
        out.append(kept)
    return out


cpdef list participants(list particle_list):
    cdef list out = []
    cdef list kept
    cdef object elem
    cdef double ncoll
    cdef Py_ssize_t i, n = len(particle_list)
    for i in range(n):
        kept = []
        for elem in particle_list[i]:
            ncoll = elem.ncoll
            if (ncoll != 0.0) and (not math.isnan(ncoll)):
                kept.append(elem)
        out.append(kept)
    return out


cpdef list spectators(list particle_list):
    cdef list out = []
    cdef list kept
    cdef object elem
    cdef double ncoll
    cdef Py_ssize_t i, n = len(particle_list)
    for i in range(n):
        kept = []
        for elem in particle_list[i]:
            ncoll = elem.ncoll
            if (ncoll == 0.0) and (not math.isnan(ncoll)):
                kept.append(elem)
        out.append(kept)
    return out


cpdef list particle_species_keep(list particle_list, set pdg_set):
    cdef list out = []
    cdef list kept
    cdef object elem
    cdef long pdg
    cdef Py_ssize_t i, n = len(particle_list)
    for i in range(n):
        kept = []
        for elem in particle_list[i]:
            if not np.isnan(elem.pdg):
                pdg = <long> int(elem.pdg)
                if pdg in pdg_set:
                    kept.append(elem)
        out.append(kept)
    return out


cpdef list particle_species_remove(list particle_list, set pdg_set):
    cdef list out = []
    cdef list kept
    cdef object elem
    cdef long pdg
    cdef Py_ssize_t i, n = len(particle_list)
    for i in range(n):
        kept = []
        for elem in particle_list[i]:
            if not np.isnan(elem.pdg):
                pdg = <long> int(elem.pdg)
                if pdg not in pdg_set:
                    kept.append(elem)
        out.append(kept)
    return out


cpdef list particle_status_keep(list particle_list, set status_set):
    cdef list out = []
    cdef list kept
    cdef object elem
    cdef long st
    cdef Py_ssize_t i, n = len(particle_list)
    for i in range(n):
        kept = []
        for elem in particle_list[i]:
            if not np.isnan(elem.status):
                st = <long> int(elem.status)
                if st in status_set:
                    kept.append(elem)
        out.append(kept)
    return out


cpdef list remove_photons(list particle_list):
    cdef list out = []
    cdef list kept
    cdef object elem
    cdef long pdg
    cdef Py_ssize_t i, n = len(particle_list)
    for i in range(n):
        kept = []
        for elem in particle_list[i]:
            if not np.isnan(elem.pdg):
                pdg = <long> int(elem.pdg)
                if pdg != 22:
                    kept.append(elem)
        out.append(kept)
    return out


cpdef list multiplicity_cut_range(list particle_list, double lim_min, double lim_max):
    cdef list out = []
    cdef Py_ssize_t i, n = len(particle_list)
    cdef Py_ssize_t multiplicity
    for i in range(n):
        multiplicity = len(particle_list[i])
        if multiplicity >= lim_min and multiplicity < lim_max:
            out.append(particle_list[i])
    if len(out) == 0:
        out = [[]]
    return out


cpdef list lower_event_energy_cut_threshold(list particle_list, double threshold):
    cdef list out = []
    cdef object elem
    cdef double total
    cdef double e
    cdef Py_ssize_t i, n = len(particle_list)
    for i in range(n):
        total = 0.0
        for elem in particle_list[i]:
            e = elem.E
            if not math.isnan(e):
                total += e
        if total >= threshold:
            out.append(particle_list[i])
    if len(out) == 0:
        out = [[]]
    return out


cpdef list spacetime_cut_dim_range(list particle_list, int dim_idx, double lim_min, double lim_max):
    cdef list out = []
    cdef list kept
    cdef object elem
    cdef double v
    cdef Py_ssize_t i, n = len(particle_list)
    for i in range(n):
        kept = []
        for elem in particle_list[i]:
            if dim_idx == 0:
                v = elem.t
            elif dim_idx == 1:
                v = elem.x
            elif dim_idx == 2:
                v = elem.y
            else:
                v = elem.z
            if (not math.isnan(v)) and (v >= lim_min) and (v <= lim_max):
                kept.append(elem)
        out.append(kept)
    return out


cpdef list pT_cut_range(list particle_list, double lim_min, double lim_max):
    cdef list out = []
    cdef list kept
    cdef object elem
    cdef double v
    cdef Py_ssize_t i, n = len(particle_list)
    for i in range(n):
        kept = []
        for elem in particle_list[i]:
            v = elem.pT_abs()
            if (not math.isnan(v)) and (v >= lim_min) and (v <= lim_max):
                kept.append(elem)
        out.append(kept)
    return out


cpdef list mT_cut_range(list particle_list, double lim_min, double lim_max):
    cdef list out = []
    cdef list kept
    cdef object elem
    cdef double v
    cdef Py_ssize_t i, n = len(particle_list)
    for i in range(n):
        kept = []
        for elem in particle_list[i]:
            v = elem.mT()
            if (not math.isnan(v)) and (v >= lim_min) and (v <= lim_max):
                kept.append(elem)
        out.append(kept)
    return out


cpdef list rapidity_cut_range(list particle_list, double lim_min, double lim_max):
    cdef list out = []
    cdef list kept
    cdef object elem
    cdef double v
    cdef Py_ssize_t i, n = len(particle_list)
    for i in range(n):
        kept = []
        for elem in particle_list[i]:
            v = elem.rapidity()
            if (not math.isnan(v)) and (v >= lim_min) and (v <= lim_max):
                kept.append(elem)
        out.append(kept)
    return out


cpdef list pseudorapidity_cut_range(list particle_list, double lim_min, double lim_max):
    cdef list out = []
    cdef list kept
    cdef object elem
    cdef double v
    cdef Py_ssize_t i, n = len(particle_list)
    for i in range(n):
        kept = []
        for elem in particle_list[i]:
            v = elem.pseudorapidity()
            if (not math.isnan(v)) and (v >= lim_min) and (v <= lim_max):
                kept.append(elem)
        out.append(kept)
    return out


cpdef list spacetime_rapidity_cut_range(list particle_list, double lim_min, double lim_max):
    cdef list out = []
    cdef list kept
    cdef object elem
    cdef double v
    cdef Py_ssize_t i, n = len(particle_list)
    for i in range(n):
        kept = []
        for elem in particle_list[i]:
            v = elem.spacetime_rapidity()
            if (not math.isnan(v)) and (v >= lim_min) and (v <= lim_max):
                kept.append(elem)
        out.append(kept)
    return out


# Classification filters using Particle methods
cpdef list keep_hadrons(list particle_list):
    cdef list out = []
    cdef list kept
    cdef object elem
    cdef object v
    cdef Py_ssize_t i, n = len(particle_list)
    for i in range(n):
        kept = []
        for elem in particle_list[i]:
            v = elem.is_hadron()
            if v and not np.isnan(v):
                kept.append(elem)
        out.append(kept)
    return out


cpdef list keep_leptons(list particle_list):
    cdef list out = []
    cdef list kept
    cdef object elem
    cdef object v
    cdef Py_ssize_t i, n = len(particle_list)
    for i in range(n):
        kept = []
        for elem in particle_list[i]:
            v = elem.is_lepton()
            if v and not np.isnan(v):
                kept.append(elem)
        out.append(kept)
    return out


cpdef list keep_quarks(list particle_list):
    cdef list out = []
    cdef list kept
    cdef object elem
    cdef object v
    cdef Py_ssize_t i, n = len(particle_list)
    for i in range(n):
        kept = []
        for elem in particle_list[i]:
            v = elem.is_quark()
            if v and not np.isnan(v):
                kept.append(elem)
        out.append(kept)
    return out


cpdef list keep_mesons(list particle_list):
    cdef list out = []
    cdef list kept
    cdef object elem
    cdef object v
    cdef Py_ssize_t i, n = len(particle_list)
    for i in range(n):
        kept = []
        for elem in particle_list[i]:
            v = elem.is_meson()
            if v and not np.isnan(v):
                kept.append(elem)
        out.append(kept)
    return out


cpdef list keep_baryons(list particle_list):
    cdef list out = []
    cdef list kept
    cdef object elem
    cdef object v
    cdef Py_ssize_t i, n = len(particle_list)
    for i in range(n):
        kept = []
        for elem in particle_list[i]:
            v = elem.is_baryon()
            if v and not np.isnan(v):
                kept.append(elem)
        out.append(kept)
    return out


cpdef list keep_up(list particle_list):
    cdef list out = []
    cdef list kept
    cdef object elem
    cdef object v
    cdef Py_ssize_t i, n = len(particle_list)
    for i in range(n):
        kept = []
        for elem in particle_list[i]:
            v = elem.has_up()
            if v and not np.isnan(v):
                kept.append(elem)
        out.append(kept)
    return out


cpdef list keep_down(list particle_list):
    cdef list out = []
    cdef list kept
    cdef object elem
    cdef object v
    cdef Py_ssize_t i, n = len(particle_list)
    for i in range(n):
        kept = []
        for elem in particle_list[i]:
            v = elem.has_down()
            if v and not np.isnan(v):
                kept.append(elem)
        out.append(kept)
    return out


cpdef list keep_strange(list particle_list):
    cdef list out = []
    cdef list kept
    cdef object elem
    cdef object v
    cdef Py_ssize_t i, n = len(particle_list)
    for i in range(n):
        kept = []
        for elem in particle_list[i]:
            v = elem.has_strange()
            if v and not np.isnan(v):
                kept.append(elem)
        out.append(kept)
    return out


cpdef list keep_charm(list particle_list):
    cdef list out = []
    cdef list kept
    cdef object elem
    cdef object v
    cdef Py_ssize_t i, n = len(particle_list)
    for i in range(n):
        kept = []
        for elem in particle_list[i]:
            v = elem.has_charm()
            if v and not np.isnan(v):
                kept.append(elem)
        out.append(kept)
    return out


cpdef list keep_bottom(list particle_list):
    cdef list out = []
    cdef list kept
    cdef object elem
    cdef object v
    cdef Py_ssize_t i, n = len(particle_list)
    for i in range(n):
        kept = []
        for elem in particle_list[i]:
            v = elem.has_bottom()
            if v and not np.isnan(v):
                kept.append(elem)
        out.append(kept)
    return out


cpdef list keep_top(list particle_list):
    cdef list out = []
    cdef list kept
    cdef object elem
    cdef object v
    cdef Py_ssize_t i, n = len(particle_list)
    for i in range(n):
        kept = []
        for elem in particle_list[i]:
            v = elem.has_top()
            if v and not np.isnan(v):
                kept.append(elem)
        out.append(kept)
    return out
