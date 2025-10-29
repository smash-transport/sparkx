"""
Pure-Python fallback helpers for sparkx.Filter accelerated paths.
These mirror the API of the Cython module _filter_accel.pyx.
Using these keeps Filter.py simple; if the C-extension is present it will be
used instead.
"""
from typing import List, Set
import math
import numpy as np


def charged_particles(particle_list: list) -> list:
    out = []
    for ev in particle_list:
        kept = [elem for elem in ev if (elem.charge != 0 and not math.isnan(elem.charge))]
        out.append(kept)
    return out


def uncharged_particles(particle_list: list) -> list:
    out = []
    for ev in particle_list:
        kept = [elem for elem in ev if (elem.charge == 0 and not math.isnan(elem.charge))]
        out.append(kept)
    return out


def participants(particle_list: list) -> list:
    out = []
    for ev in particle_list:
        kept = [elem for elem in ev if (elem.ncoll != 0 and not math.isnan(elem.ncoll))]
        out.append(kept)
    return out


def spectators(particle_list: list) -> list:
    out = []
    for ev in particle_list:
        kept = [elem for elem in ev if (elem.ncoll == 0 and not math.isnan(elem.ncoll))]
        out.append(kept)
    return out


def particle_species_keep(particle_list: list, pdg_set: Set[int]) -> list:
    out = []
    for ev in particle_list:
        kept = [elem for elem in ev if (not np.isnan(elem.pdg) and int(elem.pdg) in pdg_set)]
        out.append(kept)
    return out


def particle_species_remove(particle_list: list, pdg_set: Set[int]) -> list:
    out = []
    for ev in particle_list:
        kept = [elem for elem in ev if (not np.isnan(elem.pdg) and int(elem.pdg) not in pdg_set)]
        out.append(kept)
    return out


def particle_status_keep(particle_list: list, status_set: Set[int]) -> list:
    out = []
    for ev in particle_list:
        kept = [elem for elem in ev if (not np.isnan(elem.status) and int(elem.status) in status_set)]
        out.append(kept)
    return out


def remove_photons(particle_list: list) -> list:
    out = []
    for ev in particle_list:
        kept = [elem for elem in ev if (not np.isnan(elem.pdg) and int(elem.pdg) != 22)]
        out.append(kept)
    return out


def multiplicity_cut_range(particle_list: list, lim_min: float, lim_max: float) -> list:
    out = [ev for ev in particle_list if (len(ev) >= lim_min and len(ev) < lim_max)]
    if len(out) == 0:
        out = [[]]
    return out


def lower_event_energy_cut_threshold(particle_list: list, threshold: float) -> list:
    out = []
    for ev in particle_list:
        total = 0.0
        for elem in ev:
            if not math.isnan(elem.E):
                total += elem.E
        if total >= threshold:
            out.append(ev)
    if len(out) == 0:
        out = [[]]
    return out


def spacetime_cut_dim_range(particle_list: list, dim_idx: int, lim_min: float, lim_max: float) -> list:
    out = []
    for ev in particle_list:
        kept = []
        for elem in ev:
            if dim_idx == 0:
                v = elem.t
            elif dim_idx == 1:
                v = elem.x
            elif dim_idx == 2:
                v = elem.y
            else:
                v = elem.z
            if (not math.isnan(v)) and (lim_min <= v <= lim_max):
                kept.append(elem)
        out.append(kept)
    return out


def pT_cut_range(particle_list: list, lim_min: float, lim_max: float) -> list:
    out = []
    for ev in particle_list:
        kept = []
        for elem in ev:
            v = elem.pT_abs()
            if (not math.isnan(v)) and (lim_min <= v <= lim_max):
                kept.append(elem)
        out.append(kept)
    return out


def mT_cut_range(particle_list: list, lim_min: float, lim_max: float) -> list:
    out = []
    for ev in particle_list:
        kept = []
        for elem in ev:
            v = elem.mT()
            if (not math.isnan(v)) and (lim_min <= v <= lim_max):
                kept.append(elem)
        out.append(kept)
    return out


def rapidity_cut_range(particle_list: list, lim_min: float, lim_max: float) -> list:
    out = []
    for ev in particle_list:
        kept = []
        for elem in ev:
            v = elem.rapidity()
            if (not math.isnan(v)) and (lim_min <= v <= lim_max):
                kept.append(elem)
        out.append(kept)
    return out


def pseudorapidity_cut_range(particle_list: list, lim_min: float, lim_max: float) -> list:
    out = []
    for ev in particle_list:
        kept = []
        for elem in ev:
            v = elem.pseudorapidity()
            if (not math.isnan(v)) and (lim_min <= v <= lim_max):
                kept.append(elem)
        out.append(kept)
    return out


def spacetime_rapidity_cut_range(particle_list: list, lim_min: float, lim_max: float) -> list:
    out = []
    for ev in particle_list:
        kept = []
        for elem in ev:
            v = elem.spacetime_rapidity()
            if (not math.isnan(v)) and (lim_min <= v <= lim_max):
                kept.append(elem)
        out.append(kept)
    return out


# Classification filters
def keep_hadrons(particle_list: list) -> list:
    out = []
    for ev in particle_list:
        kept = [elem for elem in ev if (elem.is_hadron() and not np.isnan(elem.is_hadron()))]
        out.append(kept)
    return out


def keep_leptons(particle_list: list) -> list:
    out = []
    for ev in particle_list:
        kept = [elem for elem in ev if (elem.is_lepton() and not np.isnan(elem.is_lepton()))]
        out.append(kept)
    return out


def keep_quarks(particle_list: list) -> list:
    out = []
    for ev in particle_list:
        kept = [elem for elem in ev if (elem.is_quark() and not np.isnan(elem.is_quark()))]
        out.append(kept)
    return out


def keep_mesons(particle_list: list) -> list:
    out = []
    for ev in particle_list:
        kept = [elem for elem in ev if (elem.is_meson() and not np.isnan(elem.is_meson()))]
        out.append(kept)
    return out


def keep_baryons(particle_list: list) -> list:
    out = []
    for ev in particle_list:
        kept = [elem for elem in ev if (elem.is_baryon() and not np.isnan(elem.is_baryon()))]
        out.append(kept)
    return out


def keep_up(particle_list: list) -> list:
    out = []
    for ev in particle_list:
        kept = [elem for elem in ev if (elem.has_up() and not np.isnan(elem.has_up()))]
        out.append(kept)
    return out


def keep_down(particle_list: list) -> list:
    out = []
    for ev in particle_list:
        kept = [elem for elem in ev if (elem.has_down() and not np.isnan(elem.has_down()))]
        out.append(kept)
    return out


def keep_strange(particle_list: list) -> list:
    out = []
    for ev in particle_list:
        kept = [elem for elem in ev if (elem.has_strange() and not np.isnan(elem.has_strange()))]
        out.append(kept)
    return out


def keep_charm(particle_list: list) -> list:
    out = []
    for ev in particle_list:
        kept = [elem for elem in ev if (elem.has_charm() and not np.isnan(elem.has_charm()))]
        out.append(kept)
    return out


def keep_bottom(particle_list: list) -> list:
    out = []
    for ev in particle_list:
        kept = [elem for elem in ev if (elem.has_bottom() and not np.isnan(elem.has_bottom()))]
        out.append(kept)
    return out


def keep_top(particle_list: list) -> list:
    out = []
    for ev in particle_list:
        kept = [elem for elem in ev if (elem.has_top() and not np.isnan(elem.has_top()))]
        out.append(kept)
    return out
