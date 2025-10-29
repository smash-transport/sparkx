#!/usr/bin/env python3
"""
Benchmark common sparkx.Filter functions on synthetic event data.

This script reports timings for selected filters and whether the accelerated
helpers are available (and if compiled). Use it to gauge relative speed, not
as an exact profiler.

Usage (optional args):
    python scripts/bench_filter.py --events 200 --particles 500 --repeats 3 --seed 42
"""
import argparse
import importlib
import random
import time
import math
from typing import List

import numpy as np

from sparkx.Particle import Particle
import sparkx.Filter as F


def detect_accel() -> str:
    """
    Detect whether sparkx._filter_accel is importable and if it's a compiled extension.

    Returns a human-readable string with the path and compiled status, or the error.
    """
    try:
        m = importlib.import_module("sparkx._filter_accel")
        path = getattr(m, "__file__", "<unknown>")
        # Heuristic: compiled modules typically end with .so/.pyd
        is_compiled = isinstance(path, str) and path.endswith((".so", ".pyd"))
        return f"{path} (compiled={is_compiled})"
    except Exception as e:
        return f"not importable: {e}"


def make_particle(pdg: int) -> Particle:
    p = Particle()
    # Kinematics: avoid NaN across filters
    px, py, pz = np.random.normal(0.0, 1.5, size=3)
    m = random.choice([0.0, 0.139, 0.494, 0.938])  # pi, K, p masses approx
    E = math.sqrt(px * px + py * py + pz * pz + m * m)
    p.px = px
    p.py = py
    p.pz = pz
    p.E = E

    # Space-time: ensure t > |z| for spacetime_rapidity stability
    z = np.random.normal(0.0, 3.0)
    t = abs(z) + 0.1 + abs(np.random.normal(0.0, 0.5))
    p.z = z
    p.t = t
    p.x = np.random.normal(0.0, 3.0)
    p.y = np.random.normal(0.0, 3.0)

    p.pdg = pdg
    # Derive charge when PDG is valid
    ch = p.charge_from_pdg()
    if not math.isnan(ch):
        p.charge = ch
    else:
        # some PDGs may not be known; fall back to 0
        p.charge = 0

    p.status = random.choice([0, 1, 2, 3])
    p.ncoll = random.randint(0, 3)
    return p


def build_dataset(n_events: int, n_particles: int, seed: int = 0) -> List[List[Particle]]:
    rng = np.random.default_rng(seed)
    random.seed(seed)

    # A mix of hadrons, leptons, photons, and quarks
    pdg_pool = [
        211, -211, 321, -321, 2212, -2212, 2112, -2112,  # mesons/baryons
        11, -11, 13, -13,  # leptons
        22,  # photons
        1, -1, 2, -2, 3, -3  # quarks
    ]

    dataset: List[List[Particle]] = []
    for _ in range(n_events):
        ev = [make_particle(int(rng.choice(pdg_pool))) for _ in range(n_particles)]
        dataset.append(ev)
    return dataset


def time_it(label: str, fn, *args, repeats: int = 1) -> float:
    start = time.perf_counter()
    res = None
    for _ in range(repeats):
        res = fn(*args)
    end = time.perf_counter()
    dt = (end - start) / repeats
    # minimize optimizer effects
    if isinstance(res, list) and len(res) > 0:
        _ = sum(len(x) for x in res)
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=200)
    ap.add_argument("--particles", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    # Acceleration is automatic: if a compiled extension is installed, it will be used.
    args = ap.parse_args()

    # Report helper status like bench_particle.py
    print(f"Filter helper module: {detect_accel()}")
    print(f"Building dataset: events={args.events}, particles={args.particles} ...")
    base = build_dataset(args.events, args.particles, seed=args.seed)

    # Choose a subset of filters to benchmark
    benches = [
        ("charged_particles", F.charged_particles),
        ("uncharged_particles", F.uncharged_particles),
        ("remove_photons", F.remove_photons),
        ("participants", F.participants),
        ("spectators", F.spectators),
        ("particle_species_keep(211,-211)", lambda pl: F.particle_species(pl, [211, -211])),
        ("remove_particle_species(22)", lambda pl: F.remove_particle_species(pl, 22)),
        ("particle_status(1,2)", lambda pl: F.particle_status(pl, [1, 2])),
        ("multiplicity_cut[50, inf)", lambda pl: F.multiplicity_cut(pl, (50, None))),
        ("lower_event_energy_cut>=200", lambda pl: F.lower_event_energy_cut(pl, 200.0)),
        ("spacetime_cut z in [-2,2]", lambda pl: F.spacetime_cut(pl, 'z', (-2.0, 2.0))),
        ("pT_cut [0.5, 2.5]", lambda pl: F.pT_cut(pl, (0.5, 2.5))),
        ("mT_cut [0.5, 2.5]", lambda pl: F.mT_cut(pl, (0.5, 2.5))),
        ("rapidity_cut [-1.5, 1.5]", lambda pl: F.rapidity_cut(pl, (-1.5, 1.5))),
        ("pseudorapidity_cut [-2.0, 2.0]", lambda pl: F.pseudorapidity_cut(pl, (-2.0, 2.0))),
        ("spacetime_rapidity_cut [-1.0, 1.0]", lambda pl: F.spacetime_rapidity_cut(pl, (-1.0, 1.0))),
        ("keep_hadrons", F.keep_hadrons),
        ("keep_leptons", F.keep_leptons),
        ("keep_quarks", F.keep_quarks),
        ("keep_mesons", F.keep_mesons),
        ("keep_baryons", F.keep_baryons),
        ("keep_up", F.keep_up),
        ("keep_down", F.keep_down),
        ("keep_strange", F.keep_strange),
        ("keep_charm", F.keep_charm),
        ("keep_bottom", F.keep_bottom),
        ("keep_top", F.keep_top),
    ]

    # For fairness, create a fresh copy per benchmark since some filters mutate per-event lists
    import copy

    print("\nTimings (seconds):")
    results = []
    for label, func in benches:
        pl_copy = copy.deepcopy(base)
        dt = time_it(label, func, pl_copy, repeats=args.repeats)
        results.append((label, dt))
        print(f"- {label:38s} {dt:.6f}")

    # Summary
    total_one_pass = sum(dt for _, dt in results)
    print(f"\nTotal time (one pass across all filters): {total_one_pass:.6f} s")
    if args.repeats and args.repeats > 1:
        print(
            f"Total time including repeats ({args.repeats}x): {total_one_pass * args.repeats:.6f} s"
        )


if __name__ == "__main__":
    main()
