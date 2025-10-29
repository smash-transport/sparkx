#!/usr/bin/env python3
import argparse
import importlib
import math
import os
import random
import sys
import time
from statistics import mean

import numpy as np

from sparkx.Particle import Particle


def detect_accel() -> str:
    try:
        m = importlib.import_module("sparkx._particle_accel")
        path = getattr(m, "__file__", "<unknown>")
        is_compiled = path.endswith((".so", ".pyd"))
        return f"{path} (compiled={is_compiled})"
    except Exception as e:
        return f"not importable: {e}"


def make_particles(n: int, seed: int = 123) -> list[Particle]:
    rnd = random.Random(seed)
    parts: list[Particle] = []
    for _ in range(n):
        # Random but physically plausible values
        px = rnd.uniform(-1.0, 1.0)
        py = rnd.uniform(-1.0, 1.0)
        pz = rnd.uniform(-1.0, 1.0)
        pabs = math.sqrt(px * px + py * py + pz * pz)
        # Ensure E >= |p| to avoid NaN mass during benchmark
        E = pabs + rnd.uniform(0.0, 0.5)

        p = Particle()
        p.px = px
        p.py = py
        p.pz = pz
        p.E = E
        p.pdg = 211  # pi+

        # Provide spacetime coordinates for spacetime_rapidity/proper_time
        # Ensure t > |z| so no ValueError is raised
        z_pos = rnd.uniform(-2.0, 2.0)
        t_pos = abs(z_pos) + rnd.uniform(0.001, 0.5)
        p.z = z_pos
        p.t = t_pos
        parts.append(p)
    return parts


def time_call(label: str, fn, repeat: int = 1) -> float:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    t = mean(times)
    print(f"{label:30s}: {t*1000:.2f} ms")
    return t


def bench(n: int, reps: int) -> None:
    print("sparkx._particle_accel:", detect_accel())
    print(f"N={n}, reps={reps}")
    parts = make_particles(n)

    # Warm-up
    _ = sum(p.p_abs() for p in parts)

    def bench_loop(method_name: str):
        if method_name == "p_abs":
            fn = lambda: sum(p.p_abs() for p in parts)
        elif method_name == "pT_abs":
            fn = lambda: sum(p.pT_abs() for p in parts)
        elif method_name == "phi":
            fn = lambda: sum(p.phi() for p in parts)
        elif method_name == "theta":
            fn = lambda: sum(p.theta() for p in parts)
        elif method_name == "mT":
            fn = lambda: sum(p.mT() for p in parts)
        elif method_name == "rapidity":
            fn = lambda: sum((p.rapidity() for p in parts if not np.isnan(p.rapidity())))
        elif method_name == "pseudorapidity":
            fn = lambda: sum((p.pseudorapidity() for p in parts if not np.isnan(p.pseudorapidity())))
        elif method_name == "mass":
            fn = lambda: sum((p.mass_from_energy_momentum() for p in parts if not np.isnan(p.mass_from_energy_momentum())))
        elif method_name == "spacetime_rapidity":
            fn = lambda: sum(p.spacetime_rapidity() for p in parts)
        elif method_name == "proper_time":
            fn = lambda: sum(p.proper_time() for p in parts)
        else:
            raise ValueError(method_name)
        return time_call(method_name, fn, repeat=reps)

    print("Timing (lower is better):")
    total = 0.0
    for name in [
        "p_abs",
        "pT_abs",
        "phi",
        "theta",
        "mT",
        "rapidity",
        "pseudorapidity",
        "mass",
        "spacetime_rapidity",
        "proper_time",
    ]:
        total += bench_loop(name)
    print(f"Total: {total*1000:.2f} ms")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Benchmark sparkx.Particle methods")
    ap.add_argument("-n", "--num", type=int, default=200000, help="number of particles")
    ap.add_argument("-r", "--reps", type=int, default=3, help="repetitions per method")
    args = ap.parse_args(argv)
    bench(args.num, args.reps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
