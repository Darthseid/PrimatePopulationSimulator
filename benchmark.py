"""
Benchmark the simulator at several starting population sizes.

    python benchmark.py                          # 1,000 / 5,000 / 10,000 / 25,000 agents, 3 repeats each
    python benchmark.py --sizes 1000 10000 --repeats 5 --cycles 30
    python benchmark.py --species orc,elf        # a different (mixed) starting species

Each run simulates a fixed number of cycles (not years), so every size does comparable work and the time per cycle
can be compared between code versions. Only run_simulation() is timed (time.perf_counter): the population is built
beforehand, and console output and plotting are switched off. The simulator itself is not modified by this script.

Population trajectories depend on the random seed. Repeat k of every size uses seed (--seed + k), so two versions of
the code see the same seeds; the reported end population shows whether the two did comparable work.
"""
import argparse
import contextlib
import io
import os
import random
import statistics
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")   # never open a plot window

import numpy as np

import PrimatePopulationSimulator as sim_module
from PopulationObjects import Locale

EARTH_YEAR = 365.2422


def run_once(species, start_population, cycles, seed, locale_name):
    random.seed(seed)
    np.random.seed(seed)
    sim_module.starting_population = start_population           # read when the random population is created
    sim_module.display_population_pyramid = lambda *a, **k: None  # reporting/plotting are not what is measured
    sim_module.plot_population_history = lambda *a, **k: None

    with contextlib.redirect_stdout(io.StringIO()):
        locale = Locale.from_json("locales.json", locale_name)
        sim = sim_module.PrimateSimulation(species, locale)

    # (cycles - 0.5) cycles' worth of years makes the while-loop run exactly `cycles` times.
    years = (cycles - 0.5) * sim.cycle_days / EARTH_YEAR
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        sim.run_simulation(num_years=years)
    seconds = time.perf_counter() - start

    ran = max(1, round(sim.current_day / sim.cycle_days))
    history_pops = [h["population"] for h in sim.history]
    return seconds, ran, len(sim.population), statistics.mean(history_pops)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000, 10000, 25000])
    parser.add_argument("--cycles", type=int, default=20, help="cycles simulated per run (default 20)")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--species", default="modern_human", help="comma-separated species names")
    parser.add_argument("--locale", default="south_african_savanna")
    args = parser.parse_args()
    species = args.species.split(",")

    print(f"{'size':>7} {'rep':>3} {'seconds':>8} {'cycles':>6} {'ms/cycle':>9} {'avg pop':>8} {'end pop':>8}", flush=True)
    summary = []
    for size in args.sizes:
        times = []
        for rep in range(args.repeats):
            seconds, ran, end_pop, avg_pop = run_once(species, size, args.cycles, args.seed + rep, args.locale)
            times.append(seconds / ran)
            print(f"RESULT {size:>7} {rep:>3} {seconds:8.2f} {ran:6d} {1000 * seconds / ran:9.1f} {avg_pop:8.0f} {end_pop:8d}", flush=True)
        summary.append((size, min(times), statistics.median(times)))

    print("\nsize   best ms/cycle   median ms/cycle")
    for size, best, median in summary:
        print(f"{size:>6} {1000 * best:>14.1f} {1000 * median:>16.1f}")


if __name__ == "__main__":
    sys.exit(main())
