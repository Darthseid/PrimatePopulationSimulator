"""
Invariant tests for the population simulation loop (array-based population state).

    python -m unittest test_simulation_invariants -v        (run from the repository folder)

These check properties that must hold after every cycle, however the cycle is implemented: nobody is duplicated or
resurrected, unions and their members agree, everyone is aged exactly once, sex transitions and seasonal uncoupling
happen, food allocation never exceeds what exists, and the disaster / warfare code still runs.

The simulation is advanced one cycle at a time (see step()), so each invariant is checked after every cycle.
Disasters are off unless `disasters_enabled=True` is passed to PrimateSimulation (the disaster tests do that).
"""
import contextlib
import io
import os
import random
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")   # never open a plot window

import numpy as np

import PrimatePopulationSimulator as sim_module
import population_ops as ops
from PopulationObjects import Locale, Primate, SimulationParameters
from graphing import log_population_stats

HERE = os.path.dirname(os.path.abspath(__file__))
EARTH_YEAR = 365.2422

sim_module.display_population_pyramid = lambda *a, **k: None   # reporting/plotting are not under test
sim_module.plot_population_history = lambda *a, **k: None


def setUpModule():
    os.chdir(HERE)   # the simulator reads demographics.json / locales.json / disasters.json from the working directory


def make_sim(species, size=200, seed=1, module=sim_module, locale=None, **settings):
    """species: comma-separated names. Returns a ready (not yet run) PrimateSimulation."""
    random.seed(seed)
    np.random.seed(seed)
    module.starting_population = size
    with contextlib.redirect_stdout(io.StringIO()):
        locale = locale or Locale.from_json("locales.json", "south_african_savanna")
        return module.PrimateSimulation(species.split(","), locale, **settings)


def step(sim):
    """Advance exactly one more cycle and return what run_simulation printed."""
    years = (sim.current_day / sim.cycle_days + 0.5) * sim.cycle_days / EARTH_YEAR
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        sim.run_simulation(num_years=years)
    return out.getvalue()


def printed_number(output, label):
    for line in output.splitlines():
        if line.startswith(label):
            return int(line.split(":")[1].strip().replace(",", ""))
    raise AssertionError(f"'{label}' not found in output")


def is_mating_season(sim):
    day_of_year = sim.current_day % EARTH_YEAR
    return day_of_year < 30 or day_of_year > EARTH_YEAR - 30 or sim.cycle_days >= 335


class InvariantMixin:
    def check_population(self, sim, label=""):
        state = sim.state
        n = len(state)
        for name, _ in state.COLUMNS:
            self.assertEqual(len(getattr(state, name)), n, f"{label}: column {name} has the wrong length")
        self.assertTrue(state.alive.all(), f"{label}: a dead individual was left in the population")
        self.assertTrue(np.isfinite(state.age_days).all(), f"{label}: an age is not a number")
        self.assertTrue((state.params_id < sim.table.size).all() and (state.species_id < len(sim.names.names)).all())
        stats = ops.UnionStats(state, sim.table)
        self.assertFalse(stats.dissolved.any(), f"{label}: a dissolved union survived the end-of-cycle cleanup")
        self.assertTrue((stats.size >= 1).all())


def tag_individuals(sim):
    """
    Writes a unique number into each individual's next_breeding_day (which also stops everyone breeding), so the same
    individual can be recognised again after a cycle. Arrays have no object identity to follow.
    """
    n = len(sim.state)
    sim.state.next_breeding_day[:] = 1e12 + np.arange(n)


def tags_of(sim):
    tags = sim.state.next_breeding_day - 1e12
    assert np.array_equal(tags, np.round(tags)) and tags.min(initial=0) >= 0, "an untagged individual appeared"
    return tags.astype(int)


class TestPopulationInvariants(InvariantMixin, unittest.TestCase):
    SPECIES = [
        "modern_human", "kaleidar", "sequent", "simult", "merlin", "widow", "double", "shikamimi", "merfolk",
        "orc,elf,goblin", "modern_human,neanderthal", "shikamimi,orc",
    ]

    def test_state_stays_consistent_and_the_report_agrees_with_it(self):
        for species in self.SPECIES:
            with self.subTest(species=species):
                sim = make_sim(species, 200)
                for cycle in range(1, 11):
                    before = len(sim.state)
                    out = step(sim)
                    self.check_population(sim, f"{species} cycle {cycle}")
                    # The simulator's own summary must agree with the actual population.
                    self.assertEqual(printed_number(out, "Final Population"), len(sim.state))
                    births, deaths = printed_number(out, "Total Births"), printed_number(out, "Total Deaths")
                    self.assertGreaterEqual(len(sim.state), before - deaths)
                    if "double" not in species:   # 'double' respawns dead females as new males without a birth
                        self.assertLessEqual(len(sim.state), before + births)
                    if not len(sim.state):
                        break

    def test_each_survivor_ages_exactly_once_per_cycle(self):
        for species in ("modern_human", "merlin", "sequent"):
            with self.subTest(species=species):
                sim = make_sim(species, 200)
                tag_individuals(sim)
                for _ in range(8):
                    age_before, female_before = sim.state.age_days.copy(), sim.state.is_female.copy()
                    step(sim)
                    tags = tags_of(sim)
                    backward = sim.table.ages_backward[sim.state.params_id]
                    expected = age_before[tags] + np.where(backward, -sim.cycle_days, sim.cycle_days)
                    changed_sex = sim.state.is_female & ~female_before[tags]
                    expected = np.where(changed_sex, 5479, expected)   # a sex change resets age to 15 years
                    np.testing.assert_array_equal(sim.state.age_days, expected)
                    tag_individuals(sim)   # re-tag: the tags are positions in the (compacted, reordered) population

    def test_sequential_hermaphrodites_change_sex(self):
        sim = make_sim("sequent", 300)
        changed = 0
        for _ in range(30):
            tag_individuals(sim)
            female_before = sim.state.is_female.copy()
            step(sim)
            changed += int(np.count_nonzero(sim.state.is_female & ~female_before[tags_of(sim)]))
            old_males = ~sim.state.is_female & (sim.state.age_days / EARTH_YEAR > sim_module.SEQUENT_TRANSITION_YEARS)
            self.assertFalse(old_males.any(), "a male older than the transition age was left unchanged")
            if not len(sim.state):
                break
        self.assertGreater(changed, 0, "no sex change happened, so this test exercised nothing")

    def test_seasonal_maters_are_uncoupled_and_not_paired_off_season(self):
        sim = make_sim("shikamimi,orc", 300)   # orc has a short cycle, so cycle_days < 335 and there is a real off-season
        self.assertLess(sim.cycle_days, 335)
        off_season_cycles = 0
        for _ in range(12):
            step(sim)
            self.check_population(sim, "seasonal mix")
            # current_day has already advanced, which is the day the cycle just run used
            if not is_mating_season(sim):
                off_season_cycles += 1
                seasonal = sim.table.seasonal_truthy[sim.state.params_id]
                self.assertFalse((seasonal & (sim.state.union_id >= 0)).any(),
                                 "a seasonal mater was coupled at the end of an off-season cycle")
            if not len(sim.state):
                break
        self.assertGreater(off_season_cycles, 0, "no off-season cycle occurred, so this test exercised nothing")

    def test_cross_species_reproduction_creates_hybrids(self):
        sim = make_sim("modern_human,neanderthal", 300, hybridization_type="midpoint")
        for _ in range(25):
            step(sim)
            self.check_population(sim, "hybrids")
        self.assertTrue([name for name in sim.state.species_counts() if "-" in name], "no hybrid was ever born")

    def test_a_cycle_creates_no_primate_objects(self):
        """The point of the array state: the cycle works on arrays, not on one Python object per individual."""
        created = []
        original_init = Primate.__init__

        def counting_init(self, *args, **kwargs):
            created.append(1)
            original_init(self, *args, **kwargs)

        sim = make_sim("modern_human", 500)
        Primate.__init__ = counting_init
        try:
            for _ in range(5):
                step(sim)
        finally:
            Primate.__init__ = original_init
        self.assertEqual(created, [])

    def test_the_same_seed_gives_the_same_run(self):
        """Seeded runs used to differ between processes (iteration order of sets of objects); they no longer do."""
        runs = []
        for _ in range(2):
            sim = make_sim("modern_human,neanderthal", 300, seed=7)
            for _ in range(10):
                step(sim)
            runs.append((len(sim.state), sim.state.age_days.sum(), sim.state.children.sum(), sim.state.union_id.max()))
        self.assertEqual(runs[0], runs[1])


class TestEndConditions(InvariantMixin, unittest.TestCase):
    def test_simulation_stops_when_one_sex_is_gone(self):
        for keep_female in (True, False):
            with self.subTest(keep_female=keep_female):
                sim = make_sim("modern_human", 200)
                sim.state = sim.state.take(sim.state.is_female == keep_female)
                self.assertIn("One gender has gone extinct", step(sim))

    def test_everyone_starves_without_food(self):
        locale = Locale.from_json("locales.json", "south_african_savanna")
        locale.carnivore_calories = locale.herbivore_calories = locale.ruminant_calories = 0
        sim = make_sim("modern_human", 200, locale=locale)
        out = step(sim)
        self.assertEqual(len(sim.state), 0)
        self.assertIn("extinct", out)

    def test_food_allocation_never_exceeds_what_exists(self):
        locale = Locale.from_json("locales.json", "south_african_savanna")
        sim = make_sim("modern_human", 300, locale=locale)
        need_of = lambda state: float(ops.caloric_need(state, sim.table).sum() * sim.cycle_days)
        # modern humans are omnivores: they eat vegetation first, meat second. Leave only meat and give ~30% of need.
        locale.herbivore_calories = locale.ruminant_calories = locale.water_availability_m3 = 0
        locale.carnivore_calories = 0.3 * need_of(sim.state) / sim.cycle_days
        start = len(sim.state)
        step(sim)
        self.assertLess(len(sim.state), start, "nobody starved, so this test exercised nothing")
        self.assertLessEqual(need_of(sim.state), locale.carnivore_calories * sim.cycle_days + 1e-6)
        self.check_population(sim, "after starvation")


class TestDisasters(InvariantMixin, unittest.TestCase):
    def force(self, sim, names):
        for d in sim.disasters:
            d.is_possible = d.name in names
            d.trigger_chance = 1.0

    def test_warfare_kills_and_leaves_a_consistent_population(self):
        sim = make_sim("modern_human", 800, disasters_enabled=True)
        self.force(sim, {"Warfare"})
        # resolve_warfare pairs off the post-puberty, not-elderly males and kills one of each pair
        combatants = ops.combatants_of(sim.state, sim.table)
        self.assertGreater(len(combatants), 10, "too few combatants to test anything")
        out = step(sim)
        self.assertIn("DISASTER TRIGGERED: Warfare", out)
        self.assertGreaterEqual(printed_number(out, "Total Deaths"), len(combatants) // 2)
        self.check_population(sim, "after warfare")
        for _ in range(4):
            step(sim)
            self.check_population(sim, "after warfare")

    def test_all_disasters_together_keep_the_population_consistent(self):
        sim = make_sim("modern_human", 800, disasters_enabled=True)
        self.force(sim, {"Warfare", "Plague", "Pollution", "Famine"})
        out = step(sim)
        for name in ("Warfare", "Plague", "Pollution", "Famine"):
            self.assertIn(f"DISASTER TRIGGERED: {name}", out)
        for _ in range(5):
            self.check_population(sim, "under disasters")
            if not len(sim.state):
                break
            step(sim)


class TestStatsLogging(unittest.TestCase):
    def test_report_matches_an_independent_count(self):
        random.seed(3)
        modern = SimulationParameters.from_json("demographics.json", "modern_human")
        orc = SimulationParameters.from_json("demographics.json", "orc")
        people = ([Primate("modern_human", i % 2 == 0, 9000 + i, True, modern) for i in range(7)]
                  + [Primate("orc", False, 5000, True, orc) for _ in range(3)])
        history = []
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            log_population_stats(1000, people, history, 4, 5, 6, 7)
        text = out.getvalue()
        females = sum(p.is_female for p in people)
        self.assertIn(f"Total Population: {len(people)}", text)
        self.assertIn(f"Females: {females}", text)
        self.assertIn(f"Males: {len(people) - females}", text)
        self.assertIn(f"{modern.species_name}: 7", text)
        self.assertIn(f"{orc.species_name}: 3", text)
        self.assertEqual(history[0]["population"], len(people))
        self.assertEqual(history[0]["females"], females)

        # the array state must produce exactly the same report
        from population_state import NameTable, ParameterTable, PopulationState
        names = NameTable()
        table = ParameterTable(names)
        state = PopulationState.from_primates(people, table, names)
        array_history = []
        array_out = io.StringIO()
        with contextlib.redirect_stdout(array_out):
            log_population_stats(1000, state, array_history, 4, 5, 6, 7)
        self.assertEqual(array_out.getvalue(), text)
        self.assertEqual(array_history, history)


if __name__ == "__main__":
    unittest.main()
