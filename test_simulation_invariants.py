"""
Invariant tests for the population simulation loop.

    python -m unittest test_simulation_invariants -v        (run from the repository folder)

These check properties that must hold after every cycle, however the cycle is implemented: nobody is duplicated or
resurrected, unions and their members agree, everyone is aged exactly once, sex transitions and seasonal uncoupling
happen, food allocation never exceeds what exists, and the disaster / warfare code still runs.

The simulation is advanced one cycle at a time (see step()), so each invariant is checked after every cycle.
Disasters are switched off by a hard-coded flag inside run_simulation(), so the disaster tests load a copy of the
module in memory with that flag turned on; nothing on disk is changed.
"""
import contextlib
import io
import os
import random
import types
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")   # never open a plot window

import numpy as np

import PrimatePopulationSimulator as sim_module
from PopulationObjects import Locale, Primate, SimulationParameters, Union, find_union_for_primate
from graphing import log_population_stats

HERE = os.path.dirname(os.path.abspath(__file__))
EARTH_YEAR = 365.2422

sim_module.display_population_pyramid = lambda *a, **k: None   # reporting/plotting are not under test
sim_module.plot_population_history = lambda *a, **k: None


def setUpModule():
    os.chdir(HERE)   # the simulator reads demographics.json / locales.json / disasters.json from the working directory


def make_sim(species, size=200, seed=1, module=sim_module, locale=None):
    """species: comma-separated names. Returns a ready (not yet run) PrimateSimulation."""
    random.seed(seed)
    np.random.seed(seed)
    module.starting_population = size
    with contextlib.redirect_stdout(io.StringIO()):
        locale = locale or Locale.from_json("locales.json", "south_african_savanna")
        return module.PrimateSimulation(species.split(","), locale)


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
        pop = sim.population
        ids = {id(p) for p in pop}
        self.assertEqual(len(ids), len(pop), f"{label}: the same individual is in the population twice")

        unions = {}
        for p in pop:
            if p.union is None:
                continue
            self.assertIn(p, p.union.members, f"{label}: individual points at a union that does not list them")
            unions[id(p.union)] = p.union
        for union in unions.values():
            self.assertFalse(union.is_dissolved(), f"{label}: a dissolved union is still referenced after cleanup")
            for member in union.members:
                self.assertIn(id(member), ids, f"{label}: a union still holds someone no longer in the population")
                self.assertIs(member.union, union, f"{label}: a union member does not point back at its union")


class TestPopulationInvariants(InvariantMixin, unittest.TestCase):
    SPECIES = [
        "modern_human", "kaleidar", "sequent", "simult", "merlin", "widow", "double", "shikamimi", "merfolk",
        "orc,elf,goblin", "modern_human,neanderthal", "shikamimi,orc",
    ]

    def test_no_duplicates_resurrection_stale_unions_or_miscounts(self):
        for species in self.SPECIES:
            with self.subTest(species=species):
                sim = make_sim(species, 200)
                dead = {}   # id -> object (kept alive so ids cannot be reused by newborns)
                for cycle in range(1, 11):
                    before = {id(p): p for p in sim.population}
                    out = step(sim)
                    now = {id(p): p for p in sim.population}
                    self.check_population(sim, f"{species} cycle {cycle}")

                    self.assertFalse(set(now) & set(dead), f"{species}: someone who died came back")
                    dead.update({i: p for i, p in before.items() if i not in now})

                    # The simulator's own summary must agree with the actual population.
                    self.assertEqual(printed_number(out, "Final Population"), len(sim.population))
                    births, deaths = printed_number(out, "Total Births"), printed_number(out, "Total Deaths")
                    self.assertGreaterEqual(len(sim.population), len(before) - deaths)
                    if "double" not in species:   # 'double' respawns dead females as new males without a birth
                        self.assertLessEqual(len(sim.population), len(before) + births)
                    if not sim.population:
                        break

    def test_each_survivor_ages_exactly_once_per_cycle(self):
        for species in ("modern_human", "merlin", "sequent"):
            with self.subTest(species=species):
                sim = make_sim(species, 200)
                for _ in range(8):
                    before = {id(p): (p, p.age_days, p.is_female) for p in sim.population}   # keeps refs: no id reuse
                    step(sim)
                    for p in sim.population:
                        if id(p) not in before:
                            continue   # newborn
                        _, age_before, female_before = before[id(p)]
                        if p.is_female and not female_before:
                            self.assertEqual(p.age_days, 5479, "a sex change resets age to 15 years")
                        else:
                            expected = age_before - sim.cycle_days if p.params.ages_backward else age_before + sim.cycle_days
                            self.assertEqual(p.age_days, expected)

    def test_sequential_hermaphrodites_change_sex(self):
        sim = make_sim("sequent", 300)
        changed = 0
        for _ in range(30):
            before = {id(p): (p, p.is_female) for p in sim.population}
            step(sim)
            for p in sim.population:
                if id(p) in before and p.is_female and not before[id(p)][1]:
                    changed += 1
            self.assertFalse([p for p in sim.population if not p.is_female and p.age_days / EARTH_YEAR > sim_module.SEQUENT_TRANSITION_YEARS],
                             "a male older than the transition age was left unchanged")
            if not sim.population:
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
                self.assertFalse([p for p in sim.population if p.params.seasonal_mater and p.union is not None],
                                 "a seasonal mater was coupled at the end of an off-season cycle")
            if not sim.population:
                break
        self.assertGreater(off_season_cycles, 0, "no off-season cycle occurred, so this test exercised nothing")

    def test_cross_species_reproduction_creates_hybrids(self):
        sim = make_sim("modern_human,neanderthal", 300)
        for _ in range(25):
            step(sim)
            self.check_population(sim, "hybrids")
        self.assertTrue([p for p in sim.population if "-" in p.params.species_name], "no hybrid was ever born")


class TestEndConditions(InvariantMixin, unittest.TestCase):
    def test_simulation_stops_when_one_sex_is_gone(self):
        for keep_female in (True, False):
            with self.subTest(keep_female=keep_female):
                sim = make_sim("modern_human", 200)
                sim.population = [p for p in sim.population if p.is_female == keep_female]
                self.assertIn("One gender has gone extinct", step(sim))

    def test_everyone_starves_without_food(self):
        locale = Locale.from_json("locales.json", "south_african_savanna")
        locale.carnivore_calories = locale.herbivore_calories = locale.ruminant_calories = 0
        sim = make_sim("modern_human", 200, locale=locale)
        out = step(sim)
        self.assertEqual(len(sim.population), 0)
        self.assertIn("extinct", out)

    def test_food_allocation_never_exceeds_what_exists(self):
        locale = Locale.from_json("locales.json", "south_african_savanna")
        sim = make_sim("modern_human", 300, locale=locale)
        need_of = lambda people: sum(p.get_caloric_need() * sim.cycle_days for p in people)
        # modern humans are omnivores: they eat vegetation first, meat second. Leave only meat and give ~30% of need.
        locale.herbivore_calories = locale.ruminant_calories = locale.water_availability_m3 = 0
        locale.carnivore_calories = 0.3 * need_of(sim.population) / sim.cycle_days
        start = len(sim.population)
        step(sim)
        self.assertLess(len(sim.population), start, "nobody starved, so this test exercised nothing")
        self.assertLessEqual(need_of(sim.population), locale.carnivore_calories * sim.cycle_days + 1e-6)
        self.check_population(sim, "after starvation")


class TestDisasters(InvariantMixin, unittest.TestCase):
    @staticmethod
    def module_with_disasters_on():
        path = sim_module.__file__
        with open(path, encoding="utf-8-sig") as f:
            source = f.read()
        assert "disasters_enabled = False" in source, "the hard-coded disaster switch moved; update this test"
        module = types.ModuleType("primate_sim_disasters_on")
        module.__file__ = path
        exec(compile(source.replace("disasters_enabled = False", "disasters_enabled = True"), path, "exec"), module.__dict__)
        module.display_population_pyramid = lambda *a, **k: None
        module.plot_population_history = lambda *a, **k: None
        return module

    def force(self, sim, names):
        for d in sim.disasters:
            d.is_possible = d.name in names
            d.trigger_chance = 1.0

    def test_warfare_kills_and_leaves_a_consistent_population(self):
        module = self.module_with_disasters_on()
        sim = make_sim("modern_human", 800, module=module)
        self.force(sim, {"Warfare"})
        # resolve_warfare pairs off the post-puberty, not-elderly males and kills one of each pair
        combatants = [p for p in sim.population if not p.is_female
                      and p.params.puberty_age_days <= (p.age_days / EARTH_YEAR) * EARTH_YEAR < p.params.lifespan_days]
        self.assertGreater(len(combatants), 10, "too few combatants to test anything")
        out = step(sim)
        self.assertIn("DISASTER TRIGGERED: Warfare", out)
        self.assertGreaterEqual(printed_number(out, "Total Deaths"), len(combatants) // 2)
        self.check_population(sim, "after warfare")
        for _ in range(4):
            step(sim)
            self.check_population(sim, "after warfare")

    def test_all_disasters_together_keep_the_population_consistent(self):
        module = self.module_with_disasters_on()
        sim = make_sim("modern_human", 800, module=module)
        self.force(sim, {"Warfare", "Plague", "Pollution", "Famine"})
        out = step(sim)
        for name in ("Warfare", "Plague", "Pollution", "Famine"):
            self.assertIn(f"DISASTER TRIGGERED: {name}", out)
        for _ in range(5):
            self.check_population(sim, "under disasters")
            if not sim.population:
                break
            step(sim)


class TestMating(unittest.TestCase):
    def setUp(self):
        self.params = SimulationParameters.from_json("demographics.json", "modern_human")

    def person(self, female, age_days):
        return Primate("modern_human", female, age_days, True, self.params)

    def test_opposite_sexes_pair_and_same_sexes_do_not(self):
        man, woman = self.person(False, 9000), self.person(True, 9000)
        find_union_for_primate(man, [woman], "monogamy", [])
        self.assertIsNotNone(man.union)
        self.assertIs(man.union, woman.union)
        self.assertCountEqual(man.union.members, [man, woman])

        other_woman = self.person(True, 9000)
        find_union_for_primate(other_woman, [self.person(True, 9000)], "monogamy", [])
        self.assertIsNone(other_woman.union)

    def test_monogamy_picks_the_closest_age_and_ignores_the_unions_argument(self):
        """The cycle loop no longer samples unions for monogamy; this is the property that makes that safe."""
        chosen = []
        for active_unions in ([], [Union(), Union("monogamy", 2)]):
            man = self.person(False, 9000)
            pool = [self.person(True, age) for age in (5000, 8700, 9400, 12000)]
            find_union_for_primate(man, pool, "monogamy", active_unions)
            chosen.append(man.union.members[1].age_days)
        self.assertEqual(chosen, [8700, 8700])

    def test_cycle_loop_draws_no_union_sample_for_monogamy(self):
        sizes = []
        real_sample = random.sample

        def recording_sample(population, k, *a, **kw):
            if population and isinstance(population[0], Primate): # the end-of-run report also samples up to 40 *Unions*; ignore that
                sizes.append(k)
            return real_sample(population, k, *a, **kw)

        random.sample = recording_sample
        try:
            sim = make_sim("modern_human", 500)
            for _ in range(4):
                step(sim)
        finally:
            random.sample = real_sample
        self.assertTrue(sizes, "no partners were ever sampled, so this test exercised nothing")
        self.assertLessEqual(max(sizes), 20, "something is still drawing the 60-person union sample")


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


if __name__ == "__main__":
    unittest.main()
