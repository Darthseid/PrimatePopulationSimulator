"""
Tests for the array-based population state (population_state.py) and its operations (population_ops.py).

    python -m unittest test_population_ops -v        (run from the repository folder)

Most of these are differential tests: the vectorized code is compared with a plain Python implementation of the rule it
replaced (the original per-individual loops), on random inputs and on every species in demographics.json.
"""
import json
import math
import os
import random
import unittest

import numpy as np

import population_ops as ops
from PopulationObjects import Primate, SimulationParameters, calculate_age_based_fertility, earth_year as E
from population_state import NameTable, ParameterTable, PopulationState

HERE = os.path.dirname(os.path.abspath(__file__))


def setUpModule():
    os.chdir(HERE)


def all_species():
    with open("demographics.json") as f:
        keys = list(json.load(f))
    return {k: SimulationParameters.from_json("demographics.json", k) for k in keys}


def new_world(params_by_name):
    names = NameTable()
    table = ParameterTable(names)
    rows = {k: table.row_for(p) for k, p in params_by_name.items()}
    return names, table, rows


class FixedRng:
    """Stands in for numpy's Generator where a test needs to control the random numbers."""

    def __init__(self, uniform=None):
        self.uniform = uniform

    def random(self, n):
        return self.uniform[:n]

    def permutation(self, n):
        return np.arange(n)   # keeps the given order, so a test can compare feeding with a plain loop


class TestState(unittest.TestCase):
    def test_round_trip_through_primates_keeps_everything_including_union_and_species_name(self):
        params = all_species()
        names, table, _ = new_world({"modern_human": params["modern_human"], "orc": params["orc"]})
        man = Primate("modern_human", False, 9000, True, params["modern_human"])     # founder naming (the profile key)
        woman = Primate("modern_humans", True, 9100, True, params["modern_human"])   # newborn naming (species_name)
        from PopulationObjects import Union
        union = Union("monogamy", 2)
        union.add_member(man)
        union.add_member(woman)
        man.number_of_healthy_children = 2
        woman.next_breeding_day = 123.0
        single = Primate("orc", True, 5000, False, params["orc"])
        state = PopulationState.from_primates([man, woman, single], table, names)

        self.assertNotEqual(state.species_id[0], state.species_id[1], "founder and newborn names must stay distinct")
        self.assertEqual(state.params_id[0], state.params_id[1])
        self.assertEqual(state.union_id[0], state.union_id[1])
        self.assertEqual(state.union_id[2], PopulationState.NO_UNION)

        back = state.to_primates(table, names)
        self.assertEqual([(p.species_name, p.is_female, p.age_days, p.is_fertile, p.number_of_healthy_children) for p in back],
                         [("modern_human", False, 9000, True, 2), ("modern_humans", True, 9100, True, 0), ("orc", True, 5000, False, 0)])
        self.assertEqual(back[1].next_breeding_day, 123.0)
        self.assertIs(back[0].union, back[1].union)
        self.assertIsNone(back[2].union)

    def test_take_and_append_keep_all_arrays_aligned(self):
        params = all_species()
        names, table, _ = new_world({"modern_human": params["modern_human"]})
        people = [Primate("modern_human", i % 2 == 0, 1000 + i, True, params["modern_human"]) for i in range(10)]
        state = PopulationState.from_primates(people, table, names)
        kept = state.take(np.arange(10) % 3 == 0)
        both = kept.append(state.take([1, 2]))
        self.assertEqual(len(both), 6)
        for name, _ in PopulationState.COLUMNS:
            self.assertEqual(len(getattr(both, name)), 6)
        self.assertEqual(both.age_days.tolist(), [1000, 1003, 1006, 1009, 1001, 1002])

    def test_identical_hybrid_blends_share_a_parameter_row(self):
        params = all_species()
        names, table, _ = new_world({"modern_human": params["modern_human"]})
        random.seed(1)
        first = table.row_for_equivalent(SimulationParameters.from_parents(params["modern_human"], params["modern_human"]))
        second = table.row_for_equivalent(SimulationParameters.from_parents(params["modern_human"], params["modern_human"]))
        self.assertEqual(first, second)
        self.assertEqual(table.size, 2)


class TestFormulas(unittest.TestCase):
    def setUp(self):
        self.params = all_species()
        self.names, self.table, self.rows = new_world(self.params)
        random.seed(5)

    def population_of_everyone(self, per_species=40):
        people = []
        for key, p in self.params.items():
            for _ in range(per_species):
                people.append(Primate(key, random.random() < 0.5, random.uniform(1, 1.3 * p.lifespan_days), True, p))
        return people, PopulationState.from_primates(people, self.table, self.names)

    def test_caloric_need_matches_primate_for_every_species(self):
        people, state = self.population_of_everyone()
        expected = np.array([p.get_caloric_need() for p in people])
        np.testing.assert_array_equal(ops.caloric_need(state, self.table), expected)

    def test_fertility_curve_matches_the_scalar_function_for_every_species(self):
        ages = np.linspace(0, 120, 300)
        for key, q in self.params.items():
            args = (q.effective_per_cycle_fertility_rate, q.fertility_rising_steepness, q.fertility_rising_midpoint,
                    q.fertility_falling_steepness, q.fertility_declining_midpoint)
            expected = [calculate_age_based_fertility(a, *args) for a in ages]
            np.testing.assert_allclose(ops.age_based_fertility(ages, *args), expected, atol=1e-12, err_msg=key)

    def test_aging_and_old_age_death_match_the_scalar_rule_given_the_same_random_numbers(self):
        cycle_days = 730
        cycle_years = cycle_days / E
        people, state = self.population_of_everyone(60)
        uniform = np.random.default_rng(2).random(len(people))
        result = ops.age_and_old_age_deaths(state, self.table, self.names, FixedRng(uniform), cycle_days, cycle_years)
        for i, p in enumerate(people):
            q = p.params
            age_days = p.age_days + (-cycle_days if q.ages_backward else cycle_days)
            age_years = age_days / E
            is_female = p.is_female
            widow = 3 if (not is_female and q.lopsided_sex_lifespan) else 1
            if q.has_sequent_sex_transition and not is_female and age_years > ops.SEQUENT_TRANSITION_YEARS:
                is_female, age_days, age_years = True, 5479, 5479 / E
            if q.ages_backward:
                died = age_days <= 0
            else:
                if q.lifespan_years <= 0:
                    rate = 0.01
                else:
                    biological = age_years * widow * q.human_aging_factor
                    biological -= 4.5 if is_female else 0
                    rate = 1.0 - math.exp(-(ops.GOMPERTZ_A * math.exp(ops.GOMPERTZ_B * biological)) * cycle_years)
                died = uniform[i] < rate
            self.assertEqual(bool(result.died[i]), died, p.species_name)
            self.assertAlmostEqual(state.age_days[i], age_days, places=9)
            self.assertEqual(bool(state.is_female[i]), is_female)


class TestFeeding(unittest.TestCase):
    def test_greedy_take_is_the_sequential_first_come_first_served_rule(self):
        rng = np.random.default_rng(1)
        for _ in range(500):
            n = int(rng.integers(0, 300))
            sizes = rng.choice([500.0, 900.0, 1800.0, 2000.0, 2200.0, 4000.0], size=int(rng.integers(1, 5)), replace=False)
            needs = rng.choice(sizes, size=n) * rng.choice([1.0, 0.9, 0.5, 0.45], size=n) if n else np.zeros(0)
            available = float(rng.choice([0.0, needs.sum() * rng.random(), needs.sum() * 2, needs.sum() * 0.01]))
            expected, left = [], available
            for need in needs.tolist():   # the rule, written out
                ok = left >= need
                left -= need if ok else 0
                expected.append(ok)
            np.testing.assert_array_equal(ops.greedy_take(needs, available), expected)

    def test_a_need_that_exactly_equals_what_is_left_is_met(self):
        np.testing.assert_array_equal(ops.greedy_take(np.array([1000.0, 1000.0]), 2000.0), [True, True])
        np.testing.assert_array_equal(ops.greedy_take(np.array([1000.0, 1000.0]), 1999.9), [True, False])
        np.testing.assert_array_equal(ops.greedy_take(np.array([0.0, 5.0]), 0.0), [True, False])

    @staticmethod
    def original_feeding_loop(people, needs, cycle_days, meat, veg, grass, water):
        """The pre-array feeding loop, one individual at a time, in the given order."""
        fed_list = []
        for p, need in zip(people, needs):
            diet, fed = p.params.diet_key, False
            if diet == "autotroph":
                if water >= need: water -= need; fed = True
            elif diet == "carnivore":
                if meat >= need: meat -= need; fed = True
            elif diet == "herbivore":
                if veg >= need: veg -= need; fed = True
            elif diet == "ruminant":
                if grass >= need: grass -= need; fed = True
            elif diet == "omnivore":
                if veg >= need: veg -= need; fed = True
                elif meat >= need: meat -= need; fed = True
            elif diet == "everything":
                if grass >= need: grass -= need; fed = True
                if veg >= need: veg -= need; fed = True
                elif meat >= need: meat -= need; fed = True
            if p.params.is_aquatic:
                water -= cycle_days
                fed = water >= 0
            fed_list.append(fed)
        return fed_list

    def check_feed_against_original(self, species, trials, seed, scarce_water):
        params = all_species()
        names, table, _ = new_world(params)
        rng = np.random.default_rng(seed)
        random.seed(seed)
        cycle_days = 365
        for trial in range(trials):
            chosen = species() if callable(species) else species
            people = [Primate(k, random.random() < 0.5, random.uniform(0, params[k].lifespan_days * 1.1), True, params[k])
                      for k in chosen for _ in range(60)]
            state = PopulationState.from_primates(people, table, names)
            needs = np.array([p.get_caloric_need() * cycle_days for p in people])
            pools = {name: float(needs.sum() * rng.random() * 0.6) for name in ("meat", "veg", "grass", "water")}
            if scarce_water:   # water for only a few dozen days of drinking: the aquatic rule must bite
                pools["water"] = float(cycle_days * rng.uniform(0, 120)) if trial % 2 else float(needs.sum() * rng.random() * 0.05)
            expected = self.original_feeding_loop(people, needs.tolist(), cycle_days, pools["meat"], pools["veg"], pools["grass"], pools["water"])
            _, fed = ops.feed(state, table, FixedRng(), cycle_days, pools["meat"], pools["veg"], pools["grass"], pools["water"])
            np.testing.assert_array_equal(fed, expected, err_msg=f"trial {trial}: {chosen}")

    def test_feed_matches_the_original_loop_for_random_mixed_species(self):
        keys = list(all_species())
        self.check_feed_against_original(lambda: random.sample(keys, 6), trials=40, seed=4, scarce_water=False)

    def test_feed_matches_the_original_loop_with_aquatic_autotroph_and_every_diet_and_scarce_water(self):
        # merfolk (aquatic carnivore), treant (autotroph), butamimi ("everything"), omnivore, herbivore, ruminant
        mix = ["merfolk", "treant", "butamimi", "modern_human", "gorilla", "satyr"]
        self.check_feed_against_original(mix, trials=60, seed=6, scarce_water=True)


class TestUnions(unittest.TestCase):
    def setUp(self):
        self.params = all_species()
        self.names, self.table, _ = new_world(self.params)

    def state_of(self, people, labels):
        state = PopulationState.from_primates(people, self.table, self.names)
        state.union_id[:] = labels
        return state

    def person(self, species, female):
        return Primate(species, female, 9000, True, self.params[species])

    def test_union_statistics_and_partner_lookup(self):
        people = [self.person("modern_human", False), self.person("modern_human", True),      # pair, label 7
                  self.person("modern_human", False), self.person("modern_human", True), self.person("modern_human", True),  # trio, label 3
                  self.person("modern_human", True)]                                         # single
        state = self.state_of(people, [7, 7, 3, 3, 3, -1])
        stats = ops.UnionStats(state, self.table)
        self.assertEqual(sorted(stats.size.tolist()), [2, 3])
        self.assertTrue(stats.has_females.all() and stats.has_males.all())
        partner = stats.partners_of_pairs(len(state))
        self.assertEqual(partner.tolist(), [1, 0, -1, -1, -1, -1])

    def test_cleanup_dissolves_unions_without_both_sexes_but_not_a_lone_hermaphrodite(self):
        hermaphrodite = next(k for k, p in self.params.items() if p.is_hermaphrodite)
        people = [self.person("modern_human", False), self.person("modern_human", True),      # intact pair
                  self.person("modern_human", True), self.person("modern_human", True),       # two females
                  self.person("modern_human", False),                                          # orphan: partner died
                  self.person(hermaphrodite, True)]                                            # orphaned hermaphrodite
        state = self.state_of(people, [1, 1, 2, 2, 3, 4])
        ops.clean_unions(state, self.table)
        # A lone hermaphrodite counts as both sexes, so (as in the original) it stays in its one-person union.
        self.assertEqual(state.union_id.tolist(), [1, 1, -1, -1, -1, 4])


class TestCoupling(unittest.TestCase):
    def setUp(self):
        self.params = all_species()
        self.names, self.table, self.rows = new_world(self.params)

    def test_samples_are_distinct_and_in_range(self):
        rng = np.random.default_rng(3)
        for population, k in ((5, 5), (20, 20), (100, 20), (300, 20), (5000, 20)):
            samples = ops._distinct_samples(rng, 200, population, min(k, population))
            self.assertEqual(samples.shape, (200, min(k, population)))
            self.assertTrue(((samples >= 0) & (samples < population)).all())
            self.assertTrue((np.sort(samples, axis=1)[:, 1:] != np.sort(samples, axis=1)[:, :-1]).all(), "a sample repeated someone")

    def test_pairing_follows_the_closest_age_rule_in_order(self):
        rng = np.random.default_rng(8)
        random.seed(8)
        for trial in range(100):
            n = int(rng.integers(4, 20))   # at most 20: every attempter sees the whole pool, so the result is deterministic
            # Ages with no three in arithmetic progression, so no two candidates are ever equally close to anyone
            # (equally close candidates are tied, and ties are broken by random sample order, not by index).
            ages = rng.permutation(5000 + 7 * 2.0 ** np.arange(n))
            females = rng.random(n) < 0.5
            people = [Primate("modern_human", bool(f), float(a), True, self.params["modern_human"]) for f, a in zip(females, ages)]
            state = PopulationState.from_primates(people, self.table, self.names)
            everyone = np.ones(n, dtype=bool)
            ops.couple_singles(state, self.table, rng, everyone, everyone, marriage_chance=1.0)

            expected = {}                     # the original rule: in order, closest age among still-single opposite sexes
            for i in range(n):
                if i in expected:
                    continue
                options = [j for j in range(n) if j != i and j not in expected and females[j] != females[i]]
                if options:
                    best = min(options, key=lambda j: abs(ages[j] - ages[i]))
                    expected[i], expected[best] = best, i
            actual = {i: j for i in range(n) for j in range(n) if i != j and state.union_id[i] >= 0 and state.union_id[i] == state.union_id[j]}
            self.assertEqual(actual, expected, f"trial {trial}")

    def test_nobody_ends_up_in_two_unions_or_with_the_same_sex(self):
        rng = np.random.default_rng(9)
        n = 3000
        people = [Primate("modern_human", bool(rng.random() < 0.5), float(rng.integers(5000, 20000)), True, self.params["modern_human"]) for _ in range(n)]
        state = PopulationState.from_primates(people, self.table, self.names)
        everyone = np.ones(n, dtype=bool)
        made = ops.couple_singles(state, self.table, rng, everyone, everyone, marriage_chance=0.3)
        self.assertGreater(made, 100)
        stats = ops.UnionStats(state, self.table)
        self.assertEqual(set(stats.size.tolist()), {2})
        self.assertTrue(stats.has_females.all() and stats.has_males.all())
        self.assertEqual(len(stats.labels), made)


class TestReproduction(unittest.TestCase):
    def setUp(self):
        self.params = {"modern_human": all_species()["modern_human"]}
        self.names, self.table, self.rows = new_world(self.params)
        self.table.flat_fertility[:] = True                       # make conception near certain so the tests are not flaky
        self.table.effective_per_cycle_fertility_rate[:] = 5.0
        self.table.contraception_abortion_use_rate[:] = 0.0
        self.table.chance_of_multiple_birth[:] = 0.0
        self.table.infant_mortality_rate[:] = 0.0

    def couples(self, count, founder_names=False):
        people = []
        for i in range(count):
            for female in (False, True):
                name = "modern_human" if founder_names and not female else "modern_humans"
                people.append(Primate(name, female, 9000.0, True, self.params["modern_human"]))
        state = PopulationState.from_primates(people, self.table, self.names)
        state.union_id[:] = np.repeat(np.arange(count), 2)
        return state

    def reproduce_once(self, state, hybrids=None):
        age_years = state.age_days / E
        calls = hybrids if hybrids is not None else []
        def make_hybrid_row(mother_row, father_row):
            calls.append((mother_row, father_row))
            return mother_row
        return ops.reproduce(state, self.table, self.names, np.random.default_rng(1), 100.0, age_years, 1.0, 0, 0, make_hybrid_row), calls

    def test_every_viable_couple_has_a_child_and_both_parents_are_counted(self):
        state = self.couples(300)
        result, hybrid_calls = self.reproduce_once(state)
        self.assertEqual(result.eligible_mothers, 300)
        self.assertEqual(result.births, 300)
        self.assertEqual(len(result.newborns), 300)
        self.assertTrue((result.newborns.age_days == 0).all())
        self.assertEqual(state.children.sum(), 600, "each birth counts for the mother and for the father")
        self.assertTrue(result.gave_birth[state.is_female].all() and not result.gave_birth[~state.is_female].any())
        self.assertEqual(hybrid_calls, [], "parents of the same species name are not hybrids")
        self.assertTrue((state.next_breeding_day[state.is_female] > 100.0).all(), "mothers must wait before breeding again")

    def test_founders_and_descendants_count_as_hybrid_matings(self):
        state = self.couples(50, founder_names=True)   # the father carries the profile key, the mother the species_name
        result, hybrid_calls = self.reproduce_once(state)
        self.assertEqual(len(hybrid_calls), result.births)

    def test_those_who_may_not_breed_do_not(self):
        state = self.couples(40)
        index = np.flatnonzero(state.is_female)
        state.union_id[index[0]] = -1                                      # no union (the partner keeps a one-person union)
        state.age_days[index[1]] = 100.0                                   # a child
        state.age_days[index[2]] = 60000.0                                 # past menopause
        state.children[index[3]] = 99                                      # at the maximum number of children
        state.next_breeding_day[index[4]] = 1e9                            # not due yet
        state.is_fertile[index[5]] = False                                 # sterile
        state.union_id[index[6]] = state.union_id[index[6] - 1]            # two males? make it a same-sex union instead
        state.is_female[index[6] - 1] = True
        result, _ = self.reproduce_once(state)
        for j in range(7):
            self.assertFalse(result.gave_birth[index[j]], f"mother #{j} should not have bred")
        self.assertEqual(result.eligible_mothers, 40 - 7 - 0)

    def test_a_father_at_the_children_cap_blocks_the_birth(self):
        state = self.couples(50)
        fathers = np.flatnonzero(~state.is_female)
        state.children[fathers[:10]] = 99
        result, _ = self.reproduce_once(state)
        mothers = np.flatnonzero(state.is_female)
        self.assertFalse(result.gave_birth[mothers[:10]].any(), "a couple whose father is at the cap must not conceive")
        self.assertTrue(result.gave_birth[mothers[10:]].all())

    def test_newborns_of_backward_ageing_species_start_near_the_maximum_age(self):
        merlin = all_species()["merlin"]
        names, table, _ = new_world({"merlin": merlin})
        table.flat_fertility[:] = True
        table.effective_per_cycle_fertility_rate[:] = 5.0
        table.contraception_abortion_use_rate[:] = 0.0
        table.chance_of_multiple_birth[:] = 0.0
        table.infant_mortality_rate[:] = 0.0
        people = [Primate("merlins", female, merlin.lifespan_days / 2, True, merlin) for _ in range(300) for female in (False, True)]
        state = PopulationState.from_primates(people, table, names)
        state.union_id[:] = np.repeat(np.arange(300), 2)
        result = ops.reproduce(state, table, names, np.random.default_rng(1), 100.0, state.age_days / E, 1.0, 0, 0, lambda a, b: a)
        self.assertGreater(result.births, 200)
        ages = result.newborns.age_days
        self.assertTrue((ages > 0.4 * merlin.lifespan_days).all(), "a backward-ageing newborn must start old")
        self.assertAlmostEqual(ages.mean() / merlin.lifespan_days, 1.0, delta=0.08)

    def test_a_cycle_without_a_breeding_population_loses_every_infant_instead_of_crashing(self):
        state = self.couples(30)
        age_years = state.age_days / E
        result = ops.reproduce(state, self.table, self.names, np.random.default_rng(1), 100.0, age_years, 0.0, 0, 0, lambda a, b: a)
        self.assertEqual(result.births, 0)
        self.assertGreater(result.infant_deaths, 0)


class TestWarfare(unittest.TestCase):
    def test_winners_take_the_losers_partners_and_losers_leave_their_unions(self):
        params = all_species()
        names, table, _ = new_world({"modern_human": params["modern_human"]})
        people = []
        for _ in range(6):   # six couples; the men are the combatants
            people.append(Primate("modern_humans", False, 9000.0, True, params["modern_human"]))
            people.append(Primate("modern_humans", True, 9000.0, True, params["modern_human"]))
        state = PopulationState.from_primates(people, table, names)
        state.union_id[:] = np.repeat(np.arange(6), 2)
        men = ops.combatants_of(state, table)
        self.assertEqual(len(men), 6)
        survivors, killed = ops.resolve_warfare(state, table, np.random.default_rng(2), men)
        self.assertEqual(len(killed), 3)
        self.assertEqual(sorted(survivors + killed), sorted(men.tolist()))
        self.assertTrue((state.union_id[killed] == -1).all())
        women = np.flatnonzero(state.is_female)
        survivor_labels = set(state.union_id[survivors].tolist())
        self.assertTrue(set(state.union_id[women].tolist()) <= survivor_labels | {-1}, "a woman belongs to a dead man's union")
        harem_sizes = sorted(int(np.count_nonzero(state.union_id == label)) for label in survivor_labels if label >= 0)
        self.assertEqual(sum(harem_sizes), 3 + 6, "three surviving men plus all six women")


if __name__ == "__main__":
    unittest.main()
