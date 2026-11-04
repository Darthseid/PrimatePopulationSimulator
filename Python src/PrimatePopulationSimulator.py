import random
import math
import json
import time

import numpy as np

from PopulationObjects import Primate, Locale, convert_years_to_string, Disaster
from PopulationObjects import SimulationParameters
from population_state import NameTable, ParameterTable, PopulationState
import population_ops as ops
from population_ops import MARRIAGE_MAX_SIZE
from graphing import log_population_stats, display_population_pyramid, plot_population_history
from typing import List

earth_year = 365.2422
starting_population = 100

class PrimateSimulation:
    """
    Manages and runs the primate population simulation.

    The controller layer: it owns the clock, the disasters, the counters and the reporting, and calls the vectorized
    operations in population_ops.py on the array-based PopulationState (population_state.py) for each phase of a cycle.
    """
    def __init__(self, species_names: List[str], locale: Locale, scenario_name: str = None,
                 hybridization_type: str = "codominant", marriage_type: str = "monogamy",
                 disasters_enabled: bool = False):
        self.hybridization_type = hybridization_type
        self.marriage_type = marriage_type
        self.disasters_enabled = disasters_enabled
        self.species_params = {
            name: SimulationParameters.from_json("JSON\demographics.json", name)
            for name in species_names
        }
        self.locale = locale
        self.current_day = 0
        self.history = []

        self.rng = np.random.default_rng(random.getrandbits(64))
        self.names = NameTable()  # Seeding python's `random` before building a simulation still makes the whole run reproducible: this
        self.table = ParameterTable(self.names)
        for params in self.species_params.values():
            self.table.row_for(params)
        self.state = PopulationState(context=(self.table, self.names))
        self._fixed_blends = {}   # (mother row, father row) -> hybrid row, see make_hybrid_row

        self.disasters = Disaster.from_json("JSON\disasters.json")

        self.cycle_days = min(params.interbirth_interval_days for params in self.species_params.values())       
        print(f"Locale: {self.locale.name} ({self.locale.biome_type})")
        self.create_initial_population(scenario_name)

    @property
    def population(self):
        """The population as Primate/Union objects. Built on demand (slow for big populations): for tests and inspection."""
        return self.state.to_primates(self.table, self.names)

    @population.setter
    def population(self, primates):
        self.state = PopulationState.from_primates(primates, self.table, self.names, self.state._union_counter)

    def create_initial_population(self, scenario_name: str = None):
        """
        Creates the initial population, either from a scenario file or randomly.
        """
        if scenario_name:
            print(f"Loading population from scenario: {scenario_name}")
            try:
                with open("JSON\scenarios.json", 'r') as f:
                    scenarios = json.load(f)
                
                if scenario_name not in scenarios:
                    raise ValueError(f"Scenario '{scenario_name}' not found in scenarios.json")
                
                scenario_data = scenarios[scenario_name]["population"]
                description = scenarios[scenario_name].get("description", "No description provided.")
                print(f"Scenario Description: {description}")
                primates = []
                for primate_data in scenario_data:    
                    species_name = primate_data["species_name"]
                    primates.append(Primate(
                        species_name=species_name,
                        is_female=primate_data["is_female"],
                        age_days=primate_data["age_days"], # Primate __init__ will handle conversion
                        is_initially_fertile=primate_data["is_initially_fertile"],
                        params=self.species_params[species_name] # If a species is missing, this will raise a KeyError
                    ))
                self.population = primates
                return

            except FileNotFoundError:
                print("Error: scenarios.json not found. Falling back to random population.")
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Error reading scenarios.json: {e}. Falling back to random population.")
        
        self.create_random_population()

    def create_random_population(self):
        print("Creating a randomized initial population.")
        species_list = list(self.species_params.keys())
        ages, females, species_ids, params_ids, fertile = [], [], [], [], []
        for _ in range(starting_population): # Randomly pick a species        
            species_name = random.choice(species_list)
            params = self.species_params[species_name]
            start_age = random.randrange(params.lifespan_days)
            is_female = True if params.is_hermaphrodite else (random.random() < params.sex_ratio_at_birth)
            if params.has_sequent_sex_transition:
                if start_age < 12783:
                    is_female = False
                else:
                    is_female = True
            ages.append(start_age)
            females.append(is_female)
            fertile.append(random.random() > params.sterile_chance)
            species_ids.append(self.names.id(species_name)) # founders are named by their profile key, as before
            params_ids.append(self.table.row_for(params))
        empty = PopulationState(context=(self.table, self.names))
        self.state = PopulationState.newborns(empty, len(ages), ages, females, species_ids, params_ids, fertile)
        print(f"Initial population created: {len(self.state)} individuals (random species).")

    BLEND_RANDOM_FIELDS = ("diet_type", "is_dominant", "seasonal_mater", "ages_backward", "lopsided_sex_lifespan",
                           "has_sequent_sex_transition", "respawn_as_male", "is_aquatic", "kid_fodder",
                           "is_humanoid", "is_xenophilic")   # This is for hybrids.

    def make_hybrid_row(self, mother_row, father_row):
        """Parameter-table row for a midpoint hybrid of two parent rows (SimulationParameters.from_parents)."""
        row = self._fixed_blends.get((mother_row, father_row))
        if row is not None:
            return row
        mother, father = self.table.objects[mother_row], self.table.objects[father_row]
        row = self.table.row_for_equivalent(SimulationParameters.from_parents(mother, father))
        if all(getattr(mother, f) == getattr(father, f) for f in self.BLEND_RANDOM_FIELDS):
            self._fixed_blends[(mother_row, father_row)] = row
        return row

    def _describe_union(self, label):
        members = np.flatnonzero(self.state.union_id == label)
        parts = []
        for i in members:
            sex = '♀️ ' if self.state.is_female[i] else '♂️ '
            name = self.names[int(self.state.species_id[i])]
            parts.append(f"{sex}{name}( Age: {self.state.age_days[i] / earth_year:.0f} Kids: {int(self.state.children[i])})")
        cap = MARRIAGE_MAX_SIZE.get(self.marriage_type, 2)
        return f"<Union ({self.marriage_type}{len(members)}/{max(cap, len(members))}) | Members: [{', '.join(parts)}]>\n"

    def run_simulation(self, num_years: float):
        start_time = time.time()  # Add this at the start of run_simulation
        print("--- Simulation Starting ---")
        print(f"Settings: hybridization={self.hybridization_type}, marriage={self.marriage_type}, disasters={self.disasters_enabled}")
        log_population_stats(self.current_day, self.state, self.history, 0, 0, 0, 0)

        total_births = 0
        total_deaths = 0
        total_OldAgeDeaths = 0
        # Childless tracking
        male_had_child = 0
        male_childless = 0
        female_had_child = 0
        female_childless = 0

        def add_tally(tally): # (female_had, female_childless, male_had, male_childless) from ops.count_adult_deaths
            nonlocal female_had_child, female_childless, male_had_child, male_childless
            female_had_child += tally[0]
            female_childless += tally[1]
            male_had_child += tally[2]
            male_childless += tally[3]

        total_days = num_years * earth_year
        cycle = 1
        cycle_days_passed = 0
        cycle_length_in_years = self.cycle_days / earth_year
        cycle_days = self.cycle_days

        if self.cycle_days <= 0: # Safety check
            cycle_interval = 1
        else:
            cycle_interval = max(1, int(total_days / (10 * self.cycle_days)))

        while self.current_day < total_days:

            self.current_day += self.cycle_days
            cycle_days_passed += self.cycle_days
            
            birth_counter = 0
            death_counter = 0
            eligible_female_counter = 0
            piglet_calories = 0

            day_of_year = self.current_day % earth_year # --- CALCULATE SEASON ---
            is_mating_season = (day_of_year < 30) or (day_of_year > (earth_year - 30)) or self.cycle_days >= 335 #December 2 to January 31.

            active_disasters = []
            disasters_enabled = self.disasters_enabled
            for disaster in self.disasters:
                if disaster.try_trigger(self.current_day, self.cycle_days, len(self.state)) and disasters_enabled:
                    print(f"*** DISASTER TRIGGERED: {disaster.name} ***")
                if disaster.is_active and disasters_enabled:
                    if disaster.check_end(self.current_day):
                        print(f"*** DISASTER ENDED: {disaster.name} ***")
                    else:
                        active_disasters.append(disaster)
            for disaster in active_disasters:
                if disaster.name == "Warfare":                     
                    combatants = ops.combatants_of(self.state, self.table) # Males, Post-Puberty, Not-Elderly
                    _, killed = ops.resolve_warfare(self.state, self.table, self.rng, combatants)
                    death_counter += len(killed)
                    killed_mask = np.zeros(len(self.state), dtype=bool)
                    killed_mask[killed] = True
                    add_tally(ops.count_adult_deaths(self.state, killed_mask, self.table))
                    self.state = self.state.take(~killed_mask)

            state = self.state # ---- 1. Aging and old-age death. Deaths leave their partner in a one-person union until the cleanup at the end of the cycle, exactly as before.
            reference = self.table.objects[int(state.params_id[-1])]
            aging = ops.age_and_old_age_deaths(state, self.table, self.names, self.rng, cycle_days, cycle_length_in_years)
            total_OldAgeDeaths += aging.old_age_deaths
            death_counter += aging.old_age_deaths
            add_tally(aging.childless_tally)
            newborns = aging.respawned
            state = state.take(~aging.died) # survivors; union labels stay with them
            age_years = state.age_days / earth_year # ages are calculated once here and reused by every later phase

            counts = ops.count_by_group(state, self.table, age_years)
            female_count, male_count = counts.female, counts.male
            fertile_female_count, fertile_male_count = counts.fertile_female, counts.fertile_male
            if reference.is_hermaphrodite:
                female_count = len(state) # Recalculate based on survivors
                male_count = 0
                fertile_male_count = 0
                fertile_female_count = ops.fertile_females_by_reference_rules(
                    state, self.table, age_years, reference.puberty_age_days, reference.menopause_age_days)
                breeding_population = fertile_female_count
                marriage_chance = reference.coupling_rate
            else:
                breeding_population = (4 * fertile_male_count * fertile_female_count) / max(1, fertile_male_count + fertile_female_count)
                sex_ratio = male_count / max(1, female_count)
                marriage_chance = reference.coupling_rate * math.sqrt(sex_ratio) * cycle_length_in_years #This means women get paired off a lot when there are few of them, and rarely get paired off if they outnumber males a lot.

            genetic_adjuster = min(1.0, breeding_population / 50.0) #This is the stand-in for incest. If the breeding population is low, mortality goes up.

            if self.locale.area_km2 > 0: #Divide by Zero safety check.      
                inhabitants_per_sq_km = counts.non_aquatic / self.locale.area_km2 if counts.non_aquatic > 0 else 1 # Avoid zero population with all merfolk
                density_penalty = min(1, 100 / inhabitants_per_sq_km) # That way below 100/km² has no advantage.
                genetic_adjuster *= density_penalty      #Species with the aquatic flag are excluded from land density (e.g., sea-dwellers).
      
            p = state.params_id
            seasonal_present = counts.seasonal > 0   # ---- 2. Coupling. Seasonal maters are uncoupled and sit out the off-season; with none alive both steps are skipped.
            if seasonal_present and (not is_mating_season or self.cycle_days >= 335): # Seasonal maters form new couples every season.
                ops.uncouple(state, (state.union_id >= 0) & self.table.seasonal_truthy[p])
            can_couple = np.ones(len(state), dtype=bool)
            if seasonal_present and not is_mating_season:
                can_couple = ~self.table.seasonal_not_false[p] # Skip seasonal maters during off-season

            eligible = (can_couple & (state.union_id < 0) & state.is_fertile &
                        (age_years * earth_year >= reference.puberty_age_days)) # uncoupled, fertile individuals who are of age
            if eligible.any():
                partner_pool = eligible & ~(state.is_female &
                                            (ops.age_in_days_roundtrip(state.age_days) >= reference.menopause_age_days)) #This excludes post-menopausal females.
                ops.couple_singles(state, self.table, self.rng, eligible, partner_pool, marriage_chance,
                                   marriage_type=self.marriage_type)
         
            plague_count = sum(1 for disaster in active_disasters if disaster.name == "Plague") # ---- 3. Reproduction
            pollution_count = sum(1 for disaster in active_disasters if disaster.name == "Pollution")
            reproduction = ops.reproduce(state, self.table, self.names, self.rng, self.current_day, age_years,
                                         genetic_adjuster, plague_count, pollution_count, self.make_hybrid_row,
                                         self.hybridization_type)
            birth_counter += reproduction.births
            death_counter += reproduction.infant_deaths
            eligible_female_counter += reproduction.eligible_mothers
            piglet_calories += reproduction.piglet_calories #Dead infants are actually piglets that others can eat.
            newborns = newborns.append(reproduction.newborns)

            # ---- 4. Final death check (maternal and adult mortality)
            adult_mortality_scale = (1.0 + (1.0 - genetic_adjuster)) ** 1.59 # Identical for everyone this cycle, so compute once.
            died, tally, respawned = ops.final_deaths(state, self.table, self.names, self.rng, age_years,
                                                      reproduction.gave_birth, adult_mortality_scale, plague_count,
                                                      cycle_length_in_years)
            death_counter += int(np.count_nonzero(died))
            add_tally(tally)
            last_survivor = self.table.objects[int(state.params_id[-1])] if len(state) else reference # legacy leaked loop variable, see above
            population = state.take(~died).append(newborns).append(respawned) # 6. Combine survivors and newborns
       
            famine_modifier = 1.0  # ---- 5. Feeding: random order, first come first served
            for disaster in active_disasters:
                if disaster.name == "Famine":
                    famine_modifier /= 3.0 # This reduces available calories to 1/3 of normal, which is a severe famine.
            avail_meat = self.locale.carnivore_calories * self.cycle_days * famine_modifier + piglet_calories
            avail_veg = self.locale.herbivore_calories * self.cycle_days * famine_modifier
            avail_grass = self.locale.ruminant_calories * self.cycle_days * famine_modifier
            avail_water = self.locale.water_availability_m3 * self.cycle_days 
            shuffled, fed = ops.feed(population, self.table, self.rng, cycle_days, avail_meat, avail_veg, avail_grass, avail_water)
            starved = ~fed
            death_counter += int(np.count_nonzero(starved))
            add_tally(ops.count_adult_deaths(shuffled, starved, self.table))

            total_births += birth_counter
            total_deaths += death_counter
            
            self.state = shuffled.take(fed)

            if self.current_day >= total_days or cycle % cycle_interval == 0: # Always log last cycle
                log_population_stats(self.current_day, self.state, self.history, cycle, birth_counter, death_counter, eligible_female_counter)
                cycle_days_passed = 0          
            if not last_survivor.is_hermaphrodite and not last_survivor.is_sequential_species:  # 9. Check for extinction
                if not self.state.is_female.any() or self.state.is_female.all():
                    print(f"\n--- Simulation Terminated Early on cycle {cycle} ---")
                    print("Reason: One gender has gone extinct.")
                    break
            
            if len(self.state) == 0:
                print(f"\n--- Simulation Terminated Early on cycle {cycle} ---")
                print("Reason: Population is extinct.")
                break

            ops.clean_unions(self.state, self.table) # a union that lost a partner (or never had both sexes) is dissolved
            cycle += 1

        print("\n--- Simulation Finished ---")
        total_duration = self.current_day / earth_year
        
        initial_pop_size = self.history[0]['population'] if self.history else 1    
        
        population_over_time = [h['population'] for h in self.history if h['cycle'] != 0]
        average_population = sum(population_over_time) / len(population_over_time) if population_over_time else initial_pop_size
        total_duration_years = max(1, total_duration)
        final_population = len(self.state)
        
        calculated_birth_rate = (total_births / average_population / total_duration_years) * 1000 if average_population > 0 and total_duration_years > 0 else 0
        calculated_death_rate = (total_deaths / average_population / total_duration_years) * 1000 if average_population > 0 and total_duration_years > 0 else 0       

        print(f"Final Population: {final_population:,d}")
        print("It has been", convert_years_to_string(total_duration))
        print(f"Total Births: {total_births:,d}")
        print(f"Total Deaths: {total_deaths:,d}")
        if total_deaths > 0:
            print(f"Percent that died of old age: {total_OldAgeDeaths / total_deaths:.2%}")
        else:
            print("Percent that died of old age: N/A (0 deaths)")
        
        final_unions = np.unique(self.state.union_id[self.state.union_id >= 0])
        print(f"Breeding Union Count: {len(final_unions)}")
        print("Total Cycle Count:", cycle - 1)
        print(f"Crude Birth Rate (per 1,000/year, based on avg pop): {calculated_birth_rate:.2f}")
        print(f"Crude Death Rate (per 1,000/year, based on avg pop): {calculated_death_rate:.2f}")
        print(f"Rate of Natural Increase: {calculated_birth_rate - calculated_death_rate:.2f} per 1,000/year")
        print(f"Population Change: {(len(self.state) / initial_pop_size * 100):.2f}%" if initial_pop_size > 0 else "N/A")
        
        sample_size = min(len(final_unions), 40)      
        print(f"Unions of Random Sample ({sample_size} coupled individuals):")
        if sample_size > 0:
            sampled_unions = [self._describe_union(int(label)) for label in self.rng.choice(final_unions, sample_size, replace=False)]
            print(sampled_unions) #Only unique unions printed
        else:
            print("[] (No coupled individuals found)")
    
        def ratio(had, childless): # Print childless ratios
            denom = had + childless
            return (childless / denom) if denom > 0 else None

        male_childless_ratio = ratio(male_had_child, male_childless)
        female_childless_ratio = ratio(female_had_child, female_childless)

        if male_childless_ratio is None:
            print("Male childless ratio: N/A (no adult male deaths recorded)")
        else:
            print(f"Male childless ratio: {male_childless_ratio:.2%} ({male_childless} childless / {male_had_child + male_childless} adult male deaths)")

        if female_childless_ratio is None:
            print("Female childless ratio: N/A (no adult female deaths recorded)")
        else:
            print(f"Female childless ratio: {female_childless_ratio:.2%} ({female_childless} childless / {female_had_child + female_childless} adult female deaths)")
   
        if len(self.state):  # Print oldest male and female primates
            for label, mask in (("Male", ~self.state.is_female), ("Female", self.state.is_female)):
                if mask.any():
                    ages = np.where(mask, self.state.age_days, -np.inf)
                    oldest = int(np.argmax(ages))
                    print(f"Oldest {label} Age: {self.state.age_days[oldest] / earth_year:.2f} years, Kids: {int(self.state.children[oldest])}"
                          if label == "Male" else
                          f"Oldest {label}:  {self.state.age_days[oldest] / earth_year:.2f} years, Kids: {int(self.state.children[oldest])}")
                else:
                    print(f"Oldest {label}: None")
               
        end_time = time.time()
        runtime = end_time - start_time
        print(f"\nSimulation Runtime: {runtime:.2f} seconds") # Add runtime calculation and display at the end
        display_population_pyramid(self.state, earth_year)
        plot_population_history(self.history, self.current_day)

if __name__ == "__main__":  # --- Settings menu: change these to configure a run ---   
    hybridization_type = "codominant"  # "codominant" (default), "midpoint", or "lineal"
    marriage_type = "polygyny"         # "monogamy" (default), "polygyny", "polyandry", or "polygamy"
    disasters_enabled = False        # True to allow Warfare, Plague, Pollution, and Famine
    num_years = 99.0
    scenario_name = None  # None for a random starting population
    locale_name = "pampas"

    with open("JSON\demographics.json", "r") as f:
        demographics_data = json.load(f)
    #starting_species = list(demographics_data.keys()) #This is for simulations, otherwise manually enter the starting species.
    starting_species = [ "coonfolk", "modern_human"]
    sim_locale = Locale.from_json("JSON\locales.json", locale_name)
    simulation = PrimateSimulation(
        starting_species, sim_locale, scenario_name,
        hybridization_type=hybridization_type,
        marriage_type=marriage_type,
        disasters_enabled=disasters_enabled,
    )
    simulation.run_simulation(num_years=num_years)

