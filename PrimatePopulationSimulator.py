import random
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import time  # Add this import at the top

from PopulationObjects import Primate, Locale, convert_years_to_string
from PopulationObjects import SimulationParameters
from PopulationObjects import double_logistic, calculate_carrying_capacity

earth_year = 365.2422
starting_population = 3000

class PrimateSimulation:
    """
    Manages and runs the primate population simulation.
    """
    def __init__(self, params: SimulationParameters, locale: Locale, scenario_name: str = None):
        self.params = params
        self.locale = locale
        self.population: list[Primate] = []
        self.current_day = 0
        self.history = []
        
        self.carrying_capacity = calculate_carrying_capacity(self.params, self.locale)
        print(f"Locale: {self.locale.name} ({self.locale.biome_type})")  # Calculate carrying capacity based on species and locale
        print(f"Species: {self.params.species_name} ({self.params.diet_type})")
        print(f"Calculated Carrying Capacity: {self.carrying_capacity:,d} individuals")
        
        self._create_initial_population(scenario_name)

    def _create_initial_population(self, scenario_name: str = None):
        """
        Creates the initial population, either from a scenario file or randomly.
        """
        if scenario_name:
            print(f"Loading population from scenario: {scenario_name}")
            try:
                with open("scenarios.json", 'r') as f:
                    scenarios = json.load(f)
                
                if scenario_name not in scenarios:
                    raise ValueError(f"Scenario '{scenario_name}' not found in scenarios.json")
                
                scenario_data = scenarios[scenario_name]["population"]
                for primate_data in scenario_data:
                    primate = Primate(
                        is_female=primate_data["is_female"],
                        age_days=primate_data["age_days"],
                        is_initially_fertile=primate_data["is_initially_fertile"]
                    )
                    self.population.append(primate)
                
                population_name = self.params.species_name
                print(f"Initial population from scenario created: {len(self.population)} {population_name}.")
                return

            except FileNotFoundError:
                print("Error: scenarios.json not found. Falling back to random population.")
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Error reading scenarios.json: {e}. Falling back to random population.")
        
        self._create_random_population()

    def _create_random_population(self):
        """
        Creates a randomized initial population based on simulation parameters.
        """
        print("Creating a randomized initial population.")
        min_age = 0
        max_age = self.params.lifespan_days
        population_name = self.params.species_name
        for _ in range(starting_population):
            start_age = random.uniform(min_age, max_age)
            is_female = random.random() < self.params.sex_ratio_at_birth
            is_initially_fertile = random.random() > self.params.sterile_chance
            primate = Primate(is_female=is_female, age_days=start_age, is_initially_fertile=is_initially_fertile)
            self.population.append(primate)
        print(f"Initial population created: {len(self.population)} {population_name}.")

    def run_simulation(self, num_years: float):
        start_time = time.time()  # Add this at the start of run_simulation
        print("--- Simulation Starting ---")
        self._log_population_stats(0, 0, 0, 0)

        total_births = 0
        total_deaths = 0
        total_OldAgeDeaths = 0
        self.genetic_diversity = self.params.genetic_diversity

        total_days = num_years * earth_year
        cycle = 1
        cycle_days_passed = 0

        # Calculate log interval based on total simulation length
        cycle_interval = max(1, int(total_days / (5 * self.params.effective_gestation_days)))

        while self.current_day < total_days:
            if cycle == 1:
                days_to_advance = self.params.gestation_days
            else:
                days_to_advance = self.params.effective_gestation_days

            self.current_day += days_to_advance
            cycle_days_passed += days_to_advance
            for primate in self.population:
                primate.age_days += days_to_advance

            birth_counter = 0
            death_counter = 0
            newborns = []
            eligible_female_counter = 0
            mothers_who_gave_birth = set()
            
            female_count = sum(1 for p in self.population if p.is_female)
            male_count = len(self.population) - female_count

            fertile_male_count = sum(1 for p in self.population if not p.is_female and p.is_fertile and p.age_days >= self.params.puberty_age_days)
            fertile_female_count = sum(1 for p in self.population if p.is_female and p.is_fertile and p.age_days >= self.params.puberty_age_days and p.age_days < self.params.menopause_age_days)

            breeding_population = (4 * fertile_male_count * fertile_female_count) / max(1, fertile_male_count + fertile_female_count)

            genetic_adjuster = min(1.0, breeding_population / 50.0)

            sex_ratio = male_count / max(1, female_count)
            sex_penalty = 1 / (1 + abs(sex_ratio - 1.0))

            genetic_adjuster *= sex_penalty

            adjusted_adult_mortality = self.params.per_cycle_adult_mortality_rate * (1.0 + (1.0 -  genetic_adjuster)) ** 1.1
            adjusted_infant_mortality = self.params.infant_mortality_rate * (1.0 + (1.0 -  genetic_adjuster)) ** 1.1
          
            marriage_chance = self.params.coupling_rate * sex_ratio
            for mother in self.population:
                if mother.age_days >= self.params.puberty_age_days and not mother.is_coupled:
                    mother.is_coupled = random.random() < marriage_chance
                is_eligible = (
                    mother.is_female and mother.is_fertile and mother.is_coupled and
                    self.params.puberty_age_days <= mother.age_days < self.params.menopause_age_days and
                    mother.number_of_healthy_children < self.params.max_kids_per_primate
                )
                if not is_eligible:
                    continue
                eligible_female_counter += 1
                
                contraceptive_use = False
                mother_age_years = mother.age_years
                fertile_years = self.params.fertile_days / earth_year
                peak_age = self.params.puberty_age_days / earth_year + fertile_years * 0.225

                rising_midpoint = (self.params.puberty_age_days / earth_year + peak_age ) / 2
                declining_midpoint = (peak_age + self.params.menopause_age_days / earth_year) / 2
                current_fertility_rate = double_logistic(mother_age_years, self.params.effective_per_cycle_fertility_rate, self.params.fertility_rising_steepness, rising_midpoint, self.params.fertility_falling_steepness, declining_midpoint)

                if contraceptive_use:
                    current_fertility_rate *= 0.123
                if random.random() <= max(0, current_fertility_rate):
                    mothers_who_gave_birth.add(mother)
                    num_births = 1
                    while random.random() <= self.params.chance_of_multiple_birth:
                        num_births += 1

                    for _ in range(num_births):
                        if random.random() > adjusted_infant_mortality:
                            child = Primate(
                                is_female=random.random() < self.params.sex_ratio_at_birth,
                                age_days=0,
                                is_initially_fertile=random.random() > self.params.sterile_chance
                            )
                            newborns.append(child)
                            mother.number_of_healthy_children += 1
                            birth_counter += 1
                        else:
                            death_counter += 1
               
            survivors = []
            for primate in self.population:
                died = False
                if primate in mothers_who_gave_birth and random.random() <= self.params.maternal_mortality_rate:
                    died = True
                else:
                    if primate.age_days > self.params.lifespan_days:
                        base_mortality = 0.0005
                        mortality_increase = 0.09
                        lifespan_modifier = 0.93 if not primate.is_female else 1.0
                        adjusted_age = (primate.age_days / earth_year) / lifespan_modifier * (primate.age_days / self.params.lifespan_days)
                        factor = (base_mortality / mortality_increase) * math.exp(mortality_increase * adjusted_age) * (math.exp(mortality_increase) - 1)
                        mortality_rate = 1 - math.exp(-factor)

                        if random.random() < mortality_rate:
                            died = True
                            total_OldAgeDeaths += 1
                    elif primate.age_days >= self.params.puberty_age_days // 2 and random.random() < adjusted_adult_mortality:
                        died = True
                if died:
                    death_counter += 1
                else:
                    survivors.append(primate)
            
            self.population = survivors + newborns
            
            if len(self.population) > self.carrying_capacity:
                num_to_cull = len(self.population) - self.carrying_capacity
                death_counter += num_to_cull
                self.population = random.sample(self.population, self.carrying_capacity)

            total_births += birth_counter
            total_deaths += death_counter

            if cycle_days_passed >= self.params.effective_gestation_days * cycle_interval:
                self._log_population_stats(cycle, birth_counter, death_counter, eligible_female_counter)
                cycle_days_passed = 0

            if not any(p.is_female for p in self.population) or not any(not p.is_female for p in self.population):
                print(f"\n--- Simulation Terminated Early on cycle {cycle} ---")
                print("Reason: One gender has gone extinct.")
                break
                
            cycle += 1

        print("\n--- Simulation Finished ---")
        total_duration = self.current_day / earth_year
        
        initial_pop_size = self.history[0]['population'] if self.history else 1
        females_past_menopause = [p for p in self.population if p.is_female and p.age_days >= self.params.menopause_age_days]
        total_fertility_rate = 0.0
        if females_past_menopause:
            total_children_for_tfr = sum(p.number_of_healthy_children for p in females_past_menopause)
            total_fertility_rate = total_children_for_tfr / (1-self.params.infant_mortality_rate) / len(females_past_menopause)
        
        population_over_time = [h['population'] for h in self.history if h['cycle'] != 0]
        average_population = sum(population_over_time) / len(population_over_time) if population_over_time else initial_pop_size
        total_duration_years = max(1, total_duration)
        
        calculated_birth_rate = (total_births / average_population / total_duration_years) * 1000
        calculated_death_rate = (total_deaths / average_population / total_duration_years) * 1000

        print("It has been", convert_years_to_string(total_duration))
        print(f"Total Births: {total_births}")
        print(f"Total Deaths: {total_deaths}")
        if total_deaths > 0:
            print(f"Percent that died of old age: {total_OldAgeDeaths / total_deaths:.2%}")
        else:
            print("Percent that died of old age: N/A (0 deaths)")
        
        print("Total Cycle Count:", cycle)
        print(f"Total Fertility Rate (avg children for females past menopause): {total_fertility_rate:.2f}")
        print(f"Crude Birth Rate (per 1,000/year, based on avg pop): {calculated_birth_rate:.2f}")
        print(f"Crude Death Rate (per 1,000/year, based on avg pop): {calculated_death_rate:.2f}")
        print(f"Rate of Natural Increase: {calculated_birth_rate - calculated_death_rate:.2f} per 1,000/year")
        print(f"Population Change: {len(self.population) / initial_pop_size * 100:.2f}%")
        
        self.display_population_pyramid()

        end_time = time.time()
        runtime = end_time - start_time
        print(f"\nSimulation Runtime: {runtime:.2f} seconds") # Add runtime calculation and display at the end
        self.plot_population_history()
        
    def _log_population_stats(self, cycle, births_this_cycle, deaths_this_cycle, potential_mother_counter):
        total_pop = len(self.population)
        females = sum(1 for p in self.population if p.is_female)
        males = total_pop - females
        
        sex_ratio = males / females if females > 0 else float('inf')

        print(f"\n--- Cycle: {cycle} (Day: {self.current_day}) Year: {self.current_day / earth_year:.1f} ---")
        print(f"Total Population: {total_pop}")
        print(f"  - Females: {females}")
        print(f"  - Males: {males}")
        print(f"  - Sex Ratio (M/F): {sex_ratio:.2f}")
        if cycle != 0 and cycle != "Final":
            print(f"Births This Cycle: {births_this_cycle}")
            print(f"Deaths This Cycle: {deaths_this_cycle}")
            print(f"Potential Mothers: {potential_mother_counter}")

        self.history.append({'cycle': cycle, 'population': total_pop, 'females': females, 'males': males, 'current_day': self.current_day})

    def display_population_pyramid(self):
        if not self.population:
            print("\n--- Population Pyramid ---")
            print("Population is extinct.")
            return

        print("\n--- Population Pyramid ---")
        
        max_age = round(max(p.age_years for p in self.population))
        lifespan_interval = self.params.lifespan_days // round(earth_year)
        bracket_size = max(1, lifespan_interval // 15)
        brackets = range(0, (max_age // bracket_size) * bracket_size + bracket_size, bracket_size)
        
        age_distribution = {f"{i}-{i+bracket_size-1}": {"male": 0, "female": 0} for i in brackets}
        
        for p in self.population:
            age_in_years = int(p.age_years)
            bracket_start = (age_in_years // bracket_size) * bracket_size
            bracket_key = f"{bracket_start}-{bracket_start+bracket_size-1}"
            if bracket_key in age_distribution:
                if p.is_female:
                    age_distribution[bracket_key]["female"] += 1
                else:
                    age_distribution[bracket_key]["male"] += 1
       
        max_count_in_bracket = 1
        for data in age_distribution.values():
            max_count_in_bracket = max(max_count_in_bracket, data['male'], data['female'])
            
        pyramid_width = 30
        scale = pyramid_width / max_count_in_bracket if max_count_in_bracket > 0 else 1
        
        print(f"{'Males'.rjust(pyramid_width)} | Age | {'Females'.ljust(pyramid_width)}")
        print(f'{"-"*pyramid_width}-+-----+--{"-"*pyramid_width}')

        for bracket_label in sorted(age_distribution.keys(), key=lambda x: int(x.split('-')[0])):
            data = age_distribution[bracket_label]
            male_bar = '█' * int(data['male'] * scale)
            female_bar = '█' * int(data['female'] * scale)
            print(f"{male_bar.rjust(pyramid_width)} | {bracket_label.center(5)} | {female_bar.ljust(pyramid_width)}")

    def plot_population_history(self):
        if not self.history:
            print("No history recorded, cannot plot graph.")
            return

        years = [r['current_day'] / earth_year for r in self.history]
        populations = [r['population'] for r in self.history]

        if not years:
            print("No data points to plot.")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(years, populations, marker='o', linestyle='-', color='b', markersize=4)
        
        plt.title(f"Population of {self.params.species_name} Over Time")
        plt.xlabel("Years")
        plt.ylabel("Total Population")

        
        total_duration_years = self.current_day / earth_year
        if total_duration_years > 1: # X-axis scaling
            tick_interval = math.ceil(total_duration_years / 20)
            max_year = int(total_duration_years) + tick_interval
            plt.xticks(range(0, max_year, tick_interval))

        
        max_population = max(populations) # Y-axis scaling
        min_population = min(populations)
        population_range = max_population - min_population
       
        if population_range > 0:
            magnitude = 5 ** math.floor(math.log10(population_range)) #Tick marks of 50 on the Y axis. 
            tick_size = magnitude  
            if population_range / magnitude < 5:
                tick_size = magnitude / 2
            elif population_range / magnitude > 10:
                tick_size = magnitude * 2
            
            y_min = math.floor(min_population / tick_size) * tick_size
            y_max = math.ceil(max_population / tick_size) * tick_size
            y_ticks = np.arange(y_min, y_max + tick_size, tick_size)
            plt.yticks(y_ticks) # Create y-axis ticks from floor to ceiling with calculated interval

        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        
        print("\nDisplaying population graph...")
        plt.show()

if __name__ == "__main__":
 sim_params = SimulationParameters.from_json("demographics.json", "giant")
 sim_locale = Locale.from_json("locales.json", "greenland_coast")
 #simulation = PrimateSimulation(params=sim_params, locale=sim_locale, scenario_name="bounty_mutiny")
 simulation = PrimateSimulation(params=sim_params, locale=sim_locale) # For a random start
 simulation.run_simulation(num_years=60.0)

