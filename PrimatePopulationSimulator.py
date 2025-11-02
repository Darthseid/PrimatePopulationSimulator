import random
import math
import json
from matplotlib.pylab import f
import numpy as np
import matplotlib.pyplot as plt
import time  # Add this import at the top

from PopulationObjects import Primate, Locale, convert_years_to_string
from PopulationObjects import SimulationParameters
from PopulationObjects import calculate_age_based_fertility, calculate_carrying_capacity

earth_year = 365.2422
starting_population = 100

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
        if self.params.is_hermaphrodite:
            print("Species Type: Hermaphroditic")
        if self.params.is_sequential_species:
            print("Species Type: Sequential Hermaphroditic")
        if self.params.ages_backward:
            print("Species Type: Ages Backward (Merlin-style)")
            
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
                    is_female = True if self.params.is_hermaphrodite else primate_data["is_female"]
                    
                    # --- PASS PARAMS TO PRIMATE ---
                    primate = Primate(
                        is_female=is_female,
                        age_days=primate_data["age_days"], # Primate __init__ will handle conversion
                        is_initially_fertile=primate_data["is_initially_fertile"],
                        params=self.params # Pass self.params
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
            
            is_female = True if self.params.is_hermaphrodite else (random.random() < self.params.sex_ratio_at_birth)
            if self.params.species_name == "sequents":
                if start_age < 12783:
                    is_female = False # Start as male
                else:
                    is_female = True # Already transformed
            
            is_initially_fertile = random.random() > self.params.sterile_chance
            
            # --- PASS PARAMS TO PRIMATE ---
            primate = Primate(
                is_female=is_female, 
                age_days=start_age, # Primate __init__ will handle conversion
                is_initially_fertile=is_initially_fertile,
                params=self.params # Pass self.params
            )
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

        if self.params.effective_gestation_days <= 0:
            print("Warning: effective_gestation_days is zero or negative. Simulation may not run correctly.")
            cycle_interval = 1
        else:
            cycle_interval = max(1, int(total_days / (5 * self.params.effective_gestation_days)))


        while self.current_day < total_days:
            if cycle == 1:
                days_to_advance = self.params.gestation_days
            else:
                days_to_advance = self.params.effective_gestation_days
            
            if days_to_advance <= 0:
                print("Error: days_to_advance is zero or negative. Stopping simulation.")
                break

            self.current_day += days_to_advance
            cycle_days_passed += days_to_advance
            
            new_population = []
            birth_counter = 0
            death_counter = 0
            eligible_female_counter = 0
            mothers_who_gave_birth = set()
            newborns = []
            
            female_count = 0
            male_count = 0
            fertile_male_count = 0
            fertile_female_count = 0

            for primate in self.population:
                # --- 1a. Age Primate (Merlin logic) ---
                if self.params.ages_backward:
                    primate.age_days -= days_to_advance # Age decreases
                else:
                    primate.age_days += days_to_advance # Age increases
                
                # 1b. Sequential hermaphrodite check
                if self.params.species_name == "sequents" and not primate.is_female and primate.age_years > (12783 / earth_year):
                    primate.is_female = True
                    primate.age_days = 5479

                # --- 1c. Check Death (Merlin logic) ---
                died = False
                
                if self.params.ages_backward:
                    if primate.age_days <= 0: # Death by old age for Merlins
                        died = True
                        total_OldAgeDeaths += 1
                else:
                    # Standard "old age" death check
                    if primate.age_days > self.params.lifespan_days:
                        base_mortality = 0.0005
                        mortality_increase = 0.09
                        lifespan_modifier = 0.93 if not primate.is_female else 1.0
                        
                        age_in_years = primate.age_years 
                        lifespan_in_years = self.params.lifespan_days / earth_year
                        
                        adjusted_age = (age_in_years / lifespan_modifier) * (age_in_years / lifespan_in_years)
                        factor = (base_mortality / mortality_increase) * math.exp(mortality_increase * adjusted_age) * (math.exp(mortality_increase) - 1)
                        mortality_rate = 1 - math.exp(-factor)

                        if random.random() < mortality_rate:
                            died = True
                            total_OldAgeDeaths += 1
                
                if died:
                    death_counter += 1
                     # --- NEW RESPAWN LOGIC (DOUBLES) ---
                    if self.params.species_name == "Doubles" and primate.is_female:
                        respawned_male = Primate(
                            params=self.params,
                            is_female=False,
                            age_days=4748, #Age 13 years
                            is_initially_fertile=random.random() > self.params.sterile_chance 
                        )
                        newborns.append(respawned_male) # Add to newborns list
                    continue  # Primate died, don't add to new population
                
                # 2. Primate survives, add to new population list and count stats
                new_population.append(primate)
                
                if primate.is_female:
                    female_count += 1
                    if primate.is_fertile and self.params.puberty_age_days <= primate.age_years * earth_year < self.params.menopause_age_days:
                        fertile_female_count += 1
                else:
                    male_count += 1
                    if primate.is_fertile and primate.age_years * earth_year >= self.params.puberty_age_days:
                        fertile_male_count += 1
            
            # 3. Calculate this cycle's population-wide modifiers
            if self.params.is_hermaphrodite:
                female_count = len(new_population) # Recalculate based on survivors
                male_count = 0
                fertile_male_count = 0
                fertile_female_count = sum(1 for p in new_population if p.is_fertile and self.params.puberty_age_days <= p.age_years * earth_year < self.params.menopause_age_days)
                breeding_population = fertile_female_count
                marriage_chance = self.params.coupling_rate
            else:
                breeding_population = (4 * fertile_male_count * fertile_female_count) / max(1, fertile_male_count + fertile_female_count)
                sex_ratio = male_count / max(1, female_count)
                marriage_chance = self.params.coupling_rate * np.sqrt(sex_ratio)

            genetic_adjuster = min(1.0, breeding_population / 50.0)
            genetic_adjuster *= self.genetic_diversity

            adjusted_adult_mortality = self.params.per_cycle_adult_mortality_rate * (1.0 + (1.0 -  genetic_adjuster)) ** 1.59 #This caps it at a 3x multiplier when genetic adjuster is very low.
            adjusted_infant_mortality = self.params.infant_mortality_rate * (1.0 + (1.0 -  genetic_adjuster)) ** 1.59

            # 4. Birth and non-age-related Death loop
            for mother in new_population:
                if mother.age_years * earth_year >= self.params.puberty_age_days and not mother.is_coupled:
                    mother.is_coupled = random.random() < marriage_chance
                
                # We use .age_years property, which works for all species
                is_eligible = (
                    mother.is_female and mother.is_fertile and mother.is_coupled and
                    self.params.puberty_age_days <= mother.age_years * earth_year < self.params.menopause_age_days and
                    mother.number_of_healthy_children < self.params.max_kids_per_primate
                )
                if not is_eligible:
                    continue
                eligible_female_counter += 1
                
                # Check for contraceptive use
                contraceptive_use = random.random() < self.params.contraception_abortion_use_rate
                mother_age_years = mother.age_years

                if self.params.fertility_rising_steepness < 0.01 and self.params.fertility_falling_steepness < 0.01:
                    current_fertility_rate = self.params.effective_per_cycle_fertility_rate
                else:
                        # Original dynamic fertility calculation
                        fertile_years = self.params.fertile_days / earth_year
                        peak_age = self.params.puberty_age_days / earth_year + fertile_years * 0.127
                        rising_midpoint = (self.params.puberty_age_days / earth_year + peak_age ) / 1.6
                        declining_midpoint = (peak_age + self.params.menopause_age_days / earth_year) / 1.95
                        current_fertility_rate = calculate_age_based_fertility(
                        current_age=mother_age_years, 
                        max_fertility=self.params.effective_per_cycle_fertility_rate, 
                        rising_steepness=self.params.fertility_rising_steepness, 
                        rising_midpoint_age=rising_midpoint, 
                        falling_steepness=self.params.fertility_falling_steepness, 
                        falling_midpoint_age=declining_midpoint
                    )
                
                if contraceptive_use:
                    current_fertility_rate *= 0.123 # More accurate
                    
                if random.random() <= max(0, current_fertility_rate):
                    mothers_who_gave_birth.add(mother)
                    num_births = 1
                    while random.random() <= self.params.chance_of_multiple_birth:
                        num_births += 1

                    for _ in range(num_births):
                        if random.random() > adjusted_infant_mortality:
                            is_female_child = True if self.params.is_hermaphrodite else (random.random() < self.params.sex_ratio_at_birth)                            
                                
                            child = Primate(
                                is_female=is_female_child,
                                age_days=self.params.lifespan_days if self.params.ages_backward else 0,
                                is_initially_fertile=random.random() > self.params.sterile_chance,
                                params=self.params # Pass params
                            )
                            newborns.append(child)
                            mother.number_of_healthy_children += 1
                            birth_counter += 1
                        else:
                            death_counter += 1
                            if self.params.species_name == "Doubles" and primate.is_female:
                                respawned_male = Primate(
                                params=self.params,
                                is_female=False,
                                age_days=4748, #Age 13 years
                                is_initially_fertile=random.random() > self.params.sterile_chance                                                        )
            
            # 5. Final death check (maternal and adult mortality)
            final_survivors = []
            for primate in new_population:
                died = False
                if primate in mothers_who_gave_birth and random.random() <= self.params.maternal_mortality_rate:
                    died = True
                # Use age_years > 0.5 to avoid killing newborns with adult mortality
                elif primate.age_years > 0.5 and random.random() < adjusted_adult_mortality:
                    died = True
                
                if died:
                    death_counter += 1
                    if self.params.species_name == "Doubles" and primate.is_female:
                        respawned_male = Primate(
                            params=self.params,
                            is_female=False,
                            age_days=4748, #Age 13 years
                            is_initially_fertile=random.random() > self.params.sterile_chance 
                        )
                else:
                    final_survivors.append(primate)
            
            # 6. Combine survivors and newborns
            self.population = final_survivors + newborns
           # self.carrying_capacity += death_counter // 10
            # 7. Apply Carrying Capacity Culling
            if len(self.population) > self.carrying_capacity:
                num_to_cull = len(self.population) - self.carrying_capacity
                death_counter += num_to_cull
                self.population = random.sample(self.population, self.carrying_capacity)

            total_births += birth_counter
            total_deaths += death_counter

            # 8. Log stats
            log_check = False
            if cycle_interval > 0:
                if cycle_days_passed >= self.params.effective_gestation_days * cycle_interval:
                    log_check = True
            elif cycle % 10 == 0: # Fallback log for very short cycles
                 log_check = True

            if log_check or (self.current_day >= total_days): # Always log last cycle
                self._log_population_stats(cycle, birth_counter, death_counter, eligible_female_counter)
                cycle_days_passed = 0
            
            # 9. Check for extinction
            if not self.params.is_hermaphrodite and not self.params.is_sequential_species:
                if not any(p.is_female for p in self.population) or not any(not p.is_female for p in self.population):
                    print(f"\n--- Simulation Terminated Early on cycle {cycle} ---")
                    print("Reason: One gender has gone extinct.")
                    break
            
            if not self.population:
                print(f"\n--- Simulation Terminated Early on cycle {cycle} ---")
                print("Reason: Population is extinct.")
                break
                
            cycle += 1

        print("\n--- Simulation Finished ---")
        total_duration = self.current_day / earth_year
        
        initial_pop_size = self.history[0]['population'] if self.history else 1

        # Use age_years for TFR calculation
        agents_past_menopause = [p for p in self.population if p.age_years * earth_year >= self.params.menopause_age_days]
        agents_at_lifespan = [
            p for p in self.population 
            if p.age_years * earth_year >= (self.params.lifespan_days * 0.98)
        ]
        
        total_fertility_rate = 0.0
        
        if agents_at_lifespan: # Priority for Peaker-style
             total_children_for_tfr = sum(p.number_of_healthy_children for p in agents_at_lifespan)
             total_fertility_rate = (total_children_for_tfr / len(agents_at_lifespan)) if len(agents_at_lifespan) > 0 else 0.0
             total_fertility_rate /= (1-self.params.infant_mortality_rate) if (1-self.params.infant_mortality_rate) > 0 else 1
        elif agents_past_menopause: # Fallback for menopause-style
            total_children_for_tfr = sum(p.number_of_healthy_children for p in agents_past_menopause)
            total_fertility_rate = (total_children_for_tfr / len(agents_past_menopause)) if len(agents_past_menopause) > 0 else 0.0
            total_fertility_rate /= (1-self.params.infant_mortality_rate) if (1-self.params.infant_mortality_rate) > 0 else 1

        population_over_time = [h['population'] for h in self.history if h['cycle'] != 0]
        average_population = sum(population_over_time) / len(population_over_time) if population_over_time else initial_pop_size
        total_duration_years = max(1, total_duration)
        final_population = len(self.population)
        
        calculated_birth_rate = (total_births / average_population / total_duration_years) * 1000 if average_population > 0 and total_duration_years > 0 else 0
        calculated_death_rate = (total_deaths / average_population / total_duration_years) * 1000 if average_population > 0 and total_duration_years > 0 else 0
        
        median_age_years = 0.0
        if final_population > 0:
            sorted_ages_years = sorted([p.age_years for p in self.population])
            mid_index = final_population // 2
            if final_population % 2 == 0: # Average of two middle elements for even population
                median_age_years = (sorted_ages_years[mid_index - 1] + sorted_ages_years[mid_index]) / 2
            else:
                median_age_years = sorted_ages_years[mid_index] # Middle element for odd population

        print(f"Final Population: {final_population:,d}")
        print("It has been", convert_years_to_string(total_duration))
        print(f"Total Births: {total_births:,d}")
        print(f"Total Deaths: {total_deaths:,d}")
        if total_deaths > 0:
            print(f"Percent that died of old age: {total_OldAgeDeaths / total_deaths:.2%}")
        else:
            print("Percent that died of old age: N/A (0 deaths)")
        
        print("Total Cycle Count:", cycle - 1)
        print(f"Total Fertility Rate (avg children for individuals past reproductive age): {total_fertility_rate:.2f}")
        print(f"Crude Birth Rate (per 1,000/year, based on avg pop): {calculated_birth_rate:.2f}")
        print(f"Crude Death Rate (per 1,000/year, based on avg pop): {calculated_death_rate:.2f}")
        print(f"Rate of Natural Increase: {calculated_birth_rate - calculated_death_rate:.2f} per 1,000/year")
        print(f"Population Change: {(len(self.population) / initial_pop_size * 100):.2f}%" if initial_pop_size > 0 else "N/A")
        
        self.display_population_pyramid()

        end_time = time.time()
        runtime = end_time - start_time
        print(f"\nSimulation Runtime: {runtime:.2f} seconds") # Add runtime calculation and display at the end
        self.plot_population_history()
        
    def _log_population_stats(self, cycle, births_this_cycle, deaths_this_cycle, potential_mother_counter):
        total_pop = len(self.population)
        
        median_age_years = 0.0
        if total_pop > 0:
            # Use .age_years property
            if total_pop > 5000:
                sample_population = random.sample(self.population, 1000)
                sample_ages_years = np.array([p.age_years for p in sample_population])
                median_age_years = np.median(sample_ages_years)
            else:    # Original, exact calculation for smaller pops
                ages_in_years = np.array([p.age_years for p in self.population])
                median_age_years = np.median(ages_in_years)

        print(f"\n--- Cycle: {cycle} (Day: {self.current_day}) Year: {self.current_day / earth_year:.1f} ---")
        print(f"Total Population: {total_pop:,d}")
        print(f"  - Median Age: {median_age_years:.1f} years")

        if not self.params.is_hermaphrodite:
            females = sum(1 for p in self.population if p.is_female)
            males = total_pop - females
            sex_ratio = males / females if females > 0 else float('inf')
            print(f"  - Females: {females:,d}")
            print(f"  - Males: {males:,d}")
            print(f"  - Sex Ratio (M/F): {sex_ratio:.2f}")
        
        if cycle != 0 and cycle != "Final":
            print(f"Births This Cycle: {births_this_cycle:,d}")
            print(f"Deaths This Cycle: {deaths_this_cycle:,d}")
            print(f"Potential Mothers: {potential_mother_counter:,d}")

        self.history.append({'cycle': cycle, 'population': total_pop, 'females': total_pop if self.params.is_hermaphrodite else females, 'males': 0 if self.params.is_hermaphrodite else males, 'current_day': self.current_day})

    def display_population_pyramid(self):
        if not self.population:
            print("\n--- Population Pyramid ---")
            print("Population is extinct.")
            return

        print("\n--- Population Pyramid ---")
        
        # Use .age_years property
        max_age_obj = max(self.population, key=lambda p: p.age_years, default=None)
        if not max_age_obj:
            print("Population is extinct.")
            return
            
        max_age = round(max_age_obj.age_years)
        
        if earth_year <= 0:
            print("Error: earth_year is zero or negative.")
            return
            
        lifespan_interval = self.params.lifespan_days // round(earth_year) if round(earth_year) != 0 else 1
        bracket_size = max(1, lifespan_interval // 15)
        
        if bracket_size <= 0:
             bracket_size = 1 # Ensure bracket size is positive
             
        brackets = range(0, (max_age // bracket_size) * bracket_size + bracket_size, bracket_size)
        
        age_distribution = {f"{i}-{i+bracket_size-1}": {"male": 0, "female": 0} for i in brackets}
        
        for p in self.population:
            age_in_years = int(p.age_years) # Use .age_years property
            bracket_start = (age_in_years // bracket_size) * bracket_size
            bracket_key = f"{bracket_start}-{bracket_start+bracket_size-1}"
            if bracket_key in age_distribution:
                if p.is_female: # This will be true for all hermaphrodites
                    age_distribution[bracket_key]["female"] += 1
                else:
                    age_distribution[bracket_key]["male"] += 1
       
        max_count_in_bracket = 1
        for data in age_distribution.values():
            max_count_in_bracket = max(max_count_in_bracket, data['male'], data['female'])
            
        pyramid_width = 30
        scale = pyramid_width / max_count_in_bracket if max_count_in_bracket > 0 else 1
        
        if self.params.is_hermaphrodite:
            print(f"| Age | {'Individuals'.ljust(pyramid_width * 2)}")
            print(f"+-----+--{'-' * (pyramid_width * 2)}")
            for bracket_label in sorted(age_distribution.keys(), key=lambda x: int(x.split('-')[0])):
                data = age_distribution[bracket_label]
                female_bar = '█' * int(data['female'] * scale) # All agents are on this side
                print(f"| {bracket_label.center(5)} | {female_bar.ljust(pyramid_width * 2)}")
        else:
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
            if tick_interval <= 0:
                tick_interval = 1
            max_year = int(total_duration_years) + tick_interval
            plt.xticks(range(0, max_year, tick_interval))

        
        max_population = max(populations) if populations else 1 # Y-axis scaling
        min_population = min(populations) if populations else 0
        population_range = max_population - min_population
       
        if population_range > 0:
            log_range = math.log10(population_range) if population_range > 1 else 0.1
            if log_range < 1.0:
                magnitude = 1
            elif population_range > 1:
                magnitude = 5 ** math.floor(log_range) #Tick marks of 50 on the Y axis. 
            else:
                magnitude = 1
            
            tick_size = magnitude  
            if population_range / magnitude < 5:
                tick_size = magnitude / 2
            elif population_range / magnitude > 10:
                tick_size = magnitude * 2
            
            tick_size = max(1, round(tick_size)) # Ensure tick size is at least 1 and an integer
            
            y_min = math.floor(min_population / tick_size) * tick_size
            y_max = math.ceil(max_population / tick_size) * tick_size
            
            if y_min == y_max:
                y_min = max(0, y_min - tick_size)
                y_max = y_max + tick_size

            y_ticks = np.arange(y_min, y_max + tick_size, tick_size)
            plt.yticks(y_ticks) # Create y-axis ticks from floor to ceiling with calculated interval
        elif max_population > 0:
             plt.yticks(np.arange(0, max_population + 1, max(1, int(max_population / 5))))
        else:
             plt.yticks([0, 1])


        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        
        print("\nDisplaying population graph...")
        plt.show()

if __name__ == "__main__":
    sim_params = SimulationParameters.from_json("demographics.json", "treant")
    sim_locale = Locale.from_json("locales.json", "greenland_coast")
    #simulation = PrimateSimulation(params=sim_params, locale=sim_locale, scenario_name="bounty_mutiny")
    simulation = PrimateSimulation(params=sim_params, locale=sim_locale) # For a random start
    simulation.run_simulation(num_years=400.0)

