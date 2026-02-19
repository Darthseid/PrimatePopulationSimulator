import random
import math
import json
import numpy as np
import time

from PopulationObjects import Primate, Locale, convert_years_to_string, find_union_for_primate, Disaster
from PopulationObjects import SimulationParameters
from PopulationObjects import calculate_age_based_fertility
from graphing import log_population_stats, display_population_pyramid, plot_population_history, resolve_warfare
from typing import List

earth_year = 365.2422
starting_population = 900

class PrimateSimulation:
    """
    Manages and runs the primate population simulation.
    """
    def __init__(self, species_names: List[str], locale: Locale, scenario_name: str = None):
        self.species_params = {
            name: SimulationParameters.from_json("demographics.json", name)
            for name in species_names
        }
        self.locale = locale
        self.population: list[Primate] = []
        self.current_day = 0
        self.history = []

        self.disasters = Disaster.from_json("disasters.json")

        self.cycle_days = min(params.interbirth_interval_days for params in self.species_params.values())       
        print(f"Locale: {self.locale.name} ({self.locale.biome_type})")
        self.create_initial_population(scenario_name)

    def create_initial_population(self, scenario_name: str = None):
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
                description = scenarios[scenario_name].get("description", "No description provided.")
                print(f"Scenario Description: {description}")
                for primate_data in scenario_data:    
                    species_name = primate_data["species_name"]
                    primate = Primate(
                        species_name=species_name,
                        is_female=primate_data["is_female"],
                        age_days=primate_data["age_days"], # Primate __init__ will handle conversion
                        is_initially_fertile=primate_data["is_initially_fertile"],
                        params=self.species_params[species_name] # If a species is missing, this will raise a KeyError
                    )
                    
                    self.population.append(primate)
                return

            except FileNotFoundError:
                print("Error: scenarios.json not found. Falling back to random population.")
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Error reading scenarios.json: {e}. Falling back to random population.")
        
        self.create_random_population()

    def create_random_population(self):
        print("Creating a randomized initial population.")
        species_list = list(self.species_params.keys())
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
            is_initially_fertile = random.random() > params.sterile_chance
            primate = Primate(
                species_name = species_name,
                is_female=is_female,
                age_days=start_age,
                is_initially_fertile=is_initially_fertile,
                params=params
            )
            self.population.append(primate)
        print(f"Initial population created: {len(self.population)} individuals (random species).")

    def run_simulation(self, num_years: float):
        start_time = time.time()  # Add this at the start of run_simulation
        print("--- Simulation Starting ---")
        log_population_stats(self.current_day, self.population, self.history, 0, 0, 0, 0)

        total_births = 0
        total_deaths = 0
        total_OldAgeDeaths = 0
        # Childless tracking
        male_had_child = 0
        male_childless = 0
        female_had_child = 0
        female_childless = 0

        total_days = num_years * earth_year
        cycle = 1
        cycle_days_passed = 0
        cycle_length_in_years = self.cycle_days / earth_year
        hybridization_type = "lineal"

        if self.cycle_days <= 0: # Safety check
            cycle_interval = 1
        else:
            cycle_interval = max(1, int(total_days / (10 * self.cycle_days)))

        while self.current_day < total_days:

            self.current_day += self.cycle_days
            cycle_days_passed += self.cycle_days
            
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
            piglet_calories = 0

            # --- CALCULATE SEASON ---
            day_of_year = self.current_day % earth_year
            is_mating_season = (day_of_year < 30) or (day_of_year > (earth_year - 30)) or self.cycle_days >= 335 #December 2 to January 31.

            active_disasters = []

            for disaster in self.disasters:
                if disaster.try_trigger(self.current_day, self.cycle_days, len(self.population)):
                    print(f"*** DISASTER TRIGGERED: {disaster.name} ***")
                if disaster.is_active:
                    if disaster.check_end(self.current_day):
                        print(f"*** DISASTER ENDED: {disaster.name} ***")
                    else:
                        active_disasters.append(disaster)
            for disaster in active_disasters:
                if disaster.name == "Warfare":                     
                        combatants = [p for p in self.population if   # Identify combatants: Males, Post-Puberty, Not-Elderly
                                      not (p.is_female) and 
                                      (p.age_years * earth_year >= p.params.puberty_age_days) and 
                                      (p.age_years * earth_year < p.params.lifespan_days)]
                        
                        surviving_males = resolve_warfare(combatants)
                 
                        # Determine which combatants died and update childless/had_child counters
                        killed_combatants = [p for p in combatants if p not in surviving_males]
                        war_deaths = len(killed_combatants)
                        death_counter += war_deaths
                        for killed in killed_combatants:
                            # Only count if at or past puberty (combatants are filtered to be post-puberty already)
                            if killed.age_years * earth_year >= killed.params.puberty_age_days:
                                if killed.is_female:
                                    if killed.number_of_healthy_children > 0:
                                        female_had_child += 1
                                    else:
                                        female_childless += 1
                                else:
                                    if killed.number_of_healthy_children > 0:
                                        male_had_child += 1
                                    else:
                                        male_childless += 1
                        
                        combatant_set = set(combatants)
                        non_combatants = [p for p in self.population if p not in combatant_set]
                        self.population = non_combatants + surviving_males # Reconstruct population: Non-combatants + Survivors

            for primate in self.population:
                if primate.params.ages_backward:
                    primate.age_days -= self.cycle_days # Age decreases
                else:
                    primate.age_days += self.cycle_days # Age increases
                
                Widow_multiplier = 1
                if not primate.is_female and primate.params.has_widow_male_aging_multiplier:
                    Widow_multiplier = 3
                if primate.params.has_sequent_sex_transition and not primate.is_female and primate.age_years > (12783 / earth_year):
                    primate.is_female = True
                    primate.age_days = 5479 # 1b. Sequential hermaphrodite check
               
                died = False # --- 1c. Check Death (Merlin logic) ---               
                if primate.params.ages_backward:
                    if primate.age_days <= 0: # Death by old age for Merlins
                        died = True
                        total_OldAgeDeaths += 1
                else: # Standard "old age" death check              
                    HUMAN_STD_LIFESPAN = 81.4 # years # --- 1. DEFINE HUMAN BASELINE ---
                    age_in_years = primate.age_years * Widow_multiplier
                    species_lifespan_years = primate.params.lifespan_days / earth_year if primate.params.lifespan_days > 0 else 0.0        
                    if species_lifespan_years <= 0:
                        mortality_rate_per_cycle = 0.01
                    else:
                        aging_factor = HUMAN_STD_LIFESPAN / species_lifespan_years
                        bio_age = age_in_years * aging_factor
                        if primate.is_female:
                            bio_age -= 4.5
                        GOMPERTZ_A = 0.0001
                        GOMPERTZ_B = 0.082
                        hazard_rate_per_year = GOMPERTZ_A * math.exp(GOMPERTZ_B * bio_age)
                        years_per_cycle = self.cycle_days / earth_year
                        mortality_rate_per_cycle = 1.0 - math.exp(-hazard_rate_per_year * years_per_cycle)

                    if random.random() < mortality_rate_per_cycle:
                        died = True
                        total_OldAgeDeaths += 1
                
                if died:
                    death_counter += 1
                    # Track childless/had_child for adults (ignore pre-pubescent deaths)
                    try:
                        is_adult = primate.age_years * earth_year >= primate.params.puberty_age_days
                    except Exception:
                        is_adult = False
                    if is_adult:
                        if primate.is_female:
                            if primate.number_of_healthy_children > 0:
                                female_had_child += 1
                            else:
                                female_childless += 1
                        else:
                            if primate.number_of_healthy_children > 0:
                                male_had_child += 1
                            else:
                                male_childless += 1
                     # --- NEW RESPAWN LOGIC (DOUBLES) ---
                    if primate.params.has_double_female_respawn and primate.is_female:
                        respawned_male = Primate(
                            species_name="Doubles",
                            params=primate.params,
                            is_female=False,
                            age_days=4748, #Age 13 years
                            is_initially_fertile=random.random() > primate.params.sterile_chance 
                        )
                        newborns.append(respawned_male) # Add to newborns list
                    if primate.union:
                        primate.union.remove_member(primate)              
                    continue  # Primate died, don't add to new population                

                new_population.append(primate)
                
                if primate.is_female:
                    female_count += 1
                    if primate.is_fertile and primate.params.puberty_age_days <= primate.age_years * earth_year < primate.params.menopause_age_days:
                        fertile_female_count += 1
                else:
                    male_count += 1
                    if primate.is_fertile and primate.age_years * earth_year >= primate.params.puberty_age_days:
                        fertile_male_count += 1           
           
            if primate.params.is_hermaphrodite:
                female_count = len(new_population) # Recalculate based on survivors
                male_count = 0
                fertile_male_count = 0
                fertile_female_count = sum(1 for p in new_population if p.is_fertile and primate.params.puberty_age_days <= p.age_years * earth_year < primate.params.menopause_age_days)
                breeding_population = fertile_female_count
                marriage_chance = primate.params.coupling_rate
            else:
                breeding_population = (4 * fertile_male_count * fertile_female_count) / max(1, fertile_male_count + fertile_female_count)
                sex_ratio = male_count / max(1, female_count)
                marriage_chance = primate.params.coupling_rate * np.sqrt(sex_ratio) * cycle_length_in_years #This means women get paired off a lot when there are few of them, and rarely get paired off if they outnumber males a lot.

            genetic_adjuster = min(1.0, breeding_population / 50.0) #This is the stand-in for incest. If the breeding population is low, mortality goes up.

            if self.locale.area_km2 > 0: #Divide by Zero safety check.
                non_merfolk_population = [p for p in new_population if not p.params.excluded_from_land_density] #Species with this flag are excluded from land density (e.g., sea-dwellers).
                inhabitants_per_sq_km = len(non_merfolk_population) / self.locale.area_km2 if len(non_merfolk_population) > 0 else 1 # Avoid zero population with all merfolk
                density_penalty = min(1, 100 / inhabitants_per_sq_km) # That way below 100/km² has no advantage.
                genetic_adjuster *= density_penalty

            coupling_population = new_population if is_mating_season else [p for p in new_population if p.params.seasonal_mater is False] # Skip seasonal maters during off-season
            if not is_mating_season or self.cycle_days >= 335: # Seasonal maters form new couples every season.
                coupled_seasonal = [p for p in new_population 
                                    if p.union is not None and p.params.seasonal_mater]
                for p in coupled_seasonal:
                    if p.union:
                        p.union.remove_member(p)
       
            eligible_for_coupling = [
                p for p in coupling_population
                if p.union is None and 
                   p.is_fertile and 
                   p.age_years * earth_year >= primate.params.puberty_age_days # Get all uncoupled, fertile individuals who are of age
            ]
            if eligible_for_coupling:                          
                partner_pool = {
                    p for p in eligible_for_coupling 
                    if not (p.is_female and p.age_years * earth_year >= primate.params.menopause_age_days) #This excludes post-menopausal females.
                }
                partner_pool_list = list(partner_pool)
                coupled_primates = [p for p in coupling_population if p.union is not None]         
                for primate in eligible_for_coupling:
                        
                    if not (random.random() < marriage_chance):
                        continue

                    if primate.union is not None:
                        continue
                        
                    sample_size = min(len(partner_pool), 20) #Limit sample size for performance
                    if sample_size > 0:
                            local_pool = random.sample(partner_pool_list, sample_size)
                            sample_size_unions = min(len(coupled_primates), sample_size * 3) #Larger since there are more people than unions 
                            sampled_people = random.sample(coupled_primates, sample_size_unions)
                            unique_sampled_unions = list({p.union for p in sampled_people})
                            sample_unions = unique_sampled_unions[:sample_size] #This is all necessary to improve performance and to randomize spouses more.
                    find_union_for_primate(primate, local_pool, "monogamy", sample_unions)

            for mother in new_population:              
                is_eligible = (
                    mother.is_female and
                    mother.is_fertile and
                    mother.next_breeding_day <= self.current_day and
                    mother.union is not None and  # Check if in a union
                    mother.union.is_viable_for_breeding() and  # Check if union can breed
                    mother.params.puberty_age_days <= mother.age_years * earth_year < mother.params.menopause_age_days and
                    mother.number_of_healthy_children < mother.params.max_kids_per_primate
                )
                if not is_eligible:
                    continue
                eligible_female_counter += 1

                contraceptive_use = random.random() < mother.params.contraception_abortion_use_rate
                mother_age_years = mother.age_years

                if mother.params.fertility_rising_steepness < 0.01 and mother.params.fertility_falling_steepness < 0.01: #For skipping the fertility calculation.
                    current_fertility_rate = mother.params.effective_per_cycle_fertility_rate
                else:
                    fertile_years = mother.params.fertile_days / earth_year
                    peak_age = mother.params.puberty_age_days / earth_year + fertile_years * 0.127
                    rising_midpoint = (mother.params.puberty_age_days / earth_year + peak_age) / 1.6
                    declining_midpoint = (peak_age + mother.params.menopause_age_days / earth_year) / 1.95
                    current_fertility_rate = calculate_age_based_fertility(
                        current_age=mother_age_years,
                        max_fertility=mother.params.effective_per_cycle_fertility_rate,
                        rising_steepness=mother.params.fertility_rising_steepness,
                        rising_midpoint_age=rising_midpoint,
                        falling_steepness=mother.params.fertility_falling_steepness,
                        falling_midpoint_age=declining_midpoint
                    )

                male_fertility = 1.0
                father = None

                potential_fathers = []
                if mother.union:
                    for member in mother.union.members:
                        if member is mother:
                            continue
                        if member.params.is_hermaphrodite or not member.is_female:
                            potential_fathers.append(member)

                if potential_fathers:
                    father = random.choice(potential_fathers)
                    if father.number_of_healthy_children >= father.params.max_kids_per_primate:
                        continue # Both parents will stop having kids if either hits max
                    male_age_days = father.age_years * earth_year
                    if father.params.lifespan_days > 0:
                        age_ratio = male_age_days / father.params.lifespan_days
                        male_fertility = 1.0 / (1 + math.exp(10 * (age_ratio - 0.8)))
                    else:
                        male_fertility = 0.01

                current_fertility_rate *= male_fertility
                if contraceptive_use:
                    current_fertility_rate *= 0.123

                if random.random() <= max(0, current_fertility_rate):
                    mothers_who_gave_birth.add(mother)
                    num_births = 1
                    while random.random() <= mother.params.chance_of_multiple_birth:
                        num_births += 1

                    if father:
                        is_hybrid = mother.species_name != father.species_name
                        birth_aborted = False
                        child_params = random.choice([mother.params, father.params]) #Default and for random hybrids.
                        if is_hybrid:
                            mom_dom = mother.params.is_dominant
                            dad_dom = father.params.is_dominant                        
                            if mom_dom and dad_dom:
                                # Two different dominant species = Non-viable
                                birth_aborted = True
                            elif mom_dom:
                                child_params = mother.params
                                is_hybrid = False # Treated as pure for logic
                            elif dad_dom:
                                child_params = father.params
                                is_hybrid = False # Treated as pure for logic

                        if birth_aborted: continue
                        if hybridization_type == "lineal":  # Sons follow father's species, daughters follow mother's species.
                            sex_ratio_at_birth_chance = random.choice([mother.params.sex_ratio_at_birth, father.params.sex_ratio_at_birth])
                            if random.random() <= sex_ratio_at_birth_chance:
                                child_params = mother.params
                            else:
                                child_params = father.params
                        elif hybridization_type == "midpoint" and is_hybrid:  
                            child_params = SimulationParameters.from_parents(mother.params, father.params)
                    else:  # Asexual or hermaphrodite reproduction
                        child_params = mother.params
                    base_infant_mortality = child_params.infant_mortality_rate
                    adjusted_infant_mortality = base_infant_mortality * (1.0 + (1.0 - genetic_adjuster)) ** 1.59
                    adjusted_infant_mortality /= mother.params.genetic_diversity
                    if is_hybrid and father:
                       adjusted_infant_mortality /= father.params.genetic_diversity
                    pollution_sterility = 0.0
                    for disaster in active_disasters:
                        if disaster.name == "Plague":
                            adjusted_infant_mortality += 0.2
                        if disaster.name == "Pollution":
                          pollution_sterility += 0.7
                    for _ in range(num_births):
                        if random.random() > adjusted_infant_mortality / genetic_adjuster:
                            hybrid_sterile_chance = child_params.sterile_chance + pollution_sterility
                            hybrid_sterile_boost = 0.2
                            if hybridization_type == "lineal" and is_hybrid:
                                is_female_child = True if mother.params == child_params else False
                            else: 
                                is_female_child = True if child_params.is_hermaphrodite else (random.random() < child_params.sex_ratio_at_birth)
                            if not is_female_child:
                                hybrid_sterile_boost *= 2 #Males are more likely to be sterile in hybrids
                            if father:
                                father.number_of_healthy_children += 1
                                if is_hybrid:
                                    hybrid_sterile_chance += hybrid_sterile_boost
                            is_initially_fertile = random.random() > hybrid_sterile_chance             
                            if child_params.ages_backward:
                                if getattr(child_params, "lifespan_days", 0) and child_params.lifespan_days > 0:
                                    sigma = 0.175 # This allows one in 10,000 to live up to 150, and other Merlinsthe same probability to  be born at 40. 
                                    mu = math.log(child_params.lifespan_days) - 0.5 * sigma**2
                                    randomized_age = int(random.lognormvariate(mu, sigma))
                                else:
                                    randomized_age = 0  # NEW: If species uses ages_backward, newborns should start near lifespan (±10%)
                                age_for_child = randomized_age
                            else:
                                age_for_child = 0
                            child = Primate(
                                species_name=child_params.species_name,
                                is_female=is_female_child,
                                age_days=age_for_child,
                                is_initially_fertile=is_initially_fertile,
                                params=child_params
                            )
                            newborns.append(child)
                            mother.number_of_healthy_children += 1
                            birth_counter += 1
                        else:
                            death_counter += 1
                            if mother.params.has_double_female_respawn and mother.is_female:
                                respawned_male = Primate(
                                    species_name=mother.species_name,
                                    params=mother.params,
                                    is_female=False,
                                    age_days=4748,
                                    is_initially_fertile=random.random() > mother.params.sterile_chance
                                )
                                newborns.append(respawned_male) 
                            if father: #safety check against asexual reproduction.
                                if mother.params.produces_piglet_calories or father.params.produces_piglet_calories:
                                    piglet_calories += 1000 #There dead infants are actually piglets that others can eat.

                    # Reset breeding timer
                    mother.next_breeding_day = self.current_day + mother.params.interbirth_interval_days         

            final_survivors = [] # 5. Final death check (maternal and adult mortality)
            for primate in new_population:
                died = False
                if primate in mothers_who_gave_birth and random.random() <= primate.params.maternal_mortality_rate:
                    died = True
                else: # Use else here to group the adult mortality check
                    # Calculate adjusted mortality for this specific primate
                    adult_mortality = primate.params.adult_mortality_rate * cycle_length_in_years
                    adjusted_adult_mortality = adult_mortality * (1.0 + (1.0 - genetic_adjuster)) ** 1.59
                    for disaster in active_disasters:
                        if disaster.name == "Plague":
                            adjusted_adult_mortality += 0.2
                    if primate.age_years > 0.5 and random.random() < adjusted_adult_mortality:
                        died = True   
                
                if died:
                    death_counter += 1
                    # Track childless/had_child for adult deaths (ignore pre-pubescent)
                    try:
                        is_adult = primate.age_years * earth_year >= primate.params.puberty_age_days
                    except Exception:
                        is_adult = False
                    if is_adult:
                        if primate.is_female:
                            if primate.number_of_healthy_children > 0:
                                female_had_child += 1
                            else:
                                female_childless += 1
                        else:
                            if primate.number_of_healthy_children > 0:
                                male_had_child += 1
                            else:
                                male_childless += 1
                    if primate.params.has_double_female_respawn and primate.is_female:
                        respawned_male = Primate(
                            species_name = "Doubles",
                            params=primate.params, # Assuming self.params is available or use primate.params
                            is_female=False,
                            age_days=4748, #Age 13 years
                            is_initially_fertile=random.random() > primate.params.sterile_chance 
                        )
                        newborns.append(respawned_male)                   
                    if primate.union:
                        primate.union.remove_member(primate) 
                else:
                    final_survivors.append(primate)
            
            self.population = final_survivors + newborns # 6. Combine survivors and newborns

            famine_modifier = 1.0
            for disaster in active_disasters:
                        if disaster.name == "Famine":
                            famine_modifier /= 3.0
            avail_meat = self.locale.carnivore_calories * self.cycle_days * famine_modifier + piglet_calories
            avail_veg = self.locale.herbivore_calories * self.cycle_days * famine_modifier
            avail_grass = self.locale.ruminant_calories * self.cycle_days * famine_modifier
            avail_water = self.locale.water_availability_m3 * self.cycle_days 
            final_population = []
            
            current_living = [p for p in self.population]
            random.shuffle(current_living) 
            
            for p in current_living:
                step_need = p.get_caloric_need() * self.cycle_days
                diet = p.params.diet_type.lower()
                fed = False       
                if diet == "autotroph":
                    if avail_water >= step_need:
                        avail_water -= step_need; fed = True
                elif diet == "carnivore":
                    if avail_meat >= step_need:
                        avail_meat -= step_need; fed = True
                elif diet == "herbivore":
                    if avail_veg >= step_need:
                        avail_veg -= step_need; fed = True
                elif diet == "ruminant":
                    if avail_grass >= step_need:
                        avail_grass -= step_need; fed = True
                elif diet == "omnivore":
                    if avail_veg >= step_need:
                        avail_veg -= step_need; fed = True
                    elif avail_meat >= step_need:
                        avail_meat -= step_need; fed = True
                elif diet == "everything":
                    if avail_grass >= step_need:
                        avail_grass -= step_need; fed = True
                    if avail_veg >= step_need:
                        avail_veg -= step_need; fed = True
                    elif avail_meat >= step_need:
                        avail_meat -= step_need; fed = True    
                if p.params.requires_extra_water:
                    avail_water -= self.cycle_days
                    fed = avail_water >= 0 #merfolk need food & water.
                if fed:
                    final_population.append(p)
                else:
                    death_counter += 1
                    # Track childless/had_child for starvation deaths (ignore pre-pubescent)
                    try:
                        is_adult = p.age_years * earth_year >= p.params.puberty_age_days
                    except Exception:
                        is_adult = False
                    if is_adult:
                        if p.is_female:
                            if p.number_of_healthy_children > 0:
                                female_had_child += 1
                            else:
                                female_childless += 1
                        else:
                            if p.number_of_healthy_children > 0:
                                male_had_child += 1
                            else:
                                male_childless += 1
                    if p.union: p.union.remove_member(p)

            total_births += birth_counter
            total_deaths += death_counter
            
            self.population = final_population

            if self.current_day >= total_days or cycle % cycle_interval == 0: # Always log last cycle
                log_population_stats(self.current_day, self.population, self.history, cycle, birth_counter, death_counter, eligible_female_counter)
                cycle_days_passed = 0          
            if not primate.params.is_hermaphrodite and not primate.params.is_sequential_species:  # 9. Check for extinction
                if not any(p.is_female for p in self.population) or not any(not p.is_female for p in self.population):
                    print(f"\n--- Simulation Terminated Early on cycle {cycle} ---")
                    print("Reason: One gender has gone extinct.")
                    break
            
            if not self.population:
                print(f"\n--- Simulation Terminated Early on cycle {cycle} ---")
                print("Reason: Population is extinct.")
                break

            coupled_primates = [p for p in self.population if p.union]
            for primate in coupled_primates:
                if primate.union.is_dissolved():
                    primate.union.remove_member(primate)                       
            cycle += 1

        print("\n--- Simulation Finished ---")
        total_duration = self.current_day / earth_year
        
        initial_pop_size = self.history[0]['population'] if self.history else 1    
        
        population_over_time = [h['population'] for h in self.history if h['cycle'] != 0]
        average_population = sum(population_over_time) / len(population_over_time) if population_over_time else initial_pop_size
        total_duration_years = max(1, total_duration)
        final_population = len(self.population)
        
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
        
        final_unions_set = {p.union for p in self.population if p.union}
        final_unions_list = list(final_unions_set)
        print(f"Breeding Union Count: {len(final_unions_set)}")
        print("Total Cycle Count:", cycle - 1)
        print(f"Crude Birth Rate (per 1,000/year, based on avg pop): {calculated_birth_rate:.2f}")
        print(f"Crude Death Rate (per 1,000/year, based on avg pop): {calculated_death_rate:.2f}")
        print(f"Rate of Natural Increase: {calculated_birth_rate - calculated_death_rate:.2f} per 1,000/year")
        print(f"Population Change: {(len(self.population) / initial_pop_size * 100):.2f}%" if initial_pop_size > 0 else "N/A")
        
        sample_size = min(len(final_unions_set), 40)      
        print(f"Unions of Random Sample ({sample_size} coupled individuals):")
        if sample_size > 0:
            sampled_unions = random.sample(final_unions_list, sample_size) #Sampling a set is deprecated
            print(sampled_unions) #Only unique unions printed due to set usage
        else:
            print("[] (No coupled individuals found)")

        # Print childless ratios
        def ratio(had, childless):
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

         # Print oldest male and female primates
        if self.population:
            males = [p for p in self.population if not p.is_female]
            females = [p for p in self.population if p.is_female]
            if males:
                oldest_male = max(males, key=lambda p: p.age_days)
                print(f"Oldest Male Age: {oldest_male.age_years:.2f} years, Kids: {oldest_male.number_of_healthy_children}")
            else:
                print("Oldest Male: None")

            if females:
                oldest_female = max(females, key=lambda p: p.age_days)
                print(f"Oldest Female:  {oldest_female.age_years:.2f} years, Kids: {oldest_female.number_of_healthy_children}")
            else:
                print("Oldest Female: None")
               
        end_time = time.time()
        runtime = end_time - start_time
        print(f"\nSimulation Runtime: {runtime:.2f} seconds") # Add runtime calculation and display at the end
        display_population_pyramid(self.population, earth_year)
        plot_population_history(self.history, self.current_day)

if __name__ == "__main__":
    with open("demographics.json", "r") as f:
        demographics_data = json.load(f)
    #starting_species = list(demographics_data.keys()) #This is for simulations, otherwise manually enter the starting species.
    starting_species = ["merlin"]
    sim_locale = Locale.from_json("locales.json", "pampas")   
    simulation = PrimateSimulation(starting_species, sim_locale)  # Load multiple species   
    simulation.run_simulation(num_years=490.0) # Run the specific scenario

