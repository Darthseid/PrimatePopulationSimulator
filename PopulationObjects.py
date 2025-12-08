import numpy as np
import random
import json
from typing import Optional, Set, List

earth_year = 365.2422  # Constants

class Primate:
    def __init__(self, species_name, is_female: bool, age_days: int, is_initially_fertile: bool, params: 'SimulationParameters'):
        self.species_name = species_name
        self.is_female: bool = is_female
        self.params = params   # Store params to know species rules (e.g., lifespan, ageing direction)
        self.age_days: int = age_days          
        self.is_fertile: bool = is_initially_fertile
        self.number_of_healthy_children: int = 0
        self.next_breeding_day = 0
        self.union: Optional['Union'] = None  # Reference to the union this primate is in

    @property
    def age_years(self) -> float:
            return self.age_days / earth_year

    def get_caloric_need(self) -> float:
        """
        Calculates the individual daily caloric/resource need for this primate.
        """
        need = self.params.calories_needed_per_primate

        if self.is_female:
            need *= 0.9
            
        biological_age_days = self.age_years * earth_year
        if biological_age_days < self.params.puberty_age_days:
            need *= 0.5
        if biological_age_days > self.params.lifespan_days:
            need *= 0.75
        return need
    
    @property
    def is_coupled(self) -> bool:
        """
        Property to check if the primate is in a union.
        """
        return self.union is not None
    

    def __repr__(self) -> str:
        species = self.species_name
        gender = "Female" if self.is_female else "Male"
        fertility = "Fertile" if self.is_fertile else "Sterile"
        coupled_status = "Coupled" if self.is_coupled else "Single"
        return (f"<Primate | Gender: {gender}, Species: {species}, Age: {self.age_years:.1f} yrs, "
                f"Status: {fertility}, {coupled_status}, Children: {self.number_of_healthy_children}>")

class Locale:
    """
    Holds all simulation parameters related to the environment/location.
    """
    @classmethod
    def from_json(cls, json_path: str, profile_name: str):
        try:
            with open(json_path, 'r') as f:
                all_params = json.load(f)
            if profile_name not in all_params:
                raise ValueError(f"Locale profile '{profile_name}' not found in {json_path}")
            params = all_params[profile_name]
            params['name'] = profile_name # Add the profile name to the params dict
            return cls(**params)
        except FileNotFoundError:
            raise FileNotFoundError(f"Locales file not found: {json_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {json_path}")

    def __init__(self, **params):
        self.name: str = params.get("name", "Unknown")
        self.biome_type: str = params.get("biome_type", "Temperate")
        self.area_km2: float = params.get("area_km2", 0)
        self.water_availability_m3: float = params.get("water_availability_m3", 0)

        self.carnivore_calories: int = params.get("carnivore_calories", 0)
        self.herbivore_calories: int = params.get("herbivore_calories", 0)
        self.ruminant_calories: int = params.get("ruminant_calories", 0) # Interpreted as gathered food (plants, fruit, nuts)

class SimulationParameters:
    """
    Holds all simulation parameters for a given species.
    """
    @classmethod
    def from_json(cls, json_path: str, profile_name: str):
        try:
            with open(json_path, 'r') as f:
                all_params = json.load(f)
            if profile_name not in all_params:
                raise ValueError(f"Species profile '{profile_name}' not found in {json_path}")
            params = all_params[profile_name]
            return cls(**params)
        except FileNotFoundError:
            raise FileNotFoundError(f"Demographics file not found: {json_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {json_path}")

    def __init__(self, **params):
        self.species_name = params["species_name"]
        self.puberty_age_days = params["puberty_age_days"]
        self.menopause_age_days = params["menopause_age_days"]
        self.lifespan_days = params["lifespan_days"] 
        
        self.coupling_rate = params["coupling_rate"] #This represents the chance of a primate being coupled with a mate per cycle.
        self.gestation_days = params["gestation_days"]
        self.interbirth_interval_days = params["interbirth_interval_days"]
        self.max_kids_per_primate = params["max_kids_per_primate"]
        self.chance_of_multiple_birth = params["chance_of_multiple_birth"]
        self.base_fertility_rate = params["base_fertility_rate"]
        self.miscarriage_stillborn_rate = params["miscarriage_stillborn_rate"]
        self.sterile_chance = params["sterile_chance"]
        self.sex_ratio_at_birth = params["sex_ratio_at_birth"]
        self.contraception_abortion_use_rate = params["contraception_abortion_use_rate"]
        
        self.is_hermaphrodite = params.get("is_hermaphrodite", False) # Use .get() for optional params
        self.is_sequential_species = params.get("is_sequential_species", False)
        self.ages_backward = params.get("ages_backward", False) # --- ADDED ---
        self.is_dominant = params.get("is_dominant", False) #Is dominant means that this species can only propagate itself.
        self.seasonal_mater = params.get("seasonal_mater", False) #This means that the species has mating cycles.
        
        self.infant_mortality_rate = params["infant_mortality_rate"]
        self.maternal_mortality_rate = params["maternal_mortality_rate"]
        self.adult_mortality_rate = params["adult_mortality_rate"]

        self.calories_needed_per_primate = params["calories_needed_per_primate"] # Calories needed *per day*
        self.diet_type = params.get("diet_type", "omnivore") # Get diet type, default to omnivore
        
        self.genetic_diversity = params.get("initial_genetic_diversity", 1.0)
        
       
        self.fertility_rising_steepness = params["fertility_rising_steepness"]  # Fertility Curve (Dynamic)
        self.fertility_falling_steepness = params["fertility_falling_steepness"]

        self.fertile_days = self.menopause_age_days - self.puberty_age_days #A female primate's reproductive lifespan.
        self.effective_gestation_days = self.gestation_days + self.interbirth_interval_days
              
        if self.effective_gestation_days > 0: # Calculate cycles per life and per-cycle fertility rate
            self.cycles_per_reproductive_life = self.fertile_days / self.effective_gestation_days #How many birthing cycles a primate potentially has.
            cycle_length_in_years = self.effective_gestation_days / earth_year
            self.per_cycle_fertility_rate = self.base_fertility_rate * cycle_length_in_years
        else:
            self.cycles_per_reproductive_life = 0
            self.per_cycle_fertility_rate = 0

        self.effective_per_cycle_fertility_rate = min(self.per_cycle_fertility_rate * (1 - self.miscarriage_stillborn_rate), 0.99999)      
        
        if self.effective_gestation_days > 0:
            cycle_length_in_years = self.effective_gestation_days / earth_year # Calculate per-cycle mortality from annual mortality rate
            self.per_cycle_adult_mortality_rate = self.adult_mortality_rate * cycle_length_in_years
        else:
            self.per_cycle_adult_mortality_rate = 0

    @classmethod
    def from_parents(cls, p1: 'SimulationParameters', p2: 'SimulationParameters'):
        """
        Creates a new SimulationParameters object by averaging two parent profiles.
        Used for "Midpoint" hybridization.
        """       
        new_params = cls.__new__(cls) # Create a new empty instance
                   
        s1_parts = set(p1.species_name.split('-'))  # 1. Strings & Categorical (Random Inheritance with De-duplication)
        s2_parts = set(p2.species_name.split('-')) # Split both parent names by hyphen to get base constituents
              
        combined_parts = sorted(list(s1_parts.union(s2_parts)))
        new_params.species_name = "-".join(combined_parts)
        new_params.diet_type = random.choice([p1.diet_type, p2.diet_type]) # Combine unique parts and sort them to ensure consistent naming (e.g. "A-B" vs "B-A")
                
        new_params.is_hermaphrodite = False
        new_params.is_sequential_species = False
        new_params.is_dominant = random.choice([p1.is_dominant, p2.is_dominant]) # This normally should not be inherited, but it throws an error otherwise.
        new_params.seasonal_mater = random.choice([p1.seasonal_mater, p2.seasonal_mater])
        new_params.ages_backward = random.choice([p1.ages_backward, p2.ages_backward]) # 2. Booleans (Midpoint Rules)
              
        new_params.puberty_age_days = (p1.puberty_age_days + p2.puberty_age_days) / 2
        new_params.menopause_age_days = (p1.menopause_age_days + p2.menopause_age_days) / 2
        new_params.lifespan_days = (p1.lifespan_days + p2.lifespan_days) / 2
        new_params.coupling_rate = (p1.coupling_rate + p2.coupling_rate) / 2  # 3. Numeric Stats (Average)
        new_params.gestation_days = (p1.gestation_days + p2.gestation_days) / 2
        new_params.interbirth_interval_days = (p1.interbirth_interval_days + p2.interbirth_interval_days) / 2
        new_params.max_kids_per_primate = int((p1.max_kids_per_primate + p2.max_kids_per_primate) / 2)
        new_params.chance_of_multiple_birth = (p1.chance_of_multiple_birth + p2.chance_of_multiple_birth) / 2
        new_params.base_fertility_rate = (p1.base_fertility_rate + p2.base_fertility_rate) / 2
        new_params.miscarriage_stillborn_rate = (p1.miscarriage_stillborn_rate + p2.miscarriage_stillborn_rate) / 2
        new_params.sterile_chance = (p1.sterile_chance + p2.sterile_chance) / 2
        new_params.sex_ratio_at_birth = (p1.sex_ratio_at_birth + p2.sex_ratio_at_birth) / 2
        new_params.infant_mortality_rate = (p1.infant_mortality_rate + p2.infant_mortality_rate) / 2
        new_params.maternal_mortality_rate = (p1.maternal_mortality_rate + p2.maternal_mortality_rate) / 2
        new_params.adult_mortality_rate = (p1.adult_mortality_rate + p2.adult_mortality_rate) / 2
        new_params.calories_needed_per_primate = (p1.calories_needed_per_primate + p2.calories_needed_per_primate) / 2
        new_params.genetic_diversity = (p1.genetic_diversity + p2.genetic_diversity) / 1.5 #Heterosis
        new_params.fertility_rising_steepness = (p1.fertility_rising_steepness + p2.fertility_rising_steepness) / 2
        new_params.fertility_falling_steepness = (p1.fertility_falling_steepness + p2.fertility_falling_steepness) / 2
        new_params.contraception_abortion_use_rate = (p1.contraception_abortion_use_rate + p2.contraception_abortion_use_rate) / 2
       
        new_params.effective_gestation_days = new_params.gestation_days + new_params.interbirth_interval_days 
        cycle_length_in_years = new_params.effective_gestation_days / earth_year
        new_params.per_cycle_fertility_rate = new_params.base_fertility_rate * cycle_length_in_years
        new_params.fertile_days = new_params.menopause_age_days - new_params.puberty_age_days  # 4. Recalculate Derived Parameters    
            
        new_params.effective_per_cycle_fertility_rate = min(new_params.per_cycle_fertility_rate * (1 - new_params.miscarriage_stillborn_rate), 0.99999 )
                                                                          
        return new_params

class Union:
    """
    Represents a relationship (couple, harem, etc.) for breeding.
    """
    def __init__(self, marriage_type: str = "monogamy", max_size: int = 2):
        self.marriage_type = marriage_type
        self.max_size = max_size
        self.members: List[Primate] = []
        self.dissolved = False   # <-- CRUCIAL FIX

    def add_member(self, primate: Primate):
        if len(self.members) < self.max_size:
            self.members.append(primate)
            primate.union = self  # Set back-reference to union

    def remove_member(self, primate):
        if primate in self.members:
            self.members.remove(primate)
            primate.union = None
        if not self.members:  # If empty, mark as dissolved
            self.dissolved = True

    def is_dissolved(self) -> bool:
        """Hard correctness rules."""
        if self.dissolved:
            return True

        if len(self.members) == 0:
            return True

        if self.marriage_type == "asexual":
            return len(self.members) != 1

        return not self.has_females() or not self.has_males()

    def has_females(self) -> bool:
        """Checks if the union has at least one female."""
        return any(m.is_female or m.params.is_hermaphrodite for m in self.members)

    def has_males(self) -> bool:
        """Checks if the union has at least one male."""
        return any(not m.is_female or m.params.is_hermaphrodite for m in self.members)

    def is_viable_for_breeding(self) -> bool:
        """Check if union can produce children"""
        if self.marriage_type == "asexual":
            return len(self.members) > 0           
        if len(self.members) < 2:
            return False            
        return self.has_females() and self.has_males()

    # In Union class
    def __repr__(self):
        member_descriptions = ", ".join(
        [f"{'♀️ ' if m.is_female else '♂️ '}{m.species_name}( Age: {m.age_years:.0f} Kids: {m.number_of_healthy_children})" for m in self.members]
        )
        return f"<Union ({self.marriage_type}{len(self.members)}/{self.max_size}) | Members: [{member_descriptions}]>\n"

def convert_years_to_string(years_float: float) -> str:
    """
    Converts a float number of years into a human-readable string (e.g., "5 years, 3 months, 2 days").
    """
    years = int(years_float)
    remaining_years = years_float - years
    months_float = remaining_years * 12
    months = int(months_float)
    remaining_months = months_float - months
    days = int(round(remaining_months * (earth_year / 12))) # Use average days in month

    parts = []
    if years > 0:
        parts.append(f"{years} year{'s' if years != 1 else ''}")
    if months > 0:
        parts.append(f"{months} month{'s' if months != 1 else ''}")
    if days > 0:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    
    return ", ".join(parts) if parts else "0 days"


def calculate_age_based_fertility(
    current_age: float, 
    max_fertility: float, 
    rising_steepness: float, 
    rising_midpoint_age: float, 
    falling_steepness: float, 
    falling_midpoint_age: float
) -> float:
    """
    Calculates the fertility rate for an individual based on their age using a double logistic function.
    This function models a fertility curve that rises, peaks, and then declines.
    
    :param current_age: The individual's current age in years.
    :param max_fertility: The species' peak fertility rate (A).
    :param rising_steepness: How quickly fertility rises after puberty (k1).
    :param rising_midpoint_age: The age (in years) at which fertility reaches 50% of its peak during the rise (t1).
    :param falling_steepness: How quickly fertility falls after its peak (k2).
    :param falling_midpoint_age: The age (in years) at which fertility falls to 50% of its peak during the decline (t2).
    :return: The calculated fertility rate for the current age.
    """
    # Logistic function for the rising part of the curve (puberty to peak)
    growth_logistic = 1.0 / (1.0 + np.exp(-rising_steepness * (current_age - rising_midpoint_age)))
    
    # Logistic function for the declining part of the curve (peak to menopause)
    # This is (1 - logistic) to create an inverse curve
    decline_logistic = 1.0 - (1.0 / (1.0 + np.exp(-falling_steepness * (current_age - falling_midpoint_age))))
    
    # The final fertility is the product of the peak rate and both logistic curves
    return max_fertility * growth_logistic * decline_logistic

def find_union_for_primate(primate: Primate, eligible_pool: Set[Primate], marriage_type, active_unions: List[Union]):
        """
        This is the new "Coupling" function.
        It finds a partner or existing union for the given primate.
        """
        if marriage_type == "asexual":
            if primate.is_hermaphrodite:
                new_union = Union(marriage_type="asexual", max_size=1)
                new_union.add_member(primate)
            return # Asexual non-hermaphrodites can't couple

        potential_partners = []
        for partner in eligible_pool:
            if partner is primate or partner.union is not None:
                continue          
            if (primate.params.is_hermaphrodite and partner.params.is_hermaphrodite) or \
               (primate.is_female != partner.is_female):
                potential_partners.append(partner) # Find opposite sex (or any other hermaphrodite)

        if not potential_partners:
            return # No partners available
        
        potential_partners.sort(key=lambda p: abs(p.age_days - primate.age_days))
        best_partner = potential_partners[0] # Sort partners by closest age

        if marriage_type == "monogamy":
            new_union = Union(marriage_type="monogamy", max_size=2)
            new_union.add_member(primate)
            new_union.add_member(best_partner)
            return

        if marriage_type == "polygyny":
            if not primate.is_female: # Male is seeking
                # Males form new unions
                new_union = Union(marriage_type="polygyny", max_size=5)
                new_union.add_member(primate)
                new_union.add_member(best_partner) # Add one female
                active_unions.append(new_union)
            else: # Female is seeking              
                for union in active_unions:
                    if union.marriage_type == "polygyny" and \
                       len(union.members) < union.max_size and \
                       union.has_males(): # Ensure union has a male
                        union.add_member(primate)
                        return # Try to join an existing union that has a male and space
               
                if not best_partner.is_female:
                    new_union = Union(marriage_type="polygyny", max_size=5)
                    new_union.add_member(best_partner) # Add the male first
                    new_union.add_member(primate)  # If no unions to join, form a new one with the best partner (who must be male)
            return

        if marriage_type == "polyandry":
            if primate.is_female:  # Female is seeking
                if not best_partner.is_female:
                    new_union = Union(marriage_type="polyandry", max_size=5)
                    new_union.add_member(primate)
                    new_union.add_member(best_partner)
            else:  # Male is seeking            
                for union in active_unions:
                    if union.marriage_type == "polyandry" and len(union.members) < union.max_size and union.has_females():
                        union.add_member(primate)
                        return    # Try to join an existing union with a female and space         
                if best_partner.is_female:
                    new_union = Union(marriage_type="polyandry", max_size=5)
                    new_union.add_member(best_partner) # Add the female first
                    new_union.add_member(primate) # If no union to join, form a new one with the best partner (who must be female)
            return

        if marriage_type == "polygamy":
            for union in active_unions:
                if union.marriage_type == "polygamy" and len(union.members) < union.max_size:
                    union.add_member(primate)
                    return
            
            new_union = Union(marriage_type="polygamy", max_size=9)
            new_union.add_member(primate)
            new_union.add_member(best_partner)  # If none, form a new one
            return

class Disaster:
    def __init__(self, name, is_possible, trigger_chance, duration_cycles):
        self.name = name
        self.is_possible = is_possible
        self.trigger_chance = trigger_chance
        self.duration_cycles = duration_cycles

        self.is_active = False
        self.end_day = None
        self.has_triggered = False   # Permanently disables retriggering after first use

    @classmethod
    def from_json(cls, json_path: str):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            disasters = []
            for name, d_data in data.items():
                disasters.append(cls(
                    name=name,
                    is_possible=d_data.get("is_possible", True),
                    trigger_chance=d_data.get("trigger_chance", 0.0),
                    duration_cycles=d_data.get("duration_cycles", 1),
                ))

            return disasters

        except FileNotFoundError:
            print("Warning: disasters.json not found.")
            return []
        except json.JSONDecodeError:
            print("Warning: Invalid JSON in disasters.json.")
            return []

    def try_trigger(self, current_day, days_per_cycle, pop_size):
        """Attempt to trigger once. If triggered, lasts duration_cycles and can't happen again."""
        if not self.is_possible:
            return False

        if self.is_active:
            return False  # already running

        if self.has_triggered:
            return False  # already used, cannot retrigger in this simulation

        if pop_size < 500:
            return False  # threshold check

        if random.random() < self.trigger_chance:
            self.is_active = True
            self.has_triggered = True
            self.end_day = current_day + (self.duration_cycles * days_per_cycle)
            return True

        return False

    def check_end(self, current_day):
        """End the disaster if its timer has expired."""
        if self.is_active and self.end_day is not None and current_day >= self.end_day:
            self.is_active = False
            return True
        return False
