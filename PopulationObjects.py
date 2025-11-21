import math
from tkinter import SE
import numpy as np
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
            
        # We use age_years * earth_year to get biological age in days
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
        # Core Lifecycle
        self.species_name = params["Species_Name"]
        self.puberty_age_days = params["puberty_age_days"]
        self.menopause_age_days = params["menopause_age_days"]
        self.lifespan_days = params["lifespan_days"] 
        
        # Reproduction
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
        
        # Gender Types
        self.is_hermaphrodite = params.get("is_hermaphrodite", False) # Use .get() for optional params
        self.is_sequential_species = params.get("is_sequential_species", False)
        self.ages_backward = params.get("ages_backward", False) # --- ADDED ---
        
        # Mortality
        self.infant_mortality_rate = params["infant_mortality_rate"]
        self.maternal_mortality_rate = params["maternal_mortality_rate"]
        self.adult_mortality_rate = params["adult_mortality_rate"]

        # Diet & Environment
        self.calories_needed_per_primate = params["calories_needed_per_primate"] # Calories needed *per day*
        self.diet_type = params.get("diet_type", "omnivore") # Get diet type, default to omnivore
        
        # Genetics
        self.genetic_diversity = params.get("initial_genetic_diversity", 1.0)
        
        # Fertility Curve (Dynamic)
        self.fertility_rising_steepness = params["fertility_rising_steepness"]
        self.fertility_falling_steepness = params["fertility_falling_steepness"]

        # --- Calculated Parameters ---
        self.fertile_days = self.menopause_age_days - self.puberty_age_days #A female primate's reproductive lifespan.
        self.effective_gestation_days = self.gestation_days + self.interbirth_interval_days
        
        # Calculate cycles per life and per-cycle fertility rate
        if self.effective_gestation_days > 0:
            self.cycles_per_reproductive_life = self.fertile_days / self.effective_gestation_days #How many birthing cycles a primate potentially has.
            cycle_length_in_years = self.effective_gestation_days / earth_year
            self.per_cycle_fertility_rate = self.base_fertility_rate * cycle_length_in_years
        else:
            self.cycles_per_reproductive_life = 0
            self.per_cycle_fertility_rate = 0

        # Calculate final effective fertility rate per cycle, capping at 99.999%
        self.effective_per_cycle_fertility_rate = min(
            self.per_cycle_fertility_rate * (1 - self.miscarriage_stillborn_rate), 0.99999
        )
        
        # Calculate per-cycle mortality from annual mortality rate
        if self.effective_gestation_days > 0:
            cycle_length_in_years = self.effective_gestation_days / earth_year
            self.per_cycle_adult_mortality_rate = (
                1 - (1 - self.adult_mortality_rate) ** cycle_length_in_years
            )
        else:
            self.per_cycle_adult_mortality_rate = 0

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

    # In Union class
    def remove_member(self, primate):
        if primate in self.members:
            self.members.remove(primate)
            primate.union = None
        if not self.members:  # If empty, mark as dissolved
            self.dissolved = True

    def is_dissolved(self, params) -> bool:
        """Hard correctness rules."""
        if self.dissolved:
            return True

        if len(self.members) == 0:
            return True

        if self.marriage_type == "asexual":
            return len(self.members) != 1

        return not self.has_females(params) or not self.has_males(params)

    def has_females(self, params) -> bool:
        """Checks if the union has at least one female."""
        if params.is_hermaphrodite:
            return len(self.members) > 0
        return any(m.is_female for m in self.members)

    def has_males(self, params) -> bool:
        """Checks if the union has at least one male."""
        if params.is_hermaphrodite:
            return len(self.members) > 0
        return any(not m.is_female for m in self.members)

    def is_viable_for_breeding(self, params) -> bool:
        """Check if union can produce children"""
        if self.marriage_type == "asexual":
            return len(self.members) > 0
            
        if len(self.members) < 2:
            return False
            
        if params.is_hermaphrodite:
            return len(self.members) >= 2 # Need at least two hermaphrodites
        
        # --- BUG FIX ---
        # Correctly call the methods with (params)
        if not (self.has_females(params) and self.has_males(params)):
            return False
        # --- END FIX ---
            
        return True

    # In Union class
    def __repr__(self):
        member_descriptions = ", ".join(
        [f"{'F' if m.is_female else 'M'}{m.species_name}({m.age_years:.0f} {m.number_of_healthy_children})" for m in self.members]
        )
        return f"<Union ({self.marriage_type}{len(self.members)}/{self.max_size}) | Members: [{member_descriptions}]>"


def calculate_total_available_resources(params: SimulationParameters, locale: Locale) -> int:
    """
    Calculates the carrying capacity based on the species' diet and the locale's calorie availability.
    """
    total_available_calories = 0
    diet = params.diet_type.lower()
    
    if diet == "omnivore":
        total_available_calories = locale.carnivore_calories + locale.herbivore_calories
    elif diet == "carnivore":
        total_available_calories = locale.carnivore_calories
    elif diet == "herbivore":
        total_available_calories = locale.herbivore_calories
    elif diet == "ruminant":
        total_available_calories = locale.ruminant_calories + locale.herbivore_calories
    elif diet == "autotroph":
        total_available_calories = locale.water_availability_m3
    else:
        print(f"Warning: Unknown diet_type '{params.diet_type}'. They can eat everything!.")
        total_available_calories = locale.carnivore_calories + locale.herbivore_calories + locale.ruminant_calories

    return total_available_calories


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

