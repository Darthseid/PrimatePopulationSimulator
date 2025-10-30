import math
import numpy as np
import json

earth_year = 365.2422  # Constants

class Primate:
    def __init__(self, is_female: bool, age_days: int, is_initially_fertile: bool):
        self.is_female: bool = is_female
        self.age_days: int = age_days
        self.is_fertile: bool = is_initially_fertile
        self.is_coupled: bool = False
        self.number_of_healthy_children: int = 0

    @property
    def age_years(self) -> float:
        return self.age_days / earth_year

    def __repr__(self) -> str:
        gender = "Female" if self.is_female else "Male"
        fertility = "Fertile" if self.is_fertile else "Sterile"
        coupled_status = "Coupled" if self.is_coupled else "Uncoupled" # --- ADDED for better representation ---
        return (f"<Primate | Gender: {gender}, Age: {self.age_years:.1f} yrs, "
                f"Status: {fertility}, {coupled_status}, Children: {self.number_of_healthy_children}>") # --- UPDATED ---

class Locale:
    """
    Represents the environment (location) where the simulation takes place.
    It defines the available resources and area for the population.
    """
    
    @classmethod
    def from_json(cls, json_path: str, locale_name: str):
        """
        Loads locale parameters from a JSON file.
        """
        try:
            with open(json_path, 'r') as f:
                all_locales = json.load(f)
            if locale_name not in all_locales:
                raise ValueError(f"Locale '{locale_name}' not found in {json_path}")
            params = all_locales[locale_name]
            return cls(**params)
        except FileNotFoundError:
            raise FileNotFoundError(f"Locales file not found: {json_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {json_path}")

    def __init__(self, **params):
        """
        Initializes the Locale with parameters loaded from the JSON.
        """
        self.name: str = params.get("name", "Unnamed Locale")
        self.biome_type: str = params.get("biome_type", "Temperate")
        self.area_km2: float = params.get("area_km2", 0.0)
        self.water_availability_m3: float = params.get("water_availability_m3", 0.0)
        
        # Available calories per year
        self.carnivore_calories: int = params.get("carnivore_calories", 0)
        self.herbivore_calories: int = params.get("herbivore_calories", 0)
        self.ruminant_calories: int = params.get("ruminant_calories", 0)

    def __repr__(self) -> str:
        return (f"<Locale | Name: {self.name}, Biome: {self.biome_type}, "
                f"Area: {self.area_km2:,.2f} square kilometers")


class SimulationParameters:
    @classmethod
    def from_json(cls, json_path: str, profile_name: str):
        try:
            with open(json_path, 'r') as f:
                all_params = json.load(f)
            if profile_name not in all_params:
                raise ValueError(f"Profile '{profile_name}' not found in {json_path}")
            params = all_params[profile_name]
            return cls(**params)
        except FileNotFoundError:
            raise FileNotFoundError(f"Demographics file not found: {json_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {json_path}")

    def __init__(self, **params):
        self.puberty_age_days = params["puberty_age_days"]
        self.species_name = params["Species_Name"]
        self.menopause_age_days = params["menopause_age_days"]
        self.lifespan_days = params["lifespan_days"] 
        self.coupling_rate = params["coupling_rate"] #This represents the chance of a primate being coupled with a mate per cycle.
        self.gestation_days = params["gestation_days"]
        self.interbirth_interval_days = params["interbirth_interval_days"]
        self.max_kids_per_primate = params["max_kids_per_primate"]

        self.chance_of_multiple_birth = params["chance_of_multiple_birth"]

        self.base_fertility_rate = params["base_fertility_rate"]
        self.miscarriage_stillborn_rate = params["miscarriage_stillborn_rate"]
        self.fertility_rising_steepness = params["fertility_rising_steepness"]
        self.fertility_falling_steepness = params["fertility_falling_steepness"]
        self.sterile_chance = params["sterile_chance"]
        self.sex_ratio_at_birth = params["sex_ratio_at_birth"] #Set to 0.3412 to maximize males. Set to 0.658 to maximize males without penalty.

        self.infant_mortality_rate = params["infant_mortality_rate"]
        self.maternal_mortality_rate = params["maternal_mortality_rate"]
        self.adult_mortality_rate = params["adult_mortality_rate"]

        self.diet_type: str = params["diet_type"] 
        self.calories_needed_per_primate = params["calories_needed_per_primate"]

        self.genetic_diversity = params["initial_genetic_diversity"]
        self.fertile_days = self.menopause_age_days - self.puberty_age_days #A female primate's reproductive lifespan.

        self.effective_gestation_days = self.gestation_days + self.interbirth_interval_days
        
        self.cycles_per_reproductive_life = self.fertile_days / self.effective_gestation_days #How many birthing cycles a primate potentially has.
        cycle_length_in_years = self.effective_gestation_days / earth_year
        self.per_cycle_fertility_rate = self.base_fertility_rate * cycle_length_in_years
        self.effective_per_cycle_fertility_rate = min(
            self.per_cycle_fertility_rate * (1 - self.miscarriage_stillborn_rate), 0.99999
        )
        self.per_cycle_adult_mortality_rate = (
            1 - (1 - self.adult_mortality_rate) ** cycle_length_in_years
        )

def convert_years_to_string(years_float):
    years = int(years_float)
    remaining_years = years_float - years
    months_float = remaining_years * 12
    months = int(months_float)
    remaining_months = months_float - months
    days = int(round(remaining_months * 30.43685))

    parts = []
    if years > 0:
        parts.append(f"{years} year{'s' if years != 1 else ''}")
    if months > 0:
        parts.append(f"{months} month{'s' if months != 1 else ''}")
    if days > 0:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    return ", ".join(parts) if parts else "0 days"

# --- FUNCTION RENAMED AND REFACTORED ---
def calculate_age_based_fertility(current_age: float, 
                                  max_fertility: float, 
                                  rising_steepness: float, 
                                  rising_midpoint_age: float, 
                                  falling_steepness: float, 
                                  falling_midpoint_age: float) -> float:
    """
    Calculates the fertility rate based on age using a double logistic function.
    This creates a fertility curve that rises, peaks, and then falls.
    
    :param current_age: The agent's current age in years.
    :param max_fertility: The species' theoretical maximum fertility rate (A).
    :param rising_steepness: The steepness of the fertility growth curve (k1).
    :param rising_midpoint_age: The age (in years) at which fertility reaches 50% of its peak (t1).
    :param falling_steepness: The steepness of the fertility decline curve (k2).
    :param falling_midpoint_age: The age (in years) at which fertility begins to decline from its peak (t2).
    :return: The calculated fertility rate as a float.
    """
    fertility_growth = 1 / (1 + np.exp(-rising_steepness * (current_age - rising_midpoint_age)))
    declining_exp_term = np.exp(-falling_steepness * (current_age - falling_midpoint_age))
    fertility_decline = declining_exp_term / (1 + declining_exp_term)
    return max_fertility * fertility_growth * fertility_decline

def calculate_carrying_capacity(params: SimulationParameters, locale: Locale) -> int:
    """
    Calculates the carrying capacity based on the species' diet and the
    locale's available calories.
    """
    
    total_calories = 0
    diet = params.diet_type.lower() # Make it case-insensitive
     
    if diet == "carnivore":
        total_calories = locale.carnivore_calories
    elif diet == "herbivore":
        total_calories = locale.herbivore_calories
    elif diet == "ruminant":
        total_calories = locale.ruminant_calories
    elif diet == "omnivore":
        total_calories = int(
            0.75 * locale.carnivore_calories + 
            0.95 * locale.herbivore_calories 
        )
    else:
        print(f"Warning: Unknown diet_type '{params.diet_type}'. Carrying capacity may be 0.")
        
    if params.calories_needed_per_primate <= 0:
        return 0 # Avoid division by zero
        
    return math.floor(total_calories / params.calories_needed_per_primate)

