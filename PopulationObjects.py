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

        self.total_calories_available = params["total_calories_available"]
        self.calories_needed_per_primate = params["calories_needed_per_primate"]
        self.genetic_diversity = params["initial_genetic_diversity"]
        self.fertile_days = self.menopause_age_days - self.puberty_age_days #A female primate's reproductive lifespan.

        self.effective_gestation_days = self.gestation_days + self.interbirth_interval_days
        self.carrying_capacity = (
            math.floor(self.total_calories_available / self.calories_needed_per_primate)
            if self.calories_needed_per_primate > 0 else 0
        )
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

def double_logistic(t, A, k1, t1, k2, t2):
    growth_logistic = 1 / (1 + np.exp(-k1 * (t - t1)))
    decline_logistic = 1 - (1 / (1 + np.exp(-k2 * (t - t2))))
    return A * growth_logistic * decline_logistic