import matplotlib.pyplot as plt
import numpy as np
import math

earth_year = 365.2422

def log_population_stats(current_day, population, history, cycle, births, deaths, eligible):
        total_pop = len(population)
        
        median_age_years = 0.0
        if total_pop > 0:
                ages_in_years = np.array([p.age_years for p in population])
                median_age_years = np.median(ages_in_years)

        print(f"\n--- Cycle: {cycle} (Day: {current_day}) Year: {current_day / earth_year:.1f} ---")
        print(f"Total Population: {total_pop:,d}")
        print(f"  - Median Age: {median_age_years:.1f} years")

        females = sum(1 for p in population if p.is_female)
        males = total_pop - females
        sex_ratio = males / females if females > 0 else float('inf')
        print(f"  - Females: {females:,d}")
        print(f"  - Males: {males:,d}")
        print(f"  - Sex Ratio (M/F): {sex_ratio:.2f}")
        
        if cycle != 0 and cycle != "Final":
            print(f"Births This Cycle: {births:,d}")
            print(f"Deaths This Cycle: {deaths:,d}")
            print(f"Potential Mothers: {eligible:,d}")

        species_counts = {}
        for p in population:
            name = p.params.species_name
            species_counts[name] = species_counts.get(name, 0) + 1     
            species_str = ", ".join([f"{name}: {count:,d}" for name, count in species_counts.items()])   
        if species_str:
            print(f"  - Species: {species_str}")

        history.append({'cycle': cycle, 'population': total_pop, 'females': females, 'males': males, 'current_day': current_day})

def display_population_pyramid(population, earth_year):
        if not population:
            print("\n--- Population Pyramid ---")
            print("Population is extinct.")
            return

        print("\n--- Population Pyramid ---")
                
        max_age_obj = max(population, key=lambda p: p.age_years, default=None)
        if not max_age_obj: 
            print("Population is extinct.")
            return
            
        max_age = round(max_age_obj.age_years)
        
        if earth_year <= 0:
            print("Error: earth_year is zero or negative.")
            return

        bracket_size = max(1, max_age // 15)
             
        brackets = range(0, (max_age // bracket_size) * bracket_size + bracket_size, bracket_size)
        
        age_distribution = {f"{i}-{i+bracket_size-1}": {"male": 0, "female": 0} for i in brackets}
        
        for p in population:
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
        
        print(f"{'Males'.rjust(pyramid_width)} | Age | {'Females'.ljust(pyramid_width)}")
        print(f'{"-"*pyramid_width}-+-----+--{"-"*pyramid_width}')
        for bracket_label in sorted(age_distribution.keys(), key=lambda x: int(x.split('-')[0])):
            data = age_distribution[bracket_label]
            male_bar = '█' * int(data['male'] * scale)
            female_bar = '█' * int(data['female'] * scale)
            print(f"{male_bar.rjust(pyramid_width)} | {bracket_label.center(5)} | {female_bar.ljust(pyramid_width)}")

def plot_population_history(history, species_names, current_day):
        if not history:
            print("No history recorded, cannot plot graph.")
            return

        years = [r['current_day'] / earth_year for r in history]
        populations = [r['population'] for r in history]

        if not years:
            print("No data points to plot.")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(years, populations, marker='o', linestyle='-', color='b', markersize=4)
        
        plt.title(f"This Population Over Time")
        plt.xlabel("Years")
        plt.ylabel("Total Population")
        
        total_duration_years = current_day / earth_year
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
