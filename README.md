# Primate & Humanoid Population Simulator

This project is an agent-based model (ABM) designed to simulate and predict population dynamics for a wide variety of humanoid and animal species.

It operates from a "bottom-up" perspective by creating a population of individual agents, each with their own attributes like age, sex, and fertility. The simulation then runs over a specified number of years, allowing complex, large-scale behaviors like population growth, decline, or extinction to emerge from the simple rules governing each individual.

At the end of a run, the simulator provides a detailed statistical summary, a text-based population pyramid, and a `matplotlib` graph to visualize the population's history.

## Features

  * **Individual Agent-Based Modeling:** Tracks the discrete lifecycle of every single agent in the population.
  * **Deeply Configurable Demographics:** All species parameters are loaded from an external `demographics.json` file. This allows for easy modification and addition of new species profiles.
  * **Included Species Profiles:** Comes with pre-built profiles for real-world primates (modern human, chimpanzee, gorilla) and fantasy humanoids (orc, goblin, elf, neanderthal).
  * **Complex Fertility Modeling:** Uses a double logistic function to model age-based fertility, simulating a fertility peak after puberty and a decline toward menopause.
  * **Detailed Mortality:** Simulates distinct mortality rates for infants, mothers (maternal mortality), and general adults, including an age-accelerated risk of death past the average lifespan.
  * **Genetic Diversity Penalties:** Includes a system to model the negative effects of a small breeding population (Ne), which can increase mortality rates when the population lacks genetic diversity.
  * **Starting Scenarios:** Can be booted from a "random" population or from a specific, pre-defined scenario (e.g., an "aging\_village" or a "bounty\_mutiny" founder group).
  * **Visualization:** Automatically generates a text-based population pyramid and a graphical plot of population over time.

## How to Run the Simulation

### 1\. Prerequisites

You must have Python installed, along with the `numpy` and `matplotlib` libraries. You can install them using pip:

```bash
pip install numpy matplotlib
```

### 2\. Running the Simulator

The main file to run is `PrimatePopulationSimulator.py`.

```bash
python PrimatePopulationSimulator.py
```

## How to Configure the Simulation

All configuration is done by editing plain text files or the main script's `__main__` block.

### Files to Configure

  * `PrimatePopulationSimulator/demographics.json`
    This is the **primary configuration file** for all species data. You can edit any species profile or add your own. Key parameters include:

  * `PrimatePopulationSimulator/Scenarios.json`
    This file holds specific starting populations. You can add new scenarios here. Each agent is defined with a starting `age_days`, `is_female` status, and `is_initially_fertile` status.

  * `PrimatePopulationSimulator.py` (at the very bottom)
    This is where you **choose which simulation to run**. Go to the `if __name__ == "__main__":` block at the end of the file to edit the simulation parameters:

      * **To change the species:**
        Modify the second argument in this line to match a profile name from `demographics.json`:

        ```python
        sim_params = SimulationParameters.from_json("demographics.json", "modern_human")
        ```

      * **To change the run duration:**
        Modify the `num_years` value in this line:

        ```python
        simulation.run_simulation(num_years=10.0)
        ```

      * **To start with a random population:**
        Make sure this line is active:

        ```python
        simulation = PrimateSimulation(params=sim_params) # For a random start
        ```

      * **To start from a scenario:**
        Comment out the line above and uncomment this one, changing `"bounty_mutiny"` to the name of your desired scenario from `Scenarios.json`:

        ```python
        #simulation = PrimateSimulation(params=sim_params, scenario_name="bounty_mutiny")
        ```
