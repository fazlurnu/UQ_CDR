import sys, os

import matplotlib.pyplot as plt
import numpy as np

from pprint import pprint

import json
import datetime
import uuid

from uncertainty_quantification.monte_carlo_noise import ConflictResolutionSimulation
from uncertainty_quantification.plot_functions import plot_uncertainty

def select_uncertainty():
    """
    Orchestrates the entire conflict-resolution demonstration by:
      1. Selecting which uncertainties and sources are active
      2. Running the simulation
      3. Running the main simulation loop (clustering + plotting + CSV logging).
    """
    # 1. Create a selector for user input

    if len(sys.argv) != 3:
        print("Usage: python main.py <nav_uncertainty> <vehicle_uncertainty>")
        print("  - nav_uncertainty: combination of s (speed), h (heading), p (position)")
        print("  - vehicle_uncertainty: combination 'o' (ownship) and 'i' (intruder)")
        
        print("Settings is set to default: shp oi")

        nav_uncertainty = 'shp'
        vehicle_uncertainty = 'oi'
    else:
        nav_uncertainty = sys.argv[1].lower().strip()  # e.g. "sh", "p", "s", "shp"
        vehicle_uncertainty = sys.argv[2].lower().strip()    # e.g. "o" or "i"

        print(f"Settings is set to: {nav_uncertainty} {vehicle_uncertainty}")

    # Validate nav_uncertainty letters
    allowed_uncertainty_letters = {'s', 'h', 'p'}
    if any(char not in allowed_uncertainty_letters for char in nav_uncertainty):
        raise ValueError("nav_uncertainty can only contain 's', 'h', 'p'")

    # Validate vehicle_uncertainty letters
    allowed_vehicle_letters = {'o', 'i'}
    if any(char not in allowed_vehicle_letters for char in vehicle_uncertainty):
        raise ValueError("vehicle_uncertainty can only be 'o' or 'i'")

    return nav_uncertainty, vehicle_uncertainty

def main():
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    uid = str(uuid.uuid4())[:8]  # short UUID for readability
    ts_id = f"{timestamp}_{uid}"

    nav_uncertainty, vehicle_uncertainty = select_uncertainty()

    ## ----- Starting MC simulations ---- ##
    print("Running Monte Carlo")
    sim = ConflictResolutionSimulation(nav_uncertainty, vehicle_uncertainty)
    dpsi_val = 20
    
    for dcpa_val in range(0, 41, 10):
        df = sim.run_simulation(dpsi_val, dcpa_val)
        df_no_noise = sim.run_simulation_no_noise(dpsi_val, dcpa_val)

        df = df.dropna()

        features = ['tcpa', 'dcpa', 'tin']

        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

        for i, feature in enumerate(features):
            counts, bin_edges = np.histogram(df[feature], bins=30)
            percentages = counts / counts.sum() * 100  # Convert counts to percentage

            axs[i].bar(bin_edges[:-1], percentages, width=np.diff(bin_edges), edgecolor='black', align='edge')

            val = df_no_noise[feature].values[0]
            axs[i].vlines(x=val, ymin=0, ymax=percentages.max(), color='red', linestyle='--')

            axs[i].set_xlabel(feature.upper())
            axs[i].set_ylabel('Percentage')
            axs[i].set_title(f'Histogram of {feature.upper()}')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()