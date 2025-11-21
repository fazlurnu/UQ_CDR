import sys, os

import matplotlib.pyplot as plt

from pprint import pprint
import pandas as pd
import json
import datetime
import uuid

from uncertainty_quantification.monte_carlo_noise import ConflictResolutionSimulation
from uncertainty_quantification.plot_functions import plot_uncertainty

import bluesky as bs


def select_uncertainty():
    """
    Orchestrates the entire conflict-resolution demonstration by:
      1. Selecting which uncertainties and sources are active
      2. Running the simulation
      3. Running the main simulation loop (clustering + plotting + CSV logging).
    """
    # 1. Create a selector for user input

    import sys

    # Handle different numbers of arguments
    if len(sys.argv) == 1:
        print("Usage: python main.py <nav_uncertainty> <vehicle_uncertainty>")
        print("  - nav_uncertainty: combination of p (position), v (velocity)")
        print("  - vehicle_uncertainty: combination 'o' (ownship) and 'i' (intruder)")
        print("Settings is set to default: pv oi")

        nav_uncertainty = 'pv'
        vehicle_uncertainty = 'oi'
        
    elif len(sys.argv) == 2:
        nav_uncertainty = sys.argv[1].lower().strip()
        vehicle_uncertainty = 'oi'

        print(f"Settings is set to: {nav_uncertainty} {vehicle_uncertainty}")

    elif len(sys.argv) == 3:
        nav_uncertainty = sys.argv[1].lower().strip()
        vehicle_uncertainty = sys.argv[2].lower().strip()

        print(f"Settings is set to: {nav_uncertainty} {vehicle_uncertainty}")

    else:
        print("Too many arguments provided.")
        sys.exit(1)

    # Validate nav_uncertainty letters
    allowed_uncertainty_letters = {'p', 'v'}
    if any(char not in allowed_uncertainty_letters for char in nav_uncertainty):
        raise ValueError("nav_uncertainty can only contain 'p', 'v', or 'pv")

    # Validate vehicle_uncertainty letters
    allowed_vehicle_letters = {'o', 'i'}
    if any(char not in allowed_vehicle_letters for char in vehicle_uncertainty):
        raise ValueError("vehicle_uncertainty can only be 'o' or 'i'")

    return nav_uncertainty, vehicle_uncertainty

def save_sim_metadata(sim, ts_id, csv_path, fig_path, output_dir="results/UQ/metadata"):
    def safe_serialize(obj):
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)

    sim_vars = {k: safe_serialize(v) for k, v in vars(sim).items()}

    filename = f"CR_metadata_{ts_id}.json"
    metadata_path = os.path.join(output_dir, filename)

    metadata = {
        "date": datetime.datetime.now().isoformat(),
        "parameters": sim_vars,
        "output_files": {
            "csv": csv_path,
            "figure": fig_path
        }
    }

    os.makedirs(output_dir, exist_ok=True)  # Make sure the directory exists

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"----> Metadata saved to {metadata_path}")

def main():
    bs.init(mode='sim', detached=True)
    nav_uncertainty, vehicle_uncertainty = select_uncertainty()

    ## ----- Starting MC simulations ---- ##
    print("Running Monte Carlo")
    sim = ConflictResolutionSimulation(nav_uncertainty, vehicle_uncertainty)

    for init_dpsi, init_dcpa in [(30, 0), (30, 45)]:
        timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        uid = str(uuid.uuid4())[:8]  # short UUID for readability
        ts_id = f"{timestamp}_{uid}"
        
        df = sim.run_simulation(init_dpsi, init_dcpa)

        csv_output_dir = "results/UQ/csv/CR"
        csv_name = f"{nav_uncertainty}_{vehicle_uncertainty}_{ts_id}.csv"
        csv_path = os.path.join(csv_output_dir, csv_name)
        os.makedirs(csv_output_dir, exist_ok=True)

        df.to_csv(csv_path, index = False)
        print(f"----> csv saved to {csv_path}")

        print("Running Plotting")
        ## ---- Plotting ---- ##
        # df = df[df['is_conflict'] == True]

        f = plot_uncertainty(df, sim)

        fig_output_dir = "results/UQ/figures/CR"
        fig_name = f"{nav_uncertainty}_{vehicle_uncertainty}_{ts_id}.png"
        os.makedirs(fig_output_dir, exist_ok=True)
        fig_path = os.path.join(fig_output_dir, fig_name)

        f.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"----> fig saved to {fig_path}")

        save_sim_metadata(sim, ts_id, csv_path, fig_path)

if __name__ == "__main__":
    main()