import sys, os

import matplotlib.pyplot as plt

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

def save_sim_metadata(sim, ts_id, csv_path, fig_path, output_dir="results/UQ/metadata"):
    def safe_serialize(obj):
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)

    sim_vars = {k: safe_serialize(v) for k, v in vars(sim).items()}

    filename = f"metadata_{ts_id}.json"
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
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

    nav_uncertainty, vehicle_uncertainty = select_uncertainty()

    ## ----- Starting MC simulations ---- ##
    sim = ConflictResolutionSimulation(nav_uncertainty, vehicle_uncertainty)

    for dpsi_val in range(0, 181, 3):
        for dcpa_val in range(0, sim.rpz - 1, 5): ## ends at rpz - 1
            print(f"#### Running dpsi: {dpsi_val}, dcpa: {dcpa_val}")
            uid = str(uuid.uuid4())[:8]  # short UUID for readability
            ts_id = f"{timestamp}_{uid}"
            
            print("Running Monte Carlo")
            df = sim.run_simulation(dpsi_val, dcpa_val)

            csv_output_dir = "results/UQ/csv"
            csv_name = f"{nav_uncertainty}_{vehicle_uncertainty}_{ts_id}.csv"
            csv_path = os.path.join(csv_output_dir, csv_name)
            df.to_csv(csv_path, index = False)
            print(f"----> csv saved to {csv_path}")

            print("Running Plotting")
            ## ---- Plotting ---- ##
            f = plot_uncertainty(df, sim)

            fig_output_dir = "results/UQ/figures"
            fig_name = f"{nav_uncertainty}_{vehicle_uncertainty}_{ts_id}.png"
            fig_path = os.path.join(fig_output_dir, fig_name)

            f.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"----> fig saved to {fig_path}")

            save_sim_metadata(sim, ts_id, csv_path, fig_path)

if __name__ == "__main__":
    main()