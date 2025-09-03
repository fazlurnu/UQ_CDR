import sys, os

import matplotlib.pyplot as plt
import numpy as np

from pprint import pprint

import pandas as pd

from itertools import product
from tqdm import tqdm

import json
import datetime
import uuid

import yaml

from uncertainty_quantification.monte_carlo_noise import ConflictResolutionSimulation
from uncertainty_quantification.plot_functions import plot_uncertainty

def save_sim_metadata(sim, ts_id, csv_paths, fig_paths, output_dir="results/UQ/metadata"):
    """
    Saves metadata about a simulation run including timestamp, simulation parameters,
    and output file paths (list of CSVs and figures).
    """
    def safe_serialize(obj):
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)

    sim_vars = {k: safe_serialize(v) for k, v in vars(sim).items()}

    metadata = {
        "date": datetime.datetime.now().isoformat(),
        "parameters": sim_vars,
        "output_files": {
            "csv": csv_paths,
            "figures": fig_paths
        }
    }

    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, f"metadata_CD_{ts_id}.json")

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"----> Metadata saved to {metadata_path}")

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

def main():
    all_csv_paths = []
    all_fig_paths = []

    with open("sim_config.yaml", "r") as f:
            config = yaml.safe_load(f)

    nav_uncertainty, vehicle_uncertainty = select_uncertainty()

    ## ----- Starting MC simulations ---- ##
    print("Running Monte Carlo")
    sim = ConflictResolutionSimulation(nav_uncertainty, vehicle_uncertainty)

    tlosh = config['cr_params']['tlookahead']
    rpz = config['cr_params']['rpz']

    pos_acc = sim.pos_acc if sim.pos_uncertainty_on else 0
    spd_sigma = sim.gs_sigma_ownship if sim.spd_uncertainty_on else 0
    hdg_sigma = sim.hdg_sigma_ownship if sim.hdg_uncertainty_on else 0
    vel_acc = sim.vel_acc if sim.vel_uncertainty_on else 0

    results = []

    # param_grid = list(product(range(2, 181, 10), range(0, 46, 5)))
    param_grid = list(product([2, 30], [0, 45]))

    for dpsi_val, dcpa_val in tqdm(param_grid, desc="Running Simulations", unit="sim"):
        timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        uid = str(uuid.uuid4())[:8]  # short UUID for readability
        ts_id = f"{timestamp}_{uid}"

        df = sim.run_simulation(dpsi_val, dcpa_val)

        df_output_dir = f"results/UQ/csv/CD/{nav_uncertainty}_{vehicle_uncertainty}"
        os.makedirs(df_output_dir, exist_ok=True)  # Create directory if it doesn't exist
        df_name = f"CD_{nav_uncertainty}_{vehicle_uncertainty}_{ts_id}.csv"
        df_path = os.path.join(df_output_dir, df_name)

        df.to_csv(df_path, index = False)

        df_tin_only = df[df['tin'] < 1e7]

        features = ['tcpa', 'dcpa', 'tin']
        feature_name = ["$t_{CPA}$ [s]", "$d_{CPA}$ [m]", "$t_{in}$ [s]"]

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        conflict_dcpa = len(df[df['dcpa'] < rpz]) / len(df) * 100
        conflict_tin = len(df[df['tin'] < tlosh]) / len(df) * 100
        conflict_tin_defined = len(df_tin_only[df_tin_only['tin'] < 15]) / len(df_tin_only['tin']) * 100
        conflict_tout = len(df[df['tout'] > 0]) / len(df) * 100
        is_conflict = len(df[(df['dcpa'] < rpz) & (df['tin'] < tlosh)]) / len(df) * 100

        # Save into list
        results.append({
            'ts_id': ts_id,
            'dpsi_val': dpsi_val,
            'dcpa_val': dcpa_val,
            'conflict_dcpa': conflict_dcpa,
            'conflict_tin': conflict_tin,
            'conflict_tin_defined': conflict_tin_defined,
            'conflict_tout': conflict_tout,
            'is_conflict': is_conflict
        })


        for i, feature in enumerate(features):
            if(feature == 'tin'):
                counts, bin_edges = np.histogram(df_tin_only[feature], bins=30)
            else:
                counts, bin_edges = np.histogram(df[feature], bins=30)

            percentages = counts / counts.sum() * 100  # Convert counts to percentage

            max_feature_y = int(percentages.max()) + 3

            axs[i].bar(bin_edges[:-1], percentages, width=np.diff(bin_edges), edgecolor='black', align='edge')

            if(feature == 'tin'):
                axs[i].axvspan(xmin=bin_edges.min(), xmax=tlosh, ymin=0, ymax=1, color='green', alpha=0.1, label='Conflict Detected', zorder=0)
                # median_value = np.median(df_tin_only[feature])  # <-- replace 'data' with your actual array
                # axs[i].axvline(x=median_value, color='red', linestyle='--', linewidth=2, label='Median')

            elif(feature == 'dcpa'):
                axs[i].axvspan(xmin=bin_edges.min(), xmax=rpz, ymin=0, ymax=1, color='green', alpha=0.1, label='Conflict Detected', zorder=0)

            axs[i].set_xlabel(feature_name[i])

            if(i == 0):
                axs[i].set_ylabel('Percentage [%]')
            else:
                # axs[i].set_yticks([])
                axs[i].set_yticklabels([])

            if i == 1:
                axs[i].legend()

        for i, feature in enumerate(features):
            axs[i].set_ylim([0, percentages.max()])

        plt.suptitle(
            f"$Accuracy_{{pos}} = {pos_acc}$, $\sigma^2_{{spd}} = {spd_sigma}$, $\sigma^2_{{hdg}} = {hdg_sigma}$, $Accuracy_{{vel}} = {vel_acc}$\n"
            f"$d_{{CPA}} = {dcpa_val}$, $d_{{\Psi}} = {dpsi_val}$"
        )
        # plt.tight_layout(rect=[0, 0, 1, 0.95])  # 0.95 leaves 5% at the top for title
        plt.tight_layout()
        
        fig_output_dir = f"results/UQ/figures/CD/{nav_uncertainty}_{vehicle_uncertainty}"
        os.makedirs(fig_output_dir, exist_ok=True)  # Create directory if it doesn't exist
        fig_name = f"CD_{nav_uncertainty}_{vehicle_uncertainty}_{ts_id}.png"
        fig_path = os.path.join(fig_output_dir, fig_name)

        plt.savefig(fig_path, dpi=300, bbox_inches='tight')

        plt.close()

        all_csv_paths.append(df_path)
        all_fig_paths.append(fig_path)

    print(f"----> csv saved to {df_output_dir}")    
    print(f"----> figures saved to {fig_output_dir}")

    save_sim_metadata(sim, ts_id, all_csv_paths, all_fig_paths)

    # After all simulations are done
    results_df = pd.DataFrame(results)

    # Optionally save to CSV
    results_output_dir = "results/UQ/data"
    os.makedirs(results_output_dir, exist_ok=True)
    results_path = os.path.join(results_output_dir, f"conflict_results_{nav_uncertainty}_{vehicle_uncertainty}_{timestamp}.csv")
    results_df.to_csv(results_path, index=False)

if __name__ == "__main__":
    main()