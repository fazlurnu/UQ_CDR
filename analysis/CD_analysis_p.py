import pandas as pd
import json
import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm  # <-- Use parametric Gaussian

import numpy as np

def compute_tcpa_stats(metadata, df):
    params = metadata['parameters']
    sigma = params['pos_acc']  # 95% bound in 2D Gaussian
    std_dev = sigma / 2.448
    var_pos = std_dev ** 2

    # Mean positions
    x_o = df['x_own_noise'].mean()
    y_o = df['y_own_noise'].mean()
    x_i = df['x_int_noise'].mean()
    y_i = df['y_int_noise'].mean()

    # Relative velocity (deterministic)
    theta_o = np.deg2rad(df['hdg_own_true'].mean())
    theta_i = np.deg2rad(df['hdg_int_true'].mean())
    s_o = df['gs_own_true'].mean()
    s_i = df['gs_int_true'].mean()

    v_o = s_o * np.array([np.cos(theta_o), np.sin(theta_o)])
    v_i = s_i * np.array([np.cos(theta_i), np.sin(theta_i)])
    v_rel = v_o - v_i
    Vx, Vy = v_rel[0], v_rel[1]

    # Mean t_CPA (Equation 39)
    delta_x = x_i - x_o
    delta_y = y_i - y_o
    mu_tcpa = (Vx * delta_x + Vy * delta_y) / (Vx**2 + Vy**2)

    # Variance t_CPA (Equation 40)
    numerator = (Vx**2 + Vy**2) * (2 * var_pos)  # same var for xi, xo, yi, yo
    denominator = (Vx**2 + Vy**2)**2
    sigma2_tcpa = numerator / denominator

    return mu_tcpa, sigma2_tcpa

def compute_dcpa_stats(metadata, df):
    params = metadata['parameters']
    
    # Position uncertainty (isotropic)
    sigma = params['pos_acc']
    std_dev = sigma / 2.448
    var_pos = std_dev ** 2
    Sigma_rel = 2 * np.array([[var_pos, 0], [0, var_pos]])  # ownship + intruder
    
    # Mean positions
    x_own, y_own = df['x_own_true'].mean(), df['y_own_true'].mean()
    x_int, y_int = df['x_int_true'].mean(), df['y_int_true'].mean()
    
    # Mean headings and speeds
    theta_own = np.deg2rad(df['hdg_own_true'].mean())
    theta_int = np.deg2rad(df['hdg_int_true'].mean())
    s_own = df['gs_own_noise'].mean()
    s_int = df['gs_int_noise'].mean()
    
    # Unit direction vectors
    u_own = np.array([np.cos(theta_own), np.sin(theta_own)])
    u_int = np.array([np.cos(theta_int), np.sin(theta_int)])
    
    # Relative motion
    mu_rel = np.array([x_int - x_own, y_int - y_own])
    v_rel = s_own * u_own - s_int * u_int
    v_rel_norm_sq = np.dot(v_rel, v_rel)
    
    # Perpendicular unit vector
    v_perp = np.array([-v_rel[1], v_rel[0]]) / np.sqrt(v_rel_norm_sq)
    
    # Projection matrix P
    P = np.eye(2) - np.outer(v_rel, v_rel) / v_rel_norm_sq
    
    # Mean of DCPA
    mu_zp = v_perp.T @ P @ mu_rel
    
    # Variance of DCPA
    sigma2_zp = v_perp.T @ P @ Sigma_rel @ P.T @ v_perp

    return np.abs(mu_zp), sigma2_zp

def compute_tin_stats(metadata, df):
    params = metadata['parameters']
    
    # Position uncertainty (isotropic)
    sigma = params['pos_acc']
    std_dev = sigma / 2.448
    var_pos = std_dev ** 2
    Sigma_rel = 2 * np.array([[var_pos, 0], [0, var_pos]])  # ownship + intruder

    # Mean positions
    x_own, y_own = df['x_own_true'].mean(), df['y_own_true'].mean()
    x_int, y_int = df['x_int_true'].mean(), df['y_int_true'].mean()
    mu_rel = np.array([x_int - x_own, y_int - y_own])

    # Relative velocity (deterministic)
    theta_own = np.deg2rad(df['hdg_own_true'].mean())
    theta_int = np.deg2rad(df['hdg_int_true'].mean())
    s_own = df['gs_own_true'].mean()
    s_int = df['gs_int_true'].mean()

    v_own = s_own * np.array([np.cos(theta_own), np.sin(theta_own)])
    v_int = s_int * np.array([np.cos(theta_int), np.sin(theta_int)])
    v_rel = v_own - v_int
    v_rel_norm = np.linalg.norm(v_rel)
    v_rel_norm_sq = v_rel_norm**2

    # t_CPA and d_CPA from mu_rel
    mu_tcpa = (v_rel @ mu_rel) / v_rel_norm_sq
    d_cpa = mu_rel - mu_tcpa * v_rel
    d_cpa_norm_sq = np.dot(d_cpa, d_cpa)

    # Radius R from metadata
    R = params['rpz']

    if d_cpa_norm_sq >= R**2:
        return np.nan, np.nan  # no conflict — undefined t_in

    # Mean of t_in
    sqrt_term = np.sqrt(R**2 - d_cpa_norm_sq)
    mu_tin = mu_tcpa - sqrt_term / v_rel_norm

    # Projection matrix P
    P = np.eye(2) - np.outer(v_rel, v_rel) / v_rel_norm_sq

    # Gradient of t_in wrt position
    term1 = v_rel / v_rel_norm_sq
    term2 = (1 / (v_rel_norm * sqrt_term)) * (P @ mu_rel)
    grad_tin = term1 + term2  # shape: (2,)

    # Delta method variance
    sigma2_tin = grad_tin.T @ Sigma_rel @ grad_tin

    print(mu_tin)
    return mu_tin, sigma2_tin


# --- Configuration ---
# csv_path = '../results/UQ/csv/CD/s_oi/CD_s_oi_250521_123537_b1592690.csv'
import glob

# csv_path_list = glob.glob('../results/UQ/csv/CD/p_oi/*.csv')
csv_path_list = glob.glob('../results/UQ/csv/CD/p_oi/*.csv')
csv_path_list = csv_path_list[:]
# csv_path_list = [csv_path_list[i] for i in [0]]

for csv_path in csv_path_list:
    metadata_dir = '../results/UQ/metadata'

    # --- Load CSV ---
    df = pd.read_csv(csv_path)
    csv_filename = os.path.basename(csv_path)

    # --- Extract Date Code from Filename ---
    match = re.search(r'CD_p_oi_(\d{6})_', csv_filename)
    if not match:
        raise ValueError("Date code not found in CSV filename")
    date_code = match.group(1)

    # --- Find All Metadata Files for This Date ---
    metadata_files = glob.glob(os.path.join(metadata_dir, f'metadata_CD_{date_code}_*.json'))
    if not metadata_files:
        raise FileNotFoundError(f"No metadata files found for date code {date_code}")

    # --- Try All Metadata Files Until Match is Found ---
    metadata = None


    for meta_file in metadata_files:
        with open(meta_file, 'r') as f:
            meta_candidate = json.load(f)
            csv_list = meta_candidate.get("output_files", {}).get("csv", [])
            if any(os.path.basename(path) == csv_filename for path in csv_list):
                metadata = meta_candidate
                metadata_file = meta_file
                break

    if metadata is None:
        raise FileNotFoundError(f"No matching metadata file lists '{csv_filename}' under date code {date_code}")
    else:
        print(f"✅ Matched metadata: {os.path.basename(metadata_file)}")


    params = metadata['parameters']

    pos_acc = params.get('pos_acc', 'N/A')
    vel_acc = params.get('vel_acc', 'N/A')

    # print(metadata['parameters'])
    # --- Check if CSV File is Listed in Metadata ---
    csv_files_list = metadata.get("output_files", {}).get("csv", [])

    print(df.columns)

    if not any(os.path.basename(path) == csv_filename for path in csv_files_list):
        print(f"❌ The file '{csv_filename}' is NOT listed in the metadata.")
    else:
        print(f"✅ The file '{csv_filename}' is listed in the metadata.")

        # --- Plot Histograms + Single Gaussian PDF ---
        variables = ['tcpa', 'dcpa', 'tin']
        var_name = {'tcpa': "$t_{CPA}$ [s]", 'dcpa': "$d_{CPA}$ [m]", 'tin': "$t_{in}$ [s]"}
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 9), constrained_layout=True)

        for ax, var in zip(axes, variables):
            if var in df.columns:
                data = df[var].dropna()

                # Apply cutoff for 'tin'
                if var == 'tin':
                    data = data[data <= 1e7]
                    ax.axvspan(0, params['tlosh'], color='green', alpha=0.2, zorder = 0, label = 'Conflict Detected')  # alpha controls transparency

                    mu_tin, var_tin = compute_tin_stats(metadata, df)
                    if not np.isnan(mu_tin):
                        sigma_tin = np.sqrt(var_tin)
                        x_vals = np.linspace(data.min() * 0.75, data.max() * 1.25, 500)
                        y_vals = norm.pdf(x_vals, mu_tin, sigma_tin)
                        ax.plot(x_vals, y_vals, color='red', label='Analytical Approximation', zorder=30)

                # Histogram as density (area = 1)
                counts, bin_edges = np.histogram(data, bins=30, density=True)
                ax.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), color = 'lightgray', edgecolor='black', align='edge', alpha=1.0, label='Samples', zorder = 20)

                if var == 'tcpa':
                    mu_tcpa, var_tcpa = compute_tcpa_stats(metadata, df)
                    sigma_tcpa = np.sqrt(var_tcpa)

                    x_vals = np.linspace(data.min(), data.max(), 500)
                    y_vals = norm.pdf(x_vals, mu_tcpa, sigma_tcpa)

                    ax.plot(x_vals, y_vals, color='red', label=f'Analytical Approximation', zorder = 30)

                elif var == 'dcpa':
                    mu_dcpa, var_dcpa = compute_dcpa_stats(metadata, df)
                    sigma_dcpa = np.sqrt(var_dcpa)

                    x_vals = np.linspace(0, data.max(), 500)  # Only non-negative domain
                    y_vals = (
                        norm.pdf(x_vals, mu_dcpa, sigma_dcpa)
                        + norm.pdf(x_vals, -mu_dcpa, sigma_dcpa)
                    )

                    # Add transparent green background between x = 0 and x = 50
                    ax.axvspan(0, params['rpz'], color='green', alpha=0.2, zorder = 0, label = 'Conflict Detected')  # alpha controls transparency

                    ax.plot(x_vals, y_vals, color='red', label=f'Analytical Approximation', zorder = 30)

                # ax.set_title(f'{var.upper()}: Histogram & Gaussian Fit')
                ax.set_xlabel(var_name[var])
                ax.set_ylabel("Probability Density")
                if var == 'dcpa':
                    ax.legend()
            else:
                ax.set_visible(False)
                print(f"⚠️ Column '{var}' not found in the dataframe.")

        fig.align_ylabels()
        dcpa_val = np.abs(np.round(mu_dcpa))
        dpsi_val = (df['hdg_int_true'].mean())

        plt.suptitle(
            rf"$Pos_{{acc bound}} = {pos_acc * 2}$, "
            rf"$Vel_{{acc bound}} = {vel_acc * 2}$, "
            + "\n" + 
            rf"$d_{{CPA}} = {dcpa_val}$, "
            rf"$\Delta\psi = {dpsi_val}$"
        )


        plt.savefig(f'/Users/mfrahman/Python/0_UQ_CDR/results/UQ/figures/CD/p_oi/final/{csv_filename[:-4]}.png')
        plt.show()