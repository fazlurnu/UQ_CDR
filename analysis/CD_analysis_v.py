import pandas as pd
import json
import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm  # <-- Use parametric Gaussian

def compute_tcpa_stats_velocity_components(metadata, df):
    params = metadata['parameters']

    # Mean relative position (assumed fixed)
    x_own, y_own = df['x_own_true'].mean(), df['y_own_true'].mean()
    x_int, y_int = df['x_int_true'].mean(), df['y_int_true'].mean()
    x_rel = np.array([x_int - x_own, y_int - y_own])

    # Velocity components
    hdg_own = np.deg2rad(df['hdg_own_true'].mean())
    hdg_int = np.deg2rad(df['hdg_int_true'].mean())
    spd_own = df['gs_own_true'].mean()
    spd_int = df['gs_int_true'].mean()

    vx_own = spd_own * np.cos(hdg_own)
    vy_own = spd_own * np.sin(hdg_own)
    vx_int = spd_int * np.cos(hdg_int)
    vy_int = spd_int * np.sin(hdg_int)

    vx_rel = vx_own - vx_int
    vy_rel = vy_own - vy_int
    v_rel = np.array([vx_rel, vy_rel])

    A = np.dot(v_rel, x_rel)
    B = np.dot(v_rel, v_rel)
    mu_tcpa = A / B

    # Gradient of t_CPA wrt [vx_rel, vy_rel]
    dA_dv = x_rel
    dB_dv = 2 * v_rel
    grad_t = (1 / B) * dA_dv - (A / B**2) * dB_dv

    # Velocity covariance (assumed same for ownship & intruder, independent components)
    vel_acc = params['vel_acc']
    std_dev = vel_acc / 2.448
    cov_v = 2 * np.array([[std_dev**2, 0], 
                        [0, std_dev**2]])

    var_tcpa = grad_t @ cov_v @ grad_t.T

    return mu_tcpa, var_tcpa


def compute_dcpa_stats_velocity_components(metadata, df):
    params = metadata['parameters']

    # Relative position (fixed)
    x_own, y_own = df['x_own_true'].mean(), df['y_own_true'].mean()
    x_int, y_int = df['x_int_true'].mean(), df['y_int_true'].mean()
    x_rel = np.array([x_int - x_own, y_int - y_own])

    # Velocity components
    hdg_own = np.deg2rad(df['hdg_own_true'].mean())
    hdg_int = np.deg2rad(df['hdg_int_true'].mean())
    spd_own = df['gs_own_true'].mean()
    spd_int = df['gs_int_true'].mean()

    vx_own = spd_own * np.cos(hdg_own)
    vy_own = spd_own * np.sin(hdg_own)
    vx_int = spd_int * np.cos(hdg_int)
    vy_int = spd_int * np.sin(hdg_int)

    vx_rel = vx_own - vx_int
    vy_rel = vy_own - vy_int
    v_rel = np.array([vx_rel, vy_rel])

    A = np.dot(v_rel, x_rel)
    B = np.dot(v_rel, v_rel)
    t_cpa = A / B

    d_cpa_vec = x_rel - t_cpa * v_rel
    v_rel_perp = np.array([-vy_rel, vx_rel])
    v_rel_perp_unit = v_rel_perp / np.linalg.norm(v_rel)

    mu_dcpa = np.dot(d_cpa_vec, v_rel_perp_unit)

    # Derivative of t_CPA wrt [vx, vy]
    dA_dv = x_rel
    dB_dv = 2 * v_rel
    grad_t = (1 / B) * dA_dv - (A / B**2) * dB_dv

    # Derivative of dCPA = x_rel - t * v_rel ⇒ chain rule
    dd_dv = -np.outer(grad_t, v_rel) - t_cpa * np.eye(2)

    # Gradient of projection along v_perp
    grad_dcpa = dd_dv.T @ v_rel_perp_unit

    # Velocity covariance matrix
    vel_acc = params['vel_acc']
    std_dev = vel_acc / 2.448
    cov_v = 2 * np.array([[std_dev**2, 0], 
                        [0, std_dev**2]])

    var_dcpa = grad_dcpa.T @ cov_v @ grad_dcpa

    return mu_dcpa, var_dcpa

def compute_tin_stats_velocity_components(metadata, df):
    params = metadata['parameters']

    # Relative position (fixed)
    x_own, y_own = df['x_own_true'].mean(), df['y_own_true'].mean()
    x_int, y_int = df['x_int_true'].mean(), df['y_int_true'].mean()
    x_rel = np.array([x_int - x_own, y_int - y_own])

    # Velocity components (mean)
    hdg_own = np.deg2rad(df['hdg_own_true'].mean())
    hdg_int = np.deg2rad(df['hdg_int_true'].mean())
    spd_own = df['gs_own_true'].mean()
    spd_int = df['gs_int_true'].mean()

    vx_own = spd_own * np.cos(hdg_own)
    vy_own = spd_own * np.sin(hdg_own)
    vx_int = spd_int * np.cos(hdg_int)
    vy_int = spd_int * np.sin(hdg_int)

    vx_rel = vx_own - vx_int
    vy_rel = vy_own - vy_int
    v_rel = np.array([vx_rel, vy_rel])
    v_rel_norm = np.linalg.norm(v_rel)
    v_rel_norm_sq = v_rel_norm ** 2

    # Compute t_CPA and d_CPA
    A = np.dot(v_rel, x_rel)
    B = np.dot(v_rel, v_rel)
    t_cpa = A / B
    d_cpa = x_rel - t_cpa * v_rel
    d_cpa_norm_sq = np.dot(d_cpa, d_cpa)

    # Conflict zone radius
    R = params['rpz']

    # Check conflict condition
    if d_cpa_norm_sq >= R**2:
        return np.nan, np.nan  # No conflict (undefined t_in)

    sqrt_term = np.sqrt(R**2 - d_cpa_norm_sq)
    mu_tin = t_cpa - sqrt_term / v_rel_norm

    # Delta method: gradient of t_in wrt v_rel
    dA_dv = x_rel
    dB_dv = 2 * v_rel
    grad_t = (1 / B) * dA_dv - (A / B**2) * dB_dv

    # d_dCPA/dv = -grad_t * v_rel - t_cpa * I
    dd_dv = -np.outer(grad_t, v_rel) - t_cpa * np.eye(2)
    d_dnorm2_dv = 2 * d_cpa.T @ dd_dv  # gradient of ||d_cpa||^2 wrt v

    # Full gradient of t_in wrt v_rel
    grad_tin = - (1 / v_rel_norm) * grad_t + (1 / (v_rel_norm * sqrt_term)) * d_dnorm2_dv

    # Velocity uncertainty covariance
    vel_acc = params['vel_acc']
    std_dev = vel_acc / 2.448
    cov_v = 2 * np.array([[std_dev**2, 0], [0, std_dev**2]])

    var_tin = grad_tin.T @ cov_v @ grad_tin

    return mu_tin, var_tin

# --- Configuration ---
# csv_path = '../results/UQ/csv/CD/s_oi/CD_s_oi_250521_123537_b1592690.csv'
import glob

csv_path_list = glob.glob('../results/UQ/csv/CD/v_oi/final/*.csv')
# csv_path_list = csv_path_list[:]
# csv_path_list = [csv_path_list[i] for i in [7, 26]]

for csv_path in csv_path_list:
    metadata_dir = '../results/UQ/metadata'

    # --- Load CSV ---
    df = pd.read_csv(csv_path)
    csv_filename = os.path.basename(csv_path)

    spd_own = df['gs_own_true'].mean()
    spd_int = df['gs_int_true'].mean()
    print(spd_own, spd_int)

    # --- Extract Date Code from Filename ---
    match = re.search(r'CD_v_oi_(\d{6})_', csv_filename)
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
    spd_sigma = params.get('gs_sigma_ownship', 'N/A')
    hdg_sigma = params.get('hdg_sigma_ownship', 'N/A')

    # print(metadata['parameters'])
    # --- Check if CSV File is Listed in Metadata ---
    csv_files_list = metadata.get("output_files", {}).get("csv", [])

    print(df.columns)

    dpsi_val = (df['hdg_int_true'].mean())

    if(True):
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

                        mu_tin, var_tin = compute_tin_stats_velocity_components(metadata, df)
                        if not np.isnan(mu_tin):
                            sigma_tin = np.sqrt(var_tin)
                            x_vals = np.linspace(10, data.max() * 1.1, 500)
                            # x_vals = np.linspace(data.min(), data.max(), 500)
                            y_vals = norm.pdf(x_vals, mu_tin, sigma_tin)
                            ax.plot(x_vals, y_vals, color='red', label='Analytical Approximation', zorder=30)

                            ax.set_xlim([10, 25])

                    # Histogram as density (area = 1)
                    counts, bin_edges = np.histogram(data, bins=30, density=True)
                    ax.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), color = 'lightgray', edgecolor='black', align='edge', alpha=1.0, label='Samples', zorder = 20)

                    if var == 'tcpa':
                        mu_tcpa, var_tcpa = compute_tcpa_stats_velocity_components(metadata, df)
                        sigma_tcpa = np.sqrt(var_tcpa)

                        x_vals = np.linspace(data.min(), data.max(), 500)
                        y_vals = norm.pdf(x_vals, mu_tcpa, sigma_tcpa)

                        ax.plot(x_vals, y_vals, color='red', label=f'Analytical Approximation', zorder = 30)

                    elif var == 'dcpa':
                        mu_dcpa, var_dcpa = compute_dcpa_stats_velocity_components(metadata, df)
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

            dcpa_val = np.abs(np.round(mu_dcpa))

            plt.suptitle(
                rf"$Pos_{{acc bound}} = {pos_acc * 2}$, "
                rf"$Vel_{{acc bound}} = {vel_acc * 2}$, "
                + "\n" + 
                rf"$d_{{CPA}} = {dcpa_val}$, "
                rf"$\Delta\psi = {dpsi_val}$"
            )

            # if(dcpa_val == 45):
            plt.savefig(f'/Users/mfrahman/Python/0_UQ_CDR/results/UQ/figures/CD/v_oi/final/{csv_filename[:-4]}.png')

            plt.show()
