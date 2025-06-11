import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define save directory and create it if it doesn't exist
save_dir = os.path.join('..', 'results', 'UQ', 'figures', 'ipr')
os.makedirs(save_dir, exist_ok=True)

# Load CSV data
def load_data(prefix, suffix, spd_list):
    return [pd.read_csv(f'../results/BlueSky/500_samples/results_{prefix}_{spd}_{suffix}.csv') for spd in spd_list]

intruder_speeds = [5, 15, 20]

df_p_m = load_data('p_oi_m', '15_0', intruder_speeds)
df_p_v = load_data('p_oi_v', '15_0', intruder_speeds)
df_v_m = load_data('v_oi_m', '0_1.5', intruder_speeds)
df_v_v = load_data('v_oi_v', '0_1.5', intruder_speeds)
df_pv_m = load_data('pv_oi_m', '15_1.5', intruder_speeds)
df_pv_v = load_data('pv_oi_v', '15_1.5', intruder_speeds)

# Plotting function
def plot_polar_comparison(df_m_list, df_v_list, intruder_speeds, title_prefix, filename):
    plt.rcParams.update({'font.size': 14})
    fig, axs = plt.subplots(1, 3, subplot_kw={'projection': 'polar'}, figsize=(15, 5))
    lines = []

    for i in range(3):
        angles_m = df_m_list[i]['angles']
        ipr_m = df_m_list[i]['ipr']
        angles_v = df_v_list[i]['angles']
        ipr_v = df_v_list[i]['ipr']

        mask_m = (angles_m > 0) & (angles_m <= 180)
        mask_v = (angles_v > 0) & (angles_v <= 180)

        theta_m = np.deg2rad(angles_m[mask_m])
        theta_v = np.deg2rad(angles_v[mask_v])

        l1, = axs[i].plot(theta_v, ipr_v[mask_v], label='VO')
        l2, = axs[i].plot(theta_m, ipr_m[mask_m], label='MVP')

        if i == 0:
            lines = [l1, l2]

        axs[i].set_title(f'Intruder Speed: {intruder_speeds[i]} kts')
        axs[i].set_theta_zero_location('S')
        axs[i].set_thetamax(45)
        axs[i].set_thetagrids(np.arange(0, 45, 10))
        axs[i].set_rlim(0.0, 1.02)
        axs[i].set_theta_direction(-1)

    fig.legend(lines, [line.get_label() for line in lines],
               loc='lower center', ncol=2, frameon=True, bbox_to_anchor=(0.5, -0.1),
               fontsize=16)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, filename), bbox_inches='tight')
    # plt.show()

# Generate and save plots
plot_polar_comparison(df_p_m, df_p_v, intruder_speeds, title_prefix="Pos", filename="ipr_position.png")
plot_polar_comparison(df_v_m, df_v_v, intruder_speeds, title_prefix="Velo", filename="ipr_velocity.png")
plot_polar_comparison(df_pv_m, df_pv_v, intruder_speeds, title_prefix="PosVelo", filename="ipr_posvelo.png")

# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define save directory and create it if it doesn't exist
save_dir = os.path.join('..', 'results', 'UQ', 'figures', 'ipr')
os.makedirs(save_dir, exist_ok=True)

# Load only the intruder_speed = 20 files
def load_data(prefix, suffix):
    return pd.read_csv(f'../results/BlueSky/500_samples/results_{prefix}_20_{suffix}.csv')

# Load data for intruder speed = 20
df_p_m = load_data('p_oi_m', '15_0')
df_p_v = load_data('p_oi_v', '15_0')
df_v_m = load_data('v_oi_m', '0_1.5')
df_v_v = load_data('v_oi_v', '0_1.5')
df_pv_m = load_data('pv_oi_m', '15_1.5')
df_pv_v = load_data('pv_oi_v', '15_1.5')

# Plotting all three uncertainties in one figure
def plot_combined_polar(df_sets, title_labels, filename):
    plt.rcParams.update({'font.size': 14})
    fig, axs = plt.subplots(1, 3, subplot_kw={'projection': 'polar'}, figsize=(18, 6))
    lines = []

    for i, (df_m, df_v) in enumerate(df_sets):
        angles_m = df_m['angles']
        ipr_m = df_m['ipr']
        angles_v = df_v['angles']
        ipr_v = df_v['ipr']

        mask_m = (angles_m > 0) & (angles_m <= 180)
        mask_v = (angles_v > 0) & (angles_v <= 180)

        theta_m = np.deg2rad(angles_m[mask_m])
        theta_v = np.deg2rad(angles_v[mask_v])

        l1, = axs[i].plot(theta_v, ipr_v[mask_v], label='VO')
        l2, = axs[i].plot(theta_m, ipr_m[mask_m], label='MVP')

        if i == 0:
            lines = [l1, l2]

        axs[i].set_title(title_labels[i])
        axs[i].set_theta_zero_location('S')
        axs[i].set_thetamax(45)
        axs[i].set_thetagrids(np.arange(0, 45, 10))
        axs[i].set_rlim(0.0, 1.02)
        axs[i].set_theta_direction(-1)

    fig.legend(lines, [line.get_label() for line in lines],
               loc='lower center', ncol=2, frameon=True, bbox_to_anchor=(0.5, -0.1),
               fontsize=16)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, filename), bbox_inches='tight')
    # plt.show()

# Call the combined plot function
plot_combined_polar(
    df_sets=[
        (df_p_m, df_p_v),
        (df_v_m, df_v_v),
        (df_pv_m, df_pv_v)
    ],
    title_labels=["Position Uncertainty", "Velocity Uncertainty", "Pos+Velo Uncertainty"],
    filename="ipr_combined_speed20.png"
)
