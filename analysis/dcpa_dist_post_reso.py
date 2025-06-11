import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, re

def compute_dcpa_tin(df, dt=1.0, R=50):
    # Predicted positions
    for method in ['mvp', 'vo']:
        df[f'x_own_next_{method}'] = df['x_own_true'] + df[f'vx_{method}'] * dt
        df[f'y_own_next_{method}'] = df['y_own_true'] + df[f'vy_{method}'] * dt
        df[f'x_int_next_{method}'] = df['x_int_true'] + df[f'vx_{method}_int'] * dt
        df[f'y_int_next_{method}'] = df['y_int_true'] + df[f'vy_{method}_int'] * dt

        # Relative positions and velocities
        x_rel = df[f'x_int_next_{method}'] - df[f'x_own_next_{method}']
        y_rel = df[f'y_int_next_{method}'] - df[f'y_own_next_{method}']
        vx_rel = df[f'vx_{method}'] - df[f'vx_{method}_int']
        vy_rel = df[f'vy_{method}'] - df[f'vy_{method}_int']
        v_rel_sq = vx_rel**2 + vy_rel**2
        v_rel = np.sqrt(v_rel_sq)

        # Time to Closest Point of Approach (TCPA)
        tcpa = (x_rel * vx_rel + y_rel * vy_rel) / v_rel_sq
        dcpa = np.sqrt((x_rel - tcpa * vx_rel) ** 2 + (y_rel - tcpa * vy_rel) ** 2)

        df[f'dcpa_next_{method}'] = dcpa

        dcpa2 = dcpa**2
        valid = dcpa2 <= R**2

        df[f'tin_next_{method}'] = np.nan
        df.loc[valid, f'tin_next_{method}'] = (
            tcpa[valid] - np.sqrt(R**2 - dcpa2[valid]) / v_rel[valid]
        )

    return df


def plot_comparison(df1, df2, uncertainty, label1, label2, max_val=110):
    def extract_dcpa(df):
        conflict = df[df['is_conflict']]
        return {key: conflict[f'dcpa_next_{key}'].dropna() for key in ['vo', 'mvp']}

    dcpa1, dcpa2 = extract_dcpa(df1), extract_dcpa(df2)
    bins = np.arange(0, np.ceil(max_val / 5) * 5 + 5, 5)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    width = (bins[1] - bins[0]) * 0.4

    def get_max_height(dcpa):
        hist_vo, _ = np.histogram(dcpa['vo'], bins=bins)
        hist_mvp, _ = np.histogram(dcpa['mvp'], bins=bins)
        norm_vo = hist_vo / hist_vo.sum() * 100 if hist_vo.sum() else 0
        norm_mvp = hist_mvp / hist_mvp.sum() * 100 if hist_mvp.sum() else 0
        return max(norm_vo.max(), norm_mvp.max())

    ymax = np.ceil(max(get_max_height(dcpa1), get_max_height(dcpa2)) * 1.1)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    def plot_hist(ax, dcpa, title, ylabel=False):
        counts_vo, _ = np.histogram(dcpa['vo'], bins=bins)
        counts_mvp, _ = np.histogram(dcpa['mvp'], bins=bins)
        norm_vo = counts_vo / counts_vo.sum() * 100
        norm_mvp = counts_mvp / counts_mvp.sum() * 100

        ax.bar(bin_centers - width / 2, norm_vo, width=width, alpha=0.6, label='VO')
        ax.bar(bin_centers + width / 2, norm_mvp, width=width, alpha=0.6, label='MVP')
        ax.set_xlim([0, max_val])
        ax.set_ylim([0, 105])
        ax.set_title(f'{uncertainty}, hdg diff = {title}')
        ax.set_xlabel('$|\mathbf{d}_{CPA}|$ post-1st-resolution [m]')
        if ylabel:
            ax.set_ylabel('Normalized Frequency [%]')
        else:
            ax.set_yticks([])
        ax.legend()

    plot_hist(axs[0], dcpa1, label1, ylabel=True)
    plot_hist(axs[1], dcpa2, label2)

    plt.tight_layout()

    # === Save Figure ===
    def clean(text):
        return re.sub(r'[^\w\-]+', '_', text.strip('$^{} '))

    save_dir = os.path.join('..', 'results', 'UQ', 'figures', 'dcpa_uncertainty')
    os.makedirs(save_dir, exist_ok=True)

    filename = f"{clean(uncertainty)}__{clean(label1)}__vs__{clean(label2)}.png"
    filepath = os.path.join(save_dir, filename)

    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Saved figure: {filepath}")

def compute_percent_within_radius(df, method='vo', R=50):
    conflict = df[df['is_conflict']]
    return (conflict[f'dcpa_next_{method}'] < R).mean() * 100


# === RUN SCENARIOS === #
def run_scenario(csv1, csv2, uncertainty, label1, label2, angle):
    df1 = compute_dcpa_tin(pd.read_csv(csv1))
    df2 = compute_dcpa_tin(pd.read_csv(csv2))
    plot_comparison(df1, df2, uncertainty, label1, label2)

    print(uncertainty)
    for method in ['vo', 'mvp']:
        a = compute_percent_within_radius(df1, method)
        b = compute_percent_within_radius(df2, method)
        print(f"{method.upper()} {angle}deg, 0m: {a:.1f}%")
        print(f"{method.upper()} {angle}deg, 45m: {b:.1f}%")


# === EXAMPLES === #
run_scenario(
    '../results/UQ/csv/p_oi_250611_101425_c88bee1a.csv',
    '../results/UQ/csv/p_oi_250611_101605_212dd82f.csv',
    'Pos uncertainty',
    '2$^\circ$, init $|\mathbf{d}_{CPA}| = 0$ m',
    '2$^\circ$, init $|\mathbf{d}_{CPA}| = 45$ m',
    angle=2
)

run_scenario(
    '../results/UQ/csv/p_oi_250611_102846_2d85a591.csv',
    '../results/UQ/csv/p_oi_250611_102900_ddab6ff2.csv',
    'Pos uncertainty',
    '30$^\circ$, init $|\mathbf{d}_{CPA}| = 0$ m',
    '30$^\circ$, init $|\mathbf{d}_{CPA}| = 45$ m',
    angle=30
)

run_scenario(
    '../results/UQ/csv/v_oi_250611_104319_0d8f668d.csv',
    '../results/UQ/csv/v_oi_250611_104326_146f54c4.csv',
    'Velo uncertainty',
    '2$^\circ$, init $|\mathbf{d}_{CPA}| = 0$ m',
    '2$^\circ$, init $|\mathbf{d}_{CPA}| = 45$ m',
    angle=2
)

run_scenario(
    '../results/UQ/csv/v_oi_250611_104715_64f6bc1e.csv',
    '../results/UQ/csv/v_oi_250611_104724_57f0ef02.csv',
    'Velo uncertainty',
    '30$^\circ$, init $|\mathbf{d}_{CPA}| = 0$ m',
    '30$^\circ$, init $|\mathbf{d}_{CPA}| = 45$ m',
    angle=30
)

run_scenario(
    '../results/UQ/csv/pv_oi_250611_105707_7ba66c92.csv',
    '../results/UQ/csv/pv_oi_250611_105713_5d1f066f.csv',
    'PosVelo uncertainty',
    '2$^\circ$, init $|\mathbf{d}_{CPA}| = 0$ m',
    '2$^\circ$, init $|\mathbf{d}_{CPA}| = 45$ m',
    angle=2
)

run_scenario(
    '../results/UQ/csv/pv_oi_250611_105958_2cd0689f.csv',
    '../results/UQ/csv/pv_oi_250611_110003_93254970.csv',
    'PosVelo uncertainty',
    '30$^\circ$, init $|\mathbf{d}_{CPA}| = 0$ m',
    '30$^\circ$, init $|\mathbf{d}_{CPA}| = 45$ m',
    angle=30
)

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

