import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os, re

# Pattern to match all relevant files
file_pattern = '../results/UQ/data/conflict_results_*_oi_*.csv'
files = glob.glob(file_pattern)

dfs = []

dfs.append(pd.read_csv('../results/UQ/data/conflict_results_p_oi_250527_180003.csv'))
# dfs.append(pd.read_csv('results/UQ/data/conflict_results_p_oi_250602_161326.csv'))

dfs.append(pd.read_csv('../results/UQ/data/conflict_results_v_oi_250603_174521.csv'))
# dfs.append(pd.read_csv('results/UQ/data/conflict_results_pv_oi_250529_144944.csv'))

# dfs.append(pd.read_csv('results/UQ/data/conflict_results_v_oi_250602_143018.csv'))
dfs[0]['type'] = 'p'
dfs[1]['type'] = 'v'

# Prepare plotting data
def prepare_plot_data(df):
    theta = np.radians(df['dpsi_val'])
    r = df['dcpa_val']
    color_val = df['is_conflict']
    size_val = (color_val - 20) * 2
    return theta, r, color_val, size_val

titles = []
for df in dfs:
    title_here = df['type'].unique()[0]
    if(title_here == 'v'):
        titles.append('Velocity Uncertainty')
    elif(title_here == 'p'):
        titles.append('Position Uncertainty')
    elif(title_here == 'shp'):
        titles.append('Integrated Uncertainty')
    else:
        titles.append('Undefined Uncertainty')

print(len(dfs))

theta1, r1, c1, s1 = prepare_plot_data(dfs[0])
theta2, r2, c2, s2 = prepare_plot_data(dfs[1])

norm = plt.Normalize(45, 55)

# Create side-by-side polar plots
fig, axs = plt.subplots(1, 2, figsize=(18, 10), subplot_kw={'projection': 'polar'}, constrained_layout=True)

for i, (ax, theta, r, color_val, size_val, title) in enumerate(zip(
    axs,
    [theta1, theta2],
    [r1, r2],
    [c1, c2],
    [s1, s2],
    titles
)):
    ax.set_theta_zero_location('S')
    ax.set_theta_direction(1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_axisbelow(True)
    ax.grid(True)
    ax.set_rlim(-4, 49)

    sc = ax.scatter(theta, r, c=color_val, cmap='Reds', norm=norm, s=50)
    ax.set_title(f'{title}', va='bottom')

    mean_val = color_val.mean()
    max_idx = color_val.argmax()
    min_idx = color_val.argmin()

    print(f"{title}:")
    print(f"  Mean color value: {mean_val:.2f}")
    print(f"  Max color value: {color_val[max_idx]:.2f} at dpsi={np.degrees(theta[max_idx]):.2f}, dcpa={r[max_idx]:.2f}")
    print(f"  Min color value: {color_val[min_idx]:.2f} at dpsi={np.degrees(theta[min_idx]):.2f}, dcpa={r[min_idx]:.2f}")

    # Radial label only for the first plot
    if i == 0:
        ax.set_ylabel('Distance at CPA [m]', fontsize=14, labelpad=-50)
        # Angular axis (theta)
        ax.set_thetagrids(range(0, 181, 30), labels=[r'${}$'.format(t) for t in range(0, 181, 30)], fontsize=12)
        ax.text(np.radians(45), ax.get_rmax() + 10.0, 'Heading\nDiff [$^\circ$]', ha='center', va='center', fontsize=14)

    else:
        ax.set_yticklabels([])
        ax.set_thetagrids(range(0, 181, 30), labels=[r'${}$'.format(t) for t in range(0, 181, 30)], fontsize=12)

cbar_ax = fig.add_axes([0.25, -0.0, 0.5, 0.03])
cbar = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Detection Probability [%]', fontsize=14, fontweight='bold')
cbar.ax.tick_params(labelsize=10)

# plt.tight_layout()
fig_output_dir = f"../results/UQ/figures/CD/detect_prob/"
os.makedirs(fig_output_dir, exist_ok=True)  # Create directory if it doesn't exist
fig_name = f"CD_detect_prob_separate.png"
# fig_path = os.path.join(fig_output_dir, fig_name)

# fig.savefig(fig_path, dpi=300)  # Save the figure with high resolution

# Create a boxplot for detection probabilities
fig_box, ax_box = plt.subplots(figsize=(8, 6))

# Prepare data for boxplot
data = [df['is_conflict'] for df in dfs]
labels = titles  # ['Velocity Uncertainty', 'Position Uncertainty']

# Create the boxplot
ax_box.boxplot(data, labels=labels, patch_artist=True,
               boxprops=dict(facecolor='tab:gray', color='black'),
               medianprops=dict(color='black'),
               whiskerprops=dict(color='black'),
               capprops=dict(color='black'),
               flierprops=dict(markerfacecolor='tab:gray', marker='o', markersize=5, linestyle='none'))

# Labeling
ax_box.set_ylabel('Detection Probability [%]', fontsize=14)
# ax_box.set_title('Distribution of Detection Probabilities by Uncertainty Type', fontsize=16)
# ax_box.grid(True, linestyle='--', alpha=0.6)
ax_box.tick_params(axis='both', labelsize=14)  # Set tick font size

ax_box.set_ylim([40, 55])

# Save the boxplot
boxplot_name = "CD_detect_prob_boxplot.png"
boxplot_path = os.path.join(fig_output_dir, boxplot_name)
fig_box.savefig(boxplot_path, dpi=300)
# fig_box.show()
