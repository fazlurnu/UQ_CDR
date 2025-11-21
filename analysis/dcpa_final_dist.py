# Re-import necessary packages due to code execution environment reset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os

# Redefine angle list
angles_of_interest = list(range(2, 31, 2))

plt.rcParams.update({'font.size': 14})

def remove_outliers(data, factor=1.5):
    if len(data) < 4:  # not enough points to define quartiles
        return data
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    return [x for x in data if lower_bound <= x <= upper_bound]

# Redefine function to extract scientific notation values
def extract_dcpa_values(df, angle):
    entry = df[df['angles'] == angle]['distance_cpa']
    if entry.empty:
        return []
    raw = entry.values[0]
    raw_cleaned = re.sub(r'dtype=\w+', '', raw)
    nums = re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?', raw_cleaned)
    return [float(num) for num in nums if num != '64']

# Re-load the uploaded files
file_m = '../results/BlueSky/50000_samples/results_p_oi_m_20_15_0_15_receptionprob80.csv'
file_v = '../results/BlueSky/50000_samples/results_p_oi_v_20_15_0_15_receptionprob80.csv'

df_m = pd.read_csv(file_m)
df_v = pd.read_csv(file_v)

# Extract the DCPA values
data_m = [extract_dcpa_values(df_m, angle) for angle in angles_of_interest]
data_v = [extract_dcpa_values(df_v, angle) for angle in angles_of_interest]

# Colors
color_m = 'tab:orange'
color_v = 'tab:blue'

# Plot half-half violin plot
plt.figure(figsize=(12, 6))

for i, angle in enumerate(angles_of_interest):
    m_data = data_m[i]
    v_data = data_v[i]

    print(min(m_data), min(v_data))
    pos = angle

    # Plot POI-M on the left
    vp_m = plt.violinplot(m_data, positions=[pos], widths=1.2, showmeans=False, showmedians=False, showextrema=False)
    for b in vp_m['bodies']:
        b.set_facecolor(color_m)
        b.set_edgecolor(color_m)
        b.set_alpha(0.7)
        b.set_clip_path(plt.matplotlib.patches.PathPatch(
            plt.matplotlib.path.Path([[pos, -1000], [pos, 1000], [pos - 0.6, 1000], [pos - 0.6, -1000]]),
            transform=plt.gca().transData
        ))

    # Plot POI-V on the right
    vp_v = plt.violinplot(v_data, positions=[pos], widths=1.2, showmeans=False, showmedians=False, showextrema=False)
    for b in vp_v['bodies']:
        b.set_facecolor(color_v)
        b.set_edgecolor(color_v)
        b.set_alpha(0.7)
        b.set_clip_path(plt.matplotlib.patches.PathPatch(
            plt.matplotlib.path.Path([[pos, -1000], [pos, 1000], [pos + 0.6, 1000], [pos + 0.6, -1000]]),
            transform=plt.gca().transData
        ))

# Final layout adjustments
plt.xticks(angles_of_interest)
plt.xlabel('Heading Angle Difference [deg]')
plt.ylabel('Distance at CPA [m]')

# Add reference lines with legend
red_line = plt.axhline(50, color='red', linestyle='--', label=r'$R_{\mathrm{PZ}} = 50~\mathrm{m}$')
gray_line = plt.axhline(0, color='gray', linestyle='--', label=r'Initial $\|\mathbf{d}_{\mathrm{CPA}}\|$')

# Add legend
legend_elements = [
    plt.Line2D([0], [0], color=color_m, lw=4, label='MVP'),
    plt.Line2D([0], [0], color=color_v, lw=4, label='VO'),
    red_line,
    gray_line
]
plt.legend(handles=legend_elements, loc='lower right')
plt.tight_layout()

save_dir = os.path.join('..', 'results', 'UQ', 'figures', 'dcpa_final')
os.makedirs(save_dir, exist_ok=True)

base_filename = f"violin"
fig_path = os.path.join(save_dir, base_filename + '.png')
# csv_path = os.path.join(save_dir, base_filename + '.csv')

plt.savefig(fig_path, dpi=300)
plt.show()
