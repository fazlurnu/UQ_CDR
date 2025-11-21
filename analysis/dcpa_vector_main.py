import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import os

def cre_conflict(xref, yref, trkref, gsref,
                 dpsi, dcpa, tlosh, spd, rpz=50):
    trkref_rad = np.radians(trkref)
    trk = trkref + dpsi
    trk_rad = np.radians(trk)

    gsx = spd * np.cos(trk_rad)
    gsy = spd * np.sin(trk_rad)

    vrelx = gsref * np.cos(trkref_rad) - gsx
    vrely = gsref * np.sin(trkref_rad) - gsy
    vrel = np.sqrt(vrelx**2 + vrely**2)

    if dcpa == 0:
        drelcpa = (tlosh * vrel + np.sqrt(rpz**2 - dcpa**2))
    elif dcpa > rpz:
        drelcpa = 0
    else:
        drelcpa = tlosh * vrel + np.sqrt(rpz**2 - dcpa**2)

    dist = np.sqrt(drelcpa**2 + dcpa**2)
    rd = drelcpa / dist
    rx = dcpa / dist
    brn = np.degrees(np.arctan2(-rx * vrelx + rd * vrely, rd * vrelx + rx * vrely))

    xint, yint = dist * np.cos(np.radians(brn)), dist * np.sin(np.radians(brn))
    return xint, yint, trk, spd

# --- USER CONFIGURATION ---
# Prompt user for uncertainty type
uncertainty_type = input("Choose uncertainty type ('position' or 'velocity'): ").strip().lower()
if uncertainty_type not in ['position', 'velocity']:
    raise ValueError("Invalid choice. Please enter 'position' or 'velocity'.")

R = 50
xref, yref = 0, 0
trkref = 0
gsref = 20 / 1.94384
dpsi = 180
dcpa = 20
tlosh = 10
spd = 15 / 1.94384

# --- Conflict geometry ---
xint, yint, trkint, gsint = cre_conflict(xref, yref, trkref, gsref, dpsi, dcpa, tlosh, spd, rpz=R)
x_rel = np.array([xint - xref, yint - yref])
mu_rel = x_rel.copy()

vx_rel = gsref * np.cos(np.deg2rad(trkref)) - gsint * np.cos(np.deg2rad(trkint))
vy_rel = gsref * np.sin(np.deg2rad(trkref)) - gsint * np.sin(np.deg2rad(trkint))
v_rel = np.array([vx_rel, vy_rel])
mu_vrel = v_rel.copy()

np.random.seed(42)
N = 100000

if uncertainty_type == 'position':
    sigma_x = 15 / 2.448
    sigma_y = 15 / 2.448
    Sigma = np.diag([sigma_x**2, sigma_y**2])
    x_rel_samples = np.random.multivariate_normal(mu_rel, Sigma, N)

    t_cpa_samples = (x_rel_samples @ v_rel) / (v_rel @ v_rel)
    d_cpa_samples = x_rel_samples - t_cpa_samples[:, None] * v_rel

    t_cpa_det = (mu_rel @ v_rel) / (v_rel @ v_rel)
    d_cpa_det = mu_rel - t_cpa_det * v_rel

    label_extra = 'Position Uncertainty'
    x_for_align = v_rel

else:  # velocity uncertainty
    sigma_vx = 0.5 / 2.448
    sigma_vy = 0.5 / 2.448
    Sigma = np.diag([sigma_vx**2, sigma_vy**2])
    v_rel_samples = np.random.multivariate_normal(mu_vrel, Sigma, N)

    t_cpa_samples = (v_rel_samples @ x_rel) / np.einsum('ij,ij->i', v_rel_samples, v_rel_samples)
    d_cpa_samples = x_rel - t_cpa_samples[:, None] * v_rel_samples

    t_cpa_det = (mu_vrel @ x_rel) / (mu_vrel @ mu_vrel)
    d_cpa_det = x_rel - t_cpa_det * mu_vrel

    label_extra = 'Velocity Uncertainty'
    x_for_align = mu_vrel

# --- Coordinate transformation ---
u1 = x_for_align / np.linalg.norm(x_for_align)
u2 = np.array([-u1[1], u1[0]])
T = np.vstack([u2, u1])

d_cpa_trans = (T @ d_cpa_samples.T).T
d_cpa_det_trans = T @ d_cpa_det

d_cpa_trans[:, 0] -= d_cpa_det_trans[0]
d_cpa_trans[:, 1] = 0.0
x_aligned = d_cpa_trans[:, 0]

estimated_std = np.std(x_aligned)
estimated_var = np.var(x_aligned)

# --- Plot: 2D Distribution ---
plt.figure(figsize=(6, 6))
plt.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.2)
plt.axvline(0, color='black', linewidth=1, linestyle='-', alpha=0.2)
plt.scatter(d_cpa_samples[:, 0], d_cpa_samples[:, 1], alpha=0.1, s=2, label='$\\mathbf{d}_{CPA}$ samples')
plt.plot(d_cpa_det[0], d_cpa_det[1], 'ro', label='Deterministic $\\mathbf{d}_{CPA}$', markersize=6)
plt.xlabel('$d_{CPA,x}$ [m]')
plt.ylabel('$d_{CPA,y}$ [m]')
plt.title(f'2D Distribution of $\\mathbf{{d}}_{{CPA}}$ under {label_extra}')
plt.axis('equal')

legend_elements = [
    Line2D([0], [0], marker='o', color='tab:blue', label='$\\mathbf{d}_{CPA}$ samples', markersize=6, linestyle='None', alpha=0.9),
    Line2D([0], [0], marker='o', color='r', label='Deterministic $\\mathbf{d}_{CPA}$', markersize=6, linestyle='None', alpha=0.9)
]
plt.legend(handles=legend_elements)

plt.tight_layout()

save_dir = os.path.join('..', 'results', 'UQ', 'figures', 'dcpa_uncertainty')
os.makedirs(save_dir, exist_ok=True)

filename = f"{uncertainty_type}_dcpa_vector_distribution.png"
filepath = os.path.join(save_dir, filename)

plt.savefig(filepath, dpi=300)
plt.show()
plt.close()
# print(f"Saved figure: {filepath}")

# plt.savefig(uncertainty_type)

# --- Output Statistics ---
print("Estimated STD:", estimated_std)
print("Estimated VAR:", estimated_var)

## tin calculation
# Re-declare essential variables to align with previous context
vrel_mag = np.linalg.norm(mu_vrel)

# Recompute t_in using analytical equation (Eq. tin)
dcpa_norm_sq = np.einsum('ij,ij->i', d_cpa_samples, d_cpa_samples)
inside_sqrt = R**2 - dcpa_norm_sq
valid_mask = inside_sqrt >= 0

tin_samples = np.full(N, np.nan)
tin_samples[valid_mask] = t_cpa_samples[valid_mask] - np.sqrt(inside_sqrt[valid_mask]) / vrel_mag
tin_samples_clean = tin_samples[~np.isnan(tin_samples)]

# Plot histogram of t_in
plt.figure(figsize=(8, 4))
plt.hist(tin_samples_clean, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Time to Intrusion Entry $t_{in}$ [s]')
plt.ylabel('Frequency')
plt.title(f'Distribution of $t_{{in}}$ under {label_extra}')
plt.grid(True)

# Save figure
# save_dir = '/mnt/data/uq_figures'
# os.makedirs(save_dir, exist_ok=True)
# filepath_tin = os.path.join(save_dir, f"{uncertainty_type}_tin_distribution_eq9.png")
# plt.savefig(filepath_tin, dpi=300)
# plt.close()

# Output summary statistics
tin_mean = np.mean(tin_samples_clean)
tin_std = np.std(tin_samples_clean)
tin_min = np.min(tin_samples_clean)
tin_max = np.max(tin_samples_clean)

print(tin_mean, tin_std, tin_min, tin_max)
plt.show()