import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

# --- Constants and Setup ---
np.random.seed(42)
N = 100000
R = 50
xref, yref = 0, 0
trkref = 0
gsref = 20 / 1.94384
tlosh = 15
spd = 15 / 1.94384

def cre_conflict(xref, yref, trkref, gsref, dpsi, dcpa, tlosh, spd, rpz=50):
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

def simulate_case(dpsi, dcpa, color, label, ax):
    xint, yint, trkint, gsint = cre_conflict(xref, yref, trkref, gsref, dpsi, dcpa, tlosh, spd, rpz=R)
    x_rel = np.array([xint - xref, yint - yref])

    mu_vrel = np.array([
        gsref * np.cos(np.deg2rad(trkref)) - gsint * np.cos(np.deg2rad(trkint)),
        gsref * np.sin(np.deg2rad(trkref)) - gsint * np.sin(np.deg2rad(trkint))
    ])

    sigma_vx = 0.5 / 2.448
    sigma_vy = 0.5 / 2.448
    Sigma = np.diag([sigma_vx**2, sigma_vy**2])
    v_rel_samples = np.random.multivariate_normal(mu_vrel, Sigma, N)

    t_cpa_samples = (v_rel_samples @ x_rel) / np.einsum('ij,ij->i', v_rel_samples, v_rel_samples)
    d_cpa_samples = x_rel - t_cpa_samples[:, None] * v_rel_samples

    t_cpa_det = (mu_vrel @ x_rel) / (mu_vrel @ mu_vrel)
    d_cpa_det = x_rel - t_cpa_det * mu_vrel

    ax.scatter(d_cpa_samples[:, 0], d_cpa_samples[:, 1], alpha=0.05, s=2, color=color, label=f'{label} Samples')
    ax.plot(d_cpa_det[0], d_cpa_det[1], '*', color=color, markersize=10, label=f'{label} Deterministic')

    x_for_align = mu_vrel
    u1 = x_for_align / np.linalg.norm(x_for_align)
    u2 = np.array([-u1[1], u1[0]])
    T = np.vstack([u2, u1])

    d_cpa_trans = (T @ d_cpa_samples.T).T
    d_cpa_det_trans = T @ d_cpa_det
    d_cpa_trans[:, 0] -= d_cpa_det_trans[0]

    x_aligned = d_cpa_trans[:, 0]
    print(f"Case {label}: dpsi = {dpsi}, dcpa = {dcpa}")
    print("  Estimated STD:", np.std(x_aligned))
    print("  Estimated VAR:", np.var(x_aligned))

# --- Combined Figure ---
fig, ax = plt.subplots(figsize=(7, 7))

# Plot both cases in one figure
simulate_case(dpsi=180, dcpa=20, color='tab:blue', label='dψ=180°, dCPA=20m', ax=ax)
simulate_case(dpsi=10, dcpa=45, color='tab:green', label='dψ=10°, dCPA=45m', ax=ax)


# Axis settings
ax.set_title('2D Distributions of $\\mathbf{d}_{CPA}$  under Velocity Uncertainty')
ax.set_xlabel('$d_{CPA,x}$ [m]')
ax.set_ylabel('$d_{CPA,y}$ [m]')
ax.axhline(0, color='black', linestyle='--', alpha=0.3)
ax.axvline(0, color='black', linestyle='--', alpha=0.3)
ax.axis('equal')

# Custom legend elements with full opacity
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], marker='o', linestyle='None', color='tab:blue', label='dψ=180°, dCPA=20m Samples',
           markerfacecolor='tab:blue', markeredgecolor='tab:blue', markersize=6),
    Line2D([0], [0], marker='*', linestyle='None', color='tab:blue', label='dψ=180°, dCPA=20m Deterministic',
           markerfacecolor='tab:blue', markersize=10),
    Line2D([0], [0], marker='o', linestyle='None', color='tab:green', label='dψ=10°, dCPA=45m Samples',
           markerfacecolor='tab:green', markeredgecolor='tab:green', markersize=6),
    Line2D([0], [0], marker='*', linestyle='None', color='tab:green', label='dψ=10°, dCPA=45m Deterministic',
           markerfacecolor='tab:green', markersize=10)
]


ax.legend(handles=legend_elements)

# Save and show
save_dir = os.path.join('..', 'results', 'UQ', 'figures', 'dcpa_uncertainty')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "velocity_uncertainty_combined.png")
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()

print(f"Saved figure: {save_path}")