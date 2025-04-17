import os

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from shapely.geometry import Point, LineString, Polygon
from shapely.affinity import translate

from autonomous_separation.conf_reso.algorithms.VO import VOResolution
from autonomous_separation.conf_reso.algorithms.MVP import MVPResolution

OPACITY = 0.4

def plot_uncertainty(df,
                     sim_params
    ):

    vo = VOResolution()
    mvp = MVPResolution()

    df_vo = df[df['is_conflict']][['vx_vo', 'vy_vo']].copy()
    df_vo['vx'] = df['vx_vo']
    df_vo['vy'] = df['vy_vo']
    df_vo['Conf Reso'] = 'VO'

    df_mvp = df[df['is_conflict']][['vx_mvp', 'vy_mvp']].copy()
    df_mvp['vx'] = df['vx_mvp']
    df_mvp['vy'] = df['vy_mvp']
    df_mvp['Conf Reso'] = 'MVP'

    df_plot = pd.concat([df_vo, df_mvp], ignore_index=True)
    all_vx = pd.concat([df_vo['vx'], df_mvp['vx']])
    all_vy = pd.concat([df_vo['vy'], df_mvp['vy']])

    # Create jointplot
    f = sns.jointplot(
        data=df_plot,
        x="vy", y="vx", hue="Conf Reso",
        kind="scatter", alpha=OPACITY,
        zorder = 100,
        palette=['tab:blue', 'tab:orange']
    )
    f.set_axis_labels("Cross-track resolution speed [kts]", "Along-track resolution speed [kts]")

    # Set x and y limits around the ownship velocity
    vy_init = sim_params.gs_own * np.sin(np.radians(sim_params.hdg_own))
    vx_init = sim_params.gs_own * np.cos(np.radians(sim_params.hdg_own))

    # Compute max absolute values
    max_vx = all_vx.abs().max() - vx_init
    max_vy = all_vy.abs().max() - vy_init

    # Update offsets if needed
    max_offset_vx = max(max_vx, 7.5)
    max_offset_vy = max(max_vy, 7.5)

    offset = max(max_offset_vx, max_offset_vy)

    # Apply limits to plot
    f.ax_joint.set_xlim(vy_init - offset, vy_init + offset)
    f.ax_joint.set_ylim(vx_init - offset, vx_init + offset)

    # --- Plot the VO region, if possible ---
    tp_1, tp_2 = vo.get_cc_tp(Point(sim_params.x_own, sim_params.y_own),
                              Point(sim_params.x_int, sim_params.y_int), sim_params.rpz)
    
    int_vel = Point(sim_params.gs_int * np.cos(np.radians(sim_params.hdg_int)),
                    sim_params.gs_int * np.sin(np.radians(sim_params.hdg_int)))
    
    own_vel = Point((sim_params.gs_own) * np.cos(np.radians(sim_params.hdg_own)),
                    (sim_params.gs_own) * np.sin(np.radians(sim_params.hdg_own)))
    
    ownship_pos = Point(sim_params.x_own, sim_params.y_own)

    if tp_1 is not None and tp_2 is not None:
        vo_0 = translate(ownship_pos, xoff=int_vel.x, yoff=int_vel.y)
        vo_1 = translate(tp_1, xoff=int_vel.x, yoff=int_vel.y)
        vo_2 = translate(tp_2, xoff=int_vel.x, yoff=int_vel.y)

        dx_1, dy_1 = (vo_1.x - vo_0.x), (vo_1.y - vo_0.y)
        dx_2, dy_2 = (vo_2.x - vo_0.x), (vo_2.y - vo_0.y)

        # Extend lines for visualization
        extension_length = 100
        extended_endpoint_1 = Point(vo_1.x + dx_1 * extension_length,
                                    vo_1.y + dy_1 * extension_length)
        extended_endpoint_2 = Point(vo_2.x + dx_2 * extension_length,
                                    vo_2.y + dy_2 * extension_length)

        f.ax_joint.plot([vo_0.y, extended_endpoint_1.y],
                        [vo_0.x, extended_endpoint_1.x],
                        color='k')
        f.ax_joint.plot([vo_0.y, extended_endpoint_2.y],
                        [vo_0.x, extended_endpoint_2.x],
                        color='k')

        # Fill polygon between the two lines
        vo_line_1 = LineString([vo_0, vo_1])
        vo_line_2 = LineString([vo_0, vo_2])
        coords_1 = list(vo_line_1.coords)
        coords_2 = list(vo_line_2.coords)
        polygon_coords = coords_1 + coords_2[::-1]
        polygon = Polygon(polygon_coords)

        ax = f.ax_joint
        x_poly, y_poly = polygon.exterior.xy
        ax.fill(y_poly, x_poly, alpha=0.3, fc='red', zorder=-1)

    # --- Plot true resolutions for VO and MVP ---
    vx_true_vo, vy_true_vo = vo.resolve(
        Point(sim_params.x_own, sim_params.y_own), sim_params.gs_own, sim_params.hdg_own,
        Point(sim_params.x_int, sim_params.y_int), sim_params.gs_int, sim_params.hdg_int,
        rpz = sim_params.rpz, tlookahead=sim_params.tlosh,
        method=0,
    )
    vx_true_mvp, vy_true_mvp, dcpa = mvp.resolve_with_dcpa(
        Point(sim_params.x_own, sim_params.y_own), sim_params.gs_own, sim_params.hdg_own,
        Point(sim_params.x_int, sim_params.y_int), sim_params.gs_int, sim_params.hdg_int,
        rpz = sim_params.rpz, tlookahead=sim_params.tlosh,
    )

    handles, labels = f.ax_joint.get_legend_handles_labels()
    f.ax_joint.legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(1.25, 1.2))

    # Mark lines/points for VO
    # f.ax_joint.axhline(vx_true_vo, color='tab:blue', linestyle='--',
    #                    linewidth=1.5, alpha=0.7)
    # f.ax_joint.axvline(vy_true_vo, color='tab:blue', linestyle='--',
    #                    linewidth=1.5, alpha=0.7)
    f.ax_joint.scatter(vy_true_vo, vx_true_vo, s=100, color='blue',
                        marker='*', zorder=10, label='True VO')

    # Mark lines/points for MVP
    # f.ax_joint.axhline(vx_true_mvp, color='tab:orange', linestyle='--',
    #                    linewidth=1.5, alpha=0.7)
    # f.ax_joint.axvline(vy_true_mvp, color='tab:orange', linestyle='--',
    #                    linewidth=1.5, alpha=0.7)
    f.ax_joint.scatter(vy_true_mvp, vx_true_mvp, s=100, color='orange',
                        marker='*', zorder=10, label='True MVP')

    # Rename legend labels for clarity
    handles, labels = f.ax_joint.get_legend_handles_labels()
    if labels:
        labels[0] = "VO Samples"
        if len(labels) > 1:
            labels[1] = "MVP Samples"
    f.ax_joint.legend(handles=handles, labels=labels, loc='upper left')

    # draw relative velocity
    # rel_vel = Point(int_vel.x - own_vel.x, int_vel.y - own_vel.y)

    # plt.plot([ownship_pos.x, ownship_pos.x-rel_vel.x], [ownship_pos.y, ownship_pos.y-rel_vel.y], color = 'g', label = 'Rel Velo')
    # plt.plot([ownship_pos.y, ownship_pos.y+y_int], [ownship_pos.x, ownship_pos.x+x_int], linestyle = '--', color = 'r')
    # plt.plot([ownship_pos.y, ownship_pos.y+int_vel.y], [ownship_pos.x, ownship_pos.x+int_vel.x], color = 'r')
    # plt.plot([ownship_pos.y, ownship_pos.y-rel_vel.y*100], [ownship_pos.x, ownship_pos.x-rel_vel.x*100], color = 'g', linestyle = '--', alpha = 0.5)
    
    if(np.sqrt(dcpa[0]**2 + dcpa[1]**2) < 1):
        dcpa = dcpa * (1/np.sqrt(dcpa[0]**2 + dcpa[1]**2))

    f.ax_joint.plot([own_vel.y, own_vel.y + dcpa[0] * 100], [own_vel.x, own_vel.x + dcpa[1] * 100], '--r', zorder = 1)
    f.ax_joint.plot([own_vel.y, own_vel.y - dcpa[0] * 100], [own_vel.x, own_vel.x - dcpa[1] * 100], '--r', zorder = 1)

    # fig_filename = (
                #     f"{sim_params.case_title_selected}_{sim_params.source_of_uncertainty}_"
                #     f"{sim_params.gs_int}_{sim_params.dpsi_val}_{sim_params.dcpa_val}_{sim_params.rpz}_{sim_params.tlosh}.png"
                # )
    
    # fig_path = os.path.join(self.clustering.OUTPUT_DIR, fig_filename)
    # f.savefig(fig_path, dpi=300, bbox_inches='tight')
    # plt.close()  # or plt.close() if you don't want pop-up windows

    # plt.show()

    return f