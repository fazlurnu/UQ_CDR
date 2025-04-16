import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

OPACITY = 0.01

def plot_uncertainty(df):
    df_vo = df[df['is_conflict']][['vx_vo', 'vy_vo']].copy()
    df_vo['vx'] = df['vx_vo']
    df_vo['vy'] = df['vy_vo']
    df_vo['Conf Reso'] = 'VO'

    df_mvp = df[df['is_conflict']][['vx_mvp', 'vy_mvp']].copy()
    df_mvp['vx'] = df['vx_mvp']
    df_mvp['vy'] = df['vy_mvp']
    df_mvp['Conf Reso'] = 'MVP'

    df_plot = pd.concat([df_vo, df_mvp], ignore_index=True)

    # Create jointplot
    f = sns.jointplot(
        data=df_plot,
        x="vy", y="vx", hue="Conf Reso",
        kind="scatter", alpha=OPACITY,
        palette=['tab:blue', 'tab:orange']
    )
    f.set_axis_labels("Cross-track resolution speed [kts]", "Along-track resolution speed [kts]")

    plt.show()