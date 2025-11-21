# %%

import matplotlib.pyplot as plt
import numpy as np

# Plot settings for both figures
def generate_plot(mean1, mean2, title1, title2, label_left, label_right, filename, xlabel, xtick_labels):
    x = np.linspace(0, 100, 1000)
    rpz, std = 50, 5
    y1 = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean1) / std) ** 2)
    y2 = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean2) / std) ** 2)
    colors = ['dimgray', 'lightgray']
    xticks = [rpz - 10, rpz, rpz + 10]

    fig, axs = plt.subplots(2, 1, figsize=(6, 8), sharey=True)
    counter = 0

    for ax, y, mean, title, labels in zip(axs, [y1, y2], [mean1, mean2], [title1, title2], label_left):
        ax.plot(x, y, color='black')

        if(counter == 0):
            ax.fill_between(x, 0, y, where=(x <= rpz), color=colors[1], alpha=0.4, label=labels[1])
            ax.fill_between(x, 0, y, where=(x > rpz), color=colors[0], alpha=0.4, label=labels[0])
        else:
            ax.fill_between(x, 0, y, where=(x > rpz), color=colors[1], alpha=0.4, label=labels[0])
            ax.fill_between(x, 0, y, where=(x <= rpz), color=colors[0], alpha=0.4, label=labels[1])
            ax.invert_xaxis()

        ax.axvline(rpz, color='black', linestyle='--', linewidth=1)
        ax.axvline(mean, color='black', linewidth=1)
        ax.set(xlim=(30, 70), ylim=(0, max(max(y1), max(y2)) * 1.1), title=title, yticks=[], xticks=xticks)
        ax.set_xticklabels(xtick_labels)
        ax.set_ylabel('Probability Density')
        ax.legend()
        ax.grid(False)
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(False)
        ax.annotate('', xy=(1.05, 0), xytext=(-0.02, 0), xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
        ax.annotate('', xy=(0, 1.05), xytext=(0, -0.02), xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
        
        if('tin' in filename):
            ax.annotate(xlabel, xy=(1.04, 0.08), xycoords='axes fraction', ha='right', va='top')
        else:
            ax.annotate(xlabel, xy=(1.08, 0.08), xycoords='axes fraction', ha='right', va='top')

        counter += 1

    plt.tight_layout()
    path = f"{filename}"
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    return path

# First figure: DCPA
dcpa_labels = [['False Negative', 'True Positive'], ['True Negative', 'False Positive']]
dcpa_xtick_labels = [r'$\mathrm{R_{PZ}} - \delta d$', r'$\mathrm{R_{PZ}}$', r'$\mathrm{R_{PZ}} + \delta d$']
dcpa_path = generate_plot(45, 55, "Conflict Case", "No Conflict Case", dcpa_labels, dcpa_labels,
                          "dcpa_plot.png", r'$\left\| \mathbf{d}_{\mathrm{CPA}} \right\|$', dcpa_xtick_labels)

# Second figure: T_in
tin_labels = [['False Positive', 'True Negative'], ['True Positive', 'False Negative']]
tin_xtick_labels = [r'$t_\mathrm{lookahead} + \delta t$', r'$t_\mathrm{lookahead}$', r'$t_\mathrm{lookahead} - \delta t$']
tin_path = generate_plot(45, 55, "No Conflict Case", "Conflict Case", tin_labels, tin_labels,
                         "tin_plot.png", r'$t_{\mathrm{in}}$', tin_xtick_labels)

dcpa_path, tin_path
