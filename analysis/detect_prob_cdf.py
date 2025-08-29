# %%
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams.update({'font.size': 12})

csv_path_list = glob.glob('../results/UQ/csv/CD/p_oi/CD_p_oi_250527_*.csv')
color = ['gray', 'darkgray']

counter = 0
dcpa = [45, 15]

for file in csv_path_list:
    df = pd.read_csv(file)
    detect_prob_list = []
    tin_list = []

    for tlookahead_ms in range(12000, 18000, 100):
        tlookahead = tlookahead_ms / 1000
        temporal = df['tin'] < tlookahead
        spatial = df['dcpa'] < 50

        detect_prob_list.append(len(df[temporal & spatial]) / len(df) * 100)
        tin_list.append(tlookahead)

    tin_list = tin_list[::-1]
    detect_prob_list = detect_prob_list[::-1]

    plt.plot(tin_list, detect_prob_list, color=color[counter], label = fr"Initial $\|\mathbf{{d}}_{{\mathrm{{CPA}}}}\|$ = {dcpa[counter]:.1f} m")

    print(len(df[df['dcpa'] < 50]) / len(df) * 100)

    # Add annotation near the end of each curve
    # print(tin_list[0])
    if(counter == 0):
        plt.text(
            16.5,                        # x-position slightly past the end
            detect_prob_list[1]-20,                      # y-position at asymptotic value
            r'$\mathbb{P}(\|\mathbf{d}_{\mathrm{CPA,p}}\| < \mathrm{R_{\mathrm{PZ}}})$', 
            va='center',
            fontsize=10,
            color='k'
        )

    counter += 1

    # plt.axhline(max(detect_prob_list))

# Custom xtick labels centered on 15
plt.axvline(15, linestyle='--', color='k', linewidth = '0.6', label = '$t_{in}=t_{lookahead}$')

xticks = plt.xticks()[0]
xtick_labels = []
for val in xticks:
    string_tlookahead = '$t_{lookahead}$'
    if abs(val - 15) < 0.01:
        xtick_labels.append('$t_{lookahead}$')
    elif val > 15:
        diff = -int(round(15 - val))
        xtick_labels.append(f'{string_tlookahead}-{diff}')
    elif val - 15:
        diff = -int(round(val - 15))
        xtick_labels.append(f'{string_tlookahead}+{diff}')
    else:
        xtick_labels.append(str(round(val, 1)))  # fallback

plt.xticks(ticks=xticks, labels=xtick_labels, rotation = 45)
plt.xlabel('Time to first intrusion [s]')
plt.ylabel('Detection Probability [%]')
plt.xlim([11.75, 18.25])

# plt.gca().invert_xaxis()

plt.legend()

fig_output_dir = f"../results/UQ/figures/CD/detect_prob/"
# Save the boxplot
boxplot_name = "CD_detect_prob_dcpa_tin_invert.png"
boxplot_path = os.path.join(fig_output_dir, boxplot_name)
plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')

# plt.savefig()
plt.show()