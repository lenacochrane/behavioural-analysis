
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---- Adobe / Illustrator friendly PDFs ----
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']




df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/grouped+isolated/time_average_msd.csv')
# df["social_experience"] = np.where(df["track_id"] <= 4, "SI", "GH")



# for cond in df['social_experience'].unique():
#     subset = df[df['social_experience'] == cond]

    # plt.figure(figsize=(1.5, 1.5))

    # plt.figure(figsize=(4, 4))
    # ax = sns.lineplot(data=subset, x='tau', y='msd', errorbar=('ci', 95), label=cond)

    # plt.xlabel('Tau', fontsize=12,fontweight='bold')
    # plt.ylabel('MSD', fontsize=12,fontweight='bold')

    # sns.despine()
    # ax.legend(frameon=False, title=None, fontsize=11, loc="upper left")
    # plt.tight_layout(rect=[0, 0, 1, 0.95])

    # plt.savefig(f'/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GHnSI/time_average_msd_{cond}.pdf', format='pdf', bbox_inches='tight')
    # plt.close()




plt.figure(figsize=(4, 4))
sns.lineplot(data=df, x='tau', y='msd', errorbar=('ci', 95))

plt.xlabel('Tau', fontsize=12,fontweight='bold')
plt.ylabel('MSD', fontsize=12,fontweight='bold')

sns.despine()
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig(f'/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GHnSI/time_average_msd.pdf', format='pdf', bbox_inches='tight')
plt.close()