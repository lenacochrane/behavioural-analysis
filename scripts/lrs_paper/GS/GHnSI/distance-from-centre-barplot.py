
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import matplotlib as mpl
from scipy.stats import mannwhitneyu

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']

df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/grouped+isolated/distance_from_centre.csv')
df["social_experience"] = np.where(df["track"] <= 4, "SI", "GH")


track_means = (
    df.groupby(['file', 'social_experience', 'track'])['distance_from_centre']
    .mean()
    .reset_index()
)

file_means = (
    track_means
    .groupby(['file', 'social_experience'])['distance_from_centre']
    .mean()
    .reset_index()
)

gh = file_means.loc[file_means['social_experience'] == 'GH', 'distance_from_centre']
si = file_means.loc[file_means['social_experience'] == 'SI', 'distance_from_centre']

u_gh_si, p_gh_si = mannwhitneyu(gh, si, alternative='two-sided')
print(f"GH vs SI: U = {u_gh_si:.3f}, p = {p_gh_si:.4e}")



plt.figure(figsize=(6, 8))

ax = sns.barplot(
    data=file_means,
    x='social_experience',
    y='distance_from_centre',
    errorbar='sd',)


plt.ylabel('Distance From Centre (mm)', fontsize=14)
plt.subplots_adjust(wspace=0.3, hspace=0.4)

ax.legend(
    title=None,
    frameon=False,
    fontsize=11,
    loc='upper right'
)

# y = file_means['distance_from_centre'].max() * 1.10
# h = file_means['distance_from_centre'].max() * 0.03

# # SI (1) vs Pseudo (2)
# ax.plot([1, 2], [y, y], lw=1.5, c='k')
# ax.text(1.5, y + h, '*', ha='center', va='bottom', fontsize=16)




sns.despine()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GHnSI/barplot-distance-from-centre.pdf', format='pdf', bbox_inches='tight')

plt.close()