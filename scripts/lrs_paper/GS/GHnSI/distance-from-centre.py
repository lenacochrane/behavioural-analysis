
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys  
import matplotlib as mpl


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']

df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/grouped+isolated/distance_from_centre.csv')
df["social_experience"] = np.where(df["track"] <= 4, "SI", "GH")


bins = np.linspace(0, 50, 25)  # 0 to 2.5 in 0.1 increments
df['distance_bin'] = pd.cut(df['distance_from_centre'], bins, include_lowest=True)
df['bin_center'] = df['distance_bin'].apply(lambda x: x.mid)


counts = (
    df.groupby(['file', 'social_experience', 'bin_center'])
    .size()
    .groupby(['file', 'social_experience'], group_keys=False)
    .apply(lambda x: x / x.sum())
    .reset_index(name='density')
)


# plt.figure(figsize=(1.5,1.5), dpi=600)
plt.figure(figsize=(5,5), dpi=600)

ax = sns.lineplot(data=counts, x='bin_center', y='density', hue='social_experience', errorbar=('ci', 95)) # legend=False)


plt.xlabel('Distance From Centre (mm) ', fontsize=12)
plt.ylabel('Probability', fontsize=12)

sns.despine()
plt.ylim(0, 0.4)
plt.xlim(0,50)
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GHnSI/distance-from-centre.pdf', format='pdf', bbox_inches='tight')
plt.close()
