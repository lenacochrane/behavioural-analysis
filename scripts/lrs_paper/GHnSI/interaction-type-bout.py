
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']



df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/grouped+isolated/interaction_type_bout.csv')


# classify each track
df["track_1_exp"] = np.where(df["track_1"] <= 4, "SI", "GH")
df["track_2_exp"] = np.where(df["track_2"] <= 4, "SI", "GH")

# make pair label
df["social_experience"] = df.apply(
    lambda row: "-".join(sorted([row["track_1_exp"], row["track_2_exp"]])),
    axis=1
)



###### RAW NUMBER OF BOUTS
plt.figure(figsize=(2,4))
grouped = (
    df.groupby(['social_experience', 'file'])['bout_id']
      .size()
      .reset_index(name='num_bouts')
)

ax = sns.barplot(data=grouped, x='social_experience', y='num_bouts', hue='social_experience', edgecolor='black', linewidth=2, errorbar='sd')

plt.xlabel('', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
sns.despine()
ax.legend(frameon=False, title=None, fontsize=11, loc="upper right")
# plt.title('Total Interaction Bouts', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[1, 1, 1, 1])
plt.ylim(0, None)
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GHnSI/number_bouts.pdf', format='pdf', bbox_inches='tight')
plt.show()


###### BOUT LENGTHS

length_bouts = df.groupby(['social_experience', 'file'])['duration'].mean().reset_index(name='length_bout')

plt.figure(figsize=(2,4))
ax = sns.barplot(data=length_bouts, x='social_experience', y='length_bout',  edgecolor='black', linewidth=2, errorbar='sd')
plt.xlabel('', fontsize=12, fontweight='bold')
plt.ylabel('Average Bout Length (S)', fontsize=12, fontweight='bold')
# plt.title('Average Contact Bout Length', fontsize=16, fontweight='bold')
sns.despine()
ax.legend(frameon=False, title=None, fontsize=11, loc="upper right")
plt.tight_layout(rect=[1, 1, 1, 1])
plt.ylim(0, None)
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GHnSI/average_bout_length.pdf', format='pdf', bbox_inches='tight')
plt.show()










