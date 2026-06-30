
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



unified_types = [
    'head_head', 'tail_tail', 'body_body',
    'body_head', 'body_tail', 'head_tail'
]


#### BOUT DURATION OVER TIME

bin_size = 600
df['time_bin'] = (df['start_frame'] // bin_size + 1) * bin_size
length_bouts = df.groupby(['social_experience', 'file', 'time_bin'])['duration'].mean().reset_index(name='length_bout')

bins = sorted(length_bouts['time_bin'].unique())

# plt.figure(figsize=(3,2))
plt.figure(figsize=(6,4))
ax = sns.lineplot(data=length_bouts, x='time_bin', y='length_bout', hue='social_experience',  errorbar=('ci', 95))
plt.xlabel('Time Bin (S)', fontsize=12, fontweight='bold')
plt.ylabel('Mean Bout Duration (S)', fontsize=12, fontweight='bold')
# plt.title("Mean duration Over Time", fontsize=14)
plt.ylim(0,8)
plt.xlim(600,3600)
plt.xticks(np.arange(600, 3601, 600))
sns.despine()
ax.legend(frameon=False, title=None, fontsize=11, loc="upper right")
plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GHnSI/bout-duration-over-time.pdf', format='pdf', bbox_inches='tight')
plt.show()


#### BOUT DURATION FREQUENCY TIME
# plt.figure(figsize=(3,2))
plt.figure(figsize=(6,4))

bin_size = 600
df['time_bin'] = (df['start_frame'] // bin_size +1) * bin_size
freq_bouts = df.groupby(['social_experience', 'file', 'time_bin'])['bout_id'].size().reset_index(name='num_bouts')
sns.lineplot(data=freq_bouts, x='time_bin', y='num_bouts', hue='social_experience',   errorbar=('ci', 95))
plt.xlabel('Time Bin (S)', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')
plt.title('Total Interaction Bouts', fontsize=16, fontweight='bold')
plt.xlim(600,3600)
plt.xticks(np.arange(600, 3601, 600))
sns.despine()
plt.tight_layout(rect=[1, 1, 1, 1])
plt.ylim(0, None)
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GHnSI/bout-frequency-over-time.pdf', format='pdf', bbox_inches='tight')
plt.show()

