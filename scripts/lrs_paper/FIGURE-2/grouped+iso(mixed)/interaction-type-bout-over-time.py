
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
plt.ylim(0,12)
plt.xlim(600,3600)
plt.xticks(np.arange(600, 3601, 600))
sns.despine()
ax.legend(frameon=False, title=None, fontsize=11, loc="upper right")
plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/FIGURE-2/group+iso(mixed)/bout-duration-over-time.pdf', format='pdf', bbox_inches='tight')
plt.show()


#### BOUT FREQUENCY OVER TIME
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
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/FIGURE-2/group+iso(mixed)/bout-frequency-over-time.pdf', format='pdf', bbox_inches='tight')
plt.show()



#### BOUT FREQUENCY OVER TIME - NORMALISED BY PAIR AVAILABILITY
# same logic as proportion-contact-time-GS.ipynb:
# 10 GH-GH, 25 GH-SI, 10 SI-SI possible pairs (45 total)
# normalised = (share of bouts in bin that are this type) / (share of pairs that are this type)
# 1 = random mixing; >1 = preferred; <1 = avoided
N_PAIRS = {'GH-GH': 10, 'GH-SI': 25, 'SI-SI': 10}
TOTAL_PAIRS = sum(N_PAIRS.values())

# full grid: every file x type x bin, so bins/types with no bouts count as 0
all_bins = np.arange(600, 3601, bin_size)
grid = pd.MultiIndex.from_product(
    [df['file'].unique(), list(N_PAIRS), all_bins],
    names=['file', 'social_experience', 'time_bin']
).to_frame(index=False)

norm_bouts = grid.merge(freq_bouts, on=['file', 'social_experience', 'time_bin'], how='left')
norm_bouts['num_bouts'] = norm_bouts['num_bouts'].fillna(0)

tot_bouts = norm_bouts.groupby(['file', 'time_bin'])['num_bouts'].transform('sum')
norm_bouts['share'] = norm_bouts['num_bouts'] / tot_bouts
norm_bouts['expected'] = norm_bouts['social_experience'].map(N_PAIRS) / TOTAL_PAIRS
norm_bouts['normalised'] = norm_bouts['share'] / norm_bouts['expected']
# a video with no bouts at all in a bin has no share -> leave it out of the mean
norm_bouts = norm_bouts.dropna(subset=['normalised'])

plt.figure(figsize=(6,4))
ax = sns.lineplot(data=norm_bouts, x='time_bin', y='normalised', hue='social_experience',
                  hue_order=list(N_PAIRS), errorbar=('ci', 95))
plt.axhline(1, color='#2a2a2a', linewidth=1, linestyle='--')
plt.xlabel('Time Bin (S)', fontsize=12, fontweight='bold')
plt.ylabel('Bouts relative to chance\n(observed / expected)', fontsize=12, fontweight='bold')
plt.title('Interaction Bouts (normalised)', fontsize=16, fontweight='bold')
plt.xlim(600,3600)
plt.xticks(np.arange(600, 3601, 600))
plt.ylim(0, None)
sns.despine()
ax.legend(frameon=False, title=None, fontsize=11, loc="upper right")
plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/FIGURE-2/group+iso(mixed)/bout-frequency-over-time-normalised.pdf', format='pdf', bbox_inches='tight')
plt.show()
