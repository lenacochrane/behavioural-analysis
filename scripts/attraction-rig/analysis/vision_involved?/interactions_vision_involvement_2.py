
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import os

df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/cropped_interactions.csv')

df['track1_approach_angle_edited'] = 180 - df['track1_approach_angle']
df['track2_approach_angle_edited'] = 180 - df['track2_approach_angle']

## difference (large differences surely reflect one approaching and the other not relaising idk come back to it)
df['approach_angle_diff'] = (df['track1_approach_angle_edited'] - df['track2_approach_angle_edited']).abs()

bins = [0, 30, 60, 90, 120, 150, 180]

df['track1_approach_bin'] = pd.cut(
    df['track1_approach_angle_edited'],
    bins=bins,
    include_lowest=True,
    right=False)

df['track2_approach_bin'] = pd.cut(
    df['track2_approach_angle_edited'],
    bins=bins,
    include_lowest=True,
    right=False)

print(df.columns)


# track 1
t1 = df[['interaction_id', 'Frame','Normalized Frame', 'track1_approach_bin', 'track1_approach_angle_edited', 'track1_speed']].copy()
t1['track'] = 1
t1 = t1.rename(columns={
    'track1_approach_angle_edited': 'approach_angle',
    'track1_approach_bin': 'approach_bin',
    'track1_speed': 'speed'
})

# track 2
t2 = df[['interaction_id', 'Frame','Normalized Frame', 'track2_approach_bin', 'track2_approach_angle_edited', 'track2_speed']].copy()
t2['track'] = 2
t2 = t2.rename(columns={
    'track2_approach_angle_edited': 'approach_angle',
    'track2_approach_bin': 'approach_bin',
    'track2_speed': 'speed'
})

# combine
long = pd.concat([t1, t2], ignore_index=True)

long = long.sort_values(by=['interaction_id', 'Normalized Frame'])
print(long.columns)
print(long.head())

# ---- NOW filter by Normalized Frame ----
pre = long[(long['Normalized Frame'] >= -10) & (long['Normalized Frame'] <= -2)]


dominant = (
    pre.dropna(subset=['approach_bin'])
      .groupby(['interaction_id', 'track', 'approach_bin'])
      .size()
      .reset_index(name='count')
)

# pick the most frequent bin per interaction_id + track
dominant = (
    dominant.sort_values(['interaction_id', 'track', 'count'], ascending=[True, True, False])
            .groupby(['interaction_id', 'track'])
            .head(1)
)

# keep a NEW column name (don’t rename approach_bin)
dominant['dominant_approach_bin'] = dominant['approach_bin']

# keep only keys + the new label
dominant = dominant[['interaction_id', 'track', 'dominant_approach_bin']]

approach_df = long.merge(dominant, on=['interaction_id', 'track'], how='left')

approach_df = approach_df.sort_values(by=['interaction_id', 'track', 'Normalized Frame'])

print(approach_df)



# ensure consistent ordering of bins
bin_order = approach_df['dominant_approach_bin'].cat.categories

fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
axes = axes.flatten()


for ax, b in zip(axes, bin_order):
    sub = approach_df[approach_df['dominant_approach_bin'] == b]

    if sub.empty:
        ax.axis('off')
        continue

    sns.lineplot(
        data=sub,
        x='Normalized Frame',
        y='speed',
        errorbar=('ci', 95),
        ax=ax,
        legend=False
    )

    ax.set_title(f"{b}")
    ax.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel("Normalized Frame")
    ax.set_ylabel("Speed")

# one legend for the whole figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title='Track', loc='upper right', frameon=False)

fig.suptitle("Speed Over Time: Grouped by Dominant Pre-Approach Bin", fontsize=16, fontweight='bold')
fig.tight_layout(rect=[0, 0, 0.95, 0.93])

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/vision_involved/interactions_approach_angle', dpi=300, bbox_inches='tight')
plt.savefig(
    '/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/vision_involved/interactions_approach_angle.pdf',
    format='pdf',
    bbox_inches='tight'
)
plt.show()

















