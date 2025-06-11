import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyarrow.feather as feather
import matplotlib.patches as mpatches

df_moseq = pd.read_csv('/Users/cochral/Desktop/MOSEQ/KEYPOINT/2025_05_21-13_29_37/moseq_df.csv')
df_stat = pd.read_csv('/Users/cochral/Desktop/MOSEQ/KEYPOINT/2025_05_21-13_29_37/stats_df.csv')

################################ SYLLABLE STATS 

# velocity_px_s_mean 
# duration
# angular_velocity_mean
# heading_mean

# sns.barplot(data=df_stat, x='syllable', y='heading_mean', hue='group', ci='sd')

# plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/moseq/heading.png', dpi=300, bbox_inches='tight')

# plt.show()


################################ TRACK ETHOGRAM

# print(df_moseq.columns)


# # Filter only the animals you want (e.g., 'N10-GH')
# df_filtered = df_moseq[df_moseq['name'].str.startswith('N10-GH')].copy()

# tracks = df_filtered['name'].unique()

# # Make a syllable-to-color map using viridis
# syllables = sorted(df_filtered['syllable'].unique())
# palette = sns.color_palette('viridis', n_colors=len(syllables))
# syl2color = {s: palette[i] for i, s in enumerate(syllables)}

# # Start plot
# fig, ax = plt.subplots(figsize=(12, len(tracks) * 0.4))

# for i, name in enumerate(tracks):
#     sub = df_filtered[df_filtered['name'] == name].sort_values('frame_index')

#     for _, row in sub.iterrows():
#         x = row['frame_index']
#         y = i  # vertical stack by animal
#         color = syl2color[row['syllable']]
#         ax.add_patch(plt.Rectangle((x, y), 1, 1, color=color, linewidth=0))

# # Formatting
# ax.set_yticks(np.arange(len(tracks)) + 0.5)
# ax.set_yticklabels(tracks)
# ax.set_xlabel("Time")
# ax.set_ylabel("Track")
# ax.set_title("Syllable Ethogram N10-GH")
# ax.set_xlim(df_filtered['frame_index'].min(), df_filtered['frame_index'].max())
# ax.set_ylim(0, len(tracks))

# handles = [mpatches.Patch(color=c, label=s) for s, c in syl2color.items()]
# plt.legend(handles=handles, title="Syllables", bbox_to_anchor=(1.01, 1), loc='upper left')

# plt.tight_layout()
# plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/moseq/ethorgram-gh-n10.png', dpi=300, bbox_inches='tight')
# plt.show()


################################ TRACK ETHOGRAM

# Filter only the animals you want (e.g., 'N10-GH')
df_filtered = df_moseq[df_moseq['name'] == 'N1-GH_2025-02-24_15-16-50_td7'].copy()

tracks = df_filtered['name'].unique()

# Make a syllable-to-color map using viridis
syllables = sorted(df_filtered['syllable'].unique())
palette = sns.color_palette('viridis', n_colors=len(syllables))
syl2color = {s: palette[i] for i, s in enumerate(syllables)}

# Start plot
fig, ax = plt.subplots(figsize=(12, len(tracks) * 0.4))

for i, name in enumerate(tracks):
    sub = df_filtered[df_filtered['name'] == name].sort_values('frame_index')

    for _, row in sub.iterrows():
        x = row['frame_index']
        y = i  # vertical stack by animal
        color = syl2color[row['syllable']]
        ax.add_patch(plt.Rectangle((x, y), 1, 1, color=color, linewidth=0))

# Formatting
ax.set_yticks(np.arange(len(tracks)) + 0.5)
ax.set_yticklabels(tracks)
ax.set_xlabel("Time")
ax.set_ylabel("Track")
ax.set_title("Syllable Ethogram N10-GH")
ax.set_xlim(df_filtered['frame_index'].min(), df_filtered['frame_index'].max())
ax.set_ylim(0, len(tracks))

handles = [mpatches.Patch(color=c, label=s) for s, c in syl2color.items()]
plt.legend(handles=handles, title="Syllables", bbox_to_anchor=(1.01, 1), loc='upper left')

plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/moseq/ethorgram-example.png', dpi=300, bbox_inches='tight')
plt.show()
