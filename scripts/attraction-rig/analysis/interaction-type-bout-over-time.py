
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches

df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed/interaction_type_bout.csv')
df1['condition'] = 'Group Housed'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/socially-isolated/interaction_type_bout.csv')
df2['condition'] = 'Socially Isolated'



palette = {
    'Socially Isolated': '#fca35d',  # yellow-orange
    'Group Housed': '#0d75b9'        # blue-green
}


df = pd.concat([df1, df2], ignore_index=True)

unified_types = [
    'head_head', 'tail_tail', 'body_body',
    'body_head', 'body_tail', 'head_tail'
]

bin_size = 600

# Filter + bin
df_filt = df[df['predominant_type'].isin(unified_types)].copy() ###### predominant_type or initial_type
df_filt['time_bin'] = (df_filt['start_frame'] // bin_size) * bin_size

# Get sorted bins (global, across all data)
bins = sorted(df_filt['time_bin'].unique())

# Create figure with 6 subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
axes = axes.flatten()

for i, tb in enumerate(bins[:6]):  # limit to first 6 bins
    bin_df = df_filt[df_filt['time_bin'] == tb]

    sns.barplot(
        data=bin_df,
        x='predominant_type', y='duration',
        order=unified_types,
        hue='condition',  ci='sd',   # mean with SD error bars
        ax=axes[i]
    )
    axes[i].set_title(f"{tb}–{tb+bin_size}")
    axes[i].set_xlabel("")
    axes[i].set_ylabel("Mean duration" if i % 3 == 0 else "")
    axes[i].tick_params(axis='x', rotation=30)
    axes[i].set_ylim(0,20)

# Remove any unused subplots
for j in range(len(bins), 6):
    fig.delaxes(axes[j])

fig.suptitle("Mean duration per node–node type across all files (per condition)", fontsize=14)
plt.tight_layout()

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/interactions/interaction_type_bout/over-time-duration-predominant-type.png', dpi=300, bbox_inches='tight')
plt.show()




########### frequency ###### predominant_type or initial_type

bin_size = 600

# filter + bin
df_filt = df[df['predominant_type'].isin(unified_types)].copy() 
df_filt['time_bin'] = (df_filt['start_frame'] // bin_size) * bin_size

# global, sorted bins and conditions
bins = sorted(df_filt['time_bin'].unique())
conds = sorted(df_filt['condition'].dropna().unique())

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
axes = axes.flatten()

for i, tb in enumerate(bins[:6]):  # first 6 bins
    ax = axes[i]
    bin_df = df_filt[df_filt['time_bin'] == tb]

    counts = (bin_df.groupby(['file', 'condition', 'predominant_type'])
                                .size()
                                .reset_index(name='count'))


    sns.barplot(
        data=counts,
        x='predominant_type', y='count',
        order=unified_types,
        hue='condition', 
        ci='sd', ax=ax
    )
    ax.set_title(f"{tb}–{tb+bin_size}")
    ax.set_xlabel("")
    ax.set_ylabel("Frequency" if i % 3 == 0 else "")
    ax.tick_params(axis='x', rotation=30)

# remove unused subplots
for j in range(len(bins), 6):
    fig.delaxes(axes[j])

fig.suptitle("Frequency of node–node types per 600‑frame bin across all files (per condition)", fontsize=14)
plt.tight_layout()

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/interactions/interaction_type_bout/over-time-frequency-PREDOM-type.png', dpi=300, bbox_inches='tight')
plt.show()



######### raw frame count 



df['time_bin'] = (df['start_frame'] // bin_size) * bin_size

cols = ['head_head','tail_tail','body_body','body_head','body_tail','head_tail']

# Make time_bin from start_frame
df['time_bin'] = (df['start_frame'] // bin_size) * bin_size

# 1) Sum frame counts per file × condition × bin
df_summed = (
    df.groupby(['condition', 'file', 'time_bin'], as_index=False)[cols]
      .sum()
)

# 2) Melt to long format
long_df = df_summed.melt(
    id_vars=['condition', 'file', 'time_bin'],
    value_vars=cols,
    var_name='type',
    value_name='count'
)

# 3) Plot: one figure, 6 subplots = first six bins
bins = sorted(long_df['time_bin'].unique())[:6]
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
axes = axes.flatten()

for i, tb in enumerate(bins):
    sub = long_df[long_df['time_bin'] == tb]

    sns.barplot(
        data=sub,
        x='type', y='count',
        order=cols,
        hue='condition',
        estimator=np.mean,     # mean across files
        errorbar='sd',         # SD across files (use ci='sd' if seaborn<0.13)
        palette=palette,
        ax=axes[i]
    )
    axes[i].set_title(f"{tb}–{tb+bin_size}")
    axes[i].set_xlabel("")
    axes[i].set_ylabel("Mean total frames" if i % 3 == 0 else "")
    axes[i].tick_params(axis='x', rotation=30)

# Hide unused axes if <6 bins
for j in range(len(bins), 6):
    fig.delaxes(axes[j])

fig.suptitle("Total frames per file → mean ± SD by condition (per 600-frame bin)", fontsize=14)
plt.tight_layout()

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/interactions/interaction_type_bout/over-time-frame-count.png', dpi=300, bbox_inches='tight')
plt.show()