
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches

df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/interaction_type_bout.csv')
df1['condition'] = 'GH'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/interaction_type_bout.csv')
df2['condition'] = 'SI'


palette = {
    'SI': '#fca35d',  # yellow-orange
    'GH': '#0d75b9'        # blue-green
}

palette_order = ['GH', 'SI']

df = pd.concat([df1, df2], ignore_index=True)

unified_types = [
    'head_head', 'tail_tail', 'body_body',
    'body_head', 'body_tail', 'head_tail'
]

#### BOUT DURATION OVER TIME

bin_size = 600
df['time_bin'] = (df['start_frame'] // bin_size) * bin_size
length_bouts = df.groupby(['condition', 'file', 'time_bin'])['duration'].mean().reset_index(name='length_bout')

bins = sorted(length_bouts['time_bin'].unique())

plt.figure(figsize=(6,4))
sns.barplot(data=length_bouts, x='time_bin', y='length_bout', hue='condition',  edgecolor='black', linewidth=2, errorbar='sd', palette=palette)
plt.xlabel('Time Bin (S)', fontsize=12, fontweight='bold')
plt.ylabel('Mean Bout Duration (S)', fontsize=12, fontweight='bold')
plt.title("Mean duration Over Time", fontsize=14)
plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/proximal_contacts/bout-subplot-over-duration.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/proximal_contacts/bout-subplot-over-duration.pdf', format='pdf', bbox_inches='tight')   
plt.close()


#### BOUT DURATION FREQUENCY TIME
plt.figure(figsize=(6,4))


bin_size = 600
df['time_bin'] = (df['start_frame'] // bin_size) * bin_size
freq_bouts = df.groupby(['condition', 'file', 'time_bin'])['bout_id'].size().reset_index(name='num_bouts')
sns.barplot(data=freq_bouts, x='time_bin', y='num_bouts', hue='condition', edgecolor='black', linewidth=2, errorbar='sd', palette=palette, alpha=0.8)
plt.xlabel('Time Bin (S)', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')
plt.title('Total Interaction Bouts', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[1, 1, 1, 1])
plt.ylim(0, None)
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/proximal_contacts/bout-subplot-over-frequency.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/proximal_contacts/bout-subplot-over-frequency.pdf', format='pdf', bbox_inches='tight')
plt.close()




#### INITIAL TYPE PER BOUT OVER TIME - DURATION

bin_size = 600

df_filt = df[df['initial_type'].isin(unified_types)].copy() ###### predominant_type or initial_type
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
        x='initial_type', y='duration',
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

fig.suptitle("Mean duration Over Time", fontsize=14)
plt.tight_layout()

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/proximal_contacts/bout-subplot-over-initial-type-duration.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/proximal_contacts/bout-subplot-over-initial-type-duration.pdf', format='pdf', bbox_inches='tight')
plt.close()



#### INITIAL TYPE PER BOUT OVER TIME - FREQUENCY

bin_size = 600

# filter + bin
df_filt = df[df['initial_type'].isin(unified_types)].copy() 
df_filt['time_bin'] = (df_filt['start_frame'] // bin_size) * bin_size

# global, sorted bins and conditions
bins = sorted(df_filt['time_bin'].unique())
conds = sorted(df_filt['condition'].dropna().unique())

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
axes = axes.flatten()

for i, tb in enumerate(bins[:6]):  # first 6 bins
    ax = axes[i]
    bin_df = df_filt[df_filt['time_bin'] == tb]

    counts = (bin_df.groupby(['file', 'condition', 'initial_type'])
                                .size()
                                .reset_index(name='count'))


    sns.barplot(
        data=counts,
        x='initial_type', y='count',
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

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/proximal_contacts/bout-subplot-over-initial-type-frequency.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/proximal_contacts/bout-subplot-over-initial-type-frequency.pdf', format='pdf', bbox_inches='tight')  
plt.close()








