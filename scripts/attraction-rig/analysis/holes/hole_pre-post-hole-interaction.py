
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/holes/N10-1-HOLE/GROUP-HOUSED/pre_post_interactions.csv')
df1['condition'] = 'GH'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/holes/N10-1-HOLE/SOCIAL-ISOLATION/pre_post_interactions.csv')
df2['condition'] = 'SI'

df = pd.concat([df1, df2], ignore_index=True)


############### AVERAGE DURATION

plt.figure(figsize=(8, 6))

sns.barplot(
    data=df,
    x='condition',
    y='duration',
    hue='status_hole',
    edgecolor='black',
    linewidth=2,
    ci='sd',
    alpha=0.6, hue_order=['pre', 'post'])

plt.title('Interaction Duration', fontsize=16, fontweight='bold')
plt.ylim(0, None)
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes/interactions/interaction-duration-overall.png', dpi=300, bbox_inches='tight')
plt.show()



fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# Plot for GH (df1)
sns.barplot(
    data=df1,
    x='initial_type',
    y='duration',
    hue='status_hole',
    edgecolor='black',
    linewidth=2,
    ci='sd',
    alpha=0.6,
    ax=axes[0]
)
axes[0].set_title('Group-Housed (GH)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Initial Interaction Type', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Duration (s)', fontsize=12, fontweight='bold')
axes[0].tick_params(axis='x', labelrotation=45)
axes[0].tick_params(axis='both', labelsize=10)
axes[0].legend(title='Hole Status Pair')
axes[0].set_xlim(0, None)

# Plot for SI (df2)
sns.barplot(
    data=df2,
    x='initial_type',
    y='duration',
    hue='status_hole',
    edgecolor='black',
    linewidth=2,
    ci='sd',
    alpha=0.6,
    ax=axes[1]
)
axes[1].set_title('Socially Isolated (SI)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Initial Interaction Type', fontsize=12, fontweight='bold')
axes[1].set_ylabel('')  # Hide y-label on second plot
axes[1].tick_params(axis='x', labelrotation=45)
axes[1].tick_params(axis='both', labelsize=10)
axes[1].legend(title='Hole Status Pair')
axes[1].set_ylim(0, None)

plt.suptitle('Interaction Duration', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])


plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes/interactions/interaction-duration-initial_type.png', dpi=300, bbox_inches='tight')

plt.show()



fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# Plot for GH (df1)
sns.barplot(
    data=df1,
    x='initial_type',
    y='duration',
    hue='status_hole',
    edgecolor='black',
    linewidth=2,
    ci='sd',
    alpha=0.6,
    ax=axes[0]
)
axes[0].set_title('Group-Housed (GH)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Initial Interaction Type', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Duration (s)', fontsize=12, fontweight='bold')
axes[0].tick_params(axis='x', labelrotation=45)
axes[0].tick_params(axis='both', labelsize=10)
axes[0].legend(title='Hole Status Pair')
axes[0].set_xlim(0, None)

# Plot for SI (df2)
sns.barplot(
    data=df2,
    x='predominant_type',
    y='duration',
    hue='status_hole',
    edgecolor='black',
    linewidth=2,
    ci='sd',
    alpha=0.6,
    ax=axes[1]
)
axes[1].set_title('Socially Isolated (SI)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Initial Interaction Type', fontsize=12, fontweight='bold')
axes[1].set_ylabel('')  # Hide y-label on second plot
axes[1].tick_params(axis='x', labelrotation=45)
axes[1].tick_params(axis='both', labelsize=10)
axes[1].legend(title='Hole Status Pair')
axes[1].set_ylim(0, None)

plt.suptitle('Interaction Duration', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])


plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes/interactions/interaction-duration-predominant_type.png', dpi=300, bbox_inches='tight')

plt.show()

##################################### TOTAL FREQUENCY BEFORE AND AFTER HOLE EXPOSURE

# 1) Count bouts per file × condition × hole status
freq = (
    df
    .groupby(['file', 'condition', 'status_hole'])
    .size()
    .reset_index(name='frequency')
)

# 2) Plot (same style as duration)
plt.figure(figsize=(8, 6))

sns.barplot(
    data=freq,
    x='condition',
    y='frequency',
    hue='status_hole',
    edgecolor='black',
    linewidth=2,
    ci='sd',
    alpha=0.6,
    hue_order=['pre', 'post'],  order=['GH', 'SI']  
)

plt.title('Interaction Frequency', fontsize=16, fontweight='bold')
plt.ylim(0, None)
plt.ylabel('Number of Interaction Bouts')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes/interactions/interaction-frequency-overall.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes/interactions/interaction-frequency-overall.pdf', format='pdf', bbox_inches='tight')
plt.show()




##################################### INITIAL TYPES



grouped_gh = df1.groupby(['file','initial_type', 'status_hole']).size().reset_index(name='count') #size for rows, count for values
grouped_sh = df2.groupby(['file','initial_type', 'status_hole']).size().reset_index(name='count') #size for rows, count for values


fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# GH plot
sns.barplot(data=grouped_gh, x='initial_type', y='count', hue='status_hole',
            edgecolor='black', linewidth=2, ci='sd', alpha=0.6, ax=axes[0], hue_order=['pre', 'post'])
axes[0].set_title('GH - Interaction Frequency', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Interaction Type', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right', fontweight='bold')

# SI plot
sns.barplot(data=grouped_sh, x='initial_type', y='count', hue='status_hole',
            edgecolor='black', linewidth=2, ci='sd', alpha=0.6, ax=axes[1], hue_order=['pre', 'post'])
axes[1].set_title('SI - Interaction Frequency', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Interaction Type', fontsize=12, fontweight='bold')
axes[1].set_ylabel('')  # Hide duplicate Y-axis label
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right', fontweight='bold')

# Final layout and save
plt.suptitle('Interaction Type Frequency Per Condition', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.92])

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes/interactions/interaction-frequency-initial_type.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes/interactions/interaction-frequency-initial_type.pdf', format='pdf', bbox_inches='tight')
plt.show()




########### PREDOMINANT TYPE




grouped_gh = df1.groupby(['file','predominant_type', 'status_hole']).size().reset_index(name='count') #size for rows, count for values
grouped_sh = df2.groupby(['file','predominant_type', 'status_hole']).size().reset_index(name='count') #size for rows, count for values

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# GH plot
sns.barplot(data=grouped_gh, x='predominant_type', y='count', hue='status_hole',
            edgecolor='black', linewidth=2, ci='sd', alpha=0.6, ax=axes[0])
axes[0].set_title('GH - Interaction Frequency', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Interaction Type', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right', fontweight='bold')

# SI plot
sns.barplot(data=grouped_sh, x='predominant_type', y='count', hue='status_hole',
            edgecolor='black', linewidth=2, ci='sd', alpha=0.6, ax=axes[1])
axes[1].set_title('SI - Interaction Frequency', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Interaction Type', fontsize=12, fontweight='bold')
axes[1].set_ylabel('')  # Hide duplicate Y-axis label
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right', fontweight='bold')

# Final layout and save
plt.suptitle('Interaction Type Frequency Per Condition', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.92])

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes/interactions/interaction-frequency-predominant_type.png', dpi=300, bbox_inches='tight')

plt.show()




# ############## INTERACTION_TYPE

interaction_cols = ['body_body', 'body_head', 'body_tail', 'head_head', 'head_tail', 'tail_tail']

# Sum interaction counts per hole_status_pair and interaction type
melted_si = df2.groupby(['status_hole', 'file'])[interaction_cols].sum().reset_index().melt(
    id_vars=['status_hole', 'file'], var_name='interaction_type', value_name='count'
)

melted_gh = df1.groupby(['status_hole', 'file'])[interaction_cols].sum().reset_index().melt(
    id_vars=['status_hole', 'file'], var_name='interaction_type', value_name='count'
)


# Set up side-by-side subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# SI subplot
sns.barplot(data=melted_si, x='interaction_type', y='count', hue='status_hole',
            edgecolor='black', linewidth=2, alpha=0.7, ax=axes[1], hue_order=['pre', 'post'])
axes[1].set_title('SI - Interaction Frequencies', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Interaction Type', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Total Frequency', fontsize=12, fontweight='bold')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right', fontweight='bold')

# GH subplot
sns.barplot(data=melted_gh, x='interaction_type', y='count', hue='status_hole',
            edgecolor='black', linewidth=2, alpha=0.7, ax=axes[0], hue_order=['pre', 'post'])
axes[0].set_title('GH - Interaction Frequencies', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Interaction Type', fontsize=12, fontweight='bold')
axes[0].set_ylabel('')  # remove duplicate ylabel
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right', fontweight='bold')

# Finalize layout
plt.suptitle('Node-Node Interaction Frequencies by Hole Status Pair', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.92])

# Save and show
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes/interactions/total_frames-interaction_type.png', dpi=300, bbox_inches='tight')
plt.show()





