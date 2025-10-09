
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



plt.figure(figsize=(8,8))

palette = {
    'Socially Isolated': '#fca35d',  # yellow-orange
    'Group Housed': '#0d75b9'        # blue-green
}


df = pd.concat([df1, df2], ignore_index=True)

# Sum across all frame bins per file + interaction type
grouped = (
    df.groupby(['file', 'condition', 'initial_type'])
    .size()
    .reset_index(name='count')
)

sns.barplot(data=grouped, x='initial_type', y='count', hue='condition', edgecolor='black', linewidth=2, errorbar='sd', palette=palette, alpha=0.8)

plt.xlabel('Initial Interaction Type', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')


# Add an overall title to the entire figure
plt.title('Initial Interaction Type', fontsize=16, fontweight='bold')

# Adjust layout to prevent overlap, considering the overall title
# plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.tight_layout(rect=[1, 1, 1, 1])

plt.ylim(0, None)

plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/interactions/interaction_type_bout/initial_n10.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()


plt.figure(figsize=(8,8))
# Sum across all frame bins per file + interaction type
grouped2 = (
    df.groupby(['file', 'condition', 'predominant_type'])
    .size()
    .reset_index(name='count')
)

sns.barplot(data=grouped2, x='predominant_type', y='count', hue='condition', edgecolor='black', linewidth=2, errorbar='sd', palette=palette, alpha=0.8)

plt.xlabel('Predominant Interaction Type per Bout', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')


# Add an overall title to the entire figure
plt.title('Predominant Interaction Type', fontsize=16, fontweight='bold')

# Adjust layout to prevent overlap, considering the overall title
# plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.tight_layout(rect=[1, 1, 1, 1])

plt.ylim(0, None)

plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/interactions/interaction_type_bout/predominant_n10.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()


plt.figure(figsize=(8,8))

sns.barplot(data=df, x='initial_type', y='duration', hue='condition', edgecolor='black', linewidth=2, errorbar='sd', palette=palette, alpha=0.8)

plt.xlabel('Interaction Type Duration', fontsize=12, fontweight='bold')
plt.ylabel('duration', fontsize=12, fontweight='bold')


# Add an overall title to the entire figure
plt.title('Interaction Type Duration', fontsize=16, fontweight='bold')

# Adjust layout to prevent overlap, considering the overall title
# plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.tight_layout(rect=[1, 1, 1, 1])

plt.ylim(0, None)

plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/interactions/interaction_type_bout/duration_n10.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()





plt.figure(figsize=(8,8))

sns.barplot(data=df, x='predominant_type', y='duration', hue='condition', edgecolor='black', linewidth=2, errorbar='sd', palette=palette, alpha=0.8)

plt.xlabel('Interaction Type Duration', fontsize=12, fontweight='bold')
plt.ylabel('duration', fontsize=12, fontweight='bold')


# Add an overall title to the entire figure
plt.title('Interaction Type Duration (predom)', fontsize=16, fontweight='bold')

# Adjust layout to prevent overlap, considering the overall title
# plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.tight_layout(rect=[1, 1, 1, 1])

plt.ylim(0, None)

plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/interactions/interaction_type_bout/duration_predom_n10.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()



plt.figure(figsize=(8,8))


group3 = (
    df
    .groupby(['file', 'condition', 'initial_type'])['duration']  # pick the column to aggregate
    .sum()                                                      # sum durations in each group
    .reset_index(name='total_duration')                         # turn the Series back into a DataFrame
)



sns.barplot(data=group3, x='initial_type', y='total_duration', hue='condition', edgecolor='black', linewidth=2, errorbar='sd', palette=palette, alpha=0.8)

plt.xlabel('Interaction Type Duration', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')


# Add an overall title to the entire figure
plt.title('Interaction Type Duration Total', fontsize=16, fontweight='bold')

# Adjust layout to prevent overlap, considering the overall title
# plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.tight_layout(rect=[1, 1, 1, 1])

plt.ylim(0, None)

plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/interactions/interaction_type_bout/duration_sum_n10.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()


plt.figure(figsize=(8,8))


group4 = (
    df
    .groupby(['file', 'condition', 'predominant_type'])['duration']  # pick the column to aggregate
    .sum()                                                      # sum durations in each group
    .reset_index(name='total_duration')                         # turn the Series back into a DataFrame
)



sns.barplot(data=group4, x='predominant_type', y='total_duration', hue='condition', edgecolor='black', linewidth=2, errorbar='sd', palette=palette, alpha=0.8)

plt.xlabel('Interaction Type Duration Based on Predominant', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')


# Add an overall title to the entire figure
plt.title('Interaction Type Duration Total', fontsize=16, fontweight='bold')

# Adjust layout to prevent overlap, considering the overall title
# plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.tight_layout(rect=[1, 1, 1, 1])

plt.ylim(0, None)

plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/interactions/interaction_type_bout/duration_sum_n10-predom.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()



#### raw number of bouts 
plt.figure(figsize=(8,8))

# Group by condition and file, count bouts
number_bouts = (
    df.groupby(['condition', 'file'])['bout_id']
      .size()
      .reset_index(name='num_bouts')
)


sns.barplot(data=number_bouts, x='condition', y='num_bouts', hue='condition', edgecolor='black', linewidth=2, errorbar='sd', palette=palette, alpha=0.8)

plt.xlabel('', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')


# Add an overall title to the entire figure
plt.title('Total Interaction Bouts', fontsize=16, fontweight='bold')

# Adjust layout to prevent overlap, considering the overall title
# plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.tight_layout(rect=[1, 1, 1, 1])

plt.ylim(0, None)

# plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/interactions/interaction_type_bout/number_bouts.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()





plt.figure(figsize=(8,8))

cols = ['head_head','tail_tail','body_body','body_head','body_tail','head_tail']

# Reshape from wide to long
long_df = df.melt(
    id_vars=['condition', 'file'], 
    value_vars=cols, 
    var_name='type', 
    value_name='count'
)

# Group by condition, file, and type
df6 = (
    long_df.groupby(['condition', 'file', 'type'])['count']
           .sum()
           .reset_index()
)





sns.barplot(data=df6, x='type', y='count', hue='condition', edgecolor='black', linewidth=2, errorbar='sd', palette=palette, alpha=0.8)

plt.xlabel('', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')


plt.title('Total Interaction Frames', fontsize=16, fontweight='bold')


plt.tight_layout(rect=[1, 1, 1, 1])

plt.ylim(0, None)

plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/interactions/interaction_type_bout/total_framecount.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
