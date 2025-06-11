
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches

df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed/interaction_types.csv')
df1['condition'] = 'Group Housed'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/socially-isolated/interaction_types.csv')
df2['condition'] = 'Socially Isolated'

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/group-housed/interaction_types.csv')
df3['condition'] = 'Group Housed'

df4 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/socially-isolated/interaction_types.csv')
df4['condition'] = 'Socially Isolated'







plt.figure(figsize=(8,8))

palette = {
    'Socially Isolated': '#fca35d',  # yellow-orange
    'Group Housed': '#0d75b9'        # blue-green
}


df = pd.concat([df3, df4], ignore_index=True)

# Sum across all frame bins per file + interaction type
grouped = (
    df.groupby(['file', 'condition', 'interaction_type'])['count']
    .sum()
    .reset_index()
)

sns.barplot(data=grouped, x='interaction_type', y='count', hue='condition', edgecolor='black', linewidth=2, errorbar='sd', palette=palette, alpha=0.8)

plt.xlabel('Interaction Type', fontsize=12, fontweight='bold')
plt.ylabel('Interaction Count', fontsize=12, fontweight='bold')


# Add an overall title to the entire figure
plt.title('Interaction Type', fontsize=16, fontweight='bold')

# Adjust layout to prevent overlap, considering the overall title
# plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.tight_layout(rect=[1, 1, 1, 1])

plt.ylim(0, None)

plt.xticks(rotation=45)


plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/interactions/node-node/n2.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
