
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches

df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/interaction_types.csv')
df1['condition'] = 'GH'

# df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/interaction_types.csv')
# df1['condition'] = 'SI'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/interaction_types.csv')
df2['condition'] = 'PSEUDO-GH'

# df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated/interaction_types.csv')
# df2['condition'] = 'PSEUDO-SI'


palette = {
    'GH': "#0665c3", 
    'SI': "#e38213",
    'PSEUDO-GH': "#c27ba0",
    'PSEUDO-SI': "#7bc294"       
}

palette_order = ['GH', 'SI', 'PSEUDO-GH', 'PSEUDO-SI']


df = pd.concat([df1, df2], ignore_index=True)

# Sum across all frame bins per file + interaction type
grouped = (
    df.groupby(['file', 'condition', 'interaction_type'])['count']
    .sum()
    .reset_index()
)


plt.figure(figsize=(4,8))
sns.barplot(data=grouped, x='interaction_type', y='count', hue='condition', edgecolor='black', linewidth=2, errorbar='sd', palette=palette, alpha=0.8)

plt.xlabel('Interaction Type', fontsize=12, fontweight='bold')
plt.ylabel('Frame Count', fontsize=12, fontweight='bold')

plt.title('Interaction Type (All Nodes)', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.ylim(0, 650)

plt.xticks(rotation=45)


plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/proximal_contacts/interaction-type-all-nodes-n10_gh.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/proximal_contacts/interaction-type-all-nodes-n10_gh.pdf', 
            format='pdf', bbox_inches='tight')
plt.close()
