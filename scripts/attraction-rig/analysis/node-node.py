import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyarrow.feather as feather


df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/group-housed/interaction_types.csv')
df3['condition'] = 'GH_N2'

df4 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/socially-isolated/interaction_types.csv')
df4['condition'] = 'SI_N2'

df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed/interaction_types.csv')
df5['condition'] = 'GH_N10'

df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/socially-isolated/interaction_types.csv')
df6['condition'] = 'SI_N10'

df7 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated/interaction_types.csv')
df7['condition'] = 'PSEUDO-SI_N10'

df8 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/interaction_types.csv')
df8['condition'] = 'PSEUDO-GH_N10'

df9 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/socially-isolated/interaction_types.csv')
df9['condition'] = 'PSEUDO-SI_N2'

df10 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/group-housed/interaction_types.csv')
df10['condition'] = 'PSEUDO-GH_N2'



plt.figure(figsize=(8,8))

df = pd.concat([df5, df6], ignore_index=True) #n2
# df = pd.concat([ df5, df6, df7, df8], ignore_index=True) #n10

palette = {
    'SI_N2': '#FF7369',  # yellow-orange
    'SI_N10': '#FF7369',
    'GH_N2': '#0d75b9',
    'GH_N10': '#0d75b9',     # blue-green
    "PSEUDO-GH_N2": '#91CEFF',
    "PSEUDO-GH_N10": '#91CEFF',
    'PSEUDO-SI_N2': '#D8C2E0',
     'PSEUDO-SI_N10': '#D8C2E0'
    
}



grouped = (
    df.groupby(['file', 'condition', 'interaction_type'])['count']
    .sum()
    .reset_index()
)

sns.barplot(data=grouped, x='interaction_type', y='count', edgecolor='black', linewidth=2, ci='sd', hue='condition', alpha=0.8, palette=palette)

plt.xlabel('', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')

plt.ylim(0,1200)

plt.title('Node Contact Frequency', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])


plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/interactions/n10-.png', dpi=300, bbox_inches='tight')

plt.show()


