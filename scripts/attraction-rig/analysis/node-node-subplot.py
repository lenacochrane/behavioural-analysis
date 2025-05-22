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


# df = pd.concat([df3, df4, df9, df10], ignore_index=True) #n2
df = pd.concat([ df5, df6, df7, df8], ignore_index=True) #n10


# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
axes = axes.flatten()

for i in range(6):
    start = i * 600
    end = start + 600
    df_interval = df[(df['time'] >= start) & (df['time'] < end)]

    sns.barplot(data=df, x='interaction_type', y='count', edgecolor='black', linewidth=2, ci='sd', hue='condition', alpha=0.8,  ax=axes[i])
    
    axes[i].set_title(f'{start}-{end}s', fontsize=14, fontweight='bold')
    axes[i].set_xlim(0, 2)
    # axes[i].set_ylim(0, 2)
    axes[i].set_xlabel('', fontsize=10, fontweight='bold')
    axes[i].set_ylabel('Frequency', fontsize=10, fontweight='bold')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].tick_params(axis='both', labelsize=10)

# Adjust layout
plt.suptitle('Node Contact Frequency <1mm', fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
 

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/interactions/node-node/n10_subplot.png', dpi=300, bbox_inches='tight')

plt.show()


