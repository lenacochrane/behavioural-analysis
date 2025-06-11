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


df = pd.concat([df3, df4], ignore_index=True) #n2

# df = pd.concat([ df5, df6, df7, df8], ignore_index=True) #n10

bins = sorted(df['frame_bin'].unique())[:6]  # only take the first 6 bins


fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
axes = axes.flatten()



for i, bin_val in enumerate(bins):
    ax = axes[i]
    df_bin = df[df['frame_bin'] == bin_val]

    grouped = (
        df_bin.groupby(['condition', 'file', 'interaction_type'])['count']
        .sum()
        .reset_index()
    )

    sns.barplot(
        data=grouped,
        x='interaction_type',
        y='count',
        hue='condition',
        ax=ax,
        edgecolor='black',
        linewidth=2, alpha=0.8, palette=palette
    )

    bin_label = f"{bin_val}-{bin_val + 599}"
    ax.set_title(f'Frames {bin_label}', fontsize=10, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Count')
    ax.set_ylim(0, 10)
    ax.tick_params(axis='x', rotation=45)
    

# Remove any unused subplots if bins < 6
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])



fig.suptitle('Interaction Types per Frame Bin', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
 

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/interactions/node-node/n2_subplot.png', dpi=300, bbox_inches='tight')

plt.show()


