
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches




df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/group-housed/correlations.csv')
df3['condition'] = 'GH_N2'

df4 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/socially-isolated/correlations.csv')
df4['condition'] = 'SI_N2'

df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed/correlations.csv')
df5['condition'] = 'GH_N10'

df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/socially-isolated/correlations.csv')
df6['condition'] = 'SI_N10'

df7 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated/correlations.csv')
df7['condition'] = 'PSEUDO-SI_N10'

df8 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/correlations.csv')
df8['condition'] = 'PSEUDO-GH_N10'

df9 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/socially-isolated/correlations.csv')
df9['condition'] = 'PSEUDO-SI_N2'

df10 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/group-housed/correlations.csv')
df10['condition'] = 'PSEUDO-GH_N2'


df = pd.concat([df3, df4], ignore_index=True)


# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
axes = axes.flatten()

bins = list(range(0, 90, 5))  # [0, 10, 20, ..., 100]
df['distance_bin'] = pd.cut(df['body-body'], bins=bins)

for i in range(6):
    start = i * 600
    end = start + 600
    df_interval = df[(df['frame'] >= start) & (df['frame'] < end)]
    
    sns.barplot(
    data=df_interval,
    x='distance_bin',
    y='speed',
    hue='condition',
    ci='sd',
    edgecolor='black', ax=axes[i])

    
    axes[i].set_title(f'{start}-{end} frames', fontsize=14, fontweight='bold')
    axes[i].set_xlabel('Body-Body Distance', fontsize=10, fontweight='bold')
    axes[i].set_ylabel('Speed', fontsize=10, fontweight='bold')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].tick_params(axis='both', labelsize=10)


# Adjust layout
plt.suptitle('Correlation: Distance and Speed Over Time', fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])



# binned_summary = df.groupby(['distance_bin', 'condition'])['angle'].mean().reset_index()









plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/nearest-neighour-distance/subplot-n2-speed.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
