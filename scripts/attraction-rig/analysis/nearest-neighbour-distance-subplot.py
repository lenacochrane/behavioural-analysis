
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches




df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/group-housed/nearest_neighbour.csv')
df3['condition'] = 'GH_N2'

df4 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/socially-isolated/nearest_neighbour.csv')
df4['condition'] = 'SI_N2'

df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed/nearest_neighbour.csv')
df5['condition'] = 'GH_N10'

df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/socially-isolated/nearest_neighbour.csv')
df6['condition'] = 'SI_N10'

df7 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated/nearest_neighbour.csv')
df7['condition'] = 'PSEUDO-SI_N10'

df8 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/nearest_neighbour.csv')
df8['condition'] = 'PSEUDO-GH_N10'

df9 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/socially-isolated/nearest_neighbour.csv')
df9['condition'] = 'PSEUDO-SI_N2'

df10 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/group-housed/nearest_neighbour.csv')
df10['condition'] = 'PSEUDO-GH_N2'



df = pd.concat([df5, df6], ignore_index=True)


# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
axes = axes.flatten()

bins = np.linspace(0, 90, 90)  # 0 to 2.5 in 0.1 increments

for i in range(6):
    start = i * 600
    end = start + 600
    df_interval = df[(df['frame'] >= start) & (df['frame'] < end)]
    
    # Bin speeds
    df_interval['bin'] = pd.cut(df_interval['body-body'], bins, include_lowest=True)
    df_interval['bin_center'] = df_interval['bin'].apply(lambda x: x.mid)

    # Per-file density
    density_df = (
        df_interval
        .groupby(['filename', 'condition', 'bin_center'])
        .size()
        .groupby(['filename', 'condition'], group_keys=False)
        .apply(lambda x: x / x.sum())
        .reset_index(name='density')
    )

    # Plot lineplot with SD as errorbar
    sns.lineplot(
        data=density_df,
        x='bin_center',
        y='density',
        hue='condition',
        errorbar='sd',
        ax=axes[i]
    )
    
    axes[i].set_title(f'{start}-{end} frames', fontsize=14, fontweight='bold')
    axes[i].set_xlabel('Nearest Neighour (mm)', fontsize=10, fontweight='bold')
    axes[i].set_ylabel('Density', fontsize=10, fontweight='bold')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].tick_params(axis='both', labelsize=10)


# Adjust layout
plt.suptitle('Nearest Neighour Distribution Over Time', fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.ylim(0, 0.1)




plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/nearest-neighour-distance/nearest-neighour-subplot-n10.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
