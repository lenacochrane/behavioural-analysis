
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






### N2
# df = pd.concat([df3, df4], ignore_index=True)

### N10
df = pd.concat([df5, df6], ignore_index=True)

### N2 PSEUDO SI
# df = pd.concat([df4, df9], ignore_index=True)

### N2 PSEUDO GH
# df = pd.concat([df3, df10], ignore_index=True)

### N10 PSEUDO GH
# df = pd.concat([df5, df8], ignore_index=True)

### N10 PSEUDO SI
# df = pd.concat([df6, df7], ignore_index=True)


bins = list(range(0, 90, 1))  # [0, 10, 20, ..., 100]
df['bin'] = pd.cut(df['body-body'], bins, include_lowest=True)
df['bin_center'] = df['bin'].apply(lambda x: x.mid)

# Create figure
fig, axes = plt.subplots(3, 3, figsize=(15, 15), sharey=True)
axes = axes.flatten()

# Loop over 10mm intervals: 0–10, 10–20, ..., 80–90
for i, start in enumerate(range(0, 90, 10)):
    end = start + 10
    ax = axes[i]

    # Filter for this window
    subset = df[(df['body-body'] >= start) & (df['body-body'] < end)]

    summary = (
    subset.groupby(['filename', 'condition', 'bin_center'])['angle']
    .mean()
    .reset_index())

    # Only add legend on the bottom-right subplot (index 8)

    # Plot mean ± sd of per-file averages
    sns.lineplot(
        data=summary,
        x='bin_center',
        y='angle',
        hue='condition',
        errorbar='sd', ax=ax, legend=(i == 3))
    
    
    ax.set_title(f'{start}–{end} mm')
    ax.set_xlim(start, end)
    if i % 3 == 0:
        ax.set_ylabel('angle')
    else:
        ax.set_ylabel('')
    if i >= 6:
        ax.set_xlabel('Distance')
    else:
        ax.set_xlabel('')

# Remove any unused axes
for j in range(i + 1, 9):
    fig.delaxes(axes[j]) 

plt.ylim(0,180)
plt.subplots_adjust(wspace=0.3, hspace=0.4)
plt.suptitle('Speed vs Nearest Neighour Distance', fontsize=16, fontweight='bold')
plt.subplots_adjust(wspace=0.3, hspace=0.4)


# plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/nearest-neighour-distance/n10-angle.png', dpi=300, bbox_inches='tight')

plt.show()
