
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches




df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/nearest_neighbour.csv')
df5['condition'] = 'GH'

# df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/nearest_neighbour.csv')
# df6['condition'] = 'SI'

df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/nearest_neighbour.csv')
df6['condition'] = 'PSEUDO-GH'

plt.figure(figsize=(8,8))

df = pd.concat([df6, df5], ignore_index=True)


bins = list(range(0, 90, 1))  # [0, 10, 20, ..., 100]
df['bin'] = pd.cut(df['closest_node_distance'], bins, include_lowest=True)
df['bin_center'] = df['bin'].apply(lambda x: x.mid)


# Count per file-condition-bin
counts = (
    df.groupby(['filename', 'condition', 'bin_center'])
    .size()
    .groupby(['filename', 'condition'], group_keys=False)
    .apply(lambda x: x / x.sum())
    .reset_index(name='density')
)


sns.lineplot(data=counts, x='bin_center', y='density', hue='condition', errorbar='sd')


# sns.histplot(data=df, x='body-body', hue='condition', stat='density', common_norm=False, alpha=0.5, binwidth=1, multiple='dodge')

plt.xlabel('Nearest Neighbour Distance (mm)', fontsize=12, fontweight='bold')
plt.ylabel('Probability', fontsize=12, fontweight='bold')

plt.title('Nearest Neighour Distance Distriubtion', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

# plt.xlim(0,15)
# plt.xscale('log')
plt.ylim(0, 0.1)

plt.xticks(rotation=45)

plt.savefig('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/is_it_vision/nearest_neighour_min_distance/distance_probability.png', dpi=300, bbox_inches='tight')
plt.show()
