
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/holes/N10-1-HOLE/GROUP-HOUSED/hole_distance.csv')
df1['condition'] = 'GH'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/holes/N10-1-HOLE/SOCIAL-ISOLATION/hole_distance.csv')
df2['condition'] = 'SI'

plt.figure(figsize=(8,8))

df = pd.concat([df1, df2], ignore_index=True)

bins = np.linspace(0, 90, 50)  # 0 to 2.5 in 0.1 increments
df['distance_bin'] = pd.cut(df['distance_from_hole'], bins, include_lowest=True)
df['bin_center'] = df['distance_bin'].apply(lambda x: x.mid)


# Count per file-condition-bin
counts = (
    df.groupby(['file', 'condition', 'bin_center'])
    .size()
    .groupby(['file', 'condition'], group_keys=False)
    .apply(lambda x: x / x.sum())
    .reset_index(name='density')
)


sns.lineplot(data=counts, x='bin_center', y='density', hue='condition', errorbar='sd')

plt.xlabel('Distance (mm)', fontsize=12, fontweight='bold')
plt.ylabel('Probability', fontsize=12, fontweight='bold')

# plt.ylim(0, 10)

plt.title('Distance from Hole', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.ylim(0, None)

plt.xticks(fontweight='bold')

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes/distance_hole/distance-from-hole.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()


