
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches



df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/nearest_neighbour.csv')
df1['condition'] = 'GH'

# df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/nearest_neighbour.csv')
# df2['condition'] = 'SI'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/nearest_neighbour.csv')
df2['condition'] = 'PSEUDO-GH'

df = pd.concat([df1, df2], ignore_index=True)


bins = list(range(0, 90, 1))  # [0, 10, 20, ..., 100]
df['bin'] = pd.cut(df['closest_node_distance'], bins, include_lowest=True)
df['bin_center'] = df['bin'].apply(lambda x: x.mid)

plt.figure(figsize=(12, 8))

sns.lineplot(
    data=df,
    x='bin_center',
    y='speed',
    hue='condition',
    errorbar=('ci', 95))



plt.ylim(0,2)
plt.xlim(0,20)
plt.subplots_adjust(wspace=0.3, hspace=0.4)
plt.suptitle('Speed vs Nearest Neighour Distance', fontsize=16, fontweight='bold')
plt.subplots_adjust(wspace=0.3, hspace=0.4)


# plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/is_it_vision/nearest_neighour_min_distance/speed.png', dpi=300, bbox_inches='tight')

plt.show()
