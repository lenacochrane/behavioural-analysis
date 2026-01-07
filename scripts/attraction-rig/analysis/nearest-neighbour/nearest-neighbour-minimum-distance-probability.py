
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import matplotlib as mpl

# ---- Adobe / Illustrator friendly PDFs ----
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# ---- Seaborn default "deep" colours ----
pal = sns.color_palette("deep", 4)

PALETTE = {
    "GH": pal[0],     # blue
    "SI": pal[1], # orange,
    "PSEUDO-GH": pal[2], # orange
    "PSEUDO-SI": pal[3]  # orange
}

HUE_ORDER = ["GH", "SI", "PSEUDO-GH", "PSEUDO-SI"]


df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/nearest_neighbour.csv')
df5['condition'] = 'GH'

df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/nearest_neighbour.csv')
df6['condition'] = 'SI'

# df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/nearest_neighbour.csv')
# df6['condition'] = 'PSEUDO-GH'

# df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated/nearest_neighbour.csv')
# df6['condition'] = 'PSEUDO-SI'

df = pd.concat([df6, df5], ignore_index=True)



""" CLOSEST NODE DISTANCE TO ANY NODE PROBABILITY DISTRIBUTION """

bins = list(range(0, 90, 1))  # [0, 10, 20, ..., 100]
df['bin'] = pd.cut(df['closest_node_distance'], bins, include_lowest=True)
df['bin_center'] = df['bin'].apply(lambda x: x.mid)


counts = (
    df.groupby(['filename', 'condition', 'bin_center'])
    .size()
    .groupby(['filename', 'condition'], group_keys=False)
    .apply(lambda x: x / x.sum())
    .reset_index(name='density')
)

plt.figure(figsize=(8,8))

sns.lineplot(data=counts, x='bin_center', y='density', hue='condition', errorbar='sd', palette=PALETTE, hue_order=HUE_ORDER)

plt.xlabel('Nearest Neighbour Distance (mm)', fontsize=12, fontweight='bold')
plt.ylabel('Probability', fontsize=12, fontweight='bold')

plt.title('Nearest Neighour Distance Distribution', fontsize=16, fontweight='bold')

plt.axvline(1, color='gray', linestyle='--', linewidth=0.8)

plt.tight_layout(rect=[1, 1, 1, 1])

# plt.xlim(0,15)
# plt.xscale('log')
plt.ylim(0, 0.1)
plt.xlim(0,60)

plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/nearest-neighour-distance/distance_probability_min-distance_n10.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/nearest-neighour-distance/distance_probability_min-distance_n10.pdf', dpi=300, bbox_inches='tight')
plt.close()



""" CLOSEST NODE DISTANCE TO HEAD PROBABILITY DISTRIBUTION """

bins = list(range(0, 90, 1))  # [0, 10, 20, ..., 100]
df['bin'] = pd.cut(df['head_distance'], bins, include_lowest=True)
df['bin_center'] = df['bin'].apply(lambda x: x.mid)


# Count per file-condition-bin
counts = (
    df.groupby(['filename', 'condition', 'bin_center'])
    .size()
    .groupby(['filename', 'condition'], group_keys=False)
    .apply(lambda x: x / x.sum())
    .reset_index(name='density')
)

plt.figure(figsize=(8,8))

sns.lineplot(data=counts, x='bin_center', y='density', hue='condition', errorbar='sd', palette=PALETTE, hue_order=HUE_ORDER)


plt.xlabel('Nearest Neighbour Distance (mm)', fontsize=12, fontweight='bold')
plt.ylabel('Probability', fontsize=12, fontweight='bold')

plt.title('Nearest Neighour Distance Distribution', fontsize=16, fontweight='bold')

plt.axvline(1, color='gray', linestyle='--', linewidth=0.8)

plt.tight_layout(rect=[1, 1, 1, 1])

# plt.xlim(0,15)
# plt.xscale('log')
plt.ylim(0, 0.1)
plt.xlim(0,60)

plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/nearest-neighour-distance/distance_probability_min-distance-to-head_n10.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/nearest-neighour-distance/distance_probability_min-distance-to-head_n10.pdf', dpi=300, bbox_inches='tight')
plt.close()