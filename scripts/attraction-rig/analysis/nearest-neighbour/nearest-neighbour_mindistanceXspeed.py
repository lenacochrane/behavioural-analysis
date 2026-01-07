
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
pal = sns.color_palette("deep", 2)

PALETTE = {
    "GH": pal[0],     # blue
    "PSEUDO-GH": pal[1] # orange
}

HUE_ORDER = ["GH", "PSEUDO-GH"]


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/nearest_neighbour.csv')
df1['condition'] = 'GH'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/nearest_neighbour.csv')
df2['condition'] = 'PSEUDO-GH'


df = pd.concat([df1, df2], ignore_index=True)

bins = list(range(0, 90, 1))  # [0, 10, 20, ..., 100]
df['bin'] = pd.cut(df['closest_node_distance'], bins, include_lowest=True) #body-body ; closest_node_distance ; closest_node_to_head
df['bin_center'] = df['bin'].apply(lambda x: x.mid)


plt.figure(figsize=(12, 8))

sns.lineplot(
    data=df,
    x='bin_center',
    y='speed',
    hue='condition',
    errorbar=('ci', 95))


plt.axvline(1, color='black', linestyle='--', linewidth=1)
plt.ylim(0,2)
plt.xlim(0,20)
plt.subplots_adjust(wspace=0.3, hspace=0.4)
plt.suptitle('Speed vs Nearest Neighour Distance (Any Node)', fontsize=16, fontweight='bold')
plt.subplots_adjust(wspace=0.3, hspace=0.4)


plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/nearest-neighour-distance/speed-min_dist.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/nearest-neighour-distance/speed-min_dist.pdf', dpi=300, bbox_inches='tight')

plt.close()


bins = list(range(0, 90, 1))  # [0, 10, 20, ..., 100]
df['bin'] = pd.cut(df['body-body'], bins, include_lowest=True) #body-body ; closest_node_distance ; closest_node_to_head
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
plt.suptitle('Speed vs Nearest Neighour Distance (Body-Body)', fontsize=16, fontweight='bold')
plt.subplots_adjust(wspace=0.3, hspace=0.4)


plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/nearest-neighour-distance/speed-body_body.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/nearest-neighour-distance/speed-body_body.pdf', dpi=300, bbox_inches='tight')

plt.close()



bins = list(range(0, 90, 1))  # [0, 10, 20, ..., 100]
df['bin'] = pd.cut(df['head_distance'], bins, include_lowest=True) #body-body ; closest_node_distance ; head_distance
df['bin_center'] = df['bin'].apply(lambda x: x.mid)


plt.figure(figsize=(12, 8))

sns.lineplot(
    data=df,
    x='bin_center',
    y='speed',
    hue='condition',
    errorbar=('ci', 95))


plt.axvline(1, color='black', linestyle='--', linewidth=1)
plt.ylim(0,2)
plt.xlim(0,20)
plt.subplots_adjust(wspace=0.3, hspace=0.4)
plt.suptitle('Speed vs Nearest Neighour Distance From Head', fontsize=16, fontweight='bold')
plt.subplots_adjust(wspace=0.3, hspace=0.4)


plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/nearest-neighour-distance/speed-min_dist_head.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/nearest-neighour-distance/speed-min_dist_head.pdf', dpi=300, bbox_inches='tight')

plt.close()