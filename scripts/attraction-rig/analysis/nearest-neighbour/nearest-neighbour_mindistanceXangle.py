
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
df['bin'] = pd.cut(df['head_distance'], bins, include_lowest=True)
df['bin_center'] = df['bin'].apply(lambda x: x.mid)



plt.figure(figsize=(12, 8))

sns.lineplot(
    data=df,
    x='bin_center',
    y='angle',
    hue='condition',
    errorbar=('ci', 95))


plt.axvline(1, color='black', linestyle='--', linewidth=1)
plt.ylim(150,170)
plt.xlim(0,10)
plt.subplots_adjust(wspace=0.3, hspace=0.4)
plt.suptitle('Angle vs Nearest Neighour Distance', fontsize=16, fontweight='bold')
plt.subplots_adjust(wspace=0.3, hspace=0.4)


# plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/nearest-neighour-distance/angle-min_dist-head.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/nearest-neighour-distance/angle-min_dist-head.pdf', dpi=300, bbox_inches='tight')

plt.close()
