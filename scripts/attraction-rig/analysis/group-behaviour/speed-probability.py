
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
    "GH_N10": pal[0],     # blue
    "SI_N10": pal[1] # orange
}

HUE_ORDER = ["GH_N10", "SI_N10"]


# df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n1/group-housed/speed_over_time.csv')
# df1['condition'] = 'GH_N1'

# df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n1/socially-isolated/speed_over_time.csv')
# df2['condition'] = 'SI_N1'

# df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/group-housed/speed_over_time.csv')
# df3['condition'] = 'GH_N2'

# df4 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/socially-isolated/speed_over_time.csv')
# df4['condition'] = 'SI_N2'

df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/speed_over_time.csv')
df5['condition'] = 'GH_N10'

df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/speed_over_time.csv')
df6['condition'] = 'SI_N10'

# df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed/speed_over_time.csv')
# df5['condition'] = 'GH_N10'

# df7 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated/speed_over_time.csv')
# df7['condition'] = 'PSEUDO-SI_N10'

# df8 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/speed_over_time.csv')
# df8['condition'] = 'PSEUDO-GH_N10'

# df9 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/socially-isolated/speed_over_time.csv')
# df9['condition'] = 'PSEUDO-SI_N2'

# df10 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/group-housed/speed_over_time.csv')
# df10['condition'] = 'PSEUDO-GH_N2'


plt.figure(figsize=(8,8))


## ALL DF
# df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10], ignore_index=True)

## N1
# df = pd.concat([df1, df2], ignore_index=True)

## N2
# df = pd.concat([df3, df4], ignore_index=True)

## N10
df = pd.concat([df5, df6], ignore_index=True)

## PEUDO N10 GH
# df = pd.concat([df5, df8], ignore_index=True)

## PEUDO N10 SI
# df = pd.concat([df6,  df7], ignore_index=True)

## PEUDO N2 GH
# df = pd.concat([df3, df10], ignore_index=True)

## PEUDO N2 SI
# df = pd.concat([df4, df9], ignore_index=True)

## GH
# df = pd.concat([df1, df3, df5], ignore_index=True)

## SI
# df = pd.concat([df2, df4, df6], ignore_index=True)


bins = np.linspace(0, 2.5, 26)  # 0 to 2.5 in 0.1 increments
df['speed_bin'] = pd.cut(df['speed'], bins, include_lowest=True)
df['bin_center'] = df['speed_bin'].apply(lambda x: x.mid)


# Count per file-condition-bin
counts = (
    df.groupby(['file', 'condition', 'bin_center'])
    .size()
    .groupby(['file', 'condition'], group_keys=False)
    .apply(lambda x: x / x.sum())
    .reset_index(name='density')
)


sns.lineplot(data=counts, x='bin_center', y='density', hue='condition', errorbar=('ci', 95), palette=PALETTE, hue_order=HUE_ORDER)


# sns.histplot(data=df, x='speed', hue='condition', stat='density', common_norm=False, alpha=0.5)


plt.ylabel('Density', fontsize=12, fontweight='bold')
plt.xlabel('Speed (mm/s)', fontsize=12, fontweight='bold')

# plt.ylim(0, 1.5)
plt.xlim(0,2.5)
plt.ylim(0, None)

plt.title('Speed', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.xticks(rotation=45)



plt.xticks(fontweight='bold')


plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/speed/si.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/speed/si.pdf', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
