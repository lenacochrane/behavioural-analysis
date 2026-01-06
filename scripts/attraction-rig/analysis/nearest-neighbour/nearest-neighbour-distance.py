
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



plt.figure(figsize=(8,8))

df = pd.concat([df6, df5], ignore_index=True)


bins = np.linspace(0, 90, 90)  # 0 to 2.5 in 0.1 increments
df['bin'] = pd.cut(df['body-body'], bins, include_lowest=True)
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
plt.ylim(0, 0.07)

plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/nearest-neighour-distance/nearest-neighour-n10.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
