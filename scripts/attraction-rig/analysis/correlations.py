
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches




df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/group-housed/correlations.csv')
df3['condition'] = 'GH_N2'

df4 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/socially-isolated/correlations.csv')
df4['condition'] = 'SI_N2'

df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed/correlations.csv')
df5['condition'] = 'GH_N10'

df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/socially-isolated/correlations.csv')
df6['condition'] = 'SI_N10'

df7 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated/correlations.csv')
df7['condition'] = 'PSEUDO-SI_N10'

df8 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/correlations.csv')
df8['condition'] = 'PSEUDO-GH_N10'

df9 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/socially-isolated/correlations.csv')
df9['condition'] = 'PSEUDO-SI_N2'

df10 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/group-housed/correlations.csv')
df10['condition'] = 'PSEUDO-GH_N2'




plt.figure(figsize=(8,8))

### N2
# df = pd.concat([df3, df4], ignore_index=True)

### N10
# df = pd.concat([df5, df6], ignore_index=True)

### N2 PSEUDO 
# df = pd.concat([df3, df4, df9, df10], ignore_index=True)

### N2 PSEUDO GH
df = pd.concat([df3, df10], ignore_index=True)

### N10 PSEUDO GH
# df = pd.concat([df5, df8], ignore_index=True)


# bins = list(range(0, 90, 1))  # [0, 10, 20, ..., 100]
# df['distance_bin'] = pd.cut(df['body-body'], bins=bins)

# # binned_summary = df.groupby(['distance_bin', 'condition'])['angle'].mean().reset_index()


# sns.lineplot(
#     data=df,
#     x='distance_bin',
#     y='angle',
#     hue='condition',
#     ci='sd',
#     edgecolor='black'
# )


bins = np.linspace(0, 90, 90)  # 0 to 2.5 in 0.1 increments
df['bin'] = pd.cut(df['body-body'], bins, include_lowest=True)
df['bin_center'] = df['bin'].apply(lambda x: x.mid)


summary = (
    df.groupby(['filename', 'condition', 'bin_center'])['speed']
    .mean()
    .reset_index()
)

# Plot mean ± sd of per-file averages
sns.lineplot(
    data=summary,
    x='bin_center',
    y='speed',
    hue='condition',
    errorbar='sd'
)




plt.xlabel('Body-Body Distance', fontsize=12, fontweight='bold')
plt.ylabel('Speed', fontsize=12, fontweight='bold')

plt.title('Correlation: Distance and Speed', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.xlim(0,20)


plt.xticks(rotation=45)


plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/nearest-neighour-distance/n2-pseudo-speed.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
