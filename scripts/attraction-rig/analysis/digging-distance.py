import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


##### 2 LARVAE DIGGING 
df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/group-housed/digging_distances_pair.csv')
df1['condition'] = 'GH_N2'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/socially-isolated/digging_distances_pair.csv')
df2['condition'] = 'SI_N2'

##### 1 LARVAE DIGGING 

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/group-housed/digging_distances_single.csv')
df3['condition'] = 'GH_N2'

df4 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/socially-isolated/digging_distances_single.csv')
df4['condition'] = 'SI_N2'

##### 2 LARVAE DIGGING 

df7 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/socially-isolated/digging_distances_pair.csv')
df7['condition'] = 'PSEUDO-SI_N2'

df8 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/group-housed/digging_distances_pair.csv')
df8['condition'] = 'PSEUDO-GH_N2'

##### 1 LARVAE DIGGING 

df9 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/socially-isolated/digging_distances_single.csv')
df9['condition'] = 'PSEUDO-SI_N2'

df10 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/group-housed/digging_distances_single.csv')
df10['condition'] = 'PSEUDO-GH_N2'




plt.figure(figsize=(8,8))

## BOTH LARVAE DIGGING
# df = pd.concat([df1, df2], ignore_index=True)

## BOTH LARVAE DIGGING- PSEUDO GH
# df = pd.concat([df1, df8], ignore_index=True)

## BOTH LARVAE DIGGING- PSEUDO SI 
# df = pd.concat([df2, df7], ignore_index=True)

## 1 LARVAE DIGGING
# df = pd.concat([df3, df4], ignore_index=True)

## 1 LARVAE DIGGING - PSEUDO GH
# df = pd.concat([df3, df10], ignore_index=True)

## 1 LARVAE DIGGING - PSEUDO SI
df = pd.concat([df4, df9], ignore_index=True)


bins = np.linspace(0, 90, 10)  # 0 to 2.5 in 0.1 increments
df['bin'] = pd.cut(df['distance'], bins, include_lowest=True)
df['bin_center'] = df['bin'].apply(lambda x: x.mid)


# Count per file-condition-bin
counts = (
    df.groupby(['file', 'condition', 'bin_center'])
    .size()
    .groupby(['file', 'condition'], group_keys=False)
    .apply(lambda x: x / x.sum())
    .reset_index(name='density')
)


sns.lineplot(data=counts, x='bin_center', y='density', hue='condition', errorbar='sd')



# sns.histplot(data=df, x='distance', hue='condition', stat='density', common_norm=False, alpha=0.5)


plt.xlabel('Distance', fontsize=12)
plt.ylabel('Density', fontsize=12)

plt.ylim(0, None)

# Add an overall title to the entire figure
plt.title('Distance Between Larval Digging', fontsize=16, fontweight='bold')

# Adjust layout to prevent overlap, considering the overall title
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/digging/n2-one-digging-pseudo-si.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

