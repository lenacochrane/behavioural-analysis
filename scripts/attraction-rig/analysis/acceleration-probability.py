
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n1/group-housed/acceleration_accross_time.csv')
df1['condition'] = 'GH_N1'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n1/socially-isolated/acceleration_accross_time.csv')
df2['condition'] = 'SI_N1'

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/group-housed/acceleration_accross_time.csv')
df3['condition'] = 'GH_N2'

df4 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/socially-isolated/acceleration_accross_time.csv')
df4['condition'] = 'SI_N2'

df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed/acceleration_accross_time.csv')
df5['condition'] = 'GH_N10'

df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/socially-isolated/acceleration_accross_time.csv')
df6['condition'] = 'SI_N10'

df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed/acceleration_accross_time.csv')
df5['condition'] = 'GH_N10'

# df7 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated/acceleration_accross_time.csv')
# df7['condition'] = 'PSEUDO-SI_N10'

# df8 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/acceleration_accross_time.csv')
# df8['condition'] = 'PSEUDO-GH_N10'

# df9 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/socially-isolated/acceleration_accross_time.csv')
# df9['condition'] = 'PSEUDO-SI_N2'

# df10 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/group-housed/acceleration_accross_time.csv')
# df10['condition'] = 'PSEUDO-GH_N2'


plt.figure(figsize=(8,8))


## ALL DF
# df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10], ignore_index=True)

## N1
# df = pd.concat([df1, df2], ignore_index=True)

## N2
df = pd.concat([df3, df4], ignore_index=True)

## N10
# df = pd.concat([df5, df6], ignore_index=True)

## PEUDO N10
# df = pd.concat([df5, df6, df8, df7], ignore_index=True)

## PEUDO N2
# df = pd.concat([df3, df4, df9, df10], ignore_index=True)




###### DF TIME FRAME
df = df[df['time'] < 601]



sns.histplot(data=df, x='acceleration', hue='condition', stat='density', common_norm=False, alpha=0.5)

plt.xlabel('', fontsize=12, fontweight='bold')
plt.ylabel('Acceleration (mm2/s)', fontsize=12, fontweight='bold')

# plt.ylim(0, 1.5)
plt.xlim(-2,2)

plt.title('Acceleration', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.xticks(rotation=45)



plt.xticks(fontweight='bold')


plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/acceleration/acceleration-probability-n2-600frames.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
