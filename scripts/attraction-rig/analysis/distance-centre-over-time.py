
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n1/group-housed/distance_from_centre.csv')
df1['condition'] = 'GH_N1'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n1/socially-isolated/distance_from_centre.csv')
df2['condition'] = 'SI_N1'

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/group-housed/distance_from_centre.csv')
df3['condition'] = 'GH_N2'

df4 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/socially-isolated/distance_from_centre.csv')
df4['condition'] = 'SI_N2'

df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed/distance_from_centre.csv')
df5['condition'] = 'GH_N10'

df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/socially-isolated/distance_from_centre.csv')
df6['condition'] = 'SI_N10'

df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed/distance_from_centre.csv')
df5['condition'] = 'GH_N10'

df7 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated/distance_over_time.csv')
df7['condition'] = 'PSEUDO-SI_N10'

df8 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/distance_over_time.csv')
df8['condition'] = 'PSEUDO-GH_N10'

df9 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/socially-isolated/distance_over_time.csv')
df9['condition'] = 'PSEUDO-SI_N2'

df10 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/group-housed/distance_over_time.csv')
df10['condition'] = 'PSEUDO-GH_N2'


plt.figure(figsize=(8,6))

## ALL DF
# df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10], ignore_index=True)

## N1
# df = pd.concat([df1, df2], ignore_index=True)

## N2
# df = pd.concat([df3, df4], ignore_index=True)

## N10
# df = pd.concat([df5, df6], ignore_index=True)

## PEUDO N10
# df = pd.concat([df5, df6, df8, df7], ignore_index=True)
df = pd.concat([df5,  df8], ignore_index=True) # GH

## PEUDO N2
# df = pd.concat([df3, df4, df9, df10], ignore_index=True)

## GH
# df = pd.concat([df1, df3, df5], ignore_index=True)


## SI
# df = pd.concat([df2, df4, df6], ignore_index=True)


# Manually assign DataFrames to specific subplots using seaborn.lineplot
sns.lineplot(data=df, x='frame', y='distance_from_centre', hue='condition', errorbar="sd")


plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Distance From Centre (mm)', fontsize=12)

plt.xlim(0,600)

# Add an overall title to the entire figure
plt.title('Distance from Centre', fontsize=16, fontweight='bold')

# Adjust layout to prevent overlap, considering the overall title
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/distance-from-centre/distance-from-centre_gh_pseudo_n10.png', dpi=300, bbox_inches='tight')


# Show the plot
plt.show()
