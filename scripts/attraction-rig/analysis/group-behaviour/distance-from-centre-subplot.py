
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




## ALL DF
# df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10], ignore_index=True)

## N1
df = pd.concat([df1, df2], ignore_index=True)

## N2
# df = pd.concat([df3, df4], ignore_index=True)

## N10
# df = pd.concat([df5, df6], ignore_index=True)

## PEUDO N10
# df = pd.concat([ df6, df7], ignore_index=True) # si
# df = pd.concat([df5, df8], ignore_index=True) #gh

## PEUDO N2
# df = pd.concat([df4, df9], ignore_index=True) #si
# df = pd.concat([df3, df10], ignore_index=True) # gh


## GH
df = pd.concat([df1, df3, df5], ignore_index=True)

## SI
df = pd.concat([df2, df4, df6], ignore_index=True)


# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
axes = axes.flatten()
bins = np.linspace(0, 50, 25)  # 0 to 2.5 in 0.1 increments

for i in range(6):
    start = i * 600
    end = start + 600
    df_interval = df[(df['frame'] >= start) & (df['frame'] < end)]
    
    # Bin speeds
    df_interval['distance_bin'] = pd.cut(df_interval['distance_from_centre'], bins, include_lowest=True)
    df_interval['bin_center'] = df_interval['distance_bin'].apply(lambda x: x.mid)

    # Per-file density
    counts = (
    df_interval.groupby(['file', 'condition', 'bin_center'])
    .size()
    .groupby(['file', 'condition'], group_keys=False)
    .apply(lambda x: x / x.sum())
    .reset_index(name='density')
)

    # Plot lineplot with SD as errorbar
    sns.lineplot(
        data=counts,
        x='bin_center',
        y='density',
        hue='condition',
        errorbar='sd',
        ax=axes[i]
    )
    
    axes[i].set_title(f'{start}-{end} frames', fontsize=14, fontweight='bold')
    axes[i].set_xlabel('Distance From Centre (mm)', fontsize=10, fontweight='bold')
    axes[i].set_ylabel('Density', fontsize=10, fontweight='bold')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].tick_params(axis='both', labelsize=10)

plt.ylim(0,None)
# Adjust layout
plt.suptitle('Distance from Centre Distribution Over Time', fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/distance-from-centre/subplot-si.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()


