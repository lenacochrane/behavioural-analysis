
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

df7 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated/acceleration_accross_time.csv')
df7['condition'] = 'PSEUDO-SI_N10'

df8 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/acceleration_accross_time.csv')
df8['condition'] = 'PSEUDO-GH_N10'

df9 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/socially-isolated/acceleration_accross_time.csv')
df9['condition'] = 'PSEUDO-SI_N2'

df10 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/group-housed/acceleration_accross_time.csv')
df10['condition'] = 'PSEUDO-GH_N2'





## ALL DF
# df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10], ignore_index=True)

## N1
# df = pd.concat([df1, df2], ignore_index=True)

## N2
df = pd.concat([df3, df4], ignore_index=True)

## N10
# df = pd.concat([df5, df6], ignore_index=True)

## PEUDO N10
# df = pd.concat([df6, df7], ignore_index=True) #si
# df = pd.concat([df5, df8], ignore_index=True) #gh

## PEUDO N2
# df = pd.concat([df3, df10], ignore_index=True) # gh
# df = pd.concat([df4, df9], ignore_index=True) # si

## Choose the relevant combination (example: PEUDO N2)
# df = pd.concat([df1, df2], ignore_index=True)

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
axes = axes.flatten()

for i in range(6):
    start = i * 600
    end = start + 600
    df_interval = df[(df['time'] >= start) & (df['time'] < end)]
    
    sns.histplot(
        data=df_interval,
        x='acceleration',
        hue='condition',
        stat='density',
        common_norm=False,
        alpha=0.5,
        ax=axes[i]
    )
    
    axes[i].set_title(f'{start}-{end}s', fontsize=14, fontweight='bold')
    # axes[i].set_xlim(0, 2)
    # axes[i].set_ylim(0, 2)
    axes[i].set_xlabel('Acceleration (mm/s)', fontsize=10, fontweight='bold')
    axes[i].set_ylabel('Density', fontsize=10, fontweight='bold')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].tick_params(axis='both', labelsize=10)

# Adjust layout
plt.suptitle('Acceleration Distributions Over Time', fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])



plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/acceleration/subplot-n2.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
