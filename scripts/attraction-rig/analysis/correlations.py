
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches

df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed/correlations.csv')
df1['condition'] = 'Group Housed'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/socially-isolated/correlations.csv')
df2['condition'] = 'Socially Isolated'

plt.figure(figsize=(8,8))

df = pd.concat([df1, df2], ignore_index=True)

bins = list(range(0, 91, 10))  # [0, 10, 20, ..., 100]
df['distance_bin'] = pd.cut(df['body-body'], bins=bins)


# df['distance_bin'] = pd.cut(df['body-body'], bins=9)

binned_summary = df.groupby(['distance_bin', 'condition'])['angle'].mean().reset_index()


sns.barplot(
    data=binned_summary,
    x='distance_bin',
    y='angle',
    hue='condition',
    edgecolor='black'
)



plt.xlabel('Body-Body Distance', fontsize=12, fontweight='bold')
plt.ylabel('Angle', fontsize=12, fontweight='bold')


# Add an overall title to the entire figure
plt.title('Correlation: Distance and Angle', fontsize=16, fontweight='bold')

# Adjust layout to prevent overlap, considering the overall title
# plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.tight_layout(rect=[1, 1, 1, 1])





plt.xticks(rotation=45)


plt.savefig('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/analysis/interactions/correlations-acceleration.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
