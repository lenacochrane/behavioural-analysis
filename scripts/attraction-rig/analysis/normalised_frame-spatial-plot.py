
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches




df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/group-housed/interactions.csv')
df3['condition'] = 'GH_N2'

df4 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/socially-isolated/interactions.csv')
df4['condition'] = 'SI_N2'

df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed/interactions.csv')
df5['condition'] = 'GH_N10'

df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/socially-isolated/interactions.csv')
df6['condition'] = 'SI_N10'




plt.figure(figsize=(8,8))

df = pd.concat([df5, df6], ignore_index=True) #n10

# df = pd.concat([df3, df4], ignore_index=True) # n2

df_frame = df[df['Normalized Frame'] == 0]  # or use .between(-5, 5) for a range

df_frame['x'] = df_frame[['Track_1 x_body', 'Track_2 x_body']].mean(axis=1)
df_frame['y'] = df_frame[['Track_1 y_body', 'Track_2 y_body']].mean(axis=1)

positions = df_frame[['x', 'y', 'condition']]

sns.scatterplot(data=positions, x='x', y='y', hue='condition', alpha=0.5)


plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)


plt.title('Normalised Frame', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/interactions/n10-normalised-frame-distance.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
