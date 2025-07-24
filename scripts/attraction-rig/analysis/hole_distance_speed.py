
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/holes/N10-1-HOLE/GROUP-HOUSED/hole_distance.csv')
df1['condition'] = 'GH'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/holes/N10-1-HOLE/SOCIAL-ISOLATION/hole_distance.csv')
df2['condition'] = 'SI'

plt.figure(figsize=(8,8))

df = pd.concat([df1, df2], ignore_index=True)

# Create 10mm bins
bins = np.arange(0, 100, 10)  # 0–90 mm in 10mm steps
labels = [f'{b}-{b+10}' for b in bins[:-1]]  # '0-10', '10-20', ..., '80-90'

df['distance_bin'] = pd.cut(df['distance_from_hole'], bins=bins, labels=labels, right=False)

palette = {'GH': "#3891d0", 'SI': "#ff8945"}  # blue and red


sns.lineplot(data=df, x='distance_bin', y='speed', hue='condition', ci='sd', palette=palette)

plt.xlabel('Distance from Hole (mm)', fontsize=12, fontweight='bold')
plt.ylabel('Speed (mm/s)', fontsize=12, fontweight='bold')

# plt.ylim(0, 10)

plt.title('Larval Speed in Relation to Distance from Hole', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.ylim(0, None)

plt.xticks(fontweight='bold')


plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes/distance_hole/distance_speed.png', dpi=300, bbox_inches='tight')

plt.show()


