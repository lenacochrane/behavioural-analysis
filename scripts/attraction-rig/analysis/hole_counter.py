
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/holes/N10-1-HOLE/GROUP-HOUSED/hole_count.csv')
df1['condition'] = 'GH'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/holes/N10-1-HOLE/SOCIAL-ISOLATION/hole_count.csv')
df2['condition'] = 'SI'

plt.figure(figsize=(8,8))

df = pd.concat([df1, df2], ignore_index=True)


sns.lineplot(data=df, x='time', y='inside_count', ci='sd', hue='condition')

plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
plt.ylabel('Hole Count', fontsize=12, fontweight='bold')

plt.ylim(0, 10)

plt.title('Number Inside Hole', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])


plt.xticks(fontweight='bold')

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes/hole_count/hole-count.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()


