
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/holes/N10-1-HOLE/GROUP-HOUSED/hole_frame_counts.csv')
df1['condition'] = 'GH'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/holes/N10-1-HOLE/SOCIAL-ISOLATION/hole_frame_counts.csv')
df2['condition'] = 'SI'

plt.figure(figsize=(8,8))

df = pd.concat([df1, df2], ignore_index=True)

sns.barplot(data=df, x='condition', y='percent_in_hole', ci='sd',alpha=0.8)

plt.xlabel('Condition', fontsize=12, fontweight='bold')
plt.ylabel('% Time Inside Hole', fontsize=12, fontweight='bold')

plt.ylim()

plt.title('% Time Inside Hole Per Larvae', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.xticks(fontweight='bold')

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes/hole_count/hole-count-frames.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()


