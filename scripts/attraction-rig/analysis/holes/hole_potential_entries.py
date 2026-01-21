
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/holes/N10-1-HOLE/GROUP-HOUSED/hole_probability.csv')
df1['condition'] = 'GH'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/holes/N10-1-HOLE/SOCIAL-ISOLATION/hole_probability.csv')
df2['condition'] = 'SI'

plt.figure(figsize=(8,8))

df = pd.concat([df1, df2], ignore_index=True)


# Filter for only non-entry events
df_false = df[df['entry'] == False]

# Group by condition and file, then count the number of False entries
grouped = df_false.groupby(['condition', 'file']).size().reset_index(name='missed_entries')

sns.barplot(data=grouped, x='condition', y='missed_entries', edgecolor='black', linewidth=2, ci='sd', color='#2E8B57', alpha=0.6)

plt.xlabel('', fontsize=12, fontweight='bold')
plt.ylabel('Number of Non Entry Decisions', fontsize=12, fontweight='bold')

# plt.ylim(0, 10)

plt.title('Rejected Hole Entries', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.ylim(0, None)

plt.xticks(fontweight='bold')

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes/probability_enter/missed_entries.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes/probability_enter/missed_entries.pdf', format='pdf', bbox_inches='tight')

# Show the plot
plt.show()


