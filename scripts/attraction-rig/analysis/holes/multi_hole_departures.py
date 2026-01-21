
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/holes/N10-4-HOLE/GROUP-HOUSED/departures_events.csv')
df1['condition'] = 'GH'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/holes/N10-4-HOLE/SOCIAL-ISOLATION/departures_events.csv')
df2['condition'] = 'SI'

plt.figure(figsize=(8,8))

df = pd.concat([df1, df2], ignore_index=True)

departures_per_file = (
    df.groupby(["condition", "file"])
      .size()
      .reset_index(name="n_departures")
)

sns.barplot(data=departures_per_file, x='condition', y='n_departures', edgecolor='black', linewidth=2, ci='sd', color='#2E8B57', alpha=0.6)

plt.xlabel('', fontsize=12, fontweight='bold')
plt.ylabel('Departures', fontsize=12, fontweight='bold')

# plt.ylim(0, 10)

plt.title('Number of Departures', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.ylim(0, None)

plt.xticks(fontweight='bold')

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes_multi/departures/departures.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes_multi/departures/departures.pdf', format='pdf', bbox_inches='tight')
plt.close()
