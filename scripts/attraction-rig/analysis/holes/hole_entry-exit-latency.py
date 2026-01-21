
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/holes/N10-1-HOLE/GROUP-HOUSED/hole_entry_departure_latency.csv')
df1['condition'] = 'GH'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/holes/N10-1-HOLE/SOCIAL-ISOLATION/hole_entry_departure_latency.csv')
df2['condition'] = 'SI'

plt.figure(figsize=(8,8))

df = pd.concat([df1, df2], ignore_index=True)

sns.barplot(data=df, x='condition', y='latency_to_next_departure', edgecolor='black', linewidth=2, ci='sd', color='#2E8B57', alpha=0.6)

plt.xlabel('', fontsize=12, fontweight='bold')
plt.ylabel('Time between entry and exit', fontsize=12, fontweight='bold')

# plt.ylim(0, 10)

plt.title('Time between entry and departure', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.ylim(0, None)

plt.xticks(fontweight='bold')

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes/departures/entry_departures_latency.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes/departures/entry_departures_latency.pdf', format='pdf', bbox_inches='tight')

# Show the plot
plt.show()


