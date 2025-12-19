
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

dep_counts = (
    df.groupby(['condition', 'file', 'n_in_hole_at_departure'])
      .size()
      .reset_index(name='n_departures')
)

dep_counts['prob_exit'] = (
    dep_counts.groupby(['condition','file'])['n_departures']
              .transform(lambda x: x / x.sum())
)


sns.lineplot(
    data=dep_counts,
    x='n_in_hole_at_departure',
    y='prob_exit',
    hue='condition',
    errorbar='sd',   # sd across files
    marker='o',
    alpha=0.8
)

plt.xlabel('', fontsize=12, fontweight='bold')
plt.ylabel('Departure Probability', fontsize=12, fontweight='bold')

# plt.ylim(0, 10)

plt.title('Departure Probability', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.ylim(0, None)

plt.xticks(fontweight='bold')

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes_multi/departures/departure_probability.png', dpi=300, bbox_inches='tight')
plt.close()
