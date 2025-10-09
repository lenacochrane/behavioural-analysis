
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/holes/N10-1-HOLE/GROUP-HOUSED/hole_entry_departure_latency.csv')
df1['condition'] = 'GH'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/holes/N10-1-HOLE/SOCIAL-ISOLATION/hole_entry_departure_latency.csv')
df2['condition'] = 'SI'

plt.figure(figsize=(8,8))

df = pd.concat([df1, df2], ignore_index=True)


# ✅ Group by condition, file, and number_inside_hole
grouped = df.groupby(["condition", "file", "number_in_hole_at_entry"])["departure_happened"].mean().reset_index()
grouped.rename(columns={"departure_happened": "probability_of_departing"}, inplace=True)


sns.lineplot(data=grouped, x='number_in_hole_at_entry' , y='probability_of_departing', hue='condition', errorbar='sd')


plt.xlabel('Number in Hole', fontsize=12, fontweight='bold')
plt.ylabel('Probability of Depearting', fontsize=12, fontweight='bold')

# plt.ylim(0, 10)

plt.title('Probability of Hole Departure Before New Entry', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.ylim(0, None)

plt.xticks(fontweight='bold')

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes/probability_enter/prob_departure.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()


