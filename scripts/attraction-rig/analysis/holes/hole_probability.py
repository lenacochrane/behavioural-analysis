
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


# ✅ Group by condition, file, and number_inside_hole
grouped = df.groupby(["condition", "file", "number_inside_hole"])["entry"].mean().reset_index()
grouped.rename(columns={"entry": "probability_of_entry"}, inplace=True)


sns.lineplot(data=grouped, x='number_inside_hole' , y='probability_of_entry', hue='condition', errorbar='sd')


plt.xlabel('Number in Hole', fontsize=12, fontweight='bold')
plt.ylabel('Probability of Entering', fontsize=12, fontweight='bold')

# plt.ylim(0, 10)

plt.title('Probability of Hole Entry', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.ylim(0, None)

plt.xticks(fontweight='bold')

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes/probability_enter/prob.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes/probability_enter/prob.pdf', format='pdf', bbox_inches='tight')

# Show the plot
plt.show()


