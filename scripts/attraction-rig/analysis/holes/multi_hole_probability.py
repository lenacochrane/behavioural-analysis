
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/holes/N10-4-HOLE/GROUP-HOUSED/probability-entry.csv')
df1['condition'] = 'GH'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/holes/N10-4-HOLE/SOCIAL-ISOLATION/probability-entry.csv')
df2['condition'] = 'SI'

plt.figure(figsize=(8,8))

df = pd.concat([df1, df2], ignore_index=True)


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

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes_multi/probability_enter/prob.png', dpi=300, bbox_inches='tight')
plt.close()




bin_size = 600
df["time_bin"] = (df["decision_frame"] // bin_size) * bin_size  # 0,600,1200,...

# --- per file per bin probability, then average across files ---
grouped = (
    df.groupby(["condition", "file", "time_bin"])["entry"]
      .mean()
      .reset_index()
      .rename(columns={"entry": "probability_of_entry"})
)

plt.figure(figsize=(8,8))
sns.lineplot(
    data=grouped,
    x="time_bin",
    y="probability_of_entry",
    hue="condition",
    errorbar="sd"
)

plt.xlabel("Time bin (frames)", fontsize=12, fontweight="bold")
plt.ylabel("Probability of entering", fontsize=12, fontweight="bold")
plt.title("Probability of hole entry over time", fontsize=16, fontweight="bold")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes_multi/probability_enter/prob_binned.png',
            dpi=300, bbox_inches='tight')
plt.close()