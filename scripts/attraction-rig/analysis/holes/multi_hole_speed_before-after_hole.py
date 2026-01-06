
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/holes/N10-4-HOLE/GROUP-HOUSED/speed.csv')
df1['condition'] = 'GH'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/holes/N10-4-HOLE/SOCIAL-ISOLATION/speed.csv')
df2['condition'] = 'SI'


df = pd.concat([df1, df2], ignore_index=True)


fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

palette = {'GH': "#3891d0", 'SI': "#ff8945"}  # blue and red

sns.barplot(data=df, x='condition', y='mean_speed_before', edgecolor='black',linewidth=2, ci='sd', alpha=0.6, ax=axes[0], palette=palette)
axes[0].set_title('Before Entry')
axes[0].set_xlabel('Condition')
axes[0].set_ylabel('Mean Speed')
axes[0].set_ylim(0, None)
axes[0].tick_params(axis='x', labelrotation=0)
axes[0].tick_params(axis='x', labelsize=10)
axes[0].set_xticklabels(axes[0].get_xticklabels(), fontweight='bold')

# After speed
sns.barplot(data=df, x='condition', y='mean_speed_after', edgecolor='black', linewidth=2, ci='sd',  alpha=0.6, ax=axes[1], palette=palette)
axes[1].set_title('After Entry')
axes[1].set_xlabel('Condition')
axes[1].set_ylabel('')
axes[1].set_ylim(0, None)
axes[1].tick_params(axis='x', labelrotation=0)
axes[1].tick_params(axis='x', labelsize=10)
axes[1].set_xticklabels(axes[1].get_xticklabels(), fontweight='bold')

# Adjust layout
fig.suptitle('Larval Speed Before vs After Hole Entry', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle

# Save
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes_multi/speed/speed_before-after_hole.png', dpi=300, bbox_inches='tight')

# Show
plt.show()