
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from matplotlib.ticker import MaxNLocator

df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/holes/N10-1-HOLE/GROUP-HOUSED/interactions_return.csv')
df1['condition'] = 'GH'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/holes/N10-1-HOLE/SOCIAL-ISOLATION/interactions_return.csv')
df2['condition'] = 'SI'

df = pd.concat([df1, df2], ignore_index=True)


df['return_time'] = pd.to_numeric(df['return_time'], errors='coerce')

############ RETURN TIME VERSUS INTERACTION NUMBER

# summary = (
#     df.groupby(['file', 'condition', 'exiting_larva', 'exit_index', 'returned_to_hole'])
#       .agg({
#           'interacted': 'sum',
#           'return_time': 'first',
#           'partner_status': lambda x: x[x != 'none'].iloc[0] if any(x != 'none') else 'none'
#       })
#       .reset_index()
# )

def status_summary(x):
    statuses = set(x) - {'none'}
    if len(statuses) == 1:
        return statuses.pop()
    elif len(statuses) > 1:
        return 'mixed'
    else:
        return 'none'

summary = (
    df.groupby(['file', 'condition', 'exiting_larva', 'exit_index', 'returned_to_hole'])
      .agg({
          'interacted': 'sum',
          'return_time': 'first',
          'partner_status': status_summary
      })
      .reset_index()
)


# Split by condition
si = summary[summary['condition'] == 'SI']
gh = summary[summary['condition'] == 'GH']

order = ['naive', 'exposed', 'mixed']                   
palette = {'naive': '#1f77b4', 'exposed': '#ff7f0e', 'mixed': '#2ca02c'} 



fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# SI subplot
sns.scatterplot(data=si, x='return_time', y='interacted', hue='partner_status',hue_order=order, palette=palette, ax=axes[0])
axes[0].set_title('SI: Return Duration vs. Interaction Count', fontsize=14)
axes[0].set_xlabel('Return Duration (frames)')
axes[0].set_ylabel('Interaction Count')
axes[0].legend(title='Partner Status')
axes[0].xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))

# GH subplot
sns.scatterplot(data=gh, x='return_time', y='interacted', hue='partner_status', hue_order=order, palette=palette, ax=axes[1])
axes[1].set_title('GH: Return Duration vs. Interaction Count', fontsize=14)
axes[1].set_xlabel('Return Duration (frames)')
axes[1].set_ylabel('')
axes[1].legend(title='Partner Status')
axes[1].xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))

plt.suptitle('Interaction Count vs. Return Duration by Condition', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes/interactions/returns/return_time-interactions.png', dpi=300)
plt.show()








