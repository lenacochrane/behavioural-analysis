import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import os
import matplotlib as mpl

df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/potential_interactions.csv')
df1['condition'] = 'GH'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/potential_interactions.csv')
df2['condition'] = 'SI'

df = pd.concat([df1, df2], ignore_index=True)


per_video = (
    df
    .groupby(['condition', 'filename']) #, 'filename'])
    .agg(
        n_encounters=('touch', 'size'),   # total encounters
        n_touch=('touch', 'sum'),          # encounters with touch
        touch_rate=('touch', 'mean')       # fraction that touched
    )
    .reset_index()
)



plt.figure(figsize=(2,6))

sns.stripplot(
    data=per_video,
    x='condition',
    y='touch_rate',
    jitter=True,
    alpha=0.7
)


sns.pointplot(
    data=per_video,
    x='condition',
    y='touch_rate',
    errorbar=('ci', 95),
    color='black',
    markers='_',
    linestyle='none'
)

plt.title('Potential Interactions\n(Threshold = 10 cm)')
plt.ylabel('P(touch)')
plt.xlabel('')
# plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/potential_interactions/potential_interactions_10.0.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/potential_interactions/potential_interactions_10.0.pdf', format='pdf', bbox_inches='tight')
plt.show()
