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
    .groupby(['condition', 'filename'])
    .agg(
        n_encounters=('closer', 'size'), # total number of encounters
        n_closer=('closer', 'sum'), # number of encounters where they got closer
        n_touch=('touch', 'sum') # number of encounters where they touched
    )
    .reset_index()
)

per_video['closer_percent'] = (
    per_video['n_closer'] / per_video['n_encounters'])

per_video['touch_percent'] = (
    per_video['n_touch'] / per_video['n_closer'])


plt.figure(figsize=(2,6))

sns.stripplot(
    data=per_video,
    x='condition',
    y='n_encounters',
    jitter=True,
    alpha=0.7
)


sns.pointplot(
    data=per_video,
    x='condition',
    y='n_encounters',
    errorbar=('ci', 95),
    color='black',
    markers='_',
    linestyle='none'
)

plt.ylabel('P(closer)')
plt.xlabel('')
# plt.ylim(0, 1)
plt.tight_layout()

plt.show()
