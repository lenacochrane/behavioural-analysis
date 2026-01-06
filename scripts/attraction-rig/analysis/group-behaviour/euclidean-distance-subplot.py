
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math


### N2
df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/group-housed/euclidean_distances.csv')
df4 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/socially-isolated/euclidean_distances.csv')

### N10
df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed/euclidean_distances.csv')
df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/socially-isolated/euclidean_distances.csv')

### PSEUDO N2
df7 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/group-housed/euclidean_distances.csv')
df8 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/socially-isolated/euclidean_distances.csv')

### PSEUDO N10
df9 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/euclidean_distances.csv')
df10 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated/euclidean_distances.csv')



# Group the DataFrame by 'file'
grouped = df9.groupby('file')

# How many unique files?
n = len(grouped)

# Grid layout setup
ncols = 5
nrows = math.ceil(n / ncols)

# Create subplots
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 3), sharex=True)
axes = axes.flatten()  # make it iterable

# Plot each group
for ax, (file_name, group_data) in zip(axes, grouped):
    sns.lineplot(data=group_data, x='frame', y='average_distance', color='#85C7DE', ci='sd', ax=ax, label='gh_n10')
    ax.set_title(file_name, fontsize=8)
    ax.set_xlabel('Time (s)', fontsize=8)
    ax.set_ylabel('Distance (mm)', fontsize=8)
    # ax.set_xlim(0, 600)
    ax.legend(fontsize=6, title='Number of Larvae', title_fontsize=7)

# Turn off unused subplots
for ax in axes[n:]:
    ax.axis('off')

# Set a global title
fig.suptitle('Euclidean Distances', fontsize=16, fontweight='bold')

# Adjust spacing to fit global title and avoid overlap
plt.tight_layout(rect=[0, 0, 1, 0.97])  # leave space for suptitle

# Save the full figure
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/euclidean-distance/n10-pseudo-grouphoused-subplot.png',
            dpi=300, bbox_inches='tight')
plt.show()











