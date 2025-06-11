
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/holes/N10-1-HOLE/GROUP-HOUSED/hole_orientation.csv')
df1['condition'] = 'GH'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/holes/N10-1-HOLE/SOCIAL-ISOLATION/hole_orientation.csv')
df2['condition'] = 'SI'

# Define number of bins
bins = np.linspace(0, 180, 18)  # 18 bins between 0 and 180 degrees

# Calculate histogram (counts) for each dataset
counts_gh, _ = np.histogram(df1['hole orientation'], bins=bins)
counts_si, _ = np.histogram(df2['hole orientation'], bins=bins)

# Convert the counts into probabilities (normalized by the total number of values)
prob_gh = counts_gh / sum(counts_gh)
prob_si = counts_si / sum(counts_si)

# Convert bin centers to radians for the polar plot (since we are working from 0 to 180 degrees)
bin_centers = np.deg2rad((bins[:-1] + bins[1:]) / 2)

# Create the plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 6))

# Plot the histograms on the polar plot (one value per bin)
ax.bar(bin_centers, prob_gh, width=np.diff(np.deg2rad(bins)), edgecolor='k', alpha=0.6, label='GH', color='#4A6792')
ax.bar(bin_centers, prob_si, width=np.diff(np.deg2rad(bins)), edgecolor='k', alpha=0.6, label='SI', color='#F39B6D')

# Customize the polar plot to show only a semi-circle (0 to 180 degrees)
ax.set_theta_zero_location("W")  # Set the 0 degrees at the top
ax.set_theta_direction(-1)       # Move in a clockwise direction
ax.set_thetalim(0, np.pi)        # Limit to 0 to 180 degrees (semi-circle)

# Add labels and title
plt.title('Hole Orientation', fontsize=14)
plt.legend(loc='upper right')


plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes/hole_oreintation/oreintation.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()





