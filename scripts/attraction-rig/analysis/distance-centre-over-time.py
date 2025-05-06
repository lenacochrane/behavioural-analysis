
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys


# Load your data
df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed/distance_over_time.csv')
df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/socially-isolated/distance_over_time.csv')


plt.figure(figsize=(8,6))

# Manually assign DataFrames to specific subplots using seaborn.lineplot
sns.lineplot(data=df1, x='frame', y='distance_from_centre', label='Group Housed')
sns.lineplot(data=df2, x='frame', y='distance_from_centre', label='Socially Isolated')

plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Distance From Centre (mm)', fontsize=12)

# plt.xlim(800,1000)

# Add an overall title to the entire figure
plt.title('Distance from Centre', fontsize=16, fontweight='bold')

# Adjust layout to prevent overlap, considering the overall title
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/analysis/distance-from-centre/distance-over-time.png', dpi=300, bbox_inches='tight')


# Show the plot
plt.show()
