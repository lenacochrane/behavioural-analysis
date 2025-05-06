
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed/euclidean_distances.csv')
df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/socially-isolated/euclidean_distances.csv')


plt.figure(figsize=(8,6))

# Manually assign DataFrames to specific subplots using seaborn.lineplot
sns.lineplot(data=df1, x='time', y='average_distance',  color='#4A6792', ci=None, label='Group Housed')

sns.lineplot(data=df2, x='time', y='average_distance', color='#4A2C77', ci=None, label='Socially Isolated')

plt.xlim(0,600)


plt.xlabel('Time (S)', fontsize=12, fontweight='bold')
plt.ylabel('Average Distance (mm)', fontsize=12, fontweight='bold')

# plt.ylim(0,90)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(0,100)
# plt.xlim(0, 50)

# Add an overall title to the entire figure
plt.title('Euclidean Distances', fontsize=16, fontweight='bold')

# Adjust layout to prevent overlap, considering the overall title
# plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.legend(title='Number of Larvae')

plt.savefig('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/analysis/euclidean-distance/euclidean-distance.png', dpi=300, bbox_inches='tight')


# Show the plot
plt.show()
