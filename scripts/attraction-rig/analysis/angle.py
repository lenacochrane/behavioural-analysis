
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys



df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed/angle_over_time.csv')
df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/socially-isolated/angle_over_time.csv')

plt.figure(figsize=(8,6))

sns.histplot(data=df1, x='angle', stat='probability', label='Group Housed')
sns.histplot(data=df2, x='angle', stat='probability', label='Socially Isolated')

plt.xlabel('Angle', fontsize=12)
plt.ylabel('Probability', fontsize=12)

# plt.ylim(0, 0.06)


# Add an overall title to the entire figure
plt.title('Trajectory Angle Probability Distribution', fontsize=16, fontweight='bold')

# Adjust layout to prevent overlap, considering the overall title
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/analysis/angle/angle.png', dpi=300, bbox_inches='tight')


# Show the plot
plt.show()
