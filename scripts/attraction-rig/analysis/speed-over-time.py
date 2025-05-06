
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed/speed_over_time.csv')


df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/socially-isolated/speed_over_time.csv')


plt.figure(figsize=(8,6))

# Manually assign DataFrames to specific subplots using seaborn.lineplot
sns.lineplot(data=df1, x='time', y='speed',  color='#4A6792', ci=None, label='Group Housed')

sns.lineplot(data=df2, x='time', y='speed', color='#ECB0C3', ci=None, label='Socially Isolated')



plt.xlabel('Time (S)', fontsize=12)
plt.ylabel('Speed (mm/s)', fontsize=12)

# plt.ylim(0, 90)

# Add an overall title to the entire figure
plt.title('Speed Over Time', fontsize=16, fontweight='bold')

# Adjust layout to prevent overlap, considering the overall title
plt.tight_layout(rect=[0, 0, 1, 0.95])


plt.savefig('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/analysis/speed/speed-over-time.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
