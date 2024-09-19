
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/plots/euclidean-distance/n1-euclidean_distances.csv')
df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/plots/euclidean-distance/n2-euclidean_distances.csv')
df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/plots/euclidean-distance/n2-euclidean_distances.csv')
df4 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/plots/euclidean-distance/n2-euclidean_distances.csv')
df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/plots/euclidean-distance/n5-euclidean_distances.csv')
df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/plots/euclidean-distance/n6-euclidean_distances.csv')
df7 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/plots/euclidean-distance/n7-euclidean_distances.csv')
df8 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/plots/euclidean-distance/n8-euclidean_distances.csv')
df9 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/plots/euclidean-distance/n9-euclidean_distances.csv')
df10 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/plots/euclidean-distance/n10-euclidean_distances.csv')

# Create a 2x5 grid of subplots
fig, ax = plt.subplots(2, 5, figsize=(12, 5))  # 2 rows, 5 columns

# Manually assign DataFrames to specific subplots using seaborn.lineplot
sns.lineplot(data=df1, x='time', y='average_distance', ax=ax[0, 0], color='#4A6792')
ax[0, 0].set_title('X=1')
ax[0, 0].set_xlabel('Time', fontsize=8)
ax[0, 0].set_ylabel('Average Distance (mm)', fontsize=8)
ax[0, 0].set_ylim(0, 90)


sns.lineplot(data=df2, x='time', y='average_distance', ax=ax[0, 1], color='#ECB0C3')
ax[0, 1].set_title('X=2')
ax[0, 1].set_xlabel('Time', fontsize=8)
ax[0, 1].set_ylabel('Average Distance (mm)', fontsize=8)
ax[0, 1].set_ylim(0, 90)




sns.lineplot(data=df3, x='time', y='average_distance', ax=ax[0, 2], color='#629677')
ax[0, 2].set_title('X=3')
ax[0, 2].set_xlabel('Time', fontsize=8)
ax[0, 2].set_ylabel('Average Distance (mm)', fontsize=8)
ax[0, 2].set_ylim(0, 90)



sns.lineplot(data=df4, x='time', y='average_distance', ax=ax[0, 3], color='#F39B6D')
ax[0, 3].set_title('X=4')
ax[0, 3].set_xlabel('Time', fontsize=8)
ax[0, 3].set_ylabel('Average Distance (mm)', fontsize=8)
ax[0, 3].set_ylim(0, 90)



sns.lineplot(data=df5, x='time', y='average_distance', ax=ax[0, 4], color='#85C7DE')
ax[0, 4].set_title('X=5')
ax[0, 4].set_xlabel('Time', fontsize=8)
ax[0, 4].set_ylabel('Average Distance (mm)', fontsize=8)
ax[0, 4].set_ylim(0, 90)



sns.lineplot(data=df6, x='time', y='average_distance', ax=ax[1, 0], color='#7EBC66')
ax[1, 0].set_title('X=6')
ax[1, 0].set_xlabel('Time', fontsize=8)
ax[1, 0].set_ylabel('Average Distance (mm)', fontsize=8)
ax[1, 0].set_ylim(0, 90)


sns.lineplot(data=df7, x='time', y='average_distance', ax=ax[1, 1], color='#F2CD60')
ax[1, 1].set_title('X=7')
ax[1, 1].set_xlabel('Time', fontsize=8)
ax[1, 1].set_ylabel('Average Distance (mm)', fontsize=8)
ax[1, 1].set_ylim(0, 90)


sns.lineplot(data=df8, x='time', y='average_distance', ax=ax[1, 2], color='#7CB0B5')
ax[1, 2].set_title('X=8')
ax[1, 2].set_xlabel('Time', fontsize=8)
ax[1, 2].set_ylabel('Average Distance (mm)', fontsize=8)
ax[1, 2].set_ylim(0, 90)


sns.lineplot(data=df9, x='time', y='average_distance', ax=ax[1, 3], color='#F08080')
ax[1, 3].set_title('X=9')
ax[1, 3].set_xlabel('Time', fontsize=8)
ax[1, 3].set_ylabel('Average Distance (mm)', fontsize=8)
ax[1, 3].set_ylim(0, 90)


sns.lineplot(data=df10, x='time', y='average_distance', ax=ax[1, 4], color='#6F5E76')
ax[1, 4].set_title('X=10')
ax[1, 4].set_xlabel('Time', fontsize=8)
ax[1, 4].set_ylabel('Average Distance (mm)', fontsize=8)
ax[1, 4].set_ylim(0, 90)


# Add an overall title to the entire figure
fig.suptitle('Euclidean Distances', fontsize=16, fontweight='bold')

# Adjust layout to prevent overlap, considering the overall title
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/plots/euclidean-distance/euclidean-distance.png', dpi=300, bbox_inches='tight')


# Show the plot
plt.show()
