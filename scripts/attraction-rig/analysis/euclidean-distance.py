
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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



plt.figure(figsize=(8,6))


### N2
# sns.lineplot(data=df3, x='time', y='average_distance',  color='#629677', ci='sd', label='gh_n2')
# sns.lineplot(data=df4, x='time', y='average_distance',  color='#F39B6D', ci='sd', label='si_n2')

### N10
# sns.lineplot(data=df5, x='time', y='average_distance',  color='#85C7DE', ci='sd', label='gh_n10')
# sns.lineplot(data=df6, x='time', y='average_distance',  color='#7EBC66', ci='sd', label='si_n10')

### PSEUDO N2
# sns.lineplot(data=df7, x='frame', y='average_distance', color='#F2CD60', ci='sd', label='pseudo-gh-n2')
# sns.lineplot(data=df8, x='frame', y='average_distance', color='#7CB0B5', ci='sd', label='pseudo-si-n2')

### PSEUDO N10
# sns.lineplot(data=df9, x='frame', y='average_distance',  color='#F08080', ci='sd', label='pseudo-gh-n10')
# sns.lineplot(data=df10, x='frame', y='average_distance',  color='#6F5E76', ci='sd', label='pseudo-si-n10')


# plt.xlim(0,600)


plt.xlabel('Time (S)', fontsize=12, fontweight='bold')
plt.ylabel('Average Distance (mm)', fontsize=12, fontweight='bold')


plt.title('Euclidean Distances', fontsize=16, fontweight='bold')


plt.legend(title='Number of Larvae')

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/euclidean-distance/n2.png', dpi=300, bbox_inches='tight')


# Show the plot
plt.show()
