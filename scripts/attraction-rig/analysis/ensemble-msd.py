
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

### N1
df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n1/group-housed/ensemble_msd.csv')
df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n1/socially-isolated/ensemble_msd.csv')

### N2
df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/group-housed/ensemble_msd.csv')
df4 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/socially-isolated/ensemble_msd.csv')

### N10
df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed/ensemble_msd.csv')
df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/socially-isolated/ensemble_msd.csv')

### PSEUDO N2
df7 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/group-housed/ensemble_msd.csv')
df8 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/socially-isolated/ensemble_msd.csv')

### PSEUDO N10
df9 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/ensemble_msd.csv')
df10 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated/ensemble_msd.csv')

plt.figure(figsize=(8,6))

### N1
# sns.lineplot(data=df1, x='time', y='squared_distance',  color='#4A6792', ci=None, label='gh_n1')
# sns.lineplot(data=df2, x='time', y='squared_distance', color='#ECB0C3', ci=None, label='si_n1')

### N2
# sns.lineplot(data=df3, x='time', y='squared_distance',  color='#629677', ci=None, label='gh_n2')
# sns.lineplot(data=df4, x='time', y='squared_distance',  color='#F39B6D', ci=None, label='si_n2')

### N10
sns.lineplot(data=df5, x='time', y='squared_distance',  color='#85C7DE', ci=None, label='gh_n10')
sns.lineplot(data=df6, x='time', y='squared_distance',  color='#7EBC66', ci=None, label='si_n10')

### PSEUDO N2
# sns.lineplot(data=df7, x='time', y='squared_distance', color='#F2CD60', ci=None, label='pseudo-gh-n2')
# sns.lineplot(data=df8, x='time', y='squared_distance', color='#7CB0B5', ci=None, label='pseudo-si-n2')

### PSEUDO N10
sns.lineplot(data=df9, x='time', y='squared_distance',  color='#F08080', ci=None, label='pseudo-gh-n10')
sns.lineplot(data=df10, x='time', y='squared_distance',  color='#6F5E76', ci=None, label='pseudo-si-n10')


plt.xlabel('Time (S)', fontsize=12)
plt.ylabel('Ensemble Mean Squared Distance (mm)', fontsize=12)

# plt.ylim(0, 90)
plt.xlim(0,600)
# plt.yscale('log')
# plt.xscale('log')


plt.title('Ensemble Mean Squared Distance', fontsize=16, fontweight='bold')


plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/ensemble-msd/pseudo-n10-600FRAMES.png', dpi=300, bbox_inches='tight')


plt.show()
