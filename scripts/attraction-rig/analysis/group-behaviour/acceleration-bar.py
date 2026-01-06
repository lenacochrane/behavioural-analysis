
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n1/group-housed/acceleration_accross_time.csv')
df1['condition'] = 'GH_N1'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n1/socially-isolated/acceleration_accross_time.csv')
df2['condition'] = 'SI_N1'

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/group-housed/acceleration_accross_time.csv')
df3['condition'] = 'GH_N2'

df4 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/socially-isolated/acceleration_accross_time.csv')
df4['condition'] = 'SI_N2'

df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed/acceleration_accross_time.csv')
df5['condition'] = 'GH_N10'

df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/socially-isolated/acceleration_accross_time.csv')
df6['condition'] = 'SI_N10'

df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed/acceleration_accross_time.csv')
df5['condition'] = 'GH_N10'

df7 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated/acceleration_accross_time.csv')
df7['condition'] = 'PSEUDO-SI_N10'

df8 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/acceleration_accross_time.csv')
df8['condition'] = 'PSEUDO-GH_N10'

df9 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/socially-isolated/acceleration_accross_time.csv')
df9['condition'] = 'PSEUDO-SI_N2'

df10 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/group-housed/acceleration_accross_time.csv')
df10['condition'] = 'PSEUDO-GH_N2'



plt.figure(figsize=(8,6))


df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10], ignore_index=True)


sns.barplot(data=df, x='condition', y='acceleration',  alpha=0.8, edgecolor='black', linewidth=2)

plt.xlabel('Condition', fontsize=12)
plt.ylabel('Acceleration (mm^2/s)', fontsize=12)


# Add an overall title to the entire figure
plt.title('Acceleration Distribution', fontsize=16, fontweight='bold')

# Adjust layout to prevent overlap, considering the overall title
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/acceleration/acceleration-barplot.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
