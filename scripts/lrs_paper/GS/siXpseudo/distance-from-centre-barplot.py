
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys  
import matplotlib as mpl


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']

PALETTE = {   
    "SI": 'darkorange',
     "PSEUDO": "#F7D455", }

HUE_ORDER = ["SI", "PSEUDO"]


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/distance_from_centre.csv')
df1['condition'] = 'SI'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated/distance_over_time.csv')
df2['condition'] = 'PSEUDO'

df = pd.concat([df1, df2], ignore_index=True)


plt.figure(figsize=(1.5,3))


ax = sns.barplot(data=df, x='condition', y='distance_from_centre', errorbar='sd', palette=PALETTE, order=HUE_ORDER, linewidth=2, edgecolor='black')


plt.xlabel('Distance From Centre (mm) ', fontsize=12)
plt.ylabel('Probability', fontsize=12)

sns.despine()
ax.legend(frameon=False, title=None, fontsize=11, loc="upper right")
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')

# plt.xlim(800,1000)

plt.ylim(0, None)
# Add an overall title to the entire figure
# plt.title('Distances from the Centre Distribution', fontsize=16, fontweight='bold')

# Adjust layout to prevent overlap, considering the overall title
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GS/siXpseudo/distance-from-centre-barplot.pdf', format='pdf', bbox_inches='tight')


# Show the plot
plt.show()
