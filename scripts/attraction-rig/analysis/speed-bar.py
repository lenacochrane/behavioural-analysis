
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches




# Load your data
df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed/speed_over_time.csv')
df1['condition'] = 'Group Housed'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/socially-isolated/speed_over_time.csv')
df2['condition'] = 'Socially Isolated'


# plt.figure(figsize=(8,6))

plt.figure(figsize=(8,8))

# custom_palette = {
#     'X=1': '#421B53',  
#     'X=2': '#4A2C77',
#     'X=3': '#3F4A8B',
#     'X=4': '#306990',
#     'X=5': '#25838F',
#     'X=6': '#149E89',
#     'X=7': '#35B579',
#     'X=8': '#75C15A',
#     'X=9': '#B4D336',
#     'X=10': '#FAE821'
# }

df = pd.concat([df1, df2], ignore_index=True)

# df = df[df['time'] < 601]

sns.barplot(data=df, x='condition', y='speed', edgecolor='black', linewidth=2, ci='sd')

plt.xlabel('', fontsize=12, fontweight='bold')
plt.ylabel('Speed (mm/s)', fontsize=12, fontweight='bold')

plt.ylim(0, 1.5)
# Add an overall title to the entire figure
plt.title('Speed', fontsize=16, fontweight='bold')

# Adjust layout to prevent overlap, considering the overall title
# plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.tight_layout(rect=[1, 1, 1, 1])



plt.xticks(fontweight='bold')


plt.savefig('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/analysis/speed/speed-barplot-3600.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
