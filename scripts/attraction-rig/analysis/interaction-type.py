
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches

df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed/interaction_types.csv')
df1['condition'] = 'Group Housed'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/socially-isolated/interaction_types.csv')
df2['condition'] = 'Socially Isolated'





# plt.figure(figsize=(8,6))

plt.figure(figsize=(8,8))


df = pd.concat([df1, df2], ignore_index=True)

sns.barplot(data=df, x='interaction_type', y='count', hue='condition', edgecolor='black', linewidth=2,ci=95)

plt.xlabel('Interaction Type', fontsize=12, fontweight='bold')
plt.ylabel('Interaction Count', fontsize=12, fontweight='bold')


# Add an overall title to the entire figure
plt.title('Interaction Type', fontsize=16, fontweight='bold')

# Adjust layout to prevent overlap, considering the overall title
# plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.tight_layout(rect=[1, 1, 1, 1])


plt.xticks(rotation=45)


plt.savefig('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/analysis/interactions/interaction-type.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
