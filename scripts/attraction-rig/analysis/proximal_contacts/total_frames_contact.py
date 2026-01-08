import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/closest_contacts_1mm.csv')
df1['condition'] = 'GH'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/closest_contacts_1mm.csv')
df2['condition'] = 'SI'

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/closest_contacts_1mm.csv')
df3['condition'] = 'PSEUDO-GH'

df4 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated/closest_contacts_1mm.csv')
df4['condition'] = 'PSEUDO-SI'


palette = {
    'GH': "#0665c3", 
    'PSEUDO-GH': "#c27ba0",
    'SI': "#e38213",
    'PSEUDO-SI': "#7bc294"       
}

palette_order = ['GH', 'PSEUDO-GH', 'SI', 'PSEUDO-SI']


df = pd.concat([df1, df2, df3, df4], ignore_index=True)

grouped = (
    df.groupby(['file', 'condition'])
    .size()
    .reset_index(name='count')
)


plt.figure(figsize=(4,8))
sns.barplot(data=grouped, x='condition', y='count', edgecolor='black', linewidth=2, errorbar='sd', palette=palette, alpha=0.8, order=palette_order)

plt.xlabel('', fontsize=12, fontweight='bold')
plt.ylabel('Total Count', fontsize=12, fontweight='bold')

plt.title('Total Contacts', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.ylim(0, None)

plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/proximal_contacts/total_contact_frames.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/proximal_contacts/total_contact_frames.pdf', 
            format='pdf', bbox_inches='tight')

plt.close()

