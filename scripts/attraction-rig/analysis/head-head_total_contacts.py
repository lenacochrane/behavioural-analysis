import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches


###### INTERACTION TYPE - ALL NODE-NODE PER FRAME CONTACTS ######

df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/fed-fed/closest_contacts_1mm.csv')
df1['condition'] = 'fed-fed'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-fed/closest_contacts_1mm.csv')
df2['condition'] = 'fed-starved'

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-starved/closest_contacts_1mm.csv')
df3['condition'] = 'starved-starved'


plt.figure(figsize=(4,8))

df = pd.concat([df1, df2, df3], ignore_index=True)

print(df3.columns)

# grouped = df.groupby(['condition', 'file']).size().reset_index(name='total_contacts')

# print(grouped[grouped['condition'] == 'fed-starved'])

# print(grouped[grouped['condition'] == 'fed-fed'])

### some files had 0 and wasnt being included 


# 1) Build the "all files per condition" list from the raw CSVs (so missing ones still exist)
all_files = (
    df[['condition', 'file']]
    .drop_duplicates()
)

# 2) Count REAL contact frames only (ignore placeholder rows where frame is NaN)
counts = (
    df.dropna(subset=['frame'])
      .groupby(['condition', 'file'])
      .size()
      .reset_index(name='total_contacts')
)

# 3) Reindex so every (condition, file) exists; fill missing with 0
grouped = (
    all_files
      .merge(counts, on=['condition', 'file'], how='left')
      .fillna({'total_contacts': 0})
)

# ensure it's integer
grouped['total_contacts'] = grouped['total_contacts'].astype(int)



sns.violinplot(data=grouped, x='condition', y='total_contacts', edgecolor='black', linewidth=2, inner='quartile', alpha=0.8)

plt.xlabel('Condition', fontsize=12, fontweight='bold')
plt.ylabel('Interaction Count', fontsize=12, fontweight='bold')


plt.title('Total Contact Frames', fontsize=16, fontweight='bold')


plt.tight_layout(rect=[1, 1, 1, 1])

plt.ylim(0, None)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/total_contact_frames.png', dpi=300, bbox_inches='tight')

plt.show()