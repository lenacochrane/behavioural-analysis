
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches

df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/interaction_type_bout.csv')
df1['condition'] = 'GH'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/interaction_type_bout.csv')
df2['condition'] = 'SI'


palette = {
    'SI': '#fca35d',  # yellow-orange
    'GH': '#0d75b9'        # blue-green
}

palette_order = ['GH', 'SI']

df = pd.concat([df1, df2], ignore_index=True)


###### BOUT LENGTHS

length_bouts = df.groupby(['condition', 'file'])['duration'].mean().reset_index(name='length_bout')

plt.figure(figsize=(6,4))
sns.barplot(data=length_bouts, x='condition', y='length_bout',  edgecolor='black', linewidth=2, errorbar='sd', palette=palette, alpha=0.8, order=palette_order)
plt.xlabel('', fontsize=12, fontweight='bold')
plt.ylabel('Average Bout Length (S)', fontsize=12, fontweight='bold')
plt.title('Average Contact Bout Length', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[1, 1, 1, 1])
plt.ylim(0, None)
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/proximal_contacts/average_bout_length_n10.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/proximal_contacts/average_bout_length_n10.pdf', format='pdf', bbox_inches='tight')
plt.close()


###### INITIAL TYPE PER BOUT
plt.figure(figsize=(6,4))
grouped = (
    df.groupby(['file', 'condition', 'initial_type'])
    .size()
    .reset_index(name='count')
)

sns.barplot(data=grouped, x='initial_type', y='count', hue='condition', edgecolor='black', linewidth=2, errorbar='sd', palette=palette, alpha=0.8)

plt.xlabel('Initial Interaction Type', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')
plt.title('Initial Interaction Type Per Bout', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[1, 1, 1, 1])
plt.ylim(0, None)
plt.xticks(rotation=45)
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/proximal_contacts/initial_type_per_bout.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/proximal_contacts/initial_type_per_bout.pdf', format='pdf', bbox_inches='tight')
plt.close()

###### INITIAL TYPE PER BOUT DURATION

plt.figure(figsize=(6,4))

grouped = df.groupby(['file', 'condition', 'initial_type'])['duration'].mean().reset_index()

sns.barplot(data=grouped, x='initial_type', y='duration', hue='condition', edgecolor='black', linewidth=2, errorbar='sd', palette=palette, alpha=0.8)

plt.xlabel('Interaction Type Duration', fontsize=12, fontweight='bold')
plt.ylabel('duration', fontsize=12, fontweight='bold')
plt.title('Interaction Type Duration', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[1, 1, 1, 1])

plt.ylim(0, None)
plt.xticks(rotation=45)
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/proximal_contacts/initial_type_per_bout_duration.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/proximal_contacts/initial_type_per_bout_duration.pdf', format='pdf', bbox_inches='tight')
plt.close()


###### PREDOMINANT TYPE PER BOUT

plt.figure(figsize=(6,4))
grouped = (
    df.groupby(['file', 'condition', 'predominant_type'])
    .size()
    .reset_index(name='count'))

sns.barplot(data=grouped, x='predominant_type', y='count', hue='condition', edgecolor='black', linewidth=2, errorbar='sd', palette=palette, alpha=0.8)

plt.xlabel('Predominant Interaction Type per Bout', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')
plt.title('Predominant Interaction Type', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[1, 1, 1, 1])
plt.ylim(0, None)
plt.xticks(rotation=45)
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/proximal_contacts/predominant_type_per_bout.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/proximal_contacts/predominant_type_per_bout.pdf', format='pdf', bbox_inches='tight') 
plt.close()

###### PREDOMINANT TYPE PER BOUT DURATION

plt.figure(figsize=(6,4))
grouped = df.groupby(['file', 'condition', 'predominant_type'])['duration'].mean().reset_index()
sns.barplot(data=grouped, x='predominant_type', y='duration', hue='condition', edgecolor='black', linewidth=2, errorbar='sd', palette=palette, alpha=0.8)
plt.xlabel('Predominant Interaction Type', fontsize=12, fontweight='bold')
plt.ylabel('Duration', fontsize=12, fontweight='bold')
plt.title('Predominant Interaction Type Duration', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[1, 1, 1, 1])
plt.ylim(0, None)
plt.xticks(rotation=45)
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/proximal_contacts/predominant_type_per_bout_duration.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/proximal_contacts/predominant_type_per_bout_duration.pdf', format='pdf', bbox_inches='tight')
plt.close()

###### RAW NUMBER OF BOUTS
plt.figure(figsize=(6,4))
grouped = (
    df.groupby(['condition', 'file'])['bout_id']
      .size()
      .reset_index(name='num_bouts')
)

sns.barplot(data=grouped, x='condition', y='num_bouts', hue='condition', edgecolor='black', linewidth=2, errorbar='sd', palette=palette, alpha=0.8)

plt.xlabel('', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')
plt.title('Total Interaction Bouts', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[1, 1, 1, 1])
plt.ylim(0, None)
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/proximal_contacts/number_bouts.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/proximal_contacts/number_bouts.pdf', format='pdf', bbox_inches='tight')
plt.close()










