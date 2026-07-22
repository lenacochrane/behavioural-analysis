
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import matplotlib as mpl
from scipy.stats import mannwhitneyu

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']

PALETTE = {
    "GH": 'steelblue',     
    "SI": 'darkorange',}

HUE_ORDER = ["GH", "SI"]


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/interaction_type_bout.csv')
df1['condition'] = 'GH'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/interaction_type_bout.csv')
df2['condition'] = 'SI'


df = pd.concat([df1, df2], ignore_index=True)


###### RAW NUMBER OF BOUTS
plt.figure(figsize=(2,4))
grouped = (
    df.groupby(['condition', 'file'])['bout_id']
      .size()
      .reset_index(name='num_bouts')
)


# Extract one value per file for each condition
gh_bouts = grouped.loc[
    grouped['condition'] == 'GH',
    'num_bouts'
].dropna()

si_bouts = grouped.loc[
    grouped['condition'] == 'SI',
    'num_bouts'
].dropna()

# Two-sided Mann–Whitney U test
u_stat_bouts, p_value_bouts = mannwhitneyu(
    gh_bouts,
    si_bouts,
    alternative='two-sided'
)

print(f'GH files: n = {len(gh_bouts)}')
print(f'SI files: n = {len(si_bouts)}')
print(f'Mann–Whitney U = {u_stat_bouts:.3f}')
print(f'p = {p_value_bouts:.4g}')

ax = sns.barplot(data=grouped, x='condition', y='num_bouts', hue='condition', edgecolor='black', linewidth=2, errorbar='sd', palette=PALETTE,order=HUE_ORDER)

plt.xlabel('', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
sns.despine()
ax.legend(frameon=False, title=None, fontsize=11, loc="upper right")
# plt.title('Total Interaction Bouts', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[1, 1, 1, 1])
plt.ylim(0, None)
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GS/GHxSI/number_bouts_.pdf', format='pdf', bbox_inches='tight')
plt.show()


###### BOUT LENGTHS

length_bouts = df.groupby(['condition', 'file'])['duration'].mean().reset_index(name='length_bout')

# Extract one mean value per file for each condition
gh_lengths = length_bouts.loc[
    length_bouts['condition'] == 'GH',
    'length_bout'
].dropna()

si_lengths = length_bouts.loc[
    length_bouts['condition'] == 'SI',
    'length_bout'
].dropna()

# Two-sided Mann–Whitney U test
u_stat, p_value = mannwhitneyu(
    gh_lengths,
    si_lengths,
    alternative='two-sided'
)

print(f'GH files: n = {len(gh_lengths)}')
print(f'SI files: n = {len(si_lengths)}')
print(f'Mann–Whitney U = {u_stat:.3f}')
print(f'p = {p_value:.4g}')

plt.figure(figsize=(2,4))
ax = sns.barplot(data=length_bouts, x='condition', y='length_bout',  edgecolor='black', linewidth=2, errorbar='sd', palette=PALETTE,  order=HUE_ORDER)
plt.xlabel('', fontsize=12, fontweight='bold')
plt.ylabel('Average Bout Length (S)', fontsize=12, fontweight='bold')
# plt.title('Average Contact Bout Length', fontsize=16, fontweight='bold')
sns.despine()
ax.legend(frameon=False, title=None, fontsize=11, loc="upper right")
plt.tight_layout(rect=[1, 1, 1, 1])
plt.ylim(0, None)
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GS/GHxSI/average_bout_length_.pdf', format='pdf', bbox_inches='tight')
plt.show()










