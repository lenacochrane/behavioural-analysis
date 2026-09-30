
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

# pseudo populations
df1_pseudo = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/interaction_type_bout.csv')
df1_pseudo['condition'] = 'GH'

df2_pseudo = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated/interaction_type_bout.csv')
df2_pseudo['condition'] = 'SI'

df_pseudo = pd.concat([df1_pseudo, df2_pseudo], ignore_index=True)


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
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/FIGURE-2/GROUP-ISO/number_bouts_.pdf', format='pdf', bbox_inches='tight')
plt.show()



###### NUMBER OF BOUTS NORMALISED TO PSEUDO
# bouts per file for the pseudo data
grouped_pseudo = (
    df_pseudo.groupby(['condition', 'file'])['bout_id']
      .size()
      .reset_index(name='num_bouts')
)

# mean pseudo frequency for each condition
pseudo_mean = grouped_pseudo.groupby('condition')['num_bouts'].mean()
print(pseudo_mean)

# divide each real file by its own condition's pseudo mean, log2 so 0 = same as pseudo
grouped['norm_bouts'] = np.log2(grouped['num_bouts'] / grouped['condition'].map(pseudo_mean))

gh_norm = grouped.loc[grouped['condition'] == 'GH', 'norm_bouts'].dropna()
si_norm = grouped.loc[grouped['condition'] == 'SI', 'norm_bouts'].dropna()

u_stat_norm, p_value_norm = mannwhitneyu(gh_norm, si_norm, alternative='two-sided')
print(f'Normalised: Mann–Whitney U = {u_stat_norm:.3f}, p = {p_value_norm:.4g}')

plt.figure(figsize=(2,4))
ax = sns.barplot(data=grouped, x='condition', y='norm_bouts', hue='condition', edgecolor='black', linewidth=2, errorbar='sd', palette=PALETTE, order=HUE_ORDER)
ax.axhline(0, color='black', linestyle='--', linewidth=1)

plt.xlabel('')
plt.ylabel('log2(Frequency / Pseudo mean)', fontsize=12, fontweight='bold')
sns.despine()
plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/FIGURE-2/GROUP-ISO/number_bouts_normalised.pdf', format='pdf', bbox_inches='tight')
plt.show()
