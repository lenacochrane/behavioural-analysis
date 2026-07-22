import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import matplotlib as mpl
from itertools import combinations
from scipy.stats import friedmanchisquare, wilcoxon

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']


PALETTE = {
    "GH-GH": 'steelblue',     
    "SI-SI": 'darkorange',
    "G-S": "#209961",} #228B22 #209961

HUE_ORDER = ["GH-GH","G-S", "SI-SI"]



df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/grouped+isolated/closest_contacts_1mm.csv')


def social_experience_from_pair(pair):
    tracks = (
        str(pair)
        .replace('(', '')
        .replace(')', '')
        .replace('[', '')
        .replace(']', '')
        .replace(' ', '')
        .split(',')
    )
    tracks = [int(track) for track in tracks]
    experiences = ["SI" if track <= 4 else "GH" for track in tracks]

    if experiences[0] == experiences[1]:
        return f"{experiences[0]}-{experiences[1]}"
    return "G-S"


df["social_experience"] = df["Interaction Pair"].apply(social_experience_from_pair)


pairs_long = (
    df[['file', 'frame', 'Interaction Pair']]
    .assign(
        track=lambda d: (
            d['Interaction Pair']
            .astype(str)
            .str.replace(r'[\(\)\[\]\s]', '', regex=True)   # remove (), [], spaces
            .str.split(',')                                 # -> ['0','1']
        )
    )
    .explode('track')
)

pairs_long['track'] = pairs_long['track'].astype(int)

pairs_long["social_experience"] = np.where(pairs_long["track"] <= 4, "SI", "GH")


grouped = (
    df.groupby(['file', 'social_experience'])
    .size()
    .reset_index(name='count')
)

social_experience_order = ["SI-SI", "G-S", "GH-GH"]

grouped = (
    grouped
    .set_index(['file', 'social_experience'])
    .reindex(
        pd.MultiIndex.from_product(
            [grouped['file'].unique(), social_experience_order],
            names=['file', 'social_experience']
        ),
        fill_value=0
    )
    .reset_index()
)

grouped_wide = (
    grouped
    .pivot(index='file', columns='social_experience', values='count')
    .loc[:, social_experience_order]
)

friedman_statistic, friedman_p_value = friedmanchisquare(
    grouped_wide["SI-SI"],
    grouped_wide["G-S"],
    grouped_wide["GH-GH"]
)

print('\nFriedman test on total contact time per video:')
print(f'chi2 = {friedman_statistic:.4g}, p = {friedman_p_value:.4g}')

wilcoxon_results = []
for group_a, group_b in combinations(social_experience_order, 2):
    values_a = grouped_wide[group_a]
    values_b = grouped_wide[group_b]
    if (values_a == values_b).all():
        statistic, p_value = 0, 1
    else:
        statistic, p_value = wilcoxon(values_a, values_b, alternative='two-sided')
    wilcoxon_results.append({
        'comparison': f'{group_a} vs {group_b}',
        'n_a': len(values_a),
        'median_a': values_a.median(),
        'n_b': len(values_b),
        'median_b': values_b.median(),
        'W': statistic,
        'p_value': p_value,
    })

wilcoxon_table = pd.DataFrame(wilcoxon_results).sort_values('p_value')
wilcoxon_table['p_holm'] = (
    wilcoxon_table['p_value']
    .mul(len(wilcoxon_table) - np.arange(len(wilcoxon_table)))
    .cummax()
    .clip(upper=1)
)
wilcoxon_table = wilcoxon_table.sort_index()

print('\nPairwise Wilcoxon signed-rank tests on total contact time per video:')
print(wilcoxon_table.to_string(index=False))

plt.figure(figsize=(4,8))
ax = sns.barplot(data=grouped, x='social_experience', y='count', order=social_experience_order, linewidth=2, errorbar='sd', edgecolor='black', palette=PALETTE, hue_order=HUE_ORDER)

plt.xlabel('', fontsize=12, fontweight='bold')
plt.ylabel('Total Contact Time (s)', fontsize=12, fontweight='bold')


sns.despine()
ax.legend(frameon=False, title=None, fontsize=11, loc="upper right")

plt.tight_layout(rect=[1, 1, 1, 1])

plt.ylim(0, 750)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GS/GHnSI/total_contact_frames.pdf', 
            format='pdf', bbox_inches='tight')

plt.close()
