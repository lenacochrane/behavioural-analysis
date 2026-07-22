
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
    "GH-SI": "#209961",} #228B22 #209961

HUE_ORDER = ["GH-GH","GH-SI", "SI-SI"]


def holm_correct(p_values):
    p_values = np.asarray(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    previous = 0

    for rank, index in enumerate(order):
        corrected = p_values[index] * (len(p_values) - rank)
        previous = max(previous, corrected)
        adjusted[index] = min(previous, 1)

    return adjusted


def print_repeated_measures_stats(data, value_col, label, fill_missing=None):
    if fill_missing is not None:
        data = (
            data
            .set_index(['file', 'social_experience'])
            .reindex(
                pd.MultiIndex.from_product(
                    [data['file'].unique(), HUE_ORDER],
                    names=['file', 'social_experience']
                ),
                fill_value=fill_missing
            )
            .reset_index()
        )

    wide = (
        data
        .pivot(index='file', columns='social_experience', values=value_col)
        .reindex(columns=HUE_ORDER)
        .dropna()
    )

    print(f'\n{label}')
    print(f'Videos included in paired stats: {len(wide)}')

    friedman_statistic, friedman_p_value = friedmanchisquare(
        wide["GH-GH"],
        wide["GH-SI"],
        wide["SI-SI"]
    )
    print(f'Friedman test: chi2 = {friedman_statistic:.4g}, p = {friedman_p_value:.4g}')

    results = []
    for group_a, group_b in combinations(HUE_ORDER, 2):
        values_a = wide[group_a]
        values_b = wide[group_b]

        if (values_a == values_b).all():
            statistic, p_value = 0, 1
        else:
            statistic, p_value = wilcoxon(values_a, values_b, alternative='two-sided')

        results.append({
            'comparison': f'{group_a} vs {group_b}',
            'n_a': len(values_a),
            'median_a': values_a.median(),
            'n_b': len(values_b),
            'median_b': values_b.median(),
            'W': statistic,
            'p_value': p_value,
        })

    stats_table = pd.DataFrame(results)
    stats_table['p_holm'] = holm_correct(stats_table['p_value'])

    print('Pairwise Wilcoxon signed-rank tests:')
    print(stats_table.to_string(index=False))

    return data



df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/grouped+isolated/interaction_type_bout.csv')


# classify each track
df["track_1_exp"] = np.where(df["track_1"] <= 4, "SI", "GH")
df["track_2_exp"] = np.where(df["track_2"] <= 4, "SI", "GH")

# make pair label
df["social_experience"] = df.apply(
    lambda row: "-".join(sorted([row["track_1_exp"], row["track_2_exp"]])),
    axis=1
)



###### RAW NUMBER OF BOUTS
plt.figure(figsize=(2,4))
grouped = (
    df.groupby(['social_experience', 'file'])['bout_id']
      .size()
      .reset_index(name='num_bouts')
)

grouped = print_repeated_measures_stats(
    grouped,
    value_col='num_bouts',
    label='Stats for number of bouts per video:',
    fill_missing=0
)

ax = sns.barplot(data=grouped, x='social_experience', y='num_bouts', hue='social_experience', edgecolor='black', linewidth=2, errorbar='sd', palette=PALETTE, hue_order=HUE_ORDER)

plt.xlabel('', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
sns.despine()
ax.legend(frameon=False, title=None, fontsize=11, loc="upper right")
# plt.title('Total Interaction Bouts', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[1, 1, 1, 1])
plt.ylim(0, 200)
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GS/GHnSI/number_bouts.pdf', format='pdf', bbox_inches='tight')
plt.close()


###### BOUT LENGTHS

length_bouts = df.groupby(['social_experience', 'file'])['duration'].mean().reset_index(name='length_bout')

length_bouts = print_repeated_measures_stats(
    length_bouts,
    value_col='length_bout',
    label='Stats for average bout length per video:'
)

plt.figure(figsize=(2,4))
ax = sns.barplot(data=length_bouts, x='social_experience', y='length_bout',  edgecolor='black', linewidth=2, errorbar='sd', palette=PALETTE, hue_order=HUE_ORDER)
plt.xlabel('', fontsize=12, fontweight='bold')
plt.ylabel('Average Bout Length (S)', fontsize=12, fontweight='bold')
# plt.title('Average Contact Bout Length', fontsize=16, fontweight='bold')
sns.despine()
ax.legend(frameon=False, title=None, fontsize=11, loc="upper right")
plt.tight_layout(rect=[1, 1, 1, 1])
plt.ylim(0, 8)
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GS/GHnSI/average_bout_length.pdf', format='pdf', bbox_inches='tight')
plt.close()









