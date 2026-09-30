
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']

PALETTE = {
    "GH": 'steelblue',     
    "PSEUDO": 'skyblue'}

HUE_ORDER = ["PSEUDO", "GH"]   # first entry draws on top; GH sits underneath


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/closest_contacts_1mm.csv')
df1['condition'] = 'GH'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/closest_contacts_1mm.csv')
df2['condition'] = 'PSEUDO'


df = pd.concat([df1, df2], ignore_index=True)

# Explode interaction pairs so each row is one larva (track)
pairs_long = (
    df[['condition', 'file', 'frame', 'Interaction Pair', 'Closest Interaction Type']]
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

# Sum across all frame bins per larva + interaction type
grouped = (
    pairs_long.groupby(['file', 'condition', 'track', 'Closest Interaction Type'])
    .size()
    .reset_index(name='count')
)

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

results = []

for interaction, sub in grouped.groupby("Closest Interaction Type"):
    gh = sub.loc[sub["condition"] == "GH", "count"]
    pseudo = sub.loc[sub["condition"] == "PSEUDO", "count"]

    # skip if one group is missing (safety)
    if gh.empty or pseudo.empty:
        continue

    u, p = mannwhitneyu(gh, pseudo, alternative="two-sided")

    results.append({
        "interaction_type": interaction,
        "n_GH": len(gh),
        "n_PSEUDO": len(pseudo),
        "u_stat": u,
        "p_raw": p,
        "median_GH": gh.median(),
        "median_PSEUDO": pseudo.median(),
        "delta_median": pseudo.median() - gh.median()
    })

stats_df = pd.DataFrame(results)

# --- multiple comparisons correction (FDR) ---
passed, p_corr, _, _ = multipletests(
    stats_df["p_raw"],
    alpha=0.05,
    method="fdr_bh"
)

stats_df["p_corrected"] = p_corr
stats_df["passes_multiple_test_correction"] = passed

print(stats_df)

# --- ADD THIS BLOCK RIGHT HERE ---
def p_to_stars(p):
    if p <= 1e-4:
        return "****"
    if p <= 1e-3:
        return "***"
    if p <= 1e-2:
        return "**"
    if p <= 5e-2:
        return "*"
    return ""

# IMPORTANT: use the SAME label source as your plot x-axis
star_map = dict(
    zip(stats_df["interaction_type"], stats_df["p_corrected"].apply(p_to_stars))
)



# order: least-occurring overall first (nearest y=0)
SWAP_PAIR = ('head_head', 'body_head')   # these two are swapped in the order

order = (
    grouped.groupby('Closest Interaction Type')['count']
    .sum()
    .sort_values()
    .index.tolist()
)

a, b = SWAP_PAIR
if a in order and b in order:
    i, j = order.index(a), order.index(b)
    order[i], order[j] = order[j], order[i]
else:
    print(f"WARNING: {SWAP_PAIR} not both present in {order}")

plt.figure(figsize=(8,10))
ax = sns.barplot(data=grouped, y='Closest Interaction Type', x='count', order=order, hue='condition', hue_order = HUE_ORDER, edgecolor='black', linewidth=2, errorbar='sd', palette=PALETTE, alpha=0.8)

plt.ylabel('Interaction Type', fontsize=12, fontweight='bold')
plt.xlabel('Total Contact Time (s)', fontsize=12, fontweight='bold')

sns.despine()
ax.legend(frameon=False, title=None, loc="lower right")

# plt.title('Interaction Type (Closest Node)', fontsize=16, fontweight='bold')

plt.tight_layout()

plt.xlim(0, 200)

# --- ADD STARS HERE (TRULY ROBUST) ---
# make a dict: category_label -> y_position (tick center)
tick_y = {t.get_text(): y for t, y in zip(ax.get_yticklabels(), ax.get_yticks())}

# collect bar centers + widths
bar_info = []
for p in ax.patches:
    y_center = p.get_y() + p.get_height() / 2
    bar_info.append((y_center, p.get_width()))

# for each category, find the two bars closest to its tick center, take max width, place star at tick center
for label, y in tick_y.items():
    stars = star_map.get(label, "")
    if not stars:
        continue

    # distance from each bar center to this category tick
    dists = [(abs(by - y), w) for by, w in bar_info]

    # take the TWO closest bars (GH + PSEUDO)
    dists.sort(key=lambda t: t[0])
    closest_two = dists[:2]

    # safety: if bars missing, skip
    if len(closest_two) == 0:
        continue

    max_width = max(w for _, w in closest_two)

    ax.text(
        max_width * 1.06,     # slightly right of longest bar in that category
        y,                    # ALWAYS centered on the category tick
        stars,
        ha="left",
        va="center",
        fontsize=14,
        fontweight="bold",
        zorder=10
    )








plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GS/ghXpseudo/interaction_type-closestnode_n10.pdf', 
            format='pdf', bbox_inches='tight')
plt.show()


