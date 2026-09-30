
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

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed-pseudo/closest_contacts_1mm.csv')
df2['condition'] = 'PSEUDO'


df = pd.concat([df1, df2], ignore_index=True)

# Count contact frames per video + interaction type, then turn into a proportion per video
counts = (
    df.groupby(['condition', 'file', 'Closest Interaction Type'])
    .size()
    .unstack(fill_value=0)          # videos x interaction types, 0 where a type never happened
)

proportions = counts.div(counts.sum(axis=1), axis=0)   # each video's row sums to 1

grouped = (
    proportions
    .stack()
    .reset_index(name='proportion')
)

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

results = []

for interaction, sub in grouped.groupby("Closest Interaction Type"):
    gh = sub.loc[sub["condition"] == "GH", "proportion"]
    pseudo = sub.loc[sub["condition"] == "PSEUDO", "proportion"]

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
    grouped.groupby('Closest Interaction Type')['proportion']
    .mean()
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
ax = sns.barplot(data=grouped, y='Closest Interaction Type', x='proportion', order=order, hue='condition', hue_order = HUE_ORDER, edgecolor='black', linewidth=2, errorbar='sd', palette=PALETTE, alpha=0.8)

plt.ylabel('Interaction Type', fontsize=12, fontweight='bold')
plt.xlabel('Proportion of Contacts per Video', fontsize=12, fontweight='bold')

sns.despine()
ax.legend(frameon=False, title=None, loc="lower right")

# plt.title('Interaction Type (Closest Node)', fontsize=16, fontweight='bold')

plt.tight_layout()

plt.xlim(0, 0.5)

# --- ADD STARS HERE ---
tick_y = {t.get_text(): y for t, y in zip(ax.get_yticklabels(), ax.get_yticks())}

bar_ends = (
    grouped.groupby(['Closest Interaction Type', 'condition'])['proportion']
    .agg(lambda v: v.mean() + v.std())
    .groupby('Closest Interaction Type')
    .max()
)

for label, y in tick_y.items():
    stars = star_map.get(label, "")
    if not stars:
        continue

    ax.text(
        bar_ends[label] + 0.02,
        y,
        stars,
        ha="left",
        va="center",
        fontsize=14,
        fontweight="bold",
        zorder=10
    )


plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/FIGURE-1/ghXpseudo/interaction_type-closestnode_proportion_n10.pdf', 
            format='pdf', bbox_inches='tight')
plt.show()




# ===================== NORMALISED PROPORTION (skew GH vs PSEUDO) =====================
# skew = (GH - PSEUDO) / (GH + PSEUDO), using mean proportion per interaction type
# +1 = only in GH, -1 = only in PSEUDO, 0 = equal

means = grouped.groupby(['Closest Interaction Type', 'condition'])['proportion'].mean().unstack()
skew = ((means['GH'] - means['PSEUDO']) / (means['GH'] + means['PSEUDO'])).reindex(order)

colours = [PALETTE['GH'] if v > 0 else PALETTE['PSEUDO'] for v in skew]

plt.figure(figsize=(8, 5))
ax = plt.gca()
ax.bar(skew.index, skew.values, color=colours, edgecolor='black', linewidth=2, alpha=0.8)
ax.axhline(0, color='black', linewidth=1)

plt.xlabel('Interaction Type', fontsize=12, fontweight='bold')
plt.ylabel('Normalised Proportion\n(GH - PSEUDO) / (GH + PSEUDO)', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')

lim = max(abs(skew).max() * 1.3, 0.1)
plt.ylim(-lim, lim)

for i, (label, v) in enumerate(skew.items()):
    stars = star_map.get(label, "")
    if stars:
        ax.text(i, v + (0.02 if v > 0 else -0.02) * lim / 0.5, stars,
                ha='center', va='bottom' if v > 0 else 'top',
                fontsize=14, fontweight='bold')

ax.legend(handles=[mpatches.Patch(color=PALETTE['GH'], label='GH'),
                   mpatches.Patch(color=PALETTE['PSEUDO'], label='PSEUDO')],
          frameon=False)

sns.despine()
plt.tight_layout()

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/FIGURE-1/ghXpseudo/interaction_type-closestnode_normalisedproportion_n10.pdf',
            format='pdf', bbox_inches='tight')
plt.show()
