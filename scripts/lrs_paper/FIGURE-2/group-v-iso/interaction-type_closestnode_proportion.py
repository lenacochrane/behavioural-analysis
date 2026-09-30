
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']

PALETTE = {
    "GH": 'steelblue',
    "SI": 'darkorange'}

HUE_ORDER = ["GH", "SI"]

BASE = '/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation'


def proportions_per_video(path):
    """Proportion of contact frames of each interaction type, per video (each video sums to 1)."""
    df = pd.read_csv(path)
    counts = (
        df.groupby(['file', 'Closest Interaction Type'])
        .size()
        .unstack(fill_value=0)          # 0 where a type never happened in that video
    )
    return counts.div(counts.sum(axis=1), axis=0)


gh = proportions_per_video(f'{BASE}/n10/group-housed/closest_contacts_1mm.csv')
gh_pseudo = proportions_per_video(f'{BASE}/pseudo-n10/group-housed/closest_contacts_1mm.csv')

si = proportions_per_video(f'{BASE}/n10/socially-isolated/closest_contacts_1mm.csv')
si_pseudo = proportions_per_video(f'{BASE}/pseudo-n10/socially-isolated/closest_contacts_1mm.csv')

# make sure every table has every interaction type
types = sorted(set(gh.columns) | set(gh_pseudo.columns) | set(si.columns) | set(si_pseudo.columns))
gh, gh_pseudo, si, si_pseudo = [t.reindex(columns=types, fill_value=0) for t in (gh, gh_pseudo, si, si_pseudo)]

# each real video minus its own pseudo mean (per interaction type): >0 = more than chance, <0 = less
gh_diff = gh - gh_pseudo.mean()
si_diff = si - si_pseudo.mean()

grouped = pd.concat([
    gh_diff.stack().reset_index(name='difference').assign(condition='GH'),
    si_diff.stack().reset_index(name='difference').assign(condition='SI'),
], ignore_index=True)


# --- stats: GH vs SI on the pseudo-subtracted proportions ---
results = []

for interaction, sub in grouped.groupby("Closest Interaction Type"):
    g = sub.loc[sub["condition"] == "GH", "difference"]
    s = sub.loc[sub["condition"] == "SI", "difference"]

    u, p = mannwhitneyu(g, s, alternative="two-sided")

    results.append({
        "interaction_type": interaction,
        "n_GH": len(g),
        "n_SI": len(s),
        "u_stat": u,
        "p_raw": p,
        "median_GH": g.median(),
        "median_SI": s.median(),
        "delta_median": s.median() - g.median()
    })

stats_df = pd.DataFrame(results)

passed, p_corr, _, _ = multipletests(stats_df["p_raw"], alpha=0.05, method="fdr_bh")
stats_df["p_corrected"] = p_corr
stats_df["passes_multiple_test_correction"] = passed

print(stats_df)


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

star_map = dict(zip(stats_df["interaction_type"], stats_df["p_corrected"].apply(p_to_stars)))


# --- plot ---
plt.figure(figsize=(12,8))
ax = sns.barplot(data=grouped, x='Closest Interaction Type', y='difference', hue='condition', hue_order=HUE_ORDER, edgecolor='black', linewidth=2, errorbar='sd', palette=PALETTE, alpha=0.8)

ax.axhline(0, color='black', linewidth=1.5)

plt.xlabel('Interaction Type', fontsize=12, fontweight='bold')
plt.ylabel('Proportion − Pseudo Mean', fontsize=12, fontweight='bold')

sns.despine()
ax.legend(frameon=False, title=None, loc="upper right")

plt.xticks(rotation=45)

# stars above the highest point (mean + sd) in each category
tick_x = {t.get_text(): x for t, x in zip(ax.get_xticklabels(), ax.get_xticks())}

tops = (
    grouped.groupby(['Closest Interaction Type', 'condition'])['difference']
    .agg(lambda v: max(v.mean() + v.std(), 0))
    .groupby('Closest Interaction Type')
    .max()
)

for label, x in tick_x.items():
    stars = star_map.get(label, "")
    if not stars:
        continue

    ax.text(x, tops[label] + 0.01, stars, ha="center", va="bottom", fontsize=14, fontweight="bold", zorder=10)

plt.tight_layout()

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/FIGURE-2/GROUP-ISO/interaction_type-closestnode_proportion-minus-pseudo_n10.pdf',
            format='pdf', bbox_inches='tight')
plt.show()
