import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import mannwhitneyu

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']

PALETTE = {
    "GH": 'steelblue',
    "GH Pseudo": 'lightsteelblue',
    "SI": 'darkorange',
    "SI Pseudo": '#F7D455'}

BASE = '/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation'
PLOTS = '/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/FIGURE-2/GROUP-ISO'


def contact_frames_per_video(path):
    """Total contact frames per video (one row = one pair in contact on one frame)."""
    df = pd.read_csv(path)
    return df.groupby('file').size()


gh = contact_frames_per_video(f'{BASE}/n10/group-housed/closest_contacts_1mm.csv')
gh_pseudo = contact_frames_per_video(f'{BASE}/pseudo-n10/group-housed/closest_contacts_1mm.csv')

si = contact_frames_per_video(f'{BASE}/n10/socially-isolated/closest_contacts_1mm.csv')
si_pseudo = contact_frames_per_video(f'{BASE}/pseudo-n10/socially-isolated/closest_contacts_1mm.csv')


def p_to_stars(p):
    if p <= 1e-4:
        return "****"
    if p <= 1e-3:
        return "***"
    if p <= 1e-2:
        return "**"
    if p <= 5e-2:
        return "*"
    return "ns"


def compare(a, b, name):
    u, p = mannwhitneyu(a, b, alternative='two-sided')
    print(f'{name}: n={len(a)} vs n={len(b)}, U = {u:.3f}, p = {p:.4e}')
    return p


def add_bracket(ax, x1, x2, y, text):
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.015
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='black')
    ax.text((x1 + x2) / 2, y + h, text, ha='center', va='bottom', fontsize=14, fontweight='bold')


def bar_with_points(ax, data, order):
    sns.barplot(data=data, x='condition', y='value', order=order, palette=PALETTE,
                errorbar='sd', edgecolor='black', linewidth=2, alpha=0.8, capsize=0.2,
                err_kws={'linewidth': 1.5, 'color': 'black'}, ax=ax)
    sns.stripplot(data=data, x='condition', y='value', order=order, color='black',
                  size=5, jitter=0.15, alpha=0.7, ax=ax)
    ax.set_xlabel('')
    sns.despine(ax=ax)


# ---------- 1) each condition vs its own pseudo ----------
p_gh = compare(gh, gh_pseudo, 'GH vs GH Pseudo')
p_si = compare(si, si_pseudo, 'SI vs SI Pseudo')

raw = pd.concat([
    pd.DataFrame({'condition': 'GH', 'value': gh.values}),
    pd.DataFrame({'condition': 'GH Pseudo', 'value': gh_pseudo.values}),
    pd.DataFrame({'condition': 'SI', 'value': si.values}),
    pd.DataFrame({'condition': 'SI Pseudo', 'value': si_pseudo.values}),
], ignore_index=True)

fig, ax = plt.subplots(figsize=(5, 6))
bar_with_points(ax, raw, ['GH', 'GH Pseudo', 'SI', 'SI Pseudo'])
ax.set_ylabel('Total Contact Frames per Video', fontsize=12, fontweight='bold')

top = raw['value'].max()
ax.set_ylim(0, top * 1.2)
add_bracket(ax, 0, 1, top * 1.05, p_to_stars(p_gh))
add_bracket(ax, 2, 3, top * 1.05, p_to_stars(p_si))

plt.tight_layout()
plt.savefig(f'{PLOTS}/total_contact_frames_vs_pseudo.pdf', format='pdf', bbox_inches='tight')
plt.show()


# ---------- 2) log2 fold change over own pseudo mean: GH vs SI ----------
# 0 = same as pseudo, >0 = more contact than pseudo, <0 = less
gh_fold = np.log2(gh / gh_pseudo.mean())
si_fold = np.log2(si / si_pseudo.mean())

p_fold = compare(gh_fold, si_fold, 'GH fold vs SI fold')

fold = pd.concat([
    pd.DataFrame({'condition': 'GH', 'value': gh_fold.values}),
    pd.DataFrame({'condition': 'SI', 'value': si_fold.values}),
], ignore_index=True)

fig, ax = plt.subplots(figsize=(3, 6))
bar_with_points(ax, fold, ['GH', 'SI'])
ax.axhline(0, color='black', linewidth=1.5, zorder=0)
ax.set_ylabel('Total Contact Time (log2 fold change vs pseudo)', fontsize=12, fontweight='bold')

top = max(fold['value'].max(), 0)
bottom = min(fold['value'].min(), 0)
span = top - bottom
ax.set_ylim(bottom - span * 0.05, top + span * 0.2)
add_bracket(ax, 0, 1, top + span * 0.05, p_to_stars(p_fold))

plt.tight_layout()
plt.savefig(f'{PLOTS}/total_contact_frames_fold-change_GHvSI.pdf', format='pdf', bbox_inches='tight')
plt.show()
