
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/holes/N10-4-HOLE/GROUP-HOUSED/counts.csv')
df1['condition'] = 'GH'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/holes/N10-4-HOLE/SOCIAL-ISOLATION/counts.csv')
df2['condition'] = 'SI'

plt.figure(figsize=(8,8))

df = pd.concat([df1, df2], ignore_index=True)

df_total = (
    df.groupby(['condition', 'file', 'time'], as_index=False)['inside_count']
      .sum()
      .rename(columns={'inside_count': 'total_inside'})
)

sns.lineplot(data=df_total, x='time', y='total_inside', ci='sd', hue='condition')

plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
plt.ylabel('Total Hole Count', fontsize=12, fontweight='bold')

plt.ylim(0, 10)

plt.title('Number Inside Holes', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])


plt.xticks(fontweight='bold')

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes_multi/counts/hole-count.png', dpi=300, bbox_inches='tight')
plt.close()



df_mean_within_holes = (
    df.groupby(['condition', 'file', 'time'], as_index=False)['inside_count']
      .mean()
      .rename(columns={'inside_count': 'mean_inside_per_hole'})
)


plt.figure(figsize=(8,8))

sns.lineplot(
    data=df_mean_within_holes,
    x='time', y='mean_inside_per_hole',
    ci='sd', hue='condition'
)


plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
plt.ylabel('Mean Hole Count', fontsize=12, fontweight='bold')

plt.ylim(0, 10)

plt.title('Mean Number Inside Holes', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])


plt.xticks(fontweight='bold')

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes_multi/counts/hole-count-mean-per-hole.png', dpi=300, bbox_inches='tight')
plt.close()



df_spread = (
    df.groupby(['condition', 'file', 'time'])['inside_count']
      .agg(spread=lambda x: x.max() - x.min())
      .reset_index()
)
plt.figure(figsize=(8,8))

sns.lineplot(
    data=df_spread,
    x='time', y='spread',
    ci='sd', hue='condition'
)

plt.xlabel("Time (s)", fontsize=12, fontweight='bold')
plt.ylabel("Max − Min occupancy", fontsize=12, fontweight='bold')
plt.title("Hole occupancy spread (per frame)", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes_multi/counts/hole-count-spread-per-hole.png', dpi=300, bbox_inches='tight')
plt.close()





# tidy ordering
files = df['file'].unique()
hole_order = sorted(df['hole'].unique())   # ['hole1','hole2',...]

# colors
cond_colors = {'GH': 'tab:blue', 'SI': 'tab:orange'}
hole_colors = dict(zip(hole_order, sns.color_palette("tab10", n_colors=len(hole_order))))

# --- make a grid of subplots, one per file ---
n_files = len(files)
n_cols = 3                           # pick what looks nice
n_rows = int(np.ceil(n_files / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 3.5*n_rows), sharex=True, sharey=True)
axes = np.array(axes).ravel()

for ax, file in zip(axes, files):
    sub = df[df['file'] == file].sort_values('time')
    cond = sub['condition'].iloc[0]          # file has one condition
    title_color = cond_colors.get(cond, "black")

    # plot each hole as a line
    for hole_label in hole_order:
        hsub = sub[sub['hole'] == hole_label]
        if hsub.empty:
            continue
        ax.plot(
            hsub['time'], hsub['inside_count'],
            label=hole_label,
            color=hole_colors[hole_label],
            linewidth=2,
            alpha=0.6
        )

    ax.set_title(file, color=title_color, fontweight='bold')
    ax.set_ylim(0, 10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Inside count")

    ax.legend(fontsize=8, frameon=False)

# turn off empty axes if grid bigger than files
for ax in axes[len(files):]:
    ax.axis("off")

plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes_multi/counts/hole-count-per-hole.png', dpi=300, bbox_inches='tight')
plt.close()
