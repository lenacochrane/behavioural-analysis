
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches



df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/nearest_neighbour.csv')
df1['condition'] = 'GH'

# df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/nearest_neighbour.csv')
# df2['condition'] = 'SI'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/nearest_neighbour.csv')
df2['condition'] = 'PSEUDO-GH'


df = pd.concat([df1, df2], ignore_index=True)

######### MIN DISTANCE ############
bins = list(range(0, 90, 1))  # [0, 10, 20, ..., 100]
df['bin'] = pd.cut(df['closest_node_distance'], bins, include_lowest=True)
df['bin_center'] = df['bin'].apply(lambda x: x.mid)


angle_edges = np.arange(0, 181, 30)  # 0,30,60,90,120,150,180
df['angle_bin'] = pd.cut(df['approach_angle'], angle_edges, include_lowest=True, right=False)


# df = df[df['closest_node_distance'] > 1]




# -------------------------
# Plot: 2 rows x 3 columns
# -------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
axes = axes.flatten()

for ax, angle_interval in zip(axes, df['angle_bin'].cat.categories):
    sub = df[df['angle_bin'] == angle_interval]
    if sub.empty:
        ax.set_title(f"{angle_interval.left:.0f}–{angle_interval.right:.0f}° (no data)")
        ax.axis('off')
        continue

    sns.lineplot(
        data=sub,
        x='bin_center',
        y='speed',
        hue='condition',
        errorbar=('ci', 95),
        ax=ax
    )

    ax.set_title(f"Approach angle {angle_interval.left:.0f}–{angle_interval.right:.0f}°")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 2)
    ax.set_xlabel("Nearest neighbour distance (binned)")
    ax.set_ylabel("Speed")

# Add one shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', frameon=False)

fig.suptitle('Speed vs Nearest Neighbour Distance by Approach Angle Bin', fontsize=16, fontweight='bold')
fig.tight_layout(rect=[0, 0, 0.95, 0.93])


plt.savefig('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/is_it_vision/nearest_neighour_min_distance/min_distance_approach_angle.png', dpi=300, bbox_inches='tight')

plt.show()



########## BODY-BODY DISTANCE ############
bins = list(range(0, 90, 1))  # [0, 10, 20, ..., 100]
df['bin'] = pd.cut(df['body-body'], bins, include_lowest=True)
df['bin_center'] = df['bin'].apply(lambda x: x.mid)


angle_edges = np.arange(0, 181, 30)  # 0,30,60,90,120,150,180
df['angle_bin'] = pd.cut(df['approach_angle'], angle_edges, include_lowest=True, right=False)


# -------------------------
# Plot: 2 rows x 3 columns
# -------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
axes = axes.flatten()

for ax, angle_interval in zip(axes, df['angle_bin'].cat.categories):
    sub = df[df['angle_bin'] == angle_interval]
    if sub.empty:
        ax.set_title(f"{angle_interval.left:.0f}–{angle_interval.right:.0f}° (no data)")
        ax.axis('off')
        continue

    sns.lineplot(
        data=sub,
        x='bin_center',
        y='speed',
        hue='condition',
        errorbar=('ci', 95),
        ax=ax
    )

    ax.set_title(f"Approach angle {angle_interval.left:.0f}–{angle_interval.right:.0f}°")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 2)
    ax.set_xlabel("Nearest neighbour distance (binned)")
    ax.set_ylabel("Speed")

# Add one shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', frameon=False)

fig.suptitle('Speed vs Nearest Neighbour Distance by Approach Angle Bin', fontsize=16, fontweight='bold')
fig.tight_layout(rect=[0, 0, 0.95, 0.93])


plt.savefig('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/is_it_vision/nearest_neighour_min_distance/body_body_distance_approach_angle.png', dpi=300, bbox_inches='tight')

plt.show()