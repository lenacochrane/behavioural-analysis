
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import matplotlib as mpl


# ---- Adobe / Illustrator friendly PDFs ----
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# ---- Seaborn default "deep" colours ----
pal = sns.color_palette("deep", 3)

PALETTE = {
    "GH": pal[0],     # blue
    "SI": pal[1], # orange,
    "PSEUDO-GH": pal[2] # orange
}

HUE_ORDER = ["GH", "SI", "PSEUDO-GH",]


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
        ax=ax, palette=PALETTE, hue_order=HUE_ORDER
    )

    ax.set_title(f"Approach angle {angle_interval.left:.0f}–{angle_interval.right:.0f}°")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 2)
    ax.set_xlabel("Nearest neighbour distance")
    ax.set_ylabel("Speed")
    ax.axvline(x=1, color='gray', linestyle='--', linewidth=1)

# Add one shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', frameon=False)

fig.suptitle('Speed vs Nearest Neighbour Distance (Any Node) by Approach Angle Bin ', fontsize=16, fontweight='bold')
fig.tight_layout(rect=[0, 0, 0.95, 0.93])


plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/vision_involved/approach-angle-binned_speed_min-distance.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/vision_involved/approach-angle-binned_speed_min-distance.pdf', dpi=300, bbox_inches='tight')

plt.close()



######### MIN DISTANCE TO HEAD NODE ############
bins = list(range(0, 90, 1))  # [0, 10, 20, ..., 100]
df['bin'] = pd.cut(df['head_distance'], bins, include_lowest=True)
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
        ax=ax, palette=PALETTE, hue_order=HUE_ORDER
    )

    ax.set_title(f"Approach angle {angle_interval.left:.0f}–{angle_interval.right:.0f}°")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 2)
    ax.set_xlabel("Nearest neighbour distance")
    ax.set_ylabel("Speed")
    ax.axvline(x=1, color='gray', linestyle='--', linewidth=1)

# Add one shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', frameon=False)

fig.suptitle('Speed vs Nearest Neighbour Distance (From Head) by Approach Angle Bin', fontsize=16, fontweight='bold')
fig.tight_layout(rect=[0, 0, 0.95, 0.93])


plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/vision_involved/approach-angle-binned_speed_min-distance-to-head.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/vision_involved/approach-angle-binned_speed_min-distance-to-head.pdf', dpi=300, bbox_inches='tight')

plt.close()





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
        ax=ax, palette=PALETTE, hue_order=HUE_ORDER
    )

    ax.set_title(f"Approach angle {angle_interval.left:.0f}–{angle_interval.right:.0f}°")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 2)
    ax.set_xlabel("Nearest neighbour distance")
    ax.set_ylabel("Speed")

# Add one shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', frameon=False)

fig.suptitle('Speed vs Nearest Neighbour Distance (Body-Body) by Approach Angle Bin', fontsize=16, fontweight='bold')
fig.tight_layout(rect=[0, 0, 0.95, 0.93])


plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/vision_involved/approach-angle-binned_speed_body-body.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/vision_involved/approach-angle-binned_speed_body-body.pdf', dpi=300, bbox_inches='tight')

plt.close()