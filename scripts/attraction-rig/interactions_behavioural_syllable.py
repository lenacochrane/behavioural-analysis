import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import matplotlib as mpl
import ast


df1 = pd.read_csv("/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/behaviour_detection.csv")
df1["condition"] = 'grouped'

df2 = pd.read_csv("/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/behaviour_detection.csv")
df2["condition"] = 'isolated'

df = pd.concat([df1, df2], ignore_index=True)  

interactions = pd.read_csv('/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/plots/interactions/cropped_interactions.csv')
interactions['video_id'] = interactions['file'].str.replace('.mp4', '', regex=False)
cluster = pd.read_csv('/Users/cochral/Desktop/MOSEQ2/KEYPOINT-KAPPA1000/plots/interactions/pca-data2-F18.csv')

interactions = pd.merge(
        interactions,
        cluster[['interaction_id', 'Yhat.idt.pca']],
        on='interaction_id',
        how='inner'
    )

keep = ['file', 'video_id', 'Frame', 'Interaction Number', 'Normalized Frame', 'Interaction Pair', 'Yhat.idt.pca']

interactions['Interaction Pair'] = interactions['Interaction Pair'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

interaction_tracks = (
    interactions[keep]
    .assign(track=interactions['Interaction Pair'])
    .explode('track')
    .drop(columns='Interaction Pair')
    .reset_index(drop=True)
)

interaction_tracks["track_id"] = interaction_tracks["track"].astype(int)
interaction_tracks["frame"] = interaction_tracks["Frame"]
interaction_tracks["file"] = interaction_tracks["video_id"]


interaction_tracks = interaction_tracks.merge(
    df[["file", "track_id", "frame", "behaviour"]],
    on=["file", "track_id", "frame"],
    how="left"
)



import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# =============================================================================
# save ethogram per cluster
# =============================================================================

save_dir = "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/behavioural-detection-testing/track_videos/interaction"
os.makedirs(save_dir, exist_ok=True)

plot_df = interaction_tracks.copy()

# drop rows where behaviour is missing
plot_df = plot_df.dropna(subset=["behaviour", "Yhat.idt.pca", "Normalized Frame"])

# make a unique row ID per larva-within-interaction
plot_df["instance_id"] = (
    plot_df["file"].astype(str) + "_int" +
    plot_df["Interaction Number"].astype(str) + "_track" +
    plot_df["track_id"].astype(str)
)

# make sure frame is integer
plot_df["Normalized Frame"] = plot_df["Normalized Frame"].astype(int)

# consistent behaviour order
behaviour_order = [
    "stationary",
    "forward_run",
    "steering",
    "backward",
    "small_turn",
    "turn",
    "sharp_turn",
    "casting",
    "digging"
]

# include any extra labels not listed above
extra_behaviours = sorted([b for b in plot_df["behaviour"].dropna().unique() if b not in behaviour_order])
behaviour_order = behaviour_order + extra_behaviours

behaviour_to_code = {b: i for i, b in enumerate(behaviour_order)}
plot_df["behaviour_code"] = plot_df["behaviour"].map(behaviour_to_code)

# choose enough distinct colours
base_colors = list(plt.cm.tab20.colors) + list(plt.cm.Set3.colors) + list(plt.cm.Dark2.colors)
colors = base_colors[:len(behaviour_order)]
cmap = ListedColormap(colors)

for cluster_id in sorted(plot_df["Yhat.idt.pca"].dropna().unique()):
    cluster_df = plot_df[plot_df["Yhat.idt.pca"] == cluster_id].copy()

    if cluster_df.empty:
        continue

    # sort instances nicely
    cluster_df = cluster_df.sort_values(
        ["file", "Interaction Number", "track_id", "Normalized Frame"]
    )

    # pivot to ethogram matrix
    ethogram = cluster_df.pivot_table(
        index="instance_id",
        columns="Normalized Frame",
        values="behaviour_code",
        aggfunc="first"
    )

    if ethogram.empty:
        continue

    # sort columns just in case
    ethogram = ethogram.reindex(sorted(ethogram.columns), axis=1)

    # figure size scales with number of rows
    fig_height = max(4, ethogram.shape[0] * 0.18)
    fig_width = max(8, ethogram.shape[1] * 0.35)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    masked = np.ma.masked_invalid(ethogram.to_numpy())
    im = ax.imshow(
        masked,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=-0.5,
        vmax=len(behaviour_order) - 0.5
    )

    ax.set_title(f"Cluster {cluster_id}: behaviour ethogram")
    ax.set_xlabel("Normalized Frame")
    ax.set_ylabel("Interaction-track instance")

    ax.set_xticks(np.arange(len(ethogram.columns)))
    ax.set_xticklabels(ethogram.columns, rotation=0)

    # optional: hide row labels if too many
    if ethogram.shape[0] <= 40:
        ax.set_yticks(np.arange(len(ethogram.index)))
        ax.set_yticklabels(ethogram.index, fontsize=6)
    else:
        ax.set_yticks([])

    # legend
    legend_handles = [
        mpatches.Patch(color=colors[i], label=behaviour)
        for i, behaviour in enumerate(behaviour_order)
    ]
    ax.legend(
        handles=legend_handles,
        title="Behaviour",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0
    )

    plt.tight_layout()

    save_path = os.path.join(save_dir, f"cluster_{cluster_id}_ethogram.pdf")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")





