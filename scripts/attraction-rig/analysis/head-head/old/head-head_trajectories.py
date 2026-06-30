import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import matplotlib as mpl

# ---- Adobe-friendly fonts (must be set BEFORE plotting) ----
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/fed-fed/head_head_contacts_kinematics_over_time.csv')
df1['condition'] = 'fed-fed'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-fed/head_head_contacts_kinematics_over_time.csv')
df2['condition'] = 'fed-starved'

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-starved/head_head_contacts_kinematics_over_time.csv')
df3['condition'] = 'starved-starved'

df = pd.concat([df1, df2, df3], ignore_index=True)

df['heading_angle_change'] = (
    df
    .sort_values(['file', 'interaction_number', 'track_id', 'frame'])
    .groupby(['file', 'condition', 'interaction_number', 'track_id'])['heading_angle']
    .diff()
    .abs() )


df['interaction_group'] = np.where(df['interaction_number'] == 1, 'first', 'other')


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

PARTS = ["head", "body", "tail"]


def role_for_row(condition, track_id):
    # Adjust this mapping if your fed/starved track_ids are swapped in fed-starved files
    if condition == "fed-starved":
        return "fed" if track_id == 0 else "starved"
    if condition == "fed-fed":
        return "fed"
    if condition == "starved-starved":
        return "starved"
    return "unknown"

def cmaps_for_role(role):
    # temporal spectrum but stays “red family” for fed and “blue family” for starved
    return "Reds" if role == "fed" else "Blues"

def plot_condition_interactions(df, condition, out_png, out_pdf=None, max_rel_frame=10):
    sub = df[df["condition"] == condition].copy()

    # crop to 0..10 frames after interaction start
    sub = sub[(sub["rel_frame"] >= 0) & (sub["rel_frame"] <= max_rel_frame)].copy()

    # list interactions (each subplot)
    interactions = (
        sub[["file", "interaction_number", "interaction_group"]]
        .drop_duplicates()
        .sort_values(["file", "interaction_number"])
        .to_records(index=False)
    )

    n = len(interactions)
    if n == 0:
        print(f"No interactions found for {condition}")
        return

    # layout
    ncols = 4 if n >= 4 else n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), squeeze=False)
    axes = axes.ravel()

    for ax in axes[n:]:
        ax.axis("off")

    for i, (file, interaction_number, interaction_group) in enumerate(interactions):
        ax = axes[i]
        one = sub[(sub["file"] == file) & (sub["interaction_number"] == interaction_number)].copy()

        # ensure consistent time ordering
        one = one.sort_values(["track_id", "frame"])

        # title: filename + first/other
        fname = os.path.basename(str(file))
        # ax.set_title(f"{fname}\n{interaction_group} (#{interaction_number})", fontsize=10)

        title_color = "green" if interaction_group == "first" else "purple"
        ax.set_title(
            f"{fname}\n{interaction_group} (#{interaction_number})",
            fontsize=10,
            color=title_color
        )

        # # plot both larvae
        # for track_id in sorted(one["track_id"].unique()):
        #     tr = one[one["track_id"] == track_id].sort_values("frame")
        #     role = role_for_row(condition, track_id)
        #     color = "red" if role == "fed" else "blue"

        # plot both larvae
        for track_id in sorted(one["track_id"].unique()):
            tr = one[one["track_id"] == track_id].sort_values("frame")
            role = role_for_row(condition, track_id)

            # -------- TEMPORAL BODY LINE (UNDERLAY) --------
            tr = tr.sort_values("rel_frame")

            x = tr["x_head"].to_numpy()
            y = tr["y_head"].to_numpy()
            t = tr["rel_frame"].to_numpy()

            cmap = mpl.cm.Reds if role == "fed" else mpl.cm.Blues
            norm = mpl.colors.Normalize(vmin=0, vmax=max_rel_frame)

            if len(x) > 1:  # need at least 2 points to draw a line
                points = np.column_stack([x, y])
                segments = np.stack([points[:-1], points[1:]], axis=1)

                
                lc = LineCollection(
                    segments,
                    cmap=cmap,
                    norm=norm,
                    linewidths=5,
                    alpha=0.4,
                    zorder=1
                )
                lc.set_array(t[:-1])   # temporal colouring
                ax.add_collection(lc)
            # -------- END BODY LINE --------

            # -------- HEAD DOTS (ON TOP) --------
            ax.scatter(
                tr["x_head"], tr["y_head"],
                c=tr["rel_frame"],
                cmap=cmap,
                norm=norm,
                s=30,
                alpha=0.9,
                linewidths=0,
                zorder=5
            )


#             cmap = mpl.cm.Reds if role == "fed" else mpl.cm.Blues
#             norm = mpl.colors.Normalize(vmin=0, vmax=max_rel_frame)

#             ax.scatter(
#                 tr["x_head"], tr["y_head"],
#                 c=tr["rel_frame"], cmap=cmap, norm=norm,
#                 s=25, alpha=1, linewidths=0, zorder=3
# ) 
            
#             ax.scatter(
#                 tr["x_body"], tr["y_body"],
#                 c=tr["rel_frame"], cmap=cmap, norm=norm,
#                 s=25, alpha=0.8, linewidths=0, zorder=3
# )
            
#             ax.scatter(
#                 tr["x_tail"], tr["y_tail"],
#                 c=tr["rel_frame"], cmap=cmap, norm=norm,
#                 s=25, alpha=0.8, linewidths=0, zorder=3
# )


            # mark start (rel_frame==0) with a dot on body
            start = tr[tr["rel_frame"] == tr["rel_frame"].min()]
            if not start.empty:
                ax.scatter(start["x_head"], start["y_head"], s=40, marker="^",
                           color=("red" if role == "fed" else "blue"),
                           alpha=1, zorder=15)

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        # If your coordinates are image coords and look flipped vertically, uncomment:
        # ax.invert_yaxis()

    fig.suptitle(f"{condition}: head/body/tail trajectories (rel_frame 0–{max_rel_frame})", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    if out_pdf:
        plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)

# ---- call for each condition ----
plot_condition_interactions(
    df, "fed-fed",
    out_png="/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/traj/fed-fed.png",
    out_pdf="/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/traj/fed-fed.pdf",
    max_rel_frame=10
)

plot_condition_interactions(
    df, "fed-starved",
    out_png="/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/traj/fed-starved.png",
    out_pdf="/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/traj/fed-starved.pdf",
    max_rel_frame=10
)

plot_condition_interactions(
    df, "starved-starved",
    out_png="/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/traj/starved-starved.png",
    out_pdf="/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/traj/starved-starved.pdf",
    max_rel_frame=10
)






