import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import matplotlib as mpl
import ast
from sklearn.decomposition import PCA


df1 = pd.read_csv("/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/behaviour_detection.csv")
df1["condition"] = 'grouped'

df2 = pd.read_csv("/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/behaviour_detection.csv")
df2["condition"] = 'isolated'

df = pd.concat([df1, df2], ignore_index=True)  

interactions = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/clustered-interactions_pt2/cropped_interactions.csv')
cluster = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/clustered-interactions_pt2/F15_KMAX3_DMAX4_NMIN1500--/pca-data3-F15-mcmodels4-Kmax3-Dmax4-Nmin1500-05-2026.csv')


def anchor_partner(df):


        # make sure Interaction Pair is a real tuple
    df['Interaction Pair'] = df['Interaction Pair'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # because (left,right) = (Track_1_id, Track_2_id)
    df['track1_id'] = df['Interaction Pair'].str[0]
    df['track2_id'] = df['Interaction Pair'].str[1]

    # prep columns
    df['anchor_track_id'] = np.nan
    df['partner_track_id'] = np.nan

    ## Returns a straightness score 
    def compute_pca_axis(points):
        pca = PCA(n_components=2).fit(points)
        axis = pca.components_[0]
        score = pca.explained_variance_ratio_[0]
        # ensure the axis points upward
        return (axis if axis[1] >= 0 else -axis), score
    
    ## == Align the tracks (anchor 0,0) and rotate partner accordingly (on the right)
    def align_and_flip(track, anchor_axis, anchor_start):
        X = track - anchor_start
        phi = np.arctan2(anchor_axis[1], anchor_axis[0])  # angle of axis
        alpha = np.pi/2 - phi                            # rotate to +y
        R = np.array([[np.cos(alpha), -np.sin(alpha)],
                    [np.sin(alpha),  np.cos(alpha)]])
        X_rot = X.dot(R.T)
        return X_rot
    
    df['anchor x_body'] = np.nan
    df['anchor y_body'] = np.nan
    df['partner x_body'] = np.nan
    df['partner y_body'] = np.nan

    ## == Generate the anchor and partner x,y coordinates for future

    for interaction_id, group in df.groupby('interaction_id'):
        group = group.sort_values('Frame')
        coords1 = group[['Track_1 x_body','Track_1 y_body']].values
        coords2 = group[['Track_2 x_body','Track_2 y_body']].values
        if len(coords1) < 2 or len(coords2) < 2:
            continue
        # Compute PCA axes & scores
        axis1, s1 = compute_pca_axis(coords1)
        axis2, s2 = compute_pca_axis(coords2)
        # Choose anchor and partner
        if s1 >= s2:
            winner = 1
            anchor_pts, partner_pts, anchor_axis = coords1, coords2, axis1
        else:
            winner = 2
            anchor_pts, partner_pts, anchor_axis = coords2, coords1, axis2

        # Align both
        start = anchor_pts[0]
        A_al = align_and_flip(anchor_pts, anchor_axis, start)
        B_al = align_and_flip(partner_pts, anchor_axis, start)

        # --- NEW: align head/tail using the SAME reference (before flips) ---
        h1 = group[['Track_1 x_head','Track_1 y_head']].dropna().values
        t1 = group[['Track_1 x_tail','Track_1 y_tail']].dropna().values
        h2 = group[['Track_2 x_head','Track_2 y_head']].dropna().values
        t2 = group[['Track_2 x_tail','Track_2 y_tail']].dropna().values

        A_head = align_and_flip(h1 if winner == 1 else h2, anchor_axis, start) if (len(h1) or len(h2)) else np.empty((0,2))
        A_tail = align_and_flip(t1 if winner == 1 else t2, anchor_axis, start) if (len(t1) or len(t2)) else np.empty((0,2))
        B_head = align_and_flip(h2 if winner == 1 else h1, anchor_axis, start) if (len(h1) or len(h2)) else np.empty((0,2))
        B_tail = align_and_flip(t2 if winner == 1 else t1, anchor_axis, start) if (len(t1) or len(t2)) else np.empty((0,2))
    # --------------------------------------------------------------------

        # Horizontal flip if partner is left
        # if np.median(B_al[:,0]) < 0:
        #     A_al[:,0] *= -1
        #     B_al[:,0] *= -1

        # Horizontal flip if partner starts on the left
        if B_al[0, 0] < 0:
            A_al[:, 0] *= -1
            B_al[:, 0] *= -1
            # --- NEW: apply same horizontal flip to head/tail
            if A_head.size: A_head[:, 0] *= -1
            if A_tail.size: A_tail[:, 0] *= -1
            if B_head.size: B_head[:, 0] *= -1
            if B_tail.size: B_tail[:, 0] *= -1

        # Vertical flip if anchor is predominantly down
        if np.mean(A_al[:,1]) < 0:
            A_al[:,1] *= -1
            B_al[:,1] *= -1
                # --- NEW: apply same vertical flip to head/tail
            if A_head.size: A_head[:, 1] *= -1
            if A_tail.size: A_tail[:, 1] *= -1
            if B_head.size: B_head[:, 1] *= -1
            if B_tail.size: B_tail[:, 1] *= -1


        # Assign back to DataFrame
        # idx = group.index[:len(A_al)]

        idx = group.index  # safer- why 


        df.loc[idx, ['anchor x_body','anchor y_body']]  = A_al
        df.loc[idx, ['partner x_body','partner y_body']] = B_al# Initialize aligned columns

        # --- NEW: write aligned head/tail back (each uses its own length) ---
        if A_head.size:
            df.loc[group.index[:len(A_head)], ['anchor x_head','anchor y_head']] = A_head
        if A_tail.size:
            df.loc[group.index[:len(A_tail)], ['anchor x_tail','anchor y_tail']] = A_tail
        if B_head.size:
            df.loc[group.index[:len(B_head)], ['partner x_head','partner y_head']] = B_head
        if B_tail.size:
            df.loc[group.index[:len(B_tail)], ['partner x_tail','partner y_tail']] = B_tail
        # --------------------------------------------------------------------

        # → tag which original track was anchor (1 or 2)
        df.loc[idx, 'anchor_track']  = winner
        df.loc[idx, 'partner_track'] = 3 - winner

        # --- THIS is where the anchor/partner REAL ids go ---
        anchor_id  = group['track1_id'].iloc[0] if winner == 1 else group['track2_id'].iloc[0]
        partner_id = group['track2_id'].iloc[0] if winner == 1 else group['track1_id'].iloc[0]

        df.loc[idx, 'anchor_track_id']  = anchor_id
        df.loc[idx, 'partner_track_id'] = partner_id






    # === HEADING ANGLE CHANGE ===
    df['track1_heading_angle_change'] = df.groupby("interaction_id")["track1_angle"].diff().abs()
    df['track2_heading_angle_change'] = df.groupby("interaction_id")["track2_angle"].diff().abs()

    # === APPROACH ANGLE CHANGE ===
    df['track1_approach_angle_change'] = df.groupby("interaction_id")["track1_approach_angle"].diff().abs()
    df['track2_approach_angle_change'] = df.groupby("interaction_id")["track2_approach_angle"].diff().abs()

    # metrics = [
    # 'speed',
    # 'acceleration',
    # 'angle',
    # 'approach_angle']

    # for m in metrics:
    #     t1 = df[f'track1_{m}']
    #     t2 = df[f'track2_{m}']
    #     df[f'anchor_{m}']  = np.where(df['anchor_track']==1, t1, t2)
    #     df[f'partner_{m}'] = np.where(df['anchor_track']==1, t2, t1)

    #     # === Assign anchor/partner versions
    #     df['anchor_heading_angle_change']  = np.where(df['anchor_track'] == 1, df['track1_heading_angle_change'], df['track2_heading_angle_change'])
    #     df['partner_heading_angle_change'] = np.where(df['anchor_track'] == 1, df['track2_heading_angle_change'], df['track1_heading_angle_change'])

    #     df['anchor_approach_angle_change']  = np.where(df['anchor_track'] == 1, df['track1_approach_angle_change'], df['track2_approach_angle_change'])
    #     df['partner_approach_angle_change'] = np.where(df['anchor_track'] == 1, df['track2_approach_angle_change'], df['track1_approach_angle_change'])

    
    return df




interactions = pd.merge(
        interactions,
        cluster[['interaction_id', 'Yhat.idt.pca']],
        on='interaction_id',
        how='inner'
    )


interactions = anchor_partner(interactions)
interactions['video_id'] = interactions['file'].str.replace('.mp4', '', regex=False)


base_cols = ['file', 'video_id', 'Frame', 'Interaction Number',
                'Normalized Frame', 'Yhat.idt.pca']

anchor_df = interactions[base_cols].copy()
anchor_df['role'] = 'anchor'
anchor_df['track'] = interactions['anchor_track_id'].values

partner_df = interactions[base_cols].copy()
partner_df['role'] = 'partner'
partner_df['track'] = interactions['partner_track_id'].values

interaction_tracks = pd.concat([anchor_df, partner_df], ignore_index=True)
interaction_tracks = interaction_tracks.sort_values(
    ['file', 'Interaction Number', 'Normalized Frame', 'role']
).reset_index(drop=True)

interaction_tracks["track_id"] = interaction_tracks["track"].astype("Int64")
interaction_tracks["frame"] = interaction_tracks["Frame"]
interaction_tracks["file"] = interaction_tracks["video_id"]

interaction_tracks = interaction_tracks.merge(
    df[["file", "track_id", "frame", "behaviour"]],
    on=["file", "track_id", "frame"],
    how="left"
)




import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap



save_dir = "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/clustered-interactions_pt2/F15_KMAX3_DMAX4_NMIN1500--/etho"
os.makedirs(save_dir, exist_ok=True)

plot_df = interaction_tracks.copy()

# keep only partner rows
plot_df = plot_df[plot_df["role"] == "partner"].copy()

# drop missing values needed for plotting
plot_df = plot_df.dropna(subset=["behaviour", "Yhat.idt.pca", "Normalized Frame", "track_id"])

# unique row id per partner-track within interaction
plot_df["instance_id"] = (
    plot_df["file"].astype(str) + "_int" +
    plot_df["Interaction Number"].astype(str) + "_partner_track" +
    plot_df["track_id"].astype(str)
)

plot_df["Normalized Frame"] = plot_df["Normalized Frame"].astype(int)

behaviour_order = [
    "stationary",
    "forward_run",
    "backward",
    "small_turn",
    "turn",
    "sharp_turn",
    "steering",
    "casting",
    "digging"
]

extra_behaviours = sorted(
    [b for b in plot_df["behaviour"].dropna().unique() if b not in behaviour_order]
)
behaviour_order = behaviour_order + extra_behaviours

behaviour_to_code = {b: i for i, b in enumerate(behaviour_order)}
plot_df["behaviour_code"] = plot_df["behaviour"].map(behaviour_to_code)

base_colors = (
    list(plt.cm.tab20.colors) +
    list(plt.cm.Set3.colors) +
    list(plt.cm.Dark2.colors)
)
colors = base_colors[:len(behaviour_order)]
cmap = ListedColormap(colors)

for cluster_id in sorted(plot_df["Yhat.idt.pca"].dropna().unique()):
    cluster_df = plot_df[plot_df["Yhat.idt.pca"] == cluster_id].copy()

    if cluster_df.empty:
        continue

    cluster_df = cluster_df.sort_values(
        ["file", "Interaction Number", "track_id", "Normalized Frame"]
    )

    ethogram = cluster_df.pivot_table(
        index="instance_id",
        columns="Normalized Frame",
        values="behaviour_code",
        aggfunc="first"
    )

    if ethogram.empty:
        continue

    ethogram = ethogram.reindex(sorted(ethogram.columns), axis=1)

    fig_height = max(4, ethogram.shape[0] * 0.18)
    fig_width = max(8, ethogram.shape[1] * 0.35)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    masked = np.ma.masked_invalid(ethogram.to_numpy())
    ax.imshow(
        masked,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=-0.5,
        vmax=len(behaviour_order) - 0.5
    )

    ax.set_title(f"Cluster {cluster_id}: partner behaviour ethogram")
    ax.set_xlabel("Normalized Frame")
    ax.set_ylabel("Partner track")

    ax.set_xticks(np.arange(len(ethogram.columns)))
    ax.set_xticklabels(ethogram.columns)

    if ethogram.shape[0] <= 40:
        ax.set_yticks(np.arange(len(ethogram.index)))
        ax.set_yticklabels(ethogram.index, fontsize=6)
    else:
        ax.set_yticks([])

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

    save_path = os.path.join(save_dir, f"cluster_{cluster_id}_ethogram_partner.pdf")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")