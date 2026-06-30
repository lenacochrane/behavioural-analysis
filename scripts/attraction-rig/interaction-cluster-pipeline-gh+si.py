import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyarrow.feather as feather
import cv2
import re
from scipy.spatial.distance import pdist
from shapely import wkt
import glob
from random import sample
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
import ast


def create_cropped_interactions(df, wkt_dir, output_dir):

    df = pd.read_csv(df)    
   
    def pair_condition(pair):
        if isinstance(pair, str):
            pair = ast.literal_eval(pair)

        t1, t2 = pair

        def track_group(track):
            if 0 <= track <= 4:
                return "si"
            elif 5 <= track <= 9:
                return "gh"
            else:
                return None

        g1 = track_group(t1)
        g2 = track_group(t2)

        if g1 is None or g2 is None:
            return None

        return "-".join(sorted([g1, g2]))
    
    df["condition"] = df["Interaction Pair"].apply(pair_condition)


    df["interaction_id"] = df["condition"] + "_" + df["Interaction Number"].astype(str)  # Create unique interaction ID 

    def crop_interaction(group):
        if group.empty or "Normalized Frame" not in group.columns:
            return None
        center_idx = (group["Normalized Frame"].abs()).idxmin()
        if pd.isna(center_idx):
            return None
        center_pos = group.index.get_loc(center_idx)
        if center_pos < 15 or (center_pos + 16) >= len(group):
            return None
        cropped = group.iloc[center_pos - 15 : center_pos + 16].copy()
        cropped["interaction_id"] = group["interaction_id"].iloc[0]
        expected_frames = list(range(-15, 16))
        actual_frames = list(cropped["Normalized Frame"])
        if sorted(actual_frames) != expected_frames:
            return None
        return cropped

    df_cropped = df.groupby("interaction_id", group_keys=False).apply(crop_interaction) # crop interactions 15 frames either side 

    
    coordinate_columns = [
    "Track_1 x_body", "Track_1 y_body", "Track_2 x_body", "Track_2 y_body",
    "Track_1 x_tail", "Track_1 y_tail", "Track_2 x_tail", "Track_2 y_tail",
    "Track_1 x_head", "Track_1 y_head", "Track_2 x_head", "Track_2 y_head"
    ]
   
    for col in coordinate_columns:
        df_cropped[f'norm_{col}'] = df_cropped[col]

    for col in coordinate_columns: # un normalise 
        if "x_" in col:
            df_cropped[col] += df_cropped["Normalization mid_x"]
        elif "y_" in col:
            df_cropped[col] += df_cropped["Normalization mid_y"]

    for col in coordinate_columns:
        df_cropped[f'mm_{col}'] = df_cropped[col]


    df_cropped['file'] = df_cropped['file'].str.replace('.tracks.feather', '.mp4', regex=False)

    wkt_files = glob.glob(os.path.join(wkt_dir, '*_perimeter.wkt'))

    diameter_dict = {}

    for wkt_path in wkt_files:
        with open(wkt_path, 'r') as f:
            shape = wkt.loads(f.read().strip())

        # Extract coordinates
        coords = list(shape.exterior.coords)
        dists = pdist(coords)
        diameter = max(dists)

        # Extract base filename (without _perimeter.wkt)
        base_filename = os.path.basename(wkt_path).replace('_perimeter.wkt', '.mp4')
        diameter_dict[base_filename] = diameter

    # === 2. Apply scaling based on individual video diameters ===

    for idx, row in df_cropped.iterrows():
        video_file = row['file']
        
        # Special override for a known video
        if video_file == '2025-02-28_13-00-52_td9.mp4':
            scale = 1032 / 90
            print(f"🟡 Using fixed scale {scale:.3f} for {video_file}")
        
        elif video_file in diameter_dict:
            diameter_pixels = diameter_dict[video_file]
            scale = diameter_pixels / 90
        
        else:
            print(f"⚠️ Warning: No WKT file found for video {video_file}")
            continue  # skip scaling if diameter is missing

        for col in coordinate_columns:
            df_cropped.at[idx, col] *= scale


    df_cropped.to_csv(output_dir, index=False)



def apply_cluster_ids(cluster_csv, interaction_csv, output_csv):

    cluster_df = pd.read_csv(cluster_csv)
    interaction_df = pd.read_csv(interaction_csv)

    # Standardise filenames
    interaction_df["file"] = interaction_df["file"].astype(str).str.replace(".mp4", "", regex=False)
    cluster_df["file"] = cluster_df["file"].astype(str).str.replace(".tracks.feather", "", regex=False)

    # Standardise IDs
    interaction_df["Interaction Number"] = interaction_df["Interaction Number"].astype(str)
    cluster_df["interaction_id"] = cluster_df["interaction_id"].astype(str)

    # Make cluster mapping
    cluster_map = cluster_df[["file", "interaction_id", "Yhat.idt.pca"]].drop_duplicates()

    cluster_map = cluster_map.rename(columns={
        "interaction_id": "Interaction Number",
        "Yhat.idt.pca": "cluster"
    })

    # Merge cluster onto every frame of each interaction
    interaction_df = interaction_df.merge(
        cluster_map,
        on=["file", "Interaction Number"],
        how="left"
    )

    interaction_df.to_csv(output_csv, index=False)

    print("Saved:", output_csv)
    print("Rows with missing cluster:", interaction_df["cluster"].isna().sum())
    print("Unique interactions with missing cluster:",
          interaction_df.loc[interaction_df["cluster"].isna(), ["file", "Interaction Number"]]
          .drop_duplicates()
          .shape[0])
    


def anchor_partner(df):

    ## CREATE ALIGNED AND PARTNER TRACKS FOR DRAWING TRAJECTORIES 

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
        idx = group.index[:len(A_al)]
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
    

    ## APPROACH ANGLE CALCULATION FLIPPED INITIALLY BY ACCIDENT
    df["track1_approach_angle"] = 180 - df["track1_approach_angle"]
    df["track2_approach_angle"] = 180 - df["track2_approach_angle"]

    # === HEADING ANGLE CHANGE ===
    df['track1_heading_angle_change'] = df.groupby("interaction_id")["track1_angle"].diff().abs()
    df['track2_heading_angle_change'] = df.groupby("interaction_id")["track2_angle"].diff().abs()

    # === APPROACH ANGLE CHANGE ===
    df['track1_approach_angle_change'] = df.groupby("interaction_id")["track1_approach_angle"].diff().abs()
    df['track2_approach_angle_change'] = df.groupby("interaction_id")["track2_approach_angle"].diff().abs()

    metrics = [
    'speed',
    'acceleration',
    'angle',
    'approach_angle']

    for m in metrics:
        t1 = df[f'track1_{m}']
        t2 = df[f'track2_{m}']
        df[f'anchor_{m}']  = np.where(df['anchor_track']==1, t1, t2)
        df[f'partner_{m}'] = np.where(df['anchor_track']==1, t2, t1)

        # === Assign anchor/partner versions
        df['anchor_heading_angle_change']  = np.where(df['anchor_track'] == 1, df['track1_heading_angle_change'], df['track2_heading_angle_change'])
        df['partner_heading_angle_change'] = np.where(df['anchor_track'] == 1, df['track2_heading_angle_change'], df['track1_heading_angle_change'])

        df['anchor_approach_angle_change']  = np.where(df['anchor_track'] == 1, df['track1_approach_angle_change'], df['track2_approach_angle_change'])
        df['partner_approach_angle_change'] = np.where(df['anchor_track'] == 1, df['track2_approach_angle_change'], df['track1_approach_angle_change'])

    
    return df



def barplot(df, output):

    df = df.copy()  

    inter_per_video = (
    df[['file', 'condition', 'interaction_id', 'cluster']]
    .drop_duplicates(subset=['file', 'interaction_id'])
)

    # Count interactions per (video, condition, cluster)
    counts = (
        inter_per_video
        .groupby(['file', 'condition', "cluster"])
        .size()
        .reset_index(name='count')
    )

    totals = (
        counts.groupby('file')['count']
        .transform('sum')
    )

    # Add proportion column
    counts['proportion'] = counts['count'] / totals

    summary_df = (
            counts
            .set_index(['file', 'condition', "cluster"])
            .unstack(fill_value=0)
            .stack()
            .reset_index()
        )
    summary_df.rename(columns={0: 'proportion'}, inplace=True)

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=summary_df,
        x='cluster', y='proportion',ci='sd', alpha=0.8, edgecolor='black', linewidth=2
    )
    plt.title("Proportion of Clusters")
    plt.xlabel("Cluster")
    plt.ylabel("Proportion")
    # plt.xticks(rotation=90)
    sns.despine()
    plt.tight_layout()
    output_save = os.path.join(output, 'cluster_proportions.pdf')
    plt.savefig(output_save, format='pdf', bbox_inches='tight')
    plt.close()

    # Now you can plot with seaborn
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=summary_df,
        x='cluster', y='proportion', hue='condition',ci='sd', alpha=0.8, edgecolor='black', linewidth=2)

    plt.title("Proportion of Clusters")
    plt.xlabel("Cluster")
    plt.ylabel("Proportion")
    # plt.xticks(rotation=90)
    sns.despine()
    ax.legend(frameon=False, title=None, fontsize=11, loc="upper right")
    plt.tight_layout()

    output_save = os.path.join(output, 'cluster_proportions_per_condition.pdf')
    plt.savefig(output_save, format='pdf', bbox_inches='tight')
    plt.close()



def mean_traces(df, output):

    df = df.copy()

    coord_cols = [
        'anchor x_body', 'anchor y_body',
        'partner x_body', 'partner y_body',
        'anchor x_head', 'anchor y_head',
        'partner x_head', 'partner y_head'
    ]

    df = df.dropna(subset=['cluster', 'Normalized Frame'])

    mean_df = (
        df
        .groupby(['cluster', 'Normalized Frame'])[coord_cols]
        .mean()
        .reset_index()
        .sort_values(['cluster', 'Normalized Frame'])
    )

    clusters = sorted(mean_df['cluster'].dropna().unique())
    n_clusters = len(clusters)

    # shared limits
    x_cols = [c for c in coord_cols if ' x_' in c]
    y_cols = [c for c in coord_cols if ' y_' in c]

    x_min = mean_df[x_cols].min().min()
    x_max = mean_df[x_cols].max().max()
    y_min = mean_df[y_cols].min().min()
    y_max = mean_df[y_cols].max().max()

    pad_x = (x_max - x_min) * 0.1
    pad_y = (y_max - y_min) * 0.1

    fig, axes = plt.subplots(1, n_clusters, figsize=(4*n_clusters, 4))

    if n_clusters == 1:
        axes = [axes]

    for ax, cluster_id in zip(axes, clusters):

        group = mean_df[mean_df['cluster'] == cluster_id]

        # anchor
        ax.plot(group['anchor x_body'], group['anchor y_body'],
                linewidth=2)

        # partner
        ax.plot(group['partner x_body'], group['partner y_body'],
                linestyle='--', linewidth=2)
        
            # --- ADD HEADS ---
        ax.plot(group['anchor x_head'], group['anchor y_head'],
                linewidth=1, alpha=0.7)

        ax.plot(group['partner x_head'], group['partner y_head'],
                linestyle='--', linewidth=1, alpha=0.7)

        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.set_aspect('equal')

        # REMOVE EVERYTHING UGLY
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_title(f'{cluster_id}', fontsize=12)

    plt.subplots_adjust(wspace=0.3)

    output_save = os.path.join(output, 'mean_traces_clean_row.pdf')
    plt.savefig(output_save, format='pdf', bbox_inches='tight')
    plt.show()



def interaction_contact_counts(df, threshold=1.0):

    df = df.copy()

    # Boolean: frame is within threshold
    df['within_threshold'] = df['min_distance'] < threshold

    # Group and count
    summary = (
        df
        .groupby(['file', 'cluster', 'interaction_id'])['within_threshold']
        .sum()
        .reset_index()
        .rename(columns={'within_threshold': 'frames_below_1mm'})
    )

    plt.figure(figsize=(8,5))
    sns.barplot(
        data=summary,
        x='cluster',
        y='frames_below_1mm',
        errorbar='sd'
    )
    plt.ylabel("Frames < 1 mm (mean ± SD)")
    plt.xlabel("Cluster")
    sns.despine()
    plt.tight_layout()
    plt.show()










############ CREATE CROPPED INTERACTION DATAFRAME ############

# df = '/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/grouped+isolated/interactions.csv'
# wkt_dir = '/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/grouped+isolated/interaction-cluster/video-wkt-files'
# output_dir = '/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/grouped+isolated/interaction-cluster/cropped_interactions.csv'

# create_cropped_interactions(df, wkt_dir, output_dir)


############ ASSIGN CLUSTERS ############ 

# cluster_csv = '/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/grouped+isolated/interaction-cluster/pca-data2-F18-mcmodels4-Kmax2-dhat5.csv'
# interaction_csv = '/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/grouped+isolated/interaction-cluster/cropped_interactions.csv'
# output_csv = '/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/grouped+isolated/interaction-cluster/cropped_interaction_w_clusters.csv'

# apply_cluster_ids(cluster_csv, interaction_csv, output_csv)


############ CLUSTER PIPELINE ############

df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/grouped+isolated/interaction-cluster/cropped_interaction_w_clusters.csv')
output = '/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/grouped+isolated/interaction-cluster'
df = anchor_partner(df)
# barplot(df, output)
# mean_traces(df, output)
interaction_contact_counts(df, threshold=1.0)









