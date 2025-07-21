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



def create_cropped_interactions(group_csv, iso_csv, wkt_dir, output_dir):

    df_group = pd.read_csv(group_csv)
    df_group['condition'] = 'group'

    df_iso = pd.read_csv(iso_csv)
    df_iso['condition'] = 'iso'

    df = pd.concat([df_iso, df_group], ignore_index=True)

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

    standard_pixel_diameter = 1032  # pixel diameter of the standard 90mm dish

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




def modify_cluster_ids(file, output):

    df = pd.read_csv(file)

    df["condition"] = df["behavior"].map({"grouped": "group", "isolated": "iso"})

    df['interaction_id'] = df["condition"] + "_" + df["interaction_id"].astype(str)

    df.to_csv(output, index=False)


def cluster_pipeline(cropped_interaction, cluster, cluster_name, video_path, output_dir):
 
    df_cropped_interaction = pd.read_csv(cropped_interaction)
    df_cluster = pd.read_csv(cluster) # edit youngser's cluster csv 

    df = pd.merge(
        df_cropped_interaction, 
        df_cluster[['interaction_id', cluster_name]], 
        on='interaction_id', 
        how='inner'
    )

    ### 1. GENERATE GRID VIDEOS 

    frames_per_clip = 30
    dot_radius = 3
    dot_thickness = -1  # Filled
    fps = 3
    crop_size = 400
    half_crop = crop_size // 2

    # === TRACK VALID CLIPS ===
    cluster_to_interactions = {}

    for cluster_id in sorted(df[cluster_name].unique()):
        cluster_df = df[df[cluster_name] == cluster_id]
        unique_ids = cluster_df['interaction_id'].unique()

        if len(unique_ids) < 9:
            print(f"⚠️ Skipping cluster {cluster_id} (only {len(unique_ids)} interactions)")
            continue

        chosen_ids = sample(list(unique_ids), 9)
        interaction_clips = []
        final_ids = []

        for inter_id in chosen_ids:
            inter_df = cluster_df[cluster_df['interaction_id'] == inter_id].sort_values("Frame")
            start_frame = inter_df["Frame"].iloc[0]
            end_frame = start_frame + frames_per_clip
            clip_df = inter_df[(inter_df["Frame"] >= start_frame) & (inter_df["Frame"] < end_frame)]

            if len(clip_df) < frames_per_clip:
                continue

            video_file = inter_df['file'].iloc[0]
            print(f"📼 Interaction {inter_id} comes from file {video_file}")
            full_video_path = os.path.join(video_path, video_file)

            if not os.path.exists(full_video_path):
                print(f"⚠️ Missing video: {full_video_path}")
                continue

            # Use midpoint at Normalized Frame = 0
            center_frame = inter_df[inter_df["Normalized Frame"] == 0]
            if center_frame.empty:
                print(f"⚠️ No center frame for {inter_id}")
                continue

            row_center = center_frame.iloc[0]
            cx = int((row_center['Track_1 x_body'] + row_center['Track_2 x_body']) / 2)
            cy = int((row_center['Track_1 y_body'] + row_center['Track_2 y_body']) / 2)

            # Crop logic
            def safe_crop(frame, cx, cy, crop_size):
                h, w = frame.shape[:2]
                half = crop_size // 2
                x_start, y_start = cx - half, cy - half
                x_end, y_end = cx + half, cy + half
                cropped = np.zeros((crop_size, crop_size, 3), dtype=frame.dtype)

                x1, y1 = max(0, x_start), max(0, y_start)
                x2, y2 = min(w, x_end), min(h, y_end)
                dx1, dy1 = x1 - x_start, y1 - y_start
                dx2, dy2 = dx1 + (x2 - x1), dy1 + (y2 - y1)
                cropped[dy1:dy2, dx1:dx2] = frame[y1:y2, x1:x2]
                return cropped

            # Read and annotate frames
            cap = cv2.VideoCapture(full_video_path)
            clip_frames = []
            for _, row in clip_df.iterrows():
                frame_idx = int(row['Frame'])
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                x1, y1 = int(row['Track_1 x_body']), int(row['Track_1 y_body'])
                x2, y2 = int(row['Track_2 x_body']), int(row['Track_2 y_body'])

                cv2.circle(frame, (x1, y1), dot_radius, (0, 0, 255), dot_thickness)
                cv2.circle(frame, (x2, y2), dot_radius, (255, 0, 0), dot_thickness)

                cropped = safe_crop(frame, cx, cy, crop_size)
                clip_frames.append(cropped)

            cap.release()

            if len(clip_frames) == frames_per_clip:
                interaction_clips.append(clip_frames)
                final_ids.append(inter_id)

        if len(interaction_clips) < 9:
            print(f"⚠️ Not enough good clips for cluster {cluster_id}")
            continue

        cluster_to_interactions[cluster_id] = final_ids

        # === Create grid video ===
        h, w = interaction_clips[0][0].shape[:2]
        grid_frames = []
        for i in range(frames_per_clip):
            row1 = np.hstack([interaction_clips[j][i] for j in range(0, 3)])
            row2 = np.hstack([interaction_clips[j][i] for j in range(3, 6)])
            row3 = np.hstack([interaction_clips[j][i] for j in range(6, 9)])
            grid_frame = np.vstack([row1, row2, row3])
            grid_frames.append(grid_frame)

        grid_video_dir = os.path.join(output_dir, "grid_videos")
        os.makedirs(grid_video_dir, exist_ok=True)

        output_path = os.path.join(grid_video_dir, f"cluster_{cluster_id}.mp4")

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w * 3, h * 3))

        for frame in grid_frames:
            out.write(frame)
        out.release()

        print(f"✅ Saved grid video for cluster {cluster_id} → {output_path}")

    # === SAVE CSV OF INTERACTIONS INCLUDED IN GRID VIDEOS ===
    mapping_records = []
    for cluster, interactions in cluster_to_interactions.items():
        for inter_id in interactions:
            video_file = df[df['interaction_id'] == inter_id]['file'].iloc[0]
            mapping_records.append({
                'cluster': cluster,
                'interaction_id': inter_id,
                'video_file': video_file
            })

    mapping_df = pd.DataFrame(mapping_records)
    mapping_path = os.path.join(grid_video_dir, 'gridmovies_interactions.csv')
    mapping_df.to_csv(mapping_path, index=False)


    ### 2. MEAN TRAJECTORIES

    for cluster_id in sorted(df[cluster_name].unique()):
        cluster_df = df[df[cluster_name] == cluster_id]
        grouped = cluster_df.groupby("Normalized Frame")

        t1_x = grouped["norm_Track_1 x_body"].mean()
        t1_y = grouped["norm_Track_1 y_body"].mean()
        t2_x = grouped["norm_Track_2 x_body"].mean()
        t2_y = grouped["norm_Track_2 y_body"].mean()

        plt.figure(figsize=(6, 6))
        plt.plot(t1_x, t1_y, label="Track 1", color="red")
        plt.plot(t2_x, t2_y, label="Track 2", color="blue")
        plt.scatter(t1_x.iloc[0], t1_y.iloc[0], color="red", marker="o", label="T1 Start")
        plt.scatter(t2_x.iloc[0], t2_y.iloc[0], color="blue", marker="o", label="T2 Start")
        plt.gca().invert_yaxis()
        plt.axis("equal")
        plt.title(f"Mean Trajectory - Cluster {cluster_id}")
        plt.legend()
        plt.tight_layout()

        trajectory_dir = os.path.join(output_dir, "mean_trajectories")
        os.makedirs(trajectory_dir, exist_ok=True)

        save_path = os.path.join(trajectory_dir, f"cluster_{cluster_id}.png")

        plt.savefig(save_path)
        plt.close()



    ##### 3. RAW TRAJECTORIES

    for cluster_id in sorted(df[cluster_name].unique()):
        cluster_df = df[df[cluster_name] == cluster_id]
        interaction_ids = cluster_df['interaction_id'].unique()
        np.random.shuffle(interaction_ids)
        sample_ids = interaction_ids[:81]  # max 9x9

        n = int(np.ceil(np.sqrt(len(sample_ids))))
        fig, axes = plt.subplots(n, n, figsize=(n * 2, n * 2))
        axes = axes.flatten()

        for ax, interaction_id in zip(axes, sample_ids):
            traj = cluster_df[cluster_df['interaction_id'] == interaction_id]
            ax.plot(traj["Track_1 x_body"], traj["Track_1 y_body"], color="red")
            ax.plot(traj["Track_2 x_body"], traj["Track_2 y_body"], color="blue")
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(str(interaction_id), fontsize=6)

        for i in range(len(sample_ids), len(axes)):
            axes[i].axis('off')

        plt.suptitle(f"Raw Trajectories - Cluster {cluster_id}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        raw_path = os.path.join(output_dir, "raw_trajectories")
        os.makedirs(raw_path, exist_ok=True)
        save_path = os.path.join(raw_path, f"cluster_{cluster_id}_grid.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    #### 4. BARPLOT SI v GH

    counts = df_cluster.groupby([cluster_name, 'condition']).size().unstack(fill_value=0)

    counts_reset = counts.reset_index().melt(id_vars=cluster_name, var_name='condition', value_name='count')   # Convert to long format

    plt.figure(figsize=(10, 6))
    sns.barplot(data=counts_reset, x=cluster_name, y='count', hue='condition')
    plt.title("Count of Iso vs Group per Cluster (Yhat)")
    plt.xlabel("Cluster ID (Yhat)")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    path = os.path.join(output_dir, 'cluster_barlot.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')


    #### 5. BARPLOT OF PROPORTIONS PER CLUSTER (WITHIN CONDITION)

    # Normalize counts within each condition (column-wise)
    proportions = counts.div(counts.sum(axis=0), axis=1)  # axis=1 = normalize by condition totals

    # Convert to long format
    proportions_reset = proportions.copy()
    proportions_reset[cluster_name] = proportions_reset.index  # preserve cluster column
    proportions_reset = proportions_reset.melt(
        id_vars=cluster_name,
        var_name='condition',
        value_name='proportion'
    )

    # Plot proportions
    plt.figure(figsize=(10, 6))
    sns.barplot(data=proportions_reset, x=cluster_name, y='proportion', hue='condition')
    plt.title("Proportion of Interactions per Cluster (within each condition)")
    plt.xlabel("Cluster ID (Yhat)")
    plt.ylabel("Proportion")
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Save proportions plot
    prop_path = os.path.join(output_dir, 'cluster_proportions.png')
    plt.savefig(prop_path, dpi=300, bbox_inches='tight')
    plt.close()



    #### 5. SPEEDS

    for cluster_id in sorted(df[cluster_name].unique()):
        cluster_df = df[df[cluster_name] == cluster_id]
        interaction_ids = cluster_df['interaction_id'].unique()
        np.random.shuffle(interaction_ids)
        sample_ids = interaction_ids[:81]  # max 9x9 grid

        n = int(np.ceil(np.sqrt(len(sample_ids))))
        fig, axes = plt.subplots(n, n, figsize=(n * 2, n * 2))
        axes = axes.flatten()

        for ax, interaction_id in zip(axes, sample_ids):
            traj = cluster_df[cluster_df['interaction_id'] == interaction_id]

            # Plot precomputed speed
            ax.plot(traj["Normalized Frame"], traj["track1_speed"], color='red')
            ax.plot(traj["Normalized Frame"], traj["track2_speed"], color='blue')

            # Optional: show event alignment
            ax.axvline(0, color='black', linestyle='--', linewidth=0.5)

            # Axis formatting
            ax.set_ylim(0, 2)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(str(interaction_id), fontsize=6)

        # Hide any unused subplots
        for i in range(len(sample_ids), len(axes)):
            axes[i].axis('off')

        plt.suptitle(f"Raw Speed Traces - Cluster {cluster_id}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        speed = os.path.join(output_dir, "speed")
        os.makedirs(speed, exist_ok=True)
        save_path = os.path.join(speed, f"cluster_{cluster_id}_speed_raw_grid.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    

    #### 6. AVERAGE & DIFFERENCE SPEED OVER TIME PER CLUSTER

    cluster_ids = sorted(df[cluster_name].unique())
    num_clusters = len(cluster_ids)
    cols = 5
    rows = int(np.ceil(num_clusters / cols))

    # === (a) Mean Speed Over Time ===
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, cluster_id in enumerate(cluster_ids):
        ax = axes[idx]
        cluster_df = df[df[cluster_name] == cluster_id].copy()
        
        cluster_df["mean_speed"] = (cluster_df["track1_speed"] + cluster_df["track2_speed"]) / 2

        grouped = cluster_df.groupby("Normalized Frame")["mean_speed"]
        mean_vals = grouped.mean()
        std_vals = grouped.std()

        ax.plot(mean_vals.index, mean_vals.values, color='darkorange')
        ax.fill_between(mean_vals.index, mean_vals - std_vals, mean_vals + std_vals, color='orange', alpha=0.3)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_title(f"Cluster {cluster_id}")
        ax.set_ylim(0, 2)
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(len(cluster_ids), len(axes)):
        axes[i].axis('off')

    plt.suptitle("Mean ± SD of Combined Speed Over Time per Cluster", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(speed, "mean_speed_summary.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # === (b) Speed Difference Over Time ===
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, cluster_id in enumerate(cluster_ids):
        ax = axes[idx]
        cluster_df = df[df[cluster_name] == cluster_id].copy()

        cluster_df["speed_diff"] = np.abs(cluster_df["track1_speed"] - cluster_df["track2_speed"])

        grouped = cluster_df.groupby("Normalized Frame")["speed_diff"]
        mean_vals = grouped.mean()
        std_vals = grouped.std()

        ax.plot(mean_vals.index, mean_vals.values, color='navy')
        ax.fill_between(mean_vals.index, mean_vals - std_vals, mean_vals + std_vals, color='blue', alpha=0.3)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_title(f"Cluster {cluster_id}")
        ax.set_ylim(0, 2)
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(len(cluster_ids), len(axes)):
        axes[i].axis('off')

    plt.suptitle("Mean ± SD of Speed Difference Between Tracks per Cluster", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(speed, "speed_difference_summary.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


    #### 5. ACCELERATION

    for cluster_id in sorted(df[cluster_name].unique()):
        cluster_df = df[df[cluster_name] == cluster_id]
        interaction_ids = cluster_df['interaction_id'].unique()
        np.random.shuffle(interaction_ids)
        sample_ids = interaction_ids[:81]  # max 9x9 grid

        n = int(np.ceil(np.sqrt(len(sample_ids))))
        fig, axes = plt.subplots(n, n, figsize=(n * 2, n * 2))
        axes = axes.flatten()

        for ax, interaction_id in zip(axes, sample_ids):
            traj = cluster_df[cluster_df['interaction_id'] == interaction_id]

            # Plot precomputed speed
            ax.plot(traj["Normalized Frame"], traj["track1_acceleration"], color='red')
            ax.plot(traj["Normalized Frame"], traj["track2_acceleration"], color='blue')

            # Optional: show event alignment
            ax.axvline(0, color='black', linestyle='--', linewidth=0.5)

            # Axis formatting
            ax.set_ylim(-1, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(str(interaction_id), fontsize=6)

        # Hide any unused subplots
        for i in range(len(sample_ids), len(axes)):
            axes[i].axis('off')

        plt.suptitle(f"Raw Acceleration Traces - Cluster {cluster_id}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        accel = os.path.join(output_dir, "acceleration")
        os.makedirs(accel, exist_ok=True)
        save_path = os.path.join(accel, f"cluster_{cluster_id}_accleration_raw_grid.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    

    #### 6. AVERAGE & DIFFERENCE ACCELERATION OVER TIME PER CLUSTER

    cluster_ids = sorted(df[cluster_name].unique())
    num_clusters = len(cluster_ids)
    cols = 5
    rows = int(np.ceil(num_clusters / cols))

    # === (a) Mean Speed Over Time ===
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, cluster_id in enumerate(cluster_ids):
        ax = axes[idx]
        cluster_df = df[df[cluster_name] == cluster_id].copy()
        
        cluster_df["mean_accel"] = (cluster_df["track1_acceleration"] + cluster_df["track2_acceleration"]) / 2

        grouped = cluster_df.groupby("Normalized Frame")["mean_accel"]
        mean_vals = grouped.mean()
        std_vals = grouped.std()

        ax.plot(mean_vals.index, mean_vals.values, color='darkorange')
        ax.fill_between(mean_vals.index, mean_vals - std_vals, mean_vals + std_vals, color='orange', alpha=0.3)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_title(f"Cluster {cluster_id}")
        ax.set_ylim(-1, 1)
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(len(cluster_ids), len(axes)):
        axes[i].axis('off')

    plt.suptitle("Mean ± SD of Combined Acceleration Over Time per Cluster", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(accel, "mean_accleration_summary.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() 

    # === (b) Speed Difference Over Time ===
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, cluster_id in enumerate(cluster_ids):
        ax = axes[idx]
        cluster_df = df[df[cluster_name] == cluster_id].copy()

        cluster_df["accel_diff"] = np.abs(cluster_df["track1_acceleration"] - cluster_df["track2_acceleration"])

        grouped = cluster_df.groupby("Normalized Frame")["accel_diff"]
        mean_vals = grouped.mean()
        std_vals = grouped.std()

        ax.plot(mean_vals.index, mean_vals.values, color='navy')
        ax.fill_between(mean_vals.index, mean_vals - std_vals, mean_vals + std_vals, color='blue', alpha=0.3)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_title(f"Cluster {cluster_id}")
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(len(cluster_ids), len(axes)):
        axes[i].axis('off')

    plt.suptitle("Mean ± SD of Acceleration Difference Between Tracks per Cluster", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(accel, "accleration_difference_summary.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()




    #### 5. HEADING ANGLE track1_angle

    for cluster_id in sorted(df[cluster_name].unique()):
        cluster_df = df[df[cluster_name] == cluster_id]
        interaction_ids = cluster_df['interaction_id'].unique()
        np.random.shuffle(interaction_ids)
        sample_ids = interaction_ids[:81]  # max 9x9 grid

        n = int(np.ceil(np.sqrt(len(sample_ids))))
        fig, axes = plt.subplots(n, n, figsize=(n * 2, n * 2))
        axes = axes.flatten()

        for ax, interaction_id in zip(axes, sample_ids):
            traj = cluster_df[cluster_df['interaction_id'] == interaction_id]

            # Plot precomputed speed
            ax.plot(traj["Normalized Frame"], traj["track1_angle"], color='red')
            ax.plot(traj["Normalized Frame"], traj["track2_angle"], color='blue')

            # Optional: show event alignment
            ax.axvline(0, color='black', linestyle='--', linewidth=0.5)

            # Axis formatting
            ax.set_ylim(0, 180)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(str(interaction_id), fontsize=6)

        # Hide any unused subplots
        for i in range(len(sample_ids), len(axes)):
            axes[i].axis('off')

        plt.suptitle(f"Heading Angle Traces - Cluster {cluster_id}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        angle = os.path.join(output_dir, "angle")
        os.makedirs(angle, exist_ok=True)
        save_path = os.path.join(angle, f"cluster_{cluster_id}_heading-angle_raw_grid.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    

    #### 6. AVERAGE & DIFFERENCE ACCELERATION OVER TIME PER CLUSTER

    cluster_ids = sorted(df[cluster_name].unique())
    num_clusters = len(cluster_ids)
    cols = 5
    rows = int(np.ceil(num_clusters / cols))

    # === (a) Mean Speed Over Time ===
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, cluster_id in enumerate(cluster_ids):
        ax = axes[idx]
        cluster_df = df[df[cluster_name] == cluster_id].copy()
        
        cluster_df["angle_mean"] = (cluster_df["track1_angle"] + cluster_df["track2_angle"]) / 2

        grouped = cluster_df.groupby("Normalized Frame")["angle_mean"]
        mean_vals = grouped.mean()
        std_vals = grouped.std()

        ax.plot(mean_vals.index, mean_vals.values, color='darkorange')
        ax.fill_between(mean_vals.index, mean_vals - std_vals, mean_vals + std_vals, color='orange', alpha=0.3)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_title(f"Cluster {cluster_id}")
        ax.set_ylim(0, 180)
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(len(cluster_ids), len(axes)):
        axes[i].axis('off')

    plt.suptitle("Mean ± SD of Combined Heading Angle Over Time per Cluster", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(angle, "mean_heading-angle_summary.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() 

    # === (b) Speed Difference Over Time ===
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, cluster_id in enumerate(cluster_ids):
        ax = axes[idx]
        cluster_df = df[df[cluster_name] == cluster_id].copy()

        cluster_df["angle_diff"] = np.abs(cluster_df["track1_angle"] - cluster_df["track2_angle"])

        grouped = cluster_df.groupby("Normalized Frame")["angle_diff"]
        mean_vals = grouped.mean()
        std_vals = grouped.std()

        ax.plot(mean_vals.index, mean_vals.values, color='navy')
        ax.fill_between(mean_vals.index, mean_vals - std_vals, mean_vals + std_vals, color='blue', alpha=0.3)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_title(f"Cluster {cluster_id}")
        ax.set_ylim(0, 180)
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(len(cluster_ids), len(axes)):
        axes[i].axis('off')

    plt.suptitle("Mean ± SD of Heading Angle Difference Between Tracks per Cluster", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(angle, "heading-angle_difference_summary.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    
    
    #### 5. APPROACH ANGLE track1_approach_angle

    for cluster_id in sorted(df[cluster_name].unique()):
        cluster_df = df[df[cluster_name] == cluster_id]
        interaction_ids = cluster_df['interaction_id'].unique()
        np.random.shuffle(interaction_ids)
        sample_ids = interaction_ids[:81]  # max 9x9 grid

        n = int(np.ceil(np.sqrt(len(sample_ids))))
        fig, axes = plt.subplots(n, n, figsize=(n * 2, n * 2))
        axes = axes.flatten()

        for ax, interaction_id in zip(axes, sample_ids):
            traj = cluster_df[cluster_df['interaction_id'] == interaction_id]

            # Plot precomputed speed
            ax.plot(traj["Normalized Frame"], traj["track1_approach_angle"], color='red')
            ax.plot(traj["Normalized Frame"], traj["track2_approach_angle"], color='blue')

            # Optional: show event alignment
            ax.axvline(0, color='black', linestyle='--', linewidth=0.5)

            # Axis formatting
            ax.set_ylim(0, 180)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(str(interaction_id), fontsize=6)

        # Hide any unused subplots
        for i in range(len(sample_ids), len(axes)):
            axes[i].axis('off')

        plt.suptitle(f"Approach Angle Traces - Cluster {cluster_id}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        angle = os.path.join(output_dir, "angle")
        os.makedirs(angle, exist_ok=True)
        save_path = os.path.join(angle, f"cluster_{cluster_id}_approach-angle_raw_grid.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    

    #### 6. AVERAGE & DIFFERENCE ACCELERATION OVER TIME PER CLUSTER

    cluster_ids = sorted(df[cluster_name].unique())
    num_clusters = len(cluster_ids)
    cols = 5
    rows = int(np.ceil(num_clusters / cols))

    # === (a) Mean Speed Over Time ===
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, cluster_id in enumerate(cluster_ids):
        ax = axes[idx]
        cluster_df = df[df[cluster_name] == cluster_id].copy()
        
        cluster_df["mean_angle"] = (cluster_df["track1_approach_angle"] + cluster_df["track2_approach_angle"]) / 2

        grouped = cluster_df.groupby("Normalized Frame")["mean_angle"]
        mean_vals = grouped.mean()
        std_vals = grouped.std()

        ax.plot(mean_vals.index, mean_vals.values, color='darkorange')
        ax.fill_between(mean_vals.index, mean_vals - std_vals, mean_vals + std_vals, color='orange', alpha=0.3)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_title(f"Cluster {cluster_id}")
        ax.set_ylim(0, 180)
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(len(cluster_ids), len(axes)):
        axes[i].axis('off')

    plt.suptitle("Mean ± SD of Combined Approach Angle Over Time per Cluster", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(angle, "mean_approach-angle_summary.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() 

    # === (b) Speed Difference Over Time ===
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, cluster_id in enumerate(cluster_ids):
        ax = axes[idx]
        cluster_df = df[df[cluster_name] == cluster_id].copy()

        cluster_df["diff_angle"] = np.abs(cluster_df["track1_approach_angle"] - cluster_df["track2_approach_angle"])

        grouped = cluster_df.groupby("Normalized Frame")["diff_angle"]
        mean_vals = grouped.mean()
        std_vals = grouped.std()

        ax.plot(mean_vals.index, mean_vals.values, color='navy')
        ax.fill_between(mean_vals.index, mean_vals - std_vals, mean_vals + std_vals, color='blue', alpha=0.3)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_title(f"Cluster {cluster_id}")
        ax.set_ylim(0, 180)
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(len(cluster_ids), len(axes)):
        axes[i].axis('off')

    plt.suptitle("Mean ± SD of Approach Angle Difference Between Tracks per Cluster", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(angle, "approach-angle_difference_summary.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()



    ### RAW BODY-BODY DISTANCE 

    for cluster_id in sorted(df[cluster_name].unique()):
        cluster_df = df[df[cluster_name] == cluster_id]
        interaction_ids = cluster_df['interaction_id'].unique()
        np.random.shuffle(interaction_ids)
        sample_ids = interaction_ids[:81]  # max 9x9 grid

        n = int(np.ceil(np.sqrt(len(sample_ids))))
        fig, axes = plt.subplots(n, n, figsize=(n * 2, n * 2))
        axes = axes.flatten()

        for ax, interaction_id in zip(axes, sample_ids):
            traj = cluster_df[cluster_df['interaction_id'] == interaction_id]

            # Plot Distance trace
            ax.plot(traj["Normalized Frame"], traj["Distance"], color='black')

            # Optional: vertical alignment line
            ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)

            # Axis formatting
            ax.set_ylim(0, 20)  
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(str(interaction_id), fontsize=6)

        for i in range(len(sample_ids), len(axes)):
            axes[i].axis('off')

        plt.suptitle(f"Body-Body Distance Over Time - Cluster {cluster_id}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        body = os.path.join(output_dir, "body-body-distance")
        os.makedirs(body, exist_ok=True)
        save_path = os.path.join(body, f"cluster_{cluster_id}_distance_grid.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    #### AVERAGE BODY DISTANCE
    cluster_ids = sorted(df[cluster_name].unique())
    num_clusters = len(cluster_ids)
    cols = 5
    rows = int(np.ceil(num_clusters / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), sharex=True, sharey=True)
    axes = axes.flatten()

    # Plot mean ± std of Distance per cluster
    for idx, cluster_id in enumerate(cluster_ids):
        ax = axes[idx]
        cluster_df = df[df[cluster_name] == cluster_id]
        grouped = cluster_df.groupby("Normalized Frame")["Distance"]
        mean_dist = grouped.mean()
        std_dist = grouped.std()

        ax.plot(mean_dist.index, mean_dist.values, color='black')
        ax.fill_between(
            mean_dist.index,
            mean_dist - std_dist,
            mean_dist + std_dist,
            color='gray',
            alpha=0.3
        )
        ax.axvline(0, color='red', linestyle='--', linewidth=0.5)
        ax.set_title(f"Cluster {cluster_id}")
        ax.set_ylim(0, 20)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplots
    for i in range(len(cluster_ids), len(axes)):
        axes[i].axis('off')

    plt.suptitle("Mean ± SD of Body-Body Distance Between Larvae per Cluster", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = os.path.join(body, "summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


    #### MIN DISTANCE

    for cluster_id in sorted(df[cluster_name].unique()):
        cluster_df = df[df[cluster_name] == cluster_id]
        interaction_ids = cluster_df['interaction_id'].unique()
        np.random.shuffle(interaction_ids)
        sample_ids = interaction_ids[:81]  # max 9x9 grid

        n = int(np.ceil(np.sqrt(len(sample_ids))))
        fig, axes = plt.subplots(n, n, figsize=(n * 2, n * 2))
        axes = axes.flatten()

        for ax, interaction_id in zip(axes, sample_ids):
            traj = cluster_df[cluster_df['interaction_id'] == interaction_id]

            # Plot Distance trace
            ax.plot(traj["Normalized Frame"], traj["min_distance"], color='black')

            # Optional: vertical alignment line
            ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)

            # Axis formatting
            ax.set_ylim(0, 20)  
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(str(interaction_id), fontsize=6)

        for i in range(len(sample_ids), len(axes)):
            axes[i].axis('off')

        plt.suptitle(f"Min Distance Over Time - Cluster {cluster_id}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        min_distance = os.path.join(output_dir, "min-distance")
        os.makedirs(min_distance, exist_ok=True)

        save_path = os.path.join(min_distance, f"cluster_{cluster_id}_min_distance_grid.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    

    #### AVERAGE MIN DISTANCE
    cluster_ids = sorted(df[cluster_name].unique())
    num_clusters = len(cluster_ids)
    cols = 5
    rows = int(np.ceil(num_clusters / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), sharex=True, sharey=True)
    axes = axes.flatten()

    # Plot mean ± std of Distance per cluster
    for idx, cluster_id in enumerate(cluster_ids):
        ax = axes[idx]
        cluster_df = df[df[cluster_name] == cluster_id]
        grouped = cluster_df.groupby("Normalized Frame")["min_distance"]
        mean_dist = grouped.mean()
        std_dist = grouped.std()

        ax.plot(mean_dist.index, mean_dist.values, color='black')
        ax.fill_between(
            mean_dist.index,
            mean_dist - std_dist,
            mean_dist + std_dist,
            color='gray',
            alpha=0.3
        )
        ax.axvline(0, color='red', linestyle='--', linewidth=0.5)
        ax.set_title(f"Cluster {cluster_id}")
        ax.set_ylim(0, 20)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplots
    for i in range(len(cluster_ids), len(axes)):
        axes[i].axis('off')

    plt.suptitle("Mean ± SD of Min Distance Between Larvae per Cluster", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = os.path.join(min_distance, "summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


    #### RASTA 
    interaction_colors = {
        "head-head": "red",
        "head-body": "orange",
        "head-tail": "yellow",
        "body-body": "black",
        "tail-tail": "green",
        "tail-body": "purple"
    }

    # Output directory
    raster_dir = os.path.join(output_dir, "contact_rasta")
    os.makedirs(raster_dir, exist_ok=True)

    # Loop over each cluster
    for cluster_id in sorted(df[cluster_name].unique()):
        cluster_df = df[df[cluster_name] == cluster_id]
        interaction_ids = cluster_df['interaction_id'].unique()

        plt.figure(figsize=(12, len(interaction_ids) * 0.3))
        ax = plt.gca()

        for i, interaction_id in enumerate(interaction_ids):
            sub_df = cluster_df[cluster_df['interaction_id'] == interaction_id]

            for _, row in sub_df.iterrows():
                interaction_type = row.get("interaction_type")
                min_distance = row["min_distance"]

                # Only draw a line if interaction_type is in the allowed list and distance < 1
                if interaction_type in interaction_colors and min_distance < 1:
                    color = interaction_colors[interaction_type]

                    ax.plot(
                        [row["Normalized Frame"], row["Normalized Frame"]],
                        [i - 0.4, i + 0.4],
                        color=color,
                        linewidth=2
                    )

        ax.set_yticks(np.arange(len(interaction_ids)))
        ax.set_yticklabels(interaction_ids, fontsize=6)
        ax.set_xlabel("Normalized Frame")
        ax.set_ylabel("Interaction ID")
        ax.set_title(f"Raster Plot of Contact Events - Cluster {cluster_id}")
        plt.tight_layout()

        # Save plot
        raster_path = os.path.join(raster_dir, f"cluster_{cluster_id}_raster.png")
        plt.savefig(raster_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    #### CONTACT SUMMARY

    interaction_contact_summary = []

    for cluster_id in sorted(df[cluster_name].unique()):
        cluster_df = df[df[cluster_name] == cluster_id]
        for inter_id in cluster_df["interaction_id"].unique():
            inter_df = cluster_df[cluster_df["interaction_id"] == inter_id]
            n_close = (inter_df["min_distance"] < 1).sum()
            interaction_contact_summary.append({
                "cluster": cluster_id,
                "interaction_id": inter_id,
                "frames_below_1mm": n_close
            })

    # Convert to DataFrame
    df_interaction_contact = pd.DataFrame(interaction_contact_summary)

    plt.figure(figsize=(10, 4))
    sns.barplot(data=df_interaction_contact, x="cluster", y="frames_below_1mm", color="green", ci="sd", alpha=0.8)
    plt.title("Frames <1mm per Interaction per Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Frames Below 1mm")
    plt.tight_layout()
    plt.savefig(os.path.join(raster_dir, "contact_framesn.png"), dpi=300)
    plt.close()



    interaction_merge_map = {
    "head-tail": "head-tail",
    "tail-head": "head-tail",
    "tail-body": "tail-body",
    "body-tail": "tail-body",
    "head-body": "head-body",
    "body-head": "head-body",
    "tail-tail": "tail-tail",
    "head-head": "head-head",
    "body-body": "body-body"
}


    interaction_types = ["head-head", "head-body", "head-tail", "body-body", "tail-tail", "tail-body"]

    interaction_colors = {
        "head-head": "red",
        "head-body": "orange",
        "head-tail": "yellow",
        "body-body": "black",
        "tail-tail": "green",
        "tail-body": "purple"
    }


    records = []

    for cluster_id in sorted(df[cluster_name].unique()):
        cluster_df = df[df[cluster_name] == cluster_id]
        
        for inter_id in cluster_df["interaction_id"].unique():
            inter_df = cluster_df[cluster_df["interaction_id"] == inter_id]
            
            # Filter only close-contact frames
            close = inter_df[inter_df["min_distance"] < 1].copy()

            # Merge types
            close["interaction_type_merged"] = close["interaction_type"].map(interaction_merge_map)
            
            # Count how many frames of each interaction type
            counts = close["interaction_type_merged"].value_counts().to_dict()

            # Store result
            row = {"cluster": cluster_id, "interaction_id": inter_id}
            for itype in interaction_types:
                row[itype] = counts.get(itype, 0)
            records.append(row)

    # === Build final DataFrame ===
    df_contacts = pd.DataFrame(records)

    # === Per-cluster plotting ===
    for cluster_id in sorted(df_contacts["cluster"].unique()):
        subset = df_contacts[df_contacts["cluster"] == cluster_id]

        # Compute mean per interaction type
        mean_counts = subset[interaction_types].mean().reset_index()
        mean_counts.columns = ["interaction_type", "mean_frames"]

        # Plot
        plt.figure(figsize=(6, 4))
        sns.barplot(data=mean_counts, x="interaction_type", y="mean_frames", palette=interaction_colors)
        plt.title(f"Cluster {cluster_id} – Avg. Frames <1mm by Type")
        plt.xlabel("Interaction Type")
        plt.ylabel("Avg. Frames per Interaction")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save
        save_path = os.path.join(raster_dir, f"cluster_{cluster_id}_interaction_type.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

    
        #### 7. DISTANCE TRAVELLED VIOLIN PLOT (Track-wise, agnostic to T1/T2)

        print("📏 Computing track-wise distances per interaction...")

        track_distances = []

        for cluster_id in sorted(df[cluster_name].unique()):
            cluster_df = df[df[cluster_name] == cluster_id]

            for inter_id in cluster_df['interaction_id'].unique():
                inter_df = cluster_df[cluster_df['interaction_id'] == inter_id]
                inter_df = inter_df.sort_values("Frame")

                for track in ["mm_Track_1", "mm_Track_2"]:
                    x = inter_df[f"{track} x_body"].values
                    y = inter_df[f"{track} y_body"].values

                    if len(x) < 2:
                        continue

                    dist = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))

                    track_distances.append({
                        "cluster": cluster_id,
                        "distance": dist
                    })

        df_distances = pd.DataFrame(track_distances)

        # Save raw distances CSV if needed
        dist_out = os.path.join(output_dir, "distance_travelled")
        os.makedirs(dist_out, exist_ok=True)

        # Plot violin
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df_distances, x="cluster", y="distance", inner="box", cut=0, palette="Set2")
        plt.title("Track-wise Distance Travelled per Cluster")
        plt.xlabel("Cluster")
        plt.ylabel("Distance Travelled")
        plt.tight_layout()

        plt.savefig(os.path.join(dist_out, "distance_violin.png"), dpi=300)
        plt.close()
        print("✅ Saved distance travelled violin plot.")

    

    #### 8. DIFFERENCE IN CUMULATIVE DISTANCE TRAVELLED OVER TIME

    print("📈 Computing frame-wise distance difference between tracks...")

    dist_diffs = []

    for cluster_id in sorted(df[cluster_name].unique()):
        cluster_df = df[df[cluster_name] == cluster_id]

        for inter_id in cluster_df['interaction_id'].unique():
            inter_df = cluster_df[cluster_df['interaction_id'] == inter_id].sort_values("Frame")

            frames = inter_df["Normalized Frame"].values
            x1 = inter_df["mm_Track_1 x_body"].values
            y1 = inter_df["mm_Track_1 y_body"].values
            x2 = inter_df["mm_Track_2 x_body"].values
            y2 = inter_df["mm_Track_2 y_body"].values

            # Stepwise distance per frame
            d1_stepwise = np.sqrt(np.diff(x1)**2 + np.diff(y1)**2)
            d2_stepwise = np.sqrt(np.diff(x2)**2 + np.diff(y2)**2)

            # Cumulative distance per frame
            # d1_cum = np.insert(np.cumsum(d1_stepwise), 0, 0)
            # d2_cum = np.insert(np.cumsum(d2_stepwise), 0, 0)

            # # Difference in cumulative distance travelled (signed or abs)
            # diff_over_time = np.abs(d1_cum - d2_cum)  # for magnitude only
            # # OR: diff_over_time = d1_cum - d2_cum  # for directionality

            diff_over_time = np.abs(d1_stepwise - d2_stepwise)
            frames = frames[1:]


            # Store for each frame
            for f, d in zip(frames, diff_over_time):
                dist_diffs.append({
                    "cluster": cluster_id,
                    "interaction_id": inter_id,
                    "Normalized Frame": f,
                    "distance_diff": d
    })



    df_diff = pd.DataFrame(dist_diffs)

    # Plot mean ± std distance difference over time per cluster
    cluster_ids = sorted(df_diff["cluster"].unique())
    num_clusters = len(cluster_ids)
    cols = 5
    rows = int(np.ceil(len(cluster_ids) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, cluster_id in enumerate(cluster_ids):
        ax = axes[idx]
        cluster_data = df_diff[df_diff["cluster"] == cluster_id]
        grouped = cluster_data.groupby("Normalized Frame")["distance_diff"]
        mean_diff = grouped.mean()
        std_diff = grouped.std()

        ax.plot(mean_diff.index, mean_diff.values, color='darkgreen')
        ax.fill_between(mean_diff.index,
                        mean_diff - std_diff,
                        mean_diff + std_diff,
                        color='lightgreen', alpha=0.4)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_title(f"Cluster {cluster_id}")
        ax.set_ylim(0, None)
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(len(cluster_ids), len(axes)):
        axes[i].axis('off')

    plt.suptitle("Mean ± SD of Track-wise Distance Difference per Cluster", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(dist_out, "distance_diff_summary.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()















############ CREATE CROPPED INTERACTION DATAFRAME ############

# group_csv = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser/test2/interactions_group.csv'
# iso_csv = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser/test2/interactions_iso.csv'
# wkt_dir = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser/videos_original'
# output_dir = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser/test3/cropped_interactions.csv'

# create_cropped_interactions(group_csv,iso_csv, wkt_dir, output_dir)


############ REASSIGN CLUSTER IDS ############ only run once !!!

# file = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser/test4_F29/umap2-data2-F29-L16.csv'
# output = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser/test4_F29/umap2-data2-F29-L16.csv'
# modify_cluster_ids(file, output)



############ CLUSTER PIPELINE ############


cropped_interaction = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser/test4_F29/cropped_interactions.csv'
cluster = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser/test4_F29/pca-data2-F29-L15.csv'
cluster_name = 'Yhat.idt.pca' # edit name of clusters  
video_path = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser/videos_original'
output_dir = "/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser/test4_F29/Yhat.idt.pca" 
os.makedirs(output_dir, exist_ok=True)


cluster_pipeline(cropped_interaction, cluster, cluster_name, video_path, output_dir)







