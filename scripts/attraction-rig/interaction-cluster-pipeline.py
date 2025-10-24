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
import imageio.v2 as imageio
import matplotlib as mpl


#### FUNCTION CREATE_CROPPED_INTERACTIONS: ORIGINAL GROUP AND ISO INTERACTIONS MERGED AND INTERACTIONS CROPPED 30 FRAMES
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

    # === 2. Apply scaling BACK to pixels based on individual video diameters ===

    for idx, row in df_cropped.iterrows():
        video_file = row['file']
        
        if video_file == '2025-02-28_13-00-52_td9.mp4':
            scale = 1032 / 90
            print(f"🟡 Using fixed scale {scale:.3f} for {video_file}")
        
        elif video_file in diameter_dict:
            diameter_pixels = diameter_dict[video_file]
            scale =  diameter_pixels / 90
        
        else:
            print(f"⚠️ Warning: No WKT file found for video {video_file}")
            continue  # skip scaling if diameter is missing

        for col in coordinate_columns:
            df_cropped.at[idx, col] *= scale

    df_cropped.to_csv(output_dir, index=False)



#### FUNCTION MODIFY_CLUSTER_IDS: MODIFY INTERACTION ID IN YOUNGSERS DATA 

def modify_cluster_ids(file, output):
    df = pd.read_csv(file)

    df[['prefix', 'num']] = df['interaction_id2']\
        .str.extract(r'(?i)^([GI])(\d+)$', expand=True)

    # Map to the full condition name
    df['condition'] = df['prefix'].str.upper().map({
        'G': 'group',
        'I': 'iso'})

    # Build your new interaction_id
    df['interaction_id'] = df['condition'] + '_' + df['num']

    df = df.drop(columns=['prefix', 'num'])
    df.to_csv(output, index=False)


#### CLASS TO ANALYSE THE CLUSTERS  
class ClusterPipeline:

    def __init__(self, directory, interactions, clusters, cluster_name, video_path):
        
        self.directory = directory
        self.interaction_path = interactions 
        self.cluster_path = clusters
        self.cluster_name = cluster_name
        self.video_path = video_path

        self.interactions = None
        self.clusters = None
        self.df = None
        
    
    #### METHOD LOADING_DATA: LOAD AND MERGE DATAFRAMES
    def loading_data(self):

        ## LOAD DATAFRAMES

        self.interactions = pd.read_csv(self.interaction_path)
        self.clusters = pd.read_csv(self.cluster_path)

        ## MISSING INTERACTIONS BETWEEN DATAFRAMES

        set1 = set(self.interactions['interaction_id'].unique())
        set2 = set(self.clusters['interaction_id'].unique())
        missing_from_cluster = sorted(set1 - set2)
        missing_from_cropped  = sorted(set2 - set1)
        print(f">>> {len(missing_from_cluster)} IDs in cropped not in cluster (e.g. {missing_from_cluster[:5]})")
        print(f">>> {len(missing_from_cropped)} IDs in cluster not in cropped (e.g. {missing_from_cropped[:5]})")

        ## MERGE DATAFRAMES

        self.df = pd.merge(
            self.interactions, 
            self.clusters[['interaction_id', self.cluster_name]], 
            on='interaction_id', 
            how='inner'
        )
    

    #### METHOD ANCHOR_PARTNER: CREATE ANCHOR AND PARTNER BASED ON LINEARITY OF TRACK  
    def anchor_partner(self):
        
        df = self.df

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

        
        self.df = df


    #### METHOD SANITY_CHECK_ANCHOR_PARTNER: SANITY CHECK WHEREBY ANCHOR AND PARTNER REVERSED TO LINEARITY SCORE
    def sanity_check_anchor_partner(self):
        
        df = self.df

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
                winner = 2
                anchor_pts, partner_pts, anchor_axis = coords2, coords1, axis2
            else:
                winner = 1
                anchor_pts, partner_pts, anchor_axis = coords1, coords2, axis1

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

        
        self.df = df
    


    #### METHOD GRID_VIDEOS: GENERATE GRID VIDEOS OF INTERACTION CLUSTERS
    def grid_videos(self):

        df = self.df
        cluster_name = self.cluster_name 
        video_path = self.video_path 

        grid_video_dir = os.path.join(self.directory, "grid_videos")
        os.makedirs(grid_video_dir, exist_ok=True)

        frames_per_clip = 30
        dot_radius = 3
        dot_thickness = -1  # Filled
        fps = 3
        crop_size = 400

        # === TRACK VALID CLIPS ===
        cluster_to_interactions = {}

        for cluster_id in sorted(df[cluster_name].unique()):
            cluster_df = df[df[cluster_name] == cluster_id].copy()
            print("Cluster:", cluster_id)
            print(cluster_df["condition"].value_counts())

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


        print('grid videos generated') 
    
    #### METHOD RAW_TRAJECTORIES: GENERATE RAW TRAJECTORIES OF INTERACTIONS
    def raw_trajectories(self):

        df = self.df
        cluster_name = self.cluster_name 

        output = os.path.join(self.directory, "raw_trajectories")
        os.makedirs(output, exist_ok=True)


        for cluster_id in sorted(df[cluster_name].unique()):
            cluster_df = df[df[cluster_name] == cluster_id].copy()
            interaction_ids = cluster_df['interaction_id'].unique()
            np.random.shuffle(interaction_ids)
            sample_ids = interaction_ids[:81]  # up to 9×9 grid

            # figure grid size
            n = int(np.ceil(np.sqrt(len(sample_ids))))
            fig, axes = plt.subplots(n, n, figsize=(n * 2, n * 2), 
                                    sharex=False, sharey=False, constrained_layout=True,   )
            axes = axes.flatten()

            for ax, interaction_id in zip(axes, sample_ids):
                traj = cluster_df[cluster_df['interaction_id'] == interaction_id]

                A = traj[['anchor x_body','anchor y_body']].values
                B = traj[['partner x_body','partner y_body']].values

                # plot anchor & partner
                ax.plot(A[:,0], A[:,1],
                        color='C0', linewidth=1, alpha=0.8, label='Anchor')
                ax.plot(B[:,0], B[:,1],
                        color='C1', linewidth=1, alpha=0.8, label='Partner')

                # mark starts
                ax.scatter(A[0,0], A[0,1],
                        color='C0', marker='X', s=30, label='Start Anchor')
                ax.scatter(B[0,0], B[0,1],
                        color='C1', marker='X', s=30, label='Start Partner')

                # equal scaling, grid, labels
                ax.set_aspect('equal', adjustable='box')

                # ←—— Insert the padding snippet here:
                xmin = min(A[:,0].min(), B[:,0].min())
                xmax = max(A[:,0].max(), B[:,0].max())
                ymin = min(A[:,1].min(), B[:,1].min())
                ymax = max(A[:,1].max(), B[:,1].max())
                dx, dy = xmax - xmin, ymax - ymin
                half = max(dx, dy, 150) / 2
                xmid, ymid = (xmin + xmax) / 2, (ymin + ymax) / 2
                ax.set_xlim(xmid - half, xmid + half)
                ax.set_ylim(ymid - half, ymid + half)
                # ←—— End padding snippet


                ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
                ax.set_xlabel('X', fontsize=6)
                ax.set_ylabel('Y', fontsize=6)
                ax.tick_params(axis='both', which='both', labelsize=4)
                ax.set_title(interaction_id, fontsize=6)

            # turn off any unused subplots
            for i in range(len(sample_ids), len(axes)):
                axes[i].axis('off')

            plt.suptitle(f"Raw Trajectories – Cluster {cluster_id}", fontsize=14)

            save_path = os.path.join(output, f"cluster_{cluster_id}_grid.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()


    #### METHOD MEAN_TRAJECTORIES: MEAN TRAJECTORIES OF EACH CLUSTER
    def mean_trajectories(self):
         
        df = self.df
        cluster_name = self.cluster_name 

        output = os.path.join(self.directory, "mean_trajectories")
        os.makedirs(output, exist_ok=True)

        ## MEAN TRAJECTORIES

        for cluster_id in sorted(df[cluster_name].unique()):
            cluster_df = df[df[cluster_name] == cluster_id].copy()
            grouped = cluster_df.groupby("Normalized Frame")

            t1_x = grouped["anchor x_body"].mean()
            t1_y = grouped["anchor y_body"].mean()
            t2_x = grouped["partner x_body"].mean()
            t2_y = grouped["partner y_body"].mean()

            # 2) std-dev
            t1_x_std = grouped["anchor x_body"].std()
            t1_y_std = grouped["anchor y_body"].std()
            t2_x_std = grouped["partner x_body"].std()
            t2_y_std = grouped["partner y_body"].std()

            plt.figure(figsize=(6, 6))
            plt.plot(t1_x, t1_y, label="Track 1", color="red")
            plt.plot(t2_x, t2_y, label="Track 2", color="blue")
            plt.scatter(t1_x.iloc[0], t1_y.iloc[0], color="red", marker="o", label="T1 Start")
            plt.scatter(t2_x.iloc[0], t2_y.iloc[0], color="blue", marker="o", label="T2 Start")

                # error bars in X and Y
            plt.errorbar(
                t1_x, t1_y,
                xerr=t1_x_std, yerr=t1_y_std,
                fmt="none", ecolor="red", alpha=0.3, label="Track 1 ±1 SD"
            )
            plt.errorbar(
                t2_x, t2_y,
                xerr=t2_x_std, yerr=t2_y_std,
                fmt="none", ecolor="blue", alpha=0.3, label="Track 2 ±1 SD"
            )

            plt.gca().invert_yaxis()
            plt.axis("equal")
            plt.title(f"Mean Trajectory - Cluster {cluster_id}")
            plt.legend()
            plt.tight_layout()

        
            save_path = os.path.join(output, f"cluster_{cluster_id}.png")

            plt.savefig(save_path)
            plt.close()

        ## MEAN RELATIVE TRAJECTORIES 

        for cluster_id in sorted(df[cluster_name].unique()):
            cluster_df = df[df[cluster_name] == cluster_id]

            # Compute relative positions
            cluster_df["x_rel"] = cluster_df["partner x_body"] - cluster_df["anchor x_body"]
            cluster_df["y_rel"] = cluster_df["partner y_body"] - cluster_df["anchor y_body"]

            grouped = cluster_df.groupby("Normalized Frame")

            x_rel = grouped["x_rel"].mean()
            y_rel = grouped["y_rel"].mean()
            x_rel_std = grouped["x_rel"].std()
            y_rel_std = grouped["y_rel"].std()

            # Plot
            plt.figure(figsize=(6, 6))

            # Anchor point (Track 1 always at 0,0)
            plt.scatter(0, 0, color="blue", label="Anchor")

            # Relative trajectory of Track 2
            plt.plot(x_rel, y_rel, color="orange", label="Partner Rel. Trajectory")
            plt.scatter(x_rel.iloc[0], y_rel.iloc[0], color="darkorange", marker="o", label="Partner Start")

            # Error bars
            plt.errorbar(x_rel, y_rel, xerr=x_rel_std, yerr=y_rel_std, fmt="none", ecolor="orange", alpha=0.3)

            # plt.gca().invert_yaxis()
            plt.axis("equal")
            plt.title(f"Partner relative to Anchor — Cluster {cluster_id}")
            plt.legend()
            plt.tight_layout()

            save_path = os.path.join(output, f"cluster_{cluster_id}_relative.png")
            plt.savefig(save_path)
            plt.close()
        
        ## MEAN CONTACT PHASE TRAJECTORY (PRE,DURING,POST INTERACTION)

        bins = {
        "pre": range(-15, -5),
        "contact": range(-5, 5),
        "post": range(5, 15)}

        for cluster_id in sorted(df[cluster_name].unique()):
            cluster_df = df[df[cluster_name] == cluster_id]

            cluster_df["x_rel"] = cluster_df["partner x_body"] - cluster_df["anchor x_body"]
            cluster_df["y_rel"] = cluster_df["partner y_body"] - cluster_df["anchor y_body"]

            # one r for all three panels
            all_disp = np.concatenate([
            cluster_df["x_rel"].abs().values,
            cluster_df["y_rel"].abs().values])
            r = np.percentile(all_disp, 98)
            
            fig, axs = plt.subplots(1, 3, figsize=(12, 4),  constrained_layout=True)  # 3 panels: pre/contact/post

            for ax, (phase, frames) in zip(axs, bins.items()):
                phase_df = cluster_df[cluster_df["Normalized Frame"].isin(frames)]

                grouped = phase_df.groupby("Normalized Frame")
                x_mean = grouped["x_rel"].mean()
                y_mean = grouped["y_rel"].mean()
                x_std = grouped["x_rel"].std()
                y_std = grouped["y_rel"].std()

                ax.plot(x_mean, y_mean, label=f"{phase} trajectory", color="orange")
                ax.errorbar(x_mean, y_mean, xerr=x_std, yerr=y_std, fmt="none", ecolor="orange", alpha=0.3)
                ax.scatter(0, 0, color="blue", label="Anchor (0,0)")

                ax.set_xlim(-r, r)
                ax.set_ylim(-r, r)

                # ax.invert_yaxis()
                ax.set_aspect('equal', adjustable='box')

                ax.set_title(phase.capitalize())


            fig.suptitle(f"Cluster {cluster_id} – Relative Trajectories by Phase", fontsize=14)
            fig.savefig(os.path.join(output, f"cluster_{cluster_id}_phased_trajectory.png"))
            plt.close()
    

    #### METHOD BARPLOTS: BARPLOTS OF CLUSTERS
    def barplots(self):

        df = self.clusters
        full_df = self.df
        cluster_name = self.cluster_name

        palette = {
                "group": "C0",   
                "iso": "C1"}
        
        ## RAW COUNT BARPLOT

        counts = df.groupby([cluster_name, 'condition']).size().unstack(fill_value=0)
        counts_reset = counts.reset_index().melt(id_vars=cluster_name, var_name='condition', value_name='count')   # Convert to long format

        plt.figure(figsize=(10, 6))
        sns.barplot(data=counts_reset, x=cluster_name, y='count', hue='condition', palette=palette)
        plt.title("Count of Iso vs Group per Cluster (Yhat)")
        plt.xlabel("Cluster ID (Yhat)")
        plt.ylabel("Count")
        plt.xticks(rotation=90)
        plt.tight_layout()
        path = os.path.join(self.directory, 'cluster_barplot.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')

        ## PROPORTION BARPLOT

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
        sns.barplot(data=proportions_reset, x=cluster_name, y='proportion', hue='condition', palette=palette)
        plt.title("Proportion of Interactions per Cluster")
        plt.xlabel("Cluster ID")
        plt.ylabel("Proportion")
        plt.xticks(rotation=90)
        plt.tight_layout()

        # Save proportions plot
        prop_path = os.path.join(self.directory, 'cluster_proportions.png')
        plt.savefig(prop_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    ## AVERAGE PROPORTIONS BARPLOT (PER VIDEO)

        # Collapse to unique interactions (to avoid overweighting frames)
        inter_per_video = (
            df[['file', 'condition', 'interaction_id', cluster_name]]
            .drop_duplicates(subset=['file', 'interaction_id'])
        )

        # Count interactions per (video, condition, cluster)
        counts = (
            inter_per_video
            .groupby(['file', 'condition', cluster_name])
            .size()
            .reset_index(name='count')
        )

        # Compute total interactions per video (for normalization)
        totals = (
            counts.groupby('file')['count']
            .transform('sum')
        )

        # Add proportion column
        counts['proportion'] = counts['count'] / totals

        # Ensure all cluster IDs appear (even if missing in some videos)
        all_clusters = sorted(df[cluster_name].unique())
        
        summary_df = (
            counts
            .set_index(['file', 'condition', cluster_name])
            .unstack(fill_value=0)
            .stack()
            .reset_index()
        )
        summary_df.rename(columns={0: 'proportion'}, inplace=True)

        # Save tidy summary table
        summary_csv_path = os.path.join(self.directory, 'per_video_cluster_proportions_long.csv')
        summary_df.to_csv(summary_csv_path, index=False)

        # Now you can plot with seaborn
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=summary_df,
            x=cluster_name, y='proportion', hue='condition',ci='sd', alpha=0.8, palette=palette)
        
        plt.title("Proportion of Clusters")
        plt.xlabel("Cluster ID")
        plt.ylabel("Proportion")
        plt.xticks(rotation=90)
        plt.tight_layout()

        per_video_prop_path = os.path.join(self.directory, 'cluster_proportions_per_video.png')
        plt.savefig(per_video_prop_path, dpi=300, bbox_inches='tight')
        plt.close()


        ## PROPORTION OVER TIME BARPLOT

        interaction_starts = full_df[full_df['Normalized Frame'] == 0].copy()

        # Bin based on absolute Frame
        interaction_starts['time_bin'] = pd.cut(
            interaction_starts['Frame'],
            bins=np.arange(0, 3600+600, 600),
            labels=[f"{i*600}-{(i+1)*600}" for i in range(6)],
            right=False
        )
        # Count interactions per (video, condition, cluster, time_bin)
        counts = (
            interaction_starts.groupby(['file', 'condition', cluster_name, 'time_bin'])
            .size()
            .reset_index(name='count')
        )

        # Compute totals per video × time_bin (for normalization)
        totals = counts.groupby(['file', 'time_bin'])['count'].transform('sum')
        counts['proportion'] = counts['count'] / totals

        # Ensure tidy format
        summary_df = (
            counts
            .set_index(['file', 'condition', cluster_name, 'time_bin'])
            .unstack(fill_value=0)
            .stack()
            .reset_index()
        )
        summary_df.rename(columns={0: 'proportion'}, inplace=True)

        # Save tidy table
        summary_csv_path = os.path.join(self.directory, 'per_video_cluster_proportions_by_timebin.csv')
        summary_df.to_csv(summary_csv_path, index=False)

        # --- Plot subplots ---
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
        axes = axes.flatten()

        time_bins = summary_df['time_bin'].unique()
        for idx, tb in enumerate(sorted(time_bins, key=lambda x: int(x.split('-')[0]))):
            ax = axes[idx]
            sns.barplot(
                data=summary_df[summary_df['time_bin'] == tb],
                x=cluster_name, y='proportion', hue='condition',
                ci='sd', alpha=0.8, palette=palette, ax=ax
            )
            ax.set_title(f"Time {tb} sec")
            ax.set_xlabel("Cluster ID (Yhat)")
            ax.set_ylabel("Proportion")
            ax.set_ylim(0, None)
            ax.tick_params(axis='x', rotation=90)

        plt.tight_layout()
        timebin_prop_path = os.path.join(self.directory, 'cluster_proportions_by_timebin.png')
        plt.savefig(timebin_prop_path, dpi=300, bbox_inches='tight')
        plt.close()
    

        ### OBSERVED - EXPECTED DEVIATION PLOT

        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype']  = 42
        

        cluster_counts = (df.groupby([cluster_name, 'condition']).size().unstack(fill_value=0).reindex(columns=['group', 'iso'], fill_value=0))  # count number per cluster per condition

        total_group = cluster_counts['group'].sum()
        total_iso   = cluster_counts['iso'].sum()
        total_all   = total_group + total_iso

        expected_group = total_group / total_all   # e.g., ~0.56

        observed_group_frac = cluster_counts['group'] / (cluster_counts['group'] + cluster_counts['iso']).replace({0: np.nan}) ## observed fraction
        observed_group_frac = observed_group_frac.fillna(0.0)

        deviation = observed_group_frac - expected_group ## expected fraction

        deviation_sorted = deviation.sort_values()
        colors = ['C1' if val < 0 else 'C0' for val in deviation_sorted.values]

        plt.figure(figsize=(8, 6))

        # Plot with Matplotlib's bar (since sns.barplot expects a DataFrame)
        plt.bar(deviation_sorted.index.astype(str), deviation_sorted.values, color=colors)

        # Add reference line
        plt.axhline(0, color='k', linestyle='--', linewidth=1)

        # Labels and title
        plt.title("Deviation from Expected Fraction (GH Expected)")
        plt.ylabel("Deviation from Expected")
        plt.xlabel("Cluster ID")
        plt.xticks(rotation=90)

        plt.tight_layout()
        path = os.path.join(self.directory, 'deviations.png')  
        plt.savefig(path, dpi=300, bbox_inches='tight')
        path = os.path.join(self.directory, 'figure_editable.pdf')  
        plt.savefig(path, format="pdf", bbox_inches="tight", dpi=300, transparent=True)
        plt.close()





    
    ## METHOD SPEED: CALCULATES SPEED

    def speed(self):

        df = self.df
        cluster_name = self.cluster_name

        output = os.path.join(self.directory, "speed")
        os.makedirs(output, exist_ok=True)

        ## RAW SPEED

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


            save_path = os.path.join(output, f"cluster_{cluster_id}_speed_raw_grid.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        ### MEAN SPEED OVER TIME
        cluster_ids = sorted(df[cluster_name].unique())
        num_clusters = len(cluster_ids)
        cols = 5
        rows = int(np.ceil(num_clusters / cols))

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
        save_path = os.path.join(output, "mean_speed_summary.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        ## MEAN DIFFERENCE OVER TIME
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
        save_path = os.path.join(output, "speed_difference_summary.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    

    ## METHOD ACCELERATION: CALCULATES ACCELERATION
    def acceleration(self):

        df = self.df
        cluster_name = self.cluster_name

        output = os.path.join(self.directory, "acceleration")
        os.makedirs(output, exist_ok=True)

        ## RAW ACCELERATION 

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

        
            save_path = os.path.join(output, f"cluster_{cluster_id}_accleration_raw_grid.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        

        ## MEAN ACCELERATION

        cluster_ids = sorted(df[cluster_name].unique())
        num_clusters = len(cluster_ids)
        cols = 5
        rows = int(np.ceil(num_clusters / cols))

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
        save_path = os.path.join(output, "mean_accleration_summary.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close() 

        ## ACCELERATION DIFFERENCE OVER TIME
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
        save_path = os.path.join(output, "accleration_difference_summary.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


    #### METHOD HEADING_ANGLE: CALCULATES HEADING ANGLES track1_angle
    def heading_angle(self):

        df = self.df
        cluster_name = self.cluster_name

        output = os.path.join(self.directory, 'angle')
        os.makedirs(output, exist_ok=True)

        ## RAW ANGLES 
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

            angle = os.path.join(output, "angle")
            os.makedirs(angle, exist_ok=True)
            save_path = os.path.join(output, f"cluster_{cluster_id}_heading-angle_raw_grid.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
    

        ## AVERAGE HEADING ANGLE 

        cluster_ids = sorted(df[cluster_name].unique())
        num_clusters = len(cluster_ids)
        cols = 5
        rows = int(np.ceil(num_clusters / cols))

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
        save_path = os.path.join(output, "mean_heading-angle_summary.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close() 


        ## DIFFERENCE IN HEADING ANGLE 
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
        save_path = os.path.join(output, "heading-angle_difference_summary.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


    #### METHOD APPROACH_ANGLE: track1_approach_angle
    def approach_angle(self):

        df = self.df
        cluster_name = self.cluster_name

        output = os.path.join(self.directory, 'angle')
        os.makedirs(output, exist_ok=True)

        ## RAW 
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

            save_path = os.path.join(output, f"cluster_{cluster_id}_approach-angle_raw_grid.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
    
        ## MEAN APPROACH ANGLE 

        cluster_ids = sorted(df[cluster_name].unique())
        num_clusters = len(cluster_ids)
        cols = 5
        rows = int(np.ceil(num_clusters / cols))

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
        save_path = os.path.join(output, "mean_approach-angle_summary.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close() 

        ## DIFFERENCE IN APPROACH ANGLE 
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
        save_path = os.path.join(output, "approach-angle_difference_summary.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    #### METHOD BODY_DISTANCE: DISTANCE BETWEEN BODY-BODY NODES
    def body_distance(self):

        df = self.df
        cluster_name = self.cluster_name

        output = os.path.join(self.directory, 'body-body-distance')
        os.makedirs(output, exist_ok=True)


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

            save_path = os.path.join(output, f"cluster_{cluster_id}_distance_grid.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        ## AVERAGE BODY DISTANCE
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
        output_path = os.path.join(output, "summary.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


    #### MEHTOD MIN_DISTANCE: MIN DISTANCE BETWEEN ANY NODE COMBO
    def min_distance(self):

        df = self.df
        cluster_name = self.cluster_name

        output = os.path.join(self.directory, 'min-distance')
        os.makedirs(output, exist_ok=True)

        ## RAW
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

            save_path = os.path.join(output, f"cluster_{cluster_id}_min_distance_grid.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        

        ## AVERAGE MIN DISTANCE
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
        output_path = os.path.join(output, "summary.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    #### METHOD DISTANCE: DISTANCE TRAVELLED
    def distance(self):

        df = self.df
        cluster_name = self.cluster_name

        output = os.path.join(self.directory, 'distance-travelled')
        os.makedirs(output, exist_ok=True)


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

        # Plot violin
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df_distances, x="cluster", y="distance", inner="box", cut=0, palette="Set2")
        plt.title("Track-wise Distance Travelled per Cluster")
        plt.xlabel("Cluster")
        plt.ylabel("Distance Travelled")
        plt.tight_layout()

        plt.savefig(os.path.join(output, "distance_violin.png"), dpi=300)
        plt.close()

    
        ## MEAN DISTANCE TRAVELLED OVER TIME

        mean_steps = []

        for cluster_id in sorted(df[cluster_name].unique()):
            cluster_df = df[df[cluster_name] == cluster_id]

            for inter_id in cluster_df["interaction_id"].unique():
                inter_df = cluster_df[cluster_df["interaction_id"] == inter_id].sort_values("Frame")

                frames = inter_df["Normalized Frame"].values
                x1 = inter_df["mm_Track_1 x_body"].values
                y1 = inter_df["mm_Track_1 y_body"].values
                x2 = inter_df["mm_Track_2 x_body"].values
                y2 = inter_df["mm_Track_2 y_body"].values

                d1 = np.sqrt(np.diff(x1)**2 + np.diff(y1)**2)
                d2 = np.sqrt(np.diff(x2)**2 + np.diff(y2)**2)
                mean_step = (d1 + d2) / 2
                frames = frames[1:]

                for f, d in zip(frames, mean_step):
                    mean_steps.append({
                        "cluster": cluster_id,
                        "interaction_id": inter_id,
                        "Normalized Frame": f,
                        "mean_step_distance": d
                    })

        df_mean_step = pd.DataFrame(mean_steps)

        cluster_ids = sorted(df_mean_step["cluster"].unique())
        cols = 5
        rows = int(np.ceil(len(cluster_ids) / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), sharex=True, sharey=True)
        axes = axes.flatten()

        for idx, cluster_id in enumerate(cluster_ids):
            ax = axes[idx]
            cluster_data = df_mean_step[df_mean_step["cluster"] == cluster_id]
            grouped = cluster_data.groupby("Normalized Frame")["mean_step_distance"]

            mean_vals = grouped.mean()
            std_vals = grouped.std()

            ax.plot(mean_vals.index, mean_vals.values, color='purple')
            ax.fill_between(mean_vals.index, mean_vals - std_vals, mean_vals + std_vals, color='violet', alpha=0.3)
            ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
            ax.set_ylim(0, 3)
            ax.set_title(f"Cluster {cluster_id}")
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Clean unused subplots
        for i in range(len(cluster_ids), len(axes)):
            axes[i].axis('off')

        # Save and close
        plt.suptitle("Mean ± SD Distance Travelled per Frame (Track 1+2 Avg)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        save_path = os.path.join(output, "mean_distance.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()



    #### METHOD CONTACT: DURATION AND TYPE OF CONTACT
    def contact(self):

        df = self.df
        cluster_name = self.cluster_name

        output = os.path.join(self.directory, 'contacts')
        os.makedirs(output, exist_ok=True)


        interaction_colors = {
        "head-head": "red",
        "head-body": "orange",
        "head-tail": "yellow",
        "body-body": "black",
        "tail-tail": "green",
        "tail-body": "purple"}

        # raw -> merged (symmetric) labels
        interaction_merge_map = {
            "head-tail": "head-tail", "tail-head": "head-tail",
            "tail-body": "tail-body", "body-tail": "tail-body",
            "head-body": "head-body", "body-head": "head-body",
            "tail-tail": "tail-tail", "head-head": "head-head", "body-body": "body-body",
        }

        # canonical order used in plots/tables
        interaction_types = ["head-head", "head-body", "head-tail", "body-body", "tail-tail", "tail-body"]


        ## RASTA
        for cluster_id in sorted(df[cluster_name].unique()):
            cluster_df = df[df[cluster_name] == cluster_id]
            all_ids    = cluster_df['interaction_id'].unique()
            
            # 1) Sample *up to* 120 interactions at random
            if len(all_ids) > 120:
                sample_ids = np.random.choice(all_ids, 120, replace=False)
            else:
                sample_ids = all_ids.copy()
            
            # 2) Decide how tall to make the figure:
            #    – each row takes 0.2" of height
            #    – but clamp between 5" and 20"
            row_h    = 0.2
            raw_h    = len(sample_ids) * row_h
            fig_h    = min(max(raw_h, 5), 20)      # between 5" and 20"
            fig_w    = 12                         # your usual width
            
            plt.figure(figsize=(fig_w, fig_h), dpi=300)
            ax = plt.gca()
            
            for i, interaction_id in enumerate(sample_ids):
                sub = cluster_df[cluster_df['interaction_id'] == interaction_id]
                # for _, row in sub.iterrows():
                #     if row["min_distance"] < 1 and row["interaction_type"] in interaction_colors:
                #         c = interaction_colors[row["interaction_type"]]
                #         ax.plot(
                #         [row["Normalized Frame"], row["Normalized Frame"]],
                #         [i-0.4, i+0.4],
                #         color=c, linewidth=2
                #         )
                for _, row in sub.iterrows():
                    md = row.get("min_distance", np.nan)
                    typ_raw = row.get("interaction_type", np.nan)
                    if pd.isna(md) or pd.isna(typ_raw):
                        continue
                    if md < 1:
                        typ = interaction_merge_map.get(typ_raw)  # merge before coloring
                        if typ in interaction_colors:
                            ax.plot(
                                [row["Normalized Frame"], row["Normalized Frame"]],
                                [i - 0.4, i + 0.4],
                                color=interaction_colors[typ],
                                linewidth=2
                            )

            
            ax.set_yticks(np.arange(len(sample_ids)))
            ax.set_yticklabels(sample_ids, fontsize=6)
            ax.set_xlabel("Normalized Frame")
            ax.set_ylabel("Interaction ID")
            ax.set_title(f"Raster Plot of Contact Events - Cluster {cluster_id}")
            ax.grid(True, linestyle="--", alpha=0.3)
            
            plt.tight_layout()
        
            raster_path = os.path.join(output, f"cluster_{cluster_id}_raster.png")
            plt.savefig(raster_path, dpi=300, bbox_inches='tight')
            plt.close()


        ## CONTACT SUMMARY

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
        plt.savefig(os.path.join(output, "contact_frames.png"), dpi=300)
        plt.close()



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
        for idx, cluster_id in enumerate(sorted(df_contacts["cluster"].unique())):

            subset = df_contacts[df_contacts["cluster"] == cluster_id]

            means = subset[interaction_types].mean()
            stds  = subset[interaction_types].std()

            fig, ax = plt.subplots(figsize=(6,4))
            x = np.arange(len(interaction_types))

            ax.bar(
                x,
                means.values,
                yerr=stds.values,
                capsize=5,
                color=[interaction_colors[it] for it in interaction_types],
                alpha=0.8
            )

            ax.set_xticks(x)
            ax.set_xticklabels(interaction_types, rotation=45, ha="right")
            ax.set_ylabel("Avg. Frames per Interaction")
            ax.set_title(f"Cluster {cluster_id} – Avg. Frames <1mm by Type")
            ax.set_ylim(0, (means + stds).max() * 1.1)

            plt.tight_layout()

            save_path = os.path.join(output, f"cluster_{cluster_id}_interaction_type.png")
            plt.savefig(save_path, dpi=300)
            plt.close()


    
    #### METHOD SUMMARY_ANCHOR_PARTNER: SUMMARY QUANTIFICATIONS ANCHOR/PARTNER
    def summary_anchor_partner(self):

        df = self.df
        cluster_name = self.cluster_name 

        cluster_ids = sorted(df[cluster_name].unique())
        n_clusters = len(cluster_ids)
        n_rows = 13  # number of summary plots (trajectory, speed, accel, angle, etc.)

        # Create summary canvas
        # fig_ap, axes_ap = plt.subplots(n_rows, n_clusters, figsize=(n_clusters * 4, n_rows * 2))

        # width per column and height per "unit"
        width_per_col  = 4    # you already had n_clusters*4
        height_per_unit = 1.5

        # Row 0 gets 3 units, rows 1–5 get 1 each → total units = 3 + 5*1 = 8
        height_ratios = [3] + [1]*(n_rows-1) ## want mean trajectory to get 3 times the space as the other rows 
        total_units   = sum(height_ratios)          # = 8
        fig_w = n_clusters * width_per_col          # unchanged
        fig_h = total_units * height_per_unit       # 8 * 1.5 = 12"

        fig_ap, axes_ap = plt.subplots(
        n_rows,
        n_clusters,
        figsize=(fig_w, fig_h),
        gridspec_kw={'height_ratios': height_ratios},
        constrained_layout=True
    )

        if n_clusters == 1:
            axes_ap = axes_ap.reshape(n_rows, 1)

        # Mark all as invisible initially
        for ax in axes_ap.flatten():
            ax.set_visible(False)

        row_labels = [
            "Mean Trajectory",           # 0
            "Speed",                     # 1
            "Acceleration",              # 2
            "Heading Angle",             # 3
            "Heading Angle Change",      # 4
            "Approach Angle",            # 5
            "Approach Angle Change",     # 6
            "Distance Travelled",        # 7
            "Minimum Distance",          # 8
            "Interaction Type",          # 9
            "Initial Contact",       # 10  <-- new
            "Predominant Contact",   # 11  <-- new
            "Contact Frames <1mm"        # 12  (moved down from 10)
        ]


        for i, label in enumerate(row_labels):
            ax_label = axes_ap[i, 0]  # first column of each row
            ax_label.set_ylabel(label, fontsize=10, rotation=0, labelpad=40, va='center')


        df['anchor_distance'] = df.groupby('interaction_id').apply(
        lambda x: np.sqrt((x['anchor x_body'].diff()**2 + x['anchor y_body'].diff()**2))).reset_index(level=0, drop=True)

        df['partner_distance'] =  df.groupby('interaction_id').apply(
        lambda x: np.sqrt((x['partner x_body'].diff()**2 + x['partner y_body'].diff()**2))).reset_index(level=0, drop=True)
        

        for column, cluster_id in enumerate(cluster_ids):
            cluster_df = df[df[cluster_name] == cluster_id]

            ## 0. MEAN TRAJECTORIES

            ax0 = axes_ap[0, column]
            grouped = cluster_df.groupby("Normalized Frame")

            t1_x = grouped["anchor x_body"].mean()
            t1_y = grouped["anchor y_body"].mean()
            t2_x = grouped["partner x_body"].mean()
            t2_y = grouped["partner y_body"].mean()

            t1_x_std = grouped["anchor x_body"].std()
            t1_y_std = grouped["anchor y_body"].std()
            t2_x_std = grouped["partner x_body"].std()
            t2_y_std = grouped["partner y_body"].std()

                    # Combine into a DataFrame
            sd_summary = pd.DataFrame({
                "Normalized Frame": t1_y_std.index,
                "t1_y_std": t1_y_std.values,
                "t2_y_std": t2_y_std.values,
                "t1_x_std": t1_x_std.values,
                "t2_x_std": t2_x_std.values,
            })

            # Save to CSV
            # sd_summary.to_csv(os.path.join(output_dir, "std_trajectory_summary.csv"), index=False)


            ax0.plot(t1_x, t1_y, label="Anchor", color="blue")
            ax0.plot(t2_x, t2_y, label="Partner", color="orange")

            ax0.scatter(t1_x.iloc[0], t1_y.iloc[0], color="blue", marker="o", label="Anchor Start")
            ax0.scatter(t2_x.iloc[0], t2_y.iloc[0], color="orange", marker="o", label="Partner Start")

                # error bars in X and Y
            # ax0.errorbar(
            #     t1_x, t1_y,
            #     xerr=t1_x_std, yerr=t1_y_std,
            #     fmt="none", ecolor="blue", alpha=0.3, label="Anchor ±1 SD"
            # )
            # ax0.errorbar(
            #     t2_x, t2_y,
            #     xerr=t2_x_std, yerr=t2_y_std,
            #     fmt="none", ecolor="orange", alpha=0.3, label="Partner ±1 SD"
            # )

            ax0.errorbar(
                    t1_x.values, t1_y.values,
                    xerr=t1_x_std.values, yerr=t1_y_std.values,
                    fmt="none", ecolor="blue", alpha=0.3, label="Anchor ±1 SD"
                )
            
            ax0.errorbar(
                    t2_x.values, t2_y.values,
                    xerr=t2_x_std.values, yerr=t2_y_std.values,
                    fmt="none", ecolor="orange", alpha=0.3, label="Partner ±1 SD"
                )



            # ax0.set_xticks([])
            # ax_sum.set_yticks([])
            ax0.set_aspect('equal', 'box')
            ax0.set_title(f"Cluster {cluster_id}", fontsize=8)
            ax0.set_visible(True)


            ## 1. SPEED
            ax1 = axes_ap[1, column]

            sns.lineplot(data=cluster_df, x='Normalized Frame', y='anchor_speed', label='Anchor', ci='sd', color='blue', ax=ax1)
            sns.lineplot(data=cluster_df, x='Normalized Frame', y='partner_speed', label='Partner', ci='sd', color='orange', ax=ax1)

            ax1.axvline(0, color="gray", ls="--", lw=0.5)
            ax1.set_ylim(0, 2)
            ax1.set_xticks([])
            # ax1.set_yticks([])
            ax1.set_visible(True)

            ## 2. ACCELERATION
            ax2 = axes_ap[2, column]

            sns.lineplot(data=cluster_df, x='Normalized Frame', y='anchor_acceleration', label='Anchor', ci='sd', color='blue', ax=ax2)
            sns.lineplot(data=cluster_df, x='Normalized Frame', y='partner_acceleration', label='Partner', ci='sd', color='orange', ax=ax2)

            ax2.axvline(0, color="gray", ls="--", lw=0.5)
            ax2.set_ylim(-1, 1)
            ax2.set_xticks([])
            # ax1.set_yticks([])
            ax2.set_visible(True)

            ## 3. HEADING ANGLE
            ax3 = axes_ap[3, column]

            sns.lineplot(data=cluster_df, x='Normalized Frame', y='anchor_angle', label='Anchor', ci='sd', color='blue', ax=ax3)
            sns.lineplot(data=cluster_df, x='Normalized Frame', y='partner_angle', label='Partner', ci='sd', color='orange', ax=ax3)

            ax3.axvline(0, color="gray", ls="--", lw=0.5)
            ax3.set_ylim(0, 180)
            ax3.set_xticks([])
            # ax1.set_yticks([])
            ax3.set_visible(True)

            ## 4. HEADING ANGLE CHANGE
            ax4 = axes_ap[4, column]

            sns.lineplot(data=cluster_df, x='Normalized Frame', y='anchor_heading_angle_change', label='Anchor', ci='sd', color='blue', ax=ax4)
            sns.lineplot(data=cluster_df, x='Normalized Frame', y='partner_heading_angle_change', label='Partner', ci='sd', color='orange', ax=ax4)

            ax4.axvline(0, color="gray", ls="--", lw=0.5)
            ax4.set_ylim(0, 60)
            ax4.set_xticks([])
            ax4.set_visible(True)

            ## 4. APPROACH ANGLE
            ax5 = axes_ap[5, column]

            sns.lineplot(data=cluster_df, x='Normalized Frame', y='anchor_approach_angle', label='Anchor', ci='sd', color='blue', ax=ax5)
            sns.lineplot(data=cluster_df, x='Normalized Frame', y='partner_approach_angle', label='Partner', ci='sd', color='orange', ax=ax5)

            ax5.axvline(0, color="gray", ls="--", lw=0.5)
            ax5.set_ylim(0, 180)
            ax5.set_xticks([])
            # ax1.set_yticks([])
            ax5.set_visible(True)

            ## 6. APPROACH ANGLE CHANGE
            ax6 = axes_ap[6, column]

            sns.lineplot(data=cluster_df, x='Normalized Frame', y='anchor_approach_angle_change', label='Anchor', ci='sd', color='blue', ax=ax6)
            sns.lineplot(data=cluster_df, x='Normalized Frame', y='partner_approach_angle_change', label='Partner', ci='sd', color='orange', ax=ax6)

            ax6.axvline(0, color="gray", ls="--", lw=0.5)
            ax6.set_ylim(0, 60)
            ax6.set_xticks([])
            ax6.set_visible(True)

            ## 7. DISTANCE TRAVELLED
            ax7 = axes_ap[7, column]

            sns.lineplot(data=cluster_df, x='Normalized Frame', y='anchor_distance', label='Anchor', ci='sd', color='blue', ax=ax7)
            sns.lineplot(data=cluster_df, x='Normalized Frame', y='partner_distance', label='Partner', ci='sd', color='orange', ax=ax7)

            ax7.axvline(0, color="gray", ls="--", lw=0.5)
            ax7.set_ylim(0, 30)
            # ax5.set_xticks([])
            # ax1.set_yticks([])
            ax7.set_visible(True)

            ## 8. MIN DISTANCE BETWEEN

            ax8 = axes_ap[8, column]
            grouped_min = cluster_df.groupby("Normalized Frame")["min_distance"]
            mean_min = grouped_min.mean()
            std_min  = grouped_min.std()

            ax8.plot(mean_min.index, mean_min.values, color='black')
            ax8.fill_between(
                mean_min.index,
                mean_min - std_min,
                mean_min + std_min,
                color='gray',
                alpha=0.3
            )
            ax8.axvline(0, color='red', linestyle='--', linewidth=0.5)
            ax8.set_ylim(0, 20)
            ax8.set_xticks([])
            ax8.set_visible(True)

            # ---- 9–12. CONTACT SUMMARY (match standalone) ----
            interaction_colors = {
                "head-head": "red",
                "head-body": "orange",
                "head-tail": "yellow",
                "body-body": "black",
                "tail-tail": "green",
                "tail-body": "purple",
            }
            interaction_merge_map = {
                "head-tail": "head-tail", "tail-head": "head-tail",
                "tail-body": "tail-body", "body-tail": "tail-body",
                "head-body": "head-body", "body-head": "head-body",
                "tail-tail": "tail-tail", "head-head": "head-head", "body-body": "body-body",
            }
            interaction_types = ["head-head", "head-body", "head-tail", "body-body", "tail-tail", "tail-body"]
            palette_list = [interaction_colors[t] for t in interaction_types]

            inter_ids = cluster_df["interaction_id"].unique()
            records = []
            init_labels = []
            pred_labels = []
            frames_close_list = []

            for inter_id in inter_ids:
                inter = cluster_df[cluster_df["interaction_id"] == inter_id].sort_values("Frame")

                # frames in contact (<1mm), merge symmetric labels
                close = inter[inter["min_distance"] < 1].copy()
                close["interaction_type_merged"] = close["interaction_type"].map(interaction_merge_map)

                # counts per interaction (merged types)
                counts = close["interaction_type_merged"].value_counts().to_dict()
                row = {"interaction_id": inter_id}
                for it in interaction_types:
                    row[it] = counts.get(it, 0)
                records.append(row)

                # initial & predominant labels, if any contact exists
                tm = close["interaction_type_merged"]
                if not tm.empty:
                    init_labels.append(tm.iloc[0])
                    pred_labels.append(tm.value_counts().idxmax())

                # total frames <1mm for this interaction
                frames_close_list.append((inter_id, (inter["min_distance"] < 1).sum()))

            # ---------- ROW 9: Interaction Type (mean ± sd frames per interaction) ----------
            ax9 = axes_ap[9, column]
            df_counts = pd.DataFrame(records)
            means = df_counts[interaction_types].mean()
            stds  = df_counts[interaction_types].std()

            x = np.arange(len(interaction_types))
            ax9.bar(x, means.values, yerr=stds.values, capsize=5,
                    color=[interaction_colors[it] for it in interaction_types], alpha=0.8)
            ax9.set_xticks(x)
            ax9.set_xticklabels(interaction_types, rotation=45, fontsize=6)
            ax9.set_ylim(0, (means + stds).max() * 1.1 if len(means) else 1)
            ax9.set_xticks([])
            ax9.set_visible(True)

            # ---------- ROW 10: Initial Contact (%) ----------
            ax10 = axes_ap[10, column]
            if len(init_labels):
                tmp_init = pd.DataFrame({
                    "contact_type": np.repeat(interaction_types, len(init_labels)),
                    "val": np.concatenate([(np.array(init_labels) == t).astype(int) for t in interaction_types])
                })
            else:
                tmp_init = pd.DataFrame({"contact_type": interaction_types, "val": np.zeros(len(interaction_types), dtype=int)})

            sns.barplot(
                data=tmp_init, x="contact_type", y="val",
                estimator=np.mean, errorbar="sd",
                order=interaction_types, palette=palette_list, ax=ax10
            )
            ax10.set_ylim(0, 1)
            ax10.set_xticks([])
            ax10.set_visible(True)

            # ---------- ROW 11: Predominant Contact (%) ----------
            ax11 = axes_ap[11, column]
            if len(pred_labels):
                tmp_pred = pd.DataFrame({
                    "contact_type": np.repeat(interaction_types, len(pred_labels)),
                    "val": np.concatenate([(np.array(pred_labels) == t).astype(int) for t in interaction_types])
                })
            else:
                tmp_pred = pd.DataFrame({"contact_type": interaction_types, "val": np.zeros(len(interaction_types), dtype=int)})

            sns.barplot(
                data=tmp_pred, x="contact_type", y="val",
                estimator=np.mean, errorbar="sd",
                order=interaction_types, palette=palette_list, ax=ax11
            )
            ax11.set_ylim(0, 1)
            ax11.set_xticks([])
            ax11.set_visible(True)

            # ---------- ROW 12: Contact Frames <1mm (mean ± sd) ----------
            ax12 = axes_ap[12, column]
            frames_vals = pd.Series([v for _, v in frames_close_list])
            mean_val = float(frames_vals.mean()) if len(frames_vals) else 0.0
            std_val  = float(frames_vals.std())  if len(frames_vals) else 0.0

            ax12.bar(0, mean_val, yerr=std_val, color='green', alpha=0.8, capsize=5)
            ax12.text(0, mean_val + 1, f"{mean_val:.1f}", ha='left', fontsize=12)
            ax12.set_ylim(0, 15)
            ax12.set_xticks([])
            ax12.set_visible(True)


        out_path = os.path.join(self.directory, "summary_anchor_partner.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig_ap)



    #### METHOD SUMMARY_QUANTIFICATIONS: SUMMARY QUANTIFICATIONS PER CLUSTER
    def summary_quantifications(self):
        
        df = self.df
        cluster_name = self.cluster_name 

        cluster_ids = sorted(df[cluster_name].unique())
        n_clusters = len(cluster_ids)
        n_rows = 10  # number of summary plots (trajectory, speed, accel, angle, etc.)

        # Create summary canvas
        fig_sum, axes_sum = plt.subplots(n_rows, n_clusters, figsize=(n_clusters * 4, n_rows * 2))
        if n_clusters == 1:
            axes_sum = axes_sum.reshape(n_rows, 1)

        # Mark all as invisible initially
        for ax in axes_sum.flatten():
            ax.set_visible(False)

        row_labels = [
        "Mean Speed",
        "Mean Acceleration",
        "Heading Angle Mean",
        "Heading Angle Difference",
        "Approach Angle",
        "Approach Angle Difference",
        "Min Distance",
        "Mean Distance Travelled",
        "Interaction Type",
        "Contact Frames"
    ]

        for i, label in enumerate(row_labels):
            ax_label = axes_sum[i, 0]  # first column of each row
            ax_label.set_ylabel(label, fontsize=10, rotation=0, labelpad=40, va='center')
        
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_df = df[df[cluster_name] == cluster_id].copy()

            ## 0. SPEED

            cluster_df["mean_speed"] = (cluster_df["track1_speed"] + cluster_df["track2_speed"]) / 2

            grouped = cluster_df.groupby("Normalized Frame")["mean_speed"]
            mean_vals = grouped.mean()
            std_vals = grouped.std()

            ax_sum = axes_sum[0, idx]
            ax_sum.plot(mean_vals.index, mean_vals.values, color='darkorange')
            ax_sum.fill_between(mean_vals.index, mean_vals - std_vals, mean_vals + std_vals, color='orange', alpha=0.3)
            ax_sum.axvline(0, color='gray', linestyle='--', linewidth=0.5)
            ax_sum.set_ylim(0, 2)
            ax_sum.set_xticks([])
            ax_sum.set_title(f"Cluster {cluster_id}", fontsize=8)
            ax_sum.set_visible(True)

            ## 1. ACCELERATION 

            cluster_df["mean_accel"] = (cluster_df["track1_acceleration"] + cluster_df["track2_acceleration"]) / 2
            grouped = cluster_df.groupby("Normalized Frame")["mean_accel"]
            mean_vals = grouped.mean()
            std_vals = grouped.std()

            ax_sum = axes_sum[1, idx]
            ax_sum.plot(mean_vals.index, mean_vals.values, color='darkorange')
            ax_sum.fill_between(mean_vals.index, mean_vals - std_vals, mean_vals + std_vals, color='orange', alpha=0.3)
            ax_sum.axvline(0, color='gray', linestyle='--', linewidth=0.5)
            ax_sum.set_ylim(-1, 1)
            ax_sum.set_xticks([])
            ax_sum.set_visible(True)

            ## 2. MEAN HEADING ANGLE 

            cluster_df["angle_mean"] = (cluster_df["track1_angle"] + cluster_df["track2_angle"]) / 2
            grouped = cluster_df.groupby("Normalized Frame")["angle_mean"]
            mean_vals = grouped.mean()
            std_vals = grouped.std()

            ax_sum = axes_sum[2, idx]
            ax_sum.plot(mean_vals.index, mean_vals.values, color='darkorange')
            ax_sum.fill_between(mean_vals.index, mean_vals - std_vals, mean_vals + std_vals, color='orange', alpha=0.3)
            ax_sum.axvline(0, color='gray', linestyle='--', linewidth=0.5)
            ax_sum.set_ylim(0, 180)
            ax_sum.set_xticks([])
            # ax_sum.set_yticks([])
            ax_sum.set_visible(True)

            ## 3. HEADING ANGLE DIFFERENCE

            cluster_df["angle_diff"] = np.abs(cluster_df["track1_angle"] - cluster_df["track2_angle"])
            grouped = cluster_df.groupby("Normalized Frame")["angle_diff"]
            mean_vals = grouped.mean()
            std_vals = grouped.std()

            ax_sum = axes_sum[3, idx]
            ax_sum.plot(mean_vals.index, mean_vals.values, color='navy')
            ax_sum.fill_between(mean_vals.index, mean_vals - std_vals, mean_vals + std_vals, color='blue', alpha=0.3)
            ax_sum.axvline(0, color='gray', linestyle='--', linewidth=0.5)
            ax_sum.set_ylim(0, 80)
            ax_sum.set_xticks([])
            # ax_sum.set_yticks([])
            ax_sum.set_visible(True)

            ## 4. MEAN APPROACH ANGLE 

            cluster_df["mean_angle"] = (cluster_df["track1_approach_angle"] + cluster_df["track2_approach_angle"]) / 2
            grouped = cluster_df.groupby("Normalized Frame")["mean_angle"]
            mean_vals = grouped.mean()
            std_vals = grouped.std()

            ax_sum = axes_sum[4, idx]
            ax_sum.plot(mean_vals.index, mean_vals.values, color='darkorange')
            ax_sum.fill_between(mean_vals.index, mean_vals - std_vals, mean_vals + std_vals, color='orange', alpha=0.3)
            ax_sum.axvline(0, color='gray', linestyle='--', linewidth=0.5)
            ax_sum.set_ylim(0, 180)
            ax_sum.set_xticks([])
            # ax_sum.set_yticks([])
            ax_sum.set_visible(True)

            ## 5. DIFFERENCE APPROACH ANGLE

            cluster_df["diff_angle"] = np.abs(cluster_df["track1_approach_angle"] - cluster_df["track2_approach_angle"])
            grouped = cluster_df.groupby("Normalized Frame")["diff_angle"]
            mean_vals = grouped.mean()
            std_vals = grouped.std()

            ax_sum = axes_sum[5, idx]
            ax_sum.plot(mean_vals.index, mean_vals.values, color='darkorange')
            ax_sum.fill_between(mean_vals.index, mean_vals - std_vals, mean_vals + std_vals, color='orange', alpha=0.3)
            ax_sum.axvline(0, color='gray', linestyle='--', linewidth=0.5)
            ax_sum.set_ylim(0, 180)
            ax_sum.set_xticks([])
            # ax_sum.set_yticks([])
            ax_sum.set_visible(True)

            ## 6. AVERAGE MIN DISTANCE

            grouped = cluster_df.groupby("Normalized Frame")["min_distance"]
            mean_dist = grouped.mean()
            std_dist = grouped.std()

            ax_sum = axes_sum[6, idx]
            ax_sum.plot(mean_dist.index, mean_dist.values, color='black')
            ax_sum.fill_between(mean_dist.index, mean_dist - std_dist, mean_dist + std_dist, color='gray', alpha=0.3)
            ax_sum.axvline(0, color='red', linestyle='--', linewidth=0.5)
            ax_sum.set_ylim(0, 20)
            ax_sum.set_xticks([])
            # ax_sum.set_yticks([])
            ax_sum.set_visible(True)

            ## 7. MEAN DISTANCE TRAVELLED

            # grouped = cluster_df.groupby("Normalized Frame")["mean_step_distance"]
            # mean_vals = grouped.mean()
            # std_vals = grouped.std()

            # ax_sum = axes_sum[7, idx]
            # ax_sum.plot(mean_vals.index, mean_vals.values, color='purple')
            # ax_sum.fill_between(mean_vals.index, mean_vals - std_vals, mean_vals + std_vals, color='violet', alpha=0.3)
            # ax_sum.axvline(0, color='gray', linestyle='--', linewidth=0.5)
            # ax_sum.set_ylim(0, 2)
            # ax_sum.set_xticks([])
            # # ax_sum.set_yticks([])
            # ax_sum.set_visible(True)


            inter_steps = []

            for inter_id in cluster_df["interaction_id"].unique():
                inter_df = cluster_df[cluster_df["interaction_id"] == inter_id].sort_values("Frame")

                frames = inter_df["Normalized Frame"].values
                x1 = inter_df["mm_Track_1 x_body"].values
                y1 = inter_df["mm_Track_1 y_body"].values
                x2 = inter_df["mm_Track_2 x_body"].values
                y2 = inter_df["mm_Track_2 y_body"].values

                if len(x1) < 2 or len(x2) < 2:
                    continue

                d1 = np.sqrt(np.diff(x1)**2 + np.diff(y1)**2)
                d2 = np.sqrt(np.diff(x2)**2 + np.diff(y2)**2)
                mean_step = (d1 + d2) / 2
                frames = frames[1:]

                inter_steps.append(
                    pd.DataFrame({
                        "interaction_id": inter_id,
                        "Normalized Frame": frames,
                        "mean_step_distance": mean_step
                    })
                )


            tmp_steps = pd.concat(inter_steps, ignore_index=True)
            grouped = tmp_steps.groupby("Normalized Frame")["mean_step_distance"]
            mean_vals = grouped.mean()
            std_vals  = grouped.std()


            # --- then plot Row 7 using mean_vals/std_vals as you already do ---
            ax7 = axes_sum[7, idx]  # or axes_sum[7, idx] in summary_quantifications
            ax7.plot(mean_vals.index, mean_vals.values)
            ax7.fill_between(mean_vals.index, mean_vals - std_vals, mean_vals + std_vals, alpha=0.3)
            ax7.axvline(0, color="gray", ls="--", lw=0.5)
            ax7.set_ylim(0, 2)  # your choice
            ax7.set_xticks([])
            ax7.set_visible(True)


            ## 8-9 INTERACTION TYPE AND AVERAGE CONTACT FRAMES

            interaction_colors = {
                "head-head": "red",
                "head-body": "orange",
                "head-tail": "yellow",
                "body-body": "black",
                "tail-tail": "green",
                "tail-body": "purple"
            }
            interaction_merge_map = {
                "head-tail": "head-tail", "tail-head": "head-tail",
                "tail-body": "tail-body", "body-tail": "tail-body",
                "head-body": "head-body", "body-head": "head-body",
                "tail-tail": "tail-tail", "head-head": "head-head", "body-body": "body-body",
            }
            interaction_types = ["head-head", "head-body", "head-tail", "body-body", "tail-tail", "tail-body"]

            inter_ids = cluster_df["interaction_id"].unique()
            records = []
            frames_close_list = []

            for inter_id in inter_ids:
                inter = cluster_df[cluster_df["interaction_id"] == inter_id].sort_values("Frame")

                # frames in contact (<1mm) — merge symmetric labels like standalone
                close = inter[inter["min_distance"] < 1].copy()
                close["interaction_type_merged"] = close["interaction_type"].map(interaction_merge_map)

                counts = close["interaction_type_merged"].value_counts().to_dict()

                row = {"interaction_id": inter_id}
                for it in interaction_types:
                    row[it] = counts.get(it, 0)
                records.append(row)

                frames_close_list.append((inter_id, (inter["min_distance"] < 1).sum()))

            # Row 8: Interaction type counts (mean ± sd)
            ax9 = axes_sum[8, idx]
            df_counts = pd.DataFrame(records)
            means = df_counts[interaction_types].mean()
            stds  = df_counts[interaction_types].std()

            x = np.arange(len(interaction_types))
            ax9.bar(
                x, means.values, yerr=stds.values, capsize=5,
                color=[interaction_colors[it] for it in interaction_types], alpha=0.8
            )
            ax9.set_xticks(x)
            ax9.set_xticklabels(interaction_types, rotation=45, fontsize=6)
            ax9.set_ylim(0, (means + stds).max() * 1.1)
            ax9.set_xticks([])   # match your style
            ax9.set_visible(True)

            # Row 9: Contact frames <1mm (mean ± sd)
            ax12 = axes_sum[9, idx]
            frames_vals = pd.Series([v for _, v in frames_close_list])
            mean_val = frames_vals.mean()
            std_val  = frames_vals.std()

            ax12.bar(0, mean_val, yerr=std_val, color='green', alpha=0.8, capsize=5)
            ax12.text(0, mean_val + 1, f"{mean_val:.1f}", ha='left', fontsize=12)
            ax12.set_ylim(0, 15)
            ax12.set_xticks([])
            ax12.set_visible(True)


        out_path = os.path.join(self.directory, "summary.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig_sum)
    
    
    #### METHOD MEAN_TRACES: HEAD-BODY-TAIL MEAN TRACES
    def mean_traces(self):

        df = self.df
        cluster_name = self.cluster_name

        # fixed parameters
        xlim = (-50, 100)
        ylim = (-10, 300)
        anchor_color = "#4F7942"
        partner_color = "#916288"
        marker_size = 18

        # all clusters available
        clusters = sorted(df[cluster_name].unique())

        # helper to render one figure for a given node
        def render_figure(node_label: str, out_name: str):
            n = len(clusters)
            fig_w = max(8, n * 2.5)
            fig_h = 6
            fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h))
            if n == 1:
                axes = [axes]

            for ax, cluster_id in zip(axes, clusters):
                cluster_df = df[df[cluster_name] == cluster_id].copy()
                grouped = cluster_df.sort_values("Normalized Frame").groupby("Normalized Frame")

                # body positions for dashed lines
                t1_x = grouped["anchor x_body"].mean()
                t1_y = grouped["anchor y_body"].mean()
                t2_x = grouped["partner x_body"].mean()
                t2_y = grouped["partner y_body"].mean()

                # node-specific scatter (head/body/tail)
                a_x = grouped[f"anchor x_{node_label}"].mean()
                a_y = grouped[f"anchor y_{node_label}"].mean()
                p_x = grouped[f"partner x_{node_label}"].mean()
                p_y = grouped[f"partner y_{node_label}"].mean()

                # dashed body markers
                ax.plot(t1_x, t1_y, color=anchor_color, linestyle="None", marker="|", markersize=10, alpha=0.6)
                ax.plot(t2_x, t2_y, color=partner_color, linestyle="None", marker="_", markersize=10, alpha=0.6)

                # scatter for the chosen node
                ax.scatter(a_x, a_y, s=marker_size, alpha=0.85, color=anchor_color, label=f"Anchor {node_label}")
                ax.scatter(p_x, p_y, s=marker_size, alpha=0.85, color=partner_color, label=f"Partner {node_label}")

                # mark start points
                if len(t1_x) and len(t2_x):
                    ax.scatter(t1_x.iloc[0], t1_y.iloc[0], color=anchor_color, marker="o")
                    ax.scatter(t2_x.iloc[0], t2_y.iloc[0], color=partner_color, marker="o")

                # formatting
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
                ax.set_aspect('equal', adjustable='box')
                ax.set_title(f"Cluster {cluster_id}", fontweight='bold', fontsize=11)
                ax.axis('off')

            # legend: only node entries
            handles, labels = axes[0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            keep = [f"Anchor {node_label}", f"Partner {node_label}"]
            filtered = {lab: by_label[lab] for lab in keep if lab in by_label}
            if filtered:
                fig.legend(filtered.values(), filtered.keys(),
                        loc="center right", bbox_to_anchor=(1.05, 0.8),
                        fontsize=11, markerscale=1)

            plt.tight_layout()
            out_dir = os.path.join(self.directory, "mean_traces")
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(os.path.join(out_dir, out_name), dpi=300, bbox_inches='tight')
            plt.close(fig)

        # generate the three overviews
        render_figure("head", "overview_head.png")
        render_figure("body", "overview_body.png")
        render_figure("tail", "overview_tail.png")
    
    def mean_traces_gifs(self):
    
        df = self.df
        cluster_name = self.cluster_name

        # fixed look (same as your mean_traces)
        xlim = (-50, 100)
        ylim = (-10, 300)
        anchor_color = "#4F7942"
        partner_color = "#916288"
        marker_size = 70

        clusters = sorted(df[cluster_name].unique())
        frames = sorted(df['Normalized Frame'].dropna().unique())

        fig_w = max(8, len(clusters) * 2.5)
        fig_h = 6

        def render_gif(node_label: str, out_name: str):
            # Precompute static series per cluster (body traces) + node series
            pre = {}
            for cid in clusters:
                cdf = df[df[cluster_name] == cid].sort_values("Normalized Frame")
                g = cdf.groupby("Normalized Frame")

                pre[cid] = {
                    't1x': g["anchor x_body"].mean(),
                    't1y': g["anchor y_body"].mean(),
                    't2x': g["partner x_body"].mean(),
                    't2y': g["partner y_body"].mean(),
                    'ax':  g[f"anchor x_{node_label}"].mean(),
                    'ay':  g[f"anchor y_{node_label}"].mean(),
                    'px':  g[f"partner x_{node_label}"].mean(),
                    'py':  g[f"partner y_{node_label}"].mean(),
                }

            out_dir = os.path.join(self.directory, "mean_traces")
            os.makedirs(out_dir, exist_ok=True)
            tmpdir = os.path.join(out_dir, f"_tmp_{node_label}")
            os.makedirs(tmpdir, exist_ok=True)

            frame_paths = []
            for f in frames:
                fig, axes = plt.subplots(1, len(clusters), figsize=(fig_w, fig_h))
                if len(clusters) == 1:
                    axes = [axes]

                for ax, cid in zip(axes, clusters):
                    d = pre[cid]

                    # static dashed body markers
                    ax.plot(d['t1x'], d['t1y'], color=anchor_color, linestyle="None",
                            marker="|", markersize=10, alpha=0.6)
                    ax.plot(d['t2x'], d['t2y'], color=partner_color, linestyle="None",
                            marker="_", markersize=10, alpha=0.6)

                    # animated node dots at this frame
                    if f in d['ax'].index:
                        ax.scatter(d['ax'].loc[f], d['ay'].loc[f], s=marker_size,
                                alpha=0.9, color=anchor_color)
                    if f in d['px'].index:
                        ax.scatter(d['px'].loc[f], d['py'].loc[f], s=marker_size,
                                alpha=0.9, color=partner_color)

                    ax.set_xlim(*xlim)
                    ax.set_ylim(*ylim)
                    ax.set_aspect('equal', adjustable='box')
                    ax.set_title(f"Cluster {cid}", fontweight='bold', fontsize=11)
                    ax.axis('off')

                plt.tight_layout()
                frame_path = os.path.join(tmpdir, f"frame_{int(f):04d}.png")
                plt.savefig(frame_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                frame_paths.append(frame_path)

            # stitch into a gif
            images = [imageio.imread(p) for p in frame_paths]
            imageio.mimsave(os.path.join(out_dir, out_name), images, duration=0.08)  # ~12.5 fps

            # clean up temp frames
            for p in frame_paths:
                try:
                    os.remove(p)
                except OSError:
                    pass
            try:
                os.rmdir(tmpdir)
            except OSError:
                pass

        # make the three GIFs
        render_gif("head", "overview_head.gif")
        render_gif("body", "overview_body.gif")
        render_gif("tail", "overview_tail.gif")

    
    ## METHOD SPATIAL_CLUSTER: MAP OUT WHERE EACH INTERACTION OCCURED ON THE PETRI DISH 
    def spatial_cluster(self):

        df = self.df

        df = df[df['Normalized Frame'] == 0] 

        output = os.path.join(self.directory, "spatial")
        os.makedirs(output, exist_ok=True)

        cluster_ids = sorted(df[cluster_name].unique())
        num_clusters = len(cluster_ids)
        cols = 4
        rows = int(np.ceil(num_clusters / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), sharex=True, sharey=True)
        axes = axes.flatten()

        for idx, cluster_id in enumerate(cluster_ids):
            ax = axes[idx]
            cluster_df = df[df[cluster_name] == cluster_id].copy()

            cluster_df["mid_x"] = (cluster_df["Track_1 x_body"] + cluster_df["Track_2 x_body"]) / 2
            cluster_df["mid_y"] = (cluster_df["Track_1 y_body"] + cluster_df["Track_2 y_body"]) / 2

            ax.scatter(cluster_df['mid_x'], cluster_df['mid_y'], s=10, alpha=0.5, edgecolors='none')
            ax.set_title(f"Cluster {cluster_id}")
            ax.set_ylim(0, 1400)
            ax.set_xlim(0, 1400)
            ax.set_xticks([])
            ax.set_yticks([])
        
        for i in range(len(cluster_ids), len(axes)):
            axes[i].axis('off')

        plt.suptitle("Normalised Frame per Cluster", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join(output, "spatial_plot_per_cluster.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    
    ## METHOD CLUSTER_OVER_TIME: RAW COUNT OF INTERACTIONS OVER TIME PER CLUSTER 
    def cluster_over_time(self):

        df = self.df 
        cluster_name = self.cluster_name

        df = df[df['Normalized Frame'] == 0].copy() ## middle of the interaction

        bin_size = 300
        df["time_bin"] = (df["Frame"] // bin_size) * bin_size

        counts = (
        df.groupby(["file", "condition", cluster_name, "time_bin"])
        .size()
        .reset_index(name="count"))

        output = os.path.join(self.directory, 'clusters-timecourse')
        os.makedirs(output, exist_ok=True)

        ## RAW COUNT

        cluster_ids = sorted(df[cluster_name].unique())
        num_clusters = len(cluster_ids)
        cols = 4
        rows = int(np.ceil(num_clusters / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), sharex=True, sharey=True)
        axes = axes.flatten()

        for idx, cluster_id in enumerate(cluster_ids):
            ax = axes[idx]
            cluster_df = counts[counts[cluster_name] == cluster_id].copy()

            sns.lineplot(data=cluster_df, x='time_bin', y='count', hue='condition', ax=ax)
        
        for i in range(len(cluster_ids), len(axes)):
            axes[i].axis('off')

        plt.suptitle("Cluster Timecourse", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join(output, "clusters-rawcount-time.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


        # ---- PROPORTIONS: normalize FIRST per video, THEN bin plot ----

        # 1) one row per interaction (you already have df filtered to Normalized Frame == 0)
        inter_per_video = (
            df[['file', 'condition', 'interaction_id', cluster_name, 'Frame']]
            .drop_duplicates(subset=['file', 'interaction_id'])
        )

        # 2) total interactions per video (global denominator)
        video_totals = (
            inter_per_video.groupby('file')
            .size()
            .rename('total_interactions_video')
            .reset_index()
        )

        # 3) bin by 300s (using the interaction’s Frame)
        bin_size = 300
        inter_per_video['time_bin'] = (inter_per_video['Frame'] // bin_size) * bin_size

        # 4) counts per (video, condition, cluster, time_bin)
        counts = (
            inter_per_video
            .groupby(['file', 'condition', cluster_name, 'time_bin'])
            .size()
            .reset_index(name='count_in_bin')
        )

        # 5) attach the video total, then compute PROPORTION with global denom
        counts = counts.merge(video_totals, on='file', how='left')
        counts['proportion'] = counts['count_in_bin'] / counts['total_interactions_video']


        # 6) plot mean±SD across videos within each condition
        cluster_ids = sorted(df[cluster_name].unique())
        num_clusters = len(cluster_ids)
        cols = 4
        rows = int(np.ceil(num_clusters / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), sharex=True, sharey=True)
        axes = axes.flatten()

        for i, cluster in enumerate(cluster_ids):
            ax = axes[i]
            sub = counts[counts[cluster_name] == cluster]
            sns.lineplot(
                data=sub, x='time_bin', y='proportion', hue='condition', ax=ax,
                estimator='mean', errorbar='sd'
            )
            ax.set_title(f"Cluster {cluster}")
            ax.set_xlabel("Time bin (s)")
            ax.set_ylabel("Proportion of that video’s interactions")
            ax.set_ylim(0, 0.05)

        for j in range(len(cluster_ids), len(axes)):
            axes[j].axis('off')

        plt.suptitle("Cluster timecourse (per-video global-normalized proportions, mean±SD)", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(output, "clusters-proportion-time_GLOBAL-video-denom.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # --- Two-panel plot: one per condition, lines = clusters ---

        palette = dict(zip(
            cluster_ids,
            sns.color_palette("tab20", n_colors=len(cluster_ids))
        ))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        for ax, cond in zip(axes, ["iso", "group"]):
            sub = counts[counts["condition"] == cond]

            # mean±SD across videos for each (cluster, time_bin)
            sns.lineplot(
                data=sub,
                x="time_bin", y="proportion",
                hue=cluster_name,
                estimator="mean", errorbar="sd",
                palette=palette,
                ax=ax
            )

            ax.set_title(f"{cond}")
            ax.set_xlabel("Time bin (s)")
            ax.set_ylabel("Proportion of that video's interactions" if cond == "iso" else "")
            ax.set_ylim(0, None)             # set a fixed cap if you want, e.g. ax.set_ylim(0, 0.06)
            ax.grid(alpha=0.2)

        # # single combined legend (clusters)
        # handles, labels = axes[1].get_legend_handles_labels()
        # fig.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)), frameon=False)

        handles, labels = axes[1].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center",
                    ncol=min(6, len(labels)), frameon=False)
        for ax in axes:
            ax.legend_.remove() if ax.get_legend() else None


        fig.suptitle("Cluster proportions over time (per-video global-normalized; mean±SD across videos)", y=1.03)
        fig.tight_layout()
        plt.savefig(os.path.join(output, "clusters-proportion-time_by_condition.png"), dpi=300, bbox_inches="tight")
        plt.close()

        count2 = (
        df.groupby(["file", "condition", cluster_name, "time_bin"])
        .size()
        .reset_index(name="count"))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        for ax, cond in zip(axes, ["iso", "group"]):
            sub = count2[count2["condition"] == cond]

            # mean±SD across videos for each (cluster, time_bin)
            sns.lineplot(
                data=count2,
                x="time_bin", y="count",
                hue=cluster_name,
                estimator="mean", errorbar="sd",
                palette=palette,
                ax=ax
            )

            ax.set_title(f"{cond}")
            ax.set_xlabel("Time bin (s)")
            ax.set_ylabel("Number of interactions" if cond == "iso" else "")
            ax.set_ylim(0, None)             # set a fixed cap if you want, e.g. ax.set_ylim(0, 0.06)
            ax.grid(alpha=0.2)

        # # single combined legend (clusters)
        # handles, labels = axes[1].get_legend_handles_labels()
        # fig.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)), frameon=False)

        handles, labels = axes[1].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center",
                    ncol=min(6, len(labels)), frameon=False)
        for ax in axes:
            ax.legend_.remove() if ax.get_legend() else None


        fig.suptitle("Cluster raw count over time ", y=1.03)
        fig.tight_layout()
        plt.savefig(os.path.join(output, "clusters-rawcount-time_by_condition.png"), dpi=300, bbox_inches="tight")
        plt.close()
    

    #### METHOD PARTNER_MORPHOLOGY:
    def partner_morphology(self):

        df = self.df.copy()
        cluster_name = self.cluster_name
        eps = 1e-9

        # Relative geometry
        df['relative_partner_x'] = df['partner x_body'] - df['anchor x_body']
        df['relative_partner_y'] = df['partner y_body'] - df['anchor y_body']
        df['distance_from_anchor'] = np.sqrt(df['relative_partner_x']**2 + df['relative_partner_y']**2)

        # Order in time within each interaction
        df = df.sort_values(['interaction_id', 'Normalized Frame'])

        # Previous-frame values
        df['prev_relative_partner_x'] = df.groupby('interaction_id')['relative_partner_x'].shift(1)
        df['prev_relative_partner_y'] = df.groupby('interaction_id')['relative_partner_y'].shift(1)
        df['prev_distance_from_anchor'] = df.groupby('interaction_id')['distance_from_anchor'].shift(1)

        # Step vector & length
        df['relative_step_x'] = df['relative_partner_x'] - df['prev_relative_partner_x']
        df['relative_step_y'] = df['relative_partner_y'] - df['prev_relative_partner_y']
        df['relative_step_length'] = np.sqrt(df['relative_step_x']**2 + df['relative_step_y']**2)

        # Unit direction (anchor -> partner) at previous frame
        df['prev_direction_x'] = df['prev_relative_partner_x'] / (df['prev_distance_from_anchor'] + eps)
        df['prev_direction_y'] = df['prev_relative_partner_y'] / (df['prev_distance_from_anchor'] + eps)

        # Signed forward progress: positive = toward, negative = away
        df['forward_progress'] = -(
            df['relative_step_x'] * df['prev_direction_x'] +
            df['relative_step_y'] * df['prev_direction_y']
        )

            # Sideways wiggle: how much movement occurred off-axis (lateral deviation)
        df['sideways_wiggle'] = np.sqrt(
            np.maximum(0.0, df['relative_step_length']**2 - df['forward_progress']**2)
        )

        df['prev_forward_progress'] = df.groupby('interaction_id')['forward_progress'].shift(1)

        # Radial acceleration (per frame units): change in forward_progress frame-to-frame
        df['forward_accel'] = df['forward_progress'] - df['prev_forward_progress']
 

                # Select per-frame columns to plot
        out_cols = [
            'interaction_id',
            'Normalized Frame',
            'forward_progress',          # <-- your per-frame score
            'relative_step_length',      # optional QC
            'distance_from_anchor',      # optional context
            'sideways_wiggle',       # lateral deviation magnitude
            'forward_accel',             # rate of change of that speed (acceleration)

            cluster_name,
            'condition'
        ]
        per_frame = df[out_cols].copy()

        outdir = os.path.join(self.directory, "morphology_score")
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, "partner_morphology_trace.csv")
        per_frame.to_csv(outpath, index=False)

        # sns.lineplot(data=per_frame, x='Normalized Frame', y='forward_progress', hue=cluster_name)

        # plt.show()
        # plt.close()

        # sns.lineplot(data=per_frame, x='Normalized Frame', y='sideways_wiggle', hue=cluster_name)
        # plt.show()
        # plt.close()


        # sns.lineplot(data=per_frame, x='Normalized Frame', y='forward_accel', hue=cluster_name)
        # plt.show()
        # plt.close()

        # --- AUCs (per interaction) ---
        # Pre-AUC: total approach effort before 0
        # Post-AUC: total retreat effort after 0 (likely negative)
        # Net AUC: overall bias (pre + post)

        grp = df.groupby('interaction_id', sort=False)

        pre_auc = grp.apply(
            lambda g: g.loc[g['Normalized Frame'] < 0, 'forward_progress'].sum()
        ).rename('auc_pre')

        post_auc = grp.apply(
            lambda g: g.loc[g['Normalized Frame'] > 0, 'forward_progress'].sum()
        ).rename('auc_post')

        auc = (
            pre_auc.reset_index()
            .merge(post_auc.reset_index(), on='interaction_id', how='outer')
        )
        auc['auc_net'] = auc['auc_pre'].fillna(0) + auc['auc_post'].fillna(0)
        auc['auc_net_magntiude'] = auc['auc_pre'].fillna(0) + (-auc['auc_post'].fillna(0))

        # Attach labels (one row per interaction)
        lookup = (
            df[['interaction_id', self.cluster_name, 'condition']]
            .drop_duplicates('interaction_id')
        )
        auc = auc.merge(lookup, on='interaction_id', how='left')

        # Save alongside your other outputs
        outdir = os.path.join(self.directory, "morphology_score")
        os.makedirs(outdir, exist_ok=True)
        auc_path = os.path.join(outdir, "partner_morphology_auc.csv")
        auc.to_csv(auc_path, index=False)

        print(auc.head())

        order = (auc.groupby(cluster_name)['auc_pre']
           .mean()
           .sort_values(ascending=False)
           .index)

        sns.barplot(data=auc, x=cluster_name, y='auc_pre', order=order, estimator=np.mean, errorbar='se')
        sns.stripplot(data=auc, x=cluster_name, y='auc_pre', order=order, color='k', alpha=0.4, jitter=0.2, size=3)

        plt.show()
        plt.close()

        sns.barplot(data=auc, x=cluster_name, y='auc_post', estimator=np.mean, errorbar='se')
        plt.show()
        plt.close()

        sns.barplot(data=auc, x=cluster_name, y='auc_net', estimator=np.mean, errorbar='se')
        plt.show()
        plt.close()

        sns.barplot(data=auc, x=cluster_name, y='auc_net_magntiude', estimator=np.mean, errorbar='se')
        plt.show()
        plt.close()








#### Forward progress measures how much the partner’s position changes along the line that connects it to the anchor — 
# in other words, how much closer or farther the partner moves relative to the anchor in a single frame

















            




############ CREATE CROPPED INTERACTION DATAFRAME ############

# group_csv = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser/test2/interactions_group.csv'
# iso_csv = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser/test2/interactions_iso.csv'
# wkt_dir = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser/videos_original'
# output_dir = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser/test3/cropped_interactions.csv'

# create_cropped_interactions(group_csv,iso_csv, wkt_dir, output_dir)


############ REASSIGN CLUSTER IDS ############ only run once !!!

# file = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser/test7_F18/pca-data2-F18-mcmodels4-Kmax3.csv'
# output = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser/test7_F18/pca-data2-F18-mcmodels4-Kmax3.csv'
# modify_cluster_ids(file, output)


############ ANALYSIS PIPELINE ############

if __name__ == "__main__":
    # Set your paths
    directory = "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser/test3_F18/test-new-pipeline/Yhat.idt.pca"
    interactions = "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser/test4_F29/cropped_interactions.csv"
    clusters = "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser/test3_F18/pca-data2-F18.csv"
    cluster_name = "Yhat.idt.pca"   # or whatever your cluster column is
    video_path = "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser/videos_original"

    # Create instance
    pipeline = ClusterPipeline(directory, interactions, clusters, cluster_name, video_path)

    # Run methods
    pipeline.loading_data()
    pipeline.anchor_partner()
    #pipeline.sanity_check_anchor_partner() ## sanity check reversing anchor and partner 
    
    # pipeline.grid_videos()
    # pipeline.raw_trajectories()
    # pipeline.mean_trajectories()
    # pipeline.barplots() 
    # pipeline.speed()
    # pipeline.acceleration()
    # pipeline.heading_angle()
    # pipeline.approach_angle()
    # pipeline.body_distance()
    # pipeline.min_distance()
    # pipeline.distance()
    # pipeline.contact()
    # pipeline.summary_quantifications()
    # pipeline.summary_anchor_partner()
    # pipeline.mean_traces()
    # pipeline.mean_traces_gifs()
    # pipeline.spatial_cluster()
    # pipeline.cluster_over_time()
    pipeline.partner_morphology()














    