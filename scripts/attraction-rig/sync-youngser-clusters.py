
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyarrow.feather as feather
from scipy.spatial.distance import pdist
from shapely import wkt
import glob


#### FUNCTION CREATE_CROPPED_INTERACTIONS: ORIGINAL GROUP AND ISO INTERACTIONS MERGED AND INTERACTIONS CROPPED 30 FRAMES
def create_cropped_interactions(group_csv, iso_csv, groupniso_csv, wkt_dir, output_dir):

    df_group = pd.read_csv(group_csv)
    df_group['condition'] = 'G'

    df_iso = pd.read_csv(iso_csv)
    df_iso['condition'] = 'S'

    df_groupniso = pd.read_csv(groupniso_csv)
    df_groupniso['condition'] = 'GS'

    df = pd.concat([df_iso, df_group, df_groupniso], ignore_index=True)

    df["interaction_id"] = df["condition"] + "_" + df["Interaction Number"].astype(str)  # Create unique interaction ID 

    df = df.sort_values(["interaction_id", "Normalized Frame"])

    def crop_interaction(group):
        if group.empty or "Normalized Frame" not in group.columns:
            return None
        center_idx = (group["Normalized Frame"].abs()).idxmin()
        if pd.isna(center_idx):
            return None
        center_pos = group.index.get_loc(center_idx)
        if center_pos < 10 or (center_pos + 11) >= len(group):
            return None
        cropped = group.iloc[center_pos - 10 : center_pos + 11].copy()
        cropped["interaction_id"] = group["interaction_id"].iloc[0]
        expected_frames = list(range(-10, 11))
        actual_frames = list(cropped["Normalized Frame"])
        if sorted(actual_frames) != expected_frames:
            return None
        return cropped

    df_cropped = df.groupby("interaction_id", group_keys=False).apply(crop_interaction) # crop interactions 10 frames either side

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
            # print(f"🟡 Using fixed scale {scale:.3f} for {video_file}")
        
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


    df["youngser_no"] = df.groupby("condition").cumcount() + 1

    # Build your new interaction_id
    df['interaction_id'] = (
        df['condition'].astype(str) + '_' +
        df['youngser_no'].astype(str))
    df.to_csv(output, index=False)


########### CREATE CROPPED INTERACTION DATAFRAME ############

group_csv = '/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/clustered-interactions_pt2/interactions_G.csv'
iso_csv = '/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/clustered-interactions_pt2/interactions_S.csv'
groupniso_csv = '/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/clustered-interactions_pt2/interactions_GS.csv'
wkt_dir = '/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/clustered-interactions_pt2/videos_original'
output_dir = '/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/clustered-interactions_pt2/cropped_interactions.csv'

create_cropped_interactions(group_csv,iso_csv, groupniso_csv, wkt_dir, output_dir)



########### REASSIGN CLUSTER IDS ############ only run once !!!

file = '/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/clustered-interactions_pt2/pca-data3-F26-mcmodels4-Kmax2-05-2026.csv'
output = '/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/clustered-interactions_pt2/pca-data3-F26-mcmodels4-Kmax2-05-2026.csv'
modify_cluster_ids(file, output)