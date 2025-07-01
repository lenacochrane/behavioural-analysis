
# %% ################################################################################################################################################################

import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyarrow.feather as feather
import cv2
import re


######## ==== CREATE CROPPED INTERACTION CSV IDENTICAL TO THAT IN THE UMAP PIPELINE ==== ########

# == 1. Load GH and SI interaction CSV

df_group = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed/interactions.csv')
df_group['condition'] = 'group'

df_iso = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/socially-isolated/interactions.csv')
df_iso['condition'] = 'iso'

df = pd.concat([df_iso, df_group], ignore_index=True)

# == 2. Create unique interaction ID 

df["interaction_id"] = df["condition"] + "_" + df["Interaction Number"].astype(str)

# == 3. Crop interactions +- 15 frames around the normalised frame 

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

df_cropped = df.groupby("interaction_id", group_keys=False).apply(crop_interaction)


# === Un-normalize coordinates by re-adding the midpoint used during normalization ===

print(df_cropped)


coordinate_columns = [
    "Track_1 x_body", "Track_1 y_body", "Track_2 x_body", "Track_2 y_body",
    "Track_1 x_tail", "Track_1 y_tail", "Track_2 x_tail", "Track_2 y_tail",
    "Track_1 x_head", "Track_1 y_head", "Track_2 x_head", "Track_2 y_head"
]

for col in coordinate_columns:
    if "x_" in col:
        df_cropped[col] += df_cropped["Normalization mid_x"]
    elif "y_" in col:
        df_cropped[col] += df_cropped["Normalization mid_y"]

print(df_cropped)

# === convert mm into pixels ===

scale_factor = (1032/90)

for col in coordinate_columns:
   df_cropped[col] *= scale_factor


df_cropped.to_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/umap-pipeline/cropped_interactions.csv', index=False)



################################################################################################################################################################
# %% YOUNGSERS UMAP 

import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyarrow.feather as feather
import cv2
from random import sample


######## ==== MODIFY YOUNGSER INTERACTION ID'S TO MATCH MINE!! ==== ########

df_umap = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser_umap.csv')

## match youngsers id naming to my global id 

df_umap["condition"] = df_umap["behavior"].map({"grouped": "group", "isolated": "iso"})

df_umap['interaction_id'] = df_umap["condition"] + "_" + df_umap["interaction_id"].astype(str)


df_umap.to_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser_umap.csv', index=False)



################################################################################################################################################################
# %%
import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyarrow.feather as feather
import cv2
from random import sample

######## ==== CROPPED INTERACTION CSV ==== ########

df_cropped_interaction = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/umap-pipeline/cropped_interactions.csv')

######## ==== CLUSTERED CSV ==== ########

df_umap = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser_umap.csv')


######## ==== MERGE CSV ==== ########

df = pd.merge(
    df_cropped_interaction, 
    df_umap[['interaction_id', 'Yhat']], 
    on='interaction_id', 
    how='inner'
)


df['file'] = df['file'].str.replace('.tracks.feather', '.mp4', regex=False)

######## ==== ORIGINAL VIDEOS ==== ########

video_path = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/umap-pipeline/videos_original'


######## ==== OUTPUT PATH ==== ########
output_dir = "/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/umap-pipeline/cluster_grid_videos"
os.makedirs(output_dir, exist_ok=True)



frames_per_clip = 30
dot_radius = 5
dot_thickness = 10
fps = 10


# === MAIN LOOP ===
for cluster_id in sorted(df['Yhat'].unique()):
    cluster_df = df[df['Yhat'] == cluster_id]
    unique_ids = cluster_df['interaction_id'].unique()

    if len(unique_ids) < 9:
        print(f"⚠️ Skipping cluster {cluster_id} (only {len(unique_ids)} interactions)")
        continue

    chosen_ids = sample(list(unique_ids), 9)
    interaction_clips = []

    for inter_id in chosen_ids:
        inter_df = cluster_df[cluster_df['interaction_id'] == inter_id].sort_values("Frame")
        start_frame = inter_df["Frame"].iloc[0]
        end_frame = start_frame + frames_per_clip
        clip_df = inter_df[(inter_df["Frame"] >= start_frame) & (inter_df["Frame"] < end_frame)]

        if len(clip_df) < frames_per_clip:
            continue

        video_file = inter_df['file'].iloc[0]
        full_video_path = os.path.join(video_path, video_file)

        if not os.path.exists(full_video_path):
            print(f"⚠️ Missing video: {full_video_path}")
            continue

        cap = cv2.VideoCapture(full_video_path)
        clip_frames = []

        for _, row in clip_df.iterrows():
            frame_idx = int(row['Frame'])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Get coordinates
            x1 = int(row['Track_1 x_body'])
            y1 = int(row['Track_1 y_body'])
            x2 = int(row['Track_2 x_body'])
            y2 = int(row['Track_2 y_body'])

            # Overlay dots on full frame
            cv2.circle(frame, (x1, y1), dot_radius, (0, 0, 255), dot_thickness)  # red
            cv2.circle(frame, (x2, y2), dot_radius, (255, 0, 0), dot_thickness)  # blue

            clip_frames.append(frame)

        cap.release()

        if len(clip_frames) == frames_per_clip:
            interaction_clips.append(clip_frames)

    if len(interaction_clips) < 9:
        print(f"⚠️ Not enough good clips for cluster {cluster_id}")
        continue

    # === Create grid video ===
    h, w = interaction_clips[0][0].shape[:2]
    grid_frames = []

    for i in range(frames_per_clip):
        row1 = np.hstack([interaction_clips[j][i] for j in range(0, 3)])
        row2 = np.hstack([interaction_clips[j][i] for j in range(3, 6)])
        row3 = np.hstack([interaction_clips[j][i] for j in range(6, 9)])
        grid_frame = np.vstack([row1, row2, row3])
        grid_frames.append(grid_frame)

    # === Save grid video ===
    output_path = os.path.join(output_dir, f"cluster_{cluster_id}.mp4")
    frame_height, frame_width = grid_frames[0].shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    for frame in grid_frames:
        out.write(frame)
    out.release()

    print(f"✅ Saved grid video for cluster {cluster_id} → {output_path}")

# %%
