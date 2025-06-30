
# %%

import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyarrow.feather as feather
import cv2


######## ==== CREATE CROPPED INTERACTION CSV IDENTICAL TO THAT IN THE UMAP PIPELINE ==== ########

# == 1. Load GH and SI interaction CSV

df_group = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed/interactions.csv')
df_group['condition'] = 'group'

df_iso = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/socially-isolated/interactions.csv')
df_iso['condition'] = 'iso'

df = pd.concat([df_iso, df_group], ignore_index=True)

# 2. Create unique interaction ID 

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

df_cropped.to_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/umap-pipeline/cropped_interactions.csv', index=False)



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



crop_size = 600
frames_per_clip = 30

######## ==== MAIN LOOP ==== ########
for cluster_id in sorted(df['Yhat'].unique()):

    cluster_df = df[df['Yhat'] == cluster_id]
    unique_ids = cluster_df['interaction_id'].unique()

    if len(unique_ids) < 9:
        print(f"Skipping cluster {cluster_id} (only {len(unique_ids)} interactions)")
        continue

    chosen_ids = sample(list(unique_ids), 9)
    interaction_clips = []

    for inter_id in chosen_ids:
        inter_df = cluster_df[cluster_df['interaction_id'] == inter_id]
        inter_df = inter_df.sort_values("Frame")
        start_frame = inter_df["Frame"].iloc[0]
        end_frame = start_frame + frames_per_clip
        clip_df = inter_df[(inter_df["Frame"] >= start_frame) & (inter_df["Frame"] < end_frame)]

        if len(clip_df) < frames_per_clip:
            continue

        video_file = inter_df['file'].iloc[0]
        video_full_path = os.path.join(video_path, video_file)

        if not os.path.exists(video_full_path):
            print(f"⚠️ Missing video: {video_full_path}")
            continue

        cap = cv2.VideoCapture(video_full_path)
        clip_frames = []

        for _, row in clip_df.iterrows():
            frame_num = int(row["Frame"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue

            # === Overlay dots ===
            x1 = row["Track_1 x_body"]
            y1 = row["Track_1 y_body"]
            x2 = row["Track_2 x_body"]
            y2 = row["Track_2 y_body"]

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            cv2.circle(frame, (int(x1), int(y1)), 5, (0, 0, 255), -1)  # Red = larva 1
            cv2.circle(frame, (int(x2), int(y2)), 5, (255, 0, 0), -1)  # Blue = larva 2

            # === Crop around center ===
            h, w = frame.shape[:2]
            x1_crop = int(max(cx - crop_size // 2, 0))
            y1_crop = int(max(cy - crop_size // 2, 0))
            x2_crop = int(min(cx + crop_size // 2, w))
            y2_crop = int(min(cy + crop_size // 2, h))

            frame_cropped = frame[y1_crop:y2_crop, x1_crop:x2_crop]
            frame_cropped = cv2.resize(frame_cropped, (crop_size, crop_size))
            clip_frames.append(frame_cropped)


        cap.release()

        if len(clip_frames) == frames_per_clip:
            interaction_clips.append(clip_frames)

    if len(interaction_clips) < 9:
        print(f"⚠️ Not enough good clips for cluster {cluster_id}")
        continue

    # === Build 3x3 grid ===
    h, w = interaction_clips[0][0].shape[:2]
    grid_frames = []

    for i in range(frames_per_clip):
        row1 = np.hstack([interaction_clips[j][i] for j in range(0, 3)])
        row2 = np.hstack([interaction_clips[j][i] for j in range(3, 6)])
        row3 = np.hstack([interaction_clips[j][i] for j in range(6, 9)])
        grid_frame = np.vstack([row1, row2, row3])
        grid_frames.append(grid_frame)

    # === Save the video ===
    save_path = os.path.join(output_dir, f"cluster_{cluster_id}.mp4")
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w * 3, h * 3))

    for frame in grid_frames:
        out.write(frame)
    out.release()

    print(f"✅ Saved cluster {cluster_id} → {save_path}")





# %%
