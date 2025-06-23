import h5py
import numpy as np
import os
import pandas as pd

# === Set your folder path ===
folder_path = '/Users/cochral/Desktop/MOSEQ/predictions-edited'

# === Loop through all .h5 files ===
for filename in os.listdir(folder_path):
    if filename.endswith('.h5'):
        file_path = os.path.join(folder_path, filename)
        print(f"Processing: {filename}")

        with h5py.File(file_path, 'r+') as f:
            if 'point_scores' not in f or 'instance_scores' not in f:
                print(f"  ❌ Skipped (missing required datasets)")
                continue

            point_scores = np.array(f['point_scores'])         # (tracks, nodes, frames)
            instance_scores = np.array(f['instance_scores'])   # (tracks, frames)

            num_tracks, num_nodes, num_frames = point_scores.shape
            updated_instance_scores = instance_scores.copy()

            for track in range(num_tracks):
                for frame in range(num_frames):
                    point_vec = point_scores[track, :, frame]  # shape: (3,)
                    if np.all(np.isnan(point_vec) | (point_vec == 0)):
                        updated_instance_scores[track, frame] = 1.0

            # Overwrite instance_scores in place
            del f['instance_scores']
            f.create_dataset('instance_scores', data=updated_instance_scores)
            print(f"  ✅ instance_scores updated in {filename}")

################################################################## check h5 file conversion worked 

# === Set your input file path ===
# file_path = '/Users/cochral/Desktop/MOSEQ/predictions-edited/N10-GH_2025-03-31_17-02-11_td6.h5'

# # === Read and export instance_scores ===
# with h5py.File(file_path, 'r') as f:
#     instance_scores = np.array(f['instance_scores'])  # shape: (tracks, frames)
#     num_tracks, num_frames = instance_scores.shape

#     records = [
#         {'frame': frame, 'track': track, 'instance_score': instance_scores[track, frame]}
#         for track in range(num_tracks)
#         for frame in range(num_frames)
#     ]

# df = pd.DataFrame(records)
# df.to_csv('/Users/cochral/Desktop/MOSEQ/predictions-edited/frame_track_instance_scores.csv', index=False)
# print("✅ Saved to 'frame_track_instance_scores.csv'")