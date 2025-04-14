import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pyarrow.feather as feather
import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from shapely import wkt

# directory = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/socially-isolated'


# for file in os.listdir(directory):
#     if file.endswith('tracks.feather'):
#         path = os.path.join(directory, file)
#         df = pd.read_feather(path)
#         print(f"File: {file}")

#         # Unique tracks
#         track_ids = df['track_id'].unique()
#         print(f"Number of unique tracks: {len(track_ids)}")

#         # Check for gaps
#         track_gaps = {}
#         for track_id in track_ids:
#             track_df = df[df['track_id'] == track_id].sort_values(by='frame')
#             frames = track_df['frame'].astype(int).tolist()

#             missing_frames = [f for f in range(min(frames), max(frames) + 1) if f not in frames]
#             if missing_frames:
#                 track_gaps[track_id] = missing_frames

#         if track_gaps:
#             print("Tracks with missing frames:")
#             for track, gaps in track_gaps.items():
#                 print(f"  Track {track} has missing frames: {gaps}")
#         else:
#             print("No missing frames detected in any track.")

#         # Check for jumps > 30 pixels
#         big_jumps = []
#         for track_id in track_ids:
#             track = df[df['track_id'] == track_id].sort_values('frame')
#             coords = track[['x_body', 'y_body']].values
#             frames = track['frame'].values
#             if len(coords) < 2:
#                 continue
#             dists = np.linalg.norm(np.diff(coords, axis=0), axis=1)
#             for i, dist in enumerate(dists):
#                 if dist > 30:
#                     big_jumps.append({
#                         'track_id': track_id,
#                         'frame': frames[i+1],  # +1 because diff reduces index
#                         'jump': dist
#                     })

#         if big_jumps:
#             print("Jumps > 30 pixels:")
#             for jump in big_jumps:
#                 print(f"  Track {jump['track_id']} - Frame {jump['frame']} - Jump = {jump['jump']:.2f} pixels")
#         else:
#             print("No jumps > 30 pixels detected.")

#         print("-" * 60)


df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/socially-isolated/number_digging.csv')
df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/group-housed/number_digging.csv')
sns.lineplot(data=df, x='frame', y='number_digging', label='si')
sns.lineplot(data=df1, x='frame', y='number_digging', label='gh')
plt.show()