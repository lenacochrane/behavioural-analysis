import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from itertools import combinations
import cv2


# df = pd.read_feather('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-proximal-interactions/test-interactions-3/2024-12-10_10-55-21_td9.tracks.feather')

# pixel_columns = ['x_tail', 'y_tail', 'x_body', 'y_body', 'x_head', 'y_head']
# df[pixel_columns] = df[pixel_columns] * (90/1032)

# proximity_threshold = 10 # 10mm

# df['track_id'] = df['track_id'].astype(int)
# df['frame'] = df['frame'].astype(int)

# track_ids = df['track_id'].unique()
# track_combinations = list(combinations(track_ids, 2))

# for track_a, track_b in track_combinations:
#     results = []
#     track_a_data = df[df['track_id'] == track_a]
#     track_b_data = df[df['track_id'] == track_b]

#     common_frames = set(track_a_data['frame']).intersection(track_b_data['frame'])


#     for frame in common_frames:

#         point_a = track_a_data[track_a_data['frame'] == frame][['x_body', 'y_body']].to_numpy(dtype=float)
#         point_b = track_b_data[track_b_data['frame'] == frame][['x_body', 'y_body']].to_numpy(dtype=float)

#         dist = np.linalg.norm(point_a - point_b)
#         if dist < proximity_threshold:
#             results.append({
#                         'Frame': frame,
#                         'Track A': track_a,
#                         'Track A X Body': point_a[0, 0], #idk why this shape but fix ltr 
#                         'Track A Y Body': point_a[0, 1],
#                         'Track B': track_b,
#                         'Track B X Body': point_b[0, 0],
#                         'Track B Y Body': point_b[0, 1],
#                         'Proximal': True,
#                         'Distance (mm)': dist
#                     })

#     # Create and save DataFrame
#     results_df = pd.DataFrame(results)
#     if not results_df.empty:
#         # Save to CSV
#         filepath = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-proximal-interactions/test-interactions-3'
#         filename = f'track_{track_a}_vs_track_{track_b}_proximity_results.csv'
#         full_path = os.path.join(filepath, filename)

#         results_df.to_csv(full_path, index=False)

# # all_frames = pd.Series(index=range(0, 3601))





### VIDEO_FILES


# df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-proximal-interactions/test-interactions-3/track_4_vs_track_7_proximity_results.csv')

# pixel_columns = ['Track A X Body',  'Track A Y Body',  'Track B X Body',  'Track B Y Body']
# df[pixel_columns] = df[pixel_columns] * (1032/90)

# video_path = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-proximal-interactions/test-interactions-3/2024-12-10_10-55-21_td9.mp4'
# output_dir = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-proximal-interactions/test-interactions-3'
# os.makedirs(output_dir, exist_ok=True)


# # Open the video file
# cap = cv2.VideoCapture(video_path)
# fps = cap.get(cv2.CAP_PROP_FPS)
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# first_frame = 1965
# last_frame = 1993

# # Setup VideoWriter for the current segment
# video_filename = f"frames_{first_frame}-{last_frame}.mp4"


# out = cv2.VideoWriter(os.path.join(output_dir, video_filename), 
#                     cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 
#                     fps, (frame_width, frame_height))


# # Create a dictionary for quick frame data access
# frame_data = df.set_index('Frame').T.to_dict()

# # Process each frame in the range
# for frame_number in range(first_frame, last_frame + 1):
#     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Check if the frame is in the DataFrame and extract coordinates
#     if frame_number in frame_data:
#         row = frame_data[frame_number]

#         track_a_x = int(row['Track A X Body'])
#         track_a_y = int(row['Track A Y Body'])
#         track_b_x = int(row['Track B X Body'])
#         track_b_y = int(row['Track B Y Body'])
#         track_a_id = row['Track A']
#         track_b_id = row['Track B']

#         # Draw and label the points
#         cv2.circle(frame, (track_a_x, track_a_y), 5, (0, 255, 0), -1)
#         cv2.circle(frame, (track_b_x, track_b_y), 5, (255, 0, 0), -1)

#         cv2.putText(frame, f'Track {track_a_id}', (track_a_x - 50, track_a_y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#         cv2.putText(frame, f'Track {track_b_id}', (track_b_x + 50, track_b_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#     # Always write the frame to the output video
#     out.write(frame)

# # Close video writing and video capture
# out.release()
# cap.release()
# cv2.destroyAllWindows()






