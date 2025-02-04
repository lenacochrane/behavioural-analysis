import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from itertools import combinations
import cv2
import ast


### TRY AND GET THE CSV TO LOOK LIKE THE IDEAL DATAFRAME- UNMERGE DF 

# df = pd.read_feather('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-proximal-interactions/test-interaction-df/2024-07-12_13-18-27_td1.tracks.feather')

# pixel_columns = ['x_tail', 'y_tail', 'x_body', 'y_body', 'x_head', 'y_head']
# df[pixel_columns] = df[pixel_columns] * (90/1032)

# proximity_threshold = 20 # 10mm

# df['track_id'] = df['track_id'].astype(int)
# df['frame'] = df['frame'].astype(int)

# track_ids = df['track_id'].unique()
# track_combinations = list(combinations(track_ids, 2))

# results = []
# for track_a, track_b in track_combinations:

#     track_a_data = df[df['track_id'] == track_a]
#     track_b_data = df[df['track_id'] == track_b]

#     common_frames = set(track_a_data['frame']).intersection(track_b_data['frame'])

#     for frame in common_frames:


#         # body
#         point_a = track_a_data[track_a_data['frame'] == frame][['x_body', 'y_body']].to_numpy(dtype=float)
#         point_b = track_b_data[track_b_data['frame'] == frame][['x_body', 'y_body']].to_numpy(dtype=float)

#         ## tail
#         a_tail = track_a_data[track_a_data['frame'] == frame][['x_tail', 'y_tail']].to_numpy(dtype=float)
#         b_tail = track_b_data[track_b_data['frame'] == frame][['x_tail', 'y_tail']].to_numpy(dtype=float)

#         ## head
#         a_head = track_a_data[track_a_data['frame'] == frame][['x_head', 'y_head']].to_numpy(dtype=float)
#         b_head = track_b_data[track_b_data['frame'] == frame][['x_head', 'y_head']].to_numpy(dtype=float)
 

#         dist = np.linalg.norm(point_a - point_b)
#         if dist < proximity_threshold:
#             results.append({
#                         'Frame': frame,
#                         f'Interaction {track_a},{track_b}': True,
#                         f' Distance {track_a},{track_b}': dist,
#                         ## track a coordinates
#                         f'track_{track_a} x_tail': a_tail[0,0], 
#                         f'track_{track_a} y_tail': a_tail[0,1],
#                         f'track_{track_a} x_body': point_a[0,0], 
#                         f'track_{track_a} y_body': point_a[0,1],
#                         f'track_{track_a} x_head': a_head[0,0], 
#                         f'track_{track_a} y_head': a_head[0,1],
#                         ## track b coordinates
#                         f'track_{track_b} x_tail': b_tail[0,0], 
#                         f'track_{track_b} y_tail': b_tail[0,1],
#                         f'track_{track_b} x_body': point_b[0,0],
#                         f'track_{track_b} y_body': point_b[0,0],
#                         f'track_{track_b} x_head': b_head[0,0], 
#                         f'track_{track_b} y_head': b_head[0,1]
                        
#                     })


# # Create and save DataFrame
# results_df = pd.DataFrame(results)
# if not results_df.empty:
#         # Save to CSV

#     results_df.set_index('Frame', inplace=True, drop=False)


#     filepath = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-proximal-interactions/test-interaction-df'
#     filename =  'proximity_results.csv'
#     full_path = os.path.join(filepath, filename)

#     results_df.to_csv(full_path, index=False)







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







