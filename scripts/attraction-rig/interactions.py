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
import re


## TRY AND GET THE CSV TO LOOK LIKE THE IDEAL DATAFRAME- UNMERGE DF 

df = pd.read_feather('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-proximal-interactions/interaction-test-2/cleaned_tracks.feather')

pixel_columns = ['x_tail', 'y_tail', 'x_body', 'y_body', 'x_head', 'y_head']
df[pixel_columns] = df[pixel_columns] * (90/1032)

proximity_threshold = 20 # 10mm

df['track_id'] = df['track_id'].astype(int)
df['frame'] = df['frame'].astype(int)

track_ids = df['track_id'].unique()
track_combinations = list(combinations(track_ids, 2))

results = []
interaction_id = 0  # Start interaction ID
prev_frame = None  # Keep track of previous frame

for track_a, track_b in track_combinations:

    

    track_a_data = df[df['track_id'] == track_a]
    track_b_data = df[df['track_id'] == track_b]

    common_frames = set(track_a_data['frame']).intersection(track_b_data['frame'])

    for frame in common_frames:


        # body
        point_a = track_a_data[track_a_data['frame'] == frame][['x_body', 'y_body']].to_numpy(dtype=float)
        point_b = track_b_data[track_b_data['frame'] == frame][['x_body', 'y_body']].to_numpy(dtype=float)

        ## tail
        a_tail = track_a_data[track_a_data['frame'] == frame][['x_tail', 'y_tail']].to_numpy(dtype=float)
        b_tail = track_b_data[track_b_data['frame'] == frame][['x_tail', 'y_tail']].to_numpy(dtype=float)

        ## head
        a_head = track_a_data[track_a_data['frame'] == frame][['x_head', 'y_head']].to_numpy(dtype=float)
        b_head = track_b_data[track_b_data['frame'] == frame][['x_head', 'y_head']].to_numpy(dtype=float)
 

        dist = np.linalg.norm(point_a - point_b)
        if dist < proximity_threshold:

            if prev_frame is None or frame != prev_frame + 1:
                # If it's the first interaction OR there's a gap, start a new interaction
                interaction_id += 1

            results.append({
                'Frame': frame,
                'Interaction Number': interaction_id,  # Assign interaction number
                'Interaction Pair': [track_a, track_b],
                'Distance': dist,
                ## Track 1 coord
                'Track_1 x_tail': a_tail[0, 0],
                'Track_1 x_tail': a_tail[0, 1],
                'Track_1 x_body': point_a[0, 0],
                'Track_1 y_body': point_a[0, 1],
                'Track_1 x_head': a_head[0, 0],
                'Track_1 x_head': a_head[0, 1],

                ## Track 2 coord
                'Track_2 x_tail': b_tail[0, 0],
                'Track_2 x_tail': b_tail[0, 1],
                'Track_2 x_body': point_b[0, 0],
                'Track_2 y_body': point_b[0, 1],
                'Track_2 x_head': b_head[0, 0],
                'Track_2 y_head': b_head[0, 1]
                

            })

            prev_frame = frame  # Increment interaction count

# Create and save DataFrame
results_df = pd.DataFrame(results)
if not results_df.empty:
        # Save to CSV

    results_df.set_index('Frame', inplace=True, drop=False)


    filepath = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-proximal-interactions/interaction-test-2'
    filename =  'interactions.csv'
    full_path = os.path.join(filepath, filename)

    results_df.to_csv(full_path, index=False)



# ### VIDEO FILES

# df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-proximal-interactions/interaction-df-2/proximity_results.csv')

# # pixel_columns = ['Track A X Body',  'Track A Y Body',  'Track B X Body',  'Track B Y Body']
# # df[pixel_columns] = df[pixel_columns] * (1032/90)

# columns = [col for col in df.columns if col.startswith("Interaction")]


# video_path = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-proximal-interactions/interaction-df-2/2024-07-15_12-10-38_td2.mp4'
# output_dir = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-proximal-interactions/interaction-df-2'
# os.makedirs(output_dir, exist_ok=True)


# # Open the video file
# cap = cv2.VideoCapture(video_path)
# fps = cap.get(cv2.CAP_PROP_FPS)
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# for column in columns:

#     match = re.search(r"(\d+),(\d+)", column)
#     if not match:
#         continue  # Skip if no match found

#     track_A, track_B = match.groups()
#     track_A_x = f"track_{track_A} x_body"
#     track_A_y = f"track_{track_A} y_body"
#     track_B_x = f"track_{track_B} x_body"
#     track_B_y = f"track_{track_B} y_body"

#     # Ensure track columns exist in DataFrame
#     if track_A_x not in df.columns or track_A_y not in df.columns or track_B_x not in df.columns or track_B_y not in df.columns:
#         print(f"Skipping {column} because track columns are missing.")
#         continue



#     true_frames = df.loc[df[column] == True]['Frame'].sort_values().reset_index(drop=True)


    
#     segments = []
#     start_frame = true_frames.iloc[0]
#     prev_frame = start_frame

#     for current_frame in true_frames.iloc[1:]:
#         if current_frame - prev_frame != 1:  # If gap detected, close the segment
#             segments.append((start_frame, prev_frame))
#             start_frame = current_frame
#         prev_frame = current_frame

#     segments.append((start_frame, prev_frame))  # Add last segment

#         # Process each segment
#     for idx, (start, end) in enumerate(segments):
#         output_filename = f"{column.replace(' ', '_')}_{start}-{end}.mp4"
#         output_filepath = os.path.join(output_dir, output_filename)

#         # Open video writer
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter(output_filepath, fourcc, fps, (frame_width, frame_height))

#         # Extract frames
#         for frame_num in range(start, end + 1):
#             cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
#             ret, frame = cap.read()

#             if not ret:
#                 break
            
#             # Get track positions for this frame
#             # frame_data = true_frames[true_frames['Frame'] == frame_num]
#             # if not frame_data.empty:
#             #     x_A, y_A = int(frame_data[track_A_x].values[0]), int(frame_data[track_A_y].values[0])
#             #     x_B, y_B = int(frame_data[track_B_x].values[0]), int(frame_data[track_B_y].values[0])

#             #     # Draw circles on video frame at track locations
#             #     cv2.circle(frame, (x_A, y_A), 10, (0, 0, 255), -1)  # Red for Track A
#             #     cv2.circle(frame, (x_B, y_B), 10, (255, 0, 0), -1)  # Blue for Track B

#             #     # Add text labels
#             #     cv2.putText(frame, f"Track {track_A}", (x_A + 15, y_A - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#             #     cv2.putText(frame, f"Track {track_B}", (x_B + 15, y_B - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#             # # Add label overlay (frame number & interaction name)
#             label = f"{column} - Frame {frame_num}"
#             cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

#             # Write frame to output video
#             out.write(frame)

#         out.release()  # Close writer
    
# cap.release()
