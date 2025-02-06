# %% IMPORTS

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


# %% CREATE INTERACTIONS CSV 

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
                'Track_1 y_tail': a_tail[0, 1],
                'Track_1 x_body': point_a[0, 0],
                'Track_1 y_body': point_a[0, 1],
                'Track_1 x_head': a_head[0, 0],
                'Track_1 y_head': a_head[0, 1],

                ## Track 2 coord
                'Track_2 x_tail': b_tail[0, 0],
                'Track_2 y_tail': b_tail[0, 1],
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


# %% IDENTIFY CLOSEST POINT OF INTERACTION AND NORMALISE FRAMES

df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-proximal-interactions/interaction-test-2/interactions.csv')

min_distance_frames = df.groupby("Interaction Number")["Distance"].idxmin() #identify row with smallest dist

def normalize_frames(group):
    min_frame = group.loc[min_distance_frames[group.name], "Frame"]  # Get the min distance frame
    group["Normalized Frame"] = group["Frame"] - min_frame  # Subtract min frame from all frames in group
    return group

# Apply normalization within each group
df = df.groupby("Interaction Number", group_keys=False).apply(normalize_frames)

desired_order = ["Frame", "Interaction Number", "Normalized Frame"]

# Reorder the DataFrame
df = df[desired_order + [col for col in df.columns if col not in desired_order]]

filepath = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-proximal-interactions/interaction-test-2'
filename =  'interactions.csv'
df.to_csv(os.path.join(filepath, filename), index=False)





# %% KEYPOINT VIDEO OVERLAY FILES

import pandas as pd
import ast
import cv2
import os

# Load the DataFrame
df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-proximal-interactions/interaction-test-2/interactions.csv')

columns = [
    "Track_1 x_tail", "Track_1 y_tail", "Track_1 x_body", "Track_1 y_body", "Track_1 x_head", "Track_1 y_head",
    "Track_2 x_tail", "Track_2 y_tail", "Track_2 x_body", "Track_2 y_body", "Track_2 x_head", "Track_2 y_head"
]

df[columns] *= (1032 / 90)

# Convert 'Interaction Pair' column from string to list
df["Interaction Pair"] = df["Interaction Pair"].apply(ast.literal_eval)  # Convert "[0,1]" → [0,1]

# # Extract Track A (Left) and Track B (Right)
# df["Track A"] = df["Interaction Pair"].apply(lambda x: x[0])
# df["Track B"] = df["Interaction Pair"].apply(lambda x: x[1])

# Define video paths
video_path = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-proximal-interactions/interaction-test-2/2024-07-12_13-18-27_td1.mp4'
output_dir = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-proximal-interactions/interaction-test-2'
os.makedirs(output_dir, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Iterate through each unique interaction
for interaction in df["Interaction Number"].unique():
    interaction_data = df[df["Interaction Number"] == interaction]  # Select frames for this interaction

    output_filename = f"interaction_{interaction}.mp4"
    output_filepath = os.path.join(output_dir, output_filename)

    # Open video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filepath, fourcc, fps, (frame_width, frame_height))

    for _, row in interaction_data.iterrows():
        frame_num = row["Frame"]
        track_1, track_2 = row["Interaction Pair"]  # Extract both track IDs dynamically

        # Read the video frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break

        # Get coordinates for Track_1 and Track_2 (column names are fixed)
        coords_1 = {
            "head": (row["Track_1 x_head"], row["Track_1 y_head"]),
            "body": (row["Track_1 x_body"], row["Track_1 y_body"]),
            "tail": (row["Track_1 x_tail"], row["Track_1 y_tail"]),
        }
        coords_2 = {
            "head": (row["Track_2 x_head"], row["Track_2 y_head"]),
            "body": (row["Track_2 x_body"], row["Track_2 y_body"]),
            "tail": (row["Track_2 x_tail"], row["Track_2 y_tail"]),
        }

        # Draw circles for Track_1
        cv2.circle(frame, (int(coords_1["head"][0]), int(coords_1["head"][1])), 2, (0, 255, 0), -1)  
        cv2.circle(frame, (int(coords_1["body"][0]), int(coords_1["body"][1])), 2, (0, 255, 0), -1)  
        cv2.circle(frame, (int(coords_1["tail"][0]), int(coords_1["tail"][1])), 2, (0, 255, 0), -1) 

        # Draw circles for Track_2
        cv2.circle(frame, (int(coords_2["head"][0]), int(coords_2["head"][1])), 2, (255, 0, 255), -1)  
        cv2.circle(frame, (int(coords_2["body"][0]), int(coords_2["body"][1])), 2, (255, 0, 255), -1)  
        cv2.circle(frame, (int(coords_2["tail"][0]), int(coords_2["tail"][1])), 2, (255, 0, 255), -1) 

        track_1_id, track_2_id = row["Interaction Pair"] 

        # Add Labels
        cv2.putText(frame, f"Track {track_1_id}", (int(coords_1["body"][0]) + 15, int(coords_1["body"][1]) - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Track {track_2_id}", (int(coords_2["body"][0]) + 15, int(coords_2["body"][1]) - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # Write frame to output video
        out.write(frame)

    out.release()  # Close writer

# Release the video file
cap.release()
print("Annotated interaction videos saved successfully.")


# %% KEYPOINT VIDEOS B4 NORMALISATION

import pandas as pd
import ast
import cv2
import numpy as np
import os

# Load the DataFrame
df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-proximal-interactions/interaction-test-2/interactions.csv')

columns = [
    "Track_1 x_tail", "Track_1 y_tail", "Track_1 x_body", "Track_1 y_body", "Track_1 x_head", "Track_1 y_head",
    "Track_2 x_tail", "Track_2 y_tail", "Track_2 x_body", "Track_2 y_body", "Track_2 x_head", "Track_2 y_head"
]
df[columns] *= (1032 / 90)

# Convert 'Interaction Pair' column from string to list
df["Interaction Pair"] = df["Interaction Pair"].apply(ast.literal_eval)

# Define output directory
output_dir = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-proximal-interactions/interaction-test-2/keypoints-only'
os.makedirs(output_dir, exist_ok=True)

# Video settings
fps = 30  
frame_width = 1400  # Using original resolution
frame_height = 1400

# Iterate through each unique interaction
for interaction in df["Interaction Number"].unique():
    interaction_data = df[df["Interaction Number"] == interaction]

    output_filename = f"interaction_{interaction}_keypoints.mp4"
    output_filepath = os.path.join(output_dir, output_filename)

    # Open video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filepath, fourcc, fps, (frame_width, frame_height))

    for _, row in interaction_data.iterrows():
        frame_num = row["Frame"]
        track_1, track_2 = row["Interaction Pair"]

        # Create a black background
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        # Get coordinates for Track_1 and Track_2
        coords_1 = {
            "head": (row["Track_1 x_head"], row["Track_1 y_head"]),
            "body": (row["Track_1 x_body"], row["Track_1 y_body"]),
            "tail": (row["Track_1 x_tail"], row["Track_1 y_tail"]),
        }
        coords_2 = {
            "head": (row["Track_2 x_head"], row["Track_2 y_head"]),
            "body": (row["Track_2 x_body"], row["Track_2 y_body"]),
            "tail": (row["Track_2 x_tail"], row["Track_2 y_tail"]),
        }

        # Draw circles for Track_1 (Green)
        cv2.circle(frame, (int(coords_1["head"][0]), int(coords_1["head"][1])), 5, (0, 255, 0), -1)  
        cv2.circle(frame, (int(coords_1["body"][0]), int(coords_1["body"][1])), 5, (0, 255, 0), -1)  
        cv2.circle(frame, (int(coords_1["tail"][0]), int(coords_1["tail"][1])), 5, (0, 255, 0), -1) 

        # Draw circles for Track_2 (Magenta)
        cv2.circle(frame, (int(coords_2["head"][0]), int(coords_2["head"][1])), 5, (255, 0, 255), -1)  
        cv2.circle(frame, (int(coords_2["body"][0]), int(coords_2["body"][1])), 5, (255, 0, 255), -1)  
        cv2.circle(frame, (int(coords_2["tail"][0]), int(coords_2["tail"][1])), 5, (255, 0, 255), -1) 

        # Add Labels
        cv2.putText(frame, f"Track {track_1}", (int(coords_1["body"][0]) + 15, int(coords_1["body"][1]) - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Track {track_2}", (int(coords_2["body"][0]) + 15, int(coords_2["body"][1]) - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # Write frame to output video
        out.write(frame)

    out.release()  # Close writer

print("Keypoint-only interaction videos saved successfully.")



# %% CENTRED KEYPOINT VIDEOS B4 NORMALISATION


import pandas as pd
import ast
import cv2
import numpy as np
import os

# Load the DataFrame
df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-proximal-interactions/interaction-test-2/interactions.csv')

columns = [
    "Track_1 x_tail", "Track_1 y_tail", "Track_1 x_body", "Track_1 y_body", "Track_1 x_head", "Track_1 y_head",
    "Track_2 x_tail", "Track_2 y_tail", "Track_2 x_body", "Track_2 y_body", "Track_2 x_head", "Track_2 y_head"
]
df[columns] *= (1032 / 90)

# Convert 'Interaction Pair' column from string to list
df["Interaction Pair"] = df["Interaction Pair"].apply(ast.literal_eval)

# Define output directory
output_dir = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-proximal-interactions/interaction-test-2/keypoints-only'
os.makedirs(output_dir, exist_ok=True)

# Video settings
fps = 30  
original_width = 1400  
original_height = 1400
crop_size = 500  
buffer = 150  

# Iterate through each unique interaction
for interaction in df["Interaction Number"].unique():
    interaction_data = df[df["Interaction Number"] == interaction]

    output_filename = f"interaction_{interaction}_cropped.mp4"
    output_filepath = os.path.join(output_dir, output_filename)

    # Open video writer at final output size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filepath, fourcc, fps, (crop_size, crop_size))

    for _, row in interaction_data.iterrows():
        frame_num = row["Frame"]
        track_1, track_2 = row["Interaction Pair"]

        # Get keypoint coordinates
        keypoints = np.array([
            [row["Track_1 x_head"], row["Track_1 y_head"]],
            [row["Track_1 x_body"], row["Track_1 y_body"]],
            [row["Track_1 x_tail"], row["Track_1 y_tail"]],
            [row["Track_2 x_head"], row["Track_2 y_head"]],
            [row["Track_2 x_body"], row["Track_2 y_body"]],
            [row["Track_2 x_tail"], row["Track_2 y_tail"]],
        ])

        # Compute bounding box around interaction
        x_min, y_min = np.min(keypoints, axis=0) - buffer
        x_max, y_max = np.max(keypoints, axis=0) + buffer

        # Ensure bounding box is within frame limits
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(original_width, x_max)
        y_max = min(original_height, y_max)

        # Compute the center of the interaction
        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2

        # Define the cropping box to be centered
        crop_x_min = max(0, x_center - crop_size // 2)
        crop_x_max = min(original_width, crop_x_min + crop_size)
        crop_y_min = max(0, y_center - crop_size // 2)
        crop_y_max = min(original_height, crop_y_min + crop_size)

        # Create a black background **AT THE CROP SIZE**
        cropped_frame = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)

        # Shift keypoints so they fit within the crop
        keypoints_shifted = keypoints - np.array([crop_x_min, crop_y_min])

        # Ensure keypoints stay inside the cropped region
        keypoints_shifted = np.clip(keypoints_shifted, 0, crop_size - 1)

        # Extract new positions
        coords_1 = {
            "head": tuple(keypoints_shifted[0].astype(int)),
            "body": tuple(keypoints_shifted[1].astype(int)),
            "tail": tuple(keypoints_shifted[2].astype(int)),
        }
        coords_2 = {
            "head": tuple(keypoints_shifted[3].astype(int)),
            "body": tuple(keypoints_shifted[4].astype(int)),
            "tail": tuple(keypoints_shifted[5].astype(int)),
        }

        # # Ensure keypoints are inside the frame
        # if np.any(keypoints_shifted < 0) or np.any(keypoints_shifted >= crop_size):
        #     print("⚠️ Warning: Some keypoints are out of bounds!")

        # Draw keypoints directly onto the cropped frame (not full-size frame)
        cv2.circle(cropped_frame, coords_1["head"], 5, (0, 255, 0), -1)  
        cv2.circle(cropped_frame, coords_1["body"], 5, (0, 255, 0), -1)  
        cv2.circle(cropped_frame, coords_1["tail"], 5, (0, 255, 0), -1) 

        cv2.circle(cropped_frame, coords_2["head"], 5, (255, 0, 255), -1)  
        cv2.circle(cropped_frame, coords_2["body"], 5, (255, 0, 255), -1)  
        cv2.circle(cropped_frame, coords_2["tail"], 5, (255, 0, 255), -1) 

        # Add Labels
        cv2.putText(cropped_frame, f"Track {track_1}", (coords_1["body"][0] + 15, coords_1["body"][1] - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(cropped_frame, f"Track {track_2}", (coords_2["body"][0] + 15, coords_2["body"][1] - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # Write frame to output video
        out.write(cropped_frame)

    out.release()  # Close writer

print("Cropped interaction videos saved successfully.")



# %% NORMALISE COORDINATES TO MIDPOINT OF MINIMUM DISTANCE


df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-proximal-interactions/interaction-test-2/interactions.csv')





# %%

# %% KEYPOINT VIDEOS POST NORMALISATION







# %%
