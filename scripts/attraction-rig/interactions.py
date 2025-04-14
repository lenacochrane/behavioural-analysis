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

######################
# %% IDETNFIY INTERACTIONS AND CREATE CSV 

df = pd.read_feather('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/test-umap/2025-03-04_17-08-33_td2.tracks.feather')

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


    filepath = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/test-umap'
    filename =  'interactions_group.csv'
    full_path = os.path.join(filepath, filename)

    results_df.to_csv(full_path, index=False)




######################
#%% DISTANCES BETWEEN ALL BODY PART COMBINATIONS 

df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/test-umap/interactions_group.csv')



def euclidean_distance(df, x1, y1, x2, y2):
    return np.sqrt((df[x1] - df[x2])**2 + (df[y1] - df[y2])**2)

## need to compute per row the distances between body parts (interaction doesnt matter its per row)

df['t1_tail-tail_t2'] = euclidean_distance(df, 'Track_1 x_tail', 'Track_1 y_tail', 'Track_2 x_tail', 'Track_2 y_tail')
df['t1_tail-body_t2'] = euclidean_distance(df, 'Track_1 x_tail', 'Track_1 y_tail', 'Track_2 x_body', 'Track_2 y_body')
df['t1_tail-head_t2'] = euclidean_distance(df, 'Track_1 x_tail', 'Track_1 y_tail', 'Track_2 x_head', 'Track_2 y_head')

df['t1_body-tail_t2'] = euclidean_distance(df,'Track_1 x_body', 'Track_1 y_body', 'Track_2 x_tail', 'Track_2 y_tail')
df['t1_body-body_t2'] = euclidean_distance(df, 'Track_1 x_body', 'Track_1 y_body', 'Track_2 x_body', 'Track_2 y_body')
df['t1_body-head_t2'] = euclidean_distance(df, 'Track_1 x_body', 'Track_1 y_body', 'Track_2 x_head', 'Track_2 y_head')

df['t1_head-tail_t2'] = euclidean_distance(df, 'Track_1 x_head', 'Track_1 y_head', 'Track_2 x_tail', 'Track_2 y_tail')
df['t1_head-body_t2'] = euclidean_distance(df, 'Track_1 x_head', 'Track_1 y_head', 'Track_2 x_body', 'Track_2 y_body')
df['t1_head-head_t2'] = euclidean_distance(df, 'Track_1 x_head', 'Track_1 y_head', 'Track_2 x_head', 'Track_2 y_head')


filepath = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/test-umap'
filename =  'interactions_group.csv'
full_path = os.path.join(filepath, filename)

df.to_csv(full_path, index=False)


#%% QUANTIFICATIONS OF INTERACTIONS FOR UMAP


df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/test-umap/interactions_group.csv')

df['Frame'] = df['Frame'].astype(int)

# SPEED

def speed(group, x, y):

    dx = group[x].diff()
    dy = group[y].diff()
    
    distance = np.sqrt(dx**2 + dy**2)
    dt = group['Frame'].diff()
    
    # Avoid division by zero
    speed = distance / dt.replace(0, np.nan)
    
    return speed

df['track1_speed'] = df.groupby('Interaction Number').apply(lambda group: speed(group, 'Track_1 x_body', 'Track_1 y_body')).reset_index(level=0, drop=True)

df['track2_speed'] = df.groupby('Interaction Number').apply(lambda group: speed(group, 'Track_2 x_body', 'Track_2 y_body')).reset_index(level=0, drop=True)


# ACCELERATION

df['track1_acceleration'] = df.groupby('Interaction Number')['track1_speed'].diff() / df.groupby('Interaction Number')['Frame'].diff()
df['track2_acceleration'] = df.groupby('Interaction Number')['track2_speed'].diff() / df.groupby('Interaction Number')['Frame'].diff()

# TAIL-BODY-HEAD LENGTH


import numpy as np

df['track1_length'] = (
    np.sqrt((df['Track_1 x_body'] - df['Track_1 x_tail'])**2 + 
            (df['Track_1 y_body'] - df['Track_1 y_tail'])**2) 
    +
    np.sqrt((df['Track_1 x_head'] - df['Track_1 x_body'])**2 + 
            (df['Track_1 y_head'] - df['Track_1 y_body'])**2)
)


df['track2_length'] = (
    np.sqrt((df['Track_2 x_body'] - df['Track_2 x_tail'])**2 + 
            (df['Track_2 y_body'] - df['Track_2 y_tail'])**2) 
    +
    np.sqrt((df['Track_2 x_head'] - df['Track_2 x_body'])**2 + 
            (df['Track_2 y_head'] - df['Track_2 y_body'])**2)
)

# ANGLE BETWEEN TAIL-BODY AND BODY-HEAD PARTS

# Tail-Body Vector for Track 1
df['track1 TB_x'] =  df['Track_1 x_tail'] - df['Track_1 x_body'] 
df['track1 TB_y'] =  df['Track_1 y_tail'] - df['Track_1 y_body'] 
# Body-Head Vector for Track 1
df['track1 BH_x'] = df['Track_1 x_head'] - df['Track_1 x_body']
df['track1 BH_y'] = df['Track_1 y_head'] - df['Track_1 y_body']
# Tail-Body Vector for Track 2
df['track2 TB_x'] = df['Track_2 x_tail'] - df['Track_2 x_body'] 
df['track2 TB_y'] = df['Track_2 y_tail'] - df['Track_2 y_body'] 
# Body-Head Vector for Track 2
df['track2 BH_x'] = df['Track_2 x_head'] - df['Track_2 x_body']
df['track2 BH_y'] = df['Track_2 y_head'] - df['Track_2 y_body']


def calculate_angle(df, v1_x, v1_y, v2_x, v2_y):
    dot_product = (df[v1_x] * df[v2_x]) + (df[v1_y] * df[v2_y])

    magnitude_v1 = np.hypot(df[v1_x], df[v1_y])  # Same as sqrt(x^2 + y^2)
    magnitude_v2 = np.hypot(df[v2_x], df[v2_y])
    
    # Avoid division by zero
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure values are in valid range for arccos
    
    return np.degrees(np.arccos(cos_theta))  # Convert radians to degrees


# Calculate angles for each track
df['track1_angle'] = calculate_angle(df,'track1 TB_x', 'track1 TB_y', 'track1 BH_x', 'track1 BH_y')
df['track2_angle'] = calculate_angle(df, 'track2 TB_x', 'track2 TB_y', 'track2 BH_x', 'track2 BH_y')


filepath = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/test-umap/interactions_group.csv'
full_path = os.path.join(filepath)

df.to_csv(full_path, index=False)

# ANGLE BETWEEN INTERACTION PARTNERS




######################
# %% IDENTIFY CLOSEST POINT OF INTERACTION AND NORMALISE FRAMES

df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/test-umap/interactions_group.csv')


# this

# Define distance columns
distance_columns = [
    't1_tail-tail_t2', 't1_tail-body_t2', 't1_tail-head_t2',
    't1_body-tail_t2', 't1_body-body_t2', 't1_body-head_t2',
    't1_head-tail_t2', 't1_head-body_t2', 't1_head-head_t2'
]

df["min_distance"] = df[distance_columns].min(axis=1) # identifies smallest numerical value
df["interaction_type"] = df[distance_columns].idxmin(axis=1) # returns column name holding smallest value
df["interaction_type"] = df["interaction_type"].str.extract(r"t1_(.*-.*)_t2")


min_distance_frames = df.groupby("Interaction Number")["min_distance"].idxmin()

# Function to normalize frames based on min distance
def normalize_frames(group):
    min_frame = group.loc[min_distance_frames[group.name], "Frame"]  # Get the min distance frame
    group["Normalized Frame"] = group["Frame"] - min_frame  # Normalize all frames in the group
    return group

# Apply normalization within each group
df = df.groupby("Interaction Number", group_keys=False).apply(normalize_frames)


desired_order = ["Frame", "Interaction Number", "Normalized Frame"]
df = df[desired_order + [col for col in df.columns if col not in desired_order]]


filepath = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/test-umap'
filename = 'interactions_group.csv'
df.to_csv(os.path.join(filepath, filename), index=False)

#df.to_feather('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-proximal-interactions/test-interaction-3/interactions.feather')





#####################
# %% KEYPOINT VIDEO OVERLAY FILES

import pandas as pd
import ast
import cv2
import os

# Load the DataFrame
df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-proximal-interactions/test-interaction-3/interactions.csv')

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
video_path = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-proximal-interactions/test-interaction-3/n9_agarose-2024-08-27_16-15-39_td3.mp4'
output_dir = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-proximal-interactions/test-interaction-3'
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



#####################
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


#####################
# %% CENTRED KEYPOINT VIDEOS B4 NORMALISATION


import pandas as pd
import ast
import cv2
import numpy as np
import os

# Load the DataFrame
df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-proximal-interactions/test-n2/interactions.csv')

columns = [
    "Track_1 x_tail", "Track_1 y_tail", "Track_1 x_body", "Track_1 y_body", "Track_1 x_head", "Track_1 y_head",
    "Track_2 x_tail", "Track_2 y_tail", "Track_2 x_body", "Track_2 y_body", "Track_2 x_head", "Track_2 y_head"

]
df[columns] *= (1032 / 90)

# Convert 'Interaction Pair' column from string to list


df["Interaction Pair"] = df["Interaction Pair"].apply(ast.literal_eval)

# Define output directory
output_dir = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-proximal-interactions/test-n2'
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



#####################
# %% NORMALISE COORDINATES TO MIDPOINT OF BODY COORDINATES AT THE CLOSEST DISTANCE  

import pandas as pd
import numpy as np
import os

# Load DataFrame
df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/test-umap/interactions_group.csv')

distance_columns = [
    't1_tail-tail_t2', 't1_tail-body_t2', 't1_tail-head_t2',
    't1_body-tail_t2', 't1_body-body_t2', 't1_body-head_t2',
    't1_head-tail_t2', 't1_head-body_t2', 't1_head-head_t2'
]

# Ensure all distance columns are numeric
df[distance_columns] = df[distance_columns].apply(pd.to_numeric, errors='coerce')
print(df[distance_columns].dtypes) # floats


coordinate_columns = [
    "Track_1 x_body", "Track_1 y_body", "Track_2 x_body", "Track_2 y_body",
    "Track_1 x_tail", "Track_1 y_tail", "Track_2 x_tail", "Track_2 y_tail",
    "Track_1 x_head", "Track_1 y_head", "Track_2 x_head", "Track_2 y_head"
]

for interaction in df["Interaction Number"].unique():
    interaction_data = df[df["Interaction Number"] == interaction]

    min_distance_row = interaction_data[interaction_data["Normalized Frame"] == 0]

    if min_distance_row.empty:
        continue  # Skip if no frame is marked as "0"

    min_distance_row = min_distance_row.iloc[0]  # Get the first occurrence #idk why more than 1 

    # Step 2: Identify which body part pair had the smallest distance at this frame
    closest_part = min_distance_row[distance_columns].astype(float).idxmin()  # Ensure numeric dtype
    closest_part = str(closest_part)  # Force conversion to string

    # Step 3: Extract the x and y coordinates for that closest part
    part1, part2 = closest_part.split("-")  # Extract body part names from column name

    # Convert body part names into corresponding coordinate column names
    part1_x = f"Track_1 x_{part1.split('_')[-1]}"
    part1_y = f"Track_1 y_{part1.split('_')[-1]}"
    part2_x = f"Track_2 x_{part2.split('_')[0]}"
    part2_y = f"Track_2 y_{part2.split('_')[0]}"

    # Compute midpoint of the closest body part pair
    mid_x = (min_distance_row[part1_x] + min_distance_row[part2_x]) / 2
    mid_y = (min_distance_row[part1_y] + min_distance_row[part2_y]) / 2

    # Step 4: Normalize all coordinates for this interaction
    for col in coordinate_columns:
        if "x_" in col:
            df.loc[df["Interaction Number"] == interaction, col] -= mid_x
        elif "y_" in col:
            df.loc[df["Interaction Number"] == interaction, col] -= mid_y

# Save DataFrame
filepath = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/test-umap'
filename = 'interactions_group_normalised.csv'
df.to_csv(os.path.join(filepath, filename), index=False)




#######################################################
# %% KEYPOINT VIDEOS POST NORMALISATION

import pandas as pd
import ast
import cv2
import numpy as np
import os

# Load the DataFrame
df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-proximal-interactions/test-n2/interactions.csv')

columns = [
    "Track_1 x_tail", "Track_1 y_tail", "Track_1 x_body", "Track_1 y_body", "Track_1 x_head", "Track_1 y_head",
    "Track_2 x_tail", "Track_2 y_tail", "Track_2 x_body", "Track_2 y_body", "Track_2 x_head", "Track_2 y_head"
]
df[columns] *= (1032 / 90)

# Convert 'Interaction Pair' column from string to list
df["Interaction Pair"] = df["Interaction Pair"].apply(ast.literal_eval)

# Define output directory
output_dir = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-proximal-interactions/test-n2/'
os.makedirs(output_dir, exist_ok=True)

# Video settings
fps = 30  
crop_size = 1300  


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

        # Create a black background **AT THE CROP SIZE**
        cropped_frame = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)

        keypoints_shifted = keypoints + (crop_size // 2)

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



#####################
# %% UMAP 


## confused different sizes is an issue apparanelt 
### i shd do a pairplot first 


import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 📌 Load interaction dataset
df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-proximal-interactions/test-interaction-3/interactions.csv')


feature_columns = [
    "min_distance",  
    "track1_speed", "track2_speed", 
    "track1_acceleration", "track2_acceleration",
    "track1_length", "track2_length",  
    "track1_angle", "track2_angle"]


# 📌 Function to crop interaction to [-15, +15] around Normalized Frame == 0
def crop_interaction(group):
    center_frame = group.loc[group["Normalized Frame"] == 0]  # Find the reference frame
    if center_frame.empty:
        return None  # Skip if there's no frame == 0

    center_index = center_frame.index[0]  # Get index of the frame == 0
    start_index = max(center_index - 15, group.index.min())  # Ensure within range
    end_index = min(center_index + 15, group.index.max())  # Ensure within range
    
    return group.loc[start_index:end_index]  # Crop the dataframe

# 📌 Apply cropping to all interactions
df_cropped = df.groupby("Interaction Number", group_keys=False).apply(crop_interaction)
print(df_cropped.head(100))

# 📌 Pivot to turn cropped interactions into fixed-length vectors
df_vectorized = df_cropped.pivot_table(index="Interaction Number", columns="Normalized Frame", values=feature_columns)

print(df_vectorized.shape)  # Should be (num_interactions, num_features)
print(df_vectorized.index.nunique())  # Should match number of interactions


# 📌 Flatten multi-index column names (e.g., "track1_speed_-10" instead of multi-index)
df_vectorized.columns = [f"{col[0]}_frame{col[1]}" for col in df_vectorized.columns]

# 📌 Fill missing values if any interaction has fewer than 31 frames (unlikely)
df_vectorized = df_vectorized.fillna(0)


# 📌 Normalize the feature vectors (UMAP works best with normalized input)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_vectorized)

# 📌 Apply UMAP
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X_scaled)

# 📌 Store UMAP results
df_vectorized["UMAP_1"], df_vectorized["UMAP_2"] = X_umap[:, 0], X_umap[:, 1]

# 📌 Plot UMAP projection
plt.figure(figsize=(8, 6))
sns.scatterplot(x="UMAP_1", y="UMAP_2", data=df_vectorized, alpha=0.7)
plt.title("UMAP Projection of Cropped Interactions")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.show()






# %%
