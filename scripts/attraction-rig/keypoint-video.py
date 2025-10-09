
# %% IMPORTS 

import pandas as pd 
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import random
import shapely.wkt


######################################################################################################################################################## 
# %% DETERMINE COLOUR OF TRACK DEPENDING ON BOOLEAN VARIABLE 


df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-digging-2/n2/test.csv')
original_video = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-digging-2/n2/2025-03-03_14-03-34_td4.mp4'
output = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-digging-2/n2/2025-03-03_14-03-34_td4_digging.mp4'



mm_to_pixel = ['y_body', 'x_body']
df[mm_to_pixel] = df[mm_to_pixel] * (1046/90)


# Set the im= age size
image_size = 1400

# Open the original MP4 video
original_video_path = original_video
original_video = cv2.VideoCapture(original_video_path)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec used to compress the frames
video_output = cv2.VideoWriter(output, fourcc, 25.0, (image_size, image_size))

df = df.sort_values('frame')

# Iterate over frames in the original video
frame_number = 0
while original_video.isOpened():
    ret, frame = original_video.read()  # Read the next frame from the original video
    if not ret:
        break  # Exit if no more frames

    # Get the data for the current frame from the CSV
    frame_df = df[df['frame'] == frame_number]

    # Overlay the circles from the CSV data onto the frame
    for i, row in frame_df.iterrows():
        body_x = row['x_body']
        body_y = row['y_body']

        # Skip if coordinates are NaN
        if not np.isnan([body_x, body_y]).any():
            # Convert coordinates to integers
            body_x, body_y = int(body_x), int(body_y)

            # Determine color based on the boolean column
            if row['digging_status']:
                color = (255, 0, 0)  # RED for moving outside
            else:
                color = (0, 0, 255)  # BLUE for not moving outside

            # Overlay the circle on the original video frame
            cv2.circle(frame, (body_x, body_y), radius=4, color=color, thickness=-1)

    # Write the updated frame to the output video
    video_output.write(frame)

    # Move to the next frame
    frame_number += 1


original_video.release()
video_output.release()



######################################################################################################################################################## 
# %% DETERMINE COLOUR OF TRACK DEPENDING ON BOOLEAN VARIABLE 


df = pd.read_csv('/Users/cochral/Desktop/SLAEP/TRain/testing-nemo-down/test.csv')
original_video = '/Users/cochral/Desktop/SLAEP/TRain/testing-nemo-down/2025-02-25_14-25-25_td11_holes.mp4'
output = '/Users/cochral/Desktop/SLAEP/TRain/testing-nemo-down/2025-02-25_14-25-25_td11_holes_DIGGING.mp4'



mm_to_pixel = ['y_body', 'x_body']
df[mm_to_pixel] = df[mm_to_pixel] * (1046/90)


# Set the im= age size
image_size = 1400

# Open the original MP4 video
original_video_path = original_video
original_video = cv2.VideoCapture(original_video_path)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec used to compress the frames
video_output = cv2.VideoWriter(output, fourcc, 25.0, (image_size, image_size))

df = df.sort_values('frame')

# Iterate over frames in the original video
frame_number = 0
while original_video.isOpened():
    ret, frame = original_video.read()  # Read the next frame from the original video
    if not ret:
        break  # Exit if no more frames

    # Get the data for the current frame from the CSV
    frame_df = df[df['frame'] == frame_number]

    # Overlay the circles from the CSV data onto the frame
    for i, row in frame_df.iterrows():
        body_x = row['x_body']
        body_y = row['y_body']

        # Skip if coordinates are NaN
        if not np.isnan([body_x, body_y]).any():
            # Convert coordinates to integers
            body_x, body_y = int(body_x), int(body_y)

            # Determine color based on the boolean column
            if row['digging_outside_hole']:
                color = (255, 0, 0)  # RED for moving outside
            else:
                continue
            #     color = (0, 0, 255)  # BLUE for not moving outside

            # Overlay the circle on the original video frame
            cv2.circle(frame, (body_x, body_y), radius=4, color=color, thickness=-1)

    # Write the updated frame to the output video
    video_output.write(frame)

    # Move to the next frame
    frame_number += 1


original_video.release()
video_output.release()



######################################################################################################################################################## 
# %% TRACK COLOUR INSIDE SIDE-HOLE

df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/side-hole/food/2024-11-05_15-46-39_td3.tracks.feather_hole_data.csv')
original_video = ''
hole_wkt = ''
ouptput = ''


mm_to_pixel = ['x_body', 'y_body']
df[mm_to_pixel] = df[mm_to_pixel] * (1022.73/90)

# Set the image size
image_size = 1400

# Open the original MP4 video
original_video_path = original_video
original_video = cv2.VideoCapture(original_video_path)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec used to compress the frames
video_output = cv2.VideoWriter(ouptput, fourcc, 25.0, (image_size, image_size))

df = df.sort_values('frame')

# Read the WKT file
wkt_file_path = hole_wkt
with open(wkt_file_path, 'r') as file:
    wkt_data = file.read()

# Parse the WKT polygon
polygon = shapely.wkt.loads(wkt_data)

# Extract the polygon coordinates
boundary_coords = np.array(polygon.exterior.coords)

# Convert the coordinates from mm to pixels
# boundary_coords *= (1034.48 / 90)

# Iterate over frames in the original video
frame_number = 0
while original_video.isOpened():
    ret, frame = original_video.read()  # Read the next frame from the original video
    if not ret:
        break  # Exit if no more frames

    # Get the data for the current frame from the CSV
    frame_df = df[df['frame'] == frame_number]

    # Overlay the circles from the CSV data onto the frame
    for i, row in frame_df.iterrows():
        body_x = row['x_body']
        body_y = row['y_body']

        # Skip if coordinates are NaN
        if not np.isnan([body_x, body_y]).any():
            # Convert coordinates to integers
            body_x, body_y = int(body_x), int(body_y)

            # Determine color based on the boolean column
            if row['moving_outside']:
                color = (255, 0, 0)  # RED for moving outside
            else:
                color = (0, 0, 255)  # BLUE for not moving outside

            # Overlay the circle on the original video frame
            cv2.circle(frame, (body_x, body_y), radius=4, color=color, thickness=-1)

    # Draw the polygon (hole boundary) on the frame
    boundary_coords_int = boundary_coords.astype(int)
    cv2.polylines(frame, [boundary_coords_int], isClosed=True, color=(0, 255, 0), thickness=2)  # Green boundary

    # Write the updated frame to the output video
    video_output.write(frame)

    # Move to the next frame
    frame_number += 1

# Release video resources
original_video.release()
video_output.release()







######################################################################################################################################################## 
# %% VIDEO OF TRACKS ON BLACK BACKGROUND


df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/pseudo_population_9.csv')
ouptput = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/pseudo_population_9.mp4'

mm_to_pixel = ['x_head', 'y_head', 'x_body', 'y_body', 'x_tail', 'y_tail']
df[mm_to_pixel] = df[mm_to_pixel] * (1032/90) 

# for coord in ['x_head', 'x_body', 'x_tail']:
#     df[coord] += 700
# for coord in ['y_head', 'y_body', 'y_tail']:
#     df[coord] += 700


def generate_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# IMAGE SIZE SHOULD BE 1400
image_size = 1400 

# Directory and output settings
output_path = ouptput

# Define the codec and create VideoWriter object
# This object is used to write frames to the output video at 10 frames per second with the defined image size
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec used to compress the frames
video_output = cv2.VideoWriter(output_path, fourcc, 25.0, (image_size, image_size))


# Dictionary to store the last index of each track
  # Stores the last index processed for each track to link the current frame with the previous one
last_index = {}
# Dictionary to maintain unique color assignment for each track
track_colors = {}

# Initialize a cumulative image
  # Initializes a zero-valued image array (cumulative_image) that will be used to draw the paths of the tracked objects
  # dont get this as much 
cumulative_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

# Iterate over each frame
for frame in range(df['frame'].min(), df['frame'].max() + 1):
    # ensure frame is current frame 
    frame_df = df[df['frame'] == frame]

    # Update the track_colors dictionary for any new tracks in the current frame
    for track in frame_df['track_id'].unique():
        if track not in track_colors:
            track_colors[track] = generate_random_color()

    # iterate over each track

    for track in frame_df['track_id'].unique():
        # ensure the track is the current track iteration
        track_df = frame_df[frame_df['track_id'] == track]
        # if there is no data for the track in the current frame it skips the rest of the loop
        if track_df.empty:
            continue

        # Stores the current index of the track data
           # This helps in linking positions between consecutive frames.

        current_index = track_df.index[0]

        # If the track was present in the previous frame (checked using last_index), it retrieves the previous coordinates and draws lines from the previous positions to the current positions for each body part (head, body, tail)

        if track in last_index:
            # Retrieve the previous frame's data using the stored last index
            prev_index = last_index[track]
            prev_track_df = df.loc[prev_index]

            # iterate over each part 
            for part in ['head', 'body', 'tail']:
                prev_x = prev_track_df[f'x_{part}']
                prev_y = prev_track_df[f'y_{part}']
                curr_x = track_df[f'x_{part}'].values[0]
                curr_y = track_df[f'y_{part}'].values[0]

                # Check if any of the coordinates are NaN before drawing
        
                if not np.isnan([prev_x, prev_y, curr_x, curr_y]).any():
                    # Converts coordinates to integers
                    prev_x, prev_y, curr_x, curr_y = map(int, [prev_x, prev_y, curr_x, curr_y])
                    # Draw the line on cumulative_image with the corresponding track color
                    cv2.line(cumulative_image, (prev_x, prev_y), (curr_x, curr_y),
                             color=track_colors[track], thickness=1)

        # Update the last index for this track
          # important for next frame processing 
        last_index[track] = current_index

    # Write the cumulative image to the video output
    video_output.write(cumulative_image)


video_output.release()


print("Track Colors:")
for track_id, color in track_colors.items():
    print(f"Track {track_id}: RGB{color}")


######################################################################################################################################################## 
#%% TRACKS WITH DESIGNATED PALETTE ON BLACK BACKGROUND

df = pd.read_feather('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/video-keypoint/original-videos/2025-03-25_09-11-04_td13.tracks.feather')
output_path = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/video-keypoint/2025-03-25_09-11-04_td13.mp4'

track_colors = {
    0: (249, 236, 48),   # Red
    # 1: (0, 255, 0),   # Green
    # 2: (0, 0, 255),   # Blue
    # 3: (255, 255, 0), # Yellow
    # # Add more track_id: (R, G, B) pairs as needed
}

# Image size and output settings
image_size = 1400
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_output = cv2.VideoWriter(output_path, fourcc, 25.0, (image_size, image_size))


last_index = {}
cumulative_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

# Frame loop
for frame in range(df['frame'].min(), df['frame'].max() + 1):
    frame_df = df[df['frame'] == frame]
    for track in frame_df['track_id'].unique():
        track_df = frame_df[frame_df['track_id'] == track]
        if track_df.empty:
            continue

        current_index = track_df.index[0]

        if track in last_index:
            prev_index = last_index[track]
            prev_track_df = df.loc[prev_index]

            for part in ['head', 'body', 'tail']:
                prev_x = prev_track_df[f'x_{part}']
                prev_y = prev_track_df[f'y_{part}']
                curr_x = track_df[f'x_{part}'].values[0]
                curr_y = track_df[f'y_{part}'].values[0]

                if not np.isnan([prev_x, prev_y, curr_x, curr_y]).any():
                    prev_x, prev_y, curr_x, curr_y = map(int, [prev_x, prev_y, curr_x, curr_y])
                    color = track_colors.get(track, (255, 255, 255))  # Default to white if not in dict
                    cv2.line(cumulative_image, (prev_x, prev_y), (curr_x, curr_y), color=color, thickness=1)

        last_index[track] = current_index

    video_output.write(cumulative_image)

video_output.release()



######################################################################################################################################################## 
# %% KEYPOINT VIDEOS OF TRACK DATA FROM TRACKS.FEATHER FILES


import pandas as pd
import numpy as np
import cv2
import random

df = pd.read_feather('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/sleap-keypoint-videos/2025-03-31_16-57-35_td12.tracks.feather')
output = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/sleap-keypoint-videos/2025-03-31_16-57-35_td12.tracks.mp4'

image_size = 1400

# Define the codec and create the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_output = cv2.VideoWriter(output, fourcc, 25.0, (image_size, image_size))

# Assign random colors to tracks
def generate_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

last_index = {}
track_colors = {}


for frame in range(df['frame'].min(), df['frame'].max() + 1):
    cumulative_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    frame_df = df[df['frame'] == frame]

    for track in frame_df['track_id'].unique():
        track_df = frame_df[frame_df['track_id'] == track]
        if track_df.empty:
            continue

        if track not in track_colors:
            track_colors[track] = generate_random_color()

        for part in ['head', 'body', 'tail']:
            curr_x = track_df[f'x_{part}'].values[0]
            curr_y = track_df[f'y_{part}'].values[0]

            if not np.any(np.isnan([curr_x, curr_y])):
                cv2.circle(cumulative_image, (int(curr_x), int(curr_y)), 6, track_colors[track], -1)

        # Add track ID text above body
        x_text = int(track_df['x_body'].values[0])
        y_text = int(track_df['y_body'].values[0]) - 10

        cv2.putText(
            cumulative_image,
            f"{track}",
            (x_text, y_text),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=track_colors[track],
            thickness=2,
            lineType=cv2.LINE_AA
        )

    video_output.write(cumulative_image)

video_output.release()








######################################################################################################################################################## 
# %% TRACK OVERLAY ON ORIGINAL VIDEO



import cv2
import numpy as np
import random
import pandas as pd


df = pd.read_feather('/Volumes/proj-data-challenge/2024/proj-Cochrane/videos-topdown/food/2024-07-26_14-08-00_td8.tracks.feather')
df = df.sort_values('frame')

mm_to_pixel = ['x_body', 'y_body', 'x_head', 'y_head', 'y_tail', 'x_tail']
df[mm_to_pixel] = df[mm_to_pixel] * (1032/90)


original_video = cv2.VideoCapture('/Volumes/proj-data-challenge/2024/proj-Cochrane/videos-topdown/food/2024-07-26_14-08-00_td8.mp4')

# Set the image size
image_size = 1400

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec used to compress the frames
video_output = cv2.VideoWriter('/Volumes/proj-data-challenge/2024/proj-Cochrane/videos-topdown/food/2024-07-26_14-08-00_td8.tracks-visualisation.mp4', fourcc, 25.0, (image_size, image_size))

def generate_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# Dictionary to store the last index of each track
last_index = {}
# Dictionary to maintain color assignment for each track
track_colors = {}

# Iterate over each frame
for frame in range(int(df['frame'].min()), int(df['frame'].max()) + 1):
    frame_df = df[df['frame'] == frame]

    # Initialize a blank image for this frame
    frame_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    # Determine which tracks are currently active
    current_tracks = set(frame_df['track_id'].unique())

    # Identify tracks that have disappeared
    disappeared_tracks = set(last_index.keys()) - current_tracks
    for track in disappeared_tracks:
        del last_index[track]  # Remove disappeared tracks from last_index

    # Update the track_colors dictionary for any new tracks
    for track in current_tracks:
        if track not in track_colors:
            track_colors[track] = generate_random_color()

    for track in current_tracks:
        track_df = frame_df[frame_df['track_id'] == track]
        if track_df.empty:
            continue

        current_index = track_df.index[0]

        if track in last_index:
            # Retrieve the previous frame's data using the stored last index
            prev_index = last_index[track]
            prev_track_df = df.loc[prev_index]

            for part in ['head', 'body', 'tail']:
                prev_x = prev_track_df[f'x_{part}']
                prev_y = prev_track_df[f'y_{part}']
                curr_x = track_df[f'x_{part}'].values[0]
                curr_y = track_df[f'y_{part}'].values[0]

                # Check if any of the coordinates are NaN before drawing
                if not np.isnan([prev_x, prev_y, curr_x, curr_y]).any():
                    # Convert to integers
                    prev_x, prev_y, curr_x, curr_y = map(int, [prev_x, prev_y, curr_x, curr_y])
                    # Draw the line with the corresponding track color
                    cv2.line(frame_image, (prev_x, prev_y), (curr_x, curr_y),
                             color=track_colors[track], thickness=4)

        # Update the last index for this track
        last_index[track] = current_index

    # Write the frame to the video output
    video_output.write(frame_image)

# Release the video writer
video_output.release()


###################################################
# %% MOSEQ SYLLABLE OVERLAY

import pandas as pd
import numpy as np
import cv2

# Load CSV
df = pd.read_csv('/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA50/2025_06_03-13_48_49/moseq_df.csv')

print(df.head(10))
f = df[df['name'] == 'N1-GH_2025-02-24_15-16-50_td7']

print(f.head(10))


f = f.sort_values('frame_index')

# Update column names here if needed
coord_columns = ['centroid_x', 'centroid_y']  # replace with your actual centroid column names


# Set video parameters
image_size = 1400
original_video = cv2.VideoCapture('/Users/cochral/Desktop/MOSEQ/videos/N1-GH_2025-02-24_15-16-50_td7.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_output = cv2.VideoWriter('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/moseq/behavioural_syllable_videos/N1-GH_2025-02-24_15-16-50_td7_50.mp4', fourcc, 25.0, (image_size, image_size))

# Sort by frame for consistency
f = f.sort_values('frame_index')

frame_number = 0
while original_video.isOpened():
    ret, frame = original_video.read()
    if not ret:
        break

    frame_df = f[f['frame_index'] == frame_number]

    for _, row in frame_df.iterrows():
        x = row['centroid_x']
        y = row['centroid_y']
        if np.isnan([x, y]).any():
            continue

        x, y = int(x), int(y)

        # Optional: Change color based on condition
        color = (0, 255, 0)

        # Draw circle at centroid
        cv2.circle(frame, (x, y), radius=4, color=color, thickness=-1)

        # Get syllable (or any column you want to annotate)
        syllable = str(row['syllable'])  # make sure it's a string

        # Put text above the centroid
        cv2.putText(frame, syllable, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    video_output.write(frame)
    frame_number += 1

original_video.release()
video_output.release()



# %% ###################### 

############################ OVERLAYING BOUTS <1MM OF CONTACT 

import cv2
import pandas as pd
import os
from collections import defaultdict


df_bouts = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/holes/N10-1-HOLE/GROUP-HOUSED/interactions_return.csv')
df_bouts = df_bouts[df_bouts['file'] == '2025-03-07_11-48-26_td14.tracks.feather']

df_bouts = df_bouts[df_bouts['interacted'] == True] ## returns

df = pd.read_feather('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/holes/N10-1-HOLE/GROUP-HOUSED/2025-03-07_11-48-26_td14.tracks.feather')

video = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/holes/N10-1-HOLE/GROUP-HOUSED/2025-03-07_11-48-26_td14.mp4'

output = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-interaction-bouts/2025-03-07_11-48-26_td14_bout.mp4'

# --- Setup video reader/writer ---
cap = cv2.VideoCapture(video)
# fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), 1, (w, h))


# Precompute a lookup of frame → bouts
frame_to_bouts = defaultdict(list)
for _, row in df_bouts.iterrows():
    for frame in range(row['start_frame'], row['end_frame'] + 1):
        frame_to_bouts[frame].append((row['exiting_larva'], row['partner']))

frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get interactions happening in this frame
    active_bouts = frame_to_bouts.get(frame_idx, [])

    # Plot tracking overlay from df (frame_idx match)
    frame_tracks = df[df['frame'] == frame_idx]

    for _, row in frame_tracks.iterrows():
        x, y = int(row['x_body']), int(row['y_body'])
        track_id = int(row['track_id'])

        # Check if this track is in an active bout
        is_in_bout = any(track_id in bout for bout in active_bouts)

        color = (0, 0, 255) if is_in_bout else (255, 255, 255)  # Red if in bout, white otherwise
        cv2.circle(frame, (x, y), 5, color, -1)

    writer.write(frame)
    frame_idx += 1



cap.release()
writer.release()
cv2.destroyAllWindows()


# %%
