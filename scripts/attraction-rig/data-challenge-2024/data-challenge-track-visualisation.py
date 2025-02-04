
import cv2
import numpy as np
import random
import pandas as pd

# Load the tracks data
df = pd.read_feather('/Volumes/proj-data-challenge/2024/proj-Cochrane/videos-topdown/food/2024-07-26_14-08-00_td8.tracks.feather')
df = df.sort_values('frame')

# Convert millimeters to pixels for the relevant columns
mm_to_pixel = ['x_body', 'y_body', 'x_head', 'y_head', 'x_tail', 'y_tail']
df[mm_to_pixel] = df[mm_to_pixel] * (1032 / 90)

# Open the original video
original_video = cv2.VideoCapture('/Volumes/proj-data-challenge/2024/proj-Cochrane/videos-topdown/food/2024-07-26_14-08-00_td8.mp4')

# Set the output video properties
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec used to compress the frames
output_path = '/Volumes/proj-data-challenge/2024/proj-Cochrane/videos-topdown/food/2024-07-26_14-08-00_td8.tracks-visualisation.mp4'
frame_width = int(original_video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(original_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = original_video.get(cv2.CAP_PROP_FPS)
video_output = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Function to generate random color for each track
def generate_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# Dictionary to store the last index of each track
last_index = {}
# Dictionary to maintain color assignment for each track
track_colors = {}

# Iterate over each frame in the range of the data's frames
for frame_num in range(int(df['frame'].min()), int(df['frame'].max()) + 1):
    ret, frame = original_video.read()  # Read the corresponding frame from the original video
    if not ret:
        break  # Exit the loop if no more frames

    # Get the tracking data for the current frame
    frame_df = df[df['frame'] == frame_num]

    # Determine which tracks are currently active in this frame
    current_tracks = set(frame_df['track_id'].unique())

    # Remove tracks that have disappeared
    disappeared_tracks = set(last_index.keys()) - current_tracks
    for track in disappeared_tracks:
        del last_index[track]  # Remove disappeared tracks from last_index

    # Assign random colors to new tracks
    for track in current_tracks:
        if track not in track_colors:
            track_colors[track] = generate_random_color()

    # Draw the tracks directly on the original frame
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
                    # Draw the line directly on the original frame with the corresponding track color
                    cv2.line(frame, (prev_x, prev_y), (curr_x, curr_y), color=track_colors[track], thickness=8)

        # Update the last index for this track
        last_index[track] = current_index

    # Write the frame (with the tracks drawn on it) to the video output
    video_output.write(frame)

# Release the video resources
original_video.release()
video_output.release()

