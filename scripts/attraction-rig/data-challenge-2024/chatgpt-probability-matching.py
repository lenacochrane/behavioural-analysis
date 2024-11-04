import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

# Load data
df = pd.read_feather('/Volumes/proj-data-challenge/2024/proj-Cochrane/videos-topdown/food/2024-07-26_14-08-00_td8.tracks.feather')

# Scale to pixels
mm_to_pixel = ['x_body', 'y_body', 'x_head', 'y_head', 'x_tail', 'y_tail']
df[mm_to_pixel] = df[mm_to_pixel] * (1032 / 90)

# Step 1: Sort the dataframe by 'frame' to ensure proper frame sequencing
df = df.sort_values(by='frame')

# Get all unique frames (they are now sorted)
frames = df['frame'].unique()

# Step 2: Define parameters
average_pixel_movement = 12  # Movement in pixels
distance_threshold = 20  # Define your own threshold based on typical movement in pixels

# Step 3: Iterate through all pairs of frames and compute distances
for i in range(len(frames) - 1):
    current_frame = frames[i]
    next_frame = frames[i + 1]

    # Subset data for the current frame and the next frame
    df_current = df[df['frame'] == current_frame].copy()
    df_next = df[df['frame'] == next_frame].copy()

    # Get the positions of tracks in the current and next frame
    current_positions = df_current[['x_head', 'y_head']].values
    next_positions = df_next[['x_head', 'y_head']].values

    # Step 4: Calculate pairwise distances between all positions in current and next frame
    dist_matrix = cdist(current_positions, next_positions)

    # Step 5: For each track in the current frame, find the closest track in the next frame
    best_matches = np.argmin(dist_matrix, axis=1)  # Best match based on min distance
    best_distances = np.min(dist_matrix, axis=1)  # Get the minimum distance for each track

    # Step 6: Assign the track_id from the current frame to the best match in the next frame
    for j, best_match in enumerate(best_matches):
        if best_distances[j] < distance_threshold:  # Accept match if distance is below the threshold
            df_next.loc[df_next.index[best_match], 'track_id'] = df_current.iloc[j]['track_id']
     
            # Handle low-probability matches, if needed (you can flag or log them)

    # Step 7: Update the dataframe with corrected track IDs for the next frame
    df.update(df_next)

# Step 8: Save the updated dataframe
df.to_csv('/Volumes/proj-data-challenge/2024/proj-Cochrane/videos-topdown/food/2024-07-26_14-08-00_td8.probability_sync.csv', index=False)
