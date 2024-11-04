import cv2
import numpy as np
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from collections import defaultdict


df = pd.read_feather('/Volumes/proj-data-challenge/2024/proj-Cochrane/videos-topdown/food/2024-07-26_14-08-00_td8.tracks.feather')

mm_to_pixel = ['x_body', 'y_body', 'x_head', 'y_head', 'x_tail', 'y_tail']
df[mm_to_pixel] = df[mm_to_pixel] * (1032 / 90)

###### FIRST IDENTIFY FRAMES WITH NEW TRACKS 

#track_first_last_df = df.groupby('track_id').agg(first_frame=('frame', 'min'), last_frame=('frame', 'max')).reset_index()

#track_first_last_df.to_csv('/Volumes/proj-data-challenge/2024/proj-Cochrane/videos-topdown/food/2024-07-26_14-08-00_td8.track_first_frame.csv', index=False)


####### IDENTIFY TRACK JUMPS (TRACK JUMPS OCCUR WHEN ID IS MAINTAINED WRONGLY- THIS MEANS WE CAN ALWAYS DETECT THEM UNLIKE SWAPS)

  # 1. CALCULATE CHANGE IN BODY XY COORDINATES

df['dx'] = df.groupby('track_id')['x_body'].diff().fillna(0) # change in x
df['dy'] = df.groupby('track_id')['y_body'].diff().fillna(0) # change in y
df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2) # distance travelled 

df['potential_swap'] = df.groupby('track_id')['distance'].transform(lambda x: x > 20) # jumped if greater than 25 pixel movement 
df['jumped'] = df.groupby('track_id')['distance'].transform(lambda x: x > 50) # jumped if greater than 25 pixel movement 

#df.to_csv('/Volumes/proj-data-challenge/2024/proj-Cochrane/videos-topdown/food/2024-07-26_14-08-00_td8.jumping_tracks.csv', index=False)


####### FUNCTIONS TO BE CALLED ON 

def merge_tracks(df, track_id_1, track_id_2, start_frame):
    # Create a temporary ID that doesn't conflict with any existing track_id
    temp_id = df['track_id'].max() + 1  # Choose a temporary ID that is one greater than the current max track_id
    
    # From the start_frame onwards, swap the track IDs
    df.loc[(df['track_id'] == track_id_1) & (df['frame'] >= start_frame), 'track_id'] = temp_id  # Temporary replace track_id_1 with temp_id
    df.loc[(df['track_id'] == track_id_2) & (df['frame'] >= start_frame), 'track_id'] = track_id_1  # Replace track_id_2 with track_id_1
    df.loc[(df['track_id'] == temp_id) & (df['frame'] >= start_frame), 'track_id'] = track_id_2  # Replace temp_id with track_id_2

    return df

###### LINKING NEW TRACKS TO OLD TRACKS

# Dictionary to store tracks' last known positions and frame number when they went missing
missing_tracks = defaultdict(dict)

# Parameters for distance and time thresholds
distance_threshold = 20  # Proximity threshold in pixels
time_threshold = 10  # Maximum time gap allowed between frames

# Iterate over each frame
for frame in df['frame'].unique():
    
    # Get all tracks in the current frame
    current_frame_tracks = df[df['frame'] == frame]
    
    # Get track IDs for this frame
    current_track_ids = current_frame_tracks['track_id'].unique()
    
    # Check if any track is missing in this frame
    for track_id in list(missing_tracks.keys()):
        if frame - missing_tracks[track_id]['last_frame'] > time_threshold:
            # Remove tracks that have been missing for too long
            del missing_tracks[track_id]

    # Store any new tracks and try to match them with previously missing tracks
    for track_id in current_track_ids:
        track_data = current_frame_tracks[current_frame_tracks['track_id'] == track_id]
        x_body = track_data['x_body'].values[0]
        y_body = track_data['y_body'].values[0]

        # Check if this is a new track
        if frame == df[df['track_id'] == track_id]['frame'].min():
            # This is a new track, check for matches with missing tracks
            for missing_track_id, info in missing_tracks.items():
                last_x, last_y = info['coords']
                last_frame = info['last_frame']
                avg_speed = info['avg_speed']
                
                # Calculate time gap and predicted movement distance
                time_gap = frame - last_frame
                max_distance = time_gap * avg_speed
                
                # Calculate the actual distance between the new track and the missing track
                distance = np.sqrt((x_body - last_x)**2 + (y_body - last_y)**2)

                # If the new track is within the expected distance and time, link the tracks
                if distance < max_distance and distance < distance_threshold:
                    print(f"New track {track_id} matches missing track {missing_track_id} with distance {distance:.2f}")
                    
                    # Merge the new track with the missing one
                    df = merge_tracks(df, missing_track_id, track_id, frame)
                    
                    # Remove the missing track from the dictionary since it's now linked
                    del missing_tracks[missing_track_id]
                    break
            else:
                # If no match is found, store this track as a new one
                print(f"Track {track_id} is a new unlinked track in frame {frame}")
                
        # Update the last known position for active tracks
        last_x_body = x_body
        last_y_body = y_body
        
        # Calculate average speed based on the previous frame's movement, if available
        track_movement = df[(df['track_id'] == track_id) & (df['frame'] < frame)][['dx', 'dy']].values
        avg_speed = np.mean(np.sqrt(track_movement[:, 0]**2 + track_movement[:, 1]**2)) if len(track_movement) > 0 else 0

        # Update the missing_tracks dictionary if this track is lost in future frames
        if len(df[(df['track_id'] == track_id) & (df['frame'] > frame)]) == 0:
            # This track is missing in future frames, add it to the missing tracks dictionary
            missing_tracks[track_id] = {'coords': (last_x_body, last_y_body), 
                                        'last_frame': frame, 
                                        'avg_speed': avg_speed}

# Save the corrected dataframe to CSV
df.to_csv('/Volumes/proj-data-challenge/2024/proj-Cochrane/videos-topdown/food/2024-07-26_14-08-00_td8.tracks_synced_chatgpt_attempt.csv', index=False)



























##### IGNORE THE REST OF THIS 





# ### CREATE A FUNCTION WHICH DETECTS TRACK SWAPS/JUMPS AND CORRECTS SUCH 

#  # two conditions must be met: 1) larvae are close to eachothers previous position 2) track ids have swapped  

# def correct_swaps(df, distance_threshold=20):

#     jump_frames = df[df['jumped'] == True]['frame'].unique()

#     for frame in jump_frames:
#         current_frame = df[df['frame'] == frame]
#         prev_frame = df[df['frame'] == frame - 1]

#         # Extract x and y coordinates for larvae in both frames # how does this know which track its referring to 
#         current_coords = current_frame[['x_body', 'y_body']].values
#         prev_coords = prev_frame[['x_body', 'y_body']].values

#         # Compute pairwise distances between current and previous positions using cdist
#         dist_matrix = cdist(current_coords, prev_coords)

# #         Larva A (Prev)  Larva B (Prev)  Larva C (Prev)
# # Larva X     7.07          148.66           290.69
# # Larva Y    148.99            10             150
# # Larva Z    296.49          150.99             22.36

#         # Use Hungarian algorithm to find optimal assignments (one-to-one matching)
#         current_track_indices, previous_track_indices = linear_sum_assignment(dist_matrix)

# # The Hungarian algorithm finds the best one-to-one matches (minimal distances):
# # Larva X -> Larva A  (Distance = 7.07)
# # Larva Y -> Larva B  (Distance = 10)
# # Larva Z -> Larva C  (Distance = 22.36)

#         # Now, current_track_indices[i] gives the index of the larva in the current frame,
#         # and previous_track_indices[i] gives the index of the corresponding best-matched larva in the previous frame.
#         for i in range(len(current_track_indices)):
#             curr_larva_index = current_track_indices[i]   # Index in the current frame
#             prev_larva_index = previous_track_indices[i]   # Best match index in the previous frame

#             # Get the distance between the current larva and the best matched larva from the previous frame
#             dist = dist_matrix[curr_larva_index, prev_larva_index]
            
#             # Retrieve track IDs for comparison
#             curr_track_id = current_frame.iloc[curr_larva_index]['track_id']
#             prev_track_id = prev_frame.iloc[prev_larva_index]['track_id']

#             # Case 1: Swap detected - the distance is above the threshold and the IDs are different
#             if dist > distance_threshold and curr_track_id != prev_track_id:
#                 print(f"{frame=}, {curr_track_id=}, {prev_track_id=}")
#                 df = swap_tracks(df, curr_track_id, prev_track_id, frame)

#     df.to_csv('/Volumes/proj-data-challenge/2024/proj-Cochrane/videos-topdown/food/2024-07-26_14-08-00_td8.jumping_tracks_corrected?.csv', index=False)
#     return df

# correct_swaps(df)


# IDENTIFY MISSING FRAMES FROM THE TRACK DATA - IF THEY ARE WITHIN A CERTAIN PROXIMITY INTERPOLATE






# 1. IDENTIFY NEW TRACK
# 2. WHICH PREDICTION IN PREVIOUS FRAME MATCHES MOST CORRECTLY (PREDICTION NOT TRACK)
# 3. COMBINE THE TRACK OF THIS PREDICTION WITH THE NEW TRACK IF POSSIBLE 
# 4. IF TRACK IS ALREADY TAKEN UP THEN SWAP IT AROUND 
# 5. MIGHT BE MULITPLE DIFF PREDICTIONS IN AREA

# FRAME 222 GOOD EXAMPLE BUT THIS DOESNT WORK ON WHEN LARVAE ARE ALL CLOSE TOGETHER AT SIDE OF DISH


# AROUND 390 IS WEIRD ONE



# ARE THERE TRACKS WHICH ENDED IN PREVIOUS FRAMES
# ARE THESE TRACKS IN PROXIMITY TO ONE ANOTHER 






# IDENTIFY JUMPS





