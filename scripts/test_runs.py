import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pyarrow.feather as feather

# df = pd.read_feather('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/plain-petri/2024-08-20_13-17-04_td2.tracks.feather')


# this has been cleaned already 
df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/test-hole-counter/hole_count.csv')


sns.lineplot(data=df, x='time', y='count')

plt.show()




# track_23_df = df[df['track_id'] == 23]

# track_23_df['dx'] = track_23_df['x_body'].diff()
# track_23_df['dy'] = track_23_df['y_body'].diff()

# # Calculate frame-to-frame distance (displacement)
# track_23_df['distance'] = np.sqrt(track_23_df['dx']**2 + track_23_df['dy']**2)

# # Calculate cumulative distance
# track_23_df['cumulative_distance'] = track_23_df['distance'].cumsum()

# # Plot cumulative distance over time
# sns.lineplot(data=track_23_df, x='frame', y='cumulative_distance')
# plt.title('Cumulative Distance over Time')
# plt.ylabel('Cumulative Distance')
# plt.show()

# sns.lineplot(data=track_23_df, x='frame', y='y_body')

# plt.show()
# track_23_df = track_23_df.sort_values(by='frame')

# # Calculate the differences between consecutive frames
# track_23_df['dx'] = track_23_df['x_body'].diff()
# track_23_df['dy'] = track_23_df['y_body'].diff()
# track_23_df['dt'] = track_23_df['frame'].diff()

# # Calculate the distance (speed = distance / time)
# track_23_df['distance'] = np.sqrt(track_23_df['dx']**2 + track_23_df['dy']**2)
# track_23_df['speed'] = track_23_df['distance'] / track_23_df['dt']

# # Plot the speed over time (frame)
# sns.lineplot(data=track_23_df, x='frame', y='speed')
# plt.ylabel('Speed')
# plt.ylim(0,0.3)
# plt.show()



# # Plot the distribution of 'instance_score' for track_id = 0

# sns.histplot(track_0_df['instance_score'], kde=True, bins=20, color='#2b65bb')

# # Add labels and title
# plt.xlabel('Instance Score')
# plt.ylabel('Frequency')
# plt.title('Distribution of Instance Score for Larvae which Track Swapped')

# # Show the plot
# plt.show()



# THIS WAS HOLE DIGGING- USE THIS TO DETECT FIRST TIME LARVAE DIGS OR SOMETHING ALSO 
# df = pd.read_csv('/Users/cochral/Desktop/SLAEP/Topdown/videos/HOLES/identify-holes/n9-fillingingaps.csv')

# frame_counts = {}

# for track_id, group in df.groupby('track'):

#     # Sort by frame index to ensure the order is correct
#     group = group.sort_values(by='frame_idx')
    
#     # diff() calculates difference between each row for the coordinate
#     # the first row is a nan because no difference yet - this is changed to a zero
#     group['dx'] = group['body.x'].diff().fillna(0)
#     group['dy'] = group['body.y'].diff().fillna(0)

#     frames = []

#     i = 0
#     while i < len(group):
#         row = group.iloc[i]

#         # Check the initial 30 frames to confirm digging
#         if i + 30 <= len(group):
#             next_30_rows = group.iloc[i:i+30]
#             frame_average_dx = next_30_rows['dx'].abs().mean()
#             frame_average_dy = next_30_rows['dy'].abs().mean()

#             if frame_average_dx <= 2 and frame_average_dy <= 2:
#                 # If initial 30 frames confirm digging, start marking
#                 digging = True

#                 for frame in next_30_rows['frame_idx']:
#                     if frame in frame_counts:
#                         frame_counts[frame] += 1
#                     else:
#                         frame_counts[frame] = 1

#                 i += 30        

#                 while digging and i < len(group):
#                     row = group.iloc[i]

#                     if abs(row['dx']) <= 2 and abs(row['dy']) <= 2:
#                         frame = row['frame_idx']
#                         if frame in frame_counts:
#                             frame_counts[frame] += 1
#                         else:
#                             frame_counts[frame] = 1
#                         i += 1  # Continue to the next frame
#                     else:
#                         digging = False 
                
#                 continue 
        
#         i += 1


# frame_counts_df = pd.DataFrame(list(frame_counts.items()), columns=['frame', 'number digging'])


# full_frame_range = pd.DataFrame({'frame': range(df['frame_idx'].min(), df['frame_idx'].max() + 1)})


# digging_df = full_frame_range.merge(frame_counts_df, on='frame', how='left').fillna(0)


# digging_df['number digging'] = digging_df['number digging'].astype(int)


# digging_df.to_csv('/Users/cochral/Desktop/SLAEP/Topdown/videos/HOLES/identify-holes/potentialdigs.csv')



# df2 = pd.read_csv("/Users/cochral/Desktop/SLAEP/Topdown/videos/HOLES/identify-holes/potentialdigs.csv")

# sns.lineplot(data=df2, x='frame', y='number digging')

# plt.show()

# # DETECT MOVING LARVAE - THIS WORKS JUST WANT TO MAKE IT MORE EFFICIENT BELOW

# df = pd.read_csv('/Users/cochral/Desktop/SLAEP/Topdown/videos/HOLES/identify-holes/n3-identifyhole.analysis.csv')


# # Initialize a dictionary to track the counts of each frame where larvae are moving
# frame_counts = {}

# for track_id, group in df.groupby('track'):

#     # Sort by frame index to ensure the order is correct
#     group = group.sort_values(by='frame_idx')
    
#     # Calculate the difference between consecutive rows for coordinates
#     group['dx'] = group['body.x'].diff().fillna(0)
#     group['dy'] = group['body.y'].diff().fillna(0)

#     i = 0
#     while i < len(group):
#         row = group.iloc[i]

#         # Check if the movement in both x and y directions is less than 2 pixels
#         if abs(row['dx']) <= 2 and abs(row['dy']) <= 2:
            
#             # If the movement is below 2 pixels, check the next 30 frames
#             if i + 30 <= len(group):
#                 next_30_rows = group.iloc[i:i+30]
#                 frame_average_dx = next_30_rows['dx'].abs().mean()
#                 frame_average_dy = next_30_rows['dy'].abs().mean()

#                 if frame_average_dx > 2 or frame_average_dy > 2:
#                     # If the average movement over the next 30 frames is acceptable, count the current frame
#                     frame = row['frame_idx']
#                     if frame in frame_counts:
#                         frame_counts[frame] += 1
#                     else:
#                         frame_counts[frame] = 1
#                 else:
#                       # if the average movement over the 30 frames was low it may indidcate that the larvae was digging:
#                       # have to decipher whether this is ongoing 

#                       # want to identify next row which exceeds 2 pixels and then take an average from there - if its ok continue from this frame 

#                       slow_movement = True  # currently larvae f0r 30 ish frames is slowly moving 
#                       while slow_movement and i < len(group):
#                           i += 1
#                           if i >= len(group):
#                             break
                          
#                           row = group.iloc[i]
#                           if abs(row['dx']) > 2 or abs(row['dy']) > 2:
#                             slow_movement = False
                        
#                       if not slow_movement and i + 30 <= len(group):
#                           next_30_rows = group.iloc[i:i+30]
#                           frame_average_dx = next_30_rows['dx'].abs().mean()
#                           frame_average_dy = next_30_rows['dy'].abs().mean()

#                           if frame_average_dx > 2 or frame_average_dy > 2:
#                             # If the average movement over the next 30 frames is acceptable, count this frame
#                             frame = row['frame_idx']
#                             if frame in frame_counts:
#                                 frame_counts[frame] += 1
#                             else:
#                                 frame_counts[frame] = 1        
                
#         else:
#             # If the movement is already above 2 pixels, count the current frame
#             frame = row['frame_idx']
#             if frame in frame_counts:
#                 frame_counts[frame] += 1
#             else:
#                 frame_counts[frame] = 1

#         i += 1  # Continue to the next frame



# frame_counts_df = pd.DataFrame(list(frame_counts.items()), columns=['frame', 'number moving'])


# full_frame_range = pd.DataFrame({'frame': range(df['frame_idx'].min(), df['frame_idx'].max() + 1)})


# moving_df = full_frame_range.merge(frame_counts_df, on='frame', how='left').fillna(0)


# moving_df['number moving'] = moving_df['number moving'].astype(int)


# moving_df.to_csv('/Users/cochral/Desktop/SLAEP/Topdown/videos/HOLES/identify-holes/movingn3.csv')

# df2 = pd.read_csv('/Users/cochral/Desktop/SLAEP/Topdown/videos/HOLES/identify-holes/movingn3.csv')

# sns.lineplot(data=df2, x='frame', y='number moving')

# plt.show()

# # # ATTEMPED TO DO VECTORISED PER GROUPED TRACK BUT ACC PROBS EASIER JUST DO IT FOR THE DATAFRAME 

# # df = pd.read_csv('/Users/cochral/Desktop/SLAEP/Topdown/videos/HOLES/identify-holes/n3-identifyhole.analysis.csv')

# # results = []

# # for track_id, group in df.groupby('track'):

# #     group = group.sort_values(by='frame_idx')
    
# #     group['dx'] = group.groupby('track')['body.x'].diff().fillna(0)
# #     group['dy'] = group.groupby('track')['body.y'].diff().fillna(0)

# #     # new column with True if larvae is moving 
# #     group['is_moving'] = (group['dx'].abs() > 2) | (group['dy'].abs() > 2)

# #     group['future_movement'] = group['is_moving'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())

# #     df['final_movement'] = df['is_moving'] | (df['future_movement'] > 0.5)

# #     results.append(group[['frame_idx', 'track', 'final_movement']])





# df = pd.read_csv('/Users/cochral/Desktop/SLAEP/Topdown/videos/HOLES/identify-holes/n9-fillingingaps.csv')
# # mean_scores = df.groupby('track')['instance.score'].mean()
# # print(mean_scores)

# df = df[df.groupby('track')['instance.score'].transform('mean') >= 0.9]
# # print(df['track'].unique())

# # Sort by 'track' and 'frame_idx' to ensure correct order for rolling operations
# df = df.sort_values(by=['track', 'frame_idx'])

# # Calculate the difference between consecutive rows for coordinates
# df['dx'] = df.groupby('track')['body.x'].diff().fillna(0)
# df['dy'] = df.groupby('track')['body.y'].diff().fillna(0)

# # Calculate the Euclidean distance (hypotenuse) between consecutive points
# df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2)

# # Create a boolean mask where movement is greater than 1 pixels
# # Identify segments of low movement and high movement
# df['is_moving'] = (df['dx'].abs() > 0.1) | (df['dy'].abs() > 0.1)

# # Use a rolling window to check for sustained movement over the next 30 frames
# df['future_movement'] = df.groupby('track')['is_moving'].transform(lambda x: x.rolling(window=100, min_periods=1).mean())

# # Use a rolling window to check if the cumulative distance moved in the last 5 frames exceeds a threshold (e.g., 10 pixels)
# # ONLY CONSIDER IT MOVING IF IT HAS MOVED MORE THAN > PIXELS IN THE WINDOW OF A CERTAIN AMOUNT OF FRAMES  
# df['distance_rolled'] = df.groupby('track')['distance'].transform(lambda x: x.rolling(window=30, min_periods=1).sum())
# df['distance_check'] = df['distance_rolled'] > 40

# # If future_movement is high enough, we can classify as "moving"
# df['final_movement'] = (df['is_moving'] | (df['future_movement'] > 0.5)) & df['distance_check']

# # Now count the moving frames per frame_idx
# moving_counts = df.groupby('frame_idx')['final_movement'].sum().reset_index()

# # Rename the column for clarity
# moving_counts.columns = ['frame_idx', 'moving_count']

# # Ensure we have a count for every frame
# full_frame_range = pd.DataFrame({'frame_idx': range(df['frame_idx'].min(), df['frame_idx'].max() + 1)})
# full_frame_counts = full_frame_range.merge(moving_counts, on='frame_idx', how='left').fillna(0)

# # Convert counts to integers
# full_frame_counts['moving_count'] = full_frame_counts['moving_count'].astype(int)

# full_frame_counts.to_csv('/Users/cochral/Desktop/SLAEP/Topdown/videos/HOLES/identify-holes/n10-numbermoving.csv')


# sns.lineplot(data=full_frame_counts, x='frame_idx', y='moving_count')

# plt.show()
























