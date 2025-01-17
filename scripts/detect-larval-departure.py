import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pyarrow.feather as feather
import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.wkt import loads as load_wkt
from scipy.spatial.distance import euclidean



wkt_file = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-leaving-perimeter/2025-01-17-n10-agarose/2024-08-05_10-07-59_td4_perimeter.wkt'

with open(wkt_file, 'r') as f:
    wkt_content = f.read()  # Read the WKT file into the wkt_content variable
    petri_dish_boundary = load_wkt(wkt_content)  # Convert the WKT data into a Shapely polygon

buffered_inside_boundary = petri_dish_boundary.buffer(-1) 
buffered_outside_boundary = petri_dish_boundary.buffer(5) 
                            

feather_file = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-leaving-perimeter/2025-01-17-n10-agarose/2024-08-05_10-07-59_td4.tracks.feather'  # Replace with your actual Feather file path
df = feather.read_feather(feather_file)
# df_path = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-leaving-perimeter/feather.csv'
# df.to_csv(df_path, index=False)


# BUFFERS 
# df['distance_to_outer_boundary'] = df.apply(lambda row: buffered_outside_boundary.exterior.distance(Point(row['x_body'], row['y_body'])), axis=1)

# df['distance_to_inner_boundary'] = df.apply(lambda row: buffered_inside_boundary.exterior.distance(Point(row['x_body'], row['y_body'])),axis=1)

# df['distance_to_boundary'] = df[['distance_to_outer_boundary', 'distance_to_inner_boundary']].min(axis=1)

# df_close_to_boundary = df[df['distance_to_boundary'] <= 1]

# output_csv_path_1 = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-leaving-perimeter/close-perimeter.csv'
# df_close_to_boundary.to_csv(output_csv_path_1, index=False)

# BODY COORDINATES OUTSIDE PERIMETER
# df['outside_perimeter'] = df.apply(lambda row: not petri_dish_boundary.contains(Point(row['x_body'], row['y_body'])),axis=1)
# df_outside = df[df['outside_perimeter']]

# output_csv_path = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-leaving-perimeter/outside-perimeter.csv'
# df_outside.to_csv(output_csv_path, index=False)


# lets try and use body coordinates outside of perimeter- then detect if leaves or no 


# # BODY COORDINATES OUTSIDE PERIMETER
df['outside_perimeter'] = df.apply(lambda row: not petri_dish_boundary.contains(Point(row['x_body'], row['y_body'])),axis=1)

df['count'] = 10

# Function to process data and update counts
def update_larvae_count(df):
    # Iterate over each row that is marked as outside the perimeter
    for index, row in df[df['outside_perimeter']].iterrows():
        # Track subsequent 10 frames for this track
        end_frame = row['frame'] + 10
        subsequent_data = df[(df['track_id'] == row['track_id']) & (df['frame'] > row['frame']) & (df['frame'] <= end_frame)]

        if subsequent_data.empty:
            print(f"Larva with track ID {row['track_id']} left the perimeter at frame {row['frame']}.")
            df.loc[df['frame'] >= row['frame'], 'count'] -= 1
        # If there is subsequent data, assume the larva could potentially return
        else:
            # Optionally, you could handle cases where subsequent data exists
            continue  # This continues to the next larva without adjusting the count

update_larvae_count(df)


full_frame_range = range(0, 3601)  # From 0 to 3600

existing_frames = set(df['frame'].unique())

missing_frames = sorted(set(full_frame_range) - existing_frames)

missing_data = [{'frame': frame, 'count': 0} for frame in missing_frames]
df_missing = pd.DataFrame(missing_data)

# Append missing data to the original DataFrame
df = pd.concat([df, df_missing], ignore_index=True)

# Sort the DataFrame by frame to maintain chronological order
df.sort_values(by='frame', inplace=True)

# Optional: Reset index for cleanliness
df.reset_index(drop=True, inplace=True)


df_path = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-leaving-perimeter/2025-01-17-n10-agarose/track_count.csv'
df.to_csv(df_path, index=False)


sns.lineplot(data=df, x='frame', y='count')
# plt.xlim(0,600)

plt.show()

