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

# Step 1: Load the Petri dish boundary from the WKT file
wkt_file = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/test-leaving-perimeter/petri_dish_boundary.wkt'
# Ensure the wkt_content is properly defined by reading the file
with open(wkt_file, 'r') as f:
    wkt_content = f.read()  # Read the WKT file into the wkt_content variable
    # print(f"WKT File Contents: {wkt_content}")  # This will print the WKT data so you can verify it

    # Now load the WKT as a Shapely polygon
    petri_dish_boundary = load_wkt(wkt_content)  # Convert the WKT data into a Shapely polygon

# Step 2: Define the scaling factor
scaling_factor = 90 / 1032

# Step 3: Scale the coordinates of the polygon
scaled_boundary = Polygon([(x * scaling_factor, y * scaling_factor) for x, y in petri_dish_boundary.exterior.coords])

# Step 4: Create a small buffer around the boundary (e.g., 0.5 units)
buffered_boundary = scaled_boundary.buffer(10)  # You can adjust the buffer size as needed
                            
# Step 2: Load the Feather file containing the larvae tracking data
feather_file = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/test-leaving-perimeter/2024-07-30_10-58-18_td4.tracks.feather'  # Replace with your actual Feather file path
df = feather.read_feather(feather_file)

# Step 3: Iterate over the larvae coordinates in the Feather file
# Assuming your Feather file has columns: 'frame', 'track_id', 'x_body', 'y_body'
# Check if the larvae coordinates only touch the scaled boundary
df['touching_boundary'] = df.apply(lambda row: buffered_boundary.touches(Point(row['x_body'], row['y_body'])), axis=1)

# Step 3: Iterate over the larvae coordinates and check distance to boundary
df['distance_to_boundary'] = df.apply(lambda row: buffered_boundary.exterior.distance(Point(row['x_body'], row['y_body'])), axis=1)

# Check the minimum distance to the boundary
print(f"Minimum distance to boundary: {df['distance_to_boundary'].min()}")

# Filter those within a reasonable distance (e.g., within 10 units)
df_close_to_boundary = df[df['distance_to_boundary'] <= 10]

print(f"Number of larvae close to the boundary: {len(df_close_to_boundary)}")
# Step 4: Filter rows where larvae coordinates are inside or touching the boundary
larvae_touching_boundary = df[df['touching_boundary']]

# Step 5: Save the result to a new CSV or Feather file
output_csv_path = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/test-leaving-perimeter/touching-perimeter.csv'
larvae_touching_boundary.to_csv(output_csv_path, index=False)


# BUFFER SHD BENEGATIVE?