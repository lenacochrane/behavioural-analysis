import os 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from scipy.spatial import KDTree # k-dimensional spac


df = pd.read_feather('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-proximal-interactions/test-interactions/2024-07-12_13-18-27_td1.tracks.feather')


pixel_columns = ['x_tail', 'y_tail', 'x_body', 'y_body', 'x_head', 'y_head']
df[pixel_columns] = df[pixel_columns] * (90/1032)
print(df.head())

df = df[df['frame'] < 600]

# Initialize KD-Trees for each part
parts = ['head', 'body', 'tail']
trees = {part: KDTree(df[[f'x_{part}', f'y_{part}']].values) for part in parts}

# Define the proximity radius (1mm here)
radius = 1

unique_interactions = {}

# Analyze interactions for each type of part combination
for part1 in parts:
    for part2 in parts:
        tree1 = trees[part1]
        tree2 = trees[part2]
        distances, indices = tree1.query(df[[f'x_{part2}', f'y_{part2}']].values, k=1, distance_upper_bound=radius)
        print(distances)

        for i, (dist, j) in enumerate(zip(distances, indices)):
            if j != df.shape[0] and i != j:  # Ensure valid index and not comparing the same entry
                if df.at[i, 'track_id'] != df.at[j, 'track_id']:  # Check for different track IDs to avoid self-interactions
                    sorted_ids = tuple(sorted((df.at[i, 'track_id'], df.at[j, 'track_id'])))
                    frame = df.at[i, 'frame']
                    key = (frame, sorted_ids)

                    if key not in unique_interactions or dist < unique_interactions[key][1]:
                        interaction_type = f"{part1}-{part2}"
                        unique_interactions[key] = (interaction_type, dist)

# Convert interactions to a DataFrame for easier handling
interaction_details = [{
    'Frame': frame,
    'Larva1': ids[0],
    'Larva2': ids[1],
    'Interaction Type': details[0],
    'Distance': details[1]
} for (frame, ids), details in unique_interactions.items()]

interaction_df = pd.DataFrame(interaction_details)
interaction_df = interaction_df.sort_values(by='Frame')
print(interaction_df)
print(interaction_df.head(20))

interaction_df.to_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-proximal-interactions/test-interactions/df.csv', index=False)


