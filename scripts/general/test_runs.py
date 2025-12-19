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


feather_foler = '/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n2/group-housed'

files = [f for f in os.listdir(feather_foler) if f.endswith('tracks.feather')]

for file in files:
    df = feather.read_feather(os.path.join(feather_foler, file))

    print('Processing file:', file)
    print(df['track_id'].unique())

    track_gaps = {}

    for track_id in df['track_id'].unique():
        track_df = df[df['track_id'] == track_id].sort_values(by='frame')
        frames = track_df['frame'].tolist()

        missing_frames = [f for f in range(min(frames), max(frames) + 1) if f not in frames]
        
        if missing_frames:
            track_gaps[track_id] = missing_frames

    if track_gaps:
        print("Tracks with missing frames:")
        for track, gaps in track_gaps.items():
            print(f"Track {track} has missing frames: {gaps}")
    else:
        print("No missing frames detected in any track.")