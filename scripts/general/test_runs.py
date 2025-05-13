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
from shapely import wkt

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# # Load your data
# # Load data
# df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-digging-mask/N10/digging.csv')

# # Convert to int for plotting
# df['digging_status'] = df['digging_status'].astype(int)

# # Unique tracks
# track_ids = df['track_id'].unique()

# # Plot setup
# fig, axes = plt.subplots(len(track_ids), 1, figsize=(12, 3 * len(track_ids)), sharex=True)

# # Ensure axes is iterable
# if len(track_ids) == 1:
#     axes = [axes]

# # Plot for each track
# for i, track in enumerate(track_ids):
#     ax = axes[i]
#     track_df = df[df['track_id'] == track]
#     ax.plot(track_df['frame'], track_df['digging_status'], drawstyle='steps-post')
#     ax.set_title(f'Track {track} Digging Status')
#     ax.set_ylabel('Digging (0=No, 1=Yes)')
#     ax.set_ylim(-0.1, 1.1)

# # Final formatting
# axes[-1].set_xlabel('Frame')
# plt.tight_layout()
# plt.show()


df = pd.read_feather('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-digging-paramaters-gridsearch/1-grid/n2/2025-03-03_14-03-34_td4.tracks.feather')
print(df)