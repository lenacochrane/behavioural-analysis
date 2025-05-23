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



# video_folder = '/Users/cochral/Desktop/MOSEQ/si'

# # Prefix to add
# prefix = 'N1-SI_'

# # Loop through files in the folder
# for filename in os.listdir(video_folder):
#     if filename.endswith('.mp4') and not filename.startswith(prefix):
#         old_path = os.path.join(video_folder, filename)
#         new_filename = prefix + filename
#         new_path = os.path.join(video_folder, new_filename)
#         os.rename(old_path, new_path)
#         print(f"Renamed: {filename} ➝ {new_filename}")


df = pd.read_csv('/Users/cochral/Desktop/MOSEQ/KEYPOINT/2025_05_21-13_29_37/moseq_df.csv')

# sns.barplot(data=df, x='syllable', y='frequency', hue='group')

# plt.show()

print(df.columns)

print(df.head(20))