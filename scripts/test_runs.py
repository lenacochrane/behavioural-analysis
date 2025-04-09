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

directory = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n1/group-housed'

for file in os.listdir(directory):
    if file.endswith('tracks.feather'):

        df = pd.read_feather(os.path.join(directory, file))

        print(f"number of tracks = {df['track_id'].nunique()}: File {file}: Frames = {df['frame'].max()}")

    
