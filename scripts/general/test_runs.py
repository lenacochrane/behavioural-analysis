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
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# Custom function that returns total number of frames
def count_frames_opencv(video_path):
    # Capturing the input video
    video = cv2.VideoCapture(video_path)

    # Accessing the CAP_PROP_FRAME_COUNT property
    # To get the total frames
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return total_frames

# Example usage


# Input video
video= '/Volumes/lab-windingm/home/users/cochral/LRS/bertram/elegans1.mp4'

# Calling the custom function
frame_count = count_frames_opencv(video)
print(f"Total frames: {frame_count}")