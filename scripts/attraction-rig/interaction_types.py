import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from itertools import combinations
import cv2

### use that joblib within this script to do this !

## section idetnfiy all interactions present - use joblib here

#rest dont think i need joblib

directory_path = '/path/to/your/directory'

files = [f for f in os.listdir(directory_path) if f.endswith('.tracks.feather')]

for file in files:

    df = pd.read_feather(file)





