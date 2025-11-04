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


df = pd.read_csv('/Users/cochral/Desktop/disease_testing_2.csv')

df = df[["Disease", "disgenet_associations"] + [c for c in df.columns if c not in ["Disease", "disgenet_associations"]]]

df.to_csv('/Users/cochral/Desktop/disease_testing_2.csv', index=False)
