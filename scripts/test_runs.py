import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pyarrow.feather as feather

df = pd.read_feather('/Volumes/lab-windingm/home/users/cochral/AttractionRig/modelling-behaviour/michael-test-sleap-extrac/ptc/2024-07-16_11-14-59_td10.tracks.feather')

print(df)
