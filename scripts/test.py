

import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pyarrow.feather as feather

df = pd.read_feather('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n5/2024-07-12_15-38-19_td2.tracks.feather')

print(df.head())
print(df['track_id'].unique())