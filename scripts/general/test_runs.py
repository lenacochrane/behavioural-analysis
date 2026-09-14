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

import pandas as pd



df = pd.read_feather('/Volumes/lab-windingm-1/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/sensory/23129-pseudo/pseudo-23129_5.tracks.feather')

# sns.scatterplot(data=df, x='x_body', y='y_body')
# plt.show()

print(df['track_id'].unique())