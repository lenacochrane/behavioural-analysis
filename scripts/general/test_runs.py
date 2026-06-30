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



interaction_data = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/grouped+isolated/interactions_newtest.csv')    

total_interactions = interaction_data['Interaction Number'].nunique()

invalid_interactions = (
    interaction_data.groupby('Interaction Number')['Normalized Frame']
    .agg(lambda x: not set(range(-10, 11)).issubset(set(x)))
    .sum()
)

print(f"{invalid_interactions} / {total_interactions} interactions are missing full -10 to +10 normalized frames")