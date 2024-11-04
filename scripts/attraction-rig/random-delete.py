import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pyarrow.feather as feather


# # df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/test-number-digging/ground-truth/food-n10/number_digging.csv')

# ground = pd.read_excel('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/test-number-digging/ground-truth/food-n10/ground-truth.xlsx')

# df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/test-number-digging/ground-truth/food-n10/number_digging_smooth_100.csv')


# sns.lineplot(data=ground, x='frame', y='number_digging', label='ground truth', linewidth=5)

# sns.lineplot(data=df1, x='frame', y='number digging', label='RW 100', alpha=0.3)


# plt.title('Method Testing', fontweight='bold')

# plt.show()


df = pd.read_feather('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/test-number-digging/ground-truth/food-n10/2024-08-02_11-23-24_td7.tracks.feather')
moving_counts = df.groupby('frame')['track_id'].count().reset_index()

# Plotting
sns.lineplot(data=moving_counts, x='frame', y='track_id')
plt.show()

