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




import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import matplotlib as mpl
from scipy.stats import mannwhitneyu

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']

PALETTE = {
    "GH": 'steelblue',     
    "SI": 'darkorange',}

HUE_ORDER = ["GH", "SI"]


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/interaction_type_bout.csv')
df1['condition'] = 'GH'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/interaction_type_bout.csv')
df2['condition'] = 'SI'


df = pd.concat([df1, df2], ignore_index=True)


length_bouts = df.groupby(['condition', 'file', 'initial_type'])['duration'].mean().reset_index(name='length_bout')

sns.barplot(data=length_bouts, x='initial_type', y='length_bout', hue='condition', hue_order=HUE_ORDER, edgecolor='black', linewidth=2, errorbar='sd', palette=PALETTE, alpha=0.8)

plt.show()