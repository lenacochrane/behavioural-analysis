import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Add the directory containing hole_analysis.py to the Python path
sys.path.append('/Users/cochral/repos/behavioural-analysis/scripts/attraction-rig')

from hole_analysis import HoleAnalysis 



HoleAnalysis.probability_density('/Users/cochral/repos/behavioural-analysis/scripts/attraction-rig/distances_from_centre.csv')

ax.set_title('Custom Title for the Plot')

plt.show()

