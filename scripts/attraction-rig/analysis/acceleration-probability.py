
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# Add the directory containing 'hole_analysis' to sys.path
module_path = '/Users/cochral/repos/behavioural-analysis/scripts/attraction-rig'
if module_path not in sys.path:
    sys.path.append(module_path)

# Now you can import the module
from hole_analysis import HoleAnalysis



# Load your data
df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n1/acceleration.csv')
df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n2/acceleration.csv')
df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n3/acceleration.csv')
df4 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n4/acceleration.csv')
df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n5/acceleration.csv')
df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n6/acceleration.csv')
df7 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n7/acceleration.csv')
df8 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n8/acceleration.csv')
df9 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n9/acceleration.csv')
df10 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n10/acceleration.csv')

plt.figure(figsize=(8,6))

# Manually assign DataFrames to specific subplots using seaborn.lineplot
HoleAnalysis.probability_density(df1, color='#4A6792', label='X=1')
HoleAnalysis.probability_density(df2,  color='#ECB0C3', label='X=2')
HoleAnalysis.probability_density(df3,   color='#629677', label='X=3')
HoleAnalysis.probability_density(df4,  color='#F39B6D', label='X=4')
HoleAnalysis.probability_density(df5,  color='#85C7DE', label='X=5')
HoleAnalysis.probability_density(df6,  color='#7EBC66', label='X=6')
HoleAnalysis.probability_density(df7,   color='#F2CD60', label='X=7')
HoleAnalysis.probability_density(df8,   color='#7CB0B5', label='X=8')
HoleAnalysis.probability_density(df9,  color='#F08080', label='X=9')
HoleAnalysis.probability_density(df10,  color='#6F5E76', label='X=10')

plt.xlabel('Acceleration', fontsize=12)
plt.ylabel('Probability', fontsize=12)

plt.xlim(-5, 5)


# Add an overall title to the entire figure
plt.title('Acceleration Probability Distribution', fontsize=16, fontweight='bold')

# Adjust layout to prevent overlap, considering the overall title
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/plots/acceleration/acceleration-probability-overlay.png', dpi=300, bbox_inches='tight')


# Show the plot
plt.show()
