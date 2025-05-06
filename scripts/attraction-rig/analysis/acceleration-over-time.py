
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n1/acceleration_accross_time.csv')
df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n2/acceleration_accross_time.csv')
df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n3/acceleration_accross_time.csv')
df4 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n4/acceleration_accross_time.csv')
df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n5/acceleration_accross_time.csv')
df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n6/acceleration_accross_time.csv')
df7 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n7/acceleration_accross_time.csv')
df8 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n8/acceleration_accross_time.csv')
df9 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n9/acceleration_accross_time.csv')
df10 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n10/acceleration_accross_time.csv')

plt.figure(figsize=(8,6))

# Manually assign DataFrames to specific subplots using seaborn.lineplot
sns.lineplot(data=df1, x='time', y='acceleration',  color='#4A6792', ci=None, label='X=1')

sns.lineplot(data=df2, x='time', y='acceleration', color='#ECB0C3', ci=None, label='X=2')

sns.lineplot(data=df3, x='time', y='acceleration',  color='#629677', ci=None, label='X=3')

sns.lineplot(data=df4, x='time', y='acceleration',  color='#F39B6D', ci=None, label='X=4')

sns.lineplot(data=df5, x='time', y='acceleration',  color='#85C7DE', ci=None, label='X=5')

sns.lineplot(data=df6, x='time', y='acceleration',  color='#7EBC66', ci=None, label='X=6')

sns.lineplot(data=df7, x='time', y='acceleration', color='#F2CD60', ci=None, label='X=7')

sns.lineplot(data=df8, x='time', y='acceleration', color='#7CB0B5', ci=None, label='X=8')

sns.lineplot(data=df9, x='time', y='acceleration',  color='#F08080', ci=None, label='X=9')

sns.lineplot(data=df10, x='time', y='acceleration',  color='#6F5E76', ci=None, label='X=10')


plt.xlabel('Time (S)', fontsize=12)
plt.ylabel('Acceleration (mm/s^2)', fontsize=12)

# plt.ylim(0, 90)

# Add an overall title to the entire figure
plt.title('Acceleration Over Time', fontsize=16, fontweight='bold')

# Adjust layout to prevent overlap, considering the overall title
plt.tight_layout(rect=[0, 0, 1, 0.95])


plt.savefig('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/plots/acceleration/acceleration-over-time-overlay.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
