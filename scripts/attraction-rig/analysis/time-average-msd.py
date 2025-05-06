
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour/n1/time_average_msd.csv')
df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour/n2/time_average_msd.csv')
df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour/n3/time_average_msd.csv')
df4 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour/n4/time_average_msd.csv')
df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour/n5/time_average_msd.csv')
df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour/n6/time_average_msd.csv')
df7 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour/n7/time_average_msd.csv')
df8 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour/n8/time_average_msd.csv')
df9 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour/n9/time_average_msd.csv')
df10 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour/n10/time_average_msd.csv')

# plt.figure(figsize=(8,6))
plt.figure(figsize=(8,8))

# Manually assign DataFrames to specific subplots using seaborn.lineplot
sns.lineplot(data=df1, x='tau', y='msd',  color='#421B53', ci=None, label='X=1')

sns.lineplot(data=df2, x='tau', y='msd', color='#4A2C77', ci=None, label='X=2')

sns.lineplot(data=df3, x='tau', y='msd', color='#3F4A8B', ci=None, label='X=3')

sns.lineplot(data=df4, x='tau', y='msd', color='#306990', ci=None, label='X=4')

sns.lineplot(data=df5, x='tau', y='msd',  color='#25838F', ci=None, label='X=5')

sns.lineplot(data=df6, x='tau', y='msd',  color='#149E89', ci=None, label='X=6')

sns.lineplot(data=df7, x='tau', y='msd', color='#35B579', ci=None, label='X=7')

sns.lineplot(data=df8, x='tau', y='msd', color='#75C15A', ci=None, label='X=8')

sns.lineplot(data=df9, x='tau', y='msd',  color='#B4D336', ci=None, label='X=9')

sns.lineplot(data=df10, x='tau', y='msd',  color='#FAE821', ci=None, label='X=10')


plt.xlabel('Tau', fontsize=12,fontweight='bold')
plt.ylabel('MSD', fontsize=12,fontweight='bold')

# plt.ylim(0, 90)
plt.yscale('log')
plt.xscale('log')

# Add an overall title to the entire figure
plt.title('Time Average Mean Squared Distance', fontsize=16, fontweight='bold')

plt.legend(title='Number of Larvae')

# Adjust layout to prevent overlap, considering the overall title
plt.tight_layout(rect=[0, 0, 1, 0.95])


plt.savefig('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour/plots/time-average-msd/time-average-msd-overla-log.png', dpi=300, bbox_inches='tight')


# Show the plot
plt.show()
