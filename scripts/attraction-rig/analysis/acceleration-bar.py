
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



# Load your data
df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n1/acceleration_accross_time.csv')
df1['condition'] = 'X=1'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n2/acceleration_accross_time.csv')
df2['condition'] = 'X=2'

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n3/acceleration_accross_time.csv')
df3['condition'] = 'X=3'

df4 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n4/acceleration_accross_time.csv')
df4['condition'] = 'X=4'

df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n5/acceleration_accross_time.csv')
df5['condition'] = 'X=5'

df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n6/acceleration_accross_time.csv')
df6['condition'] = 'X=6'

df7 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n7/acceleration_accross_time.csv')
df7['condition'] = 'X=7'

df8 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n8/acceleration_accross_time.csv')
df8['condition'] = 'X=8'

df9 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n9/acceleration_accross_time.csv')
df9['condition'] = 'X=9'

df10 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/n10/acceleration_accross_time.csv')
df10['condition'] = 'X=10'

plt.figure(figsize=(8,6))

custom_palette = {
    'X=1': '#4A6792',  
    'X=2': '#ECB0C3',
    'X=3': '#629677',
    'X=4': '#F39B6D',
    'X=5': '#85C7DE',
    'X=6': '#7EBC66',
    'X=7': '#F2CD60',
    'X=8': '#7CB0B5',
    'X=9': '#F08080',
    'X=10': '#6F5E76'
}

df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10], ignore_index=True)

sns.barplot(data=df, x='condition', y='acceleration',  palette=custom_palette, alpha=0.8, edgecolor='black', linewidth=2)

plt.xlabel('Condition', fontsize=12)
plt.ylabel('Acceleration (mm^2/s)', fontsize=12)

plt.ylim(-0.042, 0.05)
# Add an overall title to the entire figure
plt.title('Acceleration Distribution', fontsize=16, fontweight='bold')

# Adjust layout to prevent overlap, considering the overall title
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour-day4/plots/acceleration/acceleration-barplot.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
