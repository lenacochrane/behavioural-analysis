
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches




df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/group-housed/interactions.csv')
df3['condition'] = 'GH_N2'

df4 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/socially-isolated/interactions.csv')
df4['condition'] = 'SI_N2'

df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed/interactions.csv')
df5['condition'] = 'GH_N10'

df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/socially-isolated/interactions.csv')
df6['condition'] = 'SI_N10'






plt.figure(figsize=(8,8))

# df = pd.concat([df5, df6], ignore_index=True) #n10

df = pd.concat([df3, df4], ignore_index=True) # n2

# df['avg'] = df[['track1_speed', 'track2_speed']].mean(axis=1) #speed
# df['avg'] = df[['track1_acceleration', 'track2_acceleration']].mean(axis=1) #acceleration
df['avg'] = df[['track1_angle', 'track2_angle']].mean(axis=1) # heading angle
# df['avg'] = df[['track1_approach_angle', 'track2_approach_angle']].mean(axis=1) # approach angle



# valid_interactions = df[(df['Normalized Frame'] == 0) & (df['min_distance'] < 1)][['condition', 'file', 'Interaction Number']]
valid_interactions = df[
    (df['Normalized Frame'] == 0) & 
    (df['min_distance'] >= 0) & 
    (df['min_distance'] <= 0.1)
][['condition', 'file', 'Interaction Number']]

df_filtered = df.merge(valid_interactions, on=['condition', 'file', 'Interaction Number'])



grouped = (
    df_filtered.groupby(['condition', 'file', 'Normalized Frame'])['avg']
    .mean()
    .reset_index()
)


total_interactions = df[['condition',  'Interaction Number']].drop_duplicates().shape[0]
filtered_interactions = valid_interactions[['condition', 'Interaction Number']].drop_duplicates().shape[0]

print("Total unique interactions:", total_interactions)
print("Filtered unique interactions:", filtered_interactions)



sns.lineplot(
    data=grouped,
    x='Normalized Frame',
    y='avg',
    hue='condition',
    errorbar='sd'  # seaborn now aggregates across files per condition
)

plt.xlabel('Normalized Frame', fontsize=12)
plt.ylabel('heading angle', fontsize=12)

plt.xlim(-30, 30)


plt.title('', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

# plt.xlim(0,15)
# plt.xscale('log')
# plt.ylim(0, None)
# plt.ylim(-0.25, 0.25)
# plt.ylim(140, 180)


plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/interactions/0-0.1mm_n2-headingangle.png', dpi=300, bbox_inches='tight')


plt.show()




# import seaborn as sns
# import matplotlib.pyplot as plt

# df = pd.concat([df3, df4], ignore_index=True) # n2
# # Filter to only frame 0 rows
# df_0 = df[df['Normalized Frame'] == 0]

# # Set figure size and style
# plt.figure(figsize=(8, 6))

# # Plot histogram of min_distance split by condition
# sns.histplot(
#     data=df_0,
#     x='min_distance',
#     hue='condition',
#     bins=50,
#     element='step',
#     stat='density',  # use 'count' if you prefer raw counts
#     common_norm=False
# )

# # Labeling
# plt.title('Distribution of min_distance at Normalized Frame 0', fontsize=14, fontweight='bold')
# plt.xlabel('min_distance', fontsize=12)
# plt.ylabel('Density', fontsize=12)
# # plt.legend(title='Condition')
# plt.tight_layout()

# # Show plot
# plt.show()
