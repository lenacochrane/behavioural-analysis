import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches


###### INTERACTION TYPE - ALL NODE-NODE PER FRAME CONTACTS ######

df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/fed-fed/interaction_types.csv')
df1['condition'] = 'fed-fed'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-fed/interaction_types.csv')
df2['condition'] = 'fed-starved'

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-starved/interaction_types.csv')
df3['condition'] = 'starved-starved'


plt.figure(figsize=(8,8))


df = pd.concat([df1, df2, df3], ignore_index=True)

# Sum across all frame bins per file + interaction type
grouped = (
    df.groupby(['file', 'condition', 'interaction_type'])['count']
    .sum()
    .reset_index()
)

sns.barplot(data=grouped, x='interaction_type', y='count', hue='condition', edgecolor='black', linewidth=2, errorbar='sd', alpha=0.8)

plt.xlabel('Interaction Type', fontsize=12, fontweight='bold')
plt.ylabel('Interaction Count', fontsize=12, fontweight='bold')


# Add an overall title to the entire figure
plt.title('Interaction Type', fontsize=16, fontweight='bold')

# Adjust layout to prevent overlap, considering the overall title
# plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.tight_layout(rect=[1, 1, 1, 1])

plt.ylim(0, None)

plt.xticks(rotation=45)


plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/interaction_type_all_nodes.png', dpi=300, bbox_inches='tight')
plt.close()



###### INTERACTION TYPE - CLOSEST NODE-NODE PER FRAME ######


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/fed-fed/closest_contacts_1mm.csv')
df1['condition'] = 'fed-fed'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-fed/closest_contacts_1mm.csv')
df2['condition'] = 'fed-starved'

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-starved/closest_contacts_1mm.csv')
df3['condition'] = 'starved-starved'


plt.figure(figsize=(8,8))


df = pd.concat([df1, df2, df3], ignore_index=True)

# Sum across all frame bins per file + interaction type
grouped = (
    df.groupby(['file', 'condition', 'Closest Interaction Type'])
    .size()
    .reset_index(name='count')
)

sns.barplot(data=grouped, x='Closest Interaction Type', y='count', hue='condition', edgecolor='black', linewidth=2, errorbar='sd', alpha=0.8)

plt.xlabel('Closest Interaction Type', fontsize=12, fontweight='bold')
plt.ylabel('Interaction Count', fontsize=12, fontweight='bold')


# Add an overall title to the entire figure
plt.title('Interaction Type', fontsize=16, fontweight='bold')

# Adjust layout to prevent overlap, considering the overall title
# plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.tight_layout(rect=[1, 1, 1, 1])

plt.ylim(0, None)

plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/interaction_type_closest_nodes.png', dpi=300, bbox_inches='tight')

plt.close()



###### INTERACTION TYPE - CLOSEST NODE-NODE PER FRAME - DAY OF RECORDING ######


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/fed-fed/closest_contacts_1mm.csv')
df1['condition'] = 'fed-fed'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-fed/closest_contacts_1mm.csv')
df2['condition'] = 'fed-starved'

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-starved/closest_contacts_1mm.csv')
df3['condition'] = 'starved-starved'


plt.figure(figsize=(8,8))


df = pd.concat([df1, df2, df3], ignore_index=True)
df['day'] = df['file'].str.split('_').str[0]


# Sum across all frame bins per file + interaction type
grouped = (
    df.groupby(['file', 'condition', 'Closest Interaction Type', 'day'])
    .size()
    .reset_index(name='count')
)

sns.stripplot(
    data=grouped,
    x='Closest Interaction Type',
    y='count',
    hue='day',            # <-- color by collection day
    dodge=True,           # separate days side-by-side per type
    jitter=0.25,
    alpha=0.8,
    size=5
)


plt.xlabel('Closest Interaction Type', fontsize=12, fontweight='bold')
plt.ylabel('Interaction Count', fontsize=12, fontweight='bold')


# Add an overall title to the entire figure
plt.title('Interaction Type', fontsize=16, fontweight='bold')

# Adjust layout to prevent overlap, considering the overall title
# plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.tight_layout(rect=[1, 1, 1, 1])

plt.ylim(0, None)

plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/interaction_type_closest_nodes_by_day.png', dpi=300, bbox_inches='tight')

plt.close()


###### INTERACTION TYPE - CLOSEST NODE-NODE PER FRAME - FIRST CONTACT ######


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/fed-fed/closest_contacts_1mm.csv')
df1['condition'] = 'fed-fed'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-fed/closest_contacts_1mm.csv')
df2['condition'] = 'fed-starved'

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-starved/closest_contacts_1mm.csv')
df3['condition'] = 'starved-starved'


plt.figure(figsize=(8,8))


df = pd.concat([df1, df2, df3], ignore_index=True)
df['day'] = df['file'].str.split('_').str[0]

first_contact = (
    df.groupby(['file', 'condition', 'day'])['frame']
      .min()
      .reset_index(name='first_contact_frame')
)

sns.stripplot(
    data=first_contact,
    x='condition',
    y='first_contact_frame',
    hue='day',            # <-- color by collection day
    dodge=True,
    jitter=0.25,
    alpha=0.8,
    size=5
)


plt.xlabel('Condition', fontsize=12, fontweight='bold')
plt.ylabel('First Contact Frame', fontsize=12, fontweight='bold')


# Add an overall title to the entire figure
plt.title('First Contact', fontsize=16, fontweight='bold')

# Adjust layout to prevent overlap, considering the overall title
# plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.tight_layout(rect=[1, 1, 1, 1])

plt.ylim(0, None)

plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/interaction_type_closest_nodes_first_contact.png', dpi=300, bbox_inches='tight')

plt.close()







###### CONTACT "BOUTS" < 1MM CONTINIOUS FRAMES ######


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/fed-fed/contacts_1mm.csv')
df1['condition'] = 'fed-fed'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-fed/contacts_1mm.csv')
df2['condition'] = 'fed-starved'

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-starved/contacts_1mm.csv')
df3['condition'] = 'starved-starved'


plt.figure(figsize=(8,8))


df = pd.concat([df1, df2, df3], ignore_index=True)

durations = (
    df.groupby(['file', 'condition', 'Interaction Number'])
      .size()
      .reset_index(name='duration')
)

sns.barplot(
    data=durations,
    x='condition',
    y='duration',
    edgecolor='black',
    linewidth=2,
    errorbar='sd',
    alpha=0.8
)

plt.xlabel('Interaction Number', fontsize=12, fontweight='bold')
plt.ylabel('Bout duration (frames)', fontsize=12, fontweight='bold')
plt.title('Bout duration per Interaction Number', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.xticks(rotation=45)
plt.ylim(0, None)
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/interaction_bout_durations.png', dpi=300, bbox_inches='tight')
plt.close()



###### NEAREST NEIGHOUR DISTANCE ######

df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/fed-fed/nearest_neighbour.csv')
df1['condition'] = 'fed-fed'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-fed/nearest_neighbour.csv')
df2['condition'] = 'fed-starved'

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-starved/nearest_neighbour.csv')
df3['condition'] = 'starved-starved'


plt.figure(figsize=(8,8))

df = pd.concat([df1, df2, df3], ignore_index=True)


bins = np.linspace(0, 90, 90)  # 0 to 2.5 in 0.1 increments
df['bin'] = pd.cut(df['body-body'], bins, include_lowest=True)
df['bin_center'] = df['bin'].apply(lambda x: x.mid)


# Count per file-condition-bin
counts = (
    df.groupby(['filename', 'condition', 'bin_center'])
    .size()
    .groupby(['filename', 'condition'], group_keys=False)
    .apply(lambda x: x / x.sum())
    .reset_index(name='density')
)


sns.lineplot(data=counts, x='bin_center', y='density', hue='condition', errorbar='sd')


# sns.histplot(data=df, x='body-body', hue='condition', stat='density', common_norm=False, alpha=0.5, binwidth=1, multiple='dodge')

plt.xlabel('Nearest Neighbour Distance (mm)', fontsize=12, fontweight='bold')
plt.ylabel('Probability', fontsize=12, fontweight='bold')

plt.title('Nearest Neighour Distance Distriubtion', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])



plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/nearest_neighbour_distance.png', dpi=300, bbox_inches='tight')

plt.close()


###### NEAREST NEIGHOUR DISTANCE x SPEED ######

df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/fed-fed/nearest_neighbour.csv')
df1['condition'] = 'fed-fed'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-fed/nearest_neighbour.csv')
df2['condition'] = 'fed-starved'

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-starved/nearest_neighbour.csv')
df3['condition'] = 'starved-starved'


plt.figure(figsize=(8,8))

df = pd.concat([df1, df2, df3], ignore_index=True)

bins = np.arange(0, 91, 1)   # edges: 0,1,2,...,90 (1mm bins)
df['bin'] = pd.cut(df['body-body'], bins=bins, include_lowest=True)
df['bin_center'] = df['bin'].apply(lambda x: x.mid)

grouped = (
    df
    .groupby(['file', 'condition', 'bin_center'])['speed']
    .mean()
    .reset_index()
)


sns.lineplot(data=grouped, x='bin_center', y='speed', hue='condition', errorbar='sd')

plt.xlabel('Nearest Neighbour Distance (mm)', fontsize=12, fontweight='bold')
plt.ylabel('Speed', fontsize=12, fontweight='bold')

plt.title('Speed vs Nearest Neighbour Distance', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])


plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/nearest_neighbour_distance_speed.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.close()




###### NEAREST NEIGHOUR DISTANCE x ANGLE ######

df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/fed-fed/nearest_neighbour.csv')
df1['condition'] = 'fed-fed'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-fed/nearest_neighbour.csv')
df2['condition'] = 'fed-starved'

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-starved/nearest_neighbour.csv')
df3['condition'] = 'starved-starved'


plt.figure(figsize=(8,8))

df = pd.concat([df1, df2, df3], ignore_index=True)

bins = np.arange(0, 91, 1)   # edges: 0,1,2,...,90 (1mm bins)
df['bin'] = pd.cut(df['body-body'], bins=bins, include_lowest=True)
df['bin_center'] = df['bin'].apply(lambda x: x.mid)

grouped = (
    df
    .groupby(['file', 'condition', 'bin_center'])['angle']
    .mean()
    .reset_index()
)

sns.lineplot(data=grouped, x='bin_center', y='angle', hue='condition', errorbar='sd')

plt.xlabel('Nearest Neighbour Distance (mm)', fontsize=12, fontweight='bold')
plt.ylabel('Angle', fontsize=12, fontweight='bold')

plt.title('Angle vs Nearest Neighbour Distance', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/nearest_neighbour_distance_angle.png', dpi=300, bbox_inches='tight')
plt.close()
# Show the plot





###### EUCLIDEAN DISTANCE ######

df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/fed-fed/euclidean_distances.csv')
df1['condition'] = 'fed-fed'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-fed/euclidean_distances.csv')
df2['condition'] = 'fed-starved'

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-starved/euclidean_distances.csv')
df3['condition'] = 'starved-starved'


plt.figure(figsize=(8,8))

df = pd.concat([df1, df2, df3], ignore_index=True)


sns.lineplot(data=df, x='time', y='average_distance', hue='condition', errorbar='sd')

plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
plt.ylabel('Euclidean Distance (mm)', fontsize=12, fontweight='bold')

plt.title('Euclidean Distance vs Time', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/euclidean_distance_time.png', dpi=300, bbox_inches='tight')
plt.close()
# Show the plot



###### SPEED ######

df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/fed-fed/speed_over_time.csv')
df1['condition'] = 'fed-fed'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-fed/speed_over_time.csv')
df2['condition'] = 'fed-starved'

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-starved/speed_over_time.csv')
df3['condition'] = 'starved-starved'


plt.figure(figsize=(8,8))

df = pd.concat([df1, df2, df3], ignore_index=True)

sns.histplot(
    data=df,
    x='speed',              # <-- your speed column
    hue='condition',
    bins=50,
    stat='density',         # compare shapes not raw counts
    common_norm=False,      # don't force same total area
    element='step',
    fill=False
)


plt.xlabel('Speed (mm/s)', fontsize=12, fontweight='bold')
plt.ylabel('Density', fontsize=12, fontweight='bold')

plt.title('Speed Distribution', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/speed.png', dpi=300, bbox_inches='tight')
plt.close()
# Show the plot


###### ACCELERATION ######

df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/fed-fed/acceleration_accross_time.csv')
df1['condition'] = 'fed-fed'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-fed/acceleration_accross_time.csv')
df2['condition'] = 'fed-starved'

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-starved/acceleration_accross_time.csv')
df3['condition'] = 'starved-starved'


plt.figure(figsize=(8,8))

df = pd.concat([df1, df2, df3], ignore_index=True)

sns.histplot(
    data=df,
    x='acceleration',              # <-- your speed column
    hue='condition',
    bins=50,
    stat='density',         # compare shapes not raw counts
    common_norm=False,      # don't force same total area
    element='step',
    fill=False
)


plt.xlabel('Acceleration (mm/s²)', fontsize=12, fontweight='bold')
plt.ylabel('Density', fontsize=12, fontweight='bold')

plt.title('Acceleration Distribution', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/acceleration.png', dpi=300, bbox_inches='tight')
plt.close()


###### ANGLE ######

df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/fed-fed/angle_over_time.csv')
df1['condition'] = 'fed-fed'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-fed/angle_over_time.csv')
df2['condition'] = 'fed-starved'

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-starved/angle_over_time.csv')
df3['condition'] = 'starved-starved'


plt.figure(figsize=(8,8))

df = pd.concat([df1, df2, df3], ignore_index=True)

sns.histplot(
    data=df,
    x='angle',              # <-- your speed column
    hue='condition',
    bins=50,
    stat='density',         # compare shapes not raw counts
    common_norm=False,      # don't force same total area
    element='step',
    fill=False
)


plt.xlabel('Angle (degrees)', fontsize=12, fontweight='bold')
plt.ylabel('Density', fontsize=12, fontweight='bold')

plt.title('Angle Distribution', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/angle.png', dpi=300, bbox_inches='tight')
plt.close()

###### movement direction #####

df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/fed-fed/movement_direction.csv')
df1['condition'] = 'fed-fed'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-fed/movement_direction.csv')
df2['condition'] = 'fed-starved'

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-starved/movement_direction.csv')
df3['condition'] = 'starved-starved'


plt.figure(figsize=(8,8))

df = pd.concat([df1, df2, df3], ignore_index=True)

sns.histplot(
    data=df,
    x='movement_angle',              # <-- your speed column
    hue='condition',
    bins=50,
    stat='density',         # compare shapes not raw counts
    common_norm=False,      # don't force same total area
    element='step',
    fill=False
)


plt.xlabel('Angle (degrees)', fontsize=12, fontweight='bold')
plt.ylabel('Density', fontsize=12, fontweight='bold')

plt.title('Angle Distribution', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/movement_direction.png', dpi=300, bbox_inches='tight')
plt.close()



############################################# WITHIN 10MM ONLY #######################################

###### NEAREST NEIGHOUR DISTANCE ######

df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/fed-fed/nearest_neighbour.csv')
df1['condition'] = 'fed-fed'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-fed/nearest_neighbour.csv')
df2['condition'] = 'fed-starved'

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-starved/nearest_neighbour.csv')
df3['condition'] = 'starved-starved'


plt.figure(figsize=(8,8))

df = pd.concat([df1, df2, df3], ignore_index=True)


bins = np.linspace(0, 90, 90)  # 0 to 2.5 in 0.1 increments
df['bin'] = pd.cut(df['body-body'], bins, include_lowest=True)
df['bin_center'] = df['bin'].apply(lambda x: x.mid)


# Count per file-condition-bin
counts = (
    df.groupby(['filename', 'condition', 'bin_center'])
    .size()
    .groupby(['filename', 'condition'], group_keys=False)
    .apply(lambda x: x / x.sum())
    .reset_index(name='density')
)


sns.lineplot(data=counts, x='bin_center', y='density', hue='condition', errorbar='sd')


# sns.histplot(data=df, x='body-body', hue='condition', stat='density', common_norm=False, alpha=0.5, binwidth=1, multiple='dodge')

plt.xlabel('Nearest Neighbour Distance (mm)', fontsize=12, fontweight='bold')
plt.ylabel('Probability', fontsize=12, fontweight='bold')
plt.xlim(0, 10)

plt.title('Nearest Neighour Distance Distriubtion', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])



plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/within10mm/nearest_neighbour_distance.png', dpi=300, bbox_inches='tight')

plt.close()


###### NEAREST NEIGHOUR DISTANCE x SPEED ######

df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/fed-fed/nearest_neighbour.csv')
df1['condition'] = 'fed-fed'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-fed/nearest_neighbour.csv')
df2['condition'] = 'fed-starved'

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-starved/nearest_neighbour.csv')
df3['condition'] = 'starved-starved'


plt.figure(figsize=(8,8))

df = pd.concat([df1, df2, df3], ignore_index=True)

bins = np.arange(0, 91, 1)   # edges: 0,1,2,...,90 (1mm bins)
df['bin'] = pd.cut(df['body-body'], bins=bins, include_lowest=True)
df['bin_center'] = df['bin'].apply(lambda x: x.mid)

grouped = (
    df
    .groupby(['file', 'condition', 'bin_center'])['speed']
    .mean()
    .reset_index()
)


sns.lineplot(data=grouped, x='bin_center', y='speed', hue='condition', errorbar='sd')

plt.xlabel('Nearest Neighbour Distance (mm)', fontsize=12, fontweight='bold')
plt.ylabel('Speed', fontsize=12, fontweight='bold')

plt.title('Speed vs Nearest Neighbour Distance', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])
plt.xlim(0, 10)

plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/within10mm/nearest_neighbour_distance_speed.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.close()




###### NEAREST NEIGHOUR DISTANCE x ANGLE ######

df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/fed-fed/nearest_neighbour.csv')
df1['condition'] = 'fed-fed'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-fed/nearest_neighbour.csv')
df2['condition'] = 'fed-starved'

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-starved/nearest_neighbour.csv')
df3['condition'] = 'starved-starved'


plt.figure(figsize=(8,8))

df = pd.concat([df1, df2, df3], ignore_index=True)

bins = np.arange(0, 91, 1)   # edges: 0,1,2,...,90 (1mm bins)
df['bin'] = pd.cut(df['body-body'], bins=bins, include_lowest=True)
df['bin_center'] = df['bin'].apply(lambda x: x.mid)

grouped = (
    df
    .groupby(['file', 'condition', 'bin_center'])['angle']
    .mean()
    .reset_index()
)

sns.lineplot(data=grouped, x='bin_center', y='angle', hue='condition', errorbar='sd')

plt.xlabel('Nearest Neighbour Distance (mm)', fontsize=12, fontweight='bold')
plt.ylabel('Angle', fontsize=12, fontweight='bold')

plt.title('Angle vs Nearest Neighbour Distance', fontsize=16, fontweight='bold')
plt.xlim(0, 10)
plt.tight_layout(rect=[1, 1, 1, 1])


plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/within10mm/nearest_neighbour_distance_angle.png', dpi=300, bbox_inches='tight')
plt.close()
# Show the plot





