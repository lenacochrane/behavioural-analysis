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
from shapely.geometry import Point, Polygon
from shapely import wkt

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# # video_folder = '/Users/cochral/Desktop/MOSEQ/si'

# # # Prefix to add
# # prefix = 'N1-SI_'

# # # Loop through files in the folder
# # for filename in os.listdir(video_folder):
# #     if filename.endswith('.mp4') and not filename.startswith(prefix):
# #         old_path = os.path.join(video_folder, filename)
# #         new_filename = prefix + filename
# #         new_path = os.path.join(video_folder, new_filename)
# #         os.rename(old_path, new_path)
# #         print(f"Renamed: {filename} ➝ {new_filename}")


# # df = pd.read_csv('/Users/cochral/Desktop/MOSEQ/KEYPOINT/2025_05_21-13_29_37/moseq_df.csv')

# # # sns.barplot(data=df, x='syllable', y='frequency', hue='group')

# # # plt.show()

# # print(df.columns)

# # print(df.head(20))


# # df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed/interactions_8mm.csv')
# # df5['condition'] = 'GH_N10'

# # df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/socially-isolated/interactions_8mm.csv')
# # df6['condition'] = 'SI_N10'


# # df = pd.concat([df5, df6], ignore_index=True) # n2
# # # Filter to only frame 0 rows
# # df_0 = df[df['Normalized Frame'] == 0]

# # # Set figure size and style
# # plt.figure(figsize=(8, 6))

# # # Plot histogram of min_distance split by condition
# # sns.histplot(
# #     data=df_0,
# #     x='min_distance',
# #     hue='condition',
# #     bins=50,
# #     element='step',
# #     stat='density',  # use 'count' if you prefer raw counts
# #     common_norm=False
# # )

# # # Labeling
# # plt.title('Distribution of min_distance at Normalized Frame 0', fontsize=14, fontweight='bold')
# # plt.xlabel('min_distance', fontsize=12)
# # plt.ylabel('Density', fontsize=12)
# # # plt.legend(title='Condition')
# # plt.tight_layout()

# # # Show plot
# # plt.show()



# df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/group-housed/interaction_types.csv')
# df3['condition'] = 'GH_N2'

# df4 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/socially-isolated/interaction_types.csv')
# df4['condition'] = 'SI_N2'

# df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed/interaction_types.csv')
# df5['condition'] = 'GH_N10'

# df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/socially-isolated/interaction_types.csv')
# df6['condition'] = 'SI_N10'

# df7 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated/interaction_types.csv')
# df7['condition'] = 'PSEUDO-SI_N10'

# df8 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/interaction_types.csv')
# df8['condition'] = 'PSEUDO-GH_N10'

# df9 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/socially-isolated/interaction_types.csv')
# df9['condition'] = 'PSEUDO-SI_N2'

# df10 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/group-housed/interaction_types.csv')
# df10['condition'] = 'PSEUDO-GH_N2'

# df = pd.concat([df3, df4], ignore_index=True)

# grouped = (
#     df.groupby(['file', 'condition'])['count']
#     .sum()
#     .reset_index()
# )

# plt.figure(figsize=(10, 6))
# sns.barplot(
#     data=grouped,
#     x='condition',
#     y='count',
#     edgecolor='black'
# )

# plt.xticks(rotation=90)
# plt.xlabel("Condition", fontsize=12)
# plt.ylabel("Total Count", fontsize=12)
# plt.title("Total Frames <1mm", fontsize=14, fontweight='bold')
# plt.tight_layout()
# plt.show()




# df = pd.read_csv('/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA10/2025_05_29-11_45_53/stats_df.csv')

# sns.barplot(data=df, x='syllable', y='duration', errorbar='sd')
# plt.xticks(rotation=45)
# plt.show()


# sns.barplot(data=df, x='syllable', hue='condition', y='frequency')
# plt.xlabel('Condition')
# plt.ylabel('Count')
# plt.title('Number of entries per condition')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.xlim(None,20)
# plt.show()

# df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/cleaned-tracks/socially-isolated/n10/SOCIALLY-ISOLATED/2025-02-24_11-49-59_td12.csv')

# print(df.head())


# directory = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/moseq/n1_csv'

# for filename in os.listdir(directory):
#     if "analysis" in filename and filename.endswith(".csv"):
#         # Example: 2025-03-25_16-33-40_td3.tracks.000_2025-03-25_16-33-40_td3.analysis.csv
#         # Want: GH-N1_2025-03-25_16-33-40_td3.csv
#         original_path = os.path.join(directory, filename)

#         base_name = filename.split('.tracks')[0]  # take the part before `.tracks`
#         new_filename = f"SI-N1_{base_name}.csv"
#         new_path = os.path.join(directory, new_filename)

#         os.rename(original_path, new_path)
#         print(f"✅ Renamed: {filename} ➜ {new_filename}")


# df = pd.read_csv('/Users/cochral/Desktop/MOSEQ/predictions-csv/GH-N10_2025-02-28_13-00-52_td9.csv')

# print(df)

# directory = '/Users/cochral/Desktop/MOSEQ/predictions-csv'


# # Columns to check for NaNs as a group
# check_columns = ['body.score', 'tail.score', 'head.score']
# # Column to also set if condition is met
# fill_columns = check_columns + ['instance.score']

# for filename in os.listdir(directory):
#     if filename.endswith('.csv'):
#         filepath = os.path.join(directory, filename)
#         try:
#             df = pd.read_csv(filepath)

#             # Check that all required columns are present
#             if all(col in df.columns for col in fill_columns):
#                 # Identify rows where body, tail, and head scores are all NaN
#                 condition = df[check_columns].isna().all(axis=1)

#                 # Apply fill only on those rows
#                 if condition.any():
#                     df.loc[condition, fill_columns] = 1.0
#                     df.to_csv(filepath, index=False)
#                     print(f"{filename}: Updated {condition.sum()} row(s) where all three scores were NaN.")
#                 else:
#                     print(f"{filename}: No rows with all NaNs in body, tail, and head scores.")

#             else:
#                 print(f"{filename}: Missing one or more required columns.")

#         except Exception as e:
#             print(f"Failed to process {filename}: {e}")



# df = pd.read_csv('/Users/cochral/Desktop/MOSEQ/predictions-csv/GH-N10_2025-02-28_13-00-52_td9.csv')

# print(df)



file = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed/2025-02-24_11-59-11_td14_perimeter.wkt'

# Load the WKT polygon
with open('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed/2025-02-24_11-59-11_td14_perimeter.wkt', 'r') as f:
    polygon = wkt.loads(f.read().strip())

# Use bounding box to get diameter
minx, miny, maxx, maxy = polygon.bounds
diameter_x = maxx - minx
diameter_y = maxy - miny
diameter = max(diameter_x, diameter_y)  # assuming circular shape

print(f"🟢 Diameter (bounding box): {diameter:.2f} units")