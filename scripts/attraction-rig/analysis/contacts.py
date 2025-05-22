import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyarrow.feather as feather


df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/group-housed/contacts_1mm.csv')
df3['condition'] = 'GH_N2'

df4 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/socially-isolated/contacts_1mm.csv')
df4['condition'] = 'SI_N2'

df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed/contacts_1mm.csv')
df5['condition'] = 'GH_N10'

df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/socially-isolated/contacts_1mm.csv')
df6['condition'] = 'SI_N10'

df7 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated/contacts_1mm.csv')
df7['condition'] = 'PSEUDO-SI_N10'

df8 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/contacts_1mm.csv')
df8['condition'] = 'PSEUDO-GH_N10'

df9 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/socially-isolated/contacts_1mm.csv')
df9['condition'] = 'PSEUDO-SI_N2'

df10 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/group-housed/contacts_1mm.csv')
df10['condition'] = 'PSEUDO-GH_N2'


# ALL DF
# df = pd.concat([df3, df4, df5, df6, df7, df8, df9, df10], ignore_index=True)

# N2
# df = pd.concat([df3, df4], ignore_index=True)

# N10
# df = pd.concat([df5, df6], ignore_index=True)

# # PEUDO N10
# df = pd.concat([df5, df8], ignore_index=True) # gh
# df = pd.concat([df6, df7], ignore_index=True) # si

# ## PEUDO N2
# df = pd.concat([df3, df10], ignore_index=True) # gh
# df = pd.concat([df4, df9], ignore_index=True) # si 


# ALL N10
df = pd.concat([df5, df6, df7, df8], ignore_index=True)

# ALL N2
# df = pd.concat([df3, df4, df9, df10], ignore_index=True)



### FREQUENCY ###


all_files = df[["file", "condition"]].drop_duplicates() ## account for files with no interactions

# Count interactions per file
interaction_counts = (
    df.groupby(["condition", "file"])["Interaction Number"]
    .nunique()
    .reset_index(name="interaction_count")
)
print(interaction_counts)

interaction_counts = all_files.merge(interaction_counts, on=["file", "condition"], how="left")
interaction_counts["interaction_count"] = interaction_counts["interaction_count"].fillna(0)


### DURATION ###

durations = (
    df.groupby(["condition", "file", "Interaction Number"])
    .agg(duration=("frame", "count"))
    .reset_index()
)


file_avg_duration = (
    durations.groupby(["condition", "file"])["duration"]
    .mean()
    .reset_index(name="avg_duration")
)


### % TIME IN PROXIMITY ###

# Count rows (frames) per file
frame_counts = (
    df.groupby(["condition", "file"])
    .size()
    .reset_index(name="frame_count")
)

frame_counts["percentage"] = frame_counts["frame_count"] / 3600





### CONTACT  TYPE ###

# proportions = (
#     df.groupby(["condition", "file", "Interaction Number", "Interaction Type"])
#     .size()
#     .groupby(["condition", "file", "Interaction Number"])
#     .apply(lambda x: x / x.sum())
#     .reset_index(name="proportion")
# )


# file_proportion = (
#     proportions
#     .groupby(["condition", "file", "Interaction Type"])["proportion"]
#     .mean()
#     .reset_index()
# )


### TRACK PAIR FREQUENCY ###



### PLOTTING ###

palette = {
    'GH_N10': '#4C72B0',
    'SI_N10': '#55A868',
    'PSEUDO-SI_N10': '#C44E52',
    'PSEUDO-GH_N10': '#8172B3',
    'GH_N2': '#4C72B0',
    'SI_N2': '#55A868',
    'PSEUDO-SI_N2': '#C44E52',
    'PSEUDO-GH_N2': '#8172B3'
}


fig, axes = plt.subplots(1, 3, figsize=(12, 8))

axes = axes.flatten()

# --- 1. Interaction Frequency ---
sns.violinplot(
    data=interaction_counts,
    x="condition",
    y="interaction_count",
    ax=axes[0],
    palette=palette,
    inner="box",   # shows a mini boxplot inside the violin
    linewidth=1.2
)

axes[0].set_title("Interaction Frequency")
axes[0].set_ylabel("Interaction Frequency")
axes[0].set_xlabel("Condition")
axes[0].tick_params(axis='x', rotation=45)

# --- 2. Interaction Duration ---

sns.violinplot(
    data=file_avg_duration,
    x="condition",
    y="avg_duration",
    ax=axes[1],
    palette=palette,
    inner="box",
    linewidth=1.2
)


axes[1].set_title("Average Interaction Duration")
axes[1].set_ylabel("Duration (s)")
axes[1].set_xlabel("Condition")
axes[1].tick_params(axis='x', rotation=45)

# # --- 3. Contact Type Proportions ---
# sns.boxplot(
#     data=file_proportion,
#     x="Interaction Type",
#     y="proportion",
#     hue="condition",
#     ax=axes[2]
# )
# axes[2].set_title("Contact Type Proportion Per Interaction")
# axes[2].set_ylabel("Proportion")
# axes[2].set_xlabel("Contact Type")
# axes[2].tick_params(axis='x', rotation=45)
# axes[2].legend(title="Type", bbox_to_anchor=(1.05, 1), loc='upper left')

# # --- 3. Contact Type Proportions ---

sns.barplot(
    data=frame_counts,
    x="condition",
    y="frame_count", 
    # hue="condition",
    ax=axes[2], palette=palette,
    ci="sd",
)
axes[2].set_title('')
axes[2].set_ylabel("Number of Frames")
axes[2].set_xlabel("Contact Type")
axes[2].tick_params(axis='x', rotation=45)
axes[2].legend(title="Type", bbox_to_anchor=(1.05, 1), loc='upper left')


plt.tight_layout()

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/contacts/n10-1mm.png', dpi=300, bbox_inches='tight')

plt.show()


