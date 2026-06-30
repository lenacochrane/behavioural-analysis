import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import os
import matplotlib as mpl
import re


df_clustering = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/EXPERIMENTS/ELAV_T21-D02/SIDEVIEW/ANALYSIS/percent_in_clusters.csv')

df_survival = pd.read_excel('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/EXPERIMENTS/ELAV_T21-D02/experiments.xlsx', sheet_name='exp')


## apparently turns into a proper date so format doesnt matter idk
df_survival["date_of_exp"] = pd.to_datetime(
    df_survival["date_of_exp"],
    dayfirst=True
).dt.date

survival_keep = df_survival[
    ["date_of_exp", "rig_number", "number_survived"]
].copy()


mapping_rows = []

for file_id, condition in df_clustering[["file_id", "condition"]].drop_duplicates().values:

    # mean clustering for this video
    mean_clustered = (
        df_clustering[df_clustering["file_id"] == file_id]["percent_in_clusters"]
        .mean()
    )

    # extract date
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", file_id)

    # extract SV rig number
    rig_match = re.search(r"SV(\d+)", file_id)

    if date_match and rig_match:

        exp_date = pd.to_datetime(date_match.group(1)).date()
        rig_number = int(rig_match.group(1))

        mapping_rows.append({
            "file_id": file_id,
            "condition": condition,
            "date_of_exp": exp_date,
            "rig_number": rig_number,
            "mean_percent_clustered": mean_clustered
        })

mapping_df = pd.DataFrame(mapping_rows)

print(mapping_df.head())


merged_df = mapping_df.merge(
    survival_keep,
    on=["date_of_exp", "rig_number"],
    how="left"
)

print(merged_df.head())

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=merged_df,
    x="number_survived",
    y="mean_percent_clustered",
    hue="condition",
    palette="viridis",
)

plt.xlabel("Clustering Size in Development", fontweight='bold', fontsize=12)
plt.ylabel("Mean % Time in Clusters", fontweight='bold', fontsize=12)
plt.title("Clustering vs Grouped House Survival", fontweight='bold', fontsize=14)
plt.legend(title="Condition", title_fontsize='13', fontsize='11')
plt.tight_layout()
sns.despine()
plt.savefig("/Users/cochral/repos/behavioural-analysis/plots/phd/ndd-sideview/clustering_vs_survival.pdf", format='pdf')
plt.show()
