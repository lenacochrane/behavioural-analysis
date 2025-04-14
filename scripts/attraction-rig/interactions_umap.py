import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

print("📥 Loading CSVs...")

df_group = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed/interactions.csv')
df_group['condition'] = 'group'

df_iso = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/socially-isolated/interactions.csv')
df_iso['condition'] = 'iso'

df = pd.concat([df_iso, df_group], ignore_index=True)


##### CHANGE AND CHANGE SAVING FILE
df = df[df['Frame'] <= 1800]

print("Checking Normalized Frame distribution in 'group' interactions:")
print(df_group["Normalized Frame"].describe())
print("\nHow many group interactions contain a frame close to 0 (+/-5)?")
close_to_zero = df_group[df_group["Normalized Frame"].between(-5, 5)]
print(close_to_zero["Interaction Number"].nunique(), "interactions found.")



print("\n✅ CSVs loaded and combined.")
print(df['condition'].value_counts())
print("Total rows:", len(df))


# feature_columns = [
#     "min_distance",  
#     "track1_speed", "track2_speed", 
#     "track1_acceleration", "track2_acceleration",
#     "track1_length", "track2_length",  
#     "track1_angle", "track2_angle", "track1_approach_angle", 'track2_approach_angle'
# ]

feature_columns = [
    "min_distance",  
    "track1_speed", "track2_speed", 
    "track1_acceleration", "track2_acceleration",
    "track1_length", "track2_length",  
    "track1_angle", "track2_angle", 
    "track1_approach_angle", "track2_approach_angle",

    # # Coordinates for Track 1
    "Track_1 x_tail", "Track_1 y_tail",
    "Track_1 x_body", "Track_1 y_body",
    "Track_1 x_head", "Track_1 y_head",

    # Coordinates for Track 2
    "Track_2 x_tail", "Track_2 y_tail",
    "Track_2 x_body", "Track_2 y_body",
    "Track_2 x_head", "Track_2 y_head"
]

# feature_columns = [

#     # Coordinates for Track 1
#     "Track_1 x_tail", "Track_1 y_tail",
#     "Track_1 x_body", "Track_1 y_body",
#     "Track_1 x_head", "Track_1 y_head",

#     # Coordinates for Track 2
#     "Track_2 x_tail", "Track_2 y_tail",
#     "Track_2 x_body", "Track_2 y_body",
#     "Track_2 x_head", "Track_2 y_head"
# ]


def crop_interaction(group):
    # ✅ Find the frame closest to Normalized Frame == 0
    if group.empty or "Normalized Frame" not in group.columns:
        return None
    
    center_idx = (group["Normalized Frame"].abs()).idxmin()  # Get index of closest-to-zero
    if pd.isna(center_idx):
        return None

    center_pos = group.index.get_loc(center_idx)  # position within group
    start = max(center_pos - 22, 0)
    end = min(center_pos + 22, len(group) - 1)

    cropped = group.iloc[start:end + 1].copy()
    # cropped["condition"] = group["condition"].iloc[0]
    cropped["interaction_id"] = group["interaction_id"].iloc[0]

    return cropped

df["interaction_id"] = df["condition"] + "_" + df["Interaction Number"].astype(str)




print("\n✂️ Cropping interactions (using closest to frame 0)...")
# df_cropped = df.groupby(["condition", "Interaction Number"], group_keys=False).apply(crop_interaction)
df_cropped = df.groupby("interaction_id", group_keys=False).apply(crop_interaction)

print("✅ Cropping complete.")
print("Cropped rows:", len(df_cropped))
print("Cropped conditions:")
print(df_cropped["condition"].value_counts())




# 🧱 Pivot to vectorized format
print("\n🔁 Pivoting to vectorized format...")
# df_vectorized = df_cropped.pivot_table(
#     index="Interaction Number",
#     columns="Normalized Frame",
#     values=feature_columns
# )

df_vectorized = df_cropped.pivot_table(
    index="interaction_id",
    columns="Normalized Frame",
    values=feature_columns
)



print("✅ Pivot complete.")
print("Vectorized shape:", df_vectorized.shape)

# 🧼 Flatten column names
df_vectorized.columns = [f"{col[0]}_frame{col[1]}" for col in df_vectorized.columns]
df_vectorized = df_vectorized.fillna(0)

# 🔗 Merge condition back in using interaction number
print("\n🔗 Merging condition into vectorized dataframe...")
# interaction_conditions = df_cropped.groupby("Interaction Number")["condition"].first().reset_index()
# df_vectorized = df_vectorized.reset_index().merge(
#     interaction_conditions,
#     on="Interaction Number",
#     how="left"
# ).set_index("Interaction Number")

interaction_conditions = df_cropped.groupby("interaction_id")["condition"].first().reset_index()

df_vectorized = df_vectorized.reset_index().merge(
    interaction_conditions,
    on="interaction_id",
    how="left"
).set_index("interaction_id")


print("✅ Condition merged.")
print(df_vectorized['condition'].value_counts())


# 📏 Standardize features
print("\n📐 Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_vectorized.drop(columns="condition"))
print("✅ Scaling done.")


# 🧬 Run UMAP
print("\n🌍 Running UMAP...")
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X_scaled)
print("✅ UMAP done.")
print("UMAP shape:", X_umap.shape)


# 📎 Store UMAP in dataframe
df_vectorized["UMAP_1"] = X_umap[:, 0]
df_vectorized["UMAP_2"] = X_umap[:, 1]
df_vectorized["condition"] = df_vectorized["condition"].astype(str)


print("\n📊 Ready to plot. Final condition count:")
print(df_vectorized["condition"].value_counts())

# 🖼 Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x="UMAP_1",
    y="UMAP_2",
    data=df_vectorized,
    hue="condition",
    palette="Set2",
    alpha=0.8
)
plt.title("UMAP Projection of Interactions by Condition")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(title="Condition")
plt.tight_layout()
plt.show()





import os
from sklearn.cluster import DBSCAN

# 📁 Create output folder
output_dir = "/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/umap-clusters-1800-frames"
os.makedirs(output_dir, exist_ok=True)

# 🤝 Run DBSCAN on UMAP
print("\n🔍 Running DBSCAN clustering on UMAP projection...")
clustering = DBSCAN(eps=0.5, min_samples=5).fit(df_vectorized[["UMAP_1", "UMAP_2"]])
df_vectorized["cluster"] = clustering.labels_
print("✅ Clustering complete. Cluster label counts:")
print(df_vectorized["cluster"].value_counts())

# 🔗 Merge cluster labels into the cropped interactions dataframe
df_cropped_with_clusters = df_cropped.merge(
    df_vectorized[["cluster", "UMAP_1", "UMAP_2"]],
    left_on="interaction_id",
    right_index=True,
    how="left"
)

# 💾 Export cropped interaction data with cluster + UMAP info
cropped_output_path = "/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/combined_interactions_clusters.csv.csv"
df_cropped_with_clusters.to_csv(cropped_output_path, index=False)
print(f"✅ Saved cropped interactions with cluster & UMAP to: {cropped_output_path}")



# 📊 Plot UMAP colored by cluster
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x="UMAP_1", y="UMAP_2",
    data=df_vectorized,
    hue="cluster",
    palette="tab10",
    alpha=0.8
)
plt.title("UMAP Projection Colored by Cluster")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "umap_clusters.png"))
plt.close()

# 🧠 Save mean feature values per cluster
cluster_means = df_vectorized.groupby("cluster").mean(numeric_only=True)
cluster_means.to_csv(os.path.join(output_dir, "cluster_mean_features.csv"))
print(f"✅ Saved: cluster_mean_features.csv")

# 📊 Plot selected features
selected_features = [
    "track1_speed_frame0", "track2_speed_frame0",
    "track1_acceleration_frame0", "min_distance_frame0"
]

for feat in selected_features:
    plt.figure(figsize=(6, 4))
    sns.barplot(x=cluster_means.index, y=cluster_means[feat])
    plt.title(f"{feat} by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Mean Value")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{feat}_by_cluster.png"))
    plt.close()
print("📈 Saved feature plots:", ", ".join(selected_features))

# 🌀 Plot example trajectories per cluster
print("\n📍 Plotting 3 example trajectories per cluster...")

for cluster_label in sorted(df_vectorized["cluster"].unique()):
    sample_ids = df_vectorized[df_vectorized["cluster"] == cluster_label].index[:3]

    for ex_id in sample_ids:
        ex_data = df_cropped[df_cropped["interaction_id"] == ex_id]

        plt.figure(figsize=(5, 5))
        plt.plot(ex_data["Track_1 x_body"], ex_data["Track_1 y_body"], label="Track 1", alpha=0.8)
        plt.plot(ex_data["Track_2 x_body"], ex_data["Track_2 y_body"], label="Track 2", alpha=0.8)
        plt.title(f"Cluster {cluster_label} | Interaction {ex_id}")
        plt.xlabel("X position")
        plt.ylabel("Y position")
        plt.legend()
        plt.axis("equal")
        plt.tight_layout()

        fname = f"cluster{cluster_label}_sample_traj_{ex_id}.png"
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()

# 🧬 Plot average trajectories per cluster
print("\n📈 Plotting average trajectories per cluster...")

for cluster_label in sorted(df_vectorized["cluster"].unique()):
    ids = df_vectorized[df_vectorized["cluster"] == cluster_label].index.tolist()
    cluster_data = df_cropped[df_cropped["interaction_id"].isin(ids)]

    avg_traj = cluster_data.groupby("Normalized Frame")[
        ["Track_1 x_body", "Track_1 y_body", "Track_2 x_body", "Track_2 y_body"]
    ].mean()

    plt.figure(figsize=(6, 6))
    plt.plot(avg_traj["Track_1 x_body"], avg_traj["Track_1 y_body"], label="Track 1 avg", linewidth=2)
    plt.plot(avg_traj["Track_2 x_body"], avg_traj["Track_2 y_body"], label="Track 2 avg", linewidth=2)
    plt.title(f"Average Trajectory – Cluster {cluster_label}")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"avg_traj_cluster{cluster_label}.png"))
    plt.close()

print(f"\n✅ All cluster outputs saved to: {output_dir}")
