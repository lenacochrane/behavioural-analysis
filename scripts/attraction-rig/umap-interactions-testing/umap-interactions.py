

################################################ ----- WARNING ----- ################################################

# SCRIPT WORKS AND PROCESSES DATA IDENTICALLY TO NEMO, HOWEVER DUE TO LIBRARY AND ENVIORNMENT DIFFERENCES THE UMAPS #
############# PRODUCED ARE NEVER EQUAL OR REPRODUCABLE -> I WILL NOW CREATE A VERSION TO BE RAN ON NEMO #############


########## CAN ATTEMPT TO RUN ON ENVIORNMENT CALLED MAGGOTS310 - LIBRARIES CLOSER IN VERSION TO THE CLUSTER ##########

#####################################################################################################################

import os
os.environ["OMP_NUM_THREADS"]     = "1"
os.environ["MKL_NUM_THREADS"]     = "1"
os.environ["OPENBLAS_NUM_THREADS"]= "1"

import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import cv2


print("UMAP version:", umap.__version__)
import inspect
print("UMAP.__init__ signature:\n", inspect.signature(umap.UMAP.__init__))

print("📥 Loading CSVs...")

df_group = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed/interactions.csv')
df_group['condition'] = 'group'

df_iso = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/socially-isolated/interactions.csv')
df_iso['condition'] = 'iso'

df = pd.concat([df_iso, df_group], ignore_index=True)

print("\n✅ CSVs loaded and combined.")
print(df['condition'].value_counts())


####-- DECIDE FEATURE COLUMNS --####

feature_columns = [
    "track1_speed", "track2_speed", 
    "track1_acceleration", "track2_acceleration",
    "track1_angle", "track2_angle",
    "t1_tail-tail_t2", "t1_tail-body_t2", "t1_tail-head_t2",
    "t1_body-tail_t2", "t1_body-body_t2", "t1_body-head_t2",
    "t1_head-tail_t2", "t1_head-body_t2", "t1_head-head_t2"
]


####-- CROP INTERACTIONS --####


def crop_interaction(group):
    if group.empty or "Normalized Frame" not in group.columns:
        return None
    center_idx = (group["Normalized Frame"].abs()).idxmin()
    if pd.isna(center_idx):
        return None
    center_pos = group.index.get_loc(center_idx)
    if center_pos < 15 or (center_pos + 16) >= len(group):
        return None
    cropped = group.iloc[center_pos - 15 : center_pos + 16].copy()
    cropped["interaction_id"] = group["interaction_id"].iloc[0]
    expected_frames = list(range(-15, 16))
    actual_frames = list(cropped["Normalized Frame"])
    if sorted(actual_frames) != expected_frames:
        return None
    return cropped


####-- UNIQUE ID PER INTERACTION --####
df["interaction_id"] = df["condition"] + "_" + df["Interaction Number"].astype(str)

####-- CROP INTERACTIONS --####
df_cropped = df.groupby("interaction_id", group_keys=False).apply(crop_interaction)

print("✅ Cropping complete.")
print("Cropped rows:", len(df_cropped))
print("Cropped conditions:")
print(df_cropped["condition"].value_counts())



####-- CHECK INTERACTIONS ARE CORRECT --####
interaction_lengths = df_cropped.groupby("interaction_id").size()
print("\n🧪 Frame counts per cropped interaction:")
print(interaction_lengths.value_counts().sort_index())

if (interaction_lengths != 31).any():
    print("❗Warning: Some cropped interactions are not 31 frames long.")
else:
    print("✅ All cropped interactions are exactly 31 frames long.")

####-- PIVOTING TO VECTORISED FORMAT --####

print("\n🔁 Pivoting to vectorized format...")

df_vectorized = df_cropped.pivot_table(
    index="interaction_id",
    columns="Normalized Frame",
    values=feature_columns
)

print("✅ Pivot complete.")
print("Vectorized shape:", df_vectorized.shape)


df_vectorized.columns = [f"{col[0]}_frame{col[1]}" for col in df_vectorized.columns] # 🧼 Flatten column names
df_vectorized = df_vectorized.fillna(0)

print("Feature variance check:")
print(df_vectorized.var().sort_values(ascending=True).head(20))


####-- MERGE CONDITIONS WITH INTERACTIONS  --####

print("\n🔗 Merging condition into vectorized dataframe...")

interaction_conditions = df_cropped.groupby("interaction_id")["condition"].first().reset_index()

df_vectorized = df_vectorized.reset_index().merge(
    interaction_conditions,
    on="interaction_id",
    how="left"
).set_index("interaction_id")

print("✅ Condition merged.")
print(df_vectorized['condition'].value_counts())


print('here')


####-- STANDARDISES EACH FEATURE  --####

print("\n📐 Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_vectorized.drop(columns="condition"))


# features_only = df_vectorized.drop(columns="condition")
# X_scaled = pd.DataFrame(
#     scaler.fit_transform(features_only),
#     index=features_only.index,  # <- ⬅️ This is key
#     columns=features_only.columns
# )


print("✅ Scaling done.")


####-- CREATE DIRECTORY TO SAVE RESULTS  --####

output_dir = "/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/umap_2/local"
os.makedirs(output_dir, exist_ok=True)

####-- GRID SEARCH: UMAP + DBSCAN PARAMETER COMBINATIONS  --####

print("\n🔁 Running UMAP + DBSCAN grid search...")


####-- UMAP  --####

print("\n🌍 Running UMAP...")


umap_model = umap.UMAP(metric='cosine', n_neighbors=4, min_dist=0.001, n_components=2, random_state=42, n_jobs=1,  force_approximation_algorithm=True,init="spectral" ) # cosine_n8_d0.03

X_umap = umap_model.fit_transform(X_scaled)
print("✅ UMAP done.")

# 📎 Store UMAP in dataframe
df_vectorized["UMAP_1"] = X_umap[:, 0]
df_vectorized["UMAP_2"] = X_umap[:, 1]
df_vectorized["condition"] = df_vectorized["condition"].astype(str)

# new df with umap dimensions and condiiton 
umap_df = pd.DataFrame(X_umap, columns=["UMAP1", "UMAP2"], index=df_vectorized.index)
umap_df["condition"] = df_vectorized["condition"]

print("\n📊 Ready to plot. Final condition count:")
print(df_vectorized["condition"].value_counts())

####-- PLOT THE UMAP FOR ISO AND GROUP --####

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
for i, cond in enumerate(["iso", "group"]):
    cond_df = df_vectorized[df_vectorized["condition"] == cond]
    sns.scatterplot(data=cond_df, x="UMAP_1", y="UMAP_2", ax=axes[i], alpha=0.8)
    axes[i].set_title(f"{cond}")
    axes[i].set_xlabel("UMAP Dimension 1")
    axes[i].set_ylabel("UMAP Dimension 2")

fig.suptitle(f"iso v group", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
joint_fname = f"iso_vs_group.png"
plt.savefig(os.path.join(output_dir, joint_fname))
plt.close()


####-- PLOT THE UMAP  --####

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
plt.savefig(os.path.join(output_dir, "umap.png"))
plt.close()


####-- RUN DBSCAN ON UMAP  --####

print("\n🔍 Running DBSCAN clustering on UMAP projection...")
clustering = DBSCAN(eps=0.2, min_samples=5).fit(X_umap)

labels = clustering.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

df_vectorized["cluster"] = clustering.labels_

print("✅ Clustering complete Cluster")


####-- SAVE CLUSTER COUNTS  --####

df_vectorized["cluster"] = clustering.labels_
label_col = "cluster"

cluster_counts = df_vectorized[[label_col, "condition"]].value_counts().reset_index()
cluster_counts.columns = ["cluster_label", "condition", "count"]

pivot_counts = cluster_counts.pivot(index="cluster_label", columns="condition", values="count").fillna(0).astype(int)

cluster_csv_name = f"cluster_counts.csv"
pivot_counts.to_csv(os.path.join(output_dir, cluster_csv_name))


####-- SAVE CSV CONTAINING INTERACTIONS, CLUSTER AND UMAP INFORMATION  --####

# 🔗 Merge cluster labels into the cropped interactions dataframe
df_cropped_with_clusters = df_cropped.merge(
    df_vectorized[["cluster", "UMAP_1", "UMAP_2"]],
    left_on="interaction_id",
    right_index=True,
    how="left"
)

name = "clustered_interactions"
df_cropped_with_clusters.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)



####-- PLOT THE UMAP WITH CLUSTER ANALYSIS --####

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


####-- SAVE MEAN FEATURES PER CLUSTER --####

cluster_means = df_vectorized.groupby("cluster").mean(numeric_only=True)
cluster_means.to_csv(os.path.join(output_dir, "cluster_mean_features.csv"))
print(f"✅ Saved: cluster_mean_features.csv")


####-- SAVE MEAN FEATURES PER CLUSTER --####

print("\n📈 Plotting average trajectories per cluster...")


trajectory_dir = os.path.join(output_dir, "mean_trajectories")
os.makedirs(trajectory_dir, exist_ok=True)

for cluster_label in sorted(df_vectorized["cluster"].unique()):
    interaction_ids = df_vectorized[df_vectorized["cluster"] == cluster_label].index.tolist()
    cluster_data = df_cropped[df_cropped["interaction_id"].isin(interaction_ids)]

    # Compute average trajectory over frames
    avg_traj = cluster_data.groupby("Normalized Frame")[
        ["Track_1 x_body", "Track_1 y_body", "Track_2 x_body", "Track_2 y_body"]
    ].mean()

    # Plot
    plt.figure(figsize=(6, 6))
    plt.plot(avg_traj["Track_1 x_body"], avg_traj["Track_1 y_body"], label="Track 1 avg", linewidth=2)
    plt.plot(avg_traj["Track_2 x_body"], avg_traj["Track_2 y_body"], label="Track 2 avg", linewidth=2)

    plt.title(f"Average Trajectory – Cluster {cluster_label}")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()

    # Save to mean_trajectories folder
    plot_filename = f"avg_traj_cluster{cluster_label}.png"
    plt.savefig(os.path.join(trajectory_dir, plot_filename))
    plt.close()


 ####-- SAVE KEYPOINT VIDEOS PER CLUSTER --#### 


video_out_dir = os.path.join(output_dir, f"cluster_keypoint_videos")
os.makedirs(video_out_dir, exist_ok=True)


canvas_size = (800, 800)
fps = 5
dot_radius = 5

for cluster_id in sorted(df_vectorized["cluster"].unique()):
    cluster_inter_ids = df_vectorized[df_vectorized["cluster"] == cluster_id].index
    
    # If cluster has fewer than 5, sample only that many
    selected_ids = pd.Series(cluster_inter_ids).sample(min(5, len(cluster_inter_ids)), random_state=0)


    for i, interaction_id in enumerate(selected_ids):
        clip = df_cropped[df_cropped["interaction_id"] == interaction_id]

        if clip.empty:
            continue

        x_cols = [col for col in clip.columns if " x_" in col]
        y_cols = [col for col in clip.columns if " y_" in col]
        all_x = clip[x_cols].values.flatten()
        all_y = clip[y_cols].values.flatten()
        all_x = all_x[~np.isnan(all_x)]
        all_y = all_y[~np.isnan(all_y)]

        if len(all_x) == 0 or len(all_y) == 0:
            continue

        min_x, max_x = np.min(all_x), np.max(all_x)
        min_y, max_y = np.min(all_y), np.max(all_y)
        cx = (max_x + min_x) / 2
        cy = (max_y + min_y) / 2
        bbox_w = max_x - min_x
        bbox_h = max_y - min_y
        scale = min((canvas_size[0] - 40) / bbox_w, (canvas_size[1] - 40) / bbox_h)

        filename = f"cluster{cluster_id}_example{i}.mp4"
        filepath = os.path.join(video_out_dir, filename)
        writer = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*'mp4v'), fps, canvas_size)

        for _, row in clip.iterrows():
            frame = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)

            for track, color in [('Track_1', (255, 0, 0)), ('Track_2', (0, 0, 255))]:
                for part in ['head', 'body', 'tail']:
                    x = row.get(f'{track} x_{part}', np.nan)
                    y = row.get(f'{track} y_{part}', np.nan)
                    if not np.isnan(x) and not np.isnan(y):
                        sx = int((x - cx) * scale + canvas_size[0] / 2)
                        sy = int((y - cy) * scale + canvas_size[1] / 2)
                        cv2.circle(frame, (sx, sy), dot_radius, color, -1, lineType=cv2.LINE_AA)

            writer.write(frame)

        writer.release()
  

print('umap complete :)')

