
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
from sklearn.cluster import DBSCAN

print("\U0001F4E5 Loading CSVs...")

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

print("\n✂️ Cropping interactions (using closest to frame 0)...")
df_cropped = df.groupby("interaction_id", group_keys=False).apply(crop_interaction)

print("✅ Cropping complete.")
print("Cropped rows:", len(df_cropped))
print("Cropped conditions:")
print(df_cropped["condition"].value_counts())

interaction_lengths = df_cropped.groupby("interaction_id").size()
print("\n🧪 Frame counts per cropped interaction:")
print(interaction_lengths.value_counts().sort_index())
if (interaction_lengths != 31).any():
    print("❗Warning: Some cropped interactions are not 31 frames long.")
else:
    print("✅ All cropped interactions are exactly 31 frames long.")

print("\U0001F4C8 Checking unique normalized frames before pivoting:")
print(sorted(df_cropped["Normalized Frame"].unique()))

####-- PIVOTING TO VECTORISED FORMAT --####

print("\n🔀 Pivoting to vectorized format...")
df_vectorized = df_cropped.pivot_table(
    index="interaction_id",
    columns="Normalized Frame",
    values=feature_columns
)

print("✅ Pivot complete.")
print("Vectorized shape:", df_vectorized.shape)

df_vectorized.columns = [f"{col[0]}_frame{col[1]}" for col in df_vectorized.columns] 
df_vectorized = df_vectorized.fillna(0)

print("Feature variance check:")
print(df_vectorized.var().sort_values(ascending=True).head(10))

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

####-- STANDARDISES EACH FEATURE  --####

print("\n📐 Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_vectorized.drop(columns="condition"))
print("✅ Scaling done.")

####-- GRID SEARCH: UMAP + DBSCAN PARAMETER COMBINATIONS  --####

output_dir = "/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/umap-gridsearch"
os.makedirs(output_dir, exist_ok=True)

print("\n🔁 Running UMAP + DBSCAN grid search...")

neighbors_list = [2, 4, 3, 5, 6, 7, 8, 10, 15, 30, 55]
min_dist_list = [0.01, 0.02, 0.03, 0.5, 0.1, 0.5]
metrics_list = ["cosine", "euclidean", "manhattan", "correlation"]

summary = []

for metric in metrics_list:
    for n in neighbors_list:
        for d in min_dist_list:
            print(f"\n🌍 UMAP: metric={metric}, n_neighbors={n}, min_dist={d}")
            umap_model = umap.UMAP(n_neighbors=n, min_dist=d, n_components=2,  metric=metric, random_state=42)
            X_umap = umap_model.fit_transform(X_scaled)

            # Continue with everything else unchanged, but now include metric in filenames and plots
            umap_df = pd.DataFrame(X_umap, columns=["UMAP1", "UMAP2"], index=df_vectorized.index)
            umap_df["condition"] = df_vectorized["condition"]

            # Side-by-side iso/group plots
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
            for i, cond in enumerate(["iso", "group"]):
                cond_df = umap_df[umap_df["condition"] == cond]
                sns.scatterplot(data=cond_df, x="UMAP1", y="UMAP2", ax=axes[i], alpha=0.8)
                axes[i].set_title(f"{cond} – metric={metric}, n_neighbors={n}, min_dist={d}")
                axes[i].set_xlabel("UMAP Dimension 1")
                axes[i].set_ylabel("UMAP Dimension 2")

            fig.suptitle(f"UMAP Projections – metric={metric}, n_neighbors={n}, min_dist={d}", fontsize=14)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            joint_fname = f"umap_{metric}_n{n}_d{d}_iso_vs_group.png"
            plt.savefig(os.path.join(output_dir, joint_fname))
            plt.close()

            df_vectorized[f"UMAP_1_{metric}_n{n}_d{d}"] = X_umap[:, 0]
            df_vectorized[f"UMAP_2_{metric}_n{n}_d{d}"] = X_umap[:, 1]



            dbscan_eps_list = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
            dbscan_min_samples_list = [3, 5, 8, 10]

            # DBSCAN clustering
            
            for eps in dbscan_eps_list:
                for min_samp in dbscan_min_samples_list:
                    clustering = DBSCAN(eps=eps, min_samples=min_samp).fit(X_umap)
                    labels = clustering.labels_
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = list(labels).count(-1)


                    # Save clustering results to df
                    label_col = f"cluster_{metric}_n{n}_d{d}_eps{eps}_min{min_samp}"
                    df_vectorized[label_col] = labels

                    # ✅ COUNT iso/group per cluster
                    cluster_counts = df_vectorized[[label_col, "condition"]].value_counts().reset_index()
                    cluster_counts.columns = ["cluster_label", "condition", "count"]

                    pivot_counts = cluster_counts.pivot(index="cluster_label", columns="condition", values="count").fillna(0).astype(int)

                    # Save summary CSV for this clustering
                    cluster_csv_name = f"cluster_counts_{metric}_n{n}_d{d}_eps{eps}_min{min_samp}.csv"
                    pivot_counts.to_csv(os.path.join(output_dir, cluster_csv_name))

                    # # Save clustering results to df
                    # label_col = f"cluster_{metric}_n{n}_d{d}_eps{eps}_min{min_samp}"
                    # df_vectorized[label_col] = labels

                    # Save plot
                    plt.figure(figsize=(8, 6))
                    sns.scatterplot(
                        x=f"UMAP_1_{metric}_n{n}_d{d}",
                        y=f"UMAP_2_{metric}_n{n}_d{d}",
                        hue=label_col,
                        data=df_vectorized,
                        palette="tab10",
                        alpha=0.8,
                        legend=None
                    )
                    plt.title(f"UMAP={metric}, n={n}, d={d}\nDBSCAN: eps={eps}, min_samp={min_samp} – {n_clusters} clusters, {n_noise} noise")
                    plt.xlabel("UMAP Dimension 1")
                    plt.ylabel("UMAP Dimension 2")
                    plt.tight_layout()

                    fname = f"umap_{metric}_n{n}_d{d}_eps{eps}_min{min_samp}_clusters.png"
                    plt.savefig(os.path.join(output_dir, fname))
                    plt.close()

                    summary.append({
                        "metric": metric,
                        "n_neighbors": n,
                        "min_dist": d,
                        "dbscan_eps": eps,
                        "min_samples": min_samp,
                        "n_clusters": n_clusters,
                        "n_noise_points": n_noise,
                        "plot_file": fname
                    })
            # clustering = DBSCAN(eps=0.5, min_samples=5).fit(X_umap)
            # labels = clustering.labels_
            # n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            # n_noise = list(labels).count(-1)

            # df_vectorized[f"UMAP_1_{metric}_n{n}_d{d}"] = X_umap[:, 0]
            # df_vectorized[f"UMAP_2_{metric}_n{n}_d{d}"] = X_umap[:, 1]
            # df_vectorized[f"cluster_{metric}_n{n}_d{d}"] = labels

            # # Plot clustered
            # plt.figure(figsize=(8, 6))
            # sns.scatterplot(
            #     x=f"UMAP_1_{metric}_n{n}_d{d}",
            #     y=f"UMAP_2_{metric}_n{n}_d{d}",
            #     hue=f"cluster_{metric}_n{n}_d{d}",
            #     data=df_vectorized,
            #     palette="tab10",
            #     alpha=0.8,
            #     legend=None
            # )
            # plt.title(f"UMAP ({metric}, n_neighbors={n}, min_dist={d}) – {n_clusters} clusters, {n_noise} noise")
            # plt.xlabel("UMAP Dimension 1")
            # plt.ylabel("UMAP Dimension 2")
            # plt.tight_layout()

            # fname = f"umap_{metric}_n{n}_d{d}_clusters.png"
            # plt.savefig(os.path.join(output_dir, fname))
            # plt.close()

            # summary.append({
            #     "metric": metric,
            #     "n_neighbors": n,
            #     "min_dist": d,
            #     "n_clusters": n_clusters,
            #     "n_noise_points": n_noise,
            #     "plot_file": fname
            # })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(output_dir, "umap_dbscan_summary.csv"), index=False)

print(f"\n✅ Grid search complete. Summary saved to: {os.path.join(output_dir, 'umap_dbscan_summary.csv')}")
