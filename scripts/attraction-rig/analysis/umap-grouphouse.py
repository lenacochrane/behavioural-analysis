import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

print("\U0001F4E5 Loading CSV...")

# Load only group-housed data
df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed/interactions.csv')
print("\n✅ CSV loaded.")

####-- DECIDE FEATURE COLUMNS --####

feature_columns = [
    "min_distance",  
    "track1_speed", "track2_speed"
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

    expected_frames = list(range(-15, 16))
    actual_frames = list(cropped["Normalized Frame"])
    if sorted(actual_frames) != expected_frames:
        return None

    return cropped

print("\n✂️ Cropping interactions...")
df["interaction_id"] = df["Interaction Number"]
df_cropped = df.groupby("interaction_id", group_keys=False).apply(crop_interaction)
df_cropped = df_cropped.drop(columns="interaction_id")  # Remove after grouping

print("✅ Cropping complete.")
print("Cropped rows:", len(df_cropped))

####-- PIVOTING TO VECTORISED FORMAT --####

print("\n🔀 Pivoting to vectorized format...")
df_vectorized = df_cropped.pivot_table(
    index="Interaction Number",
    columns="Normalized Frame",
    values=feature_columns
)

print("✅ Pivot complete.")
print("Vectorized shape:", df_vectorized.shape)

df_vectorized.columns = [f"{col[0]}_frame{col[1]}" for col in df_vectorized.columns] 
df_vectorized = df_vectorized.fillna(0)

print("Feature variance check:")
print(df_vectorized.var().sort_values(ascending=True).head(10))

####-- SCALING FEATURES --####

print("\n📐 Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_vectorized)
print("✅ Scaling done.")

####-- UMAP GRID SEARCH --####

output_dir = "/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/umap-grouphouse"
os.makedirs(output_dir, exist_ok=True)

print("\n🔁 Running UMAP grid search...")

neighbors_list = [2, 3, 5, 8, 10, 15, 30, 55]
min_dist_list = [0.01, 0.1, 0.5]

for n in neighbors_list:
    for d in min_dist_list:
        print(f"\n🌍 UMAP: n_neighbors={n}, min_dist={d}")
        umap_model = umap.UMAP(n_neighbors=n, min_dist=d, n_components=2, random_state=42)
        X_umap = umap_model.fit_transform(X_scaled)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x=X_umap[:, 0],
            y=X_umap[:, 1],
            alpha=0.8
        )
        plt.title(f"UMAP (n_neighbors={n}, min_dist={d})")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.tight_layout()

        fname = f"umap_n{n}_d{d}.png"
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()

print(f"\n✅ UMAP grid search complete. Plots saved to: {output_dir}")
