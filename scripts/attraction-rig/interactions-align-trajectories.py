import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyarrow.feather as feather
import cv2
import re
from scipy.spatial.distance import pdist
from shapely import wkt
import glob
from random import sample
from sklearn.decomposition import PCA

# # Load & filter
# df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser/test5_F30/cropped_interactions.csv')
# df_filter = df[df['interaction_id'] == 'iso_300']

# # Extract coords as arrays
# coords1 = df_filter[['Track_1 x_body','Track_1 y_body']].dropna().values
# coords2 = df_filter[['Track_2 x_body','Track_2 y_body']].dropna().values

# # 1) Show raw starts
# print("Raw start coords:")
# print(" Track1:", coords1[0])
# print(" Track2:", coords2[0])

# # 2) PCA straightness & axes
# pca1 = PCA(2).fit(coords1)
# axis1, s1 = pca1.components_[0], pca1.explained_variance_ratio_[0]
# pca2 = PCA(2).fit(coords2)
# axis2, s2 = pca2.components_[0], pca2.explained_variance_ratio_[0]
# print(f"Straightness → Track1: {s1:.3f}, Track2: {s2:.3f}")

# # 3) Swap so coords1 is always the anchor
# swapped = False
# if s2 > s1:
#     coords1, coords2 = coords2, coords1
#     axis1,   axis2   = axis2,   axis1
#     s1,      s2      = s2,      s1
#     swapped = True

# print("Swapped into anchor role?", swapped)
# print(" After swap start coords:")
# print("  anchor (coords1):", coords1[0])
# print("  partner(coords2):", coords2[0])

# # Alignment helper
# def align(coords, axis, origin):
#     X = coords - origin
#     v = axis.copy()
#     if v[1] < 0: v = -v
#     φ = np.arctan2(v[1], v[0])
#     α = np.pi/2 - φ
#     R = np.array([[np.cos(α), -np.sin(α)],
#                   [np.sin(α),  np.cos(α)]])
#     return X.dot(R.T)

# # 4) Align around the anchor’s start
# origin = coords1[0]
# aligned1 = align(coords1, axis1, origin)
# aligned2 = align(coords2, axis1, origin)

# # 5) Horizontal flip so partner on right
# if np.median(aligned2[:,0]) < 0:
#     aligned1[:,0] *= -1
#     aligned2[:,0] *= -1

# # # 6) Vertical flip if anchor still below
# # med_y = np.median(aligned1[:,1])
# # print("Median Y of aligned anchor before vertical flip:", med_y)
# # if med_y < 0:
# #     aligned1[:,1] *= -1
# #     aligned2[:,1] *= -1
# #     print(" Applied vertical flip")
# # else:
# #     print(" No vertical flip needed")
# # print("Median Y after flip:", np.median(aligned1[:,1]))

# # flip if the average y is negative
# mean_y = np.mean(aligned1[:,1])
# if mean_y < 0:
#     aligned1[:,1] *= -1
#     aligned2[:,1] *= -1
#     print("Applied vertical flip (mean was < 0)")



# # 7) Plot before & after with start markers
# fig, axs = plt.subplots(1, 2, figsize=(12,6))

# # Original
# axs[0].plot(coords1[:,0], coords1[:,1], '-o', label='Anchor')
# axs[0].plot(coords2[:,0], coords2[:,1], '-o', label='Partner')
# axs[0].scatter(coords1[0,0], coords1[0,1], c='C0', marker='X', s=100, label='Start A')
# axs[0].scatter(coords2[0,0], coords2[0,1], c='C1', marker='X', s=100, label='Start B')
# axs[0].set_title('Original')
# axs[0].axis('equal')
# axs[0].legend()

# # Aligned
# axs[1].plot(aligned1[:,0], aligned1[:,1], '-o', label='Anchor')
# axs[1].plot(aligned2[:,0], aligned2[:,1], '-o', label='Partner')
# axs[1].scatter(aligned1[0,0], aligned1[0,1], c='C0', marker='X', s=100, label='Start A')
# axs[1].scatter(aligned2[0,0], aligned2[0,1], c='C1', marker='X', s=100, label='Start B')
# axs[1].set_title('Aligned (Anchor→vertical)')
# axs[1].axis('equal')
# axs[1].legend()

# plt.suptitle(f'Interaction {df_filter["interaction_id"].iloc[0]}')
# plt.tight_layout()
# plt.show()




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- Helper functions ---
def compute_pca_axis(points):
    """Return the first principal component (unit vector) and its explained variance ratio."""
    pca = PCA(n_components=2).fit(points)
    axis = pca.components_[0]
    score = pca.explained_variance_ratio_[0]
    # ensure the axis points upward
    return (axis if axis[1] >= 0 else -axis), score


def align_and_flip(track, anchor_axis, anchor_start):
    """
    Translate so anchor_start -> (0,0), rotate so anchor_axis -> +y,
    and flip horizontally/vertically to keep partner on the right and anchor up.
    """
    # translate
    X = track - anchor_start
    # rotate
    phi = np.arctan2(anchor_axis[1], anchor_axis[0])  # angle of axis
    alpha = np.pi/2 - phi                            # rotate to +y
    R = np.array([[np.cos(alpha), -np.sin(alpha)],
                  [np.sin(alpha),  np.cos(alpha)]])
    X_rot = X.dot(R.T)
    return X_rot

# --- Main processing ---
df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser/test5_F30/cropped_interactions.csv')

# Initialize aligned columns
df['anchor_x_body'] = np.nan
df['anchor_y_body'] = np.nan
df['partner_x_body'] = np.nan
df['partner_y_body'] = np.nan

for interaction_id, group in df.groupby('interaction_id'):
    group = group.sort_values('Frame')
    coords1 = group[['Track_1 x_body','Track_1 y_body']].dropna().values
    coords2 = group[['Track_2 x_body','Track_2 y_body']].dropna().values
    if len(coords1) < 2 or len(coords2) < 2:
        continue
    # Compute PCA axes & scores
    axis1, s1 = compute_pca_axis(coords1)
    axis2, s2 = compute_pca_axis(coords2)
    # Choose anchor and partner
    if s1 >= s2:
        anchor_pts, partner_pts, anchor_axis = coords1, coords2, axis1
    else:
        anchor_pts, partner_pts, anchor_axis = coords2, coords1, axis2
    # Align both
    start = anchor_pts[0]
    A_al = align_and_flip(anchor_pts, anchor_axis, start)
    B_al = align_and_flip(partner_pts, anchor_axis, start)
    # Horizontal flip if partner is left
    if np.median(B_al[:,0]) < 0:
        A_al[:,0] *= -1
        B_al[:,0] *= -1
    # Vertical flip if anchor is predominantly down
    if np.mean(A_al[:,1]) < 0:
        A_al[:,1] *= -1
        B_al[:,1] *= -1
    # Assign back to DataFrame
    idx = group.index[:len(A_al)]
    df.loc[idx, ['anchor_x_body','anchor_y_body']]  = A_al
    df.loc[idx, ['partner_x_body','partner_y_body']] = B_al

# --- Sanity check plotting 15 random interactions ---
sample_ids = df['interaction_id'].drop_duplicates().sample(15, random_state=0)
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
for ax, iid in zip(axes.ravel(), sample_ids):
    sub = df[df['interaction_id'] == iid]
    A = sub[['anchor_x_body','anchor_y_body']].values
    B = sub[['partner_x_body','partner_y_body']].values
    ax.plot(A[:,0], A[:,1], '-o', markersize=2, label='Anchor')
    ax.plot(B[:,0], B[:,1], '-o', markersize=2, label='Partner')
    ax.scatter(A[0,0], A[0,1], c='C0', marker='X')
    ax.scatter(B[0,0], B[0,1], c='C1', marker='X')
    ax.set_title(iid)
    ax.axis('equal'); ax.axis('off')
plt.tight_layout(); plt.show()

