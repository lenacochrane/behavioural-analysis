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
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
import imageio.v2 as imageio
from scipy.stats import linregress
import matplotlib as mpl
from scipy.stats import binomtest
from scipy.stats import chisquare
from statsmodels.stats.multitest import multipletests
import ast
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.image as mpimg
import networkx as nx
from matplotlib.patches import FancyArrowPatch
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
import umap


#### CLASS TO ANALYSE THE CLUSTERS  
class ClusterPipeline:

    def __init__(self, directory, interactions, clusters, cluster_name, video_path, tracks_path):

        self.directory = directory
        self.interaction_path = interactions
        self.cluster_path = clusters
        self.cluster_name = cluster_name
        self.video_path = video_path
        self.tracks_path = tracks_path

        self.interactions = None
        self.clusters = None
        self.df = None
        
    
    #### METHOD LOADING_DATA: LOAD AND MERGE DATAFRAMES
    def loading_data(self):

        ## LOAD DATAFRAMES

        self.interactions = pd.read_csv(self.interaction_path)
        self.clusters = pd.read_csv(self.cluster_path)

        ## MISSING INTERACTIONS BETWEEN DATAFRAMES

        set1 = set(self.interactions['interaction_id'].unique())
        set2 = set(self.clusters['interaction_id'].unique())
        missing_from_cluster = sorted(set1 - set2)
        missing_from_cropped  = sorted(set2 - set1)
        print(f">>> {len(missing_from_cluster)} IDs in cropped not in cluster (e.g. {missing_from_cluster[:5]})")
        print(f">>> {len(missing_from_cropped)} IDs in cluster not in cropped (e.g. {missing_from_cropped[:5]})")

        ## MERGE DATAFRAMES

        self.df = pd.merge(
            self.interactions, 
            self.clusters[['interaction_id', self.cluster_name]], 
            on='interaction_id', 
            how='inner'
        )
    

    #### METHOD ANCHOR_PARTNER: CREATE ANCHOR AND PARTNER BASED ON LINEARITY OF TRACK  
    def anchor_partner(self):
        
        df = self.df

        ## CREATE ALIGNED AND PARTNER TRACKS FOR DRAWING TRAJECTORIES 

        ## Returns a straightness score 
        def compute_pca_axis(points):
            pca = PCA(n_components=2).fit(points)
            axis = pca.components_[0]
            score = pca.explained_variance_ratio_[0]
            # ensure the axis points upward
            return (axis if axis[1] >= 0 else -axis), score
        
        ## == Align the tracks (anchor 0,0) and rotate partner accordingly (on the right)
        def align_and_flip(track, anchor_axis, anchor_start):
            X = track - anchor_start
            phi = np.arctan2(anchor_axis[1], anchor_axis[0])  # angle of axis
            alpha = np.pi/2 - phi                            # rotate to +y
            R = np.array([[np.cos(alpha), -np.sin(alpha)],
                        [np.sin(alpha),  np.cos(alpha)]])
            X_rot = X.dot(R.T)
            return X_rot
        
        df['anchor x_body'] = np.nan
        df['anchor y_body'] = np.nan
        df['partner x_body'] = np.nan
        df['partner y_body'] = np.nan

        ## == Generate the anchor and partner x,y coordinates for future

        for interaction_id, group in df.groupby('interaction_id'):
            group = group.sort_values('Frame')
            coords1 = group[['Track_1 x_body','Track_1 y_body']].values
            coords2 = group[['Track_2 x_body','Track_2 y_body']].values
            if len(coords1) < 2 or len(coords2) < 2:
                continue
            # Compute PCA axes & scores
            axis1, s1 = compute_pca_axis(coords1)
            axis2, s2 = compute_pca_axis(coords2)
            # Choose anchor and partner
            if s1 >= s2:
                winner = 1
                anchor_pts, partner_pts, anchor_axis = coords1, coords2, axis1
            else:
                winner = 2
                anchor_pts, partner_pts, anchor_axis = coords2, coords1, axis2

            # Align both
            start = anchor_pts[0]
            A_al = align_and_flip(anchor_pts, anchor_axis, start)
            B_al = align_and_flip(partner_pts, anchor_axis, start)

            # # --- NEW: align head/tail using the SAME reference (before flips) ---
            # h1 = group[['Track_1 x_head','Track_1 y_head']].dropna().values
            # t1 = group[['Track_1 x_tail','Track_1 y_tail']].dropna().values
            # h2 = group[['Track_2 x_head','Track_2 y_head']].dropna().values
            # t2 = group[['Track_2 x_tail','Track_2 y_tail']].dropna().values


            h1 = group[['Track_1 x_head','Track_1 y_head']].values
            t1 = group[['Track_1 x_tail','Track_1 y_tail']].values
            h2 = group[['Track_2 x_head','Track_2 y_head']].values
            t2 = group[['Track_2 x_tail','Track_2 y_tail']].values

            A_head = align_and_flip(h1 if winner == 1 else h2, anchor_axis, start) if (len(h1) or len(h2)) else np.empty((0,2))
            A_tail = align_and_flip(t1 if winner == 1 else t2, anchor_axis, start) if (len(t1) or len(t2)) else np.empty((0,2))
            B_head = align_and_flip(h2 if winner == 1 else h1, anchor_axis, start) if (len(h1) or len(h2)) else np.empty((0,2))
            B_tail = align_and_flip(t2 if winner == 1 else t1, anchor_axis, start) if (len(t1) or len(t2)) else np.empty((0,2))
        # --------------------------------------------------------------------

            # Horizontal flip if partner is left
            # if np.median(B_al[:,0]) < 0:
            #     A_al[:,0] *= -1
            #     B_al[:,0] *= -1

            # Horizontal flip if partner starts on the left
            if B_al[0, 0] < 0:
                A_al[:, 0] *= -1
                B_al[:, 0] *= -1
                # --- NEW: apply same horizontal flip to head/tail
                if A_head.size: A_head[:, 0] *= -1
                if A_tail.size: A_tail[:, 0] *= -1
                if B_head.size: B_head[:, 0] *= -1
                if B_tail.size: B_tail[:, 0] *= -1

            # Vertical flip if anchor is predominantly down
            if np.mean(A_al[:,1]) < 0:
                A_al[:,1] *= -1
                B_al[:,1] *= -1
                    # --- NEW: apply same vertical flip to head/tail
                if A_head.size: A_head[:, 1] *= -1
                if A_tail.size: A_tail[:, 1] *= -1
                if B_head.size: B_head[:, 1] *= -1
                if B_tail.size: B_tail[:, 1] *= -1


            # Assign back to DataFrame
            idx = group.index[:len(A_al)]
            df.loc[idx, ['anchor x_body','anchor y_body']]  = A_al
            df.loc[idx, ['partner x_body','partner y_body']] = B_al# Initialize aligned columns

            # --- NEW: write aligned head/tail back (each uses its own length) ---
            if A_head.size:
                df.loc[group.index[:len(A_head)], ['anchor x_head','anchor y_head']] = A_head
            if A_tail.size:
                df.loc[group.index[:len(A_tail)], ['anchor x_tail','anchor y_tail']] = A_tail
            if B_head.size:
                df.loc[group.index[:len(B_head)], ['partner x_head','partner y_head']] = B_head
            if B_tail.size:
                df.loc[group.index[:len(B_tail)], ['partner x_tail','partner y_tail']] = B_tail
            # --------------------------------------------------------------------

            # → tag which original track was anchor (1 or 2)
            df.loc[idx, 'anchor_track']  = winner
            df.loc[idx, 'partner_track'] = 3 - winner
        

        ## APPROACH ANGLE CALCULATION FLIPPED INITIALLY BY ACCIDENT
        df["track1_approach_angle"] = 180 - df["track1_approach_angle"]
        df["track2_approach_angle"] = 180 - df["track2_approach_angle"]

        # === HEADING ANGLE CHANGE ===
        df['track1_heading_angle_change'] = df.groupby("interaction_id")["track1_angle"].diff().abs()
        df['track2_heading_angle_change'] = df.groupby("interaction_id")["track2_angle"].diff().abs()

        # === APPROACH ANGLE CHANGE ===
        df['track1_approach_angle_change'] = df.groupby("interaction_id")["track1_approach_angle"].diff().abs()
        df['track2_approach_angle_change'] = df.groupby("interaction_id")["track2_approach_angle"].diff().abs()

        metrics = [
        'speed_body',
        'speed_tail',
        'speed_head',
        'acceleration_body',
        'acceleration_tail',
        'acceleration_head',
        'angle',
        'approach_angle']

        for m in metrics:
            t1 = df[f'track1_{m}']
            t2 = df[f'track2_{m}']
            df[f'anchor_{m}']  = np.where(df['anchor_track']==1, t1, t2)
            df[f'partner_{m}'] = np.where(df['anchor_track']==1, t2, t1)

            # === Assign anchor/partner versions
            df['anchor_heading_angle_change']  = np.where(df['anchor_track'] == 1, df['track1_heading_angle_change'], df['track2_heading_angle_change'])
            df['partner_heading_angle_change'] = np.where(df['anchor_track'] == 1, df['track2_heading_angle_change'], df['track1_heading_angle_change'])

            df['anchor_approach_angle_change']  = np.where(df['anchor_track'] == 1, df['track1_approach_angle_change'], df['track2_approach_angle_change'])
            df['partner_approach_angle_change'] = np.where(df['anchor_track'] == 1, df['track2_approach_angle_change'], df['track1_approach_angle_change'])

        
        self.df = df


    
    ##########################################################################################################
    ## METHOD RAW_TRAJECTORIES: GENERATE RAW TRAJECTORIES OF INTERACTIONS
    ##########################################################################################################
    def raw_trajectories(self):

        df = self.df
        cluster_name = self.cluster_name 

        output = os.path.join(self.directory, "raw_trajectories")
        os.makedirs(output, exist_ok=True)


        for cluster_id in sorted(df[cluster_name].unique()):
            cluster_df = df[df[cluster_name] == cluster_id].copy()
            interaction_ids = cluster_df['interaction_id'].unique()
            np.random.shuffle(interaction_ids)
            sample_ids = interaction_ids[:81]  # up to 9×9 grid

            # figure grid size
            n = int(np.ceil(np.sqrt(len(sample_ids))))
            fig, axes = plt.subplots(n, n, figsize=(n * 2, n * 2), 
                                    sharex=False, sharey=False, constrained_layout=True)
            axes = axes.flatten()

            for ax, interaction_id in zip(axes, sample_ids):
                traj = cluster_df[cluster_df['interaction_id'] == interaction_id]

                A = traj[['anchor x_body','anchor y_body']].values
                B = traj[['partner x_body','partner y_body']].values

                # plot anchor & partner
                ax.plot(A[:,0], A[:,1],
                        color='C0', linewidth=1, alpha=0.8, label='Anchor')
                ax.plot(B[:,0], B[:,1],
                        color='C1', linewidth=1, alpha=0.8, label='Partner')

                # mark starts
                ax.scatter(A[0,0], A[0,1],
                        color='C0', marker='X', s=30, label='Start Anchor')
                ax.scatter(B[0,0], B[0,1],
                        color='C1', marker='X', s=30, label='Start Partner')

                # equal scaling, grid, labels
                ax.set_aspect('equal', adjustable='box')

                # ←—— Insert the padding snippet here:
                xmin = min(A[:,0].min(), B[:,0].min())
                xmax = max(A[:,0].max(), B[:,0].max())
                ymin = min(A[:,1].min(), B[:,1].min())
                ymax = max(A[:,1].max(), B[:,1].max())
                dx, dy = xmax - xmin, ymax - ymin
                half = max(dx, dy, 150) / 2
                xmid, ymid = (xmin + xmax) / 2, (ymin + ymax) / 2
                ax.set_xlim(xmid - half, xmid + half)
                ax.set_ylim(ymid - half, ymid + half)
                # ←—— End padding snippet


                ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
                ax.set_xlabel('X', fontsize=6)
                ax.set_ylabel('Y', fontsize=6)
                ax.tick_params(axis='both', which='both', labelsize=4)
                ax.set_title(interaction_id, fontsize=6)

            # turn off any unused subplots
            for i in range(len(sample_ids), len(axes)):
                axes[i].axis('off')

            plt.suptitle(f"Raw Trajectories – Cluster {cluster_id}", fontsize=14)

            save_path = os.path.join(output, f"cluster_{cluster_id}_grid.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()



    

     #### METHOD MEAN_TRAJECTORIES: MEAN TRAJECTORIES OF EACH CLUSTER
    def mean_trajectories(self):
         
        df = self.df
        cluster_name = self.cluster_name 

        output = os.path.join(self.directory, "mean_trajectories")
        os.makedirs(output, exist_ok=True)

        ## MEAN TRAJECTORIES

        for cluster_id in sorted(df[cluster_name].unique()):
            cluster_df = df[df[cluster_name] == cluster_id].copy()
            grouped = cluster_df.groupby("Normalized Frame")

            t1_x = grouped["anchor x_body"].mean()
            t1_y = grouped["anchor y_body"].mean()
            t2_x = grouped["partner x_body"].mean()
            t2_y = grouped["partner y_body"].mean()

            # 2) std-dev
            t1_x_std = grouped["anchor x_body"].std()
            t1_y_std = grouped["anchor y_body"].std()
            t2_x_std = grouped["partner x_body"].std()
            t2_y_std = grouped["partner y_body"].std()

            plt.figure(figsize=(6, 6))
            plt.plot(t1_x, t1_y, label="Track 1", color="red")
            plt.plot(t2_x, t2_y, label="Track 2", color="blue")
            plt.scatter(t1_x.iloc[0], t1_y.iloc[0], color="red", marker="o", label="T1 Start")
            plt.scatter(t2_x.iloc[0], t2_y.iloc[0], color="blue", marker="o", label="T2 Start")

                # error bars in X and Y
            plt.errorbar(
                t1_x, t1_y,
                xerr=t1_x_std, yerr=t1_y_std,
                fmt="none", ecolor="red", alpha=0.3, label="Track 1 ±1 SD"
            )
            plt.errorbar(
                t2_x, t2_y,
                xerr=t2_x_std, yerr=t2_y_std,
                fmt="none", ecolor="blue", alpha=0.3, label="Track 2 ±1 SD"
            )

            plt.gca().invert_yaxis()
            plt.axis("equal")
            plt.title(f"Mean Trajectory - Cluster {cluster_id}")
            plt.legend()
            plt.tight_layout()

        
            save_path = os.path.join(output, f"cluster_{cluster_id}.png")

            plt.savefig(save_path)
            plt.close()

        ## MEAN RELATIVE TRAJECTORIES 

        for cluster_id in sorted(df[cluster_name].unique()):
            cluster_df = df[df[cluster_name] == cluster_id]

            # Compute relative positions
            cluster_df["x_rel"] = cluster_df["partner x_body"] - cluster_df["anchor x_body"]
            cluster_df["y_rel"] = cluster_df["partner y_body"] - cluster_df["anchor y_body"]

            grouped = cluster_df.groupby("Normalized Frame")

            x_rel = grouped["x_rel"].mean()
            y_rel = grouped["y_rel"].mean()
            x_rel_std = grouped["x_rel"].std()
            y_rel_std = grouped["y_rel"].std()

            # Plot
            plt.figure(figsize=(6, 6))

            # Anchor point (Track 1 always at 0,0)
            plt.scatter(0, 0, color="blue", label="Anchor")

            # Relative trajectory of Track 2
            plt.plot(x_rel, y_rel, color="orange", label="Partner Rel. Trajectory")
            plt.scatter(x_rel.iloc[0], y_rel.iloc[0], color="darkorange", marker="o", label="Partner Start")

            # Error bars
            plt.errorbar(x_rel, y_rel, xerr=x_rel_std, yerr=y_rel_std, fmt="none", ecolor="orange", alpha=0.3)

            # plt.gca().invert_yaxis()
            plt.axis("equal")
            plt.title(f"Partner relative to Anchor — Cluster {cluster_id}")
            plt.legend()
            plt.tight_layout()

            save_path = os.path.join(output, f"cluster_{cluster_id}_relative.png")
            plt.savefig(save_path)
            plt.close()
        
        ## MEAN CONTACT PHASE TRAJECTORY (PRE,DURING,POST INTERACTION)

        bins = {
        "pre": range(-15, -5),
        "contact": range(-5, 5),
        "post": range(5, 15)}

        for cluster_id in sorted(df[cluster_name].unique()):
            cluster_df = df[df[cluster_name] == cluster_id]

            cluster_df["x_rel"] = cluster_df["partner x_body"] - cluster_df["anchor x_body"]
            cluster_df["y_rel"] = cluster_df["partner y_body"] - cluster_df["anchor y_body"]

            # one r for all three panels
            all_disp = np.concatenate([
            cluster_df["x_rel"].abs().values,
            cluster_df["y_rel"].abs().values])
            r = np.percentile(all_disp, 98)
            
            fig, axs = plt.subplots(1, 3, figsize=(12, 4),  constrained_layout=True)  # 3 panels: pre/contact/post

            for ax, (phase, frames) in zip(axs, bins.items()):
                phase_df = cluster_df[cluster_df["Normalized Frame"].isin(frames)]

                grouped = phase_df.groupby("Normalized Frame")
                x_mean = grouped["x_rel"].mean()
                y_mean = grouped["y_rel"].mean()
                x_std = grouped["x_rel"].std()
                y_std = grouped["y_rel"].std()

                ax.plot(x_mean, y_mean, label=f"{phase} trajectory", color="orange")
                ax.errorbar(x_mean, y_mean, xerr=x_std, yerr=y_std, fmt="none", ecolor="orange", alpha=0.3)
                ax.scatter(0, 0, color="blue", label="Anchor (0,0)")

                ax.set_xlim(-r, r)
                ax.set_ylim(-r, r)

                # ax.invert_yaxis()
                ax.set_aspect('equal', adjustable='box')

                ax.set_title(phase.capitalize())


            fig.suptitle(f"Cluster {cluster_id} – Relative Trajectories by Phase", fontsize=14)
            fig.savefig(os.path.join(output, f"cluster_{cluster_id}_phased_trajectory.png"))
            plt.close()

    

    def hierarchal_mean_trace_summary(self):
        
        df = self.df.copy()
        cluster_name = self.cluster_name


        ## LEVEL 0
        x1, y1 = 1, 0
        x2, y2 = 2, 0
        x3, y3 = 3, 0
        x4, y4 = 4, 0
        x5, y5 = 5, 0
        x6, y6 = 6, 0
        x7, y7 = 7, 0
        x8, y8 = 8, 0
        x9, y9 = 9, 0
        x10, y10 = 10, 0
        x11, y11 = 11, 0
        x12, y12 = 12, 0
        x13, y13 = 13, 0
        x14, y14 = 14, 0
        x15, y15 = 15, 0
        x16, y16 = 16, 0

        ## LEVEL 1
        x1_2, y1_2 = (x1 + x2) / 2, 1
        x3_4, y3_4 = (x3 + x4) / 2, 1
        x5_6, y5_6 = (x5 + x6) / 2, 1
        x7_8, y7_8 = (x7 + x8) / 2, 1
        x9_10, y9_10 = (x9 + x10) / 2, 1
        x11_12, y11_12 = (x11 + x12) / 2, 1
        x13_14, y13_14 = (x13 + x14) / 2, 1
        x15_16, y15_16 = (x15 + x16) / 2, 1

        ## LEVEL 2
        x1_4, y1_4 = (x1_2 + x3_4) / 2, 2
        x5_8, y5_8 = (x5_6 + x7_8) / 2, 2
        x9_12, y9_12 = (x9_10 + x11_12) / 2, 2
        x13_16, y13_16 = (x13_14 + x15_16) / 2, 2

        ## LEVEL 3
        x1_8, y1_8 = (x1_4 + x5_8) / 2, 3
        x9_16, y9_16 = (x9_12 + x13_16) / 2, 3

        ## LEVEL 4
        x_all, y_all = (x1_8 + x9_16) / 2, 4
            




        fig = plt.figure(figsize=(18, 4))
        gs = gridspec.GridSpec(3, 1, height_ratios=[0.3, 0.6, 0.05], hspace=0.3)

        ax1 = fig.add_subplot(gs[0])  # top: hierarchal tree
        # ax2 = fig.add_subplot(gs[1])
        sub_ax2 = gs[1].subgridspec(1, 16, wspace=0.1)
        axes_ax2 = [fig.add_subplot(sub_ax2[0, i]) for i in range(16)]

        #ax3 = fig.add_subplot(gs[2])  # bottom: average contact frames
        sub_ax3 = gs[2].subgridspec(1, 16, wspace=0.1)
        ax3 = fig.add_subplot(sub_ax3[0, :])  # one long axis spanning all 12 columns




        ## PLOTTING HIERARCHAL TREE - MANUALLY USING COORDINATES ABOVE 
        # ax.plot([x_start, x_end], [y_start, y_end])

        # 1-2
        ax1.plot([x1, x1], [y1, y1_2], color='black')
        ax1.plot([x2, x2], [y2, y1_2], color='black')
        ax1.plot([x1, x2], [y1_2, y1_2], color='black')

        # 3-4
        ax1.plot([x3, x3], [y3, y3_4], color='black')
        ax1.plot([x4, x4], [y4, y3_4], color='black')
        ax1.plot([x3, x4], [y3_4, y3_4], color='black')

        # 5-6
        ax1.plot([x5, x5], [y5, y5_6], color='black')
        ax1.plot([x6, x6], [y6, y5_6], color='black')
        ax1.plot([x5, x6], [y5_6, y5_6], color='black')

        # 7-8
        ax1.plot([x7, x7], [y7, y7_8], color='black')
        ax1.plot([x8, x8], [y8, y7_8], color='black')
        ax1.plot([x7, x8], [y7_8, y7_8], color='black')

        # 9-10
        ax1.plot([x9, x9], [y9, y9_10], color='black')
        ax1.plot([x10, x10], [y10, y9_10], color='black')
        ax1.plot([x9, x10], [y9_10, y9_10], color='black')

        # 11-12
        ax1.plot([x11, x11], [y11, y11_12], color='black')
        ax1.plot([x12, x12], [y12, y11_12], color='black')
        ax1.plot([x11, x12], [y11_12, y11_12], color='black')

        # 13-14
        ax1.plot([x13, x13], [y13, y13_14], color='black')
        ax1.plot([x14, x14], [y14, y13_14], color='black')
        ax1.plot([x13, x14], [y13_14, y13_14], color='black')

        # 15-16
        ax1.plot([x15, x15], [y15, y15_16], color='black')
        ax1.plot([x16, x16], [y16, y15_16], color='black')
        ax1.plot([x15, x16], [y15_16, y15_16], color='black')

        # 1-4
        ax1.plot([x1_2, x1_2], [y1_2, y1_4], color='black')
        ax1.plot([x3_4, x3_4], [y3_4, y1_4], color='black')
        ax1.plot([x1_2, x3_4], [y1_4, y1_4], color='black')

        # 5-8
        ax1.plot([x5_6, x5_6], [y5_6, y5_8], color='black')
        ax1.plot([x7_8, x7_8], [y7_8, y5_8], color='black')
        ax1.plot([x5_6, x7_8], [y5_8, y5_8], color='black')

        # 9-12
        ax1.plot([x9_10, x9_10], [y9_10, y9_12], color='black')
        ax1.plot([x11_12, x11_12], [y11_12, y9_12], color='black')
        ax1.plot([x9_10, x11_12], [y9_12, y9_12], color='black')

        # 13-16
        ax1.plot([x13_14, x13_14], [y13_14, y13_16], color='black')
        ax1.plot([x15_16, x15_16], [y15_16, y13_16], color='black')
        ax1.plot([x13_14, x15_16], [y13_16, y13_16], color='black')

        # 1-8
        ax1.plot([x1_4, x1_4], [y1_4, y1_8], color='black')
        ax1.plot([x5_8, x5_8], [y5_8, y1_8], color='black')
        ax1.plot([x1_4, x5_8], [y1_8, y1_8], color='black')

        # 9-16
        ax1.plot([x9_12, x9_12], [y9_12, y9_16], color='black')
        ax1.plot([x13_16, x13_16], [y13_16, y9_16], color='black')
        ax1.plot([x9_12, x13_16], [y9_16, y9_16], color='black')

        # all
        ax1.plot([x1_8, x1_8], [y1_8, y_all], color='black')
        ax1.plot([x9_16, x9_16], [y9_16, y_all], color='black')
        ax1.plot([x1_8, x9_16], [y_all, y_all], color='black')

       






        for spine in ax1.spines.values():
            spine.set_visible(False)
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        ## MEAN TRACES PLOTTING

        anchor_base = "#4F7942"   
        partner_base = "#916288"
        anchor_cmap = plt.cm.Greens
        partner_cmap = plt.cm.Purples

        ordered_frames = sorted(df["Normalized Frame"].dropna().unique())
        norm_frames = Normalize(vmin=ordered_frames[0], vmax=ordered_frames[-1]) #ltr for colour mapping

        clusters = sorted(df[cluster_name].unique())

        for cluster, ax in zip(clusters, axes_ax2):
            cluster_df = df[df[cluster_name] == cluster].copy()
            by_frame = cluster_df.sort_values("Normalized Frame").groupby("Normalized Frame")
            frames = [f for f in by_frame.groups.keys() if f == f] # list of frames (non-nan)
            
            # role+node, return mean X series and mean Y series (indexed by frame)
            def get_means(role, node):
                mx = by_frame[f"{role} x_{node}"].mean()  # Series: index = frame → mean x
                my = by_frame[f"{role} y_{node}"].mean()  # Series: index = frame → mean y
                return mx, my

            ## DRAW TRAILS
            for node in ("head", "body", "tail"):
                anchor_x, anchor_y = get_means("anchor",  node) # xy means per frame for anchor
                partner_x, partner_y = get_means("partner", node) # xy means per frame for partner

                for f0, f1 in zip(frames[:-1], frames[1:]): # consecutive frame pairs (frame 0, frame 1), (frame 1, frame 2), ...
                    # anchor
                    if f0 in anchor_x.index and f1 in anchor_x.index: # check that both frames have data
                        x0, y0 = anchor_x.loc[f0], anchor_y.loc[f0]
                        x1, y1 = anchor_x.loc[f1], anchor_y.loc[f1]
                        if np.isfinite(x0) and np.isfinite(y0) and np.isfinite(x1) and np.isfinite(y1):
                            ax.plot([x0, x1], [y0, y1],
                                    color=anchor_cmap(norm_frames(f1)), alpha=0.7, linewidth=1.2, zorder=1)
                    # partner
                    if f0 in partner_x.index and f1 in partner_x.index:
                        x0, y0 = partner_x.loc[f0], partner_y.loc[f0]
                        x1, y1 = partner_x.loc[f1], partner_y.loc[f1]
                        if np.isfinite(x0) and np.isfinite(y0) and np.isfinite(x1) and np.isfinite(y1):
                            ax.plot([x0, x1], [y0, y1],
                                    color=partner_cmap(norm_frames(f1)), alpha=0.7, linewidth=1.2, zorder=1) #zorder=1 trails goes underneath skeletons and markers
        
            ## DRAW SKELETONS: CONNECT HEAD→BODY→TAIL PER FRAME (TIME-COLORED)
            for f in frames:
                # anchor
                parts = {}
                for node in ("head", "body", "tail"):
                    x_means = by_frame[f"anchor x_{node}"].mean() 
                    y_means = by_frame[f"anchor y_{node}"].mean()

                    if f in x_means.index:
                        x, y = x_means.loc[f], y_means.loc[f]
                        if np.isfinite(x) and np.isfinite(y):
                            parts[node] = (x, y)
                
                if len(parts) == 3:
                 ax.plot([parts["head"][0], parts["body"][0], parts["tail"][0]],
                            [parts["head"][1], parts["body"][1], parts["tail"][1]],
                            color=anchor_cmap(norm_frames(f)), alpha=0.75, linewidth=1.0, zorder=2)
                # partner
                parts = {}
                for node in ("head", "body", "tail"):
                    x_means = by_frame[f"partner x_{node}"].mean()
                    y_means = by_frame[f"partner y_{node}"].mean()
                    if f in x_means.index:
                        x, y = x_means.loc[f], y_means.loc[f]
                        if np.isfinite(x) and np.isfinite(y):
                            parts[node] = (x, y)
                if len(parts) == 3:
                    ax.plot([parts["head"][0], parts["body"][0], parts["tail"][0]],
                            [parts["head"][1], parts["body"][1], parts["tail"][1]],
                            color=partner_cmap(norm_frames(f)), alpha=0.75, linewidth=1.0, zorder=2)


            ## POINTS FOR NODES AT EACH FRAME (TIME COLORED; HEAD BIGGER)
            node_marker = {"head": "o", "body": "o", "tail": "o"}
            size_map  = {"head": 6, "body": 4, "tail": 2}

            for node in ("head", "body", "tail"):
                anchor_x_means, anchor_y_means = get_means("anchor",  node)
                partner_x_means, partner_y_means = get_means("partner", node)
                for f in frames:
                    if f in anchor_x_means.index:
                        x, y = anchor_x_means.loc[f], anchor_y_means.loc[f]
                        if np.isfinite(x) and np.isfinite(y):
                            ax.scatter(x, y, s=size_map[node], marker=node_marker[node],
                                        color=anchor_cmap(norm_frames(f)), alpha=0.9, zorder=3)
                    
                    if f in partner_x_means.index:
                        x, y = partner_x_means.loc[f], partner_y_means.loc[f]
                        if np.isfinite(x) and np.isfinite(y):
                            ax.scatter(x, y, s=size_map[node], marker=node_marker[node],
                                        color=partner_cmap(norm_frames(f)), alpha=0.9, zorder=3)

            # 6) tidy the one big axis
            xlim = (-50, 100)
            ylim = (-10, 300)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            # ax.set_aspect("equal", adjustable="box")
            ax.set_title(cluster, fontsize=16, fontweight='bold')
            for ax in axes_ax2:
                ax.axis('off')



        ## AVERAGE CONTACT FRAMES 

        interaction_contact_summary = []

        for cluster_id in sorted(df[cluster_name].unique()):
            cluster_df = df[df[cluster_name] == cluster_id].copy()
            for inter_id in cluster_df["interaction_id"].unique():
                inter_df = cluster_df[cluster_df["interaction_id"] == inter_id]
                n_close = (inter_df["min_distance"] < 1).sum()
                interaction_contact_summary.append({
                    "cluster": cluster_id,
                    "interaction_id": inter_id,
                    "frames_below_1mm": n_close})

        df_interaction_contact = pd.DataFrame(interaction_contact_summary)

        mean_frames = (
            df_interaction_contact
            .groupby('cluster')['frames_below_1mm']
            .mean()
        )

        # 2) ensure index types match the deviation index (important for reindex)
        mean_frames.index = mean_frames.index.astype(int)
        order = [1,2,3,4,5,6,7,8,9,10,11,12, 13, 14, 15, 16]   # matches your dendrogram order
        mean_frames = mean_frames.reindex(order, fill_value=0)
        heat = mean_frames.to_numpy()[np.newaxis, :]


        colors = ["skyblue", "mediumseagreen", "darkgreen"]
        # build a sequential colormap from those
        my_cmap = LinearSegmentedColormap.from_list("greenblue_custom", colors)

        # 5) draw the heat “box” (single row)
        im = ax3.imshow(
            heat,
            aspect='auto',
            interpolation='nearest',
            vmin=0,  # anchor scale at 0
            cmap=my_cmap,
            # cmap='PuBuGn' 
        )

        ax3.set_yticklabels([])
        ax3.set_yticks([])
        ax3.tick_params(left=False)
        ax3.set_xlabel("")             # remove x-axis label
        ax3.set_xticklabels([])        # remove tick labels
        ax3.set_xticks([])             # remove ticks entirely
        ax3.tick_params(bottom=False)  # remove the small tick lines


        # optional: make the heat row compact and boxy
        for spine in ax3.spines.values():
            spine.set_visible(False)
        
        # === add colorbar OUTSIDE grid ===
        cbar = fig.colorbar(
            im,
            ax=[ax1, *axes_ax2, ax3],        # anchors to both axes (so it aligns with total figure height)
            fraction=0.01,         # width of colorbar relative to figure
            pad=0.01,              # horizontal gap between plots and colorbar
            location='right'       # move to right side
        )
        cbar.set_label('Average Contact Frames', rotation=270, labelpad=15)

        # fig.subplots_adjust(hspace=0.0)
        output = os.path.join(self.directory, "hierarchal_mean_trace_summary.png")
        plt.savefig(output, dpi=300, bbox_inches='tight')
        output = os.path.join(self.directory, "hierarchal_mean_trace_summary.pdf")
        plt.savefig(output, format='pdf', bbox_inches='tight')
        plt.close(fig)





    #### METHOD BARPLOTS: BARPLOTS OF CLUSTERS
    def barplots(self):

        df = self.clusters
        full_df = self.df
        cluster_name = self.cluster_name

        palette = {
            "G": "C0",
            "S": "C1",
            "GS": "mediumseagreen"
        }

        condition_order = ["G", "S", "GS"]


        df = (
            self.clusters
            .drop_duplicates(subset=['interaction_id'])
            .copy()
        )

        ## RAW COUNT BARPLOT

        counts = (
            df.groupby([cluster_name, 'condition'])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=condition_order, fill_value=0)
        )

        counts_reset = counts.reset_index().melt(
            id_vars=cluster_name,
            var_name='condition',
            value_name='count'
        )

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=counts_reset,
            x=cluster_name,
            y='count',
            hue='condition',
            hue_order=condition_order,
            palette=palette
        )
        plt.title("Count of G vs S vs GS per Cluster")
        plt.xlabel("Cluster ID")
        plt.ylabel("Count")
        plt.xticks(rotation=90)
        plt.tight_layout()
        path = os.path.join(self.directory, 'cluster_barplot.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

        ## PROPORTION BARPLOT

        proportions = counts.div(counts.sum(axis=0), axis=1)

        proportions_reset = proportions.copy()
        proportions_reset[cluster_name] = proportions_reset.index
        proportions_reset = proportions_reset.melt(
            id_vars=cluster_name,
            var_name='condition',
            value_name='proportion'
        )

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=proportions_reset,
            x=cluster_name,
            y='proportion',
            hue='condition',
            hue_order=condition_order,
            palette=palette
        )
        plt.title("Proportion of Interactions per Cluster")
        plt.xlabel("Cluster ID")
        plt.ylabel("Proportion")
        plt.xticks(rotation=90)
        plt.tight_layout()
        prop_path = os.path.join(self.directory, 'cluster_proportions.png')
        plt.savefig(prop_path, dpi=300, bbox_inches='tight')
        plt.close()

        ## AVERAGE PROPORTIONS BARPLOT PER VIDEO

        inter_per_video = (
            df[['file', 'condition', 'interaction_id', cluster_name]]
            .drop_duplicates(subset=['file', 'interaction_id'])
        )

        counts = (
            inter_per_video
            .groupby(['file', 'condition', cluster_name])
            .size()
            .reset_index(name='count')
        )

        totals = counts.groupby(['file', 'condition'])['count'].transform('sum')
        counts['proportion'] = counts['count'] / totals

        summary_df = (
            counts
            .set_index(['file', 'condition', cluster_name])['proportion']
            .unstack(fill_value=0)
            .stack()
            .reset_index(name='proportion')
        )

        summary_csv_path = os.path.join(self.directory, 'per_video_cluster_proportions_long.csv')
        summary_df.to_csv(summary_csv_path, index=False)

        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=summary_df,
            x=cluster_name,
            y='proportion',
            hue='condition',
            hue_order=condition_order,
            errorbar='sd',
            alpha=0.8,
            palette=palette
        )

        plt.title("Proportion of Clusters per Video")
        plt.xlabel("Cluster ID")
        plt.ylabel("Proportion")
        plt.xticks(rotation=90)
        plt.tight_layout()

        per_video_prop_path = os.path.join(self.directory, 'cluster_proportions_per_video.png')
        plt.savefig(per_video_prop_path, dpi=300, bbox_inches='tight')
        output = os.path.join(self.directory, 'cluster_proportions_per_video.pdf')
        plt.savefig(output, format='pdf', bbox_inches='tight')
        plt.close()

        ## PROPORTION OVER TIME BARPLOT

        interaction_starts = full_df[full_df['Normalized Frame'] == 0].copy()

        interaction_starts['time_bin'] = pd.cut(
            interaction_starts['Frame'],
            bins=np.arange(0, 3600 + 600, 600),
            labels=[f"{i*600}-{(i+1)*600}" for i in range(6)],
            right=False
        )

        counts = (
            interaction_starts
            .groupby(['file', 'condition', cluster_name, 'time_bin'])
            .size()
            .reset_index(name='count')
        )

        totals = counts.groupby(['file', 'condition', 'time_bin'])['count'].transform('sum')
        counts['proportion'] = counts['count'] / totals

        summary_df = (
            counts
            .set_index(['file', 'condition', cluster_name, 'time_bin'])['proportion']
            .unstack(fill_value=0)
            .stack()
            .reset_index(name='proportion')
        )

        summary_csv_path = os.path.join(self.directory, 'per_video_cluster_proportions_by_timebin.csv')
        summary_df.to_csv(summary_csv_path, index=False)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
        axes = axes.flatten()

        time_bins = summary_df['time_bin'].dropna().unique()

        for idx, tb in enumerate(sorted(time_bins, key=lambda x: int(str(x).split('-')[0]))):
            ax = axes[idx]
            sns.barplot(
                data=summary_df[summary_df['time_bin'] == tb],
                x=cluster_name,
                y='proportion',
                hue='condition',
                hue_order=condition_order,
                errorbar='sd',
                alpha=0.8,
                palette=palette,
                ax=ax
            )
            ax.set_title(f"Time {tb} sec")
            ax.set_xlabel("Cluster ID")
            ax.set_ylabel("Proportion")
            ax.set_ylim(0, None)
            ax.tick_params(axis='x', rotation=90)

        plt.tight_layout()
        timebin_prop_path = os.path.join(self.directory, 'cluster_proportions_by_timebin.png')
        plt.savefig(timebin_prop_path, dpi=300, bbox_inches='tight')
        plt.close()

        ## OBSERVED - EXPECTED DEVIATION PLOT FOR ALL 3 CONDITIONS

        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42

        cluster_counts = (
            df.groupby([cluster_name, 'condition'])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=condition_order, fill_value=0)
        )

        total_by_condition = cluster_counts.sum(axis=0)
        total_all = total_by_condition.sum()
        expected_fraction = total_by_condition / total_all

        observed_fraction = cluster_counts.div(cluster_counts.sum(axis=1), axis=0).fillna(0)

        deviation = observed_fraction - expected_fraction

        deviation_long = (
            deviation
            .reset_index()
            .melt(
                id_vars=cluster_name,
                var_name='condition',
                value_name='deviation'
            )
        )

        output = os.path.join(self.directory, 'deviation_observed_minus_expected_3conditions.csv')
        deviation_long.to_csv(output, index=False)

        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=deviation_long,
            x=cluster_name,
            y='deviation',
            hue='condition',
            hue_order=condition_order,
            palette=palette
        )

        plt.axhline(0, color='k', linestyle='--', linewidth=1)
        plt.title("Observed - Expected Fraction per Cluster")
        plt.ylabel("Observed - Expected Fraction")
        plt.xlabel("Cluster ID")
        plt.xticks(rotation=90)
        plt.tight_layout()

        path = os.path.join(self.directory, 'deviations_3conditions.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        path = os.path.join(self.directory, 'deviations_3conditions.pdf')
        plt.savefig(path, format="pdf", bbox_inches="tight", dpi=300, transparent=True)
        plt.close()


        ## OBSERVED - EXPECTED DEVIATION PLOT FOR G VS S ONLY

        gs_df = df[df['condition'].isin(['G', 'S'])].copy()
        gs_condition_order = ['G', 'S']
        gs_palette = {
            "G": "C0",
            "S": "C1"
        }

        gs_cluster_counts = (
            gs_df
            .groupby([cluster_name, 'condition'])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=gs_condition_order, fill_value=0)
        )

        # Expected fraction of each condition based on overall G/S totals
        gs_total_by_condition = gs_cluster_counts.sum(axis=0)
        gs_total_all = gs_total_by_condition.sum()
        gs_expected_fraction = gs_total_by_condition / gs_total_all

        # Observed fraction of G/S within each cluster
        gs_observed_fraction = (
            gs_cluster_counts
            .div(gs_cluster_counts.sum(axis=1), axis=0)
            .fillna(0)
        )

        # Observed - expected
        gs_deviation = gs_observed_fraction - gs_expected_fraction

        gs_deviation_long = (
            gs_deviation
            .reset_index()
            .melt(
                id_vars=cluster_name,
                var_name='condition',
                value_name='deviation'
            )
        )

        output = os.path.join(
            self.directory,
            'deviation_observed_minus_expected_G_vs_S.csv'
        )
        gs_deviation_long.to_csv(output, index=False)

        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=gs_deviation_long,
            x=cluster_name,
            y='deviation',
            hue='condition',
            hue_order=gs_condition_order,
            palette=gs_palette
        )

        plt.axhline(0, color='k', linestyle='--', linewidth=1)
        plt.title("Observed - Expected Fraction per Cluster: G vs S")
        plt.ylabel("Observed - Expected Fraction")
        plt.xlabel("Cluster ID")
        plt.xticks(rotation=90)
        plt.tight_layout()

        path = os.path.join(self.directory, 'deviations_G_vs_S.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')

        path = os.path.join(self.directory, 'deviations_G_vs_S.pdf')
        plt.savefig(
            path,
            format="pdf",
            bbox_inches="tight",
            dpi=300,
            transparent=True
        )
        plt.close()
        



    def barplot_deviation(self):

        df = self.clusters
        df_interaction = self.df
        cluster_name = self.cluster_name

        output = os.path.join(self.directory, "barplot_deviation")
        os.makedirs(output, exist_ok=True)

        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype']  = 42


        df = df[(df['condition'] == 'G') | (df['condition'] == 'S')].copy()
        df_interaction = df_interaction[
            (df_interaction['condition'] == 'G') | 
            (df_interaction['condition'] == 'S')
        ].copy()

        ### OBSERVED - EXPECTED DEVIATION 

        cluster_counts = (df.groupby([cluster_name, 'condition']).size().unstack(fill_value=0).reindex(columns=['G', 'S'], fill_value=0))  # count number per cluster per condition

        total_group = cluster_counts['G'].sum()
        total_iso   = cluster_counts['S'].sum()
        total_all   = total_group + total_iso

        expected_group = total_group / total_all   # e.g., ~0.56

        observed_group_frac = cluster_counts['G'] / (cluster_counts['G'] + cluster_counts['S']).replace({0: np.nan}) ## observed fraction
        observed_group_frac = observed_group_frac.fillna(0.0)

        deviation = observed_group_frac - expected_group ## expected fraction

        deviation_sorted = deviation.sort_values()
        colors = ['C1' if val < 0 else 'C0' for val in deviation_sorted.values]

        ## Binomial test per cluster
        results = []
        for cluster_id, row in cluster_counts.iterrows():
            k = row['G']
            n = row['G'] + row['S']
            if n > 0:
                p_exp = expected_group
                res = binomtest(k, n, p_exp, alternative='two-sided')
                results.append((cluster_id, res.pvalue))
            else:
                results.append((cluster_id, np.nan))

        pvals = pd.DataFrame(results, columns=['cluster_id', 'p_value'])

        ## correct for multple test - false positives
        pvals['p_adj'] = multipletests(pvals['p_value'], method='fdr_bh')[1]
        path = os.path.join(output, 'deviation_pvals.csv')
        pvals.to_csv(path, index=False, float_format='%.10f')


         ### CONTACT FRAME SUMMARY

        interaction_contact_summary = []

        for cluster_id in sorted(df_interaction[cluster_name].unique()):
            cluster_df = df_interaction[df_interaction[cluster_name] == cluster_id]
            for inter_id in cluster_df["interaction_id"].unique():
                inter_df = cluster_df[cluster_df["interaction_id"] == inter_id]
                n_close = (inter_df["min_distance"] < 1).sum()
                interaction_contact_summary.append({
                    "cluster": cluster_id,
                    "interaction_id": inter_id,
                    "frames_below_1mm": n_close
                })

        # Convert to DataFrame
        df_interaction_contact = pd.DataFrame(interaction_contact_summary)



        ## === TOP: DEVIATION BARPLOT WITH SIGNIFICANCE STARS ==
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[6, 0.3], hspace=0.05)

        ax1 = fig.add_subplot(gs[0])  # top: bar chart
        ax2 = fig.add_subplot(gs[1])  # middle: heatmap box

        x_labels = deviation_sorted.index.astype(str)
        x_pos = np.arange(len(x_labels))

        # Plot with Matplotlib's bar (since sns.barplot expects a DataFrame)
        ax1.bar(
            deviation_sorted.index.astype(str),
            deviation_sorted.values,
            color=colors,            # keep your C0/C1 mapping
            alpha=0.7,              # <— makes the fill lighter
            edgecolor='black',       # <— black border
            linewidth=1.5            # <— thicker border
        )

        # Add reference line
        ax1.axhline(0, color='k', linestyle='--', linewidth=1)

        # --- Annotate significance stars ---

        # Map adjusted p-values to cluster ids
        p_map = pvals.set_index('cluster_id')['p_adj']

        def stars(p):
            if p < 0.001: return '***'
            if p < 0.01:  return '**'
            if p < 0.05:  return '*'
            return ''

        # Vertical offset for labels
        ymin, ymax = ax1.get_ylim()
        dy = 0.015 * (ymax - ymin)

        # Annotate bars
        for i, cid in enumerate(deviation_sorted.index):
            p = p_map.get(cid, np.nan)
            if pd.notna(p):
                s = stars(p)
                if s:
                    y = deviation_sorted.loc[cid]
                    ax1.text(
                        i,
                        y + (dy if y >= 0 else -dy),
                        s,
                        ha='center',
                        va='bottom' if y >= 0 else 'top',
                        fontsize=10,
                        fontweight='bold',
                        color='black')
        
        ax1.set_title("Cluster Deviation from Expected", fontsize=16, fontweight='bold', pad=15)
        ax1.set_ylabel("Deviation from Expected")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels('')
        

        # === BOTTOM: 1×N heatmap strip of avg contact frames per cluster ===
        # 1) mean frames below 1mm per cluster
        mean_frames = (
            df_interaction_contact
            .groupby('cluster')['frames_below_1mm']
            .mean()
        )

        # 2) ensure index types match the deviation index (important for reindex)
        mean_frames.index = mean_frames.index.astype(deviation_sorted.index.dtype)

        # 3) align to the same x-order as the bar plot
        avg_contact_aligned = mean_frames.reindex(deviation_sorted.index, fill_value=0)

        # 4) reshape to 1×N for imshow
        heat = avg_contact_aligned.to_numpy()[np.newaxis, :]

        ax2.set_xlim(ax1.get_xlim())

        # define your color stops (dark → sea → cadet)
        colors = ["lightskyblue", "mediumseagreen", "darkgreen"]
        from matplotlib.colors import LinearSegmentedColormap
        # build a sequential colormap from those
        my_cmap = LinearSegmentedColormap.from_list("greenblue_custom", colors)

        # 5) draw the heat “box” (single row)
        im = ax2.imshow(
            heat,
            aspect='auto',
            interpolation='nearest',
            vmin=0,  # anchor scale at 0
            cmap=my_cmap,
            # cmap='PuBuGn' 
        )
        ax2.set_yticks([0])
        ax2.set_yticklabels(['Average Contact\nFrames'])
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(x_labels, fontweight='bold', fontsize=14)
        ax2.tick_params(axis='x', pad=15)

        # optional: make the heat row compact and boxy
        for spine in ax2.spines.values():
            spine.set_visible(False)
        
        # === add colorbar OUTSIDE grid ===
        cbar = fig.colorbar(
            im,
            ax=[ax1, ax2],        # anchors to both axes (so it aligns with total figure height)
            fraction=0.03,         # width of colorbar relative to figure
            pad=0.02,              # horizontal gap between plots and colorbar
            location='right'       # move to right side
        )
        cbar.set_label('Average Contact Frames', rotation=270, labelpad=15)


        plt.tight_layout()
        path = os.path.join(output, 'deviations.png')  
        plt.savefig(path, dpi=300, bbox_inches='tight')
        path = os.path.join(output, 'deviations.pdf')  
        plt.savefig(path, format="pdf", bbox_inches="tight", dpi=300, transparent=True)
        plt.close()
    






    def barplot_deviation_GS_social_experience(self):

        df = self.clusters.copy()
        df_interaction = self.df.copy()
        cluster_name = self.cluster_name

        output = os.path.join(self.directory, "barplot_deviation_GS_social_experience")
        os.makedirs(output, exist_ok=True)

        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42

        # Keep only GS condition
        df = df[df['condition'] == 'GS'].copy()
        df_interaction = df_interaction[df_interaction['condition'] == 'GS'].copy()

        if df.empty:
            print("No GS condition data found.")
            return

        def get_social_experience(pair):
            if pair is None or pd.isna(pair):
                return np.nan

            if isinstance(pair, str):
                pair = pair.replace("(", "").replace(")", "").replace(" ", "")
                id1, id2 = pair.split(",")
                id1 = int(id1)
                id2 = int(id2)
            else:
                id1, id2 = pair
                id1 = int(id1)
                id2 = int(id2)

            if id1 <= 4 and id2 <= 4:
                return "S-S"
            elif id1 >= 5 and id2 >= 5:
                return "G-G"
            else:
                return "G-S"

        pair_lookup = (
            df_interaction[['interaction_id', 'Interaction Pair']]
            .drop_duplicates(subset=['interaction_id'])
        )

        df = df.merge(
            pair_lookup,
            on='interaction_id',
            how='left'
        )

        df['social_experience'] = df['Interaction Pair'].apply(get_social_experience)
        df = df.dropna(subset=['social_experience'])

        social_order = ['S-S', 'G-S', 'G-G']

        cluster_counts = (
            df.groupby([cluster_name, 'social_experience'])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=social_order, fill_value=0)
        )

        cluster_counts = cluster_counts.loc[cluster_counts.sum(axis=1) > 0]


        print("\n=== SOCIAL EXPERIENCE COUNTS PER CLUSTER ===")
        print(cluster_counts)

        if cluster_counts.empty:
            print("No valid social experience data found.")
            return

        total_by_social = cluster_counts.sum(axis=0)
        total_all = total_by_social.sum()
        expected_fraction = total_by_social / total_all

        observed_fraction = (
            cluster_counts
            .div(cluster_counts.sum(axis=1), axis=0)
            .fillna(0)
        )

        deviation = observed_fraction - expected_fraction

        deviation_long = (
            deviation
            .reset_index()
            .melt(
                id_vars=cluster_name,
                var_name='social_experience',
                value_name='deviation'
            )
        )

        path = os.path.join(output, 'deviation_GS_social_experience.csv')
        deviation_long.to_csv(path, index=False)

        # Chi-square test per cluster
        results = []

        for cluster_id, row in cluster_counts.iterrows():
            observed = row[social_order].values
            expected = expected_fraction.values * observed.sum()

            try:
                stat, p = chisquare(f_obs=observed, f_exp=expected)
            except ValueError:
                p = np.nan

            results.append((cluster_id, p))

        pvals = pd.DataFrame(results, columns=['cluster_id', 'p_value'])
        pvals['p_adj'] = np.nan

        valid = pvals['p_value'].notna()

        if valid.sum() > 0:
            pvals.loc[valid, 'p_adj'] = multipletests(
                pvals.loc[valid, 'p_value'],
                method='fdr_bh'
            )[1]

        path = os.path.join(output, 'deviation_pvals_GS_social_experience.csv')
        pvals.to_csv(path, index=False, float_format='%.10f')

        axis_angles = {
            'G-S': np.deg2rad(90),
            'S-S': np.deg2rad(210),
            'G-G': np.deg2rad(330)
        }

        axis_vectors = {
            key: np.array([np.cos(angle), np.sin(angle)])
            for key, angle in axis_angles.items()
        }

        deviation_axis_df = deviation.copy()
        deviation_axis_df['axis_x'] = 0.0
        deviation_axis_df['axis_y'] = 0.0

        for social in social_order:
            deviation_axis_df['axis_x'] += deviation_axis_df[social] * axis_vectors[social][0]
            deviation_axis_df['axis_y'] += deviation_axis_df[social] * axis_vectors[social][1]

        deviation_axis_df = deviation_axis_df.reset_index()
        deviation_axis_df['total_interactions'] = deviation_axis_df[cluster_name].map(cluster_counts.sum(axis=1))

        deviation_axis_df.to_csv(
            os.path.join(output, 'GS_social_experience_three_axis_deviation_positions.csv'),
            index=False
        )

        max_dev = float(np.nanmax(np.abs(deviation[social_order].to_numpy())))
        if max_dev == 0:
            max_dev = 1e-6

        axis_limit = max_dev * 1.35

        fig, ax = plt.subplots(figsize=(8, 8))

        for social in ['G-S', 'S-S', 'G-G']:
            vec = axis_vectors[social]
            ax.arrow(
                0,
                0,
                axis_limit * vec[0],
                axis_limit * vec[1],
                head_width=axis_limit * 0.06,
                head_length=axis_limit * 0.08,
                length_includes_head=True,
                color='0.55',
                linewidth=2.5,
                zorder=1
            )
            ax.plot(
                [0, axis_limit * 0.72 * vec[0]],
                [0, axis_limit * 0.72 * vec[1]],
                color='0.55',
                linewidth=1.2,
                zorder=1
            )
            ax.text(
                axis_limit * 1.12 * vec[0],
                axis_limit * 1.12 * vec[1],
                f"+ {social}",
                ha='center',
                va='center',
                fontsize=12,
                fontweight='bold'
            )

        # ax.scatter(0, 0, marker='*', s=220, color='red', edgecolor='black', linewidth=0.7, zorder=4)
        # ax.text(0, -axis_limit * 0.11, 'expected', ha='center', va='top', fontsize=9, color='red')

        scatter = ax.scatter(
            deviation_axis_df['axis_x'],
            deviation_axis_df['axis_y'],
            s=100,
            c=deviation_axis_df[cluster_name],
            cmap='viridis',
            edgecolor='white',
            linewidth=0.6,
            zorder=3
        )

        for _, row in deviation_axis_df.iterrows():
            ax.plot(
                [0, row['axis_x']],
                [0, row['axis_y']],
                color='0.8',
                linewidth=0.8,
                zorder=2
            )
            ax.text(
                row['axis_x'],
                row['axis_y'] + axis_limit * 0.035,
                str(row[cluster_name]),
                ha='center',
                va='bottom',
                fontsize=8
            )

        cbar = fig.colorbar(scatter, ax=ax, fraction=0.045, pad=0.04)
        cbar.set_label('Cluster ID')

        ax.set_title("GS Condition: three-axis social-experience deviation")
        ax.set_aspect('equal')
        ax.set_xlim(-axis_limit * 1.25, axis_limit * 1.25)
        ax.set_ylim(-axis_limit * 1.25, axis_limit * 1.25)
        ax.axis('off')

        plt.tight_layout()

        path = os.path.join(output, 'GS_social_experience_three_axis_deviation_plot.pdf')
        plt.savefig(path, format='pdf', bbox_inches='tight', dpi=300, transparent=True)


        plt.close()

        palette = {
            'S-S': 'C1',
            'G-S': 'mediumseagreen',
            'G-G': 'C0'
        }

        cluster_order = list(deviation.index)

        plt.figure(figsize=(12, 6))

        ax = sns.barplot(
            data=deviation_long,
            x=cluster_name,
            y='deviation',
            hue='social_experience',
            hue_order=social_order,
            order=cluster_order,
            palette=palette
        )

        # Add significance stars per cluster
        p_map = pvals.set_index('cluster_id')['p_adj']

        def stars(p):
            if p < 0.001:
                return '***'
            if p < 0.01:
                return '**'
            if p < 0.05:
                return '*'
            return ''

        ymin, ymax = ax.get_ylim()
        dy = 0.03 * (ymax - ymin)

        for i, cluster_id in enumerate(cluster_order):
            p = p_map.get(cluster_id, np.nan)

            if pd.notna(p):
                s = stars(p)

                if s:
                    y = deviation.loc[cluster_id, social_order].max()

                    ax.text(
                        i,
                        y + dy,
                        s,
                        ha='center',
                        va='bottom',
                        fontsize=12,
                        fontweight='bold',
                        color='black'
                    )

        ax.axhline(0, color='k', linestyle='--', linewidth=1)
        ax.set_title("GS Condition: Observed - Expected Social Experience per Cluster")
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("Observed - Expected Fraction")
        ax.tick_params(axis='x', rotation=90)

        plt.tight_layout()


        path = os.path.join(output, 'deviations_GS_social_experience.pdf')
        plt.savefig(path, format='pdf', bbox_inches='tight', dpi=300, transparent=True)

        plt.close()
    







    def GS_pairwise_social_experience_deviation_stats(self):

        df = self.clusters.copy()
        df_interaction = self.df.copy()
        cluster_name = self.cluster_name

        output = os.path.join(self.directory, "GS_pairwise_social_experience_deviation_stats")
        os.makedirs(output, exist_ok=True)

        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42

        df = df[df['condition'] == 'GS'].copy()
        df_interaction = df_interaction[df_interaction['condition'] == 'GS'].copy()

        if df.empty:
            print("No GS condition data found.")
            return

        social_order = ['S-S', 'G-S', 'G-G']
        possible_pairs = {
            'S-S': 10,
            'G-S': 25,
            'G-G': 10
        }

        def get_social_experience(pair):
            if pair is None or pd.isna(pair):
                return np.nan

            if isinstance(pair, str):
                pair = pair.replace("(", "").replace(")", "").replace(" ", "")
                id1, id2 = pair.split(",")
                id1 = int(id1)
                id2 = int(id2)
            else:
                id1, id2 = pair
                id1 = int(id1)
                id2 = int(id2)

            if id1 <= 4 and id2 <= 4:
                return "S-S"
            elif id1 >= 5 and id2 >= 5:
                return "G-G"
            else:
                return "G-S"

        pair_lookup = (
            df_interaction[['file', 'interaction_id', 'Interaction Pair']]
            .drop_duplicates(subset=['file', 'interaction_id'])
            .copy()
        )

        pair_lookup['social_experience'] = pair_lookup['Interaction Pair'].apply(get_social_experience)
        pair_lookup = pair_lookup.dropna(subset=['social_experience'])

        df = df.merge(
            pair_lookup[['file', 'interaction_id', 'social_experience']],
            on=['file', 'interaction_id'],
            how='left'
        )
        df = df.dropna(subset=['social_experience'])

        inter_per_video = (
            df[['file', 'interaction_id', cluster_name, 'social_experience']]
            .drop_duplicates(subset=['file', 'interaction_id'])
            .copy()
        )

        if inter_per_video.empty:
            print("No valid GS social-experience interactions found.")
            return

        counts = (
            inter_per_video
            .groupby(['file', cluster_name, 'social_experience'])
            .size()
            .reset_index(name='count')
        )

        all_files = sorted(inter_per_video['file'].dropna().unique())
        all_clusters = sorted(inter_per_video[cluster_name].dropna().unique())

        full_index = pd.MultiIndex.from_product(
            [all_files, all_clusters, social_order],
            names=['file', cluster_name, 'social_experience']
        )

        counts = (
            counts
            .set_index(['file', cluster_name, 'social_experience'])
            .reindex(full_index, fill_value=0)
            .reset_index()
        )

        counts['possible_pairs'] = counts['social_experience'].map(possible_pairs)
        counts['rate_per_possible_pair'] = counts['count'] / counts['possible_pairs']

        counts.to_csv(
            os.path.join(output, 'GS_pairwise_social_experience_rates_per_video.csv'),
            index=False
        )

        pairwise_tests = [
            ('G-G', 'G-S'),
            ('S-S', 'G-S'),
            ('G-G', 'S-S')
        ]

        results = []

        for cluster_id in all_clusters:
            sub = counts[counts[cluster_name] == cluster_id]

            wide = (
                sub
                .pivot(index='file', columns='social_experience', values='rate_per_possible_pair')
                .reindex(columns=social_order)
                .fillna(0)
            )

            for group_a, group_b in pairwise_tests:
                a = wide[group_a]
                b = wide[group_b]
                diff = a - b

                if len(diff) >= 2 and not np.allclose(diff, 0):
                    stat, p = wilcoxon(a, b, alternative='two-sided')
                elif len(diff) >= 2 and np.allclose(diff, 0):
                    stat, p = 0, 1.0
                else:
                    stat, p = np.nan, np.nan

                results.append({
                    cluster_name: cluster_id,
                    'comparison': f'{group_a}_vs_{group_b}',
                    'group_a': group_a,
                    'group_b': group_b,
                    'mean_rate_group_a': a.mean(),
                    'mean_rate_group_b': b.mean(),
                    'mean_diff_group_a_minus_group_b': diff.mean(),
                    'median_diff_group_a_minus_group_b': diff.median(),
                    'W': stat,
                    'p': p,
                    'n_videos': len(diff)
                })

        stats_df = pd.DataFrame(results)
        stats_df['p_adj'] = np.nan

        valid = stats_df['p'].notna()
        if valid.any():
            stats_df.loc[valid, 'p_adj'] = multipletests(
                stats_df.loc[valid, 'p'],
                method='fdr_bh'
            )[1]

        stats_df.to_csv(
            os.path.join(output, 'GS_pairwise_social_experience_rate_stats.csv'),
            index=False,
            float_format='%.10f'
        )

        summary = (
            counts
            .groupby([cluster_name, 'social_experience'])
            .agg(
                mean_count=('count', 'mean'),
                sd_count=('count', 'std'),
                mean_rate_per_possible_pair=('rate_per_possible_pair', 'mean'),
                sd_rate_per_possible_pair=('rate_per_possible_pair', 'std'),
                possible_pairs=('possible_pairs', 'first')
            )
            .reset_index()
        )

        summary.to_csv(
            os.path.join(output, 'GS_pairwise_social_experience_rate_summary.csv'),
            index=False
        )

        plt.figure(figsize=(12, 6))

        ax = sns.barplot(
            data=counts,
            x=cluster_name,
            y='rate_per_possible_pair',
            hue='social_experience',
            hue_order=social_order,
            order=all_clusters,
            errorbar='sd',
            palette={
                'S-S': 'C1',
                'G-S': 'mediumseagreen',
                'G-G': 'C0'
            }
        )

        ax.set_title("GS condition: cluster interactions per possible pair")
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("Interaction count per possible pair")
        ax.tick_params(axis='x', rotation=90)

        plt.tight_layout()

        path = os.path.join(output, 'GS_pairwise_social_experience_rates_per_cluster.pdf')
        plt.savefig(path, format='pdf', bbox_inches='tight', dpi=300, transparent=True)

        path = os.path.join(output, 'GS_pairwise_social_experience_rates_per_cluster.png')
        plt.savefig(path, bbox_inches='tight', dpi=300)

        plt.close()

        print("Saved GS pairwise social-experience deviation stats.")




    def GS_social_experience_cluster_proportions_per_video(self):

        df = self.clusters.copy()
        df_interaction = self.df.copy()
        cluster_name = self.cluster_name

        output = os.path.join(self.directory, "GS_social_experience_cluster_proportions_per_video")
        os.makedirs(output, exist_ok=True)

        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42

        # Keep only GS condition
        df = df[df['condition'] == 'GS'].copy()
        df_interaction = df_interaction[df_interaction['condition'] == 'GS'].copy()

        if df.empty:
            print("No GS condition data found.")
            return

        def get_social_experience(pair):
            if pair is None or pd.isna(pair):
                return np.nan

            if isinstance(pair, str):
                pair = pair.replace("(", "").replace(")", "").replace(" ", "")
                id1, id2 = pair.split(",")
                id1 = int(id1)
                id2 = int(id2)
            else:
                id1, id2 = pair
                id1 = int(id1)
                id2 = int(id2)

            if id1 <= 4 and id2 <= 4:
                return "S-S"
            elif id1 >= 5 and id2 >= 5:
                return "G-G"
            else:
                return "G-S"

        # Bring Interaction Pair into self.clusters using interaction_id
        pair_lookup = (
            df_interaction[['interaction_id', 'Interaction Pair']]
            .drop_duplicates(subset=['interaction_id'])
        )

        df = df.merge(
            pair_lookup,
            on='interaction_id',
            how='left'
        )

        df['social_experience'] = df['Interaction Pair'].apply(get_social_experience)

        df = df.dropna(subset=['social_experience'])

        social_order = ['S-S', 'G-S', 'G-G']

        palette = {
            'S-S': 'C1',
            'G-S': 'mediumseagreen',
            'G-G': 'C0'
        }

        # One row per interaction
        inter_per_video = (
            df[['file', 'social_experience', 'interaction_id', cluster_name]]
            .drop_duplicates(subset=['file', 'interaction_id'])
        )

        counts = (
            inter_per_video
            .groupby(['file', 'social_experience', cluster_name])
            .size()
            .reset_index(name='count')
        )

        totals = counts.groupby(['file', 'social_experience'])['count'].transform('sum')
        counts['proportion'] = counts['count'] / totals

        summary_df = (
            counts
            .set_index(['file', 'social_experience', cluster_name])['proportion']
            .unstack(fill_value=0)
            .stack()
            .reset_index(name='proportion')
        )

        summary_csv_path = os.path.join(
            output,
            'GS_social_experience_cluster_proportions_per_video.csv'
        )
        summary_df.to_csv(summary_csv_path, index=False)

        plt.figure(figsize=(12, 6))

        sns.barplot(
            data=summary_df,
            x=cluster_name,
            y='proportion',
            hue='social_experience',
            hue_order=social_order,
            errorbar='sd',
            alpha=0.8,
            palette=palette
        )

        plt.title("GS Condition: Proportion of Clusters per Video by Social Experience")
        plt.xlabel("Cluster ID")
        plt.ylabel("Proportion")
        plt.xticks(rotation=90)
        plt.tight_layout()

        path = os.path.join(output, 'GS_social_experience_cluster_proportions_per_video.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')

        path = os.path.join(output, 'GS_social_experience_cluster_proportions_per_video.pdf')
        plt.savefig(path, format='pdf', bbox_inches='tight')

        plt.close()
    


    def grouped_clusters(self):

        df = self.df.copy()
        cluster_name = self.cluster_name

        output = os.path.join(self.directory, "grouped_clusters")
        os.makedirs(output, exist_ok=True)

        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42

        # One row per interaction.
        df = df[np.isclose(df["Normalized Frame"], 0)].copy()
        df = df.drop_duplicates(subset=["file", "interaction_id"]).copy()

        cluster_groups = {
            2: "aversive",
            5: "aversive",
            8: "uninterested",
            10: "uninterested",
            11: "uninterested",
            12: "uninterested",
            4: "tailing",
            7: "tailing",
            3: "head-head",
            9: "head-head",
            1: "general",
            6: "general"
        }

        group_order = ["aversive", "uninterested", "tailing", "head-head", "general"]
        group_palette = {
            "aversive": "#e60026",
            "uninterested": "#f28e2b",
            "tailing": "#1252b0",
            "head-head": "#28663a",
            "general": "#bdbdbd"
        }

        def grouped_cluster_label(cluster_id):
            if pd.isna(cluster_id):
                return np.nan
            try:
                return cluster_groups.get(int(float(cluster_id)), np.nan)
            except (TypeError, ValueError):
                return np.nan

        def parse_pair(pair):
            if pair is None:
                return None

            if isinstance(pair, str):
                if pair.strip() == "" or pair.lower() == "nan":
                    return None
                pair = pair.replace("(", "").replace(")", "").replace(" ", "")
                id1, id2 = pair.split(",")
                return int(id1), int(id2)

            try:
                if pd.isna(pair):
                    return None
            except ValueError:
                pass

            id1, id2 = pair
            return int(id1), int(id2)

        def get_gs_pair_type(pair):
            parsed = parse_pair(pair)
            if parsed is None:
                return np.nan

            id1, id2 = parsed

            if id1 <= 4 and id2 <= 4:
                return "S-S"
            if id1 >= 5 and id2 >= 5:
                return "G-G"
            return "G-S"

        def make_proportion_table(data, x_col, x_order):
            counts = (
                data
                .groupby([x_col, "grouped_cluster"])
                .size()
                .reset_index(name="count")
            )

            full_index = pd.MultiIndex.from_product(
                [x_order, group_order],
                names=[x_col, "grouped_cluster"]
            )

            counts = (
                counts
                .set_index([x_col, "grouped_cluster"])
                .reindex(full_index, fill_value=0)
                .reset_index()
            )

            counts["total"] = counts.groupby(x_col)["count"].transform("sum")
            counts["proportion"] = counts["count"] / counts["total"]
            counts["proportion"] = counts["proportion"].fillna(0)

            return counts

        def plot_stacked_proportions(prop_df, x_col, x_order, title, filename_stem):
            fig, ax = plt.subplots(figsize=(6, 7))
            bottoms = np.zeros(len(x_order))

            for group in group_order:
                values = (
                    prop_df[prop_df["grouped_cluster"] == group]
                    .set_index(x_col)
                    .reindex(x_order)["proportion"]
                    .fillna(0)
                    .to_numpy()
                )

                ax.bar(
                    x_order,
                    values,
                    bottom=bottoms,
                    color=group_palette[group],
                    edgecolor="white",
                    linewidth=1.2,
                    label=group
                )
                bottoms += values

            ax.set_ylim(0, 1)
            ax.set_ylabel("Proportion")
            ax.set_xlabel("")
            ax.set_title(title)
            ax.legend(title="Grouped cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="x", labelsize=14)
            ax.tick_params(axis="y", labelsize=14)

            for label in ax.get_xticklabels():
                label.set_fontweight("bold")

            plt.tight_layout()
            plt.savefig(
                os.path.join(output, f"{filename_stem}.png"),
                dpi=300,
                bbox_inches="tight"
            )
            plt.savefig(
                os.path.join(output, f"{filename_stem}.pdf"),
                format="pdf",
                dpi=300,
                bbox_inches="tight"
            )
            plt.close(fig)

        df["grouped_cluster"] = df[cluster_name].apply(grouped_cluster_label)
        df = df.dropna(subset=["condition", "grouped_cluster"])

        condition_order = ["G", "S", "GS"]
        condition_df = df[df["condition"].isin(condition_order)].copy()

        if condition_df.empty:
            print("No G, S or GS interactions found for grouped cluster proportions.")
        else:
            condition_props = make_proportion_table(
                condition_df,
                "condition",
                condition_order
            )
            condition_props.to_csv(
                os.path.join(output, "grouped_cluster_proportions_by_condition.csv"),
                index=False
            )
            plot_stacked_proportions(
                condition_props,
                "condition",
                condition_order,
                "Grouped cluster proportions by condition",
                "grouped_cluster_proportions_by_condition"
            )

        gs_df = df[df["condition"] == "GS"].copy()
        gs_df["GS_pair_type"] = gs_df["Interaction Pair"].apply(get_gs_pair_type)
        gs_df = gs_df.dropna(subset=["GS_pair_type"])

        gs_pair_order = ["G-G", "S-S", "G-S"]
        gs_df = gs_df[gs_df["GS_pair_type"].isin(gs_pair_order)].copy()

        if gs_df.empty:
            print("No GS interactions found for G-G, S-S and G-S grouped cluster proportions.")
        else:
            gs_props = make_proportion_table(
                gs_df,
                "GS_pair_type",
                gs_pair_order
            )
            gs_props.to_csv(
                os.path.join(output, "grouped_cluster_proportions_within_GS.csv"),
                index=False
            )
            plot_stacked_proportions(
                gs_props,
                "GS_pair_type",
                gs_pair_order,
                "GS grouped cluster proportions by pair type",
                "grouped_cluster_proportions_within_GS"
            )

        print(f"Saved grouped cluster proportion plots to: {output}")


        #### METHOD GRID_VIDEOS: GENERATE GRID VIDEOS OF INTERACTION CLUSTERS
    def grid_videos(self):

        df = self.df
        cluster_name = self.cluster_name 
        video_path = self.video_path 

        grid_video_dir = os.path.join(self.directory, "grid_videos")
        os.makedirs(grid_video_dir, exist_ok=True)

        frames_per_clip = 20
        dot_radius = 3
        dot_thickness = -1  # Filled
        fps = 3
        crop_size = 400
        grid_cols = 6
        grid_rows = 4
        clips_per_cluster = grid_cols * grid_rows

        # === TRACK VALID CLIPS ===
        cluster_to_interactions = {}

        for cluster_id in sorted(df[cluster_name].unique()):
            cluster_df = df[df[cluster_name] == cluster_id].copy()
            print("Cluster:", cluster_id)
            print(cluster_df["condition"].value_counts())

            unique_ids = cluster_df['interaction_id'].unique()

            if len(unique_ids) < clips_per_cluster:
                print(f"⚠️ Skipping cluster {cluster_id} (only {len(unique_ids)} interactions)")
                continue

            chosen_ids = sample(list(unique_ids), clips_per_cluster)
            interaction_clips = []
            final_ids = []

            for inter_id in chosen_ids:
                inter_df = cluster_df[cluster_df['interaction_id'] == inter_id].sort_values("Frame")
                start_frame = inter_df["Frame"].iloc[0]
                end_frame = start_frame + frames_per_clip
                clip_df = inter_df[(inter_df["Frame"] >= start_frame) & (inter_df["Frame"] < end_frame)]

                if len(clip_df) < frames_per_clip:
                    continue

                video_file = inter_df['file'].iloc[0]
                print(f"📼 Interaction {inter_id} comes from file {video_file}")
                full_video_path = os.path.join(video_path, video_file)

                if not os.path.exists(full_video_path):
                    print(f"⚠️ Missing video: {full_video_path}")
                    continue

                # Use midpoint at Normalized Frame = 0
                center_frame = inter_df[inter_df["Normalized Frame"] == 0]
                if center_frame.empty:
                    print(f"⚠️ No center frame for {inter_id}")
                    continue

                row_center = center_frame.iloc[0]
                cx = int((row_center['Track_1 x_body'] + row_center['Track_2 x_body']) / 2)
                cy = int((row_center['Track_1 y_body'] + row_center['Track_2 y_body']) / 2)

                # Crop logic
                def safe_crop(frame, cx, cy, crop_size):
                    h, w = frame.shape[:2]
                    half = crop_size // 2
                    x_start, y_start = cx - half, cy - half
                    x_end, y_end = cx + half, cy + half
                    cropped = np.zeros((crop_size, crop_size, 3), dtype=frame.dtype)

                    x1, y1 = max(0, x_start), max(0, y_start)
                    x2, y2 = min(w, x_end), min(h, y_end)
                    dx1, dy1 = x1 - x_start, y1 - y_start
                    dx2, dy2 = dx1 + (x2 - x1), dy1 + (y2 - y1)
                    cropped[dy1:dy2, dx1:dx2] = frame[y1:y2, x1:x2]
                    return cropped

                # Read and annotate frames
                cap = cv2.VideoCapture(full_video_path)
                clip_frames = []
                for _, row in clip_df.iterrows():
                    frame_idx = int(row['Frame'])
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    x1, y1 = int(row['Track_1 x_body']), int(row['Track_1 y_body'])
                    x2, y2 = int(row['Track_2 x_body']), int(row['Track_2 y_body'])

                    cv2.circle(frame, (x1, y1), dot_radius, (0, 0, 255), dot_thickness)
                    cv2.circle(frame, (x2, y2), dot_radius, (255, 0, 0), dot_thickness)

                    cropped = safe_crop(frame, cx, cy, crop_size)
                    clip_frames.append(cropped)

                cap.release()

                if len(clip_frames) == frames_per_clip:
                    interaction_clips.append(clip_frames)
                    final_ids.append(inter_id)

            if len(interaction_clips) < clips_per_cluster:
                print(f"⚠️ Not enough good clips for cluster {cluster_id}")
                continue

            cluster_to_interactions[cluster_id] = final_ids

            # === Create grid video ===
            h, w = interaction_clips[0][0].shape[:2]
            grid_frames = []
            for i in range(frames_per_clip):
                grid_rows_frames = []
                for row_idx in range(grid_rows):
                    start = row_idx * grid_cols
                    end = start + grid_cols
                    grid_rows_frames.append(
                        np.hstack([interaction_clips[j][i] for j in range(start, end)])
                    )
                grid_frame = np.vstack(grid_rows_frames)
                grid_frames.append(grid_frame)

            
            output_path = os.path.join(grid_video_dir, f"cluster_{cluster_id}.mp4")

            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w * grid_cols, h * grid_rows))

            for frame in grid_frames:
                out.write(frame)
            out.release()

            print(f"✅ Saved grid video for cluster {cluster_id} → {output_path}")


        # === SAVE CSV OF INTERACTIONS INCLUDED IN GRID VIDEOS ===
        mapping_records = []
        for cluster, interactions in cluster_to_interactions.items():
            for inter_id in interactions:
                video_file = df[df['interaction_id'] == inter_id]['file'].iloc[0]
                mapping_records.append({
                    'cluster': cluster,
                    'interaction_id': inter_id,
                    'video_file': video_file
                })

        mapping_df = pd.DataFrame(mapping_records)
        mapping_path = os.path.join(grid_video_dir, 'gridmovies_interactions.csv')
        mapping_df.to_csv(mapping_path, index=False)


        print('grid videos generated') 

    
    #### METHOD SUMMARY_ANCHOR_PARTNER: SUMMARY QUANTIFICATIONS ANCHOR/PARTNER
    def summary_anchor_partner(self):

        df = self.df
        cluster_name = self.cluster_name 

        cluster_ids = sorted(df[cluster_name].unique())
        n_clusters = len(cluster_ids)
        n_rows = 13  # number of summary plots (trajectory, speed, accel, angle, etc.)

        # Create summary canvas
        # fig_ap, axes_ap = plt.subplots(n_rows, n_clusters, figsize=(n_clusters * 4, n_rows * 2))

        # width per column and height per "unit"
        width_per_col  = 4    # you already had n_clusters*4
        height_per_unit = 1.5

        # Row 0 gets 3 units, rows 1–5 get 1 each → total units = 3 + 5*1 = 8
        height_ratios = [3] + [1]*(n_rows-1) ## want mean trajectory to get 3 times the space as the other rows 
        total_units   = sum(height_ratios)          # = 8
        fig_w = n_clusters * width_per_col          # unchanged
        fig_h = total_units * height_per_unit       # 8 * 1.5 = 12"

        fig_ap, axes_ap = plt.subplots(
        n_rows,
        n_clusters,
        figsize=(fig_w, fig_h),
        gridspec_kw={'height_ratios': height_ratios},
        constrained_layout=True
    )

        if n_clusters == 1:
            axes_ap = axes_ap.reshape(n_rows, 1)

        # Mark all as invisible initially
        for ax in axes_ap.flatten():
            ax.set_visible(False)

        row_labels = [
            "Mean Trajectory",           # 0
            "Speed",                     # 1
            "Acceleration",              # 2
            "Heading Angle",             # 3
            "Heading Angle Change",      # 4
            "Approach Angle",            # 5
            "Approach Angle Change",     # 6
            "Distance Travelled",        # 7
            "Minimum Distance",          # 8
            "Interaction Type",          # 9
            "Initial Contact",       # 10  <-- new
            "Predominant Contact",   # 11  <-- new
            "Contact Frames <1mm"        # 12  (moved down from 10)
        ]


        for i, label in enumerate(row_labels):
            ax_label = axes_ap[i, 0]  # first column of each row
            ax_label.set_ylabel(label, fontsize=10, rotation=0, labelpad=40, va='center')


        df['anchor_distance'] = df.groupby('interaction_id').apply(
        lambda x: np.sqrt((x['anchor x_body'].diff()**2 + x['anchor y_body'].diff()**2))).reset_index(level=0, drop=True)

        df['partner_distance'] =  df.groupby('interaction_id').apply(
        lambda x: np.sqrt((x['partner x_body'].diff()**2 + x['partner y_body'].diff()**2))).reset_index(level=0, drop=True)
        

        for column, cluster_id in enumerate(cluster_ids):
            cluster_df = df[df[cluster_name] == cluster_id]

            ## 0. MEAN TRAJECTORIES

            ax0 = axes_ap[0, column]
            grouped = cluster_df.groupby("Normalized Frame")

            t1_x = grouped["anchor x_body"].mean()
            t1_y = grouped["anchor y_body"].mean()
            t2_x = grouped["partner x_body"].mean()
            t2_y = grouped["partner y_body"].mean()

            t1_x_std = grouped["anchor x_body"].std()
            t1_y_std = grouped["anchor y_body"].std()
            t2_x_std = grouped["partner x_body"].std()
            t2_y_std = grouped["partner y_body"].std()

                    # Combine into a DataFrame
            sd_summary = pd.DataFrame({
                "Normalized Frame": t1_y_std.index,
                "t1_y_std": t1_y_std.values,
                "t2_y_std": t2_y_std.values,
                "t1_x_std": t1_x_std.values,
                "t2_x_std": t2_x_std.values,
            })

            # Save to CSV
            # sd_summary.to_csv(os.path.join(output_dir, "std_trajectory_summary.csv"), index=False)


            ax0.plot(t1_x, t1_y, label="Anchor", color="blue")
            ax0.plot(t2_x, t2_y, label="Partner", color="orange")

            ax0.scatter(t1_x.iloc[0], t1_y.iloc[0], color="blue", marker="o", label="Anchor Start")
            ax0.scatter(t2_x.iloc[0], t2_y.iloc[0], color="orange", marker="o", label="Partner Start")

                # error bars in X and Y
            # ax0.errorbar(
            #     t1_x, t1_y,
            #     xerr=t1_x_std, yerr=t1_y_std,
            #     fmt="none", ecolor="blue", alpha=0.3, label="Anchor ±1 SD"
            # )
            # ax0.errorbar(
            #     t2_x, t2_y,
            #     xerr=t2_x_std, yerr=t2_y_std,
            #     fmt="none", ecolor="orange", alpha=0.3, label="Partner ±1 SD"
            # )

            ax0.errorbar(
                    t1_x.values, t1_y.values,
                    xerr=t1_x_std.values, yerr=t1_y_std.values,
                    fmt="none", ecolor="blue", alpha=0.3, label="Anchor ±1 SD"
                )
            
            ax0.errorbar(
                    t2_x.values, t2_y.values,
                    xerr=t2_x_std.values, yerr=t2_y_std.values,
                    fmt="none", ecolor="orange", alpha=0.3, label="Partner ±1 SD"
                )



            # ax0.set_xticks([])
            # ax_sum.set_yticks([])
            ax0.set_aspect('equal', 'box')
            ax0.set_title(f"Cluster {cluster_id}", fontsize=8)
            ax0.set_visible(True)



            ## 1. SPEED
            ax1 = axes_ap[1, column]

            sns.lineplot(data=cluster_df, x='Normalized Frame', y='anchor_speed_body', label='Anchor', errorbar=('ci', 95), color='blue', ax=ax1)
            sns.lineplot(data=cluster_df, x='Normalized Frame', y='partner_speed_body', label='Partner', errorbar=('ci', 95), color='orange', ax=ax1)

            ax1.axvline(0, color="gray", ls="--", lw=0.5)
            ax1.set_ylim(0, 2)
            ax1.set_xticks([])
            # ax1.set_yticks([])
            ax1.set_visible(True)

            ## 2. ACCELERATION
            ax2 = axes_ap[2, column]

            sns.lineplot(data=cluster_df, x='Normalized Frame', y='anchor_acceleration_body', label='Anchor', errorbar=('ci', 95), color='blue', ax=ax2)
            sns.lineplot(data=cluster_df, x='Normalized Frame', y='partner_acceleration_body', label='Partner', errorbar=('ci', 95), color='orange', ax=ax2)

            ax2.axvline(0, color="gray", ls="--", lw=0.5)
            ax2.set_ylim(-1, 1)
            ax2.set_xticks([])
            # ax1.set_yticks([])
            ax2.set_visible(True)

            ## 3. HEADING ANGLE
            ax3 = axes_ap[3, column]

            sns.lineplot(data=cluster_df, x='Normalized Frame', y='anchor_angle', label='Anchor', errorbar=('ci', 95), color='blue', ax=ax3)
            sns.lineplot(data=cluster_df, x='Normalized Frame', y='partner_angle', label='Partner', errorbar=('ci', 95), color='orange', ax=ax3)

            ax3.axvline(0, color="gray", ls="--", lw=0.5)
            ax3.set_ylim(0, 180)
            ax3.set_xticks([])
            # ax1.set_yticks([])
            ax3.set_visible(True)

            ## 4. HEADING ANGLE CHANGE
            ax4 = axes_ap[4, column]

            sns.lineplot(data=cluster_df, x='Normalized Frame', y='anchor_heading_angle_change', label='Anchor', errorbar=('ci', 95), color='blue', ax=ax4)
            sns.lineplot(data=cluster_df, x='Normalized Frame', y='partner_heading_angle_change', label='Partner', errorbar=('ci', 95), color='orange', ax=ax4)

            ax4.axvline(0, color="gray", ls="--", lw=0.5)
            ax4.set_ylim(0, 60)
            ax4.set_xticks([])
            ax4.set_visible(True)

            ## 4. APPROACH ANGLE
            ax5 = axes_ap[5, column]

            sns.lineplot(data=cluster_df, x='Normalized Frame', y='anchor_approach_angle', label='Anchor', errorbar=('ci', 95), color='blue', ax=ax5)
            sns.lineplot(data=cluster_df, x='Normalized Frame', y='partner_approach_angle', label='Partner', errorbar=('ci', 95), color='orange', ax=ax5)

            ax5.axvline(0, color="gray", ls="--", lw=0.5)
            ax5.set_ylim(0, 180)
            ax5.set_xticks([])
            # ax1.set_yticks([])
            ax5.set_visible(True)

            ## 6. APPROACH ANGLE CHANGE
            ax6 = axes_ap[6, column]

            sns.lineplot(data=cluster_df, x='Normalized Frame', y='anchor_approach_angle_change', label='Anchor', errorbar=('ci', 95), color='blue', ax=ax6)
            sns.lineplot(data=cluster_df, x='Normalized Frame', y='partner_approach_angle_change', label='Partner', errorbar=('ci', 95), color='orange', ax=ax6)

            ax6.axvline(0, color="gray", ls="--", lw=0.5)
            ax6.set_ylim(0, 60)
            ax6.set_xticks([])
            ax6.set_visible(True)

            ## 7. DISTANCE TRAVELLED
            ax7 = axes_ap[7, column]

            sns.lineplot(data=cluster_df, x='Normalized Frame', y='anchor_distance', label='Anchor', errorbar=('ci', 95), color='blue', ax=ax7)
            sns.lineplot(data=cluster_df, x='Normalized Frame', y='partner_distance', label='Partner', errorbar=('ci', 95), color='orange', ax=ax7)

            ax7.axvline(0, color="gray", ls="--", lw=0.5)
            ax7.set_ylim(0, 14)
            # ax5.set_xticks([])
            # ax1.set_yticks([])
            ax7.set_visible(True)

            ## 8. MIN DISTANCE BETWEEN

            ax8 = axes_ap[8, column]
            grouped_min = cluster_df.groupby("Normalized Frame")["min_distance"]
            mean_min = grouped_min.mean()
            std_min  = grouped_min.std()

            mean_min = mean_min.sort_index()
            # Split windows: pre (<0), post (>0)
            pre = mean_min[mean_min.index < 0]
            post = mean_min[mean_min.index > 0]

            if len(pre) >= 2 and pre.index.nunique() >= 2:
                res_pre = linregress(pre.index.values.astype(float), pre.values.astype(float))
                slope_pre = res_pre.slope
            else:
                slope_pre = np.nan

            if len(post) >= 2 and post.index.nunique() >= 2:
                res_post = linregress(post.index.values.astype(float), post.values.astype(float))
                slope_post = res_post.slope
            else:
                slope_post = np.nan

            # ax8.plot(mean_min.index, mean_min.values, color='black')
            # ax8.fill_between(
            #     mean_min.index,
            #     mean_min - std_min,
            #     mean_min + std_min,
            #     color='gray',
            #     alpha=0.3
            # )
            sns.lineplot(
            data=cluster_df,
            x='Normalized Frame',
            y='min_distance',
            errorbar=('ci', 95),
            color='black',
            ax=ax8
        )
            ax8.axvline(0, color='red', linestyle='--', linewidth=0.5)
            ax8.set_ylim(0, 25)
            ax8.set_xticks([])
            ax8.set_visible(True)

            ax8.text(0.55, 0.92, f"pre slope:  {slope_pre:.2f}",  transform=ax8.transAxes, fontsize=8)
            ax8.text(0.55, 0.76, f"post slope: {slope_post:.2f}", transform=ax8.transAxes, fontsize=8)



            # ---- 9–12. CONTACT SUMMARY (match standalone) ----
            interaction_colors = {
                "head-head": "red",
                "head-body": "orange",
                "head-tail": "yellow",
                "body-body": "black",
                "tail-tail": "green",
                "tail-body": "purple"}


            interaction_merge_map = {
                "head-tail": "head-tail", "tail-head": "head-tail",
                "tail-body": "tail-body", "body-tail": "tail-body",
                "head-body": "head-body", "body-head": "head-body",
                "tail-tail": "tail-tail", "head-head": "head-head", "body-body": "body-body",
            }
            interaction_types = ["head-head", "head-body", "head-tail", "body-body", "tail-tail", "tail-body"]
            palette_list = [interaction_colors[t] for t in interaction_types]

            inter_ids = cluster_df["interaction_id"].unique()
            records = []
            init_labels = []
            pred_labels = []
            frames_close_list = []

            for inter_id in inter_ids:
                inter = cluster_df[cluster_df["interaction_id"] == inter_id].sort_values("Frame")

                # frames in contact (<1mm), merge symmetric labels
                close = inter[inter["min_distance"] < 1].copy()
                close["interaction_type_merged"] = close["interaction_type"].map(interaction_merge_map)

                # counts per interaction (merged types)
                counts = close["interaction_type_merged"].value_counts().to_dict()
                row = {"interaction_id": inter_id}
                for it in interaction_types:
                    row[it] = counts.get(it, 0)
                records.append(row)

                # initial & predominant labels, if any contact exists
                tm = close["interaction_type_merged"]
                if not tm.empty:
                    init_labels.append(tm.iloc[0])
                    pred_labels.append(tm.value_counts().idxmax())

                # total frames <1mm for this interaction
                frames_close_list.append((inter_id, (inter["min_distance"] < 1).sum()))

            # ---------- ROW 9: Interaction Type (mean ± sd frames per interaction) ----------
            ax9 = axes_ap[9, column]
            df_counts = pd.DataFrame(records)
            means = df_counts[interaction_types].mean()
            stds  = df_counts[interaction_types].std()

            # x = np.arange(len(interaction_types))
            # ax9.bar(x, means.values, yerr=stds.values, capsize=5,
            #         color=[interaction_colors[it] for it in interaction_types], alpha=0.8)
            # ax9.set_xticks(x)

            df_counts_long = df_counts.melt(value_vars=interaction_types,
                                var_name="interaction_type",
                                value_name="frames")

            sns.barplot(
                    data=df_counts_long,
                    x="interaction_type",
                    y="frames",
                    order=interaction_types,
                    palette=palette_list,
                    errorbar=('ci', 95),   # <-- key change for 95% CI
                    ax=ax9
                )
                            
            ax9.set_xticklabels(interaction_types, rotation=45, fontsize=6)
            ax9.set_ylim(0, (means + stds).max() * 1.1 if len(means) else 1)
            ax9.set_xticks([])
            ax9.set_visible(True)

            # ---------- ROW 10: Initial Contact (%) ----------
            ax10 = axes_ap[10, column]
            if len(init_labels):
                tmp_init = pd.DataFrame({
                    "contact_type": np.repeat(interaction_types, len(init_labels)),
                    "val": np.concatenate([(np.array(init_labels) == t).astype(int) for t in interaction_types])
                })
            else:
                tmp_init = pd.DataFrame({"contact_type": interaction_types, "val": np.zeros(len(interaction_types), dtype=int)})

            sns.barplot(
                data=tmp_init, x="contact_type", y="val",
                errorbar=('ci', 95),
                order=interaction_types, palette=palette_list, ax=ax10
            )
            ax10.set_ylim(0, 1)
            ax10.set_xticks([])
            ax10.set_visible(True)

            # ---------- ROW 11: Predominant Contact (%) ----------
            ax11 = axes_ap[11, column]
            if len(pred_labels):
                tmp_pred = pd.DataFrame({
                    "contact_type": np.repeat(interaction_types, len(pred_labels)),
                    "val": np.concatenate([(np.array(pred_labels) == t).astype(int) for t in interaction_types])
                })
            else:
                tmp_pred = pd.DataFrame({"contact_type": interaction_types, "val": np.zeros(len(interaction_types), dtype=int)})

            sns.barplot(
                data=tmp_pred, x="contact_type", y="val",
                errorbar=('ci', 95),
                order=interaction_types, palette=palette_list, ax=ax11
            )
            ax11.set_ylim(0, 1)
            ax11.set_xticks([])
            ax11.set_visible(True)


            # ---------- ROW 12: Contact Frames <1mm (mean ± sd) ----------
            ax12 = axes_ap[12, column]
            frames_vals = pd.Series([v for _, v in frames_close_list])
            mean_val = float(frames_vals.mean()) if len(frames_vals) else 0.0
            std_val  = float(frames_vals.std())  if len(frames_vals) else 0.0

            ax12.bar(0, mean_val, yerr=std_val, color='green', alpha=0.8, capsize=5)
            ax12.text(0, mean_val + 1, f"{mean_val:.1f}", ha='left', fontsize=12)
            ax12.set_ylim(0, 15)
            ax12.set_xticks([])
            ax12.set_visible(True)



        out_path = os.path.join(self.directory, "summary_anchor_partner.pdf")
        plt.savefig(out_path, format="pdf", bbox_inches='tight')
        plt.close(fig_ap)

    
    def mean_trace_summary(self):

        df = self.df.copy()
        cluster_name = self.cluster_name

        clusters = sorted(df[cluster_name].dropna().unique())
        n_clusters = len(clusters)

        fig_width = max(6, n_clusters * 1.2)

        fig = plt.figure(figsize=(fig_width, 3.5))
        gs = gridspec.GridSpec(
            2, 1,
            height_ratios=[0.9, 0.08],
            hspace=0.15
        )

        sub_ax1 = gs[0].subgridspec(1, n_clusters, wspace=0.1)
        axes_ax1 = [fig.add_subplot(sub_ax1[0, i]) for i in range(n_clusters)]

        ax2 = fig.add_subplot(gs[1])

        anchor_cmap = plt.cm.Greens
        partner_cmap = plt.cm.Purples

        ordered_frames = sorted(df["Normalized Frame"].dropna().unique())
        norm_frames = Normalize(vmin=ordered_frames[0], vmax=ordered_frames[-1])

        for cluster, ax in zip(clusters, axes_ax1):

            cluster_df = df[df[cluster_name] == cluster].copy()
            by_frame = cluster_df.sort_values("Normalized Frame").groupby("Normalized Frame")
            frames = [f for f in by_frame.groups.keys() if f == f]

            def get_means(role, node):
                mx = by_frame[f"{role} x_{node}"].mean()
                my = by_frame[f"{role} y_{node}"].mean()
                return mx, my

            # draw trails
            for node in ("head", "body", "tail"):

                anchor_x, anchor_y = get_means("anchor", node)
                partner_x, partner_y = get_means("partner", node)

                for f0, f1 in zip(frames[:-1], frames[1:]):

                    if f0 in anchor_x.index and f1 in anchor_x.index:
                        x0, y0 = anchor_x.loc[f0], anchor_y.loc[f0]
                        x1, y1 = anchor_x.loc[f1], anchor_y.loc[f1]

                        if np.isfinite(x0) and np.isfinite(y0) and np.isfinite(x1) and np.isfinite(y1):
                            ax.plot(
                                [x0, x1], [y0, y1],
                                color=anchor_cmap(norm_frames(f1)),
                                alpha=0.7,
                                linewidth=1.2,
                                zorder=1
                            )

                    if f0 in partner_x.index and f1 in partner_x.index:
                        x0, y0 = partner_x.loc[f0], partner_y.loc[f0]
                        x1, y1 = partner_x.loc[f1], partner_y.loc[f1]

                        if np.isfinite(x0) and np.isfinite(y0) and np.isfinite(x1) and np.isfinite(y1):
                            ax.plot(
                                [x0, x1], [y0, y1],
                                color=partner_cmap(norm_frames(f1)),
                                alpha=0.7,
                                linewidth=1.2,
                                zorder=1
                            )

            # draw skeletons
            for f in frames:

                parts = {}
                for node in ("head", "body", "tail"):
                    x_means = by_frame[f"anchor x_{node}"].mean()
                    y_means = by_frame[f"anchor y_{node}"].mean()

                    if f in x_means.index:
                        x, y = x_means.loc[f], y_means.loc[f]
                        if np.isfinite(x) and np.isfinite(y):
                            parts[node] = (x, y)

                if len(parts) == 3:
                    ax.plot(
                        [parts["head"][0], parts["body"][0], parts["tail"][0]],
                        [parts["head"][1], parts["body"][1], parts["tail"][1]],
                        color=anchor_cmap(norm_frames(f)),
                        alpha=0.75,
                        linewidth=1.0,
                        zorder=2
                    )

                parts = {}
                for node in ("head", "body", "tail"):
                    x_means = by_frame[f"partner x_{node}"].mean()
                    y_means = by_frame[f"partner y_{node}"].mean()

                    if f in x_means.index:
                        x, y = x_means.loc[f], y_means.loc[f]
                        if np.isfinite(x) and np.isfinite(y):
                            parts[node] = (x, y)

                if len(parts) == 3:
                    ax.plot(
                        [parts["head"][0], parts["body"][0], parts["tail"][0]],
                        [parts["head"][1], parts["body"][1], parts["tail"][1]],
                        color=partner_cmap(norm_frames(f)),
                        alpha=0.75,
                        linewidth=1.0,
                        zorder=2
                    )

            # draw points
            size_map = {"head": 6, "body": 4, "tail": 2}

            for node in ("head", "body", "tail"):

                anchor_x_means, anchor_y_means = get_means("anchor", node)
                partner_x_means, partner_y_means = get_means("partner", node)

                for f in frames:

                    if f in anchor_x_means.index:
                        x, y = anchor_x_means.loc[f], anchor_y_means.loc[f]

                        if np.isfinite(x) and np.isfinite(y):
                            ax.scatter(
                                x, y,
                                s=size_map[node],
                                marker="o",
                                color=anchor_cmap(norm_frames(f)),
                                alpha=0.9,
                                zorder=3
                            )

                    if f in partner_x_means.index:
                        x, y = partner_x_means.loc[f], partner_y_means.loc[f]

                        if np.isfinite(x) and np.isfinite(y):
                            ax.scatter(
                                x, y,
                                s=size_map[node],
                                marker="o",
                                color=partner_cmap(norm_frames(f)),
                                alpha=0.9,
                                zorder=3
                            )

            ax.set_xlim(-50, 100)
            ax.set_ylim(-10, 300)
            ax.set_title(cluster, fontsize=12, fontweight="bold")
            ax.axis("off")

        # average contact frames
        interaction_contact_summary = []

        for cluster_id in clusters:

            cluster_df = df[df[cluster_name] == cluster_id].copy()

            for inter_id in cluster_df["interaction_id"].dropna().unique():

                inter_df = cluster_df[cluster_df["interaction_id"] == inter_id]
                n_close = (inter_df["min_distance"] < 1).sum()

                interaction_contact_summary.append({
                    "cluster": cluster_id,
                    "interaction_id": inter_id,
                    "frames_below_1mm": n_close
                })

        df_interaction_contact = pd.DataFrame(interaction_contact_summary)

        mean_frames = (
            df_interaction_contact
            .groupby("cluster")["frames_below_1mm"]
            .mean()
            .reindex(clusters, fill_value=0)
        )

        heat = mean_frames.to_numpy()[np.newaxis, :]

        colors = ["skyblue", "mediumseagreen", "darkgreen"]
        my_cmap = LinearSegmentedColormap.from_list("greenblue_custom", colors)

        im = ax2.imshow(
            heat,
            aspect="auto",
            interpolation="nearest",
            vmin=0,
            cmap=my_cmap
        )

        ax2.set_yticks([])
        ax2.set_xticks([])
        ax2.tick_params(left=False, bottom=False)

        for spine in ax2.spines.values():
            spine.set_visible(False)

        cbar = fig.colorbar(
            im,
            ax=[*axes_ax1, ax2],
            fraction=0.015,
            pad=0.01,
            location="right"
        )

        cbar.set_label("Average Contact Frames", rotation=270, labelpad=15)

        output = os.path.join(self.directory, "mean_trace_summary.png")
        plt.savefig(output, dpi=300, bbox_inches="tight")

        output = os.path.join(self.directory, "mean_trace_summary.pdf")
        plt.savefig(output, format="pdf", bbox_inches="tight")

        plt.close(fig)

    
    def merge_clusters(self, merge_dict, new_cluster_name=None):
        """
        Merge selected clusters using a dictionary.

        Example:
        merge_dict = {
            1: 1,
            2: 1,
            3: 2,
            4: 2
        }

        This means:
        old cluster 1 -> new cluster 1
        old cluster 2 -> new cluster 1
        old cluster 3 -> new cluster 2
        old cluster 4 -> new cluster 2

        Any clusters not in merge_dict stay unchanged.
        """

        cluster_name = self.cluster_name

        if new_cluster_name is None:
            new_cluster_name = f"{cluster_name}_merged"

        self.clusters[new_cluster_name] = (
            self.clusters[cluster_name]
            .map(merge_dict)
            .fillna(self.clusters[cluster_name])
        )

        self.df[new_cluster_name] = (
            self.df[cluster_name]
            .map(merge_dict)
            .fillna(self.df[cluster_name])
        )

        self.cluster_name = new_cluster_name

        print(f">>> Created merged cluster column: {new_cluster_name}")
        print(">>> New clusters:", sorted(self.df[new_cluster_name].dropna().unique()))


    
    
    def contact_percentage_per_cluster(self):

        df = self.df.copy()
        cluster_name = self.cluster_name

        # For each interaction, check whether min_distance ever falls below 1
        interaction_contact = (
            df.groupby(['condition', 'interaction_id', cluster_name])['min_distance']
            .apply(lambda x: (x < 1).any())
            .reset_index(name='has_contact')
        )

        # For each cluster, calculate percentage of interactions with contact
        summary = (
            interaction_contact
            .groupby([cluster_name, 'condition'])['has_contact']
            .agg(
                total_interactions='count',
                interactions_with_contact='sum'
            )
            .reset_index()
        )

        summary['percent_with_contact'] = (
            summary['interactions_with_contact'] / summary['total_interactions'] * 100
        )

        print("\nPercentage of interactions per cluster where min_distance < 1:")
        print(summary)

        # Save CSV
        output_csv = os.path.join(
            self.directory,
            'percent_interactions_with_contact_per_cluster.csv'
        )
        summary.to_csv(output_csv, index=False)

        # Plot
        plt.figure(figsize=(10, 6))

        palette = {
            "G": "C0",
            "S": "C1",
            "GS": "mediumseagreen"
        }

        sns.barplot(
            data=summary,
            x=cluster_name,
            y='percent_with_contact',
            hue='condition',
            hue_order=['G', 'S', 'GS'],
            palette=palette
        )

        plt.ylabel('% interactions with min_distance < 1 mm')
        plt.xlabel('Cluster ID')
        plt.title('Percentage of Interactions with Contact per Cluster')
        plt.ylim(0, 100)
        plt.xticks(rotation=90)
        plt.tight_layout()

        output_pdf = os.path.join(
            self.directory,
            'percent_interactions_with_contact_per_cluster.pdf'
        )
        plt.savefig(output_pdf, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()

        print(f"\nSaved CSV to: {output_csv}")
        print(f"Saved plot to: {output_pdf}")
    





    def GS_pair_availability_normalised_interactions(self):

        df = self.clusters.copy()
        df_interaction = self.df.copy()

        output = os.path.join(
            self.directory,
            "GS_pair_availability_normalised_interactions"
        )
        os.makedirs(output, exist_ok=True)

        df = df[df["condition"] == "GS"].copy()
        df_interaction = df_interaction[df_interaction["condition"] == "GS"].copy()

        if df.empty:
            print("No GS condition data found.")
            return

        social_order = ["S-S", "G-S", "G-G"]

        # 5 isolated tracks: 0-4
        # 5 grouped tracks: 5-9
        n_iso = 5
        n_group = 5

        possible_pairs = {
            "S-S": n_iso * (n_iso - 1) // 2,   # 10
            "G-S": n_iso * n_group,            # 25
            "G-G": n_group * (n_group - 1) // 2 # 10
        }

        total_possible_pairs = sum(possible_pairs.values())

        expected_fraction = {
            k: v / total_possible_pairs
            for k, v in possible_pairs.items()
        }

        def parse_pair(pair):

            if pd.isna(pair):
                return np.nan

            if isinstance(pair, str):
                pair = pair.replace("(", "").replace(")", "").replace(" ", "")
                id1, id2 = pair.split(",")
                id1 = int(id1)
                id2 = int(id2)
            else:
                id1, id2 = pair
                id1 = int(id1)
                id2 = int(id2)

            return tuple(sorted((id1, id2)))

        def get_social_experience(pair):

            if pair is np.nan or pd.isna(pair):
                return np.nan

            id1, id2 = pair

            if id1 <= 4 and id2 <= 4:
                return "S-S"
            elif id1 >= 5 and id2 >= 5:
                return "G-G"
            else:
                return "G-S"

        # Get one row per interaction_id with file and interaction pair
        pair_lookup = (
            df_interaction[
                ["file", "interaction_id", "Interaction Pair"]
            ]
            .drop_duplicates(subset=["interaction_id"])
            .copy()
        )

        pair_lookup["parsed_pair"] = pair_lookup["Interaction Pair"].apply(parse_pair)
        pair_lookup["social_experience"] = pair_lookup["parsed_pair"].apply(get_social_experience)

        pair_lookup = pair_lookup.dropna(subset=["social_experience"])

        # One row per interaction bout
        inter_per_video = (
            pair_lookup[
                ["file", "interaction_id", "parsed_pair", "social_experience"]
            ]
            .drop_duplicates(subset=["file", "interaction_id"])
            .copy()
        )

        # Count interaction bouts per video and social experience
        counts = (
            inter_per_video
            .groupby(["file", "social_experience"])
            .size()
            .reset_index(name="observed_interactions")
        )

        # Ensure every video has S-S, G-S, G-G rows, even if zero
        all_files = sorted(inter_per_video["file"].dropna().unique())

        full_index = pd.MultiIndex.from_product(
            [all_files, social_order],
            names=["file", "social_experience"]
        )

        counts = (
            counts
            .set_index(["file", "social_experience"])
            .reindex(full_index, fill_value=0)
            .reset_index()
        )

        # Total observed bouts per video
        counts["total_observed_interactions"] = (
            counts
            .groupby("file")["observed_interactions"]
            .transform("sum")
        )

        # Actual observed percentage per video
        counts["observed_percent"] = (
            counts["observed_interactions"] /
            counts["total_observed_interactions"]
        ) * 100

        counts["observed_percent"] = counts["observed_percent"].fillna(0)

        # Expected percentage based on possible pair availability
        counts["possible_pairs"] = counts["social_experience"].map(possible_pairs)

        counts["expected_fraction"] = counts["social_experience"].map(expected_fraction)
        counts["expected_percent"] = counts["expected_fraction"] * 100

        # Main corrected metric
        counts["interactions_per_possible_pair"] = (
            counts["observed_interactions"] /
            counts["possible_pairs"]
        )

        # Enrichment relative to random pair availability
        counts["observed_fraction"] = counts["observed_percent"] / 100

        counts["enrichment_observed_over_expected"] = (
            counts["observed_fraction"] /
            counts["expected_fraction"]
        )

        counts["enrichment_observed_over_expected"] = (
            counts["enrichment_observed_over_expected"]
            .replace([np.inf, -np.inf], np.nan)
        )

        # Save per-video CSV
        per_video_csv = os.path.join(
            output,
            "GS_pair_availability_normalised_per_video.csv"
        )
        counts.to_csv(per_video_csv, index=False)

        # Summary CSV across videos
        summary = (
            counts
            .groupby("social_experience")
            .agg(
                mean_observed_interactions=("observed_interactions", "mean"),
                sd_observed_interactions=("observed_interactions", "std"),
                mean_observed_percent=("observed_percent", "mean"),
                sd_observed_percent=("observed_percent", "std"),
                expected_percent=("expected_percent", "first"),
                possible_pairs=("possible_pairs", "first"),
                mean_interactions_per_possible_pair=("interactions_per_possible_pair", "mean"),
                sd_interactions_per_possible_pair=("interactions_per_possible_pair", "std"),
                mean_enrichment=("enrichment_observed_over_expected", "mean"),
                sd_enrichment=("enrichment_observed_over_expected", "std")
            )
            .reset_index()
        )

        summary_csv = os.path.join(
            output,
            "GS_pair_availability_normalised_summary.csv"
        )
        summary.to_csv(summary_csv, index=False)

        # Plot 1: actual observed percentage vs expected percentage
        palette = {
            "S-S": "C1",
            "G-S": "mediumseagreen",
            "G-G": "C0"
        }

        plt.figure(figsize=(7, 6))

        ax = sns.barplot(
            data=counts,
            x="social_experience",
            y="observed_percent",
            order=social_order,
            errorbar="sd",
            palette=palette,
            alpha=0.8
        )

        sns.stripplot(
            data=counts,
            x="social_experience",
            y="observed_percent",
            order=social_order,
            color="black",
            size=4,
            jitter=True,
            alpha=0.7
        )

        for i, social in enumerate(social_order):
            ax.hlines(
                y=expected_fraction[social] * 100,
                xmin=i - 0.35,
                xmax=i + 0.35,
                colors="red",
                linestyles="--",
                linewidth=2
            )

        plt.ylabel("Observed interactions (%)")
        plt.xlabel("Social experience")
        plt.title("GS interactions: observed % vs expected from pair availability")
        plt.tight_layout()

        path = os.path.join(output, "GS_observed_percent_vs_expected.pdf")
        plt.savefig(path, format="pdf", bbox_inches="tight", dpi=300)
        plt.close()

        # Plot 2: corrected interaction rate per possible pair
        plt.figure(figsize=(7, 6))

        sns.barplot(
            data=counts,
            x="social_experience",
            y="interactions_per_possible_pair",
            order=social_order,
            errorbar="sd",
            palette=palette,
            alpha=0.8
        )

        sns.stripplot(
            data=counts,
            x="social_experience",
            y="interactions_per_possible_pair",
            order=social_order,
            color="black",
            size=4,
            jitter=True,
            alpha=0.7
        )

        plt.ylabel("Interaction bouts per possible pair")
        plt.xlabel("Social experience")
        plt.title("GS interactions normalised by pair availability")
        plt.tight_layout()

        path = os.path.join(output, "GS_interactions_per_possible_pair.pdf")
        plt.savefig(path, format="pdf", bbox_inches="tight", dpi=300)
        path = os.path.join(output, "GS_interactions_per_possible_pair.png")
        plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.close()

        print("\nSaved per-video CSV:")
        print(per_video_csv)

        print("\nSaved summary CSV:")
        print(summary_csv)

        print("\nExpected percentages from possible pair availability:")
        for social in social_order:
            print(f"{social}: {expected_fraction[social] * 100:.2f}%")

        print("\nSummary:")
        print(summary)
    

    def GS_partner_type_transition_matrix(self):

        df = self.df.copy()

        output = os.path.join(self.directory, "GS_partner_type_transition_matrix")
        os.makedirs(output, exist_ok=True)

        # GS only, one row per interaction at contact frame
        df = df[
            (df["condition"] == "GS") &
            (df["Normalized Frame"] == 0)
        ].copy()

        if df.empty:
            print("No GS interactions found at Normalized Frame == 0.")
            return

        def parse_pair(pair):
            if pd.isna(pair):
                return None

            if isinstance(pair, str):
                pair = pair.replace("(", "").replace(")", "").replace(" ", "")
                id1, id2 = pair.split(",")
                id1 = int(id1)
                id2 = int(id2)
            else:
                id1, id2 = pair
                id1 = int(id1)
                id2 = int(id2)

            return tuple(sorted((id1, id2)))

        def larva_type(track_id):
            return "S" if track_id <= 4 else "G"

        df["parsed_pair"] = df["Interaction Pair"].apply(parse_pair)
        df = df.dropna(subset=["parsed_pair"])

        transition_rows = []

        # Per video
        for file, file_df in df.groupby("file"):

            histories = {i: [] for i in range(10)}

            file_df = file_df.sort_values("Frame")

            for _, row in file_df.iterrows():

                id1, id2 = row["parsed_pair"]
                frame = row["Frame"]
                interaction_id = row["interaction_id"]

                # id1's partner is id2
                histories[id1].append({
                    "frame": frame,
                    "interaction_id": interaction_id,
                    "partner": id2,
                    "partner_type": larva_type(id2)
                })

                # id2's partner is id1
                histories[id2].append({
                    "frame": frame,
                    "interaction_id": interaction_id,
                    "partner": id1,
                    "partner_type": larva_type(id1)
                })

            # Per larva transition sequence
            for larva, history in histories.items():

                history = sorted(history, key=lambda x: x["frame"])

                if len(history) < 2:
                    continue

                for prev, nxt in zip(history[:-1], history[1:]):

                    transition_rows.append({
                        "file": file,
                        "larva": larva,
                        "larva_type": larva_type(larva),
                        "previous_interaction_id": prev["interaction_id"],
                        "next_interaction_id": nxt["interaction_id"],
                        "previous_frame": prev["frame"],
                        "next_frame": nxt["frame"],
                        "previous_partner": prev["partner"],
                        "next_partner": nxt["partner"],
                        "previous_partner_type": prev["partner_type"],
                        "next_partner_type": nxt["partner_type"]
                    })

        transition_df = pd.DataFrame(transition_rows)

        raw_path = os.path.join(output, "GS_partner_type_transitions_raw.csv")
        transition_df.to_csv(raw_path, index=False)

        if transition_df.empty:
            print("No transitions found.")
            return

        # Count transitions per video
        counts = (
            transition_df
            .groupby([
                "file",
                "larva_type",
                "previous_partner_type",
                "next_partner_type"
            ])
            .size()
            .reset_index(name="count")
        )

        # Fill missing combinations with zero
        all_files = sorted(transition_df["file"].unique())
        larva_types = ["S", "G"]
        partner_types = ["S", "G"]

        full_index = pd.MultiIndex.from_product(
            [all_files, larva_types, partner_types, partner_types],
            names=[
                "file",
                "larva_type",
                "previous_partner_type",
                "next_partner_type"
            ]
        )

        counts = (
            counts
            .set_index([
                "file",
                "larva_type",
                "previous_partner_type",
                "next_partner_type"
            ])
            .reindex(full_index, fill_value=0)
            .reset_index()
        )

        # Row-normalise: for each file/larva_type/previous state,
        # next S + next G = 1
        counts["total_from_previous"] = (
            counts
            .groupby(["file", "larva_type", "previous_partner_type"])["count"]
            .transform("sum")
        )

        counts["transition_probability"] = (
            counts["count"] / counts["total_from_previous"]
        )

        counts["transition_probability"] = counts["transition_probability"].fillna(0)

        matrix_path = os.path.join(output, "GS_partner_type_transition_matrix_per_video.csv")
        counts.to_csv(matrix_path, index=False)

        # Summary across videos
        summary = (
            counts
            .groupby([
                "larva_type",
                "previous_partner_type",
                "next_partner_type"
            ])
            .agg(
                mean_probability=("transition_probability", "mean"),
                sd_probability=("transition_probability", "std"),
                mean_count=("count", "mean"),
                sd_count=("count", "std")
            )
            .reset_index()
        )

        summary_path = os.path.join(output, "GS_partner_type_transition_matrix_summary.csv")
        summary.to_csv(summary_path, index=False)

        # Heatmaps: one for S larvae, one for G larvae
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        for ax, ltype in zip(axes, ["S", "G"]):

            heat = (
                summary[summary["larva_type"] == ltype]
                .pivot(
                    index="previous_partner_type",
                    columns="next_partner_type",
                    values="mean_probability"
                )
                .reindex(index=["S", "G"], columns=["S", "G"])
            )

            sns.heatmap(
                heat,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                vmin=0,
                vmax=1,
                square=True,
                ax=ax,
                cbar=True
            )

            ax.set_title("Isolated larvae" if ltype == "S" else "Grouped larvae")
            ax.set_xlabel("Next partner type")
            ax.set_ylabel("Previous partner type")

        plt.tight_layout()

        plt.savefig(
            os.path.join(output, "GS_partner_type_transition_heatmap.pdf"),
            format="pdf",
            bbox_inches="tight",
            dpi=300
        )

        plt.savefig(
            os.path.join(output, "GS_partner_type_transition_heatmap.png"),
            bbox_inches="tight",
            dpi=300
        )

        plt.close()

        print(f"Saved raw transitions: {raw_path}")
        print(f"Saved per-video transition matrix: {matrix_path}")
        print(f"Saved summary: {summary_path}")




    #### METHOD LARVAL_PROXIMITY: DISTANCE TO NEAREST THIRD LARVA DURING EACH GS INTERACTION, PER CLUSTER / PAIR TYPE
    def larval_proximity(self):

        df = self.df.copy()
        cluster_name = self.cluster_name
        tracks_path = self.tracks_path

        output = os.path.join(self.directory, "larval_proximity")
        os.makedirs(output, exist_ok=True)

        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42


        df = df[df['condition'] == 'GS'].copy()

        if df.empty:
            print("No GS condition data found.")
            return

        def parse_pair(pair):
            if pd.isna(pair):
                return None
            if isinstance(pair, str):
                pair = pair.replace("(", "").replace(")", "").replace(" ", "")
                id1, id2 = pair.split(",")
                return int(id1), int(id2)
            id1, id2 = pair
            return int(id1), int(id2)

        def larva_type(track_id):
            return "S" if track_id <= 4 else "G"

        df["parsed_pair"] = df["Interaction Pair"].apply(parse_pair)
        df = df.dropna(subset=["parsed_pair"])
        df["id1"], df["id2"] = zip(*df["parsed_pair"])

        df["pair_type"] = np.where(
            (df["id1"].map(larva_type) == "G") & (df["id2"].map(larva_type) == "G"),
            "G-G",
            np.where(
                (df["id1"].map(larva_type) == "S") & (df["id2"].map(larva_type) == "S"),
                "S-S",
                "G-S"
            )
        )

        # mm positions of the interacting pair -- same mm_Track_1/2 columns the rest of the class uses
        df["mid_x"] = (df["mm_Track_1 x_body"] + df["mm_Track_2 x_body"]) / 2
        df["mid_y"] = (df["mm_Track_1 y_body"] + df["mm_Track_2 y_body"]) / 2

        # === LOAD FULL-TRACK POSITIONS (ALREADY IN MM) FOR EVERY LARVA, EVERY FRAME ===

        track_file = os.path.join(tracks_path, "merged.track.feather")

        if not os.path.exists(track_file):
            print(f"⚠️ Missing track file: {track_file}")
            return

        tracks = feather.read_feather(track_file)[['track_id', 'frame', 'file', 'x_body', 'y_body']]
        tracks['file'] = tracks['file'].str.replace('.tracks.feather', '.mp4', regex=False)

        # expand every interaction against every other larva present at that frame, then take the closest one
        expanded = df[[
            'interaction_id', 'file', 'Frame', 'Normalized Frame', cluster_name,
            'pair_type', 'id1', 'id2', 'mid_x', 'mid_y'
        ]].merge(
            tracks, left_on=['file', 'Frame'], right_on=['file', 'frame'], how='left'
        )

        expanded = expanded[
            (expanded['track_id'] != expanded['id1']) &
            (expanded['track_id'] != expanded['id2'])
        ]

        expanded['distance_to_third_larva_mm'] = np.sqrt(
            (expanded['x_body'] - expanded['mid_x'])**2 +
            (expanded['y_body'] - expanded['mid_y'])**2
        )

        result = (
            expanded
            .groupby(['interaction_id', 'file', 'Frame', 'Normalized Frame', cluster_name, 'pair_type'])['distance_to_third_larva_mm']
            .min()
            .reset_index()
        )

        if result.empty:
            print("No proximity data could be computed (missing track file?).")
            return

        csv_path = os.path.join(output, "larval_proximity.csv")
        result.to_csv(csv_path, index=False)
        print(f"Saved per-frame proximity data: {csv_path}")

        # === SUMMARY + BARPLOT: NEAREST THIRD-LARVA DISTANCE PER CLUSTER, BY PAIR TYPE ===

        social_order = ['S-S', 'G-S', 'G-G']

        summary = (
            result
            .groupby([cluster_name, 'pair_type'])['distance_to_third_larva_mm']
            .agg(['mean', 'std', 'count'])
            .reset_index()
        )

        summary_path = os.path.join(output, "larval_proximity_summary.csv")
        summary.to_csv(summary_path, index=False)
        print(f"Saved summary: {summary_path}")

        palette = {
            'S-S': 'C1',
            'G-S': 'mediumseagreen',
            'G-G': 'C0'
        }

        cluster_order = sorted(result[cluster_name].dropna().unique())

        plt.figure(figsize=(12, 6))

        ax = sns.barplot(
            data=result,
            x=cluster_name,
            y='distance_to_third_larva_mm',
            hue='pair_type',
            order=cluster_order,
            hue_order=social_order,
            palette=palette,
            errorbar='sd'
        )

        ax.set_title("Distance to nearest third larva during interaction, by cluster and pair type")
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("Distance to nearest third larva (mm)")
        ax.tick_params(axis='x', rotation=90)

        plt.tight_layout()

        path = os.path.join(output, 'larval_proximity_barplot.pdf')
        plt.savefig(path, format='pdf', bbox_inches='tight', dpi=300, transparent=True)


        plt.close()

        print(f"Saved barplot to {output}")




    def GS_cluster_transition_matrix(self):

        df = self.df.copy()
        cluster_name = self.cluster_name

        output = os.path.join(self.directory, "GS_cluster_transition_matrix_by_larva_type")
        os.makedirs(output, exist_ok=True)
        g_vs_s_output = os.path.join(output, "G_V_S")
        within_type_output = os.path.join(output, "within_type")
        mixed_output = os.path.join(output, "mixed")
        mixed_2_output = os.path.join(output, "mixed_2")
        group_output = os.path.join(output, "group")
        isolated_output = os.path.join(output, "isolated")
        os.makedirs(g_vs_s_output, exist_ok=True)
        os.makedirs(within_type_output, exist_ok=True)
        os.makedirs(mixed_output, exist_ok=True)
        os.makedirs(mixed_2_output, exist_ok=True)
        os.makedirs(group_output, exist_ok=True)
        os.makedirs(isolated_output, exist_ok=True)

        # GS only, one row per interaction
        df = df[
            (df["condition"] == "GS") &
            (np.isclose(df["Normalized Frame"], 0))
        ].copy()

        if df.empty:
            print("No GS interactions found at Normalized Frame == 0.")
            return

        def parse_pair(pair):
            if pd.isna(pair):
                return None

            if isinstance(pair, str):
                pair = pair.replace("(", "").replace(")", "").replace(" ", "")
                id1, id2 = pair.split(",")
                return tuple(sorted((int(id1), int(id2))))

            id1, id2 = pair
            return tuple(sorted((int(id1), int(id2))))

        def larva_type(track_id):
            return "S" if int(track_id) <= 4 else "G"

        df["parsed_pair"] = df["Interaction Pair"].apply(parse_pair)
        df = df.dropna(subset=["parsed_pair"])

        df["cluster"] = df[cluster_name].astype(int)

        rows = []

        # Expand to one row per larva per interaction
        for _, row in df.iterrows():

            id1, id2 = row["parsed_pair"]

            rows.append({
                "file": row["file"],
                "interaction_id": row["interaction_id"],
                "track_id": id1,
                "larva_type": larva_type(id1),
                "partner_id": id2,
                "partner_type": larva_type(id2),
                "time": row["Frame"],
                "cluster": row["cluster"]
            })

            rows.append({
                "file": row["file"],
                "interaction_id": row["interaction_id"],
                "track_id": id2,
                "larva_type": larva_type(id2),
                "partner_id": id1,
                "partner_type": larva_type(id1),
                "time": row["Frame"],
                "cluster": row["cluster"]
            })

        expanded = pd.DataFrame(rows)

        expanded = expanded.sort_values(
            ["file", "track_id", "time", "interaction_id"]
        )

        # Next cluster per larva within each video
        expanded["next_cluster"] = (
            expanded
            .groupby(["file", "track_id"])["cluster"]
            .shift(-1)
        )
        expanded["next_partner_id"] = (
            expanded
            .groupby(["file", "track_id"])["partner_id"]
            .shift(-1)
        )
        expanded["next_partner_type"] = (
            expanded
            .groupby(["file", "track_id"])["partner_type"]
            .shift(-1)
        )

        transitions = expanded.dropna(subset=["next_cluster"]).copy()
        transitions["next_cluster"] = transitions["next_cluster"].astype(int)

        raw_path = os.path.join(g_vs_s_output, "GS_cluster_transitions_raw_by_larva_type.csv")
        transitions.to_csv(raw_path, index=False)

        if transitions.empty:
            print("No cluster transitions found.")
            return

        clusters = sorted(
            set(transitions["cluster"].unique()) |
            set(transitions["next_cluster"].unique())
        )

        # Count transitions per video and larva type
        counts = (
            transitions
            .groupby(["file", "larva_type", "cluster", "next_cluster"])
            .size()
            .reset_index(name="count")
        )

        all_files = sorted(transitions["file"].dropna().unique())
        larva_types = ["S", "G"]

        full_index = pd.MultiIndex.from_product(
            [all_files, larva_types, clusters, clusters],
            names=["file", "larva_type", "cluster", "next_cluster"]
        )

        counts = (
            counts
            .set_index(["file", "larva_type", "cluster", "next_cluster"])
            .reindex(full_index, fill_value=0)
            .reset_index()
        )

        # Row-normalise within each file, larva type and current cluster
        counts["row_total"] = (
            counts
            .groupby(["file", "larva_type", "cluster"])["count"]
            .transform("sum")
        )

        counts["transition_probability"] = (
            counts["count"] / counts["row_total"]
        )

        counts["transition_probability"] = counts["transition_probability"].fillna(0)

        per_video_path = os.path.join(
            g_vs_s_output,
            "GS_cluster_transition_matrix_per_video_by_larva_type.csv"
        )
        counts.to_csv(per_video_path, index=False)

        # Average across videos
        summary = (
            counts
            .groupby(["larva_type", "cluster", "next_cluster"])
            .agg(
                mean_probability=("transition_probability", "mean"),
                sd_probability=("transition_probability", "std"),
                mean_count=("count", "mean"),
                sd_count=("count", "std")
            )
            .reset_index()
        )

        summary_path = os.path.join(
            g_vs_s_output,
            "GS_cluster_transition_matrix_summary_by_larva_type.csv"
        )
        summary.to_csv(summary_path, index=False)

        # Plot two heatmaps: S tracks and G tracks
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        matrices = {}

        for ax, ltype in zip(axes, larva_types):

            matrix = (
                summary[summary["larva_type"] == ltype]
                .pivot(
                    index="cluster",
                    columns="next_cluster",
                    values="mean_probability"
                )
                .reindex(index=clusters, columns=clusters)
                .fillna(0)
            )

            matrices[ltype] = matrix

        vmax = max(
            matrices["S"].to_numpy().max(),
            matrices["G"].to_numpy().max()
        )

        for ax, ltype in zip(axes, larva_types):

            sns.heatmap(
                matrices[ltype],
                cmap="viridis",
                vmin=0,
                vmax=vmax,
                square=True,
                cbar=False,
                ax=ax
            )

            ax.set_title("GS: isolated-experience larvae" if ltype == "S" else "GS: group-housed-experience larvae")
            ax.set_xlabel("Next cluster")
            ax.set_ylabel("Current cluster")

            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        fig.subplots_adjust(right=0.88, wspace=0.25)
        cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
        sm = mpl.cm.ScalarMappable(
            cmap="viridis",
            norm=mpl.colors.Normalize(vmin=0, vmax=vmax)
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Mean transition probability")

        plt.savefig(
            os.path.join(g_vs_s_output, "GS_cluster_transition_heatmaps_by_larva_type.png"),
            dpi=300,
            bbox_inches="tight"
        )

        plt.savefig(
            os.path.join(g_vs_s_output, "GS_cluster_transition_heatmaps_by_larva_type.pdf"),
            format="pdf",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(f"Saved raw transitions: {raw_path}")
        print(f"Saved per-video transition matrix: {per_video_path}")
        print(f"Saved summary: {summary_path}")


        M_S = matrices["S"].copy()
        M_G = matrices["G"].copy()

        all_clusters = sorted(
            set(M_S.index) | set(M_S.columns) |
            set(M_G.index) | set(M_G.columns)
        )

        M_S = M_S.reindex(index=all_clusters, columns=all_clusters, fill_value=0)
        M_G = M_G.reindex(index=all_clusters, columns=all_clusters, fill_value=0)

        global_max = max(M_S.to_numpy().max(), M_G.to_numpy().max())

        if global_max == 0:
            global_max = 1e-6

        pos = nx.circular_layout(all_clusters)
        cmap = plt.cm.viridis
        norm = mpl.colors.Normalize(vmin=0, vmax=global_max)

        def matrix_to_digraph(M, thresh=0.0):
            G = nx.DiGraph()

            for c in M.index:
                G.add_node(int(c))

            for i in M.index:
                for j in M.columns:
                    w = float(M.loc[i, j])

                    if w > thresh:
                        G.add_edge(int(i), int(j), weight=w)

            return G

        def draw_self_loop(
            ax,
            xy,
            w,
            loop_offset=0.10,
            loop_size=0.08,
            rad=0.8,
            arrow_size=14,
            min_w=0.2,
            max_w=6.0,
            min_a=0.05,
            max_a=0.95
        ):

            x, y = xy
            v = np.array([x, y], dtype=float)
            r = np.linalg.norm(v)

            if r == 0:
                return

            u = v / r
            perp = np.array([-u[1], u[0]])

            c = v + loop_offset * u

            start = c - loop_size * perp + 0.15 * loop_size * u
            end = c + loop_size * perp + 0.15 * loop_size * u

            lw = min_w + (w / global_max) * (max_w - min_w)
            alpha = min_a + (w / global_max) * (max_a - min_a)
            color = cmap(norm(w))

            patch = FancyArrowPatch(
                posA=start,
                posB=end,
                arrowstyle="-|>",
                mutation_scale=arrow_size,
                connectionstyle=f"arc3,rad={rad}",
                linewidth=lw,
                color=color,
                alpha=alpha,
                zorder=2
            )

            ax.add_patch(patch)

        def draw_edge(
            ax,
            xy_u,
            xy_v,
            w,
            rad=0.12,
            arrow_size=14,
            shrink=18,
            min_w=0.2,
            max_w=6.0,
            min_a=0.05,
            max_a=0.95
        ):

            lw = min_w + (w / global_max) * (max_w - min_w)
            alpha = min_a + (w / global_max) * (max_a - min_a)
            color = cmap(norm(w))

            patch = FancyArrowPatch(
                posA=xy_u,
                posB=xy_v,
                arrowstyle="-|>",
                mutation_scale=arrow_size,
                connectionstyle=f"arc3,rad={rad}",
                linewidth=lw,
                color=color,
                alpha=alpha,
                shrinkA=shrink,
                shrinkB=shrink,
                zorder=1
            )

            ax.add_patch(patch)

        def draw_transition_circle(ax, G, title):
            ax.set_title(title)
            ax.axis("off")

            nx.draw_networkx_nodes(
                G,
                pos,
                ax=ax,
                node_size=700,
                node_color="lightgray",
                edgecolors="black"
            )

            nx.draw_networkx_labels(
                G,
                pos,
                ax=ax,
                font_size=9
            )

            for u, v, d in G.edges(data=True):

                w = float(d["weight"])

                if u == v:
                    draw_self_loop(ax, pos[u], w)
                else:
                    draw_edge(ax, pos[u], pos[v], w)

        G_S = matrix_to_digraph(M_S, thresh=0.0)
        G_G = matrix_to_digraph(M_G, thresh=0.0)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        draw_transition_circle(
            axes[0],
            G_S,
            "GS isolated-experience larvae"
        )

        draw_transition_circle(
            axes[1],
            G_G,
            "GS group-housed-experience larvae"
        )

        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        fig.subplots_adjust(right=0.88, wspace=0.25)
        cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])

        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Mean transition probability")

        plt.savefig(
            os.path.join(g_vs_s_output, "GS_cluster_transition_circlegraphs_by_larva_type.png"),
            dpi=300,
            bbox_inches="tight"
        )

        plt.savefig(
            os.path.join(g_vs_s_output, "GS_cluster_transition_circlegraphs_by_larva_type.pdf"),
            format="pdf",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        # -------------------------------------------------------
        # Difference heatmap: G tracks - S tracks
        # Blue = more group-housed-experience
        # Red = more isolated-experience
        # -------------------------------------------------------

        P_diff = M_G - M_S

        lim = float(np.nanmax(np.abs(P_diff.to_numpy())))

        if lim == 0:
            lim = 1e-6

        plt.figure(figsize=(10, 8))

        sns.heatmap(
            P_diff,
            cmap="RdBu",
            center=0,
            vmin=-lim,
            vmax=lim,
            square=True,
            cbar_kws={"label": "P(G tracks) − P(S tracks)"}
        )

        plt.xlabel("Next cluster")
        plt.ylabel("Current cluster")
        plt.title("GS transition difference: group-housed tracks − isolated tracks")

        plt.tight_layout()

        plt.savefig(
            os.path.join(g_vs_s_output, "GS_cluster_transition_difference_G_minus_S.png"),
            dpi=300,
            bbox_inches="tight"
        )

        plt.savefig(
            os.path.join(g_vs_s_output, "GS_cluster_transition_difference_G_minus_S.pdf"),
            format="pdf",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        # -------------------------------------------------------
        # Difference circle graph: G tracks - S tracks
        # Blue = more group-housed-experience
        # Red = more isolated-experience
        # -------------------------------------------------------

        D = P_diff.copy()

        all_clusters = sorted(set(D.index) | set(D.columns))
        D = D.reindex(index=all_clusters, columns=all_clusters, fill_value=0)

        lim = float(np.nanmax(np.abs(D.to_numpy())))

        if lim == 0:
            lim = 1e-6

        diff_cmap = plt.cm.RdBu
        diff_norm = mpl.colors.TwoSlopeNorm(
            vmin=-lim,
            vcenter=0,
            vmax=lim
        )

        def diff_matrix_to_digraph(D, thresh=0.05):

            G = nx.DiGraph()

            for c in D.index:
                G.add_node(int(c))

            for i in D.index:
                for j in D.columns:
                    w = float(D.loc[i, j])

                    if abs(w) >= thresh:
                        G.add_edge(int(i), int(j), weight=w)

            return G

        G_diff = diff_matrix_to_digraph(D, thresh=0.05)

        pos = nx.circular_layout(all_clusters)

        def diff_lw(w, min_w=0.3, max_w=6.0):
            return min_w + (abs(w) / lim) * (max_w - min_w)

        def diff_alpha(w, min_a=0.15, max_a=0.95):
            return min_a + (abs(w) / lim) * (max_a - min_a)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        ax.set_title("GS transition difference")
        ax.axis("off")

        nx.draw_networkx_nodes(
            G_diff,
            pos,
            ax=ax,
            node_size=700,
            node_color="lightgray",
            edgecolors="black"
        )

        nx.draw_networkx_labels(
            G_diff,
            pos,
            ax=ax,
            font_size=9
        )

        for u, v, d in G_diff.edges(data=True):

            w = float(d["weight"])
            color = diff_cmap(diff_norm(w))
            lw = diff_lw(w)
            alpha = diff_alpha(w)

            if u == v:
                # signed self-loop
                x, y = pos[u]
                vec = np.array([x, y], dtype=float)
                r = np.linalg.norm(vec)

                if r == 0:
                    continue

                unit = vec / r
                perp = np.array([-unit[1], unit[0]])

                center = vec + 0.10 * unit
                start = center - 0.08 * perp + 0.012 * unit
                end = center + 0.08 * perp + 0.012 * unit

                patch = FancyArrowPatch(
                    posA=start,
                    posB=end,
                    arrowstyle="-|>",
                    mutation_scale=14,
                    connectionstyle="arc3,rad=0.8",
                    linewidth=lw,
                    color=color,
                    alpha=alpha,
                    zorder=2
                )

            else:
                patch = FancyArrowPatch(
                    posA=pos[u],
                    posB=pos[v],
                    arrowstyle="-|>",
                    mutation_scale=14,
                    connectionstyle="arc3,rad=0.12",
                    linewidth=lw,
                    color=color,
                    alpha=alpha,
                    shrinkA=18,
                    shrinkB=18,
                    zorder=1
                )

            ax.add_patch(patch)

        sm = mpl.cm.ScalarMappable(cmap=diff_cmap, norm=diff_norm)
        sm.set_array([])

        cax = fig.add_axes([0.90, 0.25, 0.015, 0.50])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("P(G tracks) − P(S tracks)")

        plt.savefig(
            os.path.join(g_vs_s_output, "GS_cluster_transition_difference_circlegraph_G_minus_S.png"),
            dpi=300,
            bbox_inches="tight"
        )

        plt.savefig(
            os.path.join(g_vs_s_output, "GS_cluster_transition_difference_circlegraph_G_minus_S.pdf"),
            format="pdf",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        # -------------------------------------------------------
        # Interaction-history filtered cluster transitions
        # -------------------------------------------------------

        def plot_probability_circlegraph(M, title, filename_stem, save_dir, prob_max=None, thresh=0.05):

            values = M.to_numpy()
            finite_values = values[np.isfinite(values)]

            if finite_values.size == 0:
                print(f"No finite values found for {title}.")
                return

            if prob_max is None:
                prob_max = float(np.nanmax(finite_values))
            if prob_max == 0:
                prob_max = 1e-6

            prob_cmap = plt.cm.viridis
            prob_norm = mpl.colors.Normalize(vmin=0, vmax=prob_max)

            G_prob = matrix_to_digraph(M, thresh=thresh)
            prob_pos = nx.circular_layout(list(M.index))

            def prob_lw(w, min_w=0.3, max_w=6.0):
                return min_w + (w / prob_max) * (max_w - min_w)

            def prob_alpha(w, min_a=0.15, max_a=0.95):
                return min_a + (w / prob_max) * (max_a - min_a)

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.set_title(title)
            ax.axis("off")

            nx.draw_networkx_nodes(
                G_prob,
                prob_pos,
                ax=ax,
                node_size=700,
                node_color="lightgray",
                edgecolors="black"
            )

            nx.draw_networkx_labels(
                G_prob,
                prob_pos,
                ax=ax,
                font_size=9
            )

            for u, v, d in G_prob.edges(data=True):

                w = float(d["weight"])
                color = prob_cmap(prob_norm(w))
                lw = prob_lw(w)
                alpha = prob_alpha(w)

                if u == v:
                    x, y = prob_pos[u]
                    vec = np.array([x, y], dtype=float)
                    r = np.linalg.norm(vec)

                    if r == 0:
                        continue

                    unit = vec / r
                    perp = np.array([-unit[1], unit[0]])

                    center = vec + 0.10 * unit
                    start = center - 0.08 * perp + 0.012 * unit
                    end = center + 0.08 * perp + 0.012 * unit

                    patch = FancyArrowPatch(
                        posA=start,
                        posB=end,
                        arrowstyle="-|>",
                        mutation_scale=14,
                        connectionstyle="arc3,rad=0.8",
                        linewidth=lw,
                        color=color,
                        alpha=alpha,
                        zorder=2
                    )

                else:
                    patch = FancyArrowPatch(
                        posA=prob_pos[u],
                        posB=prob_pos[v],
                        arrowstyle="-|>",
                        mutation_scale=14,
                        connectionstyle="arc3,rad=0.12",
                        linewidth=lw,
                        color=color,
                        alpha=alpha,
                        shrinkA=18,
                        shrinkB=18,
                        zorder=1
                    )

                ax.add_patch(patch)

            sm = mpl.cm.ScalarMappable(cmap=prob_cmap, norm=prob_norm)
            sm.set_array([])

            cax = fig.add_axes([0.90, 0.25, 0.015, 0.50])
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_label("Mean transition probability")

            plt.savefig(
                os.path.join(save_dir, f"{filename_stem}.pdf"),
                format="pdf",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

        def plot_history_cluster_comparison(
            comparison_name,
            comparison_output,
            group_a_label,
            group_a_filter,
            group_b_label,
            group_b_filter,
            difference_label,
            heatmap_title,
            circle_title
        ):

            labelled = []

            group_a = transitions[group_a_filter(transitions)].copy()
            group_a["history_group"] = group_a_label
            labelled.append(group_a)

            group_b = transitions[group_b_filter(transitions)].copy()
            group_b["history_group"] = group_b_label
            labelled.append(group_b)

            history_transitions = pd.concat(labelled, ignore_index=True)

            if history_transitions.empty:
                print(f"No {comparison_name} cluster transitions found.")
                return

            history_transitions.to_csv(
                os.path.join(
                    comparison_output,
                    f"GS_cluster_transition_{comparison_name}_raw.csv"
                ),
                index=False
            )

            counts_history = (
                history_transitions
                .groupby(["file", "history_group", "cluster", "next_cluster"])
                .size()
                .reset_index(name="count")
            )

            present_file_groups = (
                history_transitions[["file", "history_group"]]
                .drop_duplicates()
                .dropna()
            )

            full_history_index = pd.MultiIndex.from_tuples(
                [
                    (row.file, row.history_group, cluster_id, next_cluster_id)
                    for row in present_file_groups.itertuples(index=False)
                    for cluster_id in clusters
                    for next_cluster_id in clusters
                ],
                names=["file", "history_group", "cluster", "next_cluster"]
            )

            counts_history = (
                counts_history
                .set_index(["file", "history_group", "cluster", "next_cluster"])
                .reindex(full_history_index, fill_value=0)
                .reset_index()
            )

            counts_history["row_total"] = (
                counts_history
                .groupby(["file", "history_group", "cluster"])["count"]
                .transform("sum")
            )
            counts_history["transition_probability"] = (
                counts_history["count"] / counts_history["row_total"]
            )
            counts_history["transition_probability"] = counts_history["transition_probability"].fillna(0)

            counts_history.to_csv(
                os.path.join(
                    comparison_output,
                    f"GS_cluster_transition_{comparison_name}_probabilities_per_video.csv"
                ),
                index=False
            )

            history_stats = []

            for cluster_id in clusters:
                for next_cluster_id in clusters:

                    sub = counts_history[
                        (counts_history["cluster"] == cluster_id) &
                        (counts_history["next_cluster"] == next_cluster_id)
                    ]

                    wide = (
                        sub.pivot(index="file", columns="history_group", values="transition_probability")
                        .dropna(subset=[group_a_label, group_b_label])
                    )

                    a = wide[group_a_label]
                    b = wide[group_b_label]
                    diff = a - b

                    if len(diff) >= 2 and not np.allclose(diff, 0):
                        stat, p = wilcoxon(a, b, alternative="two-sided")
                    elif len(diff) >= 2 and np.allclose(diff, 0):
                        stat, p = 0, 1.0
                    else:
                        stat, p = np.nan, np.nan

                    history_stats.append({
                        "cluster": cluster_id,
                        "next_cluster": next_cluster_id,
                        "W": stat,
                        "p": p,
                        f"{group_a_label}_mean": a.mean(),
                        f"{group_b_label}_mean": b.mean(),
                        "mean_diff": diff.mean(),
                        "n_videos": len(diff)
                    })

            history_stats = pd.DataFrame(history_stats)
            mask = history_stats["p"].notna()

            if mask.any():
                history_stats.loc[mask, "p_adj"] = multipletests(
                    history_stats.loc[mask, "p"],
                    method="fdr_bh"
                )[1]
            else:
                history_stats["p_adj"] = np.nan

            history_stats.to_csv(
                os.path.join(
                    comparison_output,
                    f"GS_cluster_transition_{comparison_name}_statistics.csv"
                ),
                index=False
            )

            history_summary = (
                counts_history
                .groupby(["history_group", "cluster", "next_cluster"])
                .agg(
                    mean_probability=("transition_probability", "mean"),
                    sd_probability=("transition_probability", "std"),
                    mean_count=("count", "mean"),
                    sd_count=("count", "std")
                )
                .reset_index()
            )

            history_summary.to_csv(
                os.path.join(
                    comparison_output,
                    f"GS_cluster_transition_{comparison_name}_summary.csv"
                ),
                index=False
            )

            M_A = (
                history_summary[history_summary["history_group"] == group_a_label]
                .pivot(index="cluster", columns="next_cluster", values="mean_probability")
                .reindex(index=clusters, columns=clusters)
                .fillna(0)
            )

            M_B = (
                history_summary[history_summary["history_group"] == group_b_label]
                .pivot(index="cluster", columns="next_cluster", values="mean_probability")
                .reindex(index=clusters, columns=clusters)
                .fillna(0)
            )

            M_A.to_csv(os.path.join(comparison_output, f"GS_cluster_transition_{comparison_name}_probability_{group_a_label}.csv"))
            M_B.to_csv(os.path.join(comparison_output, f"GS_cluster_transition_{comparison_name}_probability_{group_b_label}.csv"))

            short_group_a_label = group_a_label.replace("_with_", "_").replace("_then_", "_")
            short_group_b_label = group_b_label.replace("_with_", "_").replace("_then_", "_")

            D_history = (
                history_stats
                .pivot(index="cluster", columns="next_cluster", values="mean_diff")
                .reindex(index=clusters, columns=clusters)
            )

            D_history.to_csv(
                os.path.join(
                    comparison_output,
                    f"GS_cluster_transition_{comparison_name}_difference.csv"
                )
            )

            finite_diff_values = D_history.to_numpy()[np.isfinite(D_history.to_numpy())]

            if finite_diff_values.size == 0:
                print(f"No paired per-video {comparison_name} cluster comparisons found.")
                return

            M_A_mean = (
                history_stats
                .pivot(index="cluster", columns="next_cluster", values=f"{group_a_label}_mean")
                .reindex(index=clusters, columns=clusters)
                .fillna(0)
            )

            M_B_mean = (
                history_stats
                .pivot(index="cluster", columns="next_cluster", values=f"{group_b_label}_mean")
                .reindex(index=clusters, columns=clusters)
                .fillna(0)
            )

            prob_max_history = float(np.nanmax([
                np.nanmax(M_A_mean.to_numpy()),
                np.nanmax(M_B_mean.to_numpy())
            ]))
            if prob_max_history == 0:
                prob_max_history = 1e-6

            plot_probability_circlegraph(
                M_A_mean,
                f"GS cluster transition likelihood: {group_a_label}",
                f"GS_cluster_transition_{comparison_name}_{short_group_a_label}_circlegraph",
                comparison_output,
                prob_max=prob_max_history
            )

            plot_probability_circlegraph(
                M_B_mean,
                f"GS cluster transition likelihood: {group_b_label}",
                f"GS_cluster_transition_{comparison_name}_{short_group_b_label}_circlegraph",
                comparison_output,
                prob_max=prob_max_history
            )

            D_history = D_history.fillna(0)

            history_lim = float(np.nanmax(np.abs(finite_diff_values)))
            if history_lim == 0:
                history_lim = 1e-6

            plt.figure(figsize=(10, 8))
            sns.heatmap(
                D_history,
                cmap="RdBu",
                center=0,
                vmin=-history_lim,
                vmax=history_lim,
                square=True,
                cbar_kws={"label": difference_label}
            )

            plt.xlabel("Next cluster")
            plt.ylabel("Current cluster")
            plt.title(heatmap_title)
            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    comparison_output,
                    f"GS_cluster_transition_{comparison_name}_difference_heatmap.png"
                ),
                dpi=300,
                bbox_inches="tight"
            )

            plt.savefig(
                os.path.join(
                    comparison_output,
                    f"GS_cluster_transition_{comparison_name}_difference_heatmap.pdf"
                ),
                format="pdf",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            history_norm = mpl.colors.TwoSlopeNorm(
                vmin=-history_lim,
                vcenter=0,
                vmax=history_lim
            )

            G_history = diff_matrix_to_digraph(D_history, thresh=0.05)
            history_pos = nx.circular_layout(list(D_history.index))

            def history_lw(w, min_w=0.3, max_w=6.0):
                return min_w + (abs(w) / history_lim) * (max_w - min_w)

            def history_alpha(w, min_a=0.15, max_a=0.95):
                return min_a + (abs(w) / history_lim) * (max_a - min_a)

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.set_title(circle_title)
            ax.axis("off")

            nx.draw_networkx_nodes(
                G_history,
                history_pos,
                ax=ax,
                node_size=700,
                node_color="lightgray",
                edgecolors="black"
            )

            nx.draw_networkx_labels(
                G_history,
                history_pos,
                ax=ax,
                font_size=9
            )

            for u, v, d in G_history.edges(data=True):

                w = float(d["weight"])
                color = diff_cmap(history_norm(w))
                lw = history_lw(w)
                alpha = history_alpha(w)

                if u == v:
                    x, y = history_pos[u]
                    vec = np.array([x, y], dtype=float)
                    r = np.linalg.norm(vec)

                    if r == 0:
                        continue

                    unit = vec / r
                    perp = np.array([-unit[1], unit[0]])

                    center = vec + 0.10 * unit
                    start = center - 0.08 * perp + 0.012 * unit
                    end = center + 0.08 * perp + 0.012 * unit

                    patch = FancyArrowPatch(
                        posA=start,
                        posB=end,
                        arrowstyle="-|>",
                        mutation_scale=14,
                        connectionstyle="arc3,rad=0.8",
                        linewidth=lw,
                        color=color,
                        alpha=alpha,
                        zorder=2
                    )

                else:
                    patch = FancyArrowPatch(
                        posA=history_pos[u],
                        posB=history_pos[v],
                        arrowstyle="-|>",
                        mutation_scale=14,
                        connectionstyle="arc3,rad=0.12",
                        linewidth=lw,
                        color=color,
                        alpha=alpha,
                        shrinkA=18,
                        shrinkB=18,
                        zorder=1
                    )

                ax.add_patch(patch)

            sm = mpl.cm.ScalarMappable(cmap=diff_cmap, norm=history_norm)
            sm.set_array([])

            cax = fig.add_axes([0.90, 0.25, 0.015, 0.50])
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_label(difference_label)

            plt.savefig(
                os.path.join(
                    comparison_output,
                    f"GS_cluster_transition_{comparison_name}_difference_circlegraph.pdf"
                ),
                format="pdf",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

        plot_history_cluster_comparison(
            comparison_name="within_type",
            comparison_output=within_type_output,
            group_a_label="G_with_G_then_G",
            group_a_filter=lambda d: (
                (d["larva_type"] == "G") &
                (d["partner_type"] == "G") &
                (d["next_partner_type"] == "G")
            ),
            group_b_label="S_with_S_then_S",
            group_b_filter=lambda d: (
                (d["larva_type"] == "S") &
                (d["partner_type"] == "S") &
                (d["next_partner_type"] == "S")
            ),
            difference_label="P(G with G then G) - P(S with S then S)",
            heatmap_title="GS cluster transition difference: within-type history",
            circle_title="GS cluster transition likelihood: within-type history"
        )

        plot_history_cluster_comparison(
            comparison_name="mixed_type",
            comparison_output=mixed_output,
            group_a_label="G_with_S_then_S",
            group_a_filter=lambda d: (
                (d["larva_type"] == "G") &
                (d["partner_type"] == "S") &
                (d["next_partner_type"] == "S")
            ),
            group_b_label="S_with_G_then_G",
            group_b_filter=lambda d: (
                (d["larva_type"] == "S") &
                (d["partner_type"] == "G") &
                (d["next_partner_type"] == "G")
            ),
            difference_label="P(G with S then S) - P(S with G then G)",
            heatmap_title="GS cluster transition difference: mixed-type history",
            circle_title="GS cluster transition likelihood: mixed-type history"
        )

        plot_history_cluster_comparison(
            comparison_name="mixed_2",
            comparison_output=mixed_2_output,
            group_a_label="G_with_S_then_G",
            group_a_filter=lambda d: (
                (d["larva_type"] == "G") &
                (d["partner_type"] == "S") &
                (d["next_partner_type"] == "G")
            ),
            group_b_label="S_with_G_then_S",
            group_b_filter=lambda d: (
                (d["larva_type"] == "S") &
                (d["partner_type"] == "G") &
                (d["next_partner_type"] == "S")
            ),
            difference_label="P(G with S then G) - P(S with G then S)",
            heatmap_title="GS cluster transition difference: mixed-2 history",
            circle_title="GS cluster transition likelihood: mixed-2 history"
        )

        plot_history_cluster_comparison(
            comparison_name="group",
            comparison_output=group_output,
            group_a_label="G_with_S_then_G",
            group_a_filter=lambda d: (
                (d["larva_type"] == "G") &
                (d["partner_type"] == "S") &
                (d["next_partner_type"] == "G")
            ),
            group_b_label="G_with_G_then_G",
            group_b_filter=lambda d: (
                (d["larva_type"] == "G") &
                (d["partner_type"] == "G") &
                (d["next_partner_type"] == "G")
            ),
            difference_label="P(G with S then G) - P(G with G then G)",
            heatmap_title="GS cluster transition difference: group tracks, switched vs stayed group",
            circle_title="GS cluster transition likelihood: group tracks, switched vs stayed group"
        )

        plot_history_cluster_comparison(
            comparison_name="isolated",
            comparison_output=isolated_output,
            group_a_label="S_with_G_then_S",
            group_a_filter=lambda d: (
                (d["larva_type"] == "S") &
                (d["partner_type"] == "G") &
                (d["next_partner_type"] == "S")
            ),
            group_b_label="S_with_S_then_S",
            group_b_filter=lambda d: (
                (d["larva_type"] == "S") &
                (d["partner_type"] == "S") &
                (d["next_partner_type"] == "S")
            ),
            difference_label="P(S with G then S) - P(S with S then S)",
            heatmap_title="GS cluster transition difference: isolated tracks, switched vs stayed isolated",
            circle_title="GS cluster transition likelihood: isolated tracks, switched vs stayed isolated"
        )




    def cluster_transition_matrix_G_vs_S(self):

        df = self.df.copy()
        cluster_name = self.cluster_name

        output = os.path.join(self.directory, "cluster_transition_matrix_G_vs_S")
        os.makedirs(output, exist_ok=True)

        # G and S conditions only, one row per interaction
        df = df[
            (df["condition"].isin(["G", "S"])) &
            (np.isclose(df["Normalized Frame"], 0))
        ].copy()

        if df.empty:
            print("No G/S interactions found at Normalized Frame == 0.")
            return

        def parse_pair(pair):
            if pd.isna(pair):
                return None

            if isinstance(pair, str):
                pair = pair.replace("(", "").replace(")", "").replace(" ", "")
                id1, id2 = pair.split(",")
                return tuple(sorted((int(id1), int(id2))))

            id1, id2 = pair
            return tuple(sorted((int(id1), int(id2))))

        df["parsed_pair"] = df["Interaction Pair"].apply(parse_pair)
        df = df.dropna(subset=["parsed_pair"])

        df["cluster"] = df[cluster_name].astype(int)

        # -------------------------------------------------------
        # Expand to one row per larva per interaction
        # -------------------------------------------------------

        rows = []

        for _, row in df.iterrows():

            id1, id2 = row["parsed_pair"]

            rows.append({
                "file": row["file"],
                "condition": row["condition"],
                "interaction_id": row["interaction_id"],
                "track_id": id1,
                "time": row["Frame"],
                "cluster": row["cluster"]
            })

            rows.append({
                "file": row["file"],
                "condition": row["condition"],
                "interaction_id": row["interaction_id"],
                "track_id": id2,
                "time": row["Frame"],
                "cluster": row["cluster"]
            })

        expanded = pd.DataFrame(rows)

        expanded = expanded.sort_values(
            ["file", "track_id", "time", "interaction_id"]
        )

        expanded["next_cluster"] = (
            expanded
            .groupby(["file", "track_id"])["cluster"]
            .shift(-1)
        )

        transitions = expanded.dropna(subset=["next_cluster"]).copy()
        transitions["next_cluster"] = transitions["next_cluster"].astype(int)

        if transitions.empty:
            print("No cluster transitions found.")
            return

        raw_path = os.path.join(output, "G_vs_S_cluster_transitions_raw.csv")
        transitions.to_csv(raw_path, index=False)

        clusters = sorted(
            set(transitions["cluster"].unique()) |
            set(transitions["next_cluster"].unique())
        )

        # -------------------------------------------------------
        # Count transitions per video and condition
        # -------------------------------------------------------

        counts = (
            transitions
            .groupby(["file", "condition", "cluster", "next_cluster"])
            .size()
            .reset_index(name="count")
        )

        # IMPORTANT:
        # Each file has only one real condition.
        # Do NOT create fake S rows for G files or fake G rows for S files.
        file_condition_pairs = (
            transitions[["file", "condition"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        cluster_pairs = pd.DataFrame(
            [(i, j) for i in clusters for j in clusters],
            columns=["cluster", "next_cluster"]
        )

        full_index_df = file_condition_pairs.merge(cluster_pairs, how="cross")

        counts = (
            counts
            .set_index(["file", "condition", "cluster", "next_cluster"])
            .reindex(
                pd.MultiIndex.from_frame(
                    full_index_df[["file", "condition", "cluster", "next_cluster"]]
                ),
                fill_value=0
            )
            .reset_index()
        )

        # Row-normalise within each video, condition, and current cluster
        counts["row_total"] = (
            counts
            .groupby(["file", "condition", "cluster"])["count"]
            .transform("sum")
        )

        counts["transition_probability"] = (
            counts["count"] / counts["row_total"]
        ).fillna(0)

        per_video_path = os.path.join(
            output,
            "G_vs_S_cluster_transition_matrix_per_video.csv"
        )
        counts.to_csv(per_video_path, index=False)

        # -------------------------------------------------------
        # Average across videos
        # -------------------------------------------------------

        summary = (
            counts
            .groupby(["condition", "cluster", "next_cluster"])
            .agg(
                mean_probability=("transition_probability", "mean"),
                sd_probability=("transition_probability", "std"),
                mean_count=("count", "mean"),
                sd_count=("count", "std")
            )
            .reset_index()
        )

        summary_path = os.path.join(
            output,
            "G_vs_S_cluster_transition_matrix_summary.csv"
        )
        summary.to_csv(summary_path, index=False)

        condition_order = ["S", "G"]
        matrices = {}

        for condition in condition_order:
            matrices[condition] = (
                summary[summary["condition"] == condition]
                .pivot(
                    index="cluster",
                    columns="next_cluster",
                    values="mean_probability"
                )
                .reindex(index=clusters, columns=clusters)
                .fillna(0)
            )

        # -------------------------------------------------------
        # Heatmaps: S vs G
        # -------------------------------------------------------

        vmax = max(
            matrices["S"].to_numpy().max(),
            matrices["G"].to_numpy().max()
        )

        if vmax == 0:
            vmax = 1e-6

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, condition in zip(axes, condition_order):

            sns.heatmap(
                matrices[condition],
                cmap="viridis",
                vmin=0,
                vmax=vmax,
                square=True,
                cbar=False,
                ax=ax
            )

            ax.set_title("Isolated" if condition == "S" else "Group housed")
            ax.set_xlabel("Next cluster")
            ax.set_ylabel("Current cluster")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        fig.subplots_adjust(right=0.88, wspace=0.25)
        cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])

        sm = mpl.cm.ScalarMappable(
            cmap="viridis",
            norm=mpl.colors.Normalize(vmin=0, vmax=vmax)
        )
        sm.set_array([])

        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Mean transition probability")

        plt.savefig(
            os.path.join(output, "G_vs_S_cluster_transition_heatmaps.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.savefig(
            os.path.join(output, "G_vs_S_cluster_transition_heatmaps.pdf"),
            format="pdf",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        # -------------------------------------------------------
        # Circle graphs: S and G transition probabilities
        # -------------------------------------------------------

        M_S = matrices["S"].copy()
        M_G = matrices["G"].copy()

        all_clusters = sorted(
            set(M_S.index) | set(M_S.columns) |
            set(M_G.index) | set(M_G.columns)
        )

        M_S = M_S.reindex(index=all_clusters, columns=all_clusters, fill_value=0)
        M_G = M_G.reindex(index=all_clusters, columns=all_clusters, fill_value=0)

        global_max = max(M_S.to_numpy().max(), M_G.to_numpy().max())

        if global_max == 0:
            global_max = 1e-6

        pos = nx.circular_layout(all_clusters)

        cmap = plt.cm.viridis
        norm = mpl.colors.Normalize(vmin=0, vmax=global_max)

        def matrix_to_digraph(M, thresh=0.0):
            G = nx.DiGraph()

            for c in M.index:
                G.add_node(int(c))

            for i in M.index:
                for j in M.columns:
                    w = float(M.loc[i, j])
                    if w > thresh:
                        G.add_edge(int(i), int(j), weight=w)

            return G

        def draw_self_loop(ax, xy, w):

            x, y = xy
            v = np.array([x, y], dtype=float)
            r = np.linalg.norm(v)

            if r == 0:
                return

            u = v / r
            perp = np.array([-u[1], u[0]])

            c = v + 0.10 * u

            start = c - 0.08 * perp + 0.012 * u
            end = c + 0.08 * perp + 0.012 * u

            lw = 1 + (w / global_max) * 5
            alpha = 0.4 + (w / global_max) * 0.6
            color = cmap(norm(w))

            patch = FancyArrowPatch(
                posA=start,
                posB=end,
                arrowstyle="-|>",
                mutation_scale=14,
                connectionstyle="arc3,rad=0.8",
                linewidth=lw,
                color=color,
                alpha=alpha,
                zorder=2
            )

            ax.add_patch(patch)

        def draw_edge(ax, xy_u, xy_v, w):

            lw = 1 + (w / global_max) * 5
            alpha = 0.4 + (w / global_max) * 0.6
            color = cmap(norm(w))

            patch = FancyArrowPatch(
                posA=xy_u,
                posB=xy_v,
                arrowstyle="-|>",
                mutation_scale=14,
                connectionstyle="arc3,rad=0.12",
                linewidth=lw,
                color=color,
                alpha=alpha,
                shrinkA=18,
                shrinkB=18,
                zorder=1
            )

            ax.add_patch(patch)

        def draw_transition_circle(ax, G, title):

            ax.set_title(title)
            ax.axis("off")

            nx.draw_networkx_nodes(
                G,
                pos,
                ax=ax,
                node_size=700,
                node_color="lightgray",
                edgecolors="black"
            )

            nx.draw_networkx_labels(
                G,
                pos,
                ax=ax,
                font_size=9
            )

            for u, v, d in G.edges(data=True):

                w = float(d["weight"])

                if u == v:
                    draw_self_loop(ax, pos[u], w)
                else:
                    draw_edge(ax, pos[u], pos[v], w)

        G_S = matrix_to_digraph(M_S, thresh=0.10)
        G_G = matrix_to_digraph(M_G, thresh=0.10)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        draw_transition_circle(axes[0], G_S, "Isolated")
        draw_transition_circle(axes[1], G_G, "Group housed")

        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        fig.subplots_adjust(right=0.88, wspace=0.25)
        cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])

        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Mean transition probability")

        plt.savefig(
            os.path.join(output, "G_vs_S_cluster_transition_circlegraphs.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.savefig(
            os.path.join(output, "G_vs_S_cluster_transition_circlegraphs.pdf"),
            format="pdf",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        # -------------------------------------------------------
        # Difference heatmap: G - S
        # Blue = more group housed
        # Red = more isolated
        # -------------------------------------------------------

        P_diff = M_G - M_S

        lim = float(np.nanmax(np.abs(P_diff.to_numpy())))

        if lim == 0:
            lim = 1e-6

        diff_cmap = plt.cm.RdBu

        plt.figure(figsize=(10, 8))

        sns.heatmap(
            P_diff,
            cmap=diff_cmap,
            center=0,
            vmin=-lim,
            vmax=lim,
            square=True,
            cbar_kws={"label": "P(group) - P(isolated)"}
        )

        plt.xlabel("Next cluster")
        plt.ylabel("Current cluster")
        plt.title("Transition difference: group housed - isolated")
        plt.tight_layout()

        plt.savefig(
            os.path.join(output, "G_vs_S_cluster_transition_difference_heatmap.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.savefig(
            os.path.join(output, "G_vs_S_cluster_transition_difference_heatmap.pdf"),
            format="pdf",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        # -------------------------------------------------------
        # Difference circle graph: G - S
        # -------------------------------------------------------

        diff_norm = mpl.colors.TwoSlopeNorm(
            vmin=-lim,
            vcenter=0,
            vmax=lim
        )

        def diff_matrix_to_digraph(D, thresh=0.02):

            G = nx.DiGraph()

            for c in D.index:
                G.add_node(int(c))

            for i in D.index:
                for j in D.columns:
                    w = float(D.loc[i, j])

                    if abs(w) >= thresh:
                        G.add_edge(int(i), int(j), weight=w)

            return G

        G_diff = diff_matrix_to_digraph(P_diff, thresh=0.05)

        def diff_lw(w):
            return 1 + (abs(w) / lim) * 7

        def diff_alpha(w):
            return 0.4 + (abs(w) / lim) * 0.6

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        ax.set_title("Transition difference: group housed - isolated")
        ax.axis("off")

        nx.draw_networkx_nodes(
            G_diff,
            pos,
            ax=ax,
            node_size=700,
            node_color="lightgray",
            edgecolors="black"
        )

        nx.draw_networkx_labels(
            G_diff,
            pos,
            ax=ax,
            font_size=9
        )

        for u, v, d in G_diff.edges(data=True):

            w = float(d["weight"])
            color = diff_cmap(diff_norm(w))
            lw = diff_lw(w)
            alpha = diff_alpha(w)

            if u == v:

                x, y = pos[u]
                vec = np.array([x, y], dtype=float)
                r = np.linalg.norm(vec)

                if r == 0:
                    continue

                unit = vec / r
                perp = np.array([-unit[1], unit[0]])

                center = vec + 0.10 * unit
                start = center - 0.08 * perp + 0.012 * unit
                end = center + 0.08 * perp + 0.012 * unit

                patch = FancyArrowPatch(
                    posA=start,
                    posB=end,
                    arrowstyle="-|>",
                    mutation_scale=14,
                    connectionstyle="arc3,rad=0.8",
                    linewidth=lw,
                    color=color,
                    alpha=alpha,
                    zorder=2
                )

            else:

                patch = FancyArrowPatch(
                    posA=pos[u],
                    posB=pos[v],
                    arrowstyle="-|>",
                    mutation_scale=14,
                    connectionstyle="arc3,rad=0.12",
                    linewidth=lw,
                    color=color,
                    alpha=alpha,
                    shrinkA=18,
                    shrinkB=18,
                    zorder=1
                )

            ax.add_patch(patch)

        sm = mpl.cm.ScalarMappable(cmap=diff_cmap, norm=diff_norm)
        sm.set_array([])

        cax = fig.add_axes([0.90, 0.25, 0.015, 0.50])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("P(group) - P(isolated)")

        plt.savefig(
            os.path.join(output, "G_vs_S_cluster_transition_difference_circlegraph.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.savefig(
            os.path.join(output, "G_vs_S_cluster_transition_difference_circlegraph.pdf"),
            format="pdf",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        print(f"Saved raw transitions: {raw_path}")
        print(f"Saved per-video transition matrix: {per_video_path}")
        print(f"Saved summary: {summary_path}")
    


    def GS_deviations(self):

        df = self.df.copy()
        cluster_name = self.cluster_name

        output = os.path.join(self.directory, "GS_deviations")
        os.makedirs(output, exist_ok=True)

        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype']  = 42

        # one row per interaction, GS only
        df_start = df[
            (df["condition"] == "GS") &
            (np.isclose(df["Normalized Frame"], 0))
        ].copy()

        df_all = df[df["condition"] == "GS"].copy()

        if df_start.empty:
            print("No GS data found.")
            return

        def parse_pair(pair):
            if pd.isna(pair):
                return None

            if isinstance(pair, str):
                pair = pair.replace("(", "").replace(")", "").replace(" ", "")
                id1, id2 = pair.split(",")
                return tuple(sorted((int(id1), int(id2))))

            id1, id2 = pair
            return tuple(sorted((int(id1), int(id2))))

        def larva_type(track_id):
            return "S" if int(track_id) <= 4 else "G"

        df_start["parsed_pair"] = df_start["Interaction Pair"].apply(parse_pair)
        df_start = df_start.dropna(subset=["parsed_pair"])

        rows = []

        for _, row in df_start.iterrows():

            id1, id2 = row["parsed_pair"]

            rows.append({
                "file": row["file"],
                "interaction_id": row["interaction_id"],
                cluster_name: row[cluster_name],
                "larva_type": larva_type(id1)
            })

            rows.append({
                "file": row["file"],
                "interaction_id": row["interaction_id"],
                cluster_name: row[cluster_name],
                "larva_type": larva_type(id2)
            })

        expanded = pd.DataFrame(rows)

        # --------------------------------------------------
        # OBSERVED - EXPECTED DEVIATION: G larvae vs S larvae
        # --------------------------------------------------

        cluster_counts = (
            expanded
            .groupby([cluster_name, "larva_type"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=["G", "S"], fill_value=0)
        )

        cluster_counts = cluster_counts.loc[cluster_counts.sum(axis=1) > 0]

        total_group = cluster_counts["G"].sum()
        total_iso = cluster_counts["S"].sum()
        total_all = total_group + total_iso

        expected_group = total_group / total_all

        observed_group_frac = (
            cluster_counts["G"] /
            (cluster_counts["G"] + cluster_counts["S"]).replace({0: np.nan})
        ).fillna(0)

        deviation = observed_group_frac - expected_group
        deviation_sorted = deviation.sort_values()

        colors = ["C1" if val < 0 else "C0" for val in deviation_sorted.values]

        deviation_df = deviation_sorted.reset_index()
        deviation_df.columns = [cluster_name, "deviation_G_minus_expected"]

        deviation_df.to_csv(
            os.path.join(output, "GS_deviation_G_vs_S.csv"),
            index=False
        )

        cluster_counts.to_csv(
            os.path.join(output, "GS_cluster_counts_G_vs_S.csv")
        )

        # binomial test per cluster
        results = []

        for cluster_id, row in cluster_counts.iterrows():

            k = row["G"]
            n = row["G"] + row["S"]

            if n > 0:
                res = binomtest(k, n, expected_group, alternative="two-sided")
                results.append((cluster_id, res.pvalue))
            else:
                results.append((cluster_id, np.nan))

        pvals = pd.DataFrame(results, columns=["cluster_id", "p_value"])

        valid = pvals["p_value"].notna()

        pvals["p_adj"] = np.nan

        if valid.sum() > 0:
            pvals.loc[valid, "p_adj"] = multipletests(
                pvals.loc[valid, "p_value"],
                method="fdr_bh"
            )[1]

        pvals.to_csv(
            os.path.join(output, "GS_deviation_pvals_G_vs_S.csv"),
            index=False,
            float_format="%.10f"
        )

        # --------------------------------------------------
        # CONTACT FRAME SUMMARY
        # --------------------------------------------------

        interaction_contact_summary = []

        for cluster_id in sorted(df_all[cluster_name].dropna().unique()):

            cluster_df = df_all[df_all[cluster_name] == cluster_id]

            for inter_id in cluster_df["interaction_id"].dropna().unique():

                inter_df = cluster_df[cluster_df["interaction_id"] == inter_id]

                n_close = (inter_df["min_distance"] < 1).sum()

                interaction_contact_summary.append({
                    "cluster": cluster_id,
                    "interaction_id": inter_id,
                    "frames_below_1mm": n_close
                })

        df_interaction_contact = pd.DataFrame(interaction_contact_summary)

        # --------------------------------------------------
        # PLOT
        # --------------------------------------------------

        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[6, 0.3], hspace=0.05)

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        x_labels = deviation_sorted.index.astype(str)
        x_pos = np.arange(len(x_labels))

        ax1.bar(
            x_labels,
            deviation_sorted.values,
            color=colors,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5
        )

        ax1.axhline(0, color="k", linestyle="--", linewidth=1)

        p_map = pvals.set_index("cluster_id")["p_adj"]

        def stars(p):
            if p < 0.001:
                return "***"
            if p < 0.01:
                return "**"
            if p < 0.05:
                return "*"
            return ""

        ymin, ymax = ax1.get_ylim()
        dy = 0.015 * (ymax - ymin)

        for i, cid in enumerate(deviation_sorted.index):

            p = p_map.get(cid, np.nan)

            if pd.notna(p):

                s = stars(p)

                if s:
                    y = deviation_sorted.loc[cid]

                    ax1.text(
                        i,
                        y + (dy if y >= 0 else -dy),
                        s,
                        ha="center",
                        va="bottom" if y >= 0 else "top",
                        fontsize=10,
                        fontweight="bold",
                        color="black"
                    )

        ax1.set_title("GS condition: cluster deviation for G-experience vs S-experience larvae",
                    fontsize=14, fontweight="bold", pad=15)
        ax1.set_ylabel("Deviation from expected G fraction")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels("")

        # contact heat strip
        mean_frames = (
            df_interaction_contact
            .groupby("cluster")["frames_below_1mm"]
            .mean()
        )

        mean_frames.index = mean_frames.index.astype(deviation_sorted.index.dtype)

        avg_contact_aligned = mean_frames.reindex(
            deviation_sorted.index,
            fill_value=0
        )

        heat = avg_contact_aligned.to_numpy()[np.newaxis, :]

        ax2.set_xlim(ax1.get_xlim())

        colors_heat = ["lightskyblue", "mediumseagreen", "darkgreen"]
        my_cmap = LinearSegmentedColormap.from_list(
            "greenblue_custom",
            colors_heat
        )

        im = ax2.imshow(
            heat,
            aspect="auto",
            interpolation="nearest",
            vmin=0,
            cmap=my_cmap
        )

        ax2.set_yticks([0])
        ax2.set_yticklabels(["Average Contact\nFrames"])
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(x_labels, fontweight="bold", fontsize=14)
        ax2.tick_params(axis="x", pad=15)

        for spine in ax2.spines.values():
            spine.set_visible(False)

        cbar = fig.colorbar(
            im,
            ax=[ax1, ax2],
            fraction=0.03,
            pad=0.02,
            location="right"
        )

        cbar.set_label("Average Contact Frames", rotation=270, labelpad=15)

        plt.tight_layout()

        plt.savefig(
            os.path.join(output, "GS_deviations_G_vs_S.png"),
            dpi=300,
            bbox_inches="tight"
        )

        plt.savefig(
            os.path.join(output, "GS_deviations_G_vs_S.pdf"),
            format="pdf",
            bbox_inches="tight",
            dpi=300,
            transparent=True
        )

        plt.close()

        print("Saved GS G-vs-S deviation analysis.")
    





    def GS_social_experience_over_time_by_cluster(self, bin_size=600):

        df = self.df.copy()
        cluster_name = self.cluster_name

        output = os.path.join(self.directory, "GS_social_experience_over_time_by_cluster")
        os.makedirs(output, exist_ok=True)

        # GS only, one row per interaction
        df = df[
            (df["condition"] == "GS") &
            (np.isclose(df["Normalized Frame"], 0))
        ].copy()

        if df.empty:
            print("No GS interactions found at Normalized Frame == 0.")
            return

        def parse_pair(pair):
            if pd.isna(pair):
                return None

            if isinstance(pair, str):
                pair = pair.replace("(", "").replace(")", "").replace(" ", "")
                id1, id2 = pair.split(",")
                return tuple(sorted((int(id1), int(id2))))

            id1, id2 = pair
            return tuple(sorted((int(id1), int(id2))))

        def get_social_experience(pair):
            if pair is None:
                return np.nan

            id1, id2 = pair

            if id1 <= 4 and id2 <= 4:
                return "S-S"
            elif id1 >= 5 and id2 >= 5:
                return "G-G"
            else:
                return "G-S"

        df["parsed_pair"] = df["Interaction Pair"].apply(parse_pair)
        df = df.dropna(subset=["parsed_pair"])

        df["social_experience"] = df["parsed_pair"].apply(get_social_experience)
        df = df.dropna(subset=["social_experience"])

        df["cluster"] = df[cluster_name].astype(int)

        # time bins using original Frame
        max_frame = int(np.ceil(df["Frame"].max() / bin_size) * bin_size)

        bins = np.arange(0, max_frame + bin_size, bin_size)
        labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]

        df["time_bin"] = pd.cut(
            df["Frame"],
            bins=bins,
            labels=labels,
            right=False,
            include_lowest=True
        )

        social_order = ["S-S", "G-S", "G-G"]
        clusters = sorted(df["cluster"].dropna().unique())
        files = sorted(df["file"].dropna().unique())

        # count interactions per video, cluster, social experience, time bin
        counts = (
            df
            .groupby(["file", "cluster", "social_experience", "time_bin"])
            .size()
            .reset_index(name="count")
        )

        full_index = pd.MultiIndex.from_product(
            [files, clusters, social_order, labels],
            names=["file", "cluster", "social_experience", "time_bin"]
        )


        counts = (
            counts
            .set_index(["file", "cluster", "social_experience", "time_bin"])
            .reindex(full_index, fill_value=0)
            .reset_index()
        )

        counts["total"] = (
            counts
            .groupby(["file", "social_experience", "time_bin"])["count"]
            .transform("sum")
        )

        counts["proportion"] = (
            counts["count"] /
            counts["total"]
        ).fillna(0)


        counts.to_csv(
            os.path.join(output, "GS_social_experience_counts_per_video_by_timebin.csv"),
            index=False
        )

        summary = (
            counts
            .groupby(["cluster", "social_experience", "time_bin"])
            .agg(
                mean_proportion=("proportion", "mean"),
                sd_proportion=("proportion", "std")
            )
            .reset_index()
        )

        summary.to_csv(
            os.path.join(output, "GS_social_experience_counts_summary_by_timebin.csv"),
            index=False
        )

        palette = {
            "S-S": "C1",
            "G-S": "mediumseagreen",
            "G-G": "C0"
        }

        n_clusters = len(clusters)
        ncols = 4
        nrows = int(np.ceil(n_clusters / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(ncols * 4, nrows * 3),
            sharey=True
        )

        axes = np.array(axes).flatten()

        for ax, cluster_id in zip(axes, clusters):

            plot_df = summary[summary["cluster"] == cluster_id].copy()

            sns.lineplot(
                data=plot_df,
                x="time_bin",
                y="mean_proportion",
                hue="social_experience",
                hue_order=social_order,
                palette=palette,
                marker="o",
                ax=ax
            )

            ax.set_title(f"Cluster {cluster_id}")
            ax.set_xlabel("Time bin")
            ax.set_ylabel("Mean proportion of interactions")
            ax.tick_params(axis="x", rotation=90)

            if ax.get_legend() is not None:
                ax.get_legend().remove()

        for ax in axes[len(clusters):]:
            ax.axis("off")

        handles, labels_legend = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels_legend,
            title="Social experience",
            loc="upper right",
            bbox_to_anchor=(1.02, 1)
        )

        plt.suptitle(
            "GS condition: S-S, G-S, and G-G interaction proportions over time by cluster",
            fontsize=14,
            fontweight="bold"
        )

        plt.tight_layout(rect=[0, 0, 0.96, 0.96])

        plt.savefig(
            os.path.join(output, "GS_social_experience_over_time_by_cluster.png"),
            dpi=300,
            bbox_inches="tight"
        )

        plt.savefig(
            os.path.join(output, "GS_social_experience_over_time_by_cluster.pdf"),
            format="pdf",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print("Saved GS social-experience over-time cluster plot.")
    



    def G_V_S_duration_transition(self):

        df = self.df.copy()
        cluster_name = self.cluster_name

        output = os.path.join(self.directory, "G_V_S_duration_transition")
        os.makedirs(output, exist_ok=True)

        dur_map = {
            1: "medium",
            2: "short",
            3: "medium",
            4: "medium",
            5: "short",
            6: "medium",
            7: "medium",
            8: "short",
            9: "long",
            10: "short",
            11: "short",
            12: "short",
        }

        dur_order = ["short", "medium", "long"]

        df = df[
            (df["condition"].isin(["G", "S"])) &
            (np.isclose(df["Normalized Frame"], 0))
        ].copy()

        if df.empty:
            print("No G/S interactions found.")
            return


        df["cluster"] = df[cluster_name].astype(int)

        transitions = (
            df
            .sort_values(["condition", "file", "Frame", "interaction_id"])
            .copy()
        )

        transitions["next_cluster"] = (
            transitions
            .groupby(["condition", "file"])["cluster"]
            .shift(-1)
        )

        transitions = transitions.dropna(subset=["next_cluster"]).copy()
        transitions["next_cluster"] = transitions["next_cluster"].astype(int)

        if transitions.empty:
            print("No duration transitions found.")
            return

        transitions["from_dur"] = transitions["cluster"].map(dur_map)
        transitions["to_dur"] = transitions["next_cluster"].map(dur_map)
        transitions = transitions.dropna(subset=["from_dur", "to_dur"])

        transitions.to_csv(
            os.path.join(output, "G_vs_S_duration_transitions_raw.csv"),
            index=False
        )


        # --------------------------------------------------
        # PER-VIDEO TRANSITION MATRICES
        # --------------------------------------------------

        video_probs = []

        for (condition, file), dsub in transitions.groupby(["condition", "file"]):

            C = (
                dsub.groupby(["from_dur", "to_dur"])
                .size()
                .unstack(fill_value=0)
                .reindex(index=dur_order, columns=dur_order, fill_value=0)
            )

            P = C.div(C.sum(axis=1), axis=0).fillna(0)

            for from_dur in dur_order:
                for to_dur in dur_order:

                    video_probs.append({
                        "condition": condition,
                        "file": file,
                        "from_dur": from_dur,
                        "to_dur": to_dur,
                        "probability": P.loc[from_dur, to_dur]
                    })

        video_probs = pd.DataFrame(video_probs)

        video_probs.to_csv(
            os.path.join(output, "duration_transition_probabilities_per_video.csv"),
            index=False
        )



        stats = []

        for from_dur in dur_order:
            for to_dur in dur_order:

                g = video_probs[
                    (video_probs["condition"] == "G") &
                    (video_probs["from_dur"] == from_dur) &
                    (video_probs["to_dur"] == to_dur)
                ]["probability"]

                s = video_probs[
                    (video_probs["condition"] == "S") &
                    (video_probs["from_dur"] == from_dur) &
                    (video_probs["to_dur"] == to_dur)
                ]["probability"]

                if len(g) > 0 and len(s) > 0:

                    stat, p = mannwhitneyu(
                        g,
                        s,
                        alternative="two-sided"
                    )

                else:
                    stat = np.nan
                    p = np.nan

                stats.append({
                    "from_dur": from_dur,
                    "to_dur": to_dur,
                    "U": stat,
                    "p": p,
                    "G_mean": g.mean(),
                    "S_mean": s.mean()
                })

        stats = pd.DataFrame(stats)

        mask = stats["p"].notna()

        stats.loc[mask, "p_adj"] = multipletests(
            stats.loc[mask, "p"],
            method="fdr_bh"
        )[1]

        stats.to_csv(
            os.path.join(output, "duration_transition_statistics.csv"),
            index=False
        )




        def dur_transition_matrix(dsub):
            C = (
                dsub.groupby(["from_dur", "to_dur"])
                .size()
                .unstack(fill_value=0)
                .reindex(index=dur_order, columns=dur_order, fill_value=0)
            )
            P = C.div(C.sum(axis=1), axis=0).fillna(0)
            return C, P

        C_S, P_S = dur_transition_matrix(transitions[transitions["condition"] == "S"])
        C_G, P_G = dur_transition_matrix(transitions[transitions["condition"] == "G"])

        C_S.to_csv(os.path.join(output, "duration_transition_counts_S.csv"))
        C_G.to_csv(os.path.join(output, "duration_transition_counts_G.csv"))
        P_S.to_csv(os.path.join(output, "duration_transition_probability_S.csv"))
        P_G.to_csv(os.path.join(output, "duration_transition_probability_G.csv"))

        # D = P_G - P_S

        D = (
            stats
            .pivot(index="from_dur", columns="to_dur", values="G_mean")
            .reindex(index=dur_order, columns=dur_order)
            -
            stats
            .pivot(index="from_dur", columns="to_dur", values="S_mean")
            .reindex(index=dur_order, columns=dur_order)
        )


        D.to_csv(os.path.join(output, "duration_transition_difference_G_minus_S.csv"))

        lim = float(np.nanmax(np.abs(D.to_numpy())))
        if lim == 0:
            lim = 1e-6

        cmap = plt.cm.RdBu
        norm = mpl.colors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)

        plt.figure(figsize=(5, 4))
        sns.heatmap(
            D,
            cmap=cmap,
            center=0,
            vmin=-lim,
            vmax=lim,
            square=True,
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "P(G) - P(S)"}
        )
        plt.xlabel("Next duration class")
        plt.ylabel("Current duration class")
        plt.title("Duration transition difference: G - S")
        plt.tight_layout()

        plt.savefig(
            os.path.join(output, "duration_transition_difference_heatmap.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.savefig(
            os.path.join(output, "duration_transition_difference_heatmap.pdf"),
            format="pdf",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        nodes = dur_order
        pos = nx.circular_layout(nodes)

        def w_to_lw(w, min_w=0.6, max_w=8.0):
            return min_w + (abs(w) / lim) * (max_w - min_w)

        def w_to_alpha(w, min_a=0.2, max_a=0.95):
            return min_a + (abs(w) / lim) * (max_a - min_a)

        def diff_matrix_to_digraph_labels(D, thresh=0.02):
            G = nx.DiGraph()
            for c in D.index:
                G.add_node(c)
            for i in D.index:
                for j in D.columns:
                    w = float(D.loc[i, j])
                    if abs(w) >= thresh:
                        G.add_edge(i, j, weight=w)
            return G

        Gd = diff_matrix_to_digraph_labels(D, thresh=0.02)

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax.set_title("Duration transition likelihood: G - S")
        ax.axis("off")

        nx.draw_networkx_nodes(Gd, pos, ax=ax, node_size=1200)
        nx.draw_networkx_labels(Gd, pos, ax=ax, font_size=10)

        for u, v, dct in Gd.edges(data=True):
            w = float(dct["weight"])
            color = cmap(norm(w))
            lw = w_to_lw(w)
            alpha = w_to_alpha(w)

            rad = 0.18 if u != v else 0.45

            patch = FancyArrowPatch(
                posA=pos[u],
                posB=pos[v],
                arrowstyle="-|>",
                mutation_scale=16,
                connectionstyle=f"arc3,rad={rad}",
                linewidth=lw,
                color=color,
                alpha=alpha,
                shrinkA=22,
                shrinkB=22
            )

            ax.add_patch(patch)

        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cax = fig.add_axes([0.95, 0.28, 0.02, 0.45])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("P(G) - P(S)")

        plt.savefig(
            os.path.join(output, "duration_transition_difference_circlegraph.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.savefig(
            os.path.join(output, "duration_transition_difference_circlegraph.pdf"),
            format="pdf",
            bbox_inches="tight"
        )
        plt.close()

        print("Saved G vs S duration transition analysis.")




    def G_V_S_apetitive_aversive_transition(self):

        df = self.df.copy()
        cluster_name = self.cluster_name

        output = os.path.join(self.directory, "G_V_S_apetitive_aversive_transition")
        os.makedirs(output, exist_ok=True)

        valence_map = {
            1: "apetitive",
            2: "aversive",
            3: "apetitive",
            4: "apetitive",
            5: "aversive",
            6: "apetitive",
            7: "apetitive",
            8: "aversive",
            9: "apetitive",
            10: "aversive",
            11: "aversive",
            12: "aversive",
        }

        valence_order = ["apetitive", "aversive"]

        df = df[
            (df["condition"].isin(["G", "S"])) &
            (np.isclose(df["Normalized Frame"], 0))
        ].copy()

        if df.empty:
            print("No G/S interactions found.")
            return


        df["cluster"] = df[cluster_name].astype(int)

        transitions = (
            df
            .sort_values(["condition", "file", "Frame", "interaction_id"])
            .copy()
        )

        transitions["next_cluster"] = (
            transitions
            .groupby(["condition", "file"])["cluster"]
            .shift(-1)
        )

        transitions = transitions.dropna(subset=["next_cluster"]).copy()
        transitions["next_cluster"] = transitions["next_cluster"].astype(int)

        if transitions.empty:
            print("No apetitive/aversive transitions found.")
            return

        transitions["from_val"] = transitions["cluster"].map(valence_map)
        transitions["to_val"] = transitions["next_cluster"].map(valence_map)
        transitions = transitions.dropna(subset=["from_val", "to_val"])

        transitions.to_csv(
            os.path.join(output, "G_vs_S_apetitive_aversive_transitions_raw.csv"),
            index=False
        )


        # --------------------------------------------------
        # PER-VIDEO TRANSITION MATRICES
        # --------------------------------------------------

        video_probs = []

        for (condition, file), dsub in transitions.groupby(["condition", "file"]):

            C = (
                dsub.groupby(["from_val", "to_val"])
                .size()
                .unstack(fill_value=0)
                .reindex(index=valence_order, columns=valence_order, fill_value=0)
            )

            P = C.div(C.sum(axis=1), axis=0).fillna(0)

            for from_val in valence_order:
                for to_val in valence_order:

                    video_probs.append({
                        "condition": condition,
                        "file": file,
                        "from_val": from_val,
                        "to_val": to_val,
                        "probability": P.loc[from_val, to_val]
                    })

        video_probs = pd.DataFrame(video_probs)

        video_probs.to_csv(
            os.path.join(output, "apetitive_aversive_transition_probabilities_per_video.csv"),
            index=False
        )



        stats = []

        for from_val in valence_order:
            for to_val in valence_order:

                g = video_probs[
                    (video_probs["condition"] == "G") &
                    (video_probs["from_val"] == from_val) &
                    (video_probs["to_val"] == to_val)
                ]["probability"]

                s = video_probs[
                    (video_probs["condition"] == "S") &
                    (video_probs["from_val"] == from_val) &
                    (video_probs["to_val"] == to_val)
                ]["probability"]

                if len(g) > 0 and len(s) > 0:

                    stat, p = mannwhitneyu(
                        g,
                        s,
                        alternative="two-sided"
                    )

                else:
                    stat = np.nan
                    p = np.nan

                stats.append({
                    "from_val": from_val,
                    "to_val": to_val,
                    "U": stat,
                    "p": p,
                    "G_mean": g.mean(),
                    "S_mean": s.mean()
                })

        stats = pd.DataFrame(stats)

        mask = stats["p"].notna()

        stats.loc[mask, "p_adj"] = multipletests(
            stats.loc[mask, "p"],
            method="fdr_bh"
        )[1]

        stats.to_csv(
            os.path.join(output, "apetitive_aversive_transition_statistics.csv"),
            index=False
        )




        def val_transition_matrix(dsub):
            C = (
                dsub.groupby(["from_val", "to_val"])
                .size()
                .unstack(fill_value=0)
                .reindex(index=valence_order, columns=valence_order, fill_value=0)
            )
            P = C.div(C.sum(axis=1), axis=0).fillna(0)
            return C, P

        C_S, P_S = val_transition_matrix(transitions[transitions["condition"] == "S"])
        C_G, P_G = val_transition_matrix(transitions[transitions["condition"] == "G"])

        C_S.to_csv(os.path.join(output, "apetitive_aversive_transition_counts_S.csv"))
        C_G.to_csv(os.path.join(output, "apetitive_aversive_transition_counts_G.csv"))
        P_S.to_csv(os.path.join(output, "apetitive_aversive_transition_probability_S.csv"))
        P_G.to_csv(os.path.join(output, "apetitive_aversive_transition_probability_G.csv"))

        # D = P_G - P_S

        D = (
            stats
            .pivot(index="from_val", columns="to_val", values="G_mean")
            .reindex(index=valence_order, columns=valence_order)
            -
            stats
            .pivot(index="from_val", columns="to_val", values="S_mean")
            .reindex(index=valence_order, columns=valence_order)
        )


        D.to_csv(os.path.join(output, "apetitive_aversive_transition_difference_G_minus_S.csv"))

        lim = float(np.nanmax(np.abs(D.to_numpy())))
        if lim == 0:
            lim = 1e-6

        cmap = plt.cm.RdBu
        norm = mpl.colors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)

        plt.figure(figsize=(5, 4))
        sns.heatmap(
            D,
            cmap=cmap,
            center=0,
            vmin=-lim,
            vmax=lim,
            square=True,
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "P(G) - P(S)"}
        )
        plt.xlabel("Next valence class")
        plt.ylabel("Current valence class")
        plt.title("Apetitive/aversive transition difference: G - S")
        plt.tight_layout()

        plt.savefig(
            os.path.join(output, "apetitive_aversive_transition_difference_heatmap.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.savefig(
            os.path.join(output, "apetitive_aversive_transition_difference_heatmap.pdf"),
            format="pdf",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        nodes = valence_order
        pos = nx.circular_layout(nodes)

        def w_to_lw(w, min_w=0.6, max_w=8.0):
            return min_w + (abs(w) / lim) * (max_w - min_w)

        def w_to_alpha(w, min_a=0.2, max_a=0.95):
            return min_a + (abs(w) / lim) * (max_a - min_a)

        def diff_matrix_to_digraph_labels(D, thresh=0.02):
            G = nx.DiGraph()
            for c in D.index:
                G.add_node(c)
            for i in D.index:
                for j in D.columns:
                    w = float(D.loc[i, j])
                    if abs(w) >= thresh:
                        G.add_edge(i, j, weight=w)
            return G

        Gd = diff_matrix_to_digraph_labels(D, thresh=0.02)

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax.set_title("Apetitive/aversive transition likelihood: G - S")
        ax.axis("off")

        nx.draw_networkx_nodes(Gd, pos, ax=ax, node_size=1200)
        nx.draw_networkx_labels(Gd, pos, ax=ax, font_size=10)

        for u, v, dct in Gd.edges(data=True):
            w = float(dct["weight"])
            color = cmap(norm(w))
            lw = w_to_lw(w)
            alpha = w_to_alpha(w)

            rad = 0.18 if u != v else 0.45

            patch = FancyArrowPatch(
                posA=pos[u],
                posB=pos[v],
                arrowstyle="-|>",
                mutation_scale=16,
                connectionstyle=f"arc3,rad={rad}",
                linewidth=lw,
                color=color,
                alpha=alpha,
                shrinkA=22,
                shrinkB=22
            )

            ax.add_patch(patch)

        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cax = fig.add_axes([0.95, 0.28, 0.02, 0.45])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("P(G) - P(S)")

        plt.savefig(
            os.path.join(output, "apetitive_aversive_transition_difference_circlegraph.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.savefig(
            os.path.join(output, "apetitive_aversive_transition_difference_circlegraph.pdf"),
            format="pdf",
            bbox_inches="tight"
        )
        plt.close()

        print("Saved G vs S apetitive/aversive transition analysis.")




    def GS_duration_transition(self):


        df = self.df.copy()
        cluster_name = self.cluster_name

        output = os.path.join(self.directory, "GS_duration_transition")
        os.makedirs(output, exist_ok=True)
        g_vs_s_output = os.path.join(output, "G_V_S")
        within_type_output = os.path.join(output, "within_type")
        mixed_output = os.path.join(output, "mixed")
        mixed_2_output = os.path.join(output, "mixed_2")
        group_output = os.path.join(output, "group")
        isolated_output = os.path.join(output, "isolated")
        os.makedirs(g_vs_s_output, exist_ok=True)
        os.makedirs(within_type_output, exist_ok=True)
        os.makedirs(mixed_output, exist_ok=True)
        os.makedirs(mixed_2_output, exist_ok=True)
        os.makedirs(group_output, exist_ok=True)
        os.makedirs(isolated_output, exist_ok=True)

        dur_map = {
            1: "medium",
            2: "short",
            3: "medium",
            4: "medium",
            5: "short",
            6: "medium",
            7: "medium",
            8: "short",
            9: "long",
            10: "short",
            11: "short",
            12: "short",
        }

        dur_order = ["short", "medium", "long"]

        df = df[
            (df["condition"] == "GS") &
            (np.isclose(df["Normalized Frame"], 0))
        ].copy()

        if df.empty:
            print("No GS interactions found.")
            return

        def parse_pair(pair):
            if pd.isna(pair):
                return None
            if isinstance(pair, str):
                pair = pair.replace("(", "").replace(")", "").replace(" ", "")
                id1, id2 = pair.split(",")
                return tuple(sorted((int(id1), int(id2))))
            id1, id2 = pair
            return tuple(sorted((int(id1), int(id2))))

        def larva_type(track_id):
            return "S" if int(track_id) <= 4 else "G"

        df["parsed_pair"] = df["Interaction Pair"].apply(parse_pair)
        df = df.dropna(subset=["parsed_pair"])
        df["cluster"] = df[cluster_name].astype(int)

        rows = []

        for _, row in df.iterrows():
            id1, id2 = row["parsed_pair"]

            rows.append({
                "file": row["file"],
                "interaction_id": row["interaction_id"],
                "track_id": id1,
                "larva_type": larva_type(id1),
                "partner_id": id2,
                "partner_type": larva_type(id2),
                "time": row["Frame"],
                "cluster": row["cluster"]
            })

            rows.append({
                "file": row["file"],
                "interaction_id": row["interaction_id"],
                "track_id": id2,
                "larva_type": larva_type(id2),
                "partner_id": id1,
                "partner_type": larva_type(id1),
                "time": row["Frame"],
                "cluster": row["cluster"]
            })

        expanded = pd.DataFrame(rows)

        expanded = expanded.sort_values(
            ["file", "track_id", "time", "interaction_id"]
        )

        expanded["next_cluster"] = (
            expanded
            .groupby(["file", "track_id"])["cluster"]
            .shift(-1)
        )
        expanded["next_partner_id"] = (
            expanded
            .groupby(["file", "track_id"])["partner_id"]
            .shift(-1)
        )
        expanded["next_partner_type"] = (
            expanded
            .groupby(["file", "track_id"])["partner_type"]
            .shift(-1)
        )

        transitions = expanded.dropna(subset=["next_cluster"]).copy()
        transitions["next_cluster"] = transitions["next_cluster"].astype(int)

        if transitions.empty:
            print("No GS duration transitions found.")
            return

        transitions["from_dur"] = transitions["cluster"].map(dur_map)
        transitions["to_dur"] = transitions["next_cluster"].map(dur_map)
        transitions = transitions.dropna(subset=["from_dur", "to_dur"])

        transitions.to_csv(
            os.path.join(g_vs_s_output, "GS_duration_transitions_raw.csv"),
            index=False
        )

        # --------------------------------------------------
        # PER-VIDEO TRANSITION PROBABILITIES
        # --------------------------------------------------

        video_probs = []

        for (file, larva_type_id), dsub in transitions.groupby(["file", "larva_type"]):

            C = (
                dsub.groupby(["from_dur", "to_dur"])
                .size()
                .unstack(fill_value=0)
                .reindex(index=dur_order, columns=dur_order, fill_value=0)
            )

            P = C.div(C.sum(axis=1), axis=0).fillna(0)

            for from_dur in dur_order:
                for to_dur in dur_order:
                    video_probs.append({
                        "file": file,
                        "larva_type": larva_type_id,
                        "from_dur": from_dur,
                        "to_dur": to_dur,
                        "probability": P.loc[from_dur, to_dur]
                    })

        video_probs = pd.DataFrame(video_probs)

        video_probs.to_csv(
            os.path.join(g_vs_s_output, "GS_duration_transition_probabilities_per_video.csv"),
            index=False
        )

        # --------------------------------------------------
        # STATS: paired G tracks vs S tracks within each GS video
        # --------------------------------------------------

        stats = []

        for from_dur in dur_order:
            for to_dur in dur_order:

                sub = video_probs[
                    (video_probs["from_dur"] == from_dur) &
                    (video_probs["to_dur"] == to_dur)
                ]

                wide = (
                    sub.pivot(index="file", columns="larva_type", values="probability")
                    .dropna(subset=["G", "S"])
                )

                g = wide["G"]
                s = wide["S"]
                diff = g - s

                if len(diff) >= 2 and not np.allclose(diff, 0):
                    stat, p = wilcoxon(g, s, alternative="two-sided")
                elif len(diff) >= 2 and np.allclose(diff, 0):
                    stat, p = 0, 1.0
                else:
                    stat, p = np.nan, np.nan

                stats.append({
                    "from_dur": from_dur,
                    "to_dur": to_dur,
                    "W": stat,
                    "p": p,
                    "G_mean": g.mean(),
                    "S_mean": s.mean(),
                    "mean_diff_G_minus_S": diff.mean(),
                    "n_videos": len(diff)
                })

        stats = pd.DataFrame(stats)

        mask = stats["p"].notna()

        stats.loc[mask, "p_adj"] = multipletests(
            stats.loc[mask, "p"],
            method="fdr_bh"
        )[1]

        stats.to_csv(
            os.path.join(g_vs_s_output, "GS_duration_transition_statistics.csv"),
            index=False
        )

        # --------------------------------------------------
        # POOLED MATRICES, SAVED ONLY
        # --------------------------------------------------

        def dur_transition_matrix(dsub):
            C = (
                dsub.groupby(["from_dur", "to_dur"])
                .size()
                .unstack(fill_value=0)
                .reindex(index=dur_order, columns=dur_order, fill_value=0)
            )
            P = C.div(C.sum(axis=1), axis=0).fillna(0)
            return C, P

        C_S, P_S = dur_transition_matrix(transitions[transitions["larva_type"] == "S"])
        C_G, P_G = dur_transition_matrix(transitions[transitions["larva_type"] == "G"])

        C_S.to_csv(os.path.join(g_vs_s_output, "GS_duration_transition_counts_S_tracks.csv"))
        C_G.to_csv(os.path.join(g_vs_s_output, "GS_duration_transition_counts_G_tracks.csv"))
        P_S.to_csv(os.path.join(g_vs_s_output, "GS_duration_transition_probability_S_tracks.csv"))
        P_G.to_csv(os.path.join(g_vs_s_output, "GS_duration_transition_probability_G_tracks.csv"))

        nodes = dur_order
        pos = nx.circular_layout(nodes)

        def probability_matrix_to_digraph(P, thresh=0.02):
            G = nx.DiGraph()
            for c in P.index:
                G.add_node(c)
            for i in P.index:
                for j in P.columns:
                    w = float(P.loc[i, j])
                    if w >= thresh:
                        G.add_edge(i, j, weight=w)
            return G

        def plot_probability_circlegraph(P, title, colorbar_label, filename_stem, save_dir, prob_lim=None, thresh=0.02):
            values = P.to_numpy()
            finite_values = values[np.isfinite(values)]

            if finite_values.size == 0:
                print(f"No finite values found for {title}.")
                return

            if prob_lim is None:
                prob_lim = float(np.nanmax(finite_values))
            if prob_lim == 0:
                prob_lim = 1e-6

            prob_norm = mpl.colors.Normalize(vmin=0, vmax=prob_lim)
            prob_cmap = plt.cm.viridis

            def prob_w_to_lw(w, min_w=0.6, max_w=8.0):
                return min_w + (w / prob_lim) * (max_w - min_w)

            def prob_w_to_alpha(w, min_a=0.2, max_a=0.95):
                return min_a + (w / prob_lim) * (max_a - min_a)

            G_prob = probability_matrix_to_digraph(P, thresh=thresh)

            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            ax.set_title(title)
            ax.axis("off")

            nx.draw_networkx_nodes(G_prob, pos, ax=ax, node_size=1200)
            nx.draw_networkx_labels(G_prob, pos, ax=ax, font_size=10)

            for u, v, dct in G_prob.edges(data=True):

                w = float(dct["weight"])
                color = prob_cmap(prob_norm(w))
                lw = prob_w_to_lw(w)
                alpha = prob_w_to_alpha(w)

                rad = 0.18 if u != v else 0.45

                patch = FancyArrowPatch(
                    posA=pos[u],
                    posB=pos[v],
                    arrowstyle="-|>",
                    mutation_scale=16,
                    connectionstyle=f"arc3,rad={rad}",
                    linewidth=lw,
                    color=color,
                    alpha=alpha,
                    shrinkA=22,
                    shrinkB=22
                )

                ax.add_patch(patch)

            sm = mpl.cm.ScalarMappable(cmap=prob_cmap, norm=prob_norm)
            sm.set_array([])

            cax = fig.add_axes([0.95, 0.28, 0.02, 0.45])
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_label(colorbar_label)

            plt.savefig(
                os.path.join(save_dir, f"{filename_stem}.pdf"),
                format="pdf",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

        P_S_mean = (
            stats
            .pivot(index="from_dur", columns="to_dur", values="S_mean")
            .reindex(index=dur_order, columns=dur_order)
        )

        P_G_mean = (
            stats
            .pivot(index="from_dur", columns="to_dur", values="G_mean")
            .reindex(index=dur_order, columns=dur_order)
        )

        if not (
            np.isfinite(P_S_mean.to_numpy()).any() and
            np.isfinite(P_G_mean.to_numpy()).any()
        ):
            print("No paired per-video G/S duration comparisons found.")
            return

        prob_lim_main = float(np.nanmax([
            np.nanmax(P_S_mean.to_numpy()),
            np.nanmax(P_G_mean.to_numpy())
        ]))
        if prob_lim_main == 0:
            prob_lim_main = 1e-6

        plot_probability_circlegraph(
            P_S_mean,
            "GS duration transition likelihood: S tracks",
            "P(S tracks)",
            "GS_duration_transition_S_tracks_circlegraph",
            g_vs_s_output,
            prob_lim=prob_lim_main
        )

        plot_probability_circlegraph(
            P_G_mean,
            "GS duration transition likelihood: G tracks",
            "P(G tracks)",
            "GS_duration_transition_G_tracks_circlegraph",
            g_vs_s_output,
            prob_lim=prob_lim_main
        )

        # --------------------------------------------------
        # D = mean per-video G probability - mean per-video S probability
        # --------------------------------------------------

        D = (
            stats
            .pivot(index="from_dur", columns="to_dur", values="mean_diff_G_minus_S")
            .reindex(index=dur_order, columns=dur_order)
        )

        D.to_csv(
            os.path.join(g_vs_s_output, "GS_duration_transition_difference_G_minus_S.csv")
        )

        lim = float(np.nanmax(np.abs(D.to_numpy())))
        if lim == 0:
            lim = 1e-6

        cmap = plt.cm.RdBu
        norm = mpl.colors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)

        plt.figure(figsize=(5, 4))
        sns.heatmap(
            D,
            cmap=cmap,
            center=0,
            vmin=-lim,
            vmax=lim,
            square=True,
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "P(G tracks) - P(S tracks)"}
        )

        plt.xlabel("Next duration class")
        plt.ylabel("Current duration class")
        plt.title("GS duration transition difference: G tracks - S tracks")
        plt.tight_layout()

        plt.savefig(
            os.path.join(g_vs_s_output, "GS_duration_transition_difference_heatmap.png"),
            dpi=300,
            bbox_inches="tight"
        )

        plt.savefig(
            os.path.join(g_vs_s_output, "GS_duration_transition_difference_heatmap.pdf"),
            format="pdf",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        nodes = dur_order
        pos = nx.circular_layout(nodes)

        def w_to_lw(w, min_w=0.6, max_w=8.0):
            return min_w + (abs(w) / lim) * (max_w - min_w)

        def w_to_alpha(w, min_a=0.2, max_a=0.95):
            return min_a + (abs(w) / lim) * (max_a - min_a)

        def diff_matrix_to_digraph_labels(D, thresh=0.02):
            G = nx.DiGraph()
            for c in D.index:
                G.add_node(c)
            for i in D.index:
                for j in D.columns:
                    w = float(D.loc[i, j])
                    if abs(w) >= thresh:
                        G.add_edge(i, j, weight=w)
            return G

        Gd = diff_matrix_to_digraph_labels(D, thresh=0.02)

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax.set_title("GS duration transition likelihood: G tracks - S tracks")
        ax.axis("off")

        nx.draw_networkx_nodes(Gd, pos, ax=ax, node_size=1200)
        nx.draw_networkx_labels(Gd, pos, ax=ax, font_size=10)

        for u, v, dct in Gd.edges(data=True):

            w = float(dct["weight"])
            color = cmap(norm(w))
            lw = w_to_lw(w)
            alpha = w_to_alpha(w)

            rad = 0.18 if u != v else 0.45

            patch = FancyArrowPatch(
                posA=pos[u],
                posB=pos[v],
                arrowstyle="-|>",
                mutation_scale=16,
                connectionstyle=f"arc3,rad={rad}",
                linewidth=lw,
                color=color,
                alpha=alpha,
                shrinkA=22,
                shrinkB=22
            )

            ax.add_patch(patch)

        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cax = fig.add_axes([0.95, 0.28, 0.02, 0.45])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("P(G tracks) - P(S tracks)")

        plt.savefig(
            os.path.join(g_vs_s_output, "GS_duration_transition_difference_circlegraph.pdf"),
            format="pdf",
            bbox_inches="tight"
        )

        plt.close()

        # --------------------------------------------------
        # INTERACTION-HISTORY FILTERED DURATION TRANSITIONS
        # --------------------------------------------------

        def plot_history_duration_comparison(
            comparison_name,
            comparison_output,
            group_a_label,
            group_a_filter,
            group_b_label,
            group_b_filter,
            difference_label,
            heatmap_title,
            circle_title
        ):

            labelled = []

            group_a = transitions[group_a_filter(transitions)].copy()
            group_a["history_group"] = group_a_label
            labelled.append(group_a)

            group_b = transitions[group_b_filter(transitions)].copy()
            group_b["history_group"] = group_b_label
            labelled.append(group_b)

            history_transitions = pd.concat(labelled, ignore_index=True)

            raw_path = os.path.join(
                comparison_output,
                f"GS_duration_transition_{comparison_name}_raw.csv"
            )
            history_transitions.to_csv(raw_path, index=False)

            count_summary = (
                history_transitions
                .groupby(["history_group", "larva_type", "partner_type", "next_partner_type"])
                .size()
                .reset_index(name="n_transitions")
            )
            count_summary.to_csv(
                os.path.join(
                    comparison_output,
                    f"GS_duration_transition_{comparison_name}_filter_counts.csv"
                ),
                index=False
            )

            history_video_probs = []

            for (file, history_group), dsub in history_transitions.groupby(["file", "history_group"]):

                C = (
                    dsub.groupby(["from_dur", "to_dur"])
                    .size()
                    .unstack(fill_value=0)
                    .reindex(index=dur_order, columns=dur_order, fill_value=0)
                )

                P = C.div(C.sum(axis=1), axis=0).fillna(0)

                for from_dur in dur_order:
                    for to_dur in dur_order:
                        history_video_probs.append({
                            "file": file,
                            "history_group": history_group,
                            "from_dur": from_dur,
                            "to_dur": to_dur,
                            "probability": P.loc[from_dur, to_dur]
                        })

            history_video_probs = pd.DataFrame(history_video_probs)

            if history_video_probs.empty:
                print(f"No {comparison_name} duration transitions found.")
                return

            history_video_probs.to_csv(
                os.path.join(
                    comparison_output,
                    f"GS_duration_transition_{comparison_name}_probabilities_per_video.csv"
                ),
                index=False
            )

            history_stats = []

            for from_dur in dur_order:
                for to_dur in dur_order:

                    sub = history_video_probs[
                        (history_video_probs["from_dur"] == from_dur) &
                        (history_video_probs["to_dur"] == to_dur)
                    ]

                    wide = (
                        sub.pivot(index="file", columns="history_group", values="probability")
                        .dropna(subset=[group_a_label, group_b_label])
                    )

                    a = wide[group_a_label]
                    b = wide[group_b_label]
                    diff = a - b

                    if len(diff) >= 2 and not np.allclose(diff, 0):
                        stat, p = wilcoxon(a, b, alternative="two-sided")
                    elif len(diff) >= 2 and np.allclose(diff, 0):
                        stat, p = 0, 1.0
                    else:
                        stat, p = np.nan, np.nan

                    history_stats.append({
                        "from_dur": from_dur,
                        "to_dur": to_dur,
                        "W": stat,
                        "p": p,
                        f"{group_a_label}_mean": a.mean(),
                        f"{group_b_label}_mean": b.mean(),
                        "mean_diff": diff.mean(),
                        "n_videos": len(diff)
                    })

            history_stats = pd.DataFrame(history_stats)
            mask = history_stats["p"].notna()

            if mask.any():
                history_stats.loc[mask, "p_adj"] = multipletests(
                    history_stats.loc[mask, "p"],
                    method="fdr_bh"
                )[1]
            else:
                history_stats["p_adj"] = np.nan

            history_stats.to_csv(
                os.path.join(
                    comparison_output,
                    f"GS_duration_transition_{comparison_name}_statistics.csv"
                ),
                index=False
            )

            C_A, P_A = dur_transition_matrix(
                history_transitions[history_transitions["history_group"] == group_a_label]
            )
            C_B, P_B = dur_transition_matrix(
                history_transitions[history_transitions["history_group"] == group_b_label]
            )

            C_A.to_csv(os.path.join(comparison_output, f"GS_duration_transition_{comparison_name}_counts_{group_a_label}.csv"))
            C_B.to_csv(os.path.join(comparison_output, f"GS_duration_transition_{comparison_name}_counts_{group_b_label}.csv"))
            P_A.to_csv(os.path.join(comparison_output, f"GS_duration_transition_{comparison_name}_probability_{group_a_label}.csv"))
            P_B.to_csv(os.path.join(comparison_output, f"GS_duration_transition_{comparison_name}_probability_{group_b_label}.csv"))

            short_group_a_label = group_a_label.replace("_with_", "_").replace("_then_", "_")
            short_group_b_label = group_b_label.replace("_with_", "_").replace("_then_", "_")

            D_history = (
                history_stats
                .pivot(index="from_dur", columns="to_dur", values="mean_diff")
                .reindex(index=dur_order, columns=dur_order)
            )

            D_history.to_csv(
                os.path.join(
                    comparison_output,
                    f"GS_duration_transition_{comparison_name}_difference.csv"
                )
            )

            values = D_history.to_numpy()
            finite_values = values[np.isfinite(values)]

            if finite_values.size == 0:
                print(f"No paired per-video {comparison_name} comparisons found.")
                return

            P_A_mean = (
                history_stats
                .pivot(index="from_dur", columns="to_dur", values=f"{group_a_label}_mean")
                .reindex(index=dur_order, columns=dur_order)
            )

            P_B_mean = (
                history_stats
                .pivot(index="from_dur", columns="to_dur", values=f"{group_b_label}_mean")
                .reindex(index=dur_order, columns=dur_order)
            )

            prob_lim_history = float(np.nanmax([
                np.nanmax(P_A_mean.to_numpy()),
                np.nanmax(P_B_mean.to_numpy())
            ]))
            if prob_lim_history == 0:
                prob_lim_history = 1e-6

            plot_probability_circlegraph(
                P_A_mean,
                f"GS duration transition likelihood: {group_a_label}",
                f"P({group_a_label})",
                f"GS_duration_transition_{comparison_name}_{short_group_a_label}_circlegraph",
                comparison_output,
                prob_lim=prob_lim_history
            )

            plot_probability_circlegraph(
                P_B_mean,
                f"GS duration transition likelihood: {group_b_label}",
                f"P({group_b_label})",
                f"GS_duration_transition_{comparison_name}_{short_group_b_label}_circlegraph",
                comparison_output,
                prob_lim=prob_lim_history
            )

            history_lim = float(np.nanmax(np.abs(finite_values)))
            if history_lim == 0:
                history_lim = 1e-6

            history_norm = mpl.colors.TwoSlopeNorm(
                vmin=-history_lim,
                vcenter=0,
                vmax=history_lim
            )

            plt.figure(figsize=(5, 4))
            sns.heatmap(
                D_history,
                cmap=cmap,
                center=0,
                vmin=-history_lim,
                vmax=history_lim,
                square=True,
                annot=True,
                fmt=".2f",
                cbar_kws={"label": difference_label}
            )

            plt.xlabel("Next duration class")
            plt.ylabel("Current duration class")
            plt.title(heatmap_title)
            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    comparison_output,
                    f"GS_duration_transition_{comparison_name}_difference_heatmap.png"
                ),
                dpi=300,
                bbox_inches="tight"
            )

            plt.savefig(
                os.path.join(
                    comparison_output,
                    f"GS_duration_transition_{comparison_name}_difference_heatmap.pdf"
                ),
                format="pdf",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            def history_w_to_lw(w, min_w=0.6, max_w=8.0):
                return min_w + (abs(w) / history_lim) * (max_w - min_w)

            def history_w_to_alpha(w, min_a=0.2, max_a=0.95):
                return min_a + (abs(w) / history_lim) * (max_a - min_a)

            G_history = diff_matrix_to_digraph_labels(D_history, thresh=0.02)

            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            ax.set_title(circle_title)
            ax.axis("off")

            nx.draw_networkx_nodes(G_history, pos, ax=ax, node_size=1200)
            nx.draw_networkx_labels(G_history, pos, ax=ax, font_size=10)

            for u, v, dct in G_history.edges(data=True):

                w = float(dct["weight"])
                color = cmap(history_norm(w))
                lw = history_w_to_lw(w)
                alpha = history_w_to_alpha(w)

                rad = 0.18 if u != v else 0.45

                patch = FancyArrowPatch(
                    posA=pos[u],
                    posB=pos[v],
                    arrowstyle="-|>",
                    mutation_scale=16,
                    connectionstyle=f"arc3,rad={rad}",
                    linewidth=lw,
                    color=color,
                    alpha=alpha,
                    shrinkA=22,
                    shrinkB=22
                )

                ax.add_patch(patch)

            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=history_norm)
            sm.set_array([])

            cax = fig.add_axes([0.95, 0.28, 0.02, 0.45])
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_label(difference_label)

            plt.savefig(
                os.path.join(
                    comparison_output,
                    f"GS_duration_transition_{comparison_name}_difference_circlegraph.pdf"
                ),
                format="pdf",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

        plot_history_duration_comparison(
            comparison_name="within_type",
            comparison_output=within_type_output,
            group_a_label="G_with_G_then_G",
            group_a_filter=lambda d: (
                (d["larva_type"] == "G") &
                (d["partner_type"] == "G") &
                (d["next_partner_type"] == "G")
            ),
            group_b_label="S_with_S_then_S",
            group_b_filter=lambda d: (
                (d["larva_type"] == "S") &
                (d["partner_type"] == "S") &
                (d["next_partner_type"] == "S")
            ),
            difference_label="P(G with G then G) - P(S with S then S)",
            heatmap_title="GS duration transition difference: within-type history",
            circle_title="GS duration transition likelihood: within-type history"
        )

        plot_history_duration_comparison(
            comparison_name="mixed_type",
            comparison_output=mixed_output,
            group_a_label="G_with_S_then_S",
            group_a_filter=lambda d: (
                (d["larva_type"] == "G") &
                (d["partner_type"] == "S") &
                (d["next_partner_type"] == "S")
            ),
            group_b_label="S_with_G_then_G",
            group_b_filter=lambda d: (
                (d["larva_type"] == "S") &
                (d["partner_type"] == "G") &
                (d["next_partner_type"] == "G")
            ),
            difference_label="P(G with S then S) - P(S with G then G)",
            heatmap_title="GS duration transition difference: mixed-type history",
            circle_title="GS duration transition likelihood: mixed-type history"
        )

        plot_history_duration_comparison(
            comparison_name="mixed_2",
            comparison_output=mixed_2_output,
            group_a_label="G_with_S_then_G",
            group_a_filter=lambda d: (
                (d["larva_type"] == "G") &
                (d["partner_type"] == "S") &
                (d["next_partner_type"] == "G")
            ),
            group_b_label="S_with_G_then_S",
            group_b_filter=lambda d: (
                (d["larva_type"] == "S") &
                (d["partner_type"] == "G") &
                (d["next_partner_type"] == "S")
            ),
            difference_label="P(G with S then G) - P(S with G then S)",
            heatmap_title="GS duration transition difference: mixed-2 history",
            circle_title="GS duration transition likelihood: mixed-2 history"
        )

        plot_history_duration_comparison(
            comparison_name="group",
            comparison_output=group_output,
            group_a_label="G_with_S_then_G",
            group_a_filter=lambda d: (
                (d["larva_type"] == "G") &
                (d["partner_type"] == "S") &
                (d["next_partner_type"] == "G")
            ),
            group_b_label="G_with_G_then_G",
            group_b_filter=lambda d: (
                (d["larva_type"] == "G") &
                (d["partner_type"] == "G") &
                (d["next_partner_type"] == "G")
            ),
            difference_label="P(G with S then G) - P(G with G then G)",
            heatmap_title="GS duration transition difference: group tracks, switched vs stayed group",
            circle_title="GS duration transition likelihood: group tracks, switched vs stayed group"
        )

        plot_history_duration_comparison(
            comparison_name="isolated",
            comparison_output=isolated_output,
            group_a_label="S_with_G_then_S",
            group_a_filter=lambda d: (
                (d["larva_type"] == "S") &
                (d["partner_type"] == "G") &
                (d["next_partner_type"] == "S")
            ),
            group_b_label="S_with_S_then_S",
            group_b_filter=lambda d: (
                (d["larva_type"] == "S") &
                (d["partner_type"] == "S") &
                (d["next_partner_type"] == "S")
            ),
            difference_label="P(S with G then S) - P(S with S then S)",
            heatmap_title="GS duration transition difference: isolated tracks, switched vs stayed isolated",
            circle_title="GS duration transition likelihood: isolated tracks, switched vs stayed isolated"
        )

        print("Saved GS duration transition analysis.")




    def GS_apetitive_aversive_transition(self):


        df = self.df.copy()
        cluster_name = self.cluster_name

        output = os.path.join(self.directory, "GS_apetitive_aversive_transition")
        os.makedirs(output, exist_ok=True)
        g_vs_s_output = os.path.join(output, "G_V_S")
        within_type_output = os.path.join(output, "within_type")
        mixed_output = os.path.join(output, "mixed")
        mixed_2_output = os.path.join(output, "mixed_2")
        os.makedirs(g_vs_s_output, exist_ok=True)
        os.makedirs(within_type_output, exist_ok=True)
        os.makedirs(mixed_output, exist_ok=True)
        os.makedirs(mixed_2_output, exist_ok=True)

        valence_map = {
            1: "apetitive",
            2: "aversive",
            3: "apetitive",
            4: "apetitive",
            5: "aversive",
            6: "apetitive",
            7: "apetitive",
            8: "aversive",
            9: "apetitive",
            10: "aversive",
            11: "aversive",
            12: "aversive",
        }

        valence_order = ["apetitive", "aversive"]

        df = df[
            (df["condition"] == "GS") &
            (np.isclose(df["Normalized Frame"], 0))
        ].copy()

        if df.empty:
            print("No GS interactions found.")
            return

        def parse_pair(pair):
            if pd.isna(pair):
                return None
            if isinstance(pair, str):
                pair = pair.replace("(", "").replace(")", "").replace(" ", "")
                id1, id2 = pair.split(",")
                return tuple(sorted((int(id1), int(id2))))
            id1, id2 = pair
            return tuple(sorted((int(id1), int(id2))))

        def larva_type(track_id):
            return "S" if int(track_id) <= 4 else "G"

        df["parsed_pair"] = df["Interaction Pair"].apply(parse_pair)
        df = df.dropna(subset=["parsed_pair"])
        df["cluster"] = df[cluster_name].astype(int)

        rows = []

        for _, row in df.iterrows():
            id1, id2 = row["parsed_pair"]

            rows.append({
                "file": row["file"],
                "interaction_id": row["interaction_id"],
                "track_id": id1,
                "larva_type": larva_type(id1),
                "partner_id": id2,
                "partner_type": larva_type(id2),
                "time": row["Frame"],
                "cluster": row["cluster"]
            })

            rows.append({
                "file": row["file"],
                "interaction_id": row["interaction_id"],
                "track_id": id2,
                "larva_type": larva_type(id2),
                "partner_id": id1,
                "partner_type": larva_type(id1),
                "time": row["Frame"],
                "cluster": row["cluster"]
            })

        expanded = pd.DataFrame(rows)

        expanded = expanded.sort_values(
            ["file", "track_id", "time", "interaction_id"]
        )

        expanded["next_cluster"] = (
            expanded
            .groupby(["file", "track_id"])["cluster"]
            .shift(-1)
        )
        expanded["next_partner_id"] = (
            expanded
            .groupby(["file", "track_id"])["partner_id"]
            .shift(-1)
        )
        expanded["next_partner_type"] = (
            expanded
            .groupby(["file", "track_id"])["partner_type"]
            .shift(-1)
        )

        transitions = expanded.dropna(subset=["next_cluster"]).copy()
        transitions["next_cluster"] = transitions["next_cluster"].astype(int)

        if transitions.empty:
            print("No GS apetitive/aversive transitions found.")
            return

        transitions["from_val"] = transitions["cluster"].map(valence_map)
        transitions["to_val"] = transitions["next_cluster"].map(valence_map)
        transitions = transitions.dropna(subset=["from_val", "to_val"])

        transitions.to_csv(
            os.path.join(g_vs_s_output, "GS_apetitive_aversive_transitions_raw.csv"),
            index=False
        )

        # --------------------------------------------------
        # PER-VIDEO TRANSITION PROBABILITIES
        # --------------------------------------------------

        video_probs = []

        for (file, larva_type_id), dsub in transitions.groupby(["file", "larva_type"]):

            C = (
                dsub.groupby(["from_val", "to_val"])
                .size()
                .unstack(fill_value=0)
                .reindex(index=valence_order, columns=valence_order, fill_value=0)
            )

            P = C.div(C.sum(axis=1), axis=0).fillna(0)

            for from_val in valence_order:
                for to_val in valence_order:
                    video_probs.append({
                        "file": file,
                        "larva_type": larva_type_id,
                        "from_val": from_val,
                        "to_val": to_val,
                        "probability": P.loc[from_val, to_val]
                    })

        video_probs = pd.DataFrame(video_probs)

        video_probs.to_csv(
            os.path.join(g_vs_s_output, "GS_apetitive_aversive_transition_probabilities_per_video.csv"),
            index=False
        )

        # --------------------------------------------------
        # STATS: paired G tracks vs S tracks within each GS video
        # --------------------------------------------------

        stats = []

        for from_val in valence_order:
            for to_val in valence_order:

                sub = video_probs[
                    (video_probs["from_val"] == from_val) &
                    (video_probs["to_val"] == to_val)
                ]

                wide = (
                    sub.pivot(index="file", columns="larva_type", values="probability")
                    .dropna(subset=["G", "S"])
                )

                g = wide["G"]
                s = wide["S"]
                diff = g - s

                if len(diff) >= 2 and not np.allclose(diff, 0):
                    stat, p = wilcoxon(g, s, alternative="two-sided")
                elif len(diff) >= 2 and np.allclose(diff, 0):
                    stat, p = 0, 1.0
                else:
                    stat, p = np.nan, np.nan

                stats.append({
                    "from_val": from_val,
                    "to_val": to_val,
                    "W": stat,
                    "p": p,
                    "G_mean": g.mean(),
                    "S_mean": s.mean(),
                    "mean_diff_G_minus_S": diff.mean(),
                    "n_videos": len(diff)
                })

        stats = pd.DataFrame(stats)

        mask = stats["p"].notna()

        stats.loc[mask, "p_adj"] = multipletests(
            stats.loc[mask, "p"],
            method="fdr_bh"
        )[1]

        stats.to_csv(
            os.path.join(g_vs_s_output, "GS_apetitive_aversive_transition_statistics.csv"),
            index=False
        )

        # --------------------------------------------------
        # POOLED MATRICES, SAVED ONLY
        # --------------------------------------------------

        def val_transition_matrix(dsub):
            C = (
                dsub.groupby(["from_val", "to_val"])
                .size()
                .unstack(fill_value=0)
                .reindex(index=valence_order, columns=valence_order, fill_value=0)
            )
            P = C.div(C.sum(axis=1), axis=0).fillna(0)
            return C, P

        C_S, P_S = val_transition_matrix(transitions[transitions["larva_type"] == "S"])
        C_G, P_G = val_transition_matrix(transitions[transitions["larva_type"] == "G"])

        C_S.to_csv(os.path.join(g_vs_s_output, "GS_apetitive_aversive_transition_counts_S_tracks.csv"))
        C_G.to_csv(os.path.join(g_vs_s_output, "GS_apetitive_aversive_transition_counts_G_tracks.csv"))
        P_S.to_csv(os.path.join(g_vs_s_output, "GS_apetitive_aversive_transition_probability_S_tracks.csv"))
        P_G.to_csv(os.path.join(g_vs_s_output, "GS_apetitive_aversive_transition_probability_G_tracks.csv"))

        nodes = valence_order
        pos = nx.circular_layout(nodes)

        def probability_matrix_to_digraph(P, thresh=0.02):
            G = nx.DiGraph()
            for c in P.index:
                G.add_node(c)
            for i in P.index:
                for j in P.columns:
                    w = float(P.loc[i, j])
                    if w >= thresh:
                        G.add_edge(i, j, weight=w)
            return G

        def plot_probability_circlegraph(P, title, colorbar_label, filename_stem, save_dir, prob_lim=None, thresh=0.02, prob_cmap=None):
            values = P.to_numpy()
            finite_values = values[np.isfinite(values)]

            if finite_values.size == 0:
                print(f"No finite values found for {title}.")
                return

            if prob_lim is None:
                prob_lim = float(np.nanmax(finite_values))
            if prob_lim == 0:
                prob_lim = 1e-6

            prob_norm = mpl.colors.Normalize(vmin=0, vmax=prob_lim)
            if prob_cmap is None:
                prob_cmap = plt.cm.viridis

            def prob_w_to_lw(w, min_w=0.6, max_w=8.0):
                return min_w + (w / prob_lim) * (max_w - min_w)

            def prob_w_to_alpha(w, min_a=0.2, max_a=0.95):
                return min_a + (w / prob_lim) * (max_a - min_a)

            G_prob = probability_matrix_to_digraph(P, thresh=thresh)

            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            ax.set_title(title)
            ax.axis("off")

            nx.draw_networkx_nodes(G_prob, pos, ax=ax, node_size=1200)
            nx.draw_networkx_labels(G_prob, pos, ax=ax, font_size=10)

            for u, v, dct in G_prob.edges(data=True):

                w = float(dct["weight"])
                color = prob_cmap(prob_norm(w))
                lw = prob_w_to_lw(w)
                alpha = prob_w_to_alpha(w)

                rad = 0.18 if u != v else 0.45

                patch = FancyArrowPatch(
                    posA=pos[u],
                    posB=pos[v],
                    arrowstyle="-|>",
                    mutation_scale=16,
                    connectionstyle=f"arc3,rad={rad}",
                    linewidth=lw,
                    color=color,
                    alpha=alpha,
                    shrinkA=22,
                    shrinkB=22
                )

                ax.add_patch(patch)

            sm = mpl.cm.ScalarMappable(cmap=prob_cmap, norm=prob_norm)
            sm.set_array([])

            cax = fig.add_axes([0.95, 0.28, 0.02, 0.45])
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_label(colorbar_label)

            plt.savefig(
                os.path.join(save_dir, f"{filename_stem}.pdf"),
                format="pdf",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

        P_S_mean = (
            stats
            .pivot(index="from_val", columns="to_val", values="S_mean")
            .reindex(index=valence_order, columns=valence_order)
        )

        P_G_mean = (
            stats
            .pivot(index="from_val", columns="to_val", values="G_mean")
            .reindex(index=valence_order, columns=valence_order)
        )

        if not (
            np.isfinite(P_S_mean.to_numpy()).any() and
            np.isfinite(P_G_mean.to_numpy()).any()
        ):
            print("No paired per-video G/S apetitive/aversive comparisons found.")
            return

        prob_lim_main = float(np.nanmax([
            np.nanmax(P_S_mean.to_numpy()),
            np.nanmax(P_G_mean.to_numpy())
        ]))
        if prob_lim_main == 0:
            prob_lim_main = 1e-6

        plot_probability_circlegraph(
            P_S_mean,
            "GS apetitive/aversive transition likelihood: S tracks",
            "P(S tracks)",
            "GS_apetitive_aversive_transition_S_tracks_circlegraph",
            g_vs_s_output,
            prob_lim=prob_lim_main,
            prob_cmap=plt.cm.Reds
        )

        plot_probability_circlegraph(
            P_G_mean,
            "GS apetitive/aversive transition likelihood: G tracks",
            "P(G tracks)",
            "GS_apetitive_aversive_transition_G_tracks_circlegraph",
            g_vs_s_output,
            prob_lim=prob_lim_main,
            prob_cmap=plt.cm.Blues
        )

        # --------------------------------------------------
        # D = mean per-video G probability - mean per-video S probability
        # --------------------------------------------------

        D = (
            stats
            .pivot(index="from_val", columns="to_val", values="mean_diff_G_minus_S")
            .reindex(index=valence_order, columns=valence_order)
        )

        D.to_csv(
            os.path.join(g_vs_s_output, "GS_apetitive_aversive_transition_difference_G_minus_S.csv")
        )

        lim = float(np.nanmax(np.abs(D.to_numpy())))
        if lim == 0:
            lim = 1e-6

        cmap = plt.cm.RdBu
        norm = mpl.colors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)

        plt.figure(figsize=(5, 4))
        sns.heatmap(
            D,
            cmap=cmap,
            center=0,
            vmin=-lim,
            vmax=lim,
            square=True,
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "P(G tracks) - P(S tracks)"}
        )

        plt.xlabel("Next valence class")
        plt.ylabel("Current valence class")
        plt.title("GS apetitive/aversive transition difference: G tracks - S tracks")
        plt.tight_layout()

        plt.savefig(
            os.path.join(g_vs_s_output, "GS_apetitive_aversive_transition_difference_heatmap.png"),
            dpi=300,
            bbox_inches="tight"
        )

        plt.savefig(
            os.path.join(g_vs_s_output, "GS_apetitive_aversive_transition_difference_heatmap.pdf"),
            format="pdf",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        nodes = valence_order
        pos = nx.circular_layout(nodes)

        def w_to_lw(w, min_w=0.6, max_w=8.0):
            return min_w + (abs(w) / lim) * (max_w - min_w)

        def w_to_alpha(w, min_a=0.2, max_a=0.95):
            return min_a + (abs(w) / lim) * (max_a - min_a)

        def diff_matrix_to_digraph_labels(D, thresh=0.02):
            G = nx.DiGraph()
            for c in D.index:
                G.add_node(c)
            for i in D.index:
                for j in D.columns:
                    w = float(D.loc[i, j])
                    if abs(w) >= thresh:
                        G.add_edge(i, j, weight=w)
            return G

        Gd = diff_matrix_to_digraph_labels(D, thresh=0.02)

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax.set_title("GS apetitive/aversive transition likelihood: G tracks - S tracks")
        ax.axis("off")

        nx.draw_networkx_nodes(Gd, pos, ax=ax, node_size=1200)
        nx.draw_networkx_labels(Gd, pos, ax=ax, font_size=10)

        for u, v, dct in Gd.edges(data=True):

            w = float(dct["weight"])
            color = cmap(norm(w))
            lw = w_to_lw(w)
            alpha = w_to_alpha(w)

            rad = 0.18 if u != v else 0.45

            patch = FancyArrowPatch(
                posA=pos[u],
                posB=pos[v],
                arrowstyle="-|>",
                mutation_scale=16,
                connectionstyle=f"arc3,rad={rad}",
                linewidth=lw,
                color=color,
                alpha=alpha,
                shrinkA=22,
                shrinkB=22
            )

            ax.add_patch(patch)

        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cax = fig.add_axes([0.95, 0.28, 0.02, 0.45])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("P(G tracks) - P(S tracks)")

        plt.savefig(
            os.path.join(g_vs_s_output, "GS_apetitive_aversive_transition_difference_circlegraph.pdf"),
            format="pdf",
            bbox_inches="tight"
        )

        plt.close()

        # --------------------------------------------------
        # INTERACTION-HISTORY FILTERED APETITIVE/AVERSIVE TRANSITIONS
        # --------------------------------------------------

        def plot_history_valence_comparison(
            comparison_name,
            comparison_output,
            group_a_label,
            group_a_filter,
            group_b_label,
            group_b_filter,
            difference_label,
            heatmap_title,
            circle_title
        ):

            labelled = []

            group_a = transitions[group_a_filter(transitions)].copy()
            group_a["history_group"] = group_a_label
            labelled.append(group_a)

            group_b = transitions[group_b_filter(transitions)].copy()
            group_b["history_group"] = group_b_label
            labelled.append(group_b)

            history_transitions = pd.concat(labelled, ignore_index=True)

            raw_path = os.path.join(
                comparison_output,
                f"GS_apetitive_aversive_transition_{comparison_name}_raw.csv"
            )
            history_transitions.to_csv(raw_path, index=False)

            count_summary = (
                history_transitions
                .groupby(["history_group", "larva_type", "partner_type", "next_partner_type"])
                .size()
                .reset_index(name="n_transitions")
            )
            count_summary.to_csv(
                os.path.join(
                    comparison_output,
                    f"GS_apetitive_aversive_transition_{comparison_name}_filter_counts.csv"
                ),
                index=False
            )

            history_video_probs = []

            for (file, history_group), dsub in history_transitions.groupby(["file", "history_group"]):

                C = (
                    dsub.groupby(["from_val", "to_val"])
                    .size()
                    .unstack(fill_value=0)
                    .reindex(index=valence_order, columns=valence_order, fill_value=0)
                )

                P = C.div(C.sum(axis=1), axis=0).fillna(0)

                for from_val in valence_order:
                    for to_val in valence_order:
                        history_video_probs.append({
                            "file": file,
                            "history_group": history_group,
                            "from_val": from_val,
                            "to_val": to_val,
                            "probability": P.loc[from_val, to_val]
                        })

            history_video_probs = pd.DataFrame(history_video_probs)

            if history_video_probs.empty:
                print(f"No {comparison_name} apetitive/aversive transitions found.")
                return

            history_video_probs.to_csv(
                os.path.join(
                    comparison_output,
                    f"GS_apetitive_aversive_transition_{comparison_name}_probabilities_per_video.csv"
                ),
                index=False
            )

            history_stats = []

            for from_val in valence_order:
                for to_val in valence_order:

                    sub = history_video_probs[
                        (history_video_probs["from_val"] == from_val) &
                        (history_video_probs["to_val"] == to_val)
                    ]

                    wide = (
                        sub.pivot(index="file", columns="history_group", values="probability")
                        .dropna(subset=[group_a_label, group_b_label])
                    )

                    a = wide[group_a_label]
                    b = wide[group_b_label]
                    diff = a - b

                    if len(diff) >= 2 and not np.allclose(diff, 0):
                        stat, p = wilcoxon(a, b, alternative="two-sided")
                    elif len(diff) >= 2 and np.allclose(diff, 0):
                        stat, p = 0, 1.0
                    else:
                        stat, p = np.nan, np.nan

                    history_stats.append({
                        "from_val": from_val,
                        "to_val": to_val,
                        "W": stat,
                        "p": p,
                        f"{group_a_label}_mean": a.mean(),
                        f"{group_b_label}_mean": b.mean(),
                        "mean_diff": diff.mean(),
                        "n_videos": len(diff)
                    })

            history_stats = pd.DataFrame(history_stats)
            mask = history_stats["p"].notna()

            if mask.any():
                history_stats.loc[mask, "p_adj"] = multipletests(
                    history_stats.loc[mask, "p"],
                    method="fdr_bh"
                )[1]
            else:
                history_stats["p_adj"] = np.nan

            history_stats.to_csv(
                os.path.join(
                    comparison_output,
                    f"GS_apetitive_aversive_transition_{comparison_name}_statistics.csv"
                ),
                index=False
            )

            C_A, P_A = val_transition_matrix(
                history_transitions[history_transitions["history_group"] == group_a_label]
            )
            C_B, P_B = val_transition_matrix(
                history_transitions[history_transitions["history_group"] == group_b_label]
            )

            C_A.to_csv(os.path.join(comparison_output, f"GS_apetitive_aversive_transition_{comparison_name}_counts_{group_a_label}.csv"))
            C_B.to_csv(os.path.join(comparison_output, f"GS_apetitive_aversive_transition_{comparison_name}_counts_{group_b_label}.csv"))
            P_A.to_csv(os.path.join(comparison_output, f"GS_apetitive_aversive_transition_{comparison_name}_probability_{group_a_label}.csv"))
            P_B.to_csv(os.path.join(comparison_output, f"GS_apetitive_aversive_transition_{comparison_name}_probability_{group_b_label}.csv"))

            short_group_a_label = group_a_label.replace("_with_", "_").replace("_then_", "_")
            short_group_b_label = group_b_label.replace("_with_", "_").replace("_then_", "_")

            D_history = (
                history_stats
                .pivot(index="from_val", columns="to_val", values="mean_diff")
                .reindex(index=valence_order, columns=valence_order)
            )

            D_history.to_csv(
                os.path.join(
                    comparison_output,
                    f"GS_apetitive_aversive_transition_{comparison_name}_difference.csv"
                )
            )

            values = D_history.to_numpy()
            finite_values = values[np.isfinite(values)]

            if finite_values.size == 0:
                print(f"No paired per-video {comparison_name} comparisons found.")
                return

            P_A_mean = (
                history_stats
                .pivot(index="from_val", columns="to_val", values=f"{group_a_label}_mean")
                .reindex(index=valence_order, columns=valence_order)
            )

            P_B_mean = (
                history_stats
                .pivot(index="from_val", columns="to_val", values=f"{group_b_label}_mean")
                .reindex(index=valence_order, columns=valence_order)
            )

            prob_lim_history = float(np.nanmax([
                np.nanmax(P_A_mean.to_numpy()),
                np.nanmax(P_B_mean.to_numpy())
            ]))
            if prob_lim_history == 0:
                prob_lim_history = 1e-6

            plot_probability_circlegraph(
                P_A_mean,
                f"GS apetitive/aversive transition likelihood: {group_a_label}",
                f"P({group_a_label})",
                f"GS_apetitive_aversive_transition_{comparison_name}_{short_group_a_label}_circlegraph",
                comparison_output,
                prob_lim=prob_lim_history,
                prob_cmap=plt.cm.Blues
            )

            plot_probability_circlegraph(
                P_B_mean,
                f"GS apetitive/aversive transition likelihood: {group_b_label}",
                f"P({group_b_label})",
                f"GS_apetitive_aversive_transition_{comparison_name}_{short_group_b_label}_circlegraph",
                comparison_output,
                prob_lim=prob_lim_history,
                prob_cmap=plt.cm.Reds
            )

            history_lim = float(np.nanmax(np.abs(finite_values)))
            if history_lim == 0:
                history_lim = 1e-6

            history_norm = mpl.colors.TwoSlopeNorm(
                vmin=-history_lim,
                vcenter=0,
                vmax=history_lim
            )

            plt.figure(figsize=(5, 4))
            sns.heatmap(
                D_history,
                cmap=cmap,
                center=0,
                vmin=-history_lim,
                vmax=history_lim,
                square=True,
                annot=True,
                fmt=".2f",
                cbar_kws={"label": difference_label}
            )

            plt.xlabel("Next valence class")
            plt.ylabel("Current valence class")
            plt.title(heatmap_title)
            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    comparison_output,
                    f"GS_apetitive_aversive_transition_{comparison_name}_difference_heatmap.png"
                ),
                dpi=300,
                bbox_inches="tight"
            )

            plt.savefig(
                os.path.join(
                    comparison_output,
                    f"GS_apetitive_aversive_transition_{comparison_name}_difference_heatmap.pdf"
                ),
                format="pdf",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            def history_w_to_lw(w, min_w=0.6, max_w=8.0):
                return min_w + (abs(w) / history_lim) * (max_w - min_w)

            def history_w_to_alpha(w, min_a=0.2, max_a=0.95):
                return min_a + (abs(w) / history_lim) * (max_a - min_a)

            G_history = diff_matrix_to_digraph_labels(D_history, thresh=0.02)

            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            ax.set_title(circle_title)
            ax.axis("off")

            nx.draw_networkx_nodes(G_history, pos, ax=ax, node_size=1200)
            nx.draw_networkx_labels(G_history, pos, ax=ax, font_size=10)

            for u, v, dct in G_history.edges(data=True):

                w = float(dct["weight"])
                color = cmap(history_norm(w))
                lw = history_w_to_lw(w)
                alpha = history_w_to_alpha(w)

                rad = 0.18 if u != v else 0.45

                patch = FancyArrowPatch(
                    posA=pos[u],
                    posB=pos[v],
                    arrowstyle="-|>",
                    mutation_scale=16,
                    connectionstyle=f"arc3,rad={rad}",
                    linewidth=lw,
                    color=color,
                    alpha=alpha,
                    shrinkA=22,
                    shrinkB=22
                )

                ax.add_patch(patch)

            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=history_norm)
            sm.set_array([])

            cax = fig.add_axes([0.95, 0.28, 0.02, 0.45])
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_label(difference_label)

            plt.savefig(
                os.path.join(
                    comparison_output,
                    f"GS_apetitive_aversive_transition_{comparison_name}_difference_circlegraph.pdf"
                ),
                format="pdf",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

        plot_history_valence_comparison(
            comparison_name="within_type",
            comparison_output=within_type_output,
            group_a_label="G_with_G_then_G",
            group_a_filter=lambda d: (
                (d["larva_type"] == "G") &
                (d["partner_type"] == "G") &
                (d["next_partner_type"] == "G")
            ),
            group_b_label="S_with_S_then_S",
            group_b_filter=lambda d: (
                (d["larva_type"] == "S") &
                (d["partner_type"] == "S") &
                (d["next_partner_type"] == "S")
            ),
            difference_label="P(G with G then G) - P(S with S then S)",
            heatmap_title="GS apetitive/aversive transition difference: within-type history",
            circle_title="GS apetitive/aversive transition likelihood: within-type history"
        )

        plot_history_valence_comparison(
            comparison_name="mixed_type",
            comparison_output=mixed_output,
            group_a_label="G_with_S_then_S",
            group_a_filter=lambda d: (
                (d["larva_type"] == "G") &
                (d["partner_type"] == "S") &
                (d["next_partner_type"] == "S")
            ),
            group_b_label="S_with_G_then_G",
            group_b_filter=lambda d: (
                (d["larva_type"] == "S") &
                (d["partner_type"] == "G") &
                (d["next_partner_type"] == "G")
            ),
            difference_label="P(G with S then S) - P(S with G then G)",
            heatmap_title="GS apetitive/aversive transition difference: mixed-type history",
            circle_title="GS apetitive/aversive transition likelihood: mixed-type history"
        )

        plot_history_valence_comparison(
            comparison_name="mixed_2",
            comparison_output=mixed_2_output,
            group_a_label="G_with_S_then_G",
            group_a_filter=lambda d: (
                (d["larva_type"] == "G") &
                (d["partner_type"] == "S") &
                (d["next_partner_type"] == "G")
            ),
            group_b_label="S_with_G_then_S",
            group_b_filter=lambda d: (
                (d["larva_type"] == "S") &
                (d["partner_type"] == "G") &
                (d["next_partner_type"] == "S")
            ),
            difference_label="P(G with S then G) - P(S with G then S)",
            heatmap_title="GS apetitive/aversive transition difference: mixed-2 history",
            circle_title="GS apetitive/aversive transition likelihood: mixed-2 history"
        )

        print("Saved GS apetitive/aversive transition analysis.")




    def correlation_contact_G_vs_S(self):



        df = self.clusters.copy()
        df_interaction = self.df.copy()
        cluster_name = self.cluster_name

        output = os.path.join(self.directory, "correlation_contact_G_vs_S")
        os.makedirs(output, exist_ok=True)

        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42

        # Keep only G and S
        df = df[df["condition"].isin(["G", "S"])].copy()
        df_interaction = df_interaction[df_interaction["condition"].isin(["G", "S"])].copy()

        if df.empty:
            print("No G/S data found.")
            return

        # --------------------------------------------------
        # OBSERVED - EXPECTED DEVIATION
        # --------------------------------------------------

        df = df.drop_duplicates(subset=["interaction_id"]).copy()

        cluster_counts = (
            df
            .groupby([cluster_name, "condition"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=["G", "S"], fill_value=0)
        )

        total_G = cluster_counts["G"].sum()
        total_S = cluster_counts["S"].sum()
        total_all = total_G + total_S

        expected_G = total_G / total_all

        observed_G_frac = (
            cluster_counts["G"] /
            (cluster_counts["G"] + cluster_counts["S"]).replace({0: np.nan})
        ).fillna(0)

        deviation = observed_G_frac - expected_G
        deviation_sorted = deviation.sort_values()

        # --------------------------------------------------
        # BINOMIAL TEST PER CLUSTER
        # --------------------------------------------------

        results = []

        for cluster_id, row in cluster_counts.iterrows():

            k = row["G"]
            n = row["G"] + row["S"]

            if n > 0:
                res = binomtest(k, n, expected_G, alternative="two-sided")
                results.append((cluster_id, res.pvalue))
            else:
                results.append((cluster_id, np.nan))

        pvals = pd.DataFrame(results, columns=["cluster_id", "p_value"])

        valid = pvals["p_value"].notna()
        pvals["p_adj"] = np.nan

        if valid.sum() > 0:
            pvals.loc[valid, "p_adj"] = multipletests(
                pvals.loc[valid, "p_value"],
                method="fdr_bh"
            )[1]

        pvals.to_csv(
            os.path.join(output, "G_vs_S_deviation_pvals.csv"),
            index=False
        )

        # --------------------------------------------------
        # CONTACT FRAME SUMMARY
        # --------------------------------------------------

        interaction_contact_summary = []

        for cluster_id in sorted(df_interaction[cluster_name].dropna().unique()):

            cluster_df = df_interaction[df_interaction[cluster_name] == cluster_id]

            for inter_id in cluster_df["interaction_id"].dropna().unique():

                inter_df = cluster_df[cluster_df["interaction_id"] == inter_id]

                n_close = (inter_df["min_distance"] < 1).sum()

                interaction_contact_summary.append({
                    "cluster": cluster_id,
                    "interaction_id": inter_id,
                    "frames_below_1mm": n_close
                })

        df_interaction_contact = pd.DataFrame(interaction_contact_summary)

        mean_frames = (
            df_interaction_contact
            .groupby("cluster")["frames_below_1mm"]
            .mean()
        )

        mean_frames.index = mean_frames.index.astype(deviation_sorted.index.dtype)

        avg_contact_aligned = mean_frames.reindex(deviation_sorted.index)

        # --------------------------------------------------
        # CORRELATION
        # --------------------------------------------------

        correlation_df = pd.DataFrame({
            "cluster": deviation_sorted.index,
            "deviation_G_bias": deviation_sorted.values,
            "avg_contact_frames": avg_contact_aligned.values
        }).dropna()

        rho, p = spearmanr(
            correlation_df["deviation_G_bias"],
            correlation_df["avg_contact_frames"]
        )

        n = len(correlation_df)

        print(f"Spearman rho = {rho:.3f}, p = {p:.3g}, N = {n}")

        correlation_df.to_csv(
            os.path.join(output, "G_vs_S_deviation_contact_correlation.csv"),
            index=False
        )

        # Linear fit only for visual guide
        x = correlation_df["deviation_G_bias"]
        y = correlation_df["avg_contact_frames"]

        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = intercept + slope * x_line

        # --------------------------------------------------
        # PLOT
        # --------------------------------------------------

        fig, ax = plt.subplots(figsize=(4, 4))

        sns.scatterplot(
            data=correlation_df,
            x="deviation_G_bias",
            y="avg_contact_frames",
            s=60,
            color="mediumseagreen",
            edgecolor="gray",
            ax=ax
        )

        ax.plot(x_line, y_line, color="darkgray", linewidth=2)

        for _, row in correlation_df.iterrows():
            ax.text(
                row["deviation_G_bias"],
                row["avg_contact_frames"],
                str(int(row["cluster"])),
                fontsize=8,
                ha="left",
                va="bottom"
            )

        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)

        ax.set_xlabel("Deviation from expected G fraction")
        ax.set_ylabel("Average contact frames")
        ax.set_title(f"Spearman rho = {rho:.2f}, p = {p:.3g}")
        ax.set_xlim(-0.3, 0.3)

        sns.despine()
        plt.tight_layout()


        plt.savefig(
            os.path.join(output, "G_vs_S_deviation_contact_correlation.pdf"),
            format="pdf",
            bbox_inches="tight"
        )

        plt.close()

        print("Saved G vs S contact correlation analysis.")
    


    def GS_directed_movement_over_time(self, contact_threshold=1):

        from scipy.stats import wilcoxon

        df = self.df.copy()
        cluster_name = self.cluster_name

        output = os.path.join(self.directory, "GS_directed_movement_over_time")
        os.makedirs(output, exist_ok=True)

        df = df[
            (df["condition"] == "GS") &
            df["Interaction Pair"].notna()
        ].copy()

        if df.empty:
            print("No GS data found.")
            return

        def parse_pair(pair):
            if pd.isna(pair):
                return None

            if isinstance(pair, str):
                pair = pair.replace("(", "").replace(")", "").replace(" ", "")
                id1, id2 = pair.split(",")
                return int(id1), int(id2)

            id1, id2 = pair
            return int(id1), int(id2)

        def larva_type(track_id):
            return "S" if int(track_id) <= 4 else "G"

        df["parsed_pair"] = df["Interaction Pair"].apply(parse_pair)
        df = df.dropna(subset=["parsed_pair"])
        df["cluster"] = df[cluster_name].astype(int)

        frame_rows = []
        interaction_rows = []

        for interaction_id, inter in df.groupby("interaction_id"):

            inter = inter.sort_values("Normalized Frame").copy()

            id1, id2 = inter["parsed_pair"].iloc[0]

            # only mixed G-S interactions
            if larva_type(id1) == larva_type(id2):
                continue

            # Interaction Pair order maps to Track_1 / Track_2
            if larva_type(id1) == "G":
                G_id = id1
                S_id = id2

                G_x = inter["Track_1 x_body"].to_numpy(dtype=float)
                G_y = inter["Track_1 y_body"].to_numpy(dtype=float)
                S_x = inter["Track_2 x_body"].to_numpy(dtype=float)
                S_y = inter["Track_2 y_body"].to_numpy(dtype=float)

            else:
                S_id = id1
                G_id = id2

                S_x = inter["Track_1 x_body"].to_numpy(dtype=float)
                S_y = inter["Track_1 y_body"].to_numpy(dtype=float)
                G_x = inter["Track_2 x_body"].to_numpy(dtype=float)
                G_y = inter["Track_2 y_body"].to_numpy(dtype=float)

            frames = inter["Normalized Frame"].to_numpy()
            min_dist = inter["min_distance"].to_numpy(dtype=float)

            contact_idx = np.where(min_dist < contact_threshold)[0]

            if len(contact_idx) == 0:
                continue

            first_contact_idx = contact_idx[0]

            if first_contact_idx < 1:
                continue

            # movement steps before/into first contact
            idx = np.arange(1, first_contact_idx + 1)

            G_prev = np.column_stack([G_x[idx - 1], G_y[idx - 1]])
            G_curr = np.column_stack([G_x[idx], G_y[idx]])

            S_prev = np.column_stack([S_x[idx - 1], S_y[idx - 1]])
            S_curr = np.column_stack([S_x[idx], S_y[idx]])

            G_move = G_curr - G_prev
            S_move = S_curr - S_prev

            # axis from G to S at previous frame
            G_to_S = S_prev - G_prev
            dist = np.linalg.norm(G_to_S, axis=1)

            valid = np.isfinite(dist) & (dist > 0)

            if valid.sum() == 0:
                continue

            unit_G_to_S = np.zeros_like(G_to_S)
            unit_G_to_S[valid] = G_to_S[valid] / dist[valid, None]

            # signed projection:
            # positive = moving toward the other larva
            # negative = moving away
            G_signed = np.sum(G_move * unit_G_to_S, axis=1)
            S_signed = np.sum(S_move * (-unit_G_to_S), axis=1)

            # positive-only closing contribution
            G_closing = np.clip(G_signed, 0, None)
            S_closing = np.clip(S_signed, 0, None)

            driving_difference = G_closing - S_closing

            for n, frame_idx in enumerate(idx):

                if not valid[n]:
                    continue

                frame_rows.append({
                    "file": inter["file"].iloc[0],
                    "interaction_id": interaction_id,
                    "cluster": inter["cluster"].iloc[0],
                    "G_id": G_id,
                    "S_id": S_id,
                    "Normalized Frame": frames[frame_idx],
                    "min_distance": min_dist[frame_idx],
                    "G_signed_projection": G_signed[n],
                    "S_signed_projection": S_signed[n],
                    "G_closing_contribution": G_closing[n],
                    "S_closing_contribution": S_closing[n],
                    "driving_difference_G_minus_S": driving_difference[n]
                })

            interaction_rows.append({
                "file": inter["file"].iloc[0],
                "interaction_id": interaction_id,
                "cluster": inter["cluster"].iloc[0],
                "G_id": G_id,
                "S_id": S_id,
                "mean_G_closing_contribution": np.nanmean(G_closing[valid]),
                "mean_S_closing_contribution": np.nanmean(S_closing[valid]),
                "mean_driving_difference_G_minus_S": np.nanmean(driving_difference[valid]),
                "first_contact_frame": frames[first_contact_idx],
                "frames_before_contact": first_contact_idx,
                "contact_threshold": contact_threshold
            })

        frame_df = pd.DataFrame(frame_rows)
        interaction_df = pd.DataFrame(interaction_rows)

        if frame_df.empty or interaction_df.empty:
            print("No valid mixed G-S pre-contact interactions found.")
            return

        frame_df.to_csv(
            os.path.join(output, "GS_directed_movement_frame_level.csv"),
            index=False
        )

        interaction_df.to_csv(
            os.path.join(output, "GS_directed_movement_per_interaction.csv"),
            index=False
        )

        # --------------------------------------------------
        # STATS PER CLUSTER
        # positive = G contributes more to closing distance
        # negative = S contributes more
        # --------------------------------------------------

        stats_rows = []

        for cluster_id, sub in interaction_df.groupby("cluster"):

            vals = sub["mean_driving_difference_G_minus_S"].dropna()

            if len(vals) >= 2 and not np.allclose(vals, 0):
                W, p = wilcoxon(vals, alternative="two-sided")
            elif len(vals) >= 2 and np.allclose(vals, 0):
                W, p = 0, 1.0
            else:
                W, p = np.nan, np.nan

            stats_rows.append({
                "cluster": cluster_id,
                "n": len(vals),
                "mean_G_closing": sub["mean_G_closing_contribution"].mean(),
                "mean_S_closing": sub["mean_S_closing_contribution"].mean(),
                "mean_difference_G_minus_S": vals.mean(),
                "median_difference_G_minus_S": vals.median(),
                "W": W,
                "p": p
            })

        stats = pd.DataFrame(stats_rows)

        stats["p_adj"] = np.nan
        mask = stats["p"].notna()

        if mask.sum() > 0:
            stats.loc[mask, "p_adj"] = multipletests(
                stats.loc[mask, "p"],
                method="fdr_bh"
            )[1]

        stats.to_csv(
            os.path.join(output, "GS_directed_movement_stats_by_cluster.csv"),
            index=False
        )

        # --------------------------------------------------
        # LONG FORMAT FOR TIMECOURSE PLOT
        # --------------------------------------------------

        plot_long = pd.concat([
            frame_df.rename(columns={"G_closing_contribution": "closing_contribution"})
            .assign(driver="G")[[
                "file", "interaction_id", "cluster", "Normalized Frame",
                "closing_contribution", "driver"
            ]],

            frame_df.rename(columns={"S_closing_contribution": "closing_contribution"})
            .assign(driver="S")[[
                "file", "interaction_id", "cluster", "Normalized Frame",
                "closing_contribution", "driver"
            ]]
        ])

        plot_long.to_csv(
            os.path.join(output, "GS_directed_movement_plot_long.csv"),
            index=False
        )

        # --------------------------------------------------
        # PLOT: G vs S over time, per cluster
        # --------------------------------------------------

        clusters = sorted(plot_long["cluster"].dropna().unique())

        ncols = 4
        nrows = int(np.ceil(len(clusters) / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(ncols * 4, nrows * 3),
            sharex=True,
            sharey=True
        )

        axes = np.array(axes).flatten()

        palette = {
            "G": "C0",
            "S": "C1"
        }

        for ax, cluster_id in zip(axes, clusters):

            d = plot_long[plot_long["cluster"] == cluster_id]

            sns.lineplot(
                data=d,
                x="Normalized Frame",
                y="closing_contribution",
                hue="driver",
                hue_order=["G", "S"],
                palette=palette,
                errorbar=("ci", 95),
                ax=ax
            )

            ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
            ax.set_title(f"Cluster {cluster_id}")
            ax.set_xlabel("Normalized Frame")
            ax.set_ylabel("Closing contribution")

            if ax.get_legend() is not None:
                ax.get_legend().remove()

        for ax in axes[len(clusters):]:
            ax.axis("off")

        handles, labels = axes[0].get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            title="Driver",
            loc="upper right",
            bbox_to_anchor=(1.02, 1)
        )

        fig.suptitle(
            "GS mixed interactions: G vs S directed movement before contact",
            fontsize=14,
            fontweight="bold"
        )

        plt.tight_layout(rect=[0, 0, 0.96, 0.95])

        plt.savefig(
            os.path.join(output, "GS_directed_movement_over_time_by_cluster.png"),
            dpi=300,
            bbox_inches="tight"
        )

        plt.savefig(
            os.path.join(output, "GS_directed_movement_over_time_by_cluster.pdf"),
            format="pdf",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        # --------------------------------------------------
        # PLOT: interaction-level difference by cluster
        # --------------------------------------------------

        plt.figure(figsize=(10, 5))

        sns.stripplot(
            data=interaction_df,
            x="cluster",
            y="mean_driving_difference_G_minus_S",
            order=clusters,
            color="black",
            alpha=0.35,
            jitter=True,
            size=3
        )

        sns.pointplot(
            data=interaction_df,
            x="cluster",
            y="mean_driving_difference_G_minus_S",
            order=clusters,
            errorbar=("ci", 95),
            color="mediumseagreen",
            join=False,
            markers="D",
            scale=0.8
        )

        plt.axhline(0, color="black", linestyle="--", linewidth=1)

        plt.xlabel("Cluster")
        plt.ylabel("Mean closing difference\nG - S")
        plt.title("GS mixed interactions: overall directed movement difference")

        plt.tight_layout()

        plt.savefig(
            os.path.join(output, "GS_directed_movement_difference_by_cluster.png"),
            dpi=300,
            bbox_inches="tight"
        )

        plt.savefig(
            os.path.join(output, "GS_directed_movement_difference_by_cluster.pdf"),
            format="pdf",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print("Saved GS directed movement over-time analysis.")
    

    def social_context_video_proportion(self):

        df = self.df.copy()
        cluster_name = self.cluster_name

        output = os.path.join(self.directory, "social_context_video_proportion")
        os.makedirs(output, exist_ok=True)

        df = df[np.isclose(df["Normalized Frame"], 0)].copy()

        def parse_pair(pair):

            if pd.isna(pair):
                return np.nan

            if isinstance(pair, str):
                pair = pair.replace("(", "").replace(")", "").replace(" ", "")
                id1, id2 = pair.split(",")
                id1 = int(id1)
                id2 = int(id2)
            else:
                id1, id2 = pair
                id1 = int(id1)
                id2 = int(id2)

            return tuple(sorted((id1, id2)))

        def get_social_experience(pair):

            if pair is None or pd.isna(pair):
                return np.nan

            id1, id2 = pair

            if id1 <= 4 and id2 <= 4:
                return "S(GS)"
            elif id1 >= 5 and id2 >= 5:
                return "G(GS)"
            else:
                return np.nan

        # assign social context
        df["parsed_pair"] = df["Interaction Pair"].apply(parse_pair)

        g_df = df[df["condition"] == "G"].copy()
        g_df["social_context"] = "G"

        s_df = df[df["condition"] == "S"].copy()
        s_df["social_context"] = "S"

        gs_df = df[df["condition"] == "GS"].copy()
        gs_df["social_context"] = gs_df["parsed_pair"].apply(get_social_experience)
        gs_df = gs_df.dropna(subset=["social_context"])

        proportion_df = pd.concat([g_df, s_df, gs_df], ignore_index=True)

        # count cluster use per video
        counts = (
            proportion_df
            .groupby(["file", "social_context", cluster_name])
            .size()
            .reset_index(name="count")
        )

        totals = (
            proportion_df
            .groupby(["file", "social_context"])
            .size()
            .reset_index(name="total")
        )

        proportions = counts.merge(
            totals,
            on=["file", "social_context"],
            how="left"
        )

        proportions["proportion"] = proportions["count"] / proportions["total"]

        proportions.to_csv(
            os.path.join(output, "raw_social_context_video_proportions.csv"),
            index=False
        )

        # stats
        results = []

        comparisons = [
            ("G", "G(GS)"),
            ("S", "S(GS)")
        ]

        clusters = sorted(proportions[cluster_name].dropna().unique())

        for cluster in clusters:

            for group1, group2 in comparisons:

                x = proportions[
                    (proportions[cluster_name] == cluster) &
                    (proportions["social_context"] == group1)
                ]["proportion"]

                y = proportions[
                    (proportions[cluster_name] == cluster) &
                    (proportions["social_context"] == group2)
                ]["proportion"]

                if len(x) > 0 and len(y) > 0:
                    stat, p = mannwhitneyu(x, y, alternative="two-sided")
                else:
                    stat, p = np.nan, np.nan

                results.append({
                    cluster_name: cluster,
                    "comparison": f"{group1} vs {group2}",
                    "group1": group1,
                    "group2": group2,
                    "n_group1": len(x),
                    "n_group2": len(y),
                    "mean_group1": x.mean(),
                    "mean_group2": y.mean(),
                    "median_group1": x.median(),
                    "median_group2": y.median(),
                    "mannwhitney_U": stat,
                    "p_value": p
                })

        stats_df = pd.DataFrame(results)

        stats_df["p_fdr"] = np.nan

        for comparison in stats_df["comparison"].unique():

            mask = (
                (stats_df["comparison"] == comparison) &
                (stats_df["p_value"].notna())
            )

            stats_df.loc[mask, "p_fdr"] = multipletests(
                stats_df.loc[mask, "p_value"],
                method="fdr_bh"
            )[1]



        stats_df.to_csv(
            os.path.join(output, "social_context_video_proportion_stats.csv"),
            index=False
        )

        # plots
        order = ["G", "G(GS)", "S", "S(GS)"]

        for cluster in clusters:

            plot_df = proportions[proportions[cluster_name] == cluster].copy()
            plot_df["social_context"] = pd.Categorical(
                plot_df["social_context"],
                categories=order,
                ordered=True
            )

            plt.figure(figsize=(6, 5))

            sns.stripplot(
                data=plot_df,
                x="social_context",
                y="proportion",
                order=order,
                jitter=True,
                alpha=0.6
            )

            sns.pointplot(
                data=plot_df,
                x="social_context",
                y="proportion",
                order=order,
                errorbar="se",
                join=False,
                color="black"
            )

            plt.title(f"Cluster {cluster}: social context proportion")
            plt.xlabel("Social context")
            plt.ylabel("Proportion of interactions")
            plt.tight_layout()

            plt.savefig(
                os.path.join(output, f"cluster_{cluster}_social_context_video_proportion.png"),
                dpi=300
            )
            plt.close()

        print(f"Saved social context video proportion analysis to: {output}")
    


    def social_context_deviation_plots(self):

        df = self.df.copy()
        cluster_name = self.cluster_name

        output = os.path.join(self.directory, "social_context_deviation_plots")
        os.makedirs(output, exist_ok=True)

        df = df[np.isclose(df["Normalized Frame"], 0)].copy()

        def parse_pair(pair):

            if pd.isna(pair):
                return np.nan

            if isinstance(pair, str):
                pair = pair.replace("(", "").replace(")", "").replace(" ", "")
                id1, id2 = pair.split(",")
                id1 = int(id1)
                id2 = int(id2)
            else:
                id1, id2 = pair
                id1 = int(id1)
                id2 = int(id2)

            return tuple(sorted((id1, id2)))

        def get_social_experience(pair):

            if pair is None or pd.isna(pair):
                return np.nan

            id1, id2 = pair

            if id1 <= 4 and id2 <= 4:
                return "S(GS)"
            elif id1 >= 5 and id2 >= 5:
                return "G(GS)"
            else:
                return np.nan

        df["parsed_pair"] = df["Interaction Pair"].apply(parse_pair)

        g_df = df[df["condition"] == "G"].copy()
        g_df["social_context"] = "G"

        s_df = df[df["condition"] == "S"].copy()
        s_df["social_context"] = "S"

        gs_df = df[df["condition"] == "GS"].copy()
        gs_df["social_context"] = gs_df["parsed_pair"].apply(get_social_experience)
        gs_df = gs_df.dropna(subset=["social_context"])

        proportion_df = pd.concat([g_df, s_df, gs_df], ignore_index=True)

        counts = (
            proportion_df
            .groupby(["file", "social_context", cluster_name])
            .size()
            .reset_index(name="count")
        )

        totals = (
            proportion_df
            .groupby(["file", "social_context"])
            .size()
            .reset_index(name="total")
        )

        proportions = counts.merge(
            totals,
            on=["file", "social_context"],
            how="left"
        )

        proportions["proportion"] = proportions["count"] / proportions["total"]

        # fill missing clusters as 0
        clusters = sorted(proportion_df[cluster_name].dropna().unique())

        full_index = (
            proportions[["file", "social_context"]]
            .drop_duplicates()
            .assign(key=1)
            .merge(pd.DataFrame({cluster_name: clusters, "key": 1}), on="key")
            .drop(columns="key")
        )

        proportions = full_index.merge(
            proportions,
            on=["file", "social_context", cluster_name],
            how="left"
        )

        proportions["count"] = proportions["count"].fillna(0)
        proportions["proportion"] = proportions["proportion"].fillna(0)

        proportions.to_csv(
            os.path.join(output, "raw_social_context_proportions_for_deviation.csv"),
            index=False
        )

        # baseline mean per cluster
        g_baseline = (
            proportions[proportions["social_context"] == "G"]
            .groupby(cluster_name)["proportion"]
            .mean()
            .reset_index(name="baseline_proportion")
        )

        s_baseline = (
            proportions[proportions["social_context"] == "S"]
            .groupby(cluster_name)["proportion"]
            .mean()
            .reset_index(name="baseline_proportion")
        )

        g_gs = proportions[proportions["social_context"] == "G(GS)"].copy()
        g_gs = g_gs.merge(g_baseline, on=cluster_name, how="left")
        g_gs["deviation"] = g_gs["proportion"] - g_gs["baseline_proportion"]
        g_gs["comparison"] = "G(GS) - mean(G)"

        s_gs = proportions[proportions["social_context"] == "S(GS)"].copy()
        s_gs = s_gs.merge(s_baseline, on=cluster_name, how="left")
        s_gs["deviation"] = s_gs["proportion"] - s_gs["baseline_proportion"]
        s_gs["comparison"] = "S(GS) - mean(S)"

        deviation_df = pd.concat([g_gs, s_gs], ignore_index=True)

        deviation_df.to_csv(
            os.path.join(output, "social_context_cluster_deviations.csv"),
            index=False
        )

        # plot G deviation
        for comparison, filename, title in [
            ("G(GS) - mean(G)", "G_GS_minus_G_deviation.png", "G(GS) deviation from G baseline"),
            ("S(GS) - mean(S)", "S_GS_minus_S_deviation.png", "S(GS) deviation from S baseline")
        ]:

            plot_df = deviation_df[deviation_df["comparison"] == comparison].copy()

            plt.figure(figsize=(9, 5))

            plt.axhline(0, color="black", linestyle="--", linewidth=1)

            sns.stripplot(
                data=plot_df,
                x=cluster_name,
                y="deviation",
                jitter=True,
                alpha=0.6
            )

            sns.pointplot(
                data=plot_df,
                x=cluster_name,
                y="deviation",
                errorbar="se",
                join=False,
                color="black"
            )

            plt.title(title)
            plt.xlabel("Cluster")
            plt.ylabel("Deviation in proportion")
            plt.tight_layout()

            plt.savefig(
                os.path.join(output, filename),
                dpi=300
            )
            plt.close()

        print(f"Saved social context deviation plots to: {output}")




    def social_context_barplot_deviation(self):

        df = self.df.copy()
        cluster_name = self.cluster_name

        output = os.path.join(self.directory, "social_context_barplot_deviation")
        os.makedirs(output, exist_ok=True)

        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype']  = 42

        df = df[np.isclose(df["Normalized Frame"], 0)].copy()

        def parse_pair(pair):

            if pd.isna(pair):
                return np.nan

            if isinstance(pair, str):
                pair = pair.replace("(", "").replace(")", "").replace(" ", "")
                id1, id2 = pair.split(",")
                id1 = int(id1)
                id2 = int(id2)
            else:
                id1, id2 = pair
                id1 = int(id1)
                id2 = int(id2)

            return tuple(sorted((id1, id2)))

        def get_social_experience(pair):

            if pair is None:
                return np.nan

            if isinstance(pair, float) and np.isnan(pair):
                return np.nan

            id1, id2 = pair

            if id1 <= 4 and id2 <= 4:
                return "S(GS)"
            elif id1 >= 5 and id2 >= 5:
                return "G(GS)"
            else:
                return "G-S"

        df["parsed_pair"] = df["Interaction Pair"].apply(parse_pair)

        g_df = df[df["condition"] == "G"].copy()
        g_df["social_context"] = "G"

        s_df = df[df["condition"] == "S"].copy()
        s_df["social_context"] = "S"

        gs_df = df[df["condition"] == "GS"].copy()
        gs_df["social_context"] = gs_df["parsed_pair"].apply(get_social_experience)

        plot_df = pd.concat([g_df, s_df, gs_df], ignore_index=True)

        comparisons = [
            ("G", "G(GS)", "G_GS_vs_G_deviation"),
            ("S", "S(GS)", "S_GS_vs_S_deviation")
        ]

        for baseline, gs_context, name in comparisons:

            comp_df = plot_df[
                (plot_df["social_context"] == baseline) |
                (plot_df["social_context"] == gs_context)
            ].copy()

            cluster_counts = (
                comp_df
                .groupby([cluster_name, "social_context"])
                .size()
                .unstack(fill_value=0)
                .reindex(columns=[baseline, gs_context], fill_value=0)
            )

            total_baseline = cluster_counts[baseline].sum()
            total_gs = cluster_counts[gs_context].sum()
            total_all = total_baseline + total_gs

            expected_gs = total_gs / total_all

            observed_gs_frac = (
                cluster_counts[gs_context] /
                (cluster_counts[baseline] + cluster_counts[gs_context]).replace({0: np.nan})
            )

            observed_gs_frac = observed_gs_frac.fillna(0.0)

            deviation = observed_gs_frac - expected_gs
            deviation_sorted = deviation.sort_values()

            colors = ['C1' if val < 0 else 'C0' for val in deviation_sorted.values]

            results = []

            for cluster_id, row in cluster_counts.iterrows():

                k = row[gs_context]
                n = row[baseline] + row[gs_context]

                if n > 0:
                    res = binomtest(k, n, expected_gs, alternative='two-sided')
                    results.append((cluster_id, res.pvalue))
                else:
                    results.append((cluster_id, np.nan))

            pvals = pd.DataFrame(results, columns=[cluster_name, "p_value"])

            valid = pvals["p_value"].notna()
            pvals.loc[valid, "p_adj"] = multipletests(
                pvals.loc[valid, "p_value"],
                method="fdr_bh"
            )[1]

            pvals.to_csv(
                os.path.join(output, f"{name}_pvals.csv"),
                index=False,
                float_format="%.10f"
            )

            deviation_out = deviation_sorted.reset_index()
            deviation_out.columns = [cluster_name, "deviation"]
            deviation_out.to_csv(
                os.path.join(output, f"{name}_deviation_values.csv"),
                index=False
            )

            fig, ax1 = plt.subplots(figsize=(10, 6))

            x_labels = deviation_sorted.index.astype(str)
            x_pos = np.arange(len(x_labels))

            ax1.bar(
                x_labels,
                deviation_sorted.values,
                color=colors,
                alpha=0.7,
                edgecolor='black',
                linewidth=1.5
            )

            ax1.axhline(0, color='k', linestyle='--', linewidth=1)

            p_map = pvals.set_index(cluster_name)["p_adj"]

            def stars(p):
                if p < 0.001: return '***'
                if p < 0.01:  return '**'
                if p < 0.05:  return '*'
                return ''

            ymin, ymax = ax1.get_ylim()
            dy = 0.015 * (ymax - ymin)

            for i, cid in enumerate(deviation_sorted.index):
                p = p_map.get(cid, np.nan)

                if pd.notna(p):
                    s = stars(p)

                    if s:
                        y = deviation_sorted.loc[cid]
                        ax1.text(
                            i,
                            y + (dy if y >= 0 else -dy),
                            s,
                            ha='center',
                            va='bottom' if y >= 0 else 'top',
                            fontsize=10,
                            fontweight='bold',
                            color='black'
                        )

            ax1.set_title(
                f"{gs_context} Deviation from {baseline} Expected",
                fontsize=16,
                fontweight='bold',
                pad=15
            )

            ax1.set_ylabel("Deviation from Expected")
            ax1.set_xlabel("Cluster")
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(x_labels, fontweight='bold', fontsize=14)

            plt.tight_layout()

            plt.savefig(
                os.path.join(output, f"{name}.png"),
                dpi=300,
                bbox_inches="tight"
            )

            plt.savefig(
                os.path.join(output, f"{name}.pdf"),
                format="pdf",
                dpi=300,
                bbox_inches="tight",
                transparent=True
            )

            plt.close()

        print(f"Saved social context barplot deviation analysis to: {output}")



    def umap(self, n_pcs=5):

        df = self.df.copy()
        cluster_name = self.cluster_name

        output = os.path.join(self.directory, "umap")
        os.makedirs(output, exist_ok=True)

        def parse_pair(pair):

            if pd.isna(pair):
                return np.nan

            if isinstance(pair, str):
                pair = pair.replace("(", "").replace(")", "").replace(" ", "")
                id1, id2 = pair.split(",")
                id1 = int(id1)
                id2 = int(id2)
            else:
                id1, id2 = pair
                id1 = int(id1)
                id2 = int(id2)

            return tuple(sorted((id1, id2)))


        def get_gs_pair_type(pair):

            if not isinstance(pair, tuple) or len(pair) != 2:
                return np.nan

            id1, id2 = pair

            if id1 <= 4 and id2 <= 4:
                return "S-S"
            elif id1 >= 5 and id2 >= 5:
                return "G-G"
            else:
                return "G-S"


        def get_social_context(row):

            if row["condition"] == "G":
                return "G"

            elif row["condition"] == "S":
                return "S"

            elif row["condition"] == "GS":

                pair = row["parsed_pair"]

                pair_type = get_gs_pair_type(pair)

                if pd.isna(pair_type):
                    return np.nan

                if pair_type == "S-S":
                    return "S(GS)"
                elif pair_type == "G-G":
                    return "G(GS)"
                else:
                    return "G-S"

            return np.nan


        def get_umap_group_label(row):

            if row["condition"] == "G":
                return "G"

            if row["condition"] == "GS" and row["gs_pair_type"] in ["G-S", "G-G"]:
                return "G in GS"

            return np.nan


        def get_umap_iso_label(row):

            if row["condition"] == "S":
                return "S"

            if row["condition"] == "GS" and row["gs_pair_type"] in ["S-S", "G-S"]:
                return "S in GS"

            return np.nan

        def get_appetitive_aversive_label(cluster_id):

            if pd.isna(cluster_id):
                return np.nan

            cluster_id = int(float(cluster_id))

            if cluster_id in [1, 3, 4, 6, 7, 9]:
                return "appetitive"

            if cluster_id in [2, 5, 8, 10, 11, 12]:
                return "aversive"

            return np.nan

        features = [
            "t1_head-head_t2",
            "track1_speed_head",
            "track1_speed_tail",
            "track2_speed_head",
            "track2_speed_tail",
            "track1_acceleration_body",
            "track1_acceleration_tail",
            "track1_acceleration_head",
            "track2_acceleration_body",
            "track2_acceleration_tail",
            "track2_acceleration_head",
            "track1_angle",
            "track2_angle",
            "track1_approach_angle",
            "track2_approach_angle",
        ]

        df = df.sort_values(["interaction_id", "Normalized Frame"]).copy()

        features = [f for f in features if f in df.columns]

        vector_df = (
            df
            .set_index(["interaction_id", "Normalized Frame"])[features]
            .unstack("Normalized Frame")
        )

        vector_df.columns = [
            f"{feature}_{frame}"
            for feature, frame in vector_df.columns
        ]

        vector_df.to_csv(
            os.path.join(output, "interaction_vectors_step1.csv")
        )


        metadata = (
            df[
                [
                    "interaction_id",
                    cluster_name,
                    "condition",
                    "file",
                    "Interaction Pair"
                ]
            ]
            .drop_duplicates(subset="interaction_id")
            .set_index("interaction_id")
        )

        metadata.to_csv(
            os.path.join(output, "interaction_metadata.csv")
        )



        # scaler = StandardScaler()
        # X_scaled = scaler.fit_transform(vector_df)

        # pca = PCA()
        # pca_scores = pca.fit_transform(X_scaled)

        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(vector_df)

        # Standardise each vector dimension
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        # Perform PCA
        pca = PCA()
        pca_scores = pca.fit_transform(X_scaled)



        explained = pd.DataFrame({
            "PC": np.arange(1, len(pca.explained_variance_ratio_) + 1),
            "explained_variance": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_)
        })

        explained.to_csv(
            os.path.join(output, "pca_explained_variance.csv"),
            index=False
        )

        scores = pd.DataFrame(
            pca_scores[:, :10],
            index=vector_df.index,
            columns=[f"PC{i}" for i in range(1, 11)]
        )

    

        scores = scores.join(metadata)

        scores["parsed_pair"] = scores["Interaction Pair"].apply(parse_pair)
        scores["gs_pair_type"] = scores["parsed_pair"].apply(get_gs_pair_type)
        scores["social_context"] = scores.apply(get_social_context, axis=1)

        scores.to_csv(
            os.path.join(output, "pca_scores.csv")
        )

        plt.figure(figsize=(6,5))

        plt.plot(
            explained["PC"],
            explained["cumulative_explained_variance"],
            marker="o"
        )

        plt.xlabel("Principal Component")
        plt.ylabel("Proportion of Variance Explained")
        plt.tight_layout()

        plt.savefig(os.path.join(output, "pca_explained_variance.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(7,6))

        sns.scatterplot(
            data=scores,
            x="PC1",
            y="PC2",
            hue=cluster_name,
            s=12,
            alpha=0.5,
            linewidth=0,
            palette='tab20'
        )

        plt.title("PCA coloured by cluster")
        plt.tight_layout()

        plt.savefig(
            os.path.join(output, "pca_cluster.pdf"),
            format="pdf",
            transparent=True
        )
        plt.close()


        plt.figure(figsize=(7,6))

        sns.scatterplot(
            data=scores,
            x="PC1",
            y="PC2",
            hue="condition",
            s=12,
            alpha=0.5,
            linewidth=0
        )

        plt.title("PCA coloured by condition")
        plt.tight_layout()


        plt.savefig(
            os.path.join(output, "pca_condition.pdf"),
            format="pdf",
            transparent=True
        )
        plt.close()


        plt.figure(figsize=(7,6))

        sns.scatterplot(
            data=scores,
            x="PC1",
            y="PC2",
            hue="social_context",
            s=12,
            alpha=0.5,
            linewidth=0
        )

        plt.title("PCA coloured by social context")
        plt.tight_layout()

        plt.savefig(
            os.path.join(output, "pca_social_context.pdf"),
            format="pdf",

            transparent=True
        )
        plt.close()


        ## UMAP
        # Number of principal components to use
        X_umap = pca_scores[:, :n_pcs]

        # Gentle cluster-assisted UMAP for a clearer visualisation of known clusters.
        # Set cluster_label_weight to 0 for a fully unsupervised UMAP.
        cluster_label_weight = 1.0
        if cluster_label_weight > 0:
            cluster_labels = pd.get_dummies(metadata[cluster_name].astype(str))
            cluster_labels = cluster_labels.reindex(vector_df.index).to_numpy(dtype=float)
            X_umap = np.hstack([X_umap, cluster_labels * cluster_label_weight])

        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.02,
            metric="euclidean",
            random_state=42,
            init="spectral",
            spread=1.2
        )

        # reducer = umap.UMAP(
        #         n_neighbors=5,
        #         min_dist=0.001,
        #         metric="cosine",
        #         random_state=42,
        #         init="spectral",
        #     )
        
        embedding = reducer.fit_transform(X_umap)

        umap_df = pd.DataFrame(
            embedding,
            index=vector_df.index,
            columns=["UMAP1", "UMAP2"]
        )

        umap_df = umap_df.join(metadata)

        umap_df["parsed_pair"] = umap_df["Interaction Pair"].apply(parse_pair)
        umap_df["gs_pair_type"] = umap_df["parsed_pair"].apply(get_gs_pair_type)
        umap_df["social_context"] = umap_df.apply(get_social_context, axis=1)
        umap_df["umap_group"] = umap_df.apply(get_umap_group_label, axis=1)
        umap_df["umap_iso"] = umap_df.apply(get_umap_iso_label, axis=1)
        umap_df["umap_aversive_appetitive"] = umap_df[cluster_name].apply(
            get_appetitive_aversive_label
        )

        umap_df.to_csv(
            os.path.join(output, "umap_coordinates.csv")
        )

        plt.figure(figsize=(7,6))

        sns.scatterplot(
            data=umap_df,
            x="UMAP1",
            y="UMAP2",
            hue=cluster_name,
            s=8,
            alpha=0.8,
            linewidth=0,
            palette='tab20'
        )

        plt.title(f"UMAP ({n_pcs} PCs) coloured by cluster")

        plt.tight_layout()

        plt.savefig(os.path.join(output, "umap_cluster.pdf"),
  
                    format="pdf",
                    transparent=True)

        plt.close()

        plt.figure(figsize=(7,6))

        sns.scatterplot(
            data=umap_df,
            x="UMAP1",
            y="UMAP2",
            hue="condition",
            s=8,
            alpha=0.4,
            linewidth=0
        )

        plt.title(f"UMAP ({n_pcs} PCs) coloured by condition")

        plt.tight_layout()

        plt.savefig(os.path.join(output, "umap_condition.pdf"),
                    format="pdf",
                    transparent=True)

        plt.close()


        gs_umap_df = umap_df[
            (umap_df["condition"] == "GS") &
            (umap_df["gs_pair_type"].notna())
        ].copy()

        plt.figure(figsize=(7,6))

        sns.scatterplot(
            data=gs_umap_df,
            x="UMAP1",
            y="UMAP2",
            hue="gs_pair_type",
            hue_order=["S-S", "G-S", "G-G"],
            palette={
                "S-S": "royalblue",
                "G-S": "darkorange",
                "G-G": "mediumseagreen"
            },
            s=8,
            alpha=0.5,
            linewidth=0
        )

        plt.title(f"UMAP ({n_pcs} PCs) within GS coloured by interaction type")

        plt.tight_layout()

        plt.savefig(os.path.join(output, "umap_social_context.pdf"),
                    format="pdf",
                    transparent=True)

        plt.close()


        plt.figure(figsize=(7,6))

        sns.scatterplot(
            data=umap_df.dropna(subset=["umap_group"]),
            x="UMAP1",
            y="UMAP2",
            hue="umap_group",
            hue_order=["G", "G in GS"],
            palette={
                "G": "mediumseagreen",
                "G in GS": "darkorange"
            },
            s=8,
            alpha=0.45,
            linewidth=0
        )

        plt.title(f"UMAP ({n_pcs} PCs): G condition vs G-including GS interactions")

        plt.tight_layout()

        plt.savefig(os.path.join(output, "umap_group.pdf"),
                    format="pdf",
                    transparent=True)

        plt.close()


        plt.figure(figsize=(7,6))

        sns.scatterplot(
            data=umap_df.dropna(subset=["umap_iso"]),
            x="UMAP1",
            y="UMAP2",
            hue="umap_iso",
            hue_order=["S", "S in GS"],
            palette={
                "S": "royalblue",
                "S in GS": "darkorange"
            },
            s=8,
            alpha=0.45,
            linewidth=0
        )

        plt.title(f"UMAP ({n_pcs} PCs): S condition vs S-including GS interactions")

        plt.tight_layout()

        plt.savefig(os.path.join(output, "umap_iso.pdf"),
                    format="pdf",
                    transparent=True)

        plt.close()


        plt.figure(figsize=(7,6))

        sns.scatterplot(
            data=umap_df.dropna(subset=["umap_aversive_appetitive"]),
            x="UMAP1",
            y="UMAP2",
            hue="umap_aversive_appetitive",
            hue_order=["appetitive", "aversive"],
            palette={
                "appetitive": "blue",
                "aversive": "red"
            },
            s=8,
            alpha=0.8,
            linewidth=0
        )

        plt.title(f"UMAP ({n_pcs} PCs) coloured by appetitive/aversive")

        plt.tight_layout()

        plt.savefig(os.path.join(output, "umap_aversive_apeititve.pdf"),
                    format="pdf",
                    transparent=True)

        plt.close()
    


        ## METHOD SPATIAL_CLUSTER: MAP OUT WHERE EACH INTERACTION OCCURED ON THE PETRI DISH
    def spatial_cluster(self):

        df = self.df

        df = df[df['Normalized Frame'] == 0] 

        output = os.path.join(self.directory, "spatial")
        os.makedirs(output, exist_ok=True)

        cluster_ids = sorted(df[cluster_name].unique())
        num_clusters = len(cluster_ids)
        cols = 4
        rows = int(np.ceil(num_clusters / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), sharex=True, sharey=True)
        axes = axes.flatten()

        for idx, cluster_id in enumerate(cluster_ids):
            ax = axes[idx]
            cluster_df = df[df[cluster_name] == cluster_id].copy()

            cluster_df["mid_x"] = (cluster_df["Track_1 x_body"] + cluster_df["Track_2 x_body"]) / 2
            cluster_df["mid_y"] = (cluster_df["Track_1 y_body"] + cluster_df["Track_2 y_body"]) / 2

            ax.scatter(cluster_df['mid_x'], cluster_df['mid_y'], s=10, alpha=0.5, edgecolors='none')
            ax.set_title(f"Cluster {cluster_id}")
            ax.set_ylim(0, 1400)
            ax.set_xlim(0, 1400)
            ax.set_xticks([])
            ax.set_yticks([])
        
        for i in range(len(cluster_ids), len(axes)):
            axes[i].axis('off')

        plt.suptitle("Normalised Frame per Cluster", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join(output, "spatial_plot_per_cluster.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()





      







                












        




if __name__ == "__main__":
    # Set your paths
    directory = "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/clustered-interactions_pt2/F15_KMAX3_DMAX4_NMIN1500--"
    interactions = "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/clustered-interactions_pt2/cropped_interactions.csv"
    clusters = "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/clustered-interactions_pt2/F15_KMAX3_DMAX4_NMIN1500--/pca-data3-F15-mcmodels4-Kmax3-Dmax4-Nmin1500-05-2026.csv"
    cluster_name = "Yhat.idt.pca"   # or whatever your cluster column is
    video_path = "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/clustered-interactions_pt2/videos_original"
    tracks_path = "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/grouped+isolated"

    # Create instance
    pipeline = ClusterPipeline(directory, interactions, clusters, cluster_name, video_path, tracks_path)

    # Run methods
    pipeline.loading_data()

    # merge_dict = {
    # 1: 1,
    # 2: 1,
    # 3: 3,
    # 4: 4,
    # 5: 5,
    # 6: 6,
    # 7: 7,
    # 8: 8,
    # 9: 9,
    # 10: 10,
    # 11: 10,
    # 12: 10}


    # pipeline.merge_clusters(merge_dict)



    # pipeline.anchor_partner()
    # pipeline.grid_videos()
    # pipeline.mean_trajectories()
    # pipeline.raw_trajectories()
    # pipeline.barplots()
    # pipeline.grouped_clusters()
    # pipeline.barplot_deviation()
    # pipeline.barplot_deviation_GS_social_experience()
    # pipeline.GS_pairwise_social_experience_deviation_stats()
    # pipeline.GS_social_experience_cluster_proportions_per_video()
    # pipeline.hierarchal_mean_trace_summary()
    # pipeline.summary_anchor_partner()
    # pipeline.mean_trace_summary()
    # pipeline.contact_percentage_per_cluster()
    # pipeline.GS_pair_availability_normalised_interactions()
    # pipeline.GS_partner_type_transition_matrix()
    # pipeline.larval_proximity()
    # pipeline.GS_cluster_transition_matrix()
    # pipeline.cluster_transition_matrix_G_vs_S()
    # pipeline.GS_deviations()
    # pipeline.GS_social_experience_over_time_by_cluster()
    # pipeline.G_V_S_duration_transition()
    # pipeline.G_V_S_apetitive_aversive_transition()
    # pipeline.GS_duration_transition()
    # pipeline.GS_apetitive_aversive_transition()
    # pipeline.correlation_contact_G_vs_S()
    # pipeline.GS_directed_movement_over_time()
    # pipeline.social_context_video_proportion()
    # pipeline.social_context_barplot_deviation()
    # pipeline.umap(n_pcs=5)
    # pipeline.spatial_cluster()
    # pipeline.larval_proximity()
    pipeline.grouped_clusters()
