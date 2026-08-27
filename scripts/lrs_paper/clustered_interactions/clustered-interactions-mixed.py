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
from matplotlib.patches import Rectangle
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
from matplotlib.cm import ScalarMappable
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


#### CLASS TO ANALYSE THE INTERACTION CLUSTERS (GH, SI AND MIXED CLUSTERS)
class ClusterPipeline:

    def __init__(self, directory, interactions, clusters, cluster_name, video_path):
        
        self.directory = directory
        self.interaction_path = interactions 
        self.cluster_path = clusters
        self.cluster_name = cluster_name
        self.video_path = video_path

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
    


    #### METHOD SUMMARY_ANCHOR_PARTNER: SUMMARY QUANTIFICATIONS ANCHOR/PARTNER
    def summary_anchor_partner(self):

        df = self.df
        cluster_name = self.cluster_name 

        cluster_ids = sorted(df[cluster_name].unique())
        n_clusters = len(cluster_ids)
        n_rows = 9  # number of summary plots (trajectory, speed, accel, angle, etc.)

        # Create summary canvas
        # fig_ap, axes_ap = plt.subplots(n_rows, n_clusters, figsize=(n_clusters * 4, n_rows * 2))

        # width per column and height per "unit"
        width_per_col  = 1.7
        height_per_unit = 1.65

        # Row 0 gets extra height for shared-scale trajectory plots.
        height_ratios = [2.6] + [1]*(n_rows-1)
        total_units   = sum(height_ratios)
        fig_w = n_clusters * width_per_col
        fig_h = total_units * height_per_unit

        fig_ap, axes_ap = plt.subplots(
        n_rows,
        n_clusters,
        figsize=(fig_w, fig_h),
        gridspec_kw={
            'height_ratios': height_ratios,
            'wspace': 0.08,
            'hspace': 0.18
        }
    )
        fig_ap.subplots_adjust(
            left=0.08,
            right=0.99,
            top=0.95,
            bottom=0.05
        )

        if n_clusters == 1:
            axes_ap = axes_ap.reshape(n_rows, 1)

        # Mark all as invisible initially
        for ax in axes_ap.flatten():
            ax.set_visible(False)

        row_labels = [
            "Mean Trajectory",           # 0
            "Speed Tail",                # 1
            "Acceleration Tail",         # 2
            "Heading Angle",             # 3
            "Heading Angle Change",      # 4
            "Approach Angle",            # 5
            "Approach Angle Change",     # 6
            "Distance Travelled",        # 7
            "Minimum Distance",          # 8
        ]


        for i, label in enumerate(row_labels):
            ax_label = axes_ap[i, 0]  # first column of each row
            ax_label.set_ylabel(label, fontsize=10, rotation=0, labelpad=40, va='center')


        df['anchor_distance'] = df.groupby('interaction_id').apply(
        lambda x: np.sqrt((x['anchor x_body'].diff()**2 + x['anchor y_body'].diff()**2))).reset_index(level=0, drop=True)

        df['partner_distance'] =  df.groupby('interaction_id').apply(
        lambda x: np.sqrt((x['partner x_body'].diff()**2 + x['partner y_body'].diff()**2))).reset_index(level=0, drop=True)

        anchor_base = "#4F7942"
        partner_base = "#916288"

        trace_x_vals = []
        trace_y_vals = []
        for cluster_id in cluster_ids:
            cluster_df = df[df[cluster_name] == cluster_id]
            grouped = cluster_df.groupby("Normalized Frame")

            for role in ("anchor", "partner"):
                x_mean = grouped[f"{role} x_body"].mean()
                y_mean = grouped[f"{role} y_body"].mean()
                x_std = grouped[f"{role} x_body"].std().fillna(0)
                y_std = grouped[f"{role} y_body"].std().fillna(0)

                trace_x_vals.extend((x_mean - x_std).dropna().tolist())
                trace_x_vals.extend((x_mean + x_std).dropna().tolist())
                trace_y_vals.extend((y_mean - y_std).dropna().tolist())
                trace_y_vals.extend((y_mean + y_std).dropna().tolist())

        if trace_x_vals and trace_y_vals:
            trace_x_min, trace_x_max = min(trace_x_vals), max(trace_x_vals)
            trace_y_min, trace_y_max = min(trace_y_vals), max(trace_y_vals)
            trace_x_pad = (trace_x_max - trace_x_min) * 0.05 or 1
            trace_y_pad = (trace_y_max - trace_y_min) * 0.05 or 1
            trace_xlim = (trace_x_min - trace_x_pad, trace_x_max + trace_x_pad)
            trace_ylim = (trace_y_min - trace_y_pad, trace_y_max + trace_y_pad)
        else:
            trace_xlim = None
            trace_ylim = None

        interaction_contact_summary = []
        for cluster_id in cluster_ids:
            cluster_df = df[df[cluster_name] == cluster_id]
            for inter_id in cluster_df["interaction_id"].unique():
                inter = cluster_df[cluster_df["interaction_id"] == inter_id]
                interaction_contact_summary.append({
                    "cluster": cluster_id,
                    "interaction_id": inter_id,
                    "frames_below_1mm": (inter["min_distance"] < 1).sum()
                })

        if interaction_contact_summary:
            mean_contact_frames = (
                pd.DataFrame(interaction_contact_summary)
                .groupby("cluster")["frames_below_1mm"]
                .mean()
            )
        else:
            mean_contact_frames = pd.Series(dtype=float)
        

        for column, cluster_id in enumerate(cluster_ids):
            cluster_df = df[df[cluster_name] == cluster_id].copy()

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


            ax0.plot(t1_x, t1_y, color=anchor_base)
            ax0.plot(t2_x, t2_y, color=partner_base)

            ax0.scatter(t1_x.iloc[0], t1_y.iloc[0], color=anchor_base, marker="o")
            ax0.scatter(t2_x.iloc[0], t2_y.iloc[0], color=partner_base, marker="o")

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
                    fmt="none", ecolor=anchor_base, alpha=0.3
                )
            
            ax0.errorbar(
                    t2_x.values, t2_y.values,
                    xerr=t2_x_std.values, yerr=t2_y_std.values,
                    fmt="none", ecolor=partner_base, alpha=0.3
                )



            # ax0.set_xticks([])
            # ax_sum.set_yticks([])
            if trace_xlim is not None and trace_ylim is not None:
                ax0.set_xlim(*trace_xlim)
                ax0.set_ylim(*trace_ylim)
            ax0.set_aspect('equal', 'box')
            ax0.set_title(f"Cluster {cluster_id}", fontsize=8)
            mean_contact = mean_contact_frames.get(cluster_id, 0.0)
            ax0.text(
                0.97,
                0.03,
                f"{mean_contact:.1f}s Contact",
                transform=ax0.transAxes,
                ha="right",
                va="bottom",
                fontsize=10,
                color="#4a4a4a"
            )
            ax0.axis("off")
            ax0.set_visible(True)



            ## 1. SPEED TAIL
            ax1 = axes_ap[1, column]

            sns.lineplot(data=cluster_df, x='Normalized Frame', y='anchor_speed_tail', errorbar=('ci', 95), color=anchor_base, legend=False, ax=ax1)
            sns.lineplot(data=cluster_df, x='Normalized Frame', y='partner_speed_tail', errorbar=('ci', 95), color=partner_base, legend=False, ax=ax1)

            ax1.axvline(0, color="gray", ls="--", lw=0.5)
            ax1.set_ylim(0, 2)
            ax1.set_xticks([])
            # ax1.set_yticks([])
            ax1.set_visible(True)

            ## 2. ACCELERATION TAIL
            ax2 = axes_ap[2, column]

            sns.lineplot(data=cluster_df, x='Normalized Frame', y='anchor_acceleration_tail', errorbar=('ci', 95), color=anchor_base, legend=False, ax=ax2)
            sns.lineplot(data=cluster_df, x='Normalized Frame', y='partner_acceleration_tail', errorbar=('ci', 95), color=partner_base, legend=False, ax=ax2)

            ax2.axvline(0, color="gray", ls="--", lw=0.5)
            ax2.set_ylim(-0.5, 0.5)
            ax2.set_xticks([])
            # ax1.set_yticks([])
            ax2.set_visible(True)

            ## 3. HEADING ANGLE
            ax3 = axes_ap[3, column]

            sns.lineplot(data=cluster_df, x='Normalized Frame', y='anchor_angle', errorbar=('ci', 95), color=anchor_base, legend=False, ax=ax3)
            sns.lineplot(data=cluster_df, x='Normalized Frame', y='partner_angle', errorbar=('ci', 95), color=partner_base, legend=False, ax=ax3)

            ax3.axvline(0, color="gray", ls="--", lw=0.5)
            ax3.set_ylim(0, 180)
            ax3.set_xticks([])
            # ax1.set_yticks([])
            ax3.set_visible(True)

            ## 4. HEADING ANGLE CHANGE
            ax4 = axes_ap[4, column]

            sns.lineplot(data=cluster_df, x='Normalized Frame', y='anchor_heading_angle_change', errorbar=('ci', 95), color=anchor_base, legend=False, ax=ax4)
            sns.lineplot(data=cluster_df, x='Normalized Frame', y='partner_heading_angle_change', errorbar=('ci', 95), color=partner_base, legend=False, ax=ax4)

            ax4.axvline(0, color="gray", ls="--", lw=0.5)
            ax4.set_ylim(0, 60)
            ax4.set_xticks([])
            ax4.set_visible(True)

            ## 4. APPROACH ANGLE
            ax5 = axes_ap[5, column]

            sns.lineplot(data=cluster_df, x='Normalized Frame', y='anchor_approach_angle', errorbar=('ci', 95), color=anchor_base, legend=False, ax=ax5)
            sns.lineplot(data=cluster_df, x='Normalized Frame', y='partner_approach_angle', errorbar=('ci', 95), color=partner_base, legend=False, ax=ax5)

            ax5.axvline(0, color="gray", ls="--", lw=0.5)
            ax5.set_ylim(0, 180)
            ax5.set_xticks([])
            # ax1.set_yticks([])
            ax5.set_visible(True)

            ## 6. APPROACH ANGLE CHANGE
            ax6 = axes_ap[6, column]

            sns.lineplot(data=cluster_df, x='Normalized Frame', y='anchor_approach_angle_change', errorbar=('ci', 95), color=anchor_base, legend=False, ax=ax6)
            sns.lineplot(data=cluster_df, x='Normalized Frame', y='partner_approach_angle_change', errorbar=('ci', 95), color=partner_base, legend=False, ax=ax6)

            ax6.axvline(0, color="gray", ls="--", lw=0.5)
            ax6.set_ylim(0, 60)
            ax6.set_xticks([])
            ax6.set_visible(True)

            ## 7. DISTANCE TRAVELLED
            ax7 = axes_ap[7, column]

            sns.lineplot(data=cluster_df, x='Normalized Frame', y='anchor_distance', errorbar=('ci', 95), color=anchor_base, legend=False, ax=ax7)
            sns.lineplot(data=cluster_df, x='Normalized Frame', y='partner_distance', errorbar=('ci', 95), color=partner_base, legend=False, ax=ax7)

            ax7.axvline(0, color="gray", ls="--", lw=0.5)
            ax7.set_ylim(0, 14)
            ax7.set_xticks([])
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
            ax8.axvline(0, color='gray', linestyle='--', linewidth=0.5)
            ax8.set_ylim(0, 25)
            ax8.set_xticks([-10, 0, 10])
            ax8.set_visible(True)

            ax8.text(0.55, 0.92, f"pre slope:  {slope_pre:.2f}",  transform=ax8.transAxes, fontsize=8, clip_on=False, zorder=10)
            ax8.text(0.55, 0.76, f"post slope: {slope_post:.2f}", transform=ax8.transAxes, fontsize=8, clip_on=False, zorder=10)

            for row_idx in range(n_rows):
                ax = axes_ap[row_idx, column]
                if not ax.get_visible():
                    continue
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.set_xlabel("")
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()

                if row_idx == 0:
                    ax.set_ylabel("")
                    ax.tick_params(
                        axis="both",
                        left=False,
                        labelleft=False,
                        bottom=False,
                        labelbottom=False
                    )
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                elif column == 0:
                    ax.set_ylabel(row_labels[row_idx], fontsize=10, rotation=0, labelpad=40, va='center')
                    ax.tick_params(axis='y', left=True, labelleft=True)
                else:
                    ax.set_ylabel("")
                    ax.spines["left"].set_visible(False)
                    ax.tick_params(axis='y', left=False, labelleft=False)

                if 0 < row_idx < n_rows - 1:
                    ax.tick_params(axis='x', bottom=False, labelbottom=False)



        legend_handles = [
            Line2D([0], [0], color=anchor_base, lw=2, label="Anchor"),
            Line2D([0], [0], color=partner_base, lw=2, label="Partner")
        ]
        fig_ap.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 1.01)
        )

        out_path = os.path.join(self.directory, "summary_anchor_partner.pdf")
        plt.savefig(out_path, format="pdf", bbox_inches='tight')
        plt.close(fig_ap)
    


    #### METHOD SUMMARY_CLUSTERS: INDIVIDUAL CLUSTER METRIC SUMMARIES
    def summary_clusters(self):

        df = self.df.copy()
        cluster_name = self.cluster_name

        output = os.path.join(self.directory, "cluster_summary")
        os.makedirs(output, exist_ok=True)

        cluster_order = sorted(
            df[cluster_name].dropna().unique(),
            key=lambda value: float(value)
        )
        cluster_labels = [str(int(float(cluster))) for cluster in cluster_order]
        cluster_palette = dict(zip(
            cluster_labels,
            sns.color_palette("viridis", n_colors=len(cluster_labels))
        ))

        df["cluster_summary_label"] = df[cluster_name].apply(
            lambda cluster: str(int(float(cluster))) if pd.notna(cluster) else np.nan
        )
        df["approach_angle_partner_to_anchor"] = df["partner_approach_angle"]
        df["approach_angle_anchor_to_partner"] = df["anchor_approach_angle"]
        df["anchor_distance"] = (
            df.groupby("interaction_id")
            .apply(lambda group: np.sqrt(
                group["anchor x_body"].diff()**2 + group["anchor y_body"].diff()**2
            ))
            .reset_index(level=0, drop=True)
        )
        df["partner_distance"] = (
            df.groupby("interaction_id")
            .apply(lambda group: np.sqrt(
                group["partner x_body"].diff()**2 + group["partner y_body"].diff()**2
            ))
            .reset_index(level=0, drop=True)
        )
        heading_change_ylim = (0, 30)

        metrics = [
            ("partner_speed_tail", "Tail Speed Partner", "tail_speed_partner"),
            ("anchor_speed_tail", "Tail Speed Anchor", "tail_speed_anchor"),
            ("partner_acceleration_tail", "Tail Acceleration Partner", "tail_acceleration_partner"),
            ("anchor_acceleration_tail", "Tail Acceleration Anchor", "tail_acceleration_anchor"),
            ("partner_heading_angle_change", "Heading Angle Change Partner", "heading_angle_change_partner"),
            ("anchor_heading_angle_change", "Heading Angle Change Anchor", "heading_angle_change_anchor"),
            ("approach_angle_partner_to_anchor", "Approach Angle Partner to Anchor", "approach_angle_partner_to_anchor"),
            ("approach_angle_anchor_to_partner", "Approach Angle Anchor to Partner", "approach_angle_anchor_to_partner"),
            ("anchor_distance", "Distance Travelled Anchor", "distance_travelled_anchor"),
            ("partner_distance", "Distance Travelled Partner", "distance_travelled_partner"),
            ("min_distance", "Minimum Distance", "minimum_distance"),
        ]

        for metric, title, filename in metrics:
            if metric not in df.columns:
                print(f"Skipping {title}: missing column {metric}")
                continue

            plot_df = (
                df[["Normalized Frame", "cluster_summary_label", metric]]
                .dropna(subset=["Normalized Frame", "cluster_summary_label", metric])
                .groupby(["cluster_summary_label", "Normalized Frame"], as_index=False)[metric]
                .mean()
            )

            fig, ax = plt.subplots(figsize=(7, 5))

            for idx, cluster_label in enumerate(cluster_labels):
                cluster_plot_df = plot_df[
                    plot_df["cluster_summary_label"] == cluster_label
                ].sort_values("Normalized Frame")

                ax.plot(
                    cluster_plot_df["Normalized Frame"],
                    cluster_plot_df[metric],
                    color=cluster_palette[cluster_label],
                    linestyle="-" if idx % 2 == 0 else "--",
                    linewidth=1.6,
                    label=cluster_label
                )

            ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.8)
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_xlabel("Normalized Frame")
            ax.set_ylabel(title)
            ax.set_xticks([-10, 0, 10])
            if metric in ["approach_angle_partner_to_anchor", "approach_angle_anchor_to_partner"]:
                ax.set_ylim(0, 180)
            elif metric in ["partner_heading_angle_change", "anchor_heading_angle_change"]:
                ax.set_ylim(*heading_change_ylim)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.legend(
                title="Cluster",
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False
            )

            fig.savefig(
                os.path.join(output, f"{filename}.pdf"),
                format="pdf",
                bbox_inches="tight",
                transparent=True
            )
            plt.close(fig)

        contact_df = (
            df[["interaction_id", "cluster_summary_label", "min_distance"]]
            .dropna(subset=["interaction_id", "cluster_summary_label", "min_distance"])
            .groupby(["cluster_summary_label", "interaction_id"], as_index=False)
            .agg(made_contact=("min_distance", lambda values: (values < 1).any()))
        )

        contact_summary = (
            contact_df
            .groupby("cluster_summary_label", as_index=False)["made_contact"]
            .mean()
        )
        contact_summary["percentage_contact"] = contact_summary["made_contact"] * 100
        contact_summary = (
            contact_summary
            .set_index("cluster_summary_label")
            .reindex(cluster_labels)
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(
            contact_summary["cluster_summary_label"],
            contact_summary["percentage_contact"],
            color=[cluster_palette[label] for label in contact_summary["cluster_summary_label"]],
            edgecolor="black",
            linewidth=0.8
        )

        ax.set_title("Percentage Contact", fontsize=14, fontweight="bold")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("% interactions with contact")
        ax.set_ylim(0, 100)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.savefig(
            os.path.join(output, "perecentage_contact.pdf"),
            format="pdf",
            bbox_inches="tight",
            transparent=True
        )
        plt.close(fig)

        if "interaction_type" in df.columns:
            interaction_merge_map = {
                "head-head": "head-head",
                "head-tail": "head-tail",
                "tail-head": "head-tail",
                "head-body": "head-body",
                "body-head": "head-body",
                "tail-tail": "other",
                "tail-body": "other",
                "body-tail": "other",
                "body-body": "other",
            }
            contact_order = ["head-head", "head-tail", "head-body", "other"]
            contact_palette = {
                "head-head": "red",
                "head-tail": "blue",
                "head-body": "green",
                "other": "lightgray",
            }

            first_contact_rows = []
            for (cluster_label, inter_id), inter_df in df.groupby(
                ["cluster_summary_label", "interaction_id"]
            ):
                if pd.isna(cluster_label):
                    continue

                close = inter_df[inter_df["min_distance"] < 1].sort_values("Frame")
                if close.empty:
                    continue

                first_contact_type = close["interaction_type"].iloc[0]
                first_contact_rows.append({
                    "cluster_summary_label": cluster_label,
                    "interaction_id": inter_id,
                    "contact_type": interaction_merge_map.get(first_contact_type, "other")
                })

            first_contact_df = pd.DataFrame(first_contact_rows)

            if not first_contact_df.empty:
                contact_props = (
                    first_contact_df
                    .groupby(["cluster_summary_label", "contact_type"])
                    .size()
                    .unstack(fill_value=0)
                    .reindex(index=cluster_labels, columns=contact_order, fill_value=0)
                )
                contact_props = contact_props.div(contact_props.sum(axis=1), axis=0).fillna(0)

                fig, ax = plt.subplots(figsize=(7, 5))
                bottoms = np.zeros(len(contact_props))
                x = np.arange(len(contact_props))

                for contact_type in contact_order:
                    values = contact_props[contact_type].to_numpy()
                    ax.bar(
                        x,
                        values,
                        bottom=bottoms,
                        color=contact_palette[contact_type],
                        edgecolor="white",
                        linewidth=0.8,
                        label=contact_type
                    )
                    bottoms += values

                ax.set_title("Contact Nodes", fontsize=14, fontweight="bold")
                ax.set_xlabel("Cluster")
                ax.set_ylabel("Proportion at first contact frame")
                ax.set_xticks(x)
                ax.set_xticklabels(cluster_labels)
                ax.set_ylim(0, 1)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.legend(
                    title="Contact type",
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    frameon=False
                )

                fig.savefig(
                    os.path.join(output, "contact_nodes.pdf"),
                    format="pdf",
                    bbox_inches="tight",
                    transparent=True
                )
                plt.close(fig)
        else:
            print("Skipping Contact Nodes: missing column interaction_type")




    #### METHOD HIERARCHAL_MEAN_TRACE_SUMMARY: MEAN TRAJECTORIES WITH HIERARCHAL TREE
    def hierarchal_mean_trace_summary(self):
        
        df = self.df.copy()
        cluster_name = self.cluster_name
        clusters = sorted(df[cluster_name].unique())
        n_clusters = len(clusters)


        fig = plt.figure(figsize=(18, 4))
        gs = gridspec.GridSpec(3, 1, height_ratios=[0.3, 0.6, 0.05], hspace=0.3)

        ax1 = fig.add_subplot(gs[0])  # top: hierarchal tree
        # ax2 = fig.add_subplot(gs[1])
        sub_ax2 = gs[1].subgridspec(1, n_clusters, wspace=0.1)
        axes_ax2 = [fig.add_subplot(sub_ax2[0, i]) for i in range(n_clusters)]

        #ax3 = fig.add_subplot(gs[2])  # bottom: average contact frames
        sub_ax3 = gs[2].subgridspec(1, n_clusters, wspace=0.1)
        ax3 = fig.add_subplot(sub_ax3[0, :])  # one long axis spanning all 12 columns


        ## PLOTTING HIERARCHAL TREE - MANUALLY MATCHING THE CLUSTER STRUCTURE
        cluster_positions = {
            int(cluster): idx + 1
            for idx, cluster in enumerate(sorted(df[cluster_name].dropna().unique()))
        }

        def draw_branch(left, right, height):
            left_node = left if isinstance(left, tuple) else (cluster_positions.get(left), 0)
            right_node = right if isinstance(right, tuple) else (cluster_positions.get(right), 0)

            if left_node[0] is None:
                return right_node
            if right_node[0] is None:
                return left_node

            ax1.plot([left_node[0], left_node[0]], [left_node[1], height], color='black')
            ax1.plot([right_node[0], right_node[0]], [right_node[1], height], color='black')
            ax1.plot([left_node[0], right_node[0]], [height, height], color='black')
            return ((left_node[0] + right_node[0]) / 2, height)

        branch_10_12 = draw_branch(10, 12, 1)
        branch_10_12_11 = draw_branch(11, branch_10_12, 2)
        branch_9_12 = draw_branch(9, branch_10_12_11, 3)
        branch_8_12 = draw_branch(8, branch_9_12, 4)

        branch_1_2 = draw_branch(1, 2, 1)
        branch_1_4_height = 3
        ax1.plot([branch_1_2[0], branch_1_2[0]], [branch_1_2[1], branch_1_4_height], color='black')
        ax1.plot([cluster_positions[3], cluster_positions[3]], [0, branch_1_4_height], color='black')
        ax1.plot([cluster_positions[4], cluster_positions[4]], [0, branch_1_4_height], color='black')
        ax1.plot([branch_1_2[0], cluster_positions[4]], [branch_1_4_height, branch_1_4_height], color='black')
        branch_1_4 = ((branch_1_2[0] + cluster_positions[3] + cluster_positions[4]) / 3, branch_1_4_height)

        branch_5_7 = draw_branch(5, 7, 1)
        branch_5_7_6 = draw_branch(branch_5_7, 6, 2)

        branch_1_7 = draw_branch(branch_1_4, branch_5_7_6, 4)
        draw_branch(branch_1_7, branch_8_12, 5)
        ax1.set_xlim(0.5, n_clusters + 0.5)
        ax1.set_ylim(-0.1, 5.2)

    

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
        order = [int(cluster) for cluster in clusters]   # matches your dendrogram order
        mean_frames = mean_frames.reindex(order, fill_value=0)


        # colors = ["skyblue", "mediumseagreen", "darkgreen"]
        colors = ["aliceblue", "#56B19C", "darkgreen"]
        # build a sequential colormap from those
        my_cmap = LinearSegmentedColormap.from_list("greenblue_custom", colors)

        vals = mean_frames.to_numpy()
        vmax = float(np.nanmax(vals)) if np.nanmax(vals) > 0 else 1.0
        norm = Normalize(vmin=0, vmax=vmax)

        # 5) draw the heat “box” as vector rectangles so Illustrator preserves colours
        ax3.set_xlim(0, len(vals))
        ax3.set_ylim(0, 1)
        for i, v in enumerate(vals):
            ax3.add_patch(
                Rectangle((i, 0), 1, 1,
                          facecolor=my_cmap(norm(v)),
                          edgecolor='none')
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
        sm = ScalarMappable(norm=norm, cmap=my_cmap)
        sm.set_array([])

        cbar = fig.colorbar(
            sm,
            ax=[ax1, *axes_ax2, ax3],        # anchors to both axes (so it aligns with total figure height)
            fraction=0.01,         # width of colorbar relative to figure
            pad=0.01,              # horizontal gap between plots and colorbar
            location='right'       # move to right side
        )
        cbar.set_label('Average Contact Frames', rotation=270, labelpad=15)

        output = os.path.join(self.directory, "hierarchal_mean_trace_summary.pdf")
        plt.savefig(output, format='pdf', bbox_inches='tight')
        plt.close(fig)
    
    
    #### METHOD BARPLOT_DEVIATION: BARPLOT DEVIATION OF OBSERVED V EXPECTED FOR G V S  
    def barplot_deviation(self):

        df = self.clusters
        df_interaction = self.df
        cluster_name = self.cluster_name

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
        cluster_one = next((cid for cid in deviation_sorted.index if str(cid) == '1'), None)
        if cluster_one is not None:
            display_order = [cid for cid in deviation_sorted.index if cid != cluster_one]
            display_order.insert(1, cluster_one)
            deviation_sorted = deviation_sorted.reindex(display_order)
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
        path = os.path.join(self.directory, 'deviation_pvals.csv')
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

        ax2.set_xlim(ax1.get_xlim())

        # define your color stops (dark → sea → cadet)
        # colors = ["lightskyblue", "mediumseagreen", "darkgreen"]
        colors = ["aliceblue", "#56B19C", "darkgreen"]
        # build a sequential colormap from those
        my_cmap = LinearSegmentedColormap.from_list("greenblue_custom", colors)

        vals = avg_contact_aligned.to_numpy()
        vmax = float(np.nanmax(vals)) if np.nanmax(vals) > 0 else 1.0
        norm = Normalize(vmin=0, vmax=vmax)

        # 5) draw the heat “box” as vector rectangles so Illustrator preserves colours
        ax2.set_xlim(-0.5, len(vals) - 0.5)
        ax2.set_ylim(-0.5, 0.5)
        for i, v in enumerate(vals):
            ax2.add_patch(
                Rectangle((i - 0.5, -0.5), 1, 1,
                          facecolor=my_cmap(norm(v)),
                          edgecolor='none')
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
        sm = ScalarMappable(norm=norm, cmap=my_cmap)
        sm.set_array([])

        cbar = fig.colorbar(
            sm,
            ax=[ax1, ax2],        # anchors to both axes (so it aligns with total figure height)
            fraction=0.03,         # width of colorbar relative to figure
            pad=0.02,              # horizontal gap between plots and colorbar
            location='right'       # move to right side
        )
        cbar.set_label('Average Contact Frames', rotation=270, labelpad=15)

        path = os.path.join(self.directory, 'deviations.pdf')  
        plt.savefig(path, format="pdf", bbox_inches="tight", dpi=300, transparent=True)
        plt.close()
    



    #### METHOD BARPLOTS_PROPORTIONS: BARPLOT PROPORTION OF CLUSTER BY CONDITION
    def barplot_proportion(self):

        df = self.clusters
        full_df = self.df
        cluster_name = self.cluster_name

        palette = {
            "G": "C0",
            "S": "C1",
            "GS": "#d26070"
        }

        # palette = {
        #     "G": "C0",
        #     "S": "C1",
        #     "GS": "#4a8763"
        # }

        condition_order = ["G", "GS", "S"]


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


        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=summary_df,
            x=cluster_name,
            y='proportion',
            hue='condition',
            hue_order=condition_order,
            errorbar='sd',

            palette=palette
        )

        plt.title("Proportion of Clusters per Video")
        plt.xlabel("Cluster ID")
        plt.ylabel("Proportion")
        plt.xticks(rotation=90)
        plt.tight_layout()

        output = os.path.join(self.directory, 'cluster_proportions.pdf')
        plt.savefig(output, format='pdf', bbox_inches='tight')
        plt.close()
    



    def correlations_G_vs_S(self):

        df = self.clusters.copy()
        df_interaction = self.df.copy()
        cluster_name = self.cluster_name


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

        # pvals.to_csv(
        #     os.path.join(self.directory, "G_vs_S_deviation_pvals.csv"),
        #     index=False
        # )

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

        # correlation_df.to_csv(
        #     os.path.join(self.directory, "G_vs_S_deviation_contact_correlation.csv"),
        #     index=False
        # )

        # Linear fit only for visual guide
        x = correlation_df["deviation_G_bias"]
        y = correlation_df["avg_contact_frames"]

        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = intercept + slope * x_line


        fig, ax = plt.subplots(figsize=(4, 4))

        sns.scatterplot(
            data=correlation_df,
            x="deviation_G_bias",
            y="avg_contact_frames",
            s=60,
            color="mediumseagreen",
            # edgecolor="gray",
            ax=ax
        )

        ax.plot(x_line, y_line, color="darkgray", linewidth=2)

        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)

        ax.set_xlabel("Deviation from expected G fraction")
        ax.set_ylabel("Average contact frames")
        ax.set_title(f"Spearman rho = {rho:.2f}, p = {p:.3g}")
        ax.set_xlim(-0.3, 0.3)

        sns.despine()
        plt.tight_layout()


        plt.savefig(
            os.path.join(self.directory, "correlation_plot.pdf"),
            format="pdf",
            bbox_inches="tight"
        )

        plt.close()
    


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
        umap_df["umap_aversive_appetitive"] = umap_df[cluster_name].apply(
            get_appetitive_aversive_label
        )
        cluster_order = sorted(
            umap_df[cluster_name].dropna().unique(),
            key=lambda value: float(value)
        )
        cluster_labels = [str(int(float(cluster))) for cluster in cluster_order]
        # cluster_cmap = mpl.colormaps["YlGn"]
        # cluster_colors = [
        #     cluster_cmap(value)
        #     for value in np.linspace(0.2, 1.0, len(cluster_labels))
        # ]
        cluster_palette = dict(zip(
            cluster_labels,
            sns.color_palette("viridis", n_colors=len(cluster_labels))
        ))
        umap_df["umap_cluster_label"] = umap_df[cluster_name].apply(
            lambda cluster: str(int(float(cluster))) if pd.notna(cluster) else np.nan
        )

        x_pad = (umap_df["UMAP1"].max() - umap_df["UMAP1"].min()) * 0.05
        y_pad = (umap_df["UMAP2"].max() - umap_df["UMAP2"].min()) * 0.05
        x_lim = (umap_df["UMAP1"].min() - x_pad, umap_df["UMAP1"].max() + x_pad)
        y_lim = (umap_df["UMAP2"].min() - y_pad, umap_df["UMAP2"].max() + y_pad)

        def setup_umap_axis():
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.set_position([0.10, 0.12, 0.62, 0.78])
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            return fig, ax

        def move_legend_outside(ax):
            sns.move_legend(
                ax,
                "center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0,
                frameon=False
            )
            legend = ax.get_legend()
            legend_handles = getattr(legend, "legend_handles", None)
            if legend_handles is None:
                legend_handles = legend.legendHandles
            for handle in legend_handles:
                if hasattr(handle, "set_sizes"):
                    handle.set_sizes([45])
                elif hasattr(handle, "set_markersize"):
                    handle.set_markersize(7)

        fig, ax = setup_umap_axis()

        sns.scatterplot(
            data=umap_df,
            x="UMAP1",
            y="UMAP2",
            hue="umap_cluster_label",
            hue_order=cluster_labels,
            s=8,
            alpha=0.8,
            linewidth=0,
            palette=cluster_palette,
            ax=ax
        )

        ax.set_title(f"UMAP ({n_pcs} PCs) coloured by cluster")
        move_legend_outside(ax)

        fig.savefig(os.path.join(output, "umap_cluster.pdf"),
  
                    format="pdf",
                    transparent=True)

        plt.close()

        fig, ax = setup_umap_axis()

        sns.scatterplot(
            data=umap_df,
            x="UMAP1",
            y="UMAP2",
            hue="condition",
            hue_order=["S", "GS", "G"],
            palette={
                "S": "C1",
                "GS": "#d26070",
                "G": "C0"
            },
            s=8,
            alpha=1.0,
            linewidth=0,
            ax=ax
        )

        ax.set_title(f"UMAP ({n_pcs} PCs) coloured by condition")
        move_legend_outside(ax)

        fig.savefig(os.path.join(output, "umap_condition.pdf"),
                    format="pdf",
                    transparent=True)

        plt.close()


        gs_umap_df = umap_df[
            (umap_df["condition"] == "GS") &
            (umap_df["gs_pair_type"].notna())
        ].copy()

        fig, ax = setup_umap_axis()

        sns.scatterplot(
            data=gs_umap_df,
            x="UMAP1",
            y="UMAP2",
            hue="gs_pair_type",
            hue_order=["S-S", "G-S", "G-G"],
            palette={
                "S-S": "C1",
                "G-S": "#d26070",
                "G-G": "C0"
            },
            s=8,
            alpha=1.0,
            linewidth=0,
            ax=ax
        )

        ax.set_title(f"UMAP ({n_pcs} PCs) within GS coloured by interaction type")
        move_legend_outside(ax)

        fig.savefig(os.path.join(output, "umap_social_context.pdf"),
                    format="pdf",
                    transparent=True)

        plt.close()


        fig, ax = setup_umap_axis()

        sns.scatterplot(
            data=umap_df.dropna(subset=["umap_aversive_appetitive"]),
            x="UMAP1",
            y="UMAP2",
            hue="umap_aversive_appetitive",
            hue_order=["appetitive", "aversive"],
            palette={
                "appetitive": "#0047AB",
                "aversive": "#D22B2B"
            },
            s=8,
            alpha=0.8,
            linewidth=0,
            ax=ax
        )

        ax.set_title(f"UMAP ({n_pcs} PCs) coloured by appetitive/aversive")
        move_legend_outside(ax)

        fig.savefig(os.path.join(output, "umap_aversive_apeititve.pdf"),
                    format="pdf",
                    transparent=True)

        plt.close()
    



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

                sub = video_probs[
                    (video_probs["from_dur"] == from_dur) &
                    (video_probs["to_dur"] == to_dur)
                ]

                g = sub[sub["condition"] == "G"]["probability"]
                s = sub[sub["condition"] == "S"]["probability"]
                diff = g.mean() - s.mean()

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
                    "test": "mannwhitneyu",
                    "U": stat,
                    "p": p,
                    "G_mean": g.mean(),
                    "S_mean": s.mean(),
                    "mean_diff_G_minus_S": diff,
                    "n_G_videos": len(g),
                    "n_S_videos": len(s)
                })

        stats = pd.DataFrame(stats)
        stats["p_adj"] = np.nan

        mask = stats["p"].notna()

        if mask.any():
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

        cmap = LinearSegmentedColormap.from_list(
            "si_to_gh",
            ["C1", "white", "C0"]
        )
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

        Gd = diff_matrix_to_digraph_labels(D, thresh=0.05)

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax.set_title("Duration transition likelihood: G - S")
        ax.axis("off")

        nx.draw_networkx_nodes(Gd, pos, ax=ax, node_size=1200)
        nx.draw_networkx_labels(Gd, pos, ax=ax, font_size=10)

        for u, v, dct in Gd.edges(data=True):
            w = float(dct["weight"])
            color = 'C1' if w < 0 else 'C0'
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
            os.path.join(output, "duration_transition_difference_circlegraph.pdf"),
            format="pdf",
            bbox_inches="tight"
        )
        plt.close()

    

    def G_V_S_cluster_transition(self):

        df = self.df.copy()
        cluster_name = self.cluster_name

        output = os.path.join(self.directory, "G_V_S_cluster_transition")
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

        stats = []

        for cluster in clusters:
            for next_cluster in clusters:

                sub = counts[
                    (counts["cluster"] == cluster) &
                    (counts["next_cluster"] == next_cluster)
                ]

                g = sub[sub["condition"] == "G"]["transition_probability"]
                s = sub[sub["condition"] == "S"]["transition_probability"]
                diff = g.mean() - s.mean()

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
                    "cluster": cluster,
                    "next_cluster": next_cluster,
                    "test": "mannwhitneyu",
                    "U": stat,
                    "p": p,
                    "G_mean": g.mean(),
                    "S_mean": s.mean(),
                    "mean_diff_G_minus_S": diff,
                    "n_G_videos": len(g),
                    "n_S_videos": len(s)
                })

        stats = pd.DataFrame(stats)
        transition_difference_filter = 0.05
        stats["max_mean_probability"] = stats[["G_mean", "S_mean"]].max(axis=1)
        stats["abs_mean_diff_G_minus_S"] = stats["mean_diff_G_minus_S"].abs()
        stats["tested_in_filtered_family"] = (
            stats["abs_mean_diff_G_minus_S"] >= transition_difference_filter
        )
        stats["p_adj"] = np.nan

        mask = stats["p"].notna()

        if mask.any():
            stats.loc[mask, "p_adj"] = multipletests(
                stats.loc[mask, "p"],
                method="fdr_bh"
            )[1]

        stats.to_csv(
            os.path.join(output, "G_vs_S_cluster_transition_statistics.csv"),
            index=False
        )

        filtered_stats = stats[stats["tested_in_filtered_family"]].copy()
        filtered_stats["p_adj_filtered"] = np.nan

        filtered_mask = filtered_stats["p"].notna()

        if filtered_mask.any():
            filtered_stats.loc[filtered_mask, "p_adj_filtered"] = multipletests(
                filtered_stats.loc[filtered_mask, "p"],
                method="fdr_bh"
            )[1]

        filtered_stats.to_csv(
            os.path.join(output, "G_vs_S_cluster_transition_statistics_filtered.csv"),
            index=False
        )

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
        # Circle graph: G - S transition probability difference
        # -------------------------------------------------------

        M_S = matrices["S"].copy()
        M_G = matrices["G"].copy()

        all_clusters = sorted(
            set(M_S.index) | set(M_S.columns) |
            set(M_G.index) | set(M_G.columns)
        )

        M_S = M_S.reindex(index=all_clusters, columns=all_clusters, fill_value=0)
        M_G = M_G.reindex(index=all_clusters, columns=all_clusters, fill_value=0)

        D = M_G - M_S
        D.to_csv(os.path.join(output, "G_vs_S_cluster_transition_difference_G_minus_S.csv"))

        lim = float(np.nanmax(np.abs(D.to_numpy())))
        if lim == 0:
            lim = 1e-6

        pos = nx.circular_layout(all_clusters)

        cmap = LinearSegmentedColormap.from_list(
            "si_to_gh_cluster",
            ["C1", "white", "C0"]
        )
        norm = mpl.colors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)

        def diff_matrix_to_digraph_labels(D, thresh=0.02, p_adj_lookup=None, alpha=0.05, allowed_edges=None):
            G = nx.DiGraph()

            for c in D.index:
                G.add_node(int(c))

            for i in D.index:
                for j in D.columns:
                    w = float(D.loc[i, j])
                    edge = (int(i), int(j))
                    is_allowed = allowed_edges is None or edge in allowed_edges
                    is_significant = True
                    if p_adj_lookup is not None:
                        p_adj = p_adj_lookup.get(edge, np.nan)
                        is_significant = pd.notna(p_adj) and p_adj < alpha

                    if abs(w) >= thresh and is_allowed and is_significant:
                        G.add_edge(edge[0], edge[1], weight=w)

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

            lw = 0.6 + (abs(w) / lim) * 7.4
            alpha = 0.2 + (abs(w) / lim) * 0.75
            color = 'C1' if w < 0 else 'C0'

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

            lw = 0.6 + (abs(w) / lim) * 7.4
            alpha = 0.2 + (abs(w) / lim) * 0.75
            color = 'C1' if w < 0 else 'C0'

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

            edges = sorted(
                G.edges(data=True),
                key=lambda edge: float(edge[2]["weight"]) > 0
            )

            for u, v, d in edges:

                w = float(d["weight"])

                if u == v:
                    draw_self_loop(ax, pos[u], w)
                else:
                    draw_edge(ax, pos[u], pos[v], w)

        Gd = diff_matrix_to_digraph_labels(
            D,
            thresh=transition_difference_filter
        )

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

        draw_transition_circle(ax, Gd, "Cluster transition differences: G - S (abs diff >= 0.05)")

        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cax = fig.add_axes([0.95, 0.28, 0.02, 0.45])

        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("P(G) - P(S)")

        plt.savefig(
            os.path.join(output, "G_vs_S_cluster_transition_circlegraphs.pdf"),
            format="pdf",

            bbox_inches="tight"
        )
        plt.close()

        p_adj_lookup = (
            stats
            .set_index(["cluster", "next_cluster"])["p_adj"]
            .to_dict()
        )
        Gd_stat = diff_matrix_to_digraph_labels(
            D,
            thresh=0.05,
            p_adj_lookup=p_adj_lookup,
            alpha=0.05
        )

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

        draw_transition_circle(ax, Gd_stat, "Cluster transition likelihood: G - S (p_adj < 0.05)")

        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cax = fig.add_axes([0.95, 0.28, 0.02, 0.45])

        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("P(G) - P(S)")

        plt.savefig(
            os.path.join(output, "G_vs_S_cluster_transition_circlegraphs_stat.pdf"),
            format="pdf",
            bbox_inches="tight"
        )
        plt.close()

        p_adj_filtered_lookup = (
            filtered_stats
            .set_index(["cluster", "next_cluster"])["p_adj_filtered"]
            .to_dict()
        )
        Gd_stat_filtered = diff_matrix_to_digraph_labels(
            D,
            thresh=transition_difference_filter,
            p_adj_lookup=p_adj_filtered_lookup,
            alpha=0.05
        )

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

        draw_transition_circle(
            ax,
            Gd_stat_filtered,
            "Cluster transition differences: G - S (abs diff >= 0.05, filtered p_adj < 0.05)"
        )

        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cax = fig.add_axes([0.95, 0.28, 0.02, 0.45])

        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("P(G) - P(S)")

        plt.savefig(
            os.path.join(output, "G_vs_S_cluster_transition_circlegraphs_stat_filtered.pdf"),
            format="pdf",
            bbox_inches="tight"
        )
        plt.close()
    

    def GS_deviation(self):

        df = self.clusters.copy()
        df_interaction = self.df.copy()
        cluster_name = self.cluster_name

        output = os.path.join(self.directory, "GS_deviation")
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
    


    def GS_barplot_proportion(self):

        df = self.clusters.copy()
        df_interaction = self.df.copy()
        cluster_name = self.cluster_name

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

        social_order = ['G-G', 'G-S', 'S-S', ]

        palette = {
            'S-S': 'C1',
            'G-S': '#d26070',
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


        path = os.path.join(self.directory, 'GS_proportions.pdf')
        plt.savefig(path, format='pdf', bbox_inches='tight')

        plt.close()
    


    def GS_cluster_transition(self):

        df = self.df.copy()
        cluster_name = self.cluster_name

        output = os.path.join(self.directory, "GS_cluster_transition")
        os.makedirs(output, exist_ok=True)
        g_vs_s_output = os.path.join(output, "G_V_S")
        within_type_output = os.path.join(output, "within_type")
        mixed_output = os.path.join(output, "mixed")
        mixed_2_output = os.path.join(output, "mixed_2")
        os.makedirs(g_vs_s_output, exist_ok=True)
        os.makedirs(within_type_output, exist_ok=True)
        os.makedirs(mixed_output, exist_ok=True)
        os.makedirs(mixed_2_output, exist_ok=True)

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

        gs_stats = []

        for cluster_id in clusters:
            for next_cluster_id in clusters:

                sub = counts[
                    (counts["cluster"] == cluster_id) &
                    (counts["next_cluster"] == next_cluster_id)
                ]

                wide = (
                    sub.pivot(index="file", columns="larva_type", values="transition_probability")
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

                gs_stats.append({
                    "cluster": cluster_id,
                    "next_cluster": next_cluster_id,
                    "W": stat,
                    "p": p,
                    "G_mean": g.mean(),
                    "S_mean": s.mean(),
                    "mean_diff_G_minus_S": diff.mean(),
                    "n_videos": len(diff)
                })

        gs_stats = pd.DataFrame(gs_stats)
        gs_stats["abs_mean_diff_G_minus_S"] = gs_stats["mean_diff_G_minus_S"].abs()
        gs_stats["tested_in_filtered_family"] = gs_stats["abs_mean_diff_G_minus_S"] >= 0.05
        gs_stats["p_adj"] = np.nan

        gs_mask = gs_stats["p"].notna()
        if gs_mask.any():
            gs_stats.loc[gs_mask, "p_adj"] = multipletests(
                gs_stats.loc[gs_mask, "p"],
                method="fdr_bh"
            )[1]

        gs_stats.to_csv(
            os.path.join(g_vs_s_output, "GS_cluster_transition_statistics_by_larva_type.csv"),
            index=False
        )

        gs_filtered_stats = gs_stats[gs_stats["tested_in_filtered_family"]].copy()
        gs_filtered_stats["p_adj_filtered"] = np.nan

        gs_filtered_mask = gs_filtered_stats["p"].notna()
        if gs_filtered_mask.any():
            gs_filtered_stats.loc[gs_filtered_mask, "p_adj_filtered"] = multipletests(
                gs_filtered_stats.loc[gs_filtered_mask, "p"],
                method="fdr_bh"
            )[1]

        gs_filtered_stats.to_csv(
            os.path.join(g_vs_s_output, "GS_cluster_transition_statistics_by_larva_type_filtered.csv"),
            index=False
        )

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

        diff_cmap = LinearSegmentedColormap.from_list(
            "si_to_gh_gs",
            ["C1", "white", "C0"]
        )
        diff_norm = mpl.colors.TwoSlopeNorm(
            vmin=-lim,
            vcenter=0,
            vmax=lim
        )

        def diff_matrix_to_digraph(D, thresh=0.05, p_adj_lookup=None, alpha=0.05):

            G = nx.DiGraph()

            for c in D.index:
                G.add_node(int(c))

            for i in D.index:
                for j in D.columns:
                    w = float(D.loc[i, j])
                    edge = (int(i), int(j))
                    is_significant = True

                    if p_adj_lookup is not None:
                        p_adj = p_adj_lookup.get(edge, np.nan)
                        is_significant = pd.notna(p_adj) and p_adj < alpha

                    if abs(w) >= thresh and is_significant:
                        G.add_edge(edge[0], edge[1], weight=w)

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

        edges = sorted(
            G_diff.edges(data=True),
            key=lambda edge: float(edge[2]["weight"]) > 0
        )

        for u, v, d in edges:

            w = float(d["weight"])
            color = 'C1' if w < 0 else 'C0'
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

        gs_filtered_lookup = (
            gs_filtered_stats
            .set_index(["cluster", "next_cluster"])["p_adj_filtered"]
            .to_dict()
        )
        G_diff_stat_filtered = diff_matrix_to_digraph(
            D,
            thresh=0.05,
            p_adj_lookup=gs_filtered_lookup,
            alpha=0.05
        )

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        ax.set_title("GS transition difference (abs diff >= 0.05, filtered p_adj < 0.05)")
        ax.axis("off")

        nx.draw_networkx_nodes(
            G_diff_stat_filtered,
            pos,
            ax=ax,
            node_size=700,
            node_color="lightgray",
            edgecolors="black"
        )

        nx.draw_networkx_labels(
            G_diff_stat_filtered,
            pos,
            ax=ax,
            font_size=9
        )

        edges = sorted(
            G_diff_stat_filtered.edges(data=True),
            key=lambda edge: float(edge[2]["weight"]) > 0
        )

        for u, v, d in edges:

            w = float(d["weight"])
            color = 'C1' if w < 0 else 'C0'
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
        cbar.set_label("P(G tracks) − P(S tracks)")

        plt.savefig(
            os.path.join(g_vs_s_output, "GS_cluster_transition_difference_circlegraph_G_minus_S_stat_filtered.pdf"),
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
            history_stats["abs_mean_diff"] = history_stats["mean_diff"].abs()
            history_stats["tested_in_filtered_family"] = history_stats["abs_mean_diff"] >= 0.05
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

            history_filtered_stats = history_stats[history_stats["tested_in_filtered_family"]].copy()
            history_filtered_stats["p_adj_filtered"] = np.nan

            filtered_mask = history_filtered_stats["p"].notna()
            if filtered_mask.any():
                history_filtered_stats.loc[filtered_mask, "p_adj_filtered"] = multipletests(
                    history_filtered_stats.loc[filtered_mask, "p"],
                    method="fdr_bh"
                )[1]

            history_filtered_stats.to_csv(
                os.path.join(
                    comparison_output,
                    f"GS_cluster_transition_{comparison_name}_statistics_filtered.csv"
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

            edges = sorted(
                G_history.edges(data=True),
                key=lambda edge: float(edge[2]["weight"]) > 0
            )

            for u, v, d in edges:

                w = float(d["weight"])
                color = 'C1' if w < 0 else 'C0'
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

            history_filtered_lookup = (
                history_filtered_stats
                .set_index(["cluster", "next_cluster"])["p_adj_filtered"]
                .to_dict()
            )
            G_history_stat_filtered = diff_matrix_to_digraph(
                D_history,
                thresh=0.05,
                p_adj_lookup=history_filtered_lookup,
                alpha=0.05
            )

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.set_title(f"{circle_title} (filtered p_adj < 0.05)")
            ax.axis("off")

            nx.draw_networkx_nodes(
                G_history_stat_filtered,
                history_pos,
                ax=ax,
                node_size=700,
                node_color="lightgray",
                edgecolors="black"
            )

            nx.draw_networkx_labels(
                G_history_stat_filtered,
                history_pos,
                ax=ax,
                font_size=9
            )

            edges = sorted(
                G_history_stat_filtered.edges(data=True),
                key=lambda edge: float(edge[2]["weight"]) > 0
            )

            for u, v, d in edges:

                w = float(d["weight"])
                color = 'C1' if w < 0 else 'C0'
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
                    f"GS_cluster_transition_{comparison_name}_difference_circlegraph_stat_filtered.pdf"
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



       




    

    






if __name__ == "__main__":
    # Set your paths
    directory = "/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/clustered_interactions"
    interactions = "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/clustered-interactions_pt2/cropped_interactions.csv"
    clusters = "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/clustered-interactions_pt2/F15_KMAX3_DMAX4_NMIN1500--/pca-data3-F15-mcmodels4-Kmax3-Dmax4-Nmin1500-05-2026.csv"
    cluster_name = "Yhat.idt.pca"   # or whatever your cluster column is
    video_path = "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/clustered-interactions_pt2/videos_original"

    # Create instance
    pipeline = ClusterPipeline(directory, interactions, clusters, cluster_name, video_path)

    # Run methods
    pipeline.loading_data()

    pipeline.anchor_partner()
    pipeline.summary_anchor_partner()
    # pipeline.summary_clusters()
    # pipeline.hierarchal_mean_trace_summary()
    # pipeline.barplot_deviation()
    # pipeline.GS_deviation()
    # pipeline.barplot_proportion()
    # pipeline.GS_barplot_proportion()
    # pipeline.correlations_G_vs_S()
    # pipeline.umap(n_pcs=5)
    # pipeline.G_V_S_duration_transition()
    # pipeline.G_V_S_cluster_transition() 
    # pipeline.GS_cluster_transition()
