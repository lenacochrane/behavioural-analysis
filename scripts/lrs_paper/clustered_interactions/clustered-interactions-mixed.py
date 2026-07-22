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

    # pipeline.anchor_partner()
    # pipeline.hierarchal_mean_trace_summary()
    pipeline.barplot_deviation()
