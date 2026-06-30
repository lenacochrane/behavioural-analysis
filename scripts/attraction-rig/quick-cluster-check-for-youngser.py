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



#### CLASS TO ANALYSE THE CLUSTERS  
class ClusterPipeline:

    def __init__(self, directory, interactions, clusters, cluster_name):
        
        self.directory = directory
        self.interaction_path = interactions 
        self.cluster_path = clusters
        self.cluster_name = cluster_name

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

                # Number of unique interactions per condition
        interaction_counts = (
            self.df[['interaction_id', 'condition']]
            .drop_duplicates()
            .groupby('condition')['interaction_id']
            .nunique()
            .sort_index()
        )

        print("\nInteractions per condition:")
        print(interaction_counts)

        print(f"\nTotal interactions: {interaction_counts.sum()}")


                # Raw cluster counts per condition
        cluster_counts = (
            self.df[['interaction_id', 'condition', self.cluster_name]]
            .drop_duplicates(subset=['interaction_id'])
            .groupby([self.cluster_name, 'condition'])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )

        print("\nRaw interaction counts per cluster and condition:")
        print(cluster_counts)
    





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


        counts = (
            df.groupby([cluster_name, 'condition'])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=condition_order, fill_value=0)
        )

        ## PROPORTION BARPLOT

        proportions = counts.div(counts.sum(axis=0), axis=1)

        proportions_reset = proportions.copy()
        proportions_reset[cluster_name] = proportions_reset.index
        proportions_reset = proportions_reset.melt(
            id_vars=cluster_name,
            var_name='condition',
            value_name='proportion'
        )

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

        # summary_csv_path = os.path.join(self.directory, 'per_video_cluster_proportions_long.csv')
        # summary_df.to_csv(summary_csv_path, index=False)

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

        output = os.path.join(self.directory, 'cluster_proportions_per_video.pdf')
        plt.savefig(output, format='pdf', bbox_inches='tight')
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

        # output = os.path.join(self.directory, 'deviation_observed_minus_expected_3conditions.csv')
        # deviation_long.to_csv(output, index=False)

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

   
        path = os.path.join(self.directory, 'deviations_3conditions.pdf')
        plt.savefig(path, format="pdf", bbox_inches="tight", dpi=300, transparent=True)
        plt.close()




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
        # path = os.path.join(self.directory, 'deviation_pvals.csv')
        # pvals.to_csv(path, index=False, float_format='%.10f')


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


        path = os.path.join(self.directory, 'deviations.pdf')  
        plt.savefig(path, format="pdf", bbox_inches="tight", dpi=300, transparent=True)
        plt.close()
    




    def barplot_deviation_GS_social_experience(self):

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

        # path = os.path.join(self.directory, 'deviation_GS_social_experience.csv')
        # deviation_long.to_csv(path, index=False)

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

        # path = os.path.join(self.directory, 'deviation_pvals_GS_social_experience.csv')
        # pvals.to_csv(path, index=False, float_format='%.10f')

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



        path = os.path.join(self.directory, 'deviations_GS_social_experience.pdf')
        plt.savefig(path, format='pdf', bbox_inches='tight', dpi=300, transparent=True)

        plt.close()
    


    def GS_social_experience_cluster_proportions_per_video(self):

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

        # summary_csv_path = os.path.join(
        #     self.directory,
        #     'GS_social_experience_cluster_proportions_per_video.csv'
        # )
        # summary_df.to_csv(summary_csv_path, index=False)

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

        path = os.path.join(self.directory, 'GS_social_experience_cluster_proportions_per_video.pdf')
        plt.savefig(path, format='pdf', bbox_inches='tight')

        plt.close()

    

    def raw_counts(self):

        df = self.clusters.copy()
        df_interaction = self.df.copy()
        cluster_name = self.cluster_name

        def get_social_experience(row):
            condition = row['condition']

            if condition == 'G':
                return 'G'
            elif condition == 'S':
                return 'S'
            elif condition == 'GS':
                pair = row['Interaction Pair']

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

            return np.nan

        pair_lookup = (
            df_interaction[['interaction_id', 'Interaction Pair']]
            .drop_duplicates(subset=['interaction_id'])
        )

        df = df.merge(
            pair_lookup,
            on='interaction_id',
            how='left'
        )

        df['social_experience'] = df.apply(get_social_experience, axis=1)

        df = df.dropna(subset=['social_experience'])

        raw_counts = (
            df[['interaction_id', cluster_name, 'social_experience']]
            .drop_duplicates(subset=['interaction_id'])
            .groupby([cluster_name, 'social_experience'])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=['G', 'S', 'S-S', 'G-S', 'G-G'], fill_value=0)
            .sort_index()
        )

        raw_counts['Total'] = raw_counts.sum(axis=1)

        totals_row = pd.DataFrame(
            [raw_counts.sum(axis=0)],
            index=['Total']
        )

        raw_counts = pd.concat([raw_counts, totals_row])

        print("\nRaw interaction counts per cluster and social experience:")
        print(raw_counts)

        output = os.path.join(self.directory, 'raw_counts_per_cluster_social_experience.csv')
        raw_counts.to_csv(output)

        print(f"\nSaved raw counts to: {output}")


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





    
    









if __name__ == "__main__":

    # folder to save outputs
    directory = "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/clustered-interactions_pt2/F15_KMAX3_DMAX4_NMIN1500--COMBINED"
    # path to interactions CSV #cropped_interactions
    interactions = "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/clustered-interactions_pt2/cropped_interactions.csv"
    # path to clusters CSV
    clusters = "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/clustered-interactions_pt2/F15_KMAX3_DMAX4_NMIN1500--COMBINED/pca-data3-F15-mcmodels4-Kmax3-Dmax4-Nmin1500-05-2026.csv"
    # Yhat.idt.pca
    cluster_name = "Yhat.idt.pca"   # or whatever your cluster column is

    # Create instance
    pipeline = ClusterPipeline(directory, interactions, clusters, cluster_name)

    # Run methods
    pipeline.loading_data()

    pipeline.raw_counts()

    merge_dict = {
    1: 1,
    2: 1,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 10,
    12: 10}


    pipeline.merge_clusters(merge_dict)


    pipeline.barplots()
    pipeline.barplot_deviation()
    pipeline.barplot_deviation_GS_social_experience()
    pipeline.GS_social_experience_cluster_proportions_per_video()