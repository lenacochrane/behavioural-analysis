
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import os
from sklearn.decomposition import PCA

###################### NEAREST NEIGHOUR PLOT 

# df0_nn = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated/nearest_neighbour.csv')
# df0_nn['condition'] = 'PSEUDO-SI'

# df1_nn = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/nearest_neighbour.csv')
# df1_nn['condition'] = 'PSEUDO-GH'

# df2_nn = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/nearest_neighbour.csv')
# df2_nn['condition'] = 'GH'

# df3_nn = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/nearest_neighbour.csv')
# df3_nn['condition'] = 'SI'

# df_nn = pd.concat([df0_nn, df1_nn, df2_nn, df3_nn], ignore_index=True)

# bins = np.arange(0, 91, 1)
# df_nn['bin'] = pd.cut(df_nn['body-body'], bins, include_lowest=True, right=False)
# df_nn['bin_center'] = df_nn['bin'].apply(lambda x: x.mid)


# counts = (
#     df_nn.groupby(['filename', 'condition', 'bin_center'])
#     .size()
#     .groupby(['filename', 'condition'], group_keys=False)
#     .apply(lambda x: x / x.sum())
#     .reset_index(name='density'))


# comparisons = [
#     ('GH', 'SI'),
#     ('GH', 'PSEUDO-GH'),
#     ('SI', 'PSEUDO-SI')]

# titles = [
#     'GH vs SI',
#     'GH vs PSEUDO-GH',
#     'SI vs PSEUDO-SI'
# ]


base_palette = sns.color_palette("tab10")

custom_palette = {
    "GH": base_palette[0],             # classic seaborn blue
    "SI": base_palette[1],             # classic seaborn orange
    "PSEUDO-GH": (0.0, 0.2, 0.4),      # dark blue
    "PSEUDO-SI": (0.6, 0.3, 0.0)       # dark orange
}

# fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)

# for i, ((cond1, cond2), title) in enumerate(zip(comparisons, titles)):
#     ax = axes[i]
#     sub = counts[counts['condition'].isin([cond1, cond2])]

#     sns.lineplot(
#         data=sub,
#         x='bin_center',
#         y='density',
#         hue='condition',
#         errorbar='sd',
#         ax=ax,
#         legend=False,
#           palette=custom_palette  # legend only on last subplot
#     )

#     ax.set_title(title, fontsize=13, fontweight='bold')
#     ax.set_xlim(0, 90)
#     ax.set_ylim(0, 0.08)
#     ax.set_xlabel('', fontsize=12, fontweight='bold')

#     if i == 0:
#         ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
#     else:
#         ax.set_ylabel('')
#         # ax.set_yticklabels([])
#         ax.tick_params(labelleft=False) 


# fig.supxlabel('Nearest Neighbour Distance (mm)', fontsize=12, fontweight='bold')

# # ---- Reserve space at the bottom and add a figure-level legend ----
# fig.subplots_adjust(wspace=0.15, right=0.82)

# import matplotlib.patches as mpatches
# legend_patches = [
#     mpatches.Patch(color=custom_palette["GH"],         label="GH"),
#     mpatches.Patch(color=custom_palette["SI"],         label="SI"),
#     mpatches.Patch(color=custom_palette["PSEUDO-GH"],  label="PSEUDO-GH"),
#     mpatches.Patch(color=custom_palette["PSEUDO-SI"],  label="PSEUDO-SI"),
# ]
# fig.legend(
#     handles=legend_patches,
#     loc="center right",
#     bbox_to_anchor=(0.81, 0.75),
#     title="Condition",
#     frameon=True
# )


# plt.suptitle('Nearest Neighbour Distributions', fontsize=16, fontweight='bold')
# # plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# plt.savefig(
#     '/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/maggot-meeting/nn.png',
#     dpi=300, bbox_inches='tight'
# )
# # plt.show()



# ###################### INTERACTION TYPE FREQUENCY

# df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/interaction_types.csv')
# df1['condition'] = 'GH'

# df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/interaction_types.csv')
# df2['condition'] = 'SI'

# df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated/interaction_types.csv')
# df3['condition'] = 'PSEUDO-SI'

# df4 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/interaction_types.csv')
# df4['condition'] = 'PSEUDO-GH'

# df_int = pd.concat([df1, df2, df3, df4], ignore_index=True)

# grouped = (
#     df_int.groupby(['file', 'condition', 'interaction_type'])['count']
#     .sum()
#     .reset_index())

# grouped['interaction_type'] = grouped['interaction_type'].str.replace('_', '-', regex=False)

# fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)

# for i, ((cond1, cond2), title) in enumerate(zip(comparisons, titles)):
#     ax = axes[i]
#     sub = grouped[grouped['condition'].isin([cond1, cond2])]

#     sns.barplot(data=sub, x='interaction_type', y='count', hue='condition', ci='sd', ax=ax, legend=False, alpha=0.8, edgecolor="black", linewidth=2, palette=custom_palette )

#     ax.set_title(title, fontsize=13, fontweight='bold')
#     ax.set_ylim(0, 600)
#     ax.set_xlabel('', fontsize=12, fontweight='bold')
#     ax.tick_params(axis='x', labelrotation=45)

#     if i == 0:
#         ax.set_ylabel('Count', fontsize=12, fontweight='bold')
#     else:
#         ax.set_ylabel('')
#         # ax.set_yticklabels([])
#         ax.tick_params(labelleft=False) 


# fig.supxlabel('Interaction Type', fontsize=12, fontweight='bold')

# # ---- Reserve space at the bottom and add a figure-level legend ----
# fig.subplots_adjust(wspace=0.15, right=0.82, bottom=0.158)



# import matplotlib.patches as mpatches
# legend_patches = [
#     mpatches.Patch(color=custom_palette["GH"],         label="GH"),
#     mpatches.Patch(color=custom_palette["SI"],         label="SI"),
#     mpatches.Patch(color=custom_palette["PSEUDO-GH"],  label="PSEUDO-GH"),
#     mpatches.Patch(color=custom_palette["PSEUDO-SI"],  label="PSEUDO-SI"),
# ]
# fig.legend(
#     handles=legend_patches,
#     loc="center right",
#     bbox_to_anchor=(0.81, 0.75),
#     title="Condition",
#     frameon=True)

# plt.suptitle('Interaction Type Count', fontsize=16, fontweight='bold')
# # plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# plt.savefig(
#     '/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/maggot-meeting/type.png',
#     dpi=300, bbox_inches='tight'
# )
# plt.show()


# ###################### SPEED

# df_speed = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/speed_over_time.csv')
# df_speed['condition'] = 'SI'

# df0_speed = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/speed_over_time.csv')
# df0_speed['condition'] = 'GH'

# plt.figure(figsize=(8,6))

# df_spd = pd.concat([df_speed, df0_speed], ignore_index=True)


# bins = np.linspace(0, 2.5, 26)  # 0 to 2.5 in 0.1 increments
# df_spd['speed_bin'] = pd.cut(df_spd['speed'], bins, include_lowest=True)
# df_spd['bin_center'] = df_spd['speed_bin'].apply(lambda x: x.mid)


# # Count per file-condition-bin
# counts = (
#     df_spd.groupby(['file', 'condition', 'bin_center'])
#     .size()
#     .groupby(['file', 'condition'], group_keys=False)
#     .apply(lambda x: x / x.sum())
#     .reset_index(name='density')
# )


# sns.lineplot(data=counts, x='bin_center', y='density', hue='condition', errorbar='sd', palette=custom_palette)

# plt.ylabel('Probability', fontsize=12, fontweight='bold')
# plt.xlabel('Speed (mm/s)', fontsize=12, fontweight='bold')

# plt.xlim(0,2.5)
# plt.ylim(0, None)

# plt.title('Speed', fontsize=16, fontweight='bold')

# plt.tight_layout(rect=[1, 1, 1, 1])

# plt.xticks(rotation=45)
# # plt.xticks(fontweight='bold')

# plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/maggot-meeting/speed.png', dpi=300, bbox_inches='tight')
# plt.show()



# ###################### ENSEMBLE

# df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/ensemble_msd.csv')
# df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/ensemble_msd.csv')

# plt.figure(figsize=(8,6))
# sns.lineplot(data=df5, x='time', y='squared_distance',  color=base_palette[0], errorbar='sd', label='GH')
# sns.lineplot(data=df6, x='time', y='squared_distance',  color=base_palette[1], errorbar='sd', label='SI')

# plt.xlabel('Time (S)', fontsize=12, fontweight='bold')
# plt.ylabel('Ensemble Mean Squared Distance (mm)', fontsize=12, fontweight='bold')

# plt.title('Ensemble Mean Squared Distance', fontsize=16, fontweight='bold')


# plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/maggot-meeting/ensemble.png', dpi=300, bbox_inches='tight')

# plt.show()

# ###################### TIME AVERAGE

# df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/time_average_msd.csv')
# df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/time_average_msd.csv')

# plt.figure(figsize=(8,6))

# ### N10
# sns.lineplot(data=df1, x='tau', y='msd', ci=None, label='SI', color=base_palette[1])
# sns.lineplot(data=df2, x='tau', y='msd', ci=None, label='GH', color=base_palette[0])

# plt.xlabel('Tau', fontsize=12,fontweight='bold')
# plt.ylabel('MSD', fontsize=12,fontweight='bold')


# plt.title('Time Average Mean Squared Distance', fontsize=16, fontweight='bold')

# plt.legend(title='Condition')



# plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/maggot-meeting/time-average.png', dpi=300, bbox_inches='tight')


# plt.show()

# ###################### DISTANCE FROM CENTRE

# d1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/distance_from_centre.csv')
# d1['condition'] = 'SI'

# d2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/distance_from_centre.csv')
# d2['condition'] = 'GH'

# df = pd.concat([d1, d2], ignore_index=True)

# plt.figure(figsize=(8,6))


# bins = np.linspace(0, 50, 25)  # 0 to 2.5 in 0.1 increments
# df['distance_bin'] = pd.cut(df['distance_from_centre'], bins, include_lowest=True)
# df['bin_center'] = df['distance_bin'].apply(lambda x: x.mid)


# # Count per file-condition-bin
# counts = (
#     df.groupby(['file', 'condition', 'bin_center'])
#     .size()
#     .groupby(['file', 'condition'], group_keys=False)
#     .apply(lambda x: x / x.sum())
#     .reset_index(name='density')
# )


# sns.lineplot(data=counts, x='bin_center', y='density', hue='condition', errorbar='sd', palette=custom_palette)

# plt.xlabel('Distance From Centre (mm) ', fontsize=12, fontweight='bold')
# plt.ylabel('Probability', fontsize=12, fontweight='bold')

# plt.ylim(0, None)

# plt.title('Distances from the Centre Distribution', fontsize=16, fontweight='bold')


# plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/maggot-meeting/distance-centre.png', dpi=300, bbox_inches='tight')

# plt.show()


# ###################### EUCLIDEAN DISTANCE

# df_d = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/euclidean_distances.csv')
# df_d0 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/euclidean_distances.csv')

# plt.figure(figsize=(8,6))

# sns.lineplot(data=df_d, x='time', y='average_distance',  color=base_palette[0], ci='sd', label='GH')
# sns.lineplot(data=df_d0, x='time', y='average_distance',  color=base_palette[1], ci='sd', label='SI')


# plt.xlabel('Time (S)', fontsize=12, fontweight='bold')
# plt.ylabel('Average Distance (mm)', fontsize=12, fontweight='bold')

# plt.title('Euclidean Distances', fontsize=16, fontweight='bold')

# plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/maggot-meeting/euc.png', dpi=300, bbox_inches='tight')

# plt.show()


# ###################### ANGLE

# df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/angle_over_time.csv')
# df5['condition'] = 'GH'

# df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/angle_over_time.csv')
# df6['condition'] = 'SI'

# df = pd.concat([df5, df6], ignore_index=True)

# plt.figure(figsize=(8,6))


# bins = np.linspace(0, 180, 18)  # 0 to 2.5 in 0.1 increments
# df['bin'] = pd.cut(df['angle'], bins, include_lowest=True)
# df['bin_center'] = df['bin'].apply(lambda x: x.mid)


# counts = (
#     df.groupby(['file', 'condition', 'bin_center'])
#     .size()
#     .groupby(['file', 'condition'], group_keys=False)
#     .apply(lambda x: x / x.sum())
#     .reset_index(name='density')
# )


# sns.lineplot(data=counts, x='bin_center', y='density', hue='condition', errorbar='sd', palette=custom_palette)

# plt.xlabel('Angle', fontsize=12, fontweight='bold')
# plt.ylabel('Probability', fontsize=12, fontweight='bold')

# plt.title('Trajectory Angle Probability Distribution', fontsize=16, fontweight='bold')

# plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/maggot-meeting/angle.png', dpi=300, bbox_inches='tight')
# plt.show()



##################### NEAREST NEIGHOUR - SPEED

# df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/nearest_neighbour.csv')
# df5['condition'] = 'GH'

# # df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/nearest_neighbour.csv')
# # df6['condition'] = 'SI'

# df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/nearest_neighbour.csv')
# df6['condition'] = 'PSEUDO-GH'

# df = pd.concat([df5, df6], ignore_index=True)

# plt.figure(figsize=(8,8))

# bins = np.arange(0, 91, 1)
# df['bin'] = pd.cut(df['body-body'], bins, include_lowest=True, right=False)
# df['bin_center'] = df['bin'].apply(lambda x: x.mid)

# summary = (
#     df.groupby(['filename', 'condition', 'bin_center'])['speed']
#     .mean()
#     .reset_index())

# sns.lineplot(
#         data=summary,
#         x='bin_center',
#         y='speed',
#         hue='condition',
#         errorbar='sd', palette=custom_palette)


# plt.ylim(0,None)
# plt.xticks(range(0, 21, 1))   # 0, 1, 2, …, 20
# plt.xlim(0,20)

# plt.title('Speed vs Nearest Neighbour Distance', fontsize=16, fontweight='bold')
# plt.xlabel('Nearest Neighbour (mm)', fontsize=12, fontweight='bold')
# plt.ylabel('Speed (mm/s)', fontsize=12, fontweight='bold')

# plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/maggot-meeting/nn-speed-PSUED.png', dpi=300, bbox_inches='tight')

# plt.show()

















###################### INTERACTIONS



def cluster_pipeline(cropped_interaction, cluster, cluster_name, video_path, output_dir):

    base_palette = sns.color_palette("tab10")

    custom_palette = {
    "group": base_palette[0],             # classic seaborn blue
    "iso": base_palette[1]}


    #### LOAD DATAFRAMES
 
    df_cropped_interaction = pd.read_csv(cropped_interaction)
    df_cluster = pd.read_csv(cluster) # edit youngser's cluster csv 

    #### MERGE DATAFRAMES

    df = pd.merge(
        df_cropped_interaction, 
        df_cluster[['interaction_id', cluster_name]], 
        on='interaction_id', 
        how='inner'
    )
    
  
    counts = (
        df
        .groupby(['file', 'condition', cluster_name])
        .size()
        .reset_index(name='count')
    )

    # Compute total interactions per video (for normalization)
    totals = (
        counts.groupby('file')['count']
        .transform('sum')
    )

    # Add proportion column
    counts['proportion'] = counts['count'] / totals
    
    summary_df = (
        counts
        .set_index(['file', 'condition', cluster_name])
        .unstack(fill_value=0)
        .stack()
        .reset_index()
    )
    summary_df.rename(columns={0: 'proportion'}, inplace=True)

    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=summary_df,
        x=cluster_name, y='proportion', hue='condition',ci='sd', alpha=0.8, palette=custom_palette, edgecolor='black', linewidth=2)
    
    plt.title("Proportion of Clusters", fontsize=16, fontweight='bold')
    plt.xlabel("Cluster ID", fontsize=12, fontweight='bold')
    plt.ylabel("Proportion", fontsize=12, fontweight='bold')
    # plt.xticks(rotation=90)
    plt.tight_layout()

    per_video_prop_path = os.path.join(output_dir, 'cluster_proportions_per_video.png')
    plt.savefig(per_video_prop_path, dpi=300, bbox_inches='tight')
    plt.show()


    ################## TOTAL NUMBER

    unique_interactions = (
    df[['file', 'condition', 'interaction_id']]
    .drop_duplicates()
    )

    file_counts = (
        unique_interactions
        .groupby(['file', 'condition'])
        .size()
        .reset_index(name='n_interactions')
    )

    plt.figure(figsize=(5, 4))
    sns.barplot(
        data=file_counts,
        x='condition', y='n_interactions',
        palette=custom_palette, edgecolor='black', linewidth=2, alpha=0.5)

    plt.title("Total Number of Interactions", fontsize=16, fontweight='bold')
    plt.xlabel(" ", fontsize=12, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.ylabel("Total Interactions", fontsize=12, fontweight='bold')
    plt.tight_layout()

    total_counts_path = os.path.join(output_dir, 'total_interactions.png')
    plt.savefig(total_counts_path, dpi=300, bbox_inches='tight')
    plt.show()


    ######## normalised frame

    # --- 1) Filter for the normalised frame ---
    df_norm = df[df['Normalized Frame'] == 0]   # adjust if your reference frame is named differently

    # --- 2) Plot distribution of min_distance ---
    plt.figure(figsize=(5, 4))
    sns.histplot(
        data=df_norm, x='min_distance', hue='condition',
        bins=30, stat='density', common_norm=False,
        palette=custom_palette, element='bars',  # ensures filled bars
    edgecolor=None,   # removes the outlines
    alpha=0.4
    )

    sns.histplot(
    data=df_norm, x='min_distance', hue='condition',
    bins=30, stat='density', common_norm=False,
    palette=custom_palette, fill=False, lw=1,
    element='step', alpha=0.6
)

    plt.title("Minimum Distance at Normalised Frame", fontsize=16, fontweight='bold')
    plt.xlabel("Min Distance (mm)", fontsize=12, fontweight='bold')
    plt.ylabel("Density", fontsize=12, fontweight='bold')
    # plt.xticks(fontsize=12, fontweight='bold')
    # plt.yticks(fontsize=12, fontweight='bold')
    plt.tight_layout()

    out_path = os.path.join(output_dir, 'min_distance_distribution.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()


    


    ## == Returns a straightness score 
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

    # for interaction_id, group in df.groupby('interaction_id'):
    #     group = group.sort_values('Frame')
    #     coords1 = group[['Track_1 x_body','Track_1 y_body']].dropna().values
    #     coords2 = group[['Track_2 x_body','Track_2 y_body']].dropna().values
    #     if len(coords1) < 2 or len(coords2) < 2:
    #         continue
    #     # Compute PCA axes & scores
    #     axis1, s1 = compute_pca_axis(coords1)
    #     axis2, s2 = compute_pca_axis(coords2)
    #     # Choose anchor and partner
    #     if s1 >= s2:
    #         winner = 1
    #         anchor_pts, partner_pts, anchor_axis = coords1, coords2, axis1
    #     else:
    #         winner = 2
    #         anchor_pts, partner_pts, anchor_axis = coords2, coords1, axis2

    #     # Align both
    #     start = anchor_pts[0]
    #     A_al = align_and_flip(anchor_pts, anchor_axis, start)
    #     B_al = align_and_flip(partner_pts, anchor_axis, start)



    #     # Horizontal flip if partner is left
    #     # if np.median(B_al[:,0]) < 0:
    #     #     A_al[:,0] *= -1
    #     #     B_al[:,0] *= -1

    #     # Horizontal flip if partner starts on the left
    #     if B_al[0, 0] < 0:
    #         A_al[:, 0] *= -1
    #         B_al[:, 0] *= -1

    #     # Vertical flip if anchor is predominantly down
    #     if np.mean(A_al[:,1]) < 0:
    #         A_al[:,1] *= -1
    #         B_al[:,1] *= -1
    #     # Assign back to DataFrame
    #     idx = group.index[:len(A_al)]
    #     df.loc[idx, ['anchor x_body','anchor y_body']]  = A_al
    #     df.loc[idx, ['partner x_body','partner y_body']] = B_al# Initialize aligned columns

    #     # → tag which original track was anchor (1 or 2)
    #     df.loc[idx, 'anchor_track']  = winner
    #     df.loc[idx, 'partner_track'] = 3 - winner

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
            winner = 1
            anchor_pts, partner_pts, anchor_axis = coords1, coords2, axis1
        else:
            winner = 2
            anchor_pts, partner_pts, anchor_axis = coords2, coords1, axis2

        # Align both (body)
        start = anchor_pts[0]
        A_al = align_and_flip(anchor_pts, anchor_axis, start)
        B_al = align_and_flip(partner_pts, anchor_axis, start)

        # --- NEW: align head/tail using the SAME reference (before flips) ---
        h1 = group[['Track_1 x_head','Track_1 y_head']].dropna().values
        t1 = group[['Track_1 x_tail','Track_1 y_tail']].dropna().values
        h2 = group[['Track_2 x_head','Track_2 y_head']].dropna().values
        t2 = group[['Track_2 x_tail','Track_2 y_tail']].dropna().values

        A_head = align_and_flip(h1 if winner == 1 else h2, anchor_axis, start) if (len(h1) or len(h2)) else np.empty((0,2))
        A_tail = align_and_flip(t1 if winner == 1 else t2, anchor_axis, start) if (len(t1) or len(t2)) else np.empty((0,2))
        B_head = align_and_flip(h2 if winner == 1 else h1, anchor_axis, start) if (len(h1) or len(h2)) else np.empty((0,2))
        B_tail = align_and_flip(t2 if winner == 1 else t1, anchor_axis, start) if (len(t1) or len(t2)) else np.empty((0,2))
        # --------------------------------------------------------------------

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

        # Assign back to DataFrame (body)
        idx = group.index[:len(A_al)]
        df.loc[idx, ['anchor x_body','anchor y_body']]   = A_al
        df.loc[idx, ['partner x_body','partner y_body']] = B_al

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
    'speed',
    'acceleration',
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
    




    # for cluster_id in sorted(df[cluster_name].unique()):
    #     cluster_df = df[df[cluster_name] == cluster_id]
    #     grouped = cluster_df.groupby("Normalized Frame")

    #     t1_x = grouped["anchor x_body"].mean()
    #     t1_y = grouped["anchor y_body"].mean()
    #     t2_x = grouped["partner x_body"].mean()
    #     t2_y = grouped["partner y_body"].mean()

    #     plt.figure(figsize=(6, 6))
    #     plt.plot(t1_x, t1_y, label="Track 1", color="red")
    #     plt.plot(t2_x, t2_y, label="Track 2", color="blue")
    #     plt.scatter(t1_x.iloc[0], t1_y.iloc[0], color="red", marker="o", label="T1 Start")
    #     plt.scatter(t2_x.iloc[0], t2_y.iloc[0], color="blue", marker="o", label="T2 Start")

    #     # plt.gca().invert_yaxis()
    #     plt.ylim(0,300)
    #     plt.xlim(-100,100)
    #     plt.title(f"Mean Trajectory - Cluster {cluster_id}")
    #     plt.legend()
    #     plt.tight_layout()

    #     trajectory_dir = os.path.join(output_dir, "mean_trajectories")
    #     os.makedirs(trajectory_dir, exist_ok=True)

    #     save_path = os.path.join(trajectory_dir, f"cluster_{cluster_id}.png")

    #     plt.savefig(save_path)
    #     plt.close()
    

    # for cluster_id in sorted(df[cluster_name].unique()):
    #     cluster_df = df[df[cluster_name] == cluster_id]
    #     grouped = cluster_df.groupby("Normalized Frame")

    #     t1_x = grouped["anchor x_body"].mean()
    #     t1_y = grouped["anchor y_body"].mean()
    #     t2_x = grouped["partner x_body"].mean()
    #     t2_y = grouped["partner y_body"].mean()

    #     plt.figure(figsize=(6, 6))
    #     plt.plot(t1_x, t1_y, label="Track 1", color="red", linestyle="None", marker="|", markersize=10, alpha=0.6)
    #     plt.plot(t2_x, t2_y, label="Track 2", color="blue", linestyle="None", marker="_", markersize=10, alpha=0.6)

    #     plt.scatter(t1_x.iloc[0], t1_y.iloc[0], color="red", marker="o", label="T1 Start")
    #     plt.scatter(t2_x.iloc[0], t2_y.iloc[0], color="blue", marker="o", label="T2 Start")

    #     # plt.gca().invert_yaxis()
    #     plt.ylim(0,300)
    #     plt.xlim(-100,100)
    #     plt.title(f"Mean Trajectory - Cluster {cluster_id}")
    #     plt.legend()
    #     plt.tight_layout()

    #     trajectory_dir = os.path.join(output_dir, "mean_trajectories")
    #     os.makedirs(trajectory_dir, exist_ok=True)

    #     save_path = os.path.join(trajectory_dir, f"cluster_{cluster_id}_dash.png")

    #     plt.savefig(save_path)
    #     plt.close()
    
    # for cluster_id in sorted(df[cluster_name].unique()):
    #     cluster_df = df[df[cluster_name] == cluster_id]
    #     grouped = cluster_df.groupby("Normalized Frame")

    #     # Body (aligned)
    #     t1_x = grouped["anchor x_body"].mean()
    #     t1_y = grouped["anchor y_body"].mean()
    #     t2_x = grouped["partner x_body"].mean()
    #     t2_y = grouped["partner y_body"].mean()

    #     # Head / Tail (aligned)
    #     a_hx = grouped["anchor x_head"].mean()
    #     a_hy = grouped["anchor y_head"].mean()
    #     a_tx = grouped["anchor x_tail"].mean()
    #     a_ty = grouped["anchor y_tail"].mean()

    #     p_hx = grouped["partner x_head"].mean()
    #     p_hy = grouped["partner y_head"].mean()
    #     p_tx = grouped["partner x_tail"].mean()
    #     p_ty = grouped["partner y_tail"].mean()

    #     plt.figure(figsize=(6, 6))

    #     # Body “dash” markers
    #     plt.plot(t1_x, t1_y, label="Anchor body", color="red",
    #             linestyle="None", marker="|", markersize=10, alpha=0.6)
    #     plt.plot(t2_x, t2_y, label="Partner body", color="blue",
    #             linestyle="None", marker="_", markersize=10, alpha=0.6)

    #     # Head/Tail points — plot ALL points (no subsampling)
    #     plt.scatter(a_hx, a_hy, s=18, alpha=0.85, color="red",   label="Anchor head")
    #     plt.scatter(a_tx, a_ty, s=18, alpha=0.85, color="orange",label="Anchor tail")
    #     plt.scatter(p_hx, p_hy, s=18, alpha=0.85, color="navy",  label="Partner head")
    #     plt.scatter(p_tx, p_ty, s=18, alpha=0.85, color="cyan",  label="Partner tail")

    #     # Start markers for body
    #     plt.scatter(t1_x.iloc[0], t1_y.iloc[0], color="red",  marker="o", label="Anchor start")
    #     plt.scatter(t2_x.iloc[0], t2_y.iloc[0], color="blue", marker="o", label="Partner start")

    #     plt.ylim(0, 300)
    #     plt.xlim(-100, 100)
    #     plt.title(f"Mean Trajectory - Cluster {cluster_id}")

    #     # dedupe legend entries (optional but nice)
    #     handles, labels = plt.gca().get_legend_handles_labels()
    #     by_label = dict(zip(labels, handles))
    #     plt.legend(by_label.values(), by_label.keys(), loc="best", fontsize=8)

    #     plt.tight_layout()

    #     trajectory_dir = os.path.join(output_dir, "mean_trajectories")
    #     os.makedirs(trajectory_dir, exist_ok=True)
    #     save_path = os.path.join(trajectory_dir, f"cluster_{cluster_id}_dash_ht.png")
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #     plt.close()

    
    # #### head-head interactions 
    # for cluster_id in sorted(df[cluster_name].unique()):
    #     cluster_df = df[df[cluster_name] == cluster_id]
    #     grouped = cluster_df.groupby("Normalized Frame")

    #     # Body (aligned)
    #     t1_x = grouped["anchor x_body"].mean()
    #     t1_y = grouped["anchor y_body"].mean()
    #     t2_x = grouped["partner x_body"].mean()
    #     t2_y = grouped["partner y_body"].mean()

    #     # Head / Tail (aligned)
    #     a_hx = grouped["anchor x_head"].mean()
    #     a_hy = grouped["anchor y_head"].mean()
    #     # a_tx = grouped["anchor x_tail"].mean()
    #     # a_ty = grouped["anchor y_tail"].mean()

    #     p_hx = grouped["partner x_head"].mean()
    #     p_hy = grouped["partner y_head"].mean()
    #     # p_tx = grouped["partner x_tail"].mean()
    #     # p_ty = grouped["partner y_tail"].mean()

    #     plt.figure(figsize=(6, 6))

    #     # Body “dash” markers
    #     plt.plot(t1_x, t1_y, label="Anchor body", color="blue",
    #             linestyle="None", marker="|", markersize=10, alpha=0.6)
    #     plt.plot(t2_x, t2_y, label="Partner body", color="orange",
    #             linestyle="None", marker="_", markersize=10, alpha=0.6)

    #     # Head/Tail points — plot ALL points (no subsampling)
    #     plt.scatter(a_hx, a_hy, s=18, alpha=0.85, color="blue",   label="Anchor head")
    #     # plt.scatter(a_tx, a_ty, s=18, alpha=0.85, color="orange",label="Anchor tail")
    #     plt.scatter(p_hx, p_hy, s=18, alpha=0.85, color="orange",  label="Partner head")
    #     # plt.scatter(p_tx, p_ty, s=18, alpha=0.85, color="cyan",  label="Partner tail")

    #     # Start markers for body
    #     plt.scatter(t1_x.iloc[0], t1_y.iloc[0], color="darkblue",  marker="o", label="Anchor start")
    #     plt.scatter(t2_x.iloc[0], t2_y.iloc[0], color="darkorange", marker="o", label="Partner start")

    #     plt.ylim(0, 300)
    #     plt.xlim(-100, 100)
    #     plt.title(f"Mean Trajectory - Cluster {cluster_id}")

    #     # dedupe legend entries (optional but nice)
    #     handles, labels = plt.gca().get_legend_handles_labels()
    #     by_label = dict(zip(labels, handles))
    #     plt.legend(by_label.values(), by_label.keys(), loc="best", fontsize=8)

    #     plt.tight_layout()

    #     trajectory_dir = os.path.join(output_dir, "mean_trajectories")
    #     os.makedirs(trajectory_dir, exist_ok=True)
    #     save_path = os.path.join(trajectory_dir, f"cluster_{cluster_id}_dash_head.png")
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #     plt.close()
    

    #### SUBPLOTS HEAD-HEAD

    # ==== 4-up head–head figure for clusters 1–4 (horizontal), no axes ====
    clusters_to_plot = [1, 2, 3, 4]

    fig, axes = plt.subplots(1, 4, figsize=(10, 6))  # wide row of 4
    for ax, cluster_id in zip(axes, clusters_to_plot):
        cluster_df = df[df[cluster_name] == cluster_id]

        grouped = cluster_df.sort_values("Normalized Frame").groupby("Normalized Frame")

        # Body (aligned)
        t1_x = grouped["anchor x_body"].mean()
        t1_y = grouped["anchor y_body"].mean()
        t2_x = grouped["partner x_body"].mean()
        t2_y = grouped["partner y_body"].mean()

        # Head (aligned)
        a_hx = grouped["anchor x_head"].mean()
        a_hy = grouped["anchor y_head"].mean()
        p_hx = grouped["partner x_head"].mean()
        p_hy = grouped["partner y_head"].mean()

        # Body “dash” markers (same colors as your example)
        ax.plot(t1_x, t1_y, label="Anchor body", color="#4F7942",
                linestyle="None", marker="|", markersize=10, alpha=0.6)
        ax.plot(t2_x, t2_y, label="Partner body", color="#916288",
                linestyle="None", marker="_", markersize=10, alpha=0.6)

        # Head points — all frames
        ax.scatter(a_hx, a_hy, s=18, alpha=0.85, color="#4F7942",   label="Anchor head")
        ax.scatter(p_hx, p_hy, s=18, alpha=0.85, color="#916288", label="Partner head")

        # Start markers for body
        ax.scatter(t1_x.iloc[0], t1_y.iloc[0], color="#4F7942",   marker="o", label="Anchor start")
        ax.scatter(t2_x.iloc[0], t2_y.iloc[0], color="#916288", marker="o", label="Partner start")

        # Same view window & proportions
        ax.set_xlim(-50, 100)
        ax.set_ylim(-10, 300)
        ax.set_aspect('equal', adjustable='box')

        # Title per panel
        ax.set_title(f"Cluster {cluster_id}", fontweight='bold', fontsize=12)

        # REMOVE axes completely
        ax.axis('off')

    # optional: a single legend for the whole figure
    # (collect from the first axis that has artists)
    # handles, labels = axes[0].get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # fig.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=12)
#     fig.legend(
#     by_label.values(), by_label.keys(),
#     loc="center right",
#     bbox_to_anchor=(1.02, 0.5),   # just outside the right edge, vertically centered
#     fontsize=14, markerscale=1.5
# )

    # Collect legend handles/labels from first axis
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    # Keep only head entries
    keep = ["Anchor head", "Partner head"]
    filtered = {lab: by_label[lab] for lab in keep if lab in by_label}

    fig.legend(
        filtered.values(), filtered.keys(),
        loc="center right", bbox_to_anchor=(1.1, 0.8),
        fontsize=12, markerscale=1
    )

    # fig.suptitle("Head–Head Interactions", fontsize=16, fontweight="bold")

    plt.tight_layout()
    # plt.subplots_adjust(wspace=-0.8)  
    out_path = os.path.join(output_dir, "mean_trajectories", "clusters_1_4_headhead_row.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    



    


 








cropped_interaction = '/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser/test4_F29/cropped_interactions.csv'
cluster = '/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser/test3_F18/pca-data2-F18.csv'
cluster_name = 'Yhat.idt.pca' # edit name of clusters  
video_path = '/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser/videos_original'
output_dir = "/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/maggot-meeting" 
os.makedirs(output_dir, exist_ok=True)


cluster_pipeline(cropped_interaction, cluster, cluster_name, video_path, output_dir)
