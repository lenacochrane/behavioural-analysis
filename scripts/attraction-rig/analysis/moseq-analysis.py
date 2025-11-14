import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyarrow.feather as feather
import matplotlib.patches as mpatches
import cv2

# df_moseq = pd.read_csv('/Users/cochral/Desktop/MOSEQ/KEYPOINT/2025_05_21-13_29_37/moseq_df.csv')
# df_stat = pd.read_csv('/Users/cochral/Desktop/MOSEQ/KEYPOINT/2025_05_21-13_29_37/stats_df.csv')

################################ SYLLABLE STATS 

# velocity_px_s_mean 
# duration
# angular_velocity_mean
# heading_mean

# sns.barplot(data=df_stat, x='syllable', y='heading_mean', hue='group', ci='sd')

# plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/moseq/heading.png', dpi=300, bbox_inches='tight')

# plt.show()


################################ TRACK ETHOGRAM

# print(df_moseq.columns)


# # Filter only the animals you want (e.g., 'N10-GH')
# df_filtered = df_moseq[df_moseq['name'].str.startswith('N10-GH')].copy()

# tracks = df_filtered['name'].unique()

# # Make a syllable-to-color map using viridis
# syllables = sorted(df_filtered['syllable'].unique())
# palette = sns.color_palette('viridis', n_colors=len(syllables))
# syl2color = {s: palette[i] for i, s in enumerate(syllables)}

# # Start plot
# fig, ax = plt.subplots(figsize=(12, len(tracks) * 0.4))

# for i, name in enumerate(tracks):
#     sub = df_filtered[df_filtered['name'] == name].sort_values('frame_index')

#     for _, row in sub.iterrows():
#         x = row['frame_index']
#         y = i  # vertical stack by animal
#         color = syl2color[row['syllable']]
#         ax.add_patch(plt.Rectangle((x, y), 1, 1, color=color, linewidth=0))

# # Formatting
# ax.set_yticks(np.arange(len(tracks)) + 0.5)
# ax.set_yticklabels(tracks)
# ax.set_xlabel("Time")
# ax.set_ylabel("Track")
# ax.set_title("Syllable Ethogram N10-GH")
# ax.set_xlim(df_filtered['frame_index'].min(), df_filtered['frame_index'].max())
# ax.set_ylim(0, len(tracks))

# handles = [mpatches.Patch(color=c, label=s) for s, c in syl2color.items()]
# plt.legend(handles=handles, title="Syllables", bbox_to_anchor=(1.01, 1), loc='upper left')

# plt.tight_layout()
# plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/moseq/ethorgram-gh-n10.png', dpi=300, bbox_inches='tight')
# plt.show()


################################ TRACK ETHOGRAM

# # Filter only the animals you want (e.g., 'N10-GH')
# df_filtered = df_moseq[df_moseq['name'] == 'N1-GH_2025-02-24_15-16-50_td7'].copy()

# tracks = df_filtered['name'].unique()

# # Make a syllable-to-color map using viridis
# syllables = sorted(df_filtered['syllable'].unique())
# palette = sns.color_palette('viridis', n_colors=len(syllables))
# syl2color = {s: palette[i] for i, s in enumerate(syllables)}

# # Start plot
# fig, ax = plt.subplots(figsize=(12, len(tracks) * 0.4))

# for i, name in enumerate(tracks):
#     sub = df_filtered[df_filtered['name'] == name].sort_values('frame_index')

#     for _, row in sub.iterrows():
#         x = row['frame_index']
#         y = i  # vertical stack by animal
#         color = syl2color[row['syllable']]
#         ax.add_patch(plt.Rectangle((x, y), 1, 1, color=color, linewidth=0))

# # Formatting
# ax.set_yticks(np.arange(len(tracks)) + 0.5)
# ax.set_yticklabels(tracks)
# ax.set_xlabel("Time")
# ax.set_ylabel("Track")
# ax.set_title("Syllable Ethogram N10-GH")
# ax.set_xlim(df_filtered['frame_index'].min(), df_filtered['frame_index'].max())
# ax.set_ylim(0, len(tracks))

# handles = [mpatches.Patch(color=c, label=s) for s, c in syl2color.items()]
# plt.legend(handles=handles, title="Syllables", bbox_to_anchor=(1.01, 1), loc='upper left')

# plt.tight_layout()
# plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/moseq/ethorgram-example.png', dpi=300, bbox_inches='tight')
# plt.show()



""" ANALYSIS PIPELINE FOR MOSEQ DATA IN ATTRACTION RIG """
# --------------------------------------------------------
# BASIC_STATS: duration, frequency from stats_df
# --------------------------------------------------------
def basic_stats(df, output):
    
    plt.figure(figsize=(8,6))
    sns.barplot(data=df, x='syllable', y='duration',  ci='sd')
    plt.title('Syllable Duration by Condition')
    plt.ylim(0, None)
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(output, 'syllable_duration.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8,6))
    sns.barplot(data=df, x='syllable', y='frequency',  ci='sd')
    plt.title('Syllable Frequency by Condition')
    plt.ylim(0, None)
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(output, 'syllable_frequency.png'), dpi=300, bbox_inches='tight')
    plt.close()
# --------------------------------------------------------
# DURATIONS: test durations of syllables
# --------------------------------------------------------
def durations(df, output):

    def durations_by_onset(df):

        out = []
        for name, g in df.groupby("name", sort=False):
            g = g.reset_index(drop=False)  # keep original index as 'index'
            onset_rows = g[g["onset"] == True]

            start_idxs = onset_rows["index"].tolist()               # Frames: 0  1  2  3  4  5  6
            sylls = onset_rows["syllable"].tolist()                 # Onset:  T  F  F  T  F  T  F
            # add sentinel “end” at one past the last row index     # Sylls:  37 37 37 40 40 41 41
            end_sentinel = g["index"].iloc[-1] + 1                  # → Start indices: [0, 3, 5]
            next_starts = start_idxs[1:] + [end_sentinel]           # → Next starts:  [3, 5, 7]
                                                                    # → Durations:    [3, 2, 2]
            for s, n, sy in zip(start_idxs, next_starts, sylls):
                out.append({
                    "name": name,
                    "syllable": sy,
                    "start_idx": s,
                    "end_idx_exclusive": n,
                    "duration_frames": n - s
                })

        return pd.DataFrame(out)
    
    df = df.sort_values(["name", "frame_index"]).reset_index(drop=True)
    durations = durations_by_onset(df)
    print(durations)

    output = os.path.join(output, 'syllable_duration_distributions')
    if not os.path.exists(output):
        os.makedirs(output)

    # number of bouts per syllable
    bout_counts = (
        durations.groupby("syllable")
                .size()
                .reset_index(name="num_bouts"))

    plt.figure(figsize=(8,6))
    sns.barplot(data=bout_counts, x="syllable", y="num_bouts", color="steelblue")
    plt.title("Number of Bouts per Syllable")
    plt.xlabel("Syllable")
    plt.ylabel("Number of Bouts")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output, "bouts_per_syllable.png"), dpi=300)
    plt.close()


    # total frames per syllable

    total_frames = (durations.groupby(["name", "syllable"])["duration_frames"].sum().reset_index(name="total_frames"))
    plt.figure(figsize=(8,6))
    sns.barplot(data=total_frames, x='syllable', y='total_frames', ci='sd')
    plt.title('Total Frames per Syllable by Condition')
    plt.ylim(0, None)
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(output, 'total_frames_per_syllable.png'), dpi=300, bbox_inches='tight')
    plt.close()

    for syllable in durations['syllable'].unique():

        sub = durations[durations['syllable'] == syllable]
        plt.figure(figsize=(8,6))
        sns.histplot(sub['duration_frames'], bins=20, kde=False)
        plt.title(f'Syllable {syllable} Duration Distribution')
        plt.xlabel('Duration (frames)')
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        plt.savefig(os.path.join(output, f'syllable_{syllable}_duration_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    

    grouped_1_frame = durations[durations['duration_frames'] < 2].groupby(['syllable', 'name']).size().reset_index(name='count_1_frame')

    plt.figure(figsize=(8,6))
    sns.barplot(data=grouped_1_frame, x='syllable', y='count_1_frame', ci='sd')
    plt.title('Frequency of 1 frame Syllable Occurrences')
    plt.ylim(0, None)
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(output, 'frequency_1_frame_syllable_occurrences.png'), dpi=300, bbox_inches='tight')
    plt.close()

    total_counts = (durations.groupby(["syllable", 'name']).size().reset_index(name="total_bouts"))
    freqs = total_counts.merge(grouped_1_frame, on=["name", "syllable"], how="left").fillna(0)
    freqs["fraction_1_frame"] = freqs["count_1_frame"] / freqs["total_bouts"]

    plt.figure(figsize=(8,6))
    sns.barplot(data=freqs, x='syllable', y='fraction_1_frame', ci='sd')
    plt.title('Fraction of 1 frame Syllable Occurrences')
    plt.ylim(0, None)
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(output, 'fraction_1_frame_syllable_occurrences.png'), dpi=300, bbox_inches='tight')
    plt.close()
# --------------------------------------------------------
# SYLLABLE_OVERLAY: overlay syllables on video 
# --------------------------------------------------------
def syllable_overlay(df, output, video_track_name):

    f = df[df['name'] == video_track_name]
    f = f.sort_values('frame_index')
    coord_columns = ['centroid_x', 'centroid_y']  # replace with your actual centroid column names

    image_size = 1400
    original_video = cv2.VideoCapture('/Users/cochral/Desktop/MOSEQ/videos/N1-GH_2025-02-24_15-16-50_td7.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_output = cv2.VideoWriter(os.path.join(output, 'N1-GH_2025-02-24_15-16-50_td7.mp4'), fourcc, 25.0, (image_size, image_size))

    frame_number = 0
    while original_video.isOpened():
        ret, frame = original_video.read()
        if not ret:
            break

        frame_df = f[f['frame_index'] == frame_number]

        for _, row in frame_df.iterrows():
            x = row['centroid_x']
            y = row['centroid_y']
            if np.isnan([x, y]).any():
                continue
            x, y = int(x), int(y)
            color = (0, 0, 255)
            cv2.circle(frame, (x, y), radius=8, color=color, thickness=-1)
            syllable = str(row['syllable'])  # make sure it's a string
            cv2.putText(frame, syllable, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.2, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

        video_output.write(frame)
        frame_number += 1
    original_video.release()
    video_output.release()
# --------------------------------------------------------
# SYLLABLE_FEATURES: quantify syllable features 
# --------------------------------------------------------
def syllable_features(df, output):

    output = os.path.join(output, 'syllable_quantifications')
    if not os.path.exists(output):
        os.makedirs(output)

    df = df.sort_values(["name", "frame_index"]).reset_index(drop=True)
    df["bout_id"] = df.groupby("name")["onset"].cumsum()
    df["frame_per_bout"] = df.groupby(["name", "bout_id"]).cumcount()

    df.to_csv(os.path.join(output, 'moseq_df_normalized_frame.csv'), index=False)

    # Fixed Y axis range 
    hmin, hmax   = -1, 1    # heading range (radians)
    avmin, avmax = -8, 8          # angular velocity
    vmin, vmax   = 0, 450          # speed (px/s)

    for syllable in sorted(df['syllable'].unique()):
        sub = df[df["syllable"] == syllable]  

        fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

        durations = ( sub.groupby(["name", "bout_id"]).size())  # length of each syllable bout
    

        median_dur = durations.median()
        low95, high95 = durations.quantile([0.025, 0.975]).values

        ax1 = axes[0]
        sns.lineplot(data=sub, x='frame_per_bout', y='heading', ax=ax1, legend=False, ci=95)
        ax2 = axes[1]
        sns.lineplot(data=sub, x='frame_per_bout', y='angular_velocity', ax=ax2, legend=False, ci=95)
        ax3 = axes[2]
        sns.lineplot(data=sub, x='frame_per_bout', y='velocity_px_s', ax=ax3, legend=False, ci=95)

        for ax in axes:
            ax.axvline(median_dur, linestyle="--", color="gray", label="median duration")
            ax.axvline(low95, linestyle=":", color="gray", label="95% CI")
            ax.axvline(high95, linestyle=":", color="gray")

        ax1.set_title(f'Syllable {syllable} Feature Quantifications')
        ax1.set_ylabel('Heading (radians)')
        ax1.set_ylim(hmin, hmax)
        ax2.set_ylabel('Angular Velocity (radians/s)')
        ax2.set_ylim(avmin, avmax)
        ax3.set_ylabel('Velocity (px/s)')
        ax3.set_ylim(vmin, vmax)
        ax3.set_xlabel('Frame (0 = onset)')

        plt.tight_layout()
        plt.savefig(os.path.join(output, f'{syllable}_mean.png'), dpi=300, bbox_inches='tight')
        plt.close() 
    
    for syllable in sorted(df['syllable'].unique()):
        sub = df[df["syllable"] == syllable]  

        fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

        durations = ( sub.groupby(["name", "bout_id"]).size() )  # length of each syllable bout
        median_dur = durations.median()
        low95, high95 = durations.quantile([0.025, 0.975]).values

        # --- NEW: plot every individual bout trace in light gray on each axis ---
        for ax, ycol in zip(axes, ['heading','angular_velocity','velocity_px_s']):
            for (_, g) in sub.groupby(["name","bout_id"]):
                ax.plot(
                    g["frame_per_bout"].values, g[ycol].values,
                    color="0.8", linewidth=0.7, alpha=0.4
                )
        # -----------------------------------------------------------------------

        ax1 = axes[0]
        sns.lineplot(data=sub, x='frame_per_bout', y='heading',
                    ax=ax1, legend=False, ci=None, color='black', linewidth=2)

        ax2 = axes[1]
        sns.lineplot(data=sub, x='frame_per_bout', y='angular_velocity',
                    ax=ax2, legend=False, ci=None, color='black', linewidth=2)

        ax3 = axes[2]
        sns.lineplot(data=sub, x='frame_per_bout', y='velocity_px_s',
                    ax=ax3, legend=False, ci=None, color='black', linewidth=2)

        for ax in axes:
            ax.axvline(median_dur, linestyle="--", color="gray", label="median duration")
            ax.axvline(low95, linestyle=":", color="gray", label="95% CI")
            ax.axvline(high95, linestyle=":", color="gray")

        ax1.set_title(f'Syllable {syllable} Feature Quantifications')
        ax1.set_ylabel('Heading (radians)')
        ax1.set_ylim(hmin, hmax)
        ax2.set_ylabel('Angular Velocity (radians/s)')
        ax2.set_ylim(avmin, avmax)
        ax3.set_ylabel('Velocity (px/s)')
        ax3.set_ylim(vmin, vmax)
        ax3.set_xlabel('Frame (0 = onset)')

        plt.tight_layout()
        plt.savefig(os.path.join(output, f'{syllable}_traces.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    ## plot syllables over normalised time

    bout_lengths = (df.groupby(["name", 'syllable', "bout_id"]).size().reset_index(name="bout_len"))
    df = df.merge(bout_lengths, on=["name", "syllable", "bout_id"], how="left")
    df["normalised_frames"] = np.where(df["bout_len"] > 1, df["frame_per_bout"] / (df["bout_len"] - 1),0.0)

    bounds = (
    bout_lengths.groupby("syllable")["bout_len"]
                .quantile([0.1, 0.9])
                .unstack())
    bounds.columns = ["low95", "high95"]

    low95  = df["syllable"].map(bounds["low95"])
    high95 = df["syllable"].map(bounds["high95"])

    df = df[(df["bout_len"] >= low95) & (df["bout_len"] <= high95)]


    # get per-syllable quantile limits
    for syllable in sorted(df['syllable'].unique()):
        sub = df[df["syllable"] == syllable]  

        fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

        ax1 = axes[0]
        sns.lineplot(data=sub, x='normalised_frames', y='heading', ax=ax1, legend=False, ci=95)
        ax2 = axes[1]
        sns.lineplot(data=sub, x='normalised_frames', y='angular_velocity', ax=ax2, legend=False, ci=95)
        ax3 = axes[2]
        sns.lineplot(data=sub, x='normalised_frames', y='velocity_px_s', ax=ax3, legend=False, ci=95)


        ax1.set_title(f'Syllable {syllable} Feature Quantifications')
        ax1.set_ylabel('Heading (radians)')
        ax1.set_ylim(hmin, hmax)
        ax1.set_xlim(0,1)
        ax2.set_ylabel('Angular Velocity (radians/s)')
        ax2.set_ylim(avmin, avmax)
        ax2.set_xlim(0,1)
        ax3.set_ylabel('Velocity (px/s)')
        ax3.set_ylim(vmin, vmax)
        ax3.set_xlim(0,1)
        ax3.set_xlabel('Frame (0 = onset)')

        plt.tight_layout()
        plt.savefig(os.path.join(output, f'{syllable}_normalised.png'), dpi=300, bbox_inches='tight')
        plt.close() 
# --------------------------------------------------------
# COMPARING_MODELS: model comparison plots
# --------------------------------------------------------
def comparing_models(directory):

    dfs = []
    for folder in os.listdir(directory):
        if not folder.startswith('KEYPOINT'):
            continue 
        kappa = folder.split("KAPPA")[-1]
        path = os.path.join(directory, folder, 'moseq_df.csv')
        df = pd.read_csv(path)
        df['kappa'] = int(kappa)
        dfs.append(df)

    output = os.path.join(directory, 'model_comparisons')
    if not os.path.exists(output):
        os.makedirs(output)
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(os.path.join(output, 'combined_moseq_df.csv'), index=False)

    dfs_stats = []
    for folder in os.listdir(directory):
        if not folder.startswith('KEYPOINT'):
            continue 
        kappa = folder.split("KAPPA")[-1]
        path = os.path.join(directory, folder, 'stats_df.csv')
        df_stat = pd.read_csv(path)
        df_stat['kappa'] = int(kappa)
        dfs_stats.append(df_stat)

    df_stat = pd.concat(dfs_stats, ignore_index=True)
    df_stat.to_csv(os.path.join(output, 'combined_summary_df.csv'), index=False)


    # 1. QUANTIFY 1 FRAME SYLLABLE OCCURRENCES

    def durations_by_onset(df):
        out = []
        for (kappa, name), g in df.groupby(["kappa", "name"], sort=False):
            g = g.reset_index(drop=False)  # keep original index as 'index'
            onset_rows = g[g["onset"] == True]

            start_idxs = onset_rows["index"].tolist()               # Frames: 0  1  2  3  4  5  6
            sylls = onset_rows["syllable"].tolist()                 # Onset:  T  F  F  T  F  T  F
            # add sentinel “end” at one past the last row index     # Sylls:  37 37 37 40 40 41 41
            end_sentinel = g["index"].iloc[-1] + 1                  # → Start indices: [0, 3, 5]
            next_starts = start_idxs[1:] + [end_sentinel]           # → Next starts:  [3, 5, 7]
                                                                    # → Durations:    [3, 2, 2]
            for s, n, sy in zip(start_idxs, next_starts, sylls):
                out.append({
                    'kappa': kappa,
                    "name": name,
                    "syllable": sy,
                    "start_idx": s,
                    "end_idx_exclusive": n,
                    "duration_frames": n - s})
        return pd.DataFrame(out)
    
    df = df.sort_values(['kappa', "name", "frame_index"]).reset_index(drop=True)
    durations = durations_by_onset(df)
    print(durations)
    
    ## per video 
    grouped_1_frame = durations[durations['duration_frames'] < 2].groupby(['kappa', 'name'])['duration_frames'].sum().reset_index(name='count_1_frame') # sum of frames 
    video_totals = durations.groupby(['kappa', 'name'])['duration_frames'].sum().reset_index(name='video_frame_count_total') # total frames per video
    video_level = video_totals.merge(grouped_1_frame, on=['kappa', 'name'], how='left')
    video_level['percent_1_frame'] = (video_level['count_1_frame'] / video_level['video_frame_count_total'] * 100)
    video_level.to_csv(os.path.join(output, 'percentage_1_frame_syllable_occurrences_per_video.csv'), index=False)

    plt.figure(figsize=(8,6))
    sns.barplot(data=video_level, x='kappa', y='percent_1_frame', ci='sd')
    plt.title('Percentage of 1 Frame Syllable Occurrences')
    plt.ylim(0, None)
    plt.xlabel('Kappa')
    plt.ylabel('Percentage of 1 Frame Syllable Occurrences (%)')
    plt.savefig(os.path.join(output, 'percentage_1_frame_syllable_occurrences_per_video.png'), dpi=300, bbox_inches='tight')
    plt.close()

    ## per syllable - identifying the % of syllables which are rubbish (dominated by 1 frame percentage is high)
    syllable_1_frame = durations[durations['duration_frames'] < 2].groupby(['kappa',  'syllable'])['duration_frames'].sum().reset_index(name='count_1_frame') 
    syllable_totals = durations.groupby(['kappa', 'syllable'])['duration_frames'].sum().reset_index(name='syllable_frame_count_total')
    syllable_level = syllable_totals.merge(syllable_1_frame, on=['kappa',  'syllable'], how='left')
    syllable_level['count_1_frame'] = syllable_level['count_1_frame'].fillna(0)
    syllable_level['percent_1_frame'] = (syllable_level['count_1_frame'] / syllable_level['syllable_frame_count_total'] * 100)
    syllable_level.to_csv(os.path.join(output, 'percentage_1_frame_syllable_occurrences_per_syllable.csv'), index=False)

    thresholds = [1, 5, 10, 30, 50, 70, 90]

    bad_syllables = []  
    for thresh in thresholds:
        syllable_level['bad_syllable'] = syllable_level['percent_1_frame'] > thresh #boolean- syllable above or over threshold
        summary = (syllable_level.groupby('kappa')['bad_syllable'].mean() * 100).reset_index(name=f'percent_junk_syllables')
        for _, row in summary.iterrows():
            bad_syllables.append({
                'kappa': int(row['kappa']),
                'threshold_percent': thresh,
                '1_frame_dominant_syllables_percent': row['percent_junk_syllables'],
            })
    
    bad_syllables_df = pd.DataFrame(bad_syllables)
    bad_syllables_df.to_csv(os.path.join(output, 'bad_syllables_summary.csv'), index=False)

    for threshold in bad_syllables_df['threshold_percent'].unique():

        sub = bad_syllables_df[bad_syllables_df['threshold_percent'] == threshold]
        plt.figure(figsize=(8,6))
        sns.barplot(data=sub, x='kappa', y='1_frame_dominant_syllables_percent')
        plt.title(f'Percentage of Syllables Dominated by 1 Frame (> {threshold}%)')
        plt.ylim(0, None)
        plt.xlabel('Kappa')
        plt.ylabel('Percentage of Junk Syllables (%)')
        plt.savefig(os.path.join(output, f'bad_syllables_percentage_threshold_{threshold}.png'), dpi=300, bbox_inches='tight')
        plt.close()


    # 2. QUANTIFY NUMBER OF SYLLABLES UNDER THRESHOLD (NOT INC IN SUMMARY GIFS)

    total_syllable_durations = durations.groupby(['kappa', 'syllable'])['duration_frames'].sum().reset_index(name='syllable_frame_count_total')

    kappa_rows = []

    for kappa in sorted(total_syllable_durations['kappa'].unique()):
        
        ## moseq_df syllables
        sub_total = total_syllable_durations[total_syllable_durations['kappa'] == kappa]
        total_syllable_number = sub_total['syllable'].nunique()
        total_frames = sub_total['syllable_frame_count_total'].sum()

        ## stats_df syllables
        df_stat['kappa'] = df_stat['kappa'].astype(int)
        sub_stats_df = df_stat[df_stat['kappa'] == int(kappa)]
        syllables_stats = sub_stats_df['syllable'].unique()

        ## identify hidden syllables 
        hidden = sub_total[~sub_total['syllable'].isin(syllables_stats)]

        n_hidden_syllables = hidden['syllable'].nunique()
        hidden_frames = hidden['syllable_frame_count_total'].sum()

        percent_hidden_frames = (
            100 * hidden_frames / total_frames if total_frames > 0 else 0.0)
        
        percentage_hidden_syllables = (
            100 * n_hidden_syllables / total_syllable_number if total_syllable_number > 0 else 0.0)

        kappa_rows.append({
            'kappa': int(kappa),
            'total_syllables': int(total_syllable_number),
            'no_underthreshold_syllables': int(n_hidden_syllables),
            'percent_underthreshold_syllables': percentage_hidden_syllables,
            'percent_frames_underthreshold_syllables': percent_hidden_frames,
        })
    
    underthreshold_syllables_df = pd.DataFrame(kappa_rows)
    underthreshold_syllables_df.to_csv(os.path.join(output, 'underthreshold_syllables_per_kappa.csv'),index=False)

    # % if syllables under threshold
    plt.figure(figsize=(8,6))
    sns.barplot(data=underthreshold_syllables_df, x='kappa', y='percent_underthreshold_syllables')
    plt.title('Percentage of Syllables Under Threshold')
    plt.ylim(0, None)
    plt.xlabel('Kappa')
    plt.ylabel('Percentage of Syllables Under Threshold (%)')
    plt.savefig(os.path.join(output, 'percentage_underthreshold_syllables.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # % frames belonging to syllables under threshold
    plt.figure(figsize=(8,6))
    sns.barplot(data=underthreshold_syllables_df, x='kappa', y='percent_frames_underthreshold_syllables')
    plt.title('Percentage of Frames Belonging to Syllables Under Threshold')
    plt.ylim(0, None)
    plt.xlabel('Kappa')
    plt.ylabel('Percentage of Frames Under Threshold (%)')
    plt.savefig(os.path.join(output, 'percentage_underthreshold_syllables_frames.png'), dpi=300, bbox_inches='tight')
    plt.close()


    # 3. QUANTIFY NUMBER OF FRAMES WHICH ARENT WITHIN THE 10-90th PERCENTILE OF DURATION OF A SYLLABLE

    quantiles = (
        durations
        .groupby(['kappa', 'syllable'])['duration_frames']
        .quantile([0.10, 0.90])
        .unstack()  # index: (kappa, syllable), columns: 0.10, 0.90
        .reset_index()
        .rename(columns={0.10: 'q10', 0.90: 'q90'}))
    
    durations_with_quantiles = durations.merge(quantiles, on=['kappa', 'syllable'], how='left')

    durations_with_quantiles['inside_quantiles_bool'] = ((durations_with_quantiles['duration_frames'] >= durations_with_quantiles['q10']) & (durations_with_quantiles['duration_frames'] <= durations_with_quantiles['q90']))
    durations_with_quantiles['frames_outside'] = np.where(durations_with_quantiles['inside_quantiles_bool'], 0, durations_with_quantiles['duration_frames'])
    durations_with_quantiles['frames_inside'] = np.where(durations_with_quantiles['inside_quantiles_bool'],durations_with_quantiles['duration_frames'],0)

    summary_quantiles = (
    durations_with_quantiles.groupby(['kappa', 'syllable'], as_index=False)
    .agg(
        frames_inside=('frames_inside', 'sum'),
        frames_outside=('frames_outside', 'sum')))
    
    summary_quantiles['fraction_outside_quantiles'] = summary_quantiles['frames_outside'] / (summary_quantiles['frames_inside'] + summary_quantiles['frames_outside']) * 100
    summary_quantiles.to_csv(os.path.join(output, 'frames_outside_10-90th_percentile_per_syllable.csv'), index=False)

    plt.figure(figsize=(8,6))
    sns.barplot(data=summary_quantiles, x='kappa', y='fraction_outside_quantiles', ci='sd')
    plt.title('Fraction of Frames Outside 10-90th Percentile of Syllable Duration')
    plt.ylim(0, None)
    plt.xlabel('Kappa')
    plt.ylabel('Fraction of Frames Outside 10-90th Percentile(%)')
    plt.savefig(os.path.join(output, 'fraction_frames_outside_10-90th_percentile.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8,6))
    sns.barplot(data=summary_quantiles, x='kappa', y='frames_outside', ci='sd')
    plt.title('Number of Frames Relating to Syllables which are outside the 10-90th Percentile of Syllable Duration')
    plt.ylim(0, None)
    plt.xlabel('Kappa')
    plt.ylabel('Number of Frames Outside 10-90th Percentile')
    plt.savefig(os.path.join(output, 'number_frames_outside_10-90th_percentile.png'), dpi=300, bbox_inches='tight')
    plt.close() 






























    
    








        

   




directory = '/Users/cochral/Desktop/MOSEQ'
comparing_models(directory)

    


## plot syllables over time - and also mean line but idk yes 

## can i add in the testing 1 video overlay yes


# df_moseq = pd.read_csv('/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA3600/moseq_df.csv')
# df_stat = pd.read_csv('/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA3600/stats_df.csv')


# output = '/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA3600/testing' 
# if not os.path.exists(output):
#     os.makedirs(output)


# basic_stats(df_stat, output)
# durations(df_moseq, output)
# syllable_overlay(df_moseq, output, 'N1-GH_2025-02-24_15-16-50_td7') #video_track_name = 'N1-GH_2025-02-24_15-16-50_td7'  
# syllable_features(df_moseq, output)