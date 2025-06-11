import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyarrow.feather as feather
import matplotlib.patches as mpatches

df_moseq = pd.read_csv('/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA10/2025_05_29-11_45_53/moseq_df.csv')
# df_stat = pd.read_csv('/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA3600/2025_06_06-11_20_54/stats_df.csv')

print(df_moseq.columns)
# make sure rows are in temporal order
df_moseq = df_moseq.sort_values(['name', 'frame_index']).reset_index(drop=True)

# 1) each time onset=True, bump the bout_id
df_moseq['bout_id'] = df_moseq.groupby('name')['onset'].cumsum()

# 2) within each (name, bout_id) group, count frames from 0,1,2,...
df_moseq['fake_frame'] = df_moseq.groupby(['name', 'bout_id']).cumcount()

# Filter to syllable 2 and get each bout’s length (number of frames)
durations = (
    df_moseq[df_moseq['syllable'] == 2]
    .groupby(['name', 'bout_id'])
    .size()
)

# Print statistics
print(f"Syllable 2 durations (in frames):")
print(f"  min  = {durations.min():.0f}")
print(f"  mean = {durations.mean():.1f}")
print(f"  max  = {durations.max():.0f}")



# ——— Fixed y-axis ranges ———
hmin, hmax   = -0.6, 0.6    # heading range (radians)
avmin, avmax = -3, 3          # angular velocity
vmin, vmax   = 0, 250           # speed (px/s)

# ——— prep output dir ———
output_dir = '/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/moseq/syllable_quantifications_10'
os.makedirs(output_dir, exist_ok=True)

features = ['heading','angular_velocity','velocity_px_s']

for syllable in sorted(df_moseq['syllable'].unique()):
    aligned = {feat: [] for feat in features}

    # ——— first, compute the duration statistics for this syllable ———
    durations = (
        df_moseq[df_moseq['syllable'] == syllable]
        .groupby(['name','bout_id'])
        .size()
    )
    mean_d   = durations.mean()
    low95, high95 = durations.quantile([0.025,0.975]).values

    # ——— then build your aligned traces exactly as before ———
    for session_name, session_df in df_moseq.groupby('name'):
        session_df = session_df.reset_index(drop=True)
        onsets = session_df[
            (session_df['syllable']==syllable)&(session_df['onset'])
        ].index.tolist()

        for onset_idx in onsets:
            trace = []
            i = onset_idx
            while i < len(session_df) and session_df.loc[i,'syllable']==syllable:
                trace.append(session_df.loc[i,features].values)
                i += 1

            if trace:
                arr = np.vstack(trace)
                for j,feat in enumerate(features):
                    aligned[feat].append(arr[:,j])

    # ——— padding + average ———
    if not aligned[features[0]]:
        continue
    max_len = max(len(x) for x in aligned[features[0]])
    for feat in features:
        aligned[feat] = [
            np.pad(x,(0,max_len-len(x)),constant_values=np.nan)
            for x in aligned[feat]
        ]
    means = {feat: np.nanmean(np.vstack(aligned[feat]),axis=0)
             for feat in features}


    # ——— plotting ———
    fig, axs = plt.subplots(3,1,figsize=(10,8),sharex=True)

# Heading
    axs[0].plot(means['heading'], label='mean trace')
    axs[0].axvline(mean_d, linestyle='--')
    axs[0].axvline(low95,   linestyle=':')
    axs[0].axvline(high95,  linestyle=':')
    axs[0].set_ylim(hmin, hmax)
    axs[0].set_ylabel('heading')
    axs[0].set_title(f'Syllable {syllable}')

    # Angular velocity
    axs[1].plot(means['angular_velocity'], label='mean trace')
    axs[1].axvline(mean_d, linestyle='--')
    axs[1].axvline(low95,   linestyle=':')
    axs[1].axvline(high95,  linestyle=':')
    axs[1].set_ylim(avmin, avmax)
    axs[1].set_ylabel('angular_velocity')

    # Velocity
    axs[2].plot(means['velocity_px_s'], label='mean trace')
    axs[2].axvline(mean_d, linestyle='--')
    axs[2].axvline(low95,   linestyle=':')
    axs[2].axvline(high95,  linestyle=':')
    axs[2].set_ylim(vmin, vmax)
    axs[2].set_ylabel('velocity_px_s')
    axs[2].set_xlabel('Frame (0 = onset)')

    # Single legend, layout
    axs[0].legend(loc='upper right')
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    # ——— save ———
    plt.savefig(os.path.join(output_dir,f'syllable_{syllable}.png'))
    plt.close()






# Output directory
# output_dir = '/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/moseq/syllable_quantifications'
# os.makedirs(output_dir, exist_ok=True)


# features = ['heading', 'angular_velocity', 'velocity_px_s']

# for syllable in sorted(df_moseq['syllable'].unique()):
#     aligned = {feat: [] for feat in features}

#     for session_name, session_df in df_moseq.groupby('name'):
#         session_df = session_df.reset_index(drop=True)

#         # Get all onsets for this session where syllable == current syllable
#         syllable_onsets = session_df[
#             (session_df['syllable'] == syllable) & (session_df['onset'] == True)
#         ].index.tolist()

#         for onset_idx in syllable_onsets:
#             trace = []
#             i = onset_idx

#             # Step forward while syllable remains the same
#             while (
#                 i < len(session_df)
#                 and session_df.loc[i, 'syllable'] == syllable
#             ):
#                 trace.append(session_df.loc[i, features].values)
#                 i += 1

#             if trace:
#                 trace_array = np.array(trace)
#                 for j, feat in enumerate(features):
#                     aligned[feat].append(trace_array[:, j])

#     # Padding and averaging
#     if aligned[features[0]]:  # Only continue if data collected
#         max_len = max(len(seq) for seq in aligned[features[0]])
#         for feat in features:
#             aligned[feat] = [np.pad(seq, (0, max_len - len(seq)), constant_values=np.nan) for seq in aligned[feat]]
#         means = {feat: np.nanmean(np.vstack(aligned[feat]), axis=0) for feat in features}

#         # Plotting
#         fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
#         for ax, feat in zip(axs, features):
#             ax.plot(means[feat])
#             ax.set_title(f'{feat} over syllable duration')
#             ax.set_ylabel(feat)

#         axs[-1].set_xlabel('Frame (0 = onset)')
#         fig.suptitle(f'Syllable {syllable}', fontsize=14)
#         plt.tight_layout()
#         plt.subplots_adjust(top=0.9)

#         # Save
#         outpath = os.path.join(output_dir, f'syllable_{syllable}.png')
#         plt.savefig(outpath)
#         plt.close()
