
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os 



df = pd.read_csv('/Users/cochral/Desktop/SLAEP/TRain/testing-nemo-down/test.csv')

# Optional: extract video name from file if needed
filename = "test"  # or parse from path
output_dir = '/Users/cochral/Desktop/SLAEP/TRain/testing-nemo-down/returns'

returns = []


# Step 1: Gather all return events across all tracks
for track in df['track_id'].unique():
    track_df = df[df['track_id'] == track].sort_values(by='frame').reset_index(drop=True)
    states = track_df['within_hole'].astype(bool).values
    frames = track_df['frame'].values
    # filename = os.path.splitext(os.path.basename(track_df['file'].iloc[0]))[0] if 'file' in track_df.columns else 'unknown'

    i = 1
    while i < len(states):
        if states[i - 1] and not states[i]:  # Exit
            exit_frame = frames[i]
            j = i + 1
            while j < len(states):
                if not states[j - 1] and states[j]:  # Return
                    return_frame = frames[j]
                    segment = track_df[(track_df['frame'] >= exit_frame) & (track_df['frame'] <= return_frame)]
                    
                    # Step 2: Plot and save this return
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.plot(segment['x_body'], segment['y_body'], marker='o')
                    ax.set_title(f"Track {track} Return\nFrames {exit_frame}-{return_frame}")
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.invert_yaxis()
                    ax.set_xlim(0, 100)
                    ax.set_ylim(0, 100)
                    plt.tight_layout()

                    # Save figure
                    save_path = os.path.join(
                        output_dir,
                        f"{filename}_track{track}_frames{exit_frame}-{return_frame}.png"
                    )
                    plt.savefig(save_path)
                    plt.close()

                    i = j
                    break
                j += 1
        i += 1









