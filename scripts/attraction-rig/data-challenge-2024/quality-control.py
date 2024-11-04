import os
import glob
import numpy as np
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

### QUALITY CONTROL TO FLAG VIDEOS WITH MORE TRACKING 

directory = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-quality-control'

data = []   

for file in os.listdir(directory):
    if file.endswith("tracks.feather"):
        file_path = os.path.join(directory, file) # join directory with file
    
        ### CREATE FOLDER WITH FILENAME 
        file_name = file.replace(".tracks.feather", "")
        # Define the path for the new folder
        folder_path = os.path.join(directory, file_name)

        # Check if the folder already exists, if not, create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)  

        ### OPEN FEATHER FILE 
        df = pd.read_feather(file_path)

        ### CONVERSION MM TO PIXELS (SHD NOT BE NECESSARY THO)
        # mm_to_pixel = ['x_body', 'y_body', 'x_head', 'y_head', 'x_tail', 'y_tail']
        # df[mm_to_pixel] = df[mm_to_pixel] * (1032 / 90) 

        ### PLOT OF NUMBER OF INSTANCES OVER TIME 

        track_counts = df.groupby('frame')['track_id'].nunique()

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=track_counts)
        plt.title(f"Number of Tracks per Frame")
        plt.xlabel("Frame")
        plt.ylabel("Number of Track IDs")

        plot_path = os.path.join(folder_path, f"number_of_tracks.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()  

        ### FIRST AND LAST FRAME OFF EVERY TRACK 
        track_first_last_df = df.groupby('track_id').agg(first_frame=('frame', 'min'), last_frame=('frame', 'max')).reset_index()
        csv_path = os.path.join(folder_path, f"track_first_last_frames.csv")
        track_first_last_df.to_csv(csv_path ,index=False)

        ### CREATE PLOTS FOR BODY X,Y COORDINATES OF EACH TRACK TRAJECTORY 

        for track_id, track_data in df.groupby('track_id'):
            plt.figure(figsize=(8, 6))
            plt.plot(track_data['x_body'], track_data['y_body']) 
            plt.title(f"Track {track_id}: Body Coordinates")
            plt.xlabel("X Body")
            plt.ylabel("Y Body")
            plt.xlim(0,1400)
            plt.ylim(0,1400)

            # Save plot for each track in the file's output folder
            track_plot_path = os.path.join(folder_path, f"track_{track_id}.png")
            plt.tight_layout()
            plt.savefig(track_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
        

        df['dx'] = df.groupby('track_id')['x_body'].diff().fillna(0) # change in x per track
        df['dy'] = df.groupby('track_id')['y_body'].diff().fillna(0) # change in y per track
        df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2) # distance travelled 

        df['track_jumps'] = df.groupby('track_id')['distance'].transform(lambda x: x > 25) # jumped if greater than 25 pixel movement 

        track_jumps = df[df['track_jumps']].copy()
        track_jump_path = os.path.join(folder_path, 'potential_track_jumps.csv')
        track_jumps.to_csv(track_jump_path, index=False)


        ### META DATA FOR DIRECTORY

        total_tracks = df['track_id'].nunique()
        track_jump_number = track_jumps.shape[0]

        track_lengths = track_first_last_df['last_frame'] - track_first_last_df['first_frame'] #df created above
        average_track_length = track_lengths.mean()


        data.append({'file':file_name, 'total tracks': total_tracks, 'average track length': average_track_length, 'track jumps': track_jump_number})
    

summary_df = pd.DataFrame(data)
summary_path = os.path.join(directory, "summary.csv")
summary_df.to_csv(summary_path, index=False)



       


        





    





















  