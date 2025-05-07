
import pandas as pd
import numpy as np
import os 
import pyarrow.feather as feather
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point
from scipy.spatial.distance import cdist
from shapely.affinity import scale
from shapely.wkt import dumps as wkt_dumps
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import cv2
from shapely import wkt
from shapely.affinity import scale
from shapely.wkt import loads as load_wkt
import random


### PSEUDO DATAFRAMES: MULTIPLE TRACKS FOR EACH FRAME (EACH TRACK HAS AN ASSOCIATED FILENAME)


def pseudo_population_euclidean_distance(directory):
        
        files = os.listdir(directory)
        pseudo_files = [f for f in files if f.startswith('pseudo_population_') and f.endswith('.csv')]

        data = []

        for pseudo_track in pseudo_files:
            file_path = os.path.join(directory, pseudo_track)
            df = pd.read_csv(file_path)
            df.sort_values(by='frame', ascending=True)

            for frame in df['frame'].unique():
                unique_frame = df[df['frame'] == frame]

                if unique_frame.empty or unique_frame[['x_body', 'y_body']].isnull().any().any():
                    print(f"Skipping frame {frame} in {pseudo_track} due to missing data.")
                    continue    

                body_coordinates = unique_frame[['x_body', 'y_body']].to_numpy()
                distance = cdist(body_coordinates, body_coordinates, 'euclidean')

                np.fill_diagonal(distance, np.nan)
                average_distance = np.nanmean(distance)

                data.append({'time': frame, 'average_distance': average_distance, 'file': pseudo_track})

        df = pd.DataFrame(data)
        df = df.sort_values(by=['time', 'file'], ascending=True)

        output_file = os.path.join(directory, 'euclidean_distances.csv')

        df.to_csv(output_file, index=False)
        return df


def time_average_msd(directory, taus):

    dfs = []

    files = os.listdir(directory)
    pseudo_files = [f for f in files if f.startswith('pseudo_population_') and f.endswith('.csv')]

    for pseudo_track in pseudo_files:
        file_path = os.path.join(directory, pseudo_track)
        df = pd.read_csv(file_path)
        df.sort_values(by='frame', ascending=True)
        # df['file'] = pseudo_track

        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = df[["filename", "track_id", "frame", "x_body", "y_body"]]

    # one value per tau 
    def msd_per_tau(df, tau):

        squared_displacements = []
        # for unique_track in df['track_id'].unique():
        grouped_data = df.groupby(['filename', 'track_id'])
        for (file, track_id), unique_track in grouped_data:

            unique_track = unique_track.sort_values(by='frame').reset_index(drop=True)
            

            if len(unique_track) > tau:

                initial_positions = unique_track[['x_body', 'y_body']].values[:-tau] # values up till tau as a NumPy array # positions from t to t-N-tau # represent starting points
                tau_positions = unique_track[['x_body', 'y_body']].values[tau:] # values from tau onwards # t+tau to t-N # representing ending points 
                disp = np.sum((tau_positions - initial_positions) ** 2, axis=1) # squared displacement for each pair
                squared_displacements.append(disp)  

        if squared_displacements:
        # Flatten the list of arrays into a single NumPy array
            flattened_displacements = np.concatenate(squared_displacements)

        # Filter out NaN and inf values
            valid_displacements = flattened_displacements[np.isfinite(flattened_displacements)]

            if valid_displacements.size > 0:
                mean_disp = np.mean(valid_displacements)
                return mean_disp


    msds = []
    for tau in taus:
        msd = msd_per_tau(df, tau)
        msds.append(msd)

    tau_msd_df = pd.DataFrame({'tau': taus, 'msd': msds})
    tau_msd_df = tau_msd_df.sort_values(by='tau', ascending=True)
    file_path = os.path.join(directory,'time_average_msd.csv')
    tau_msd_df.to_csv(file_path, index=False)

    return tau_msd_df


def ensemble_msd(directory):

    data = []

    files = os.listdir(directory)
    pseudo_files = [f for f in files if f.startswith('pseudo_population_') and f.endswith('.csv')]

    for pseudo_track in pseudo_files:
        file_path = os.path.join(directory, pseudo_track)
        df = pd.read_csv(file_path)
        df.sort_values(by='frame', ascending=True)

        ## CENTRE COORDINATES SHOULD ALL BE 0,0 ACTUALLY DUE TO CENTROID NORMALISATION 
        centre_x = 0
        centre_y = 0

        grouped_data = df.groupby(['filename', 'track_id'])
        for (file, track_id), unique_track in grouped_data:
            unique_track = unique_track.sort_values(by='frame').reset_index(drop=True)
               
            for _, row in unique_track.iterrows():
                squared_distance = (row['x_body'] - centre_x) ** 2 + (row['y_body'] - centre_y) ** 2
                data.append({
                    'time': row['frame'], 
                    'squared_distance': squared_distance, 
                    'file': row['filename']
                })
                    
    # Create a DataFrame from the MSD data
    df = pd.DataFrame(data)
    df = df.sort_values(by=['time'], ascending=True)

    # Save the DataFrame as a CSV file
    output_path = os.path.join(directory, 'ensemble_msd.csv')
    df.to_csv(output_path, index=False)

    return df 


def speed(directory):

    speed = []
    data = []
    
    files = os.listdir(directory)
    pseudo_files = [f for f in files if f.startswith('pseudo_population_') and f.endswith('.csv')]

    for pseudo_track in pseudo_files:
        file_path = os.path.join(directory, pseudo_track)
        df = pd.read_csv(file_path)
        df.sort_values(by='frame', ascending=True)

        grouped_data = df.groupby(['filename', 'track_id'])
        for (file, track_id), track_unique in grouped_data:
            track_unique = track_unique.sort_values(by='frame').reset_index(drop=True)

            for i in range(len(track_unique) - 1):

                row = track_unique.iloc[i]
                next_row = track_unique.iloc[i+1]

                distance = np.sqrt((row['x_body'] - next_row['x_body'])**2 + (row['y_body'] - next_row['y_body'])**2)

                time1 = row['frame']
                time2 = next_row['frame']

                time = time2 - time1
          
                speed_value = distance / time 

                speed.append(speed_value)

                data.append({'time': time2, 'speed': speed_value})

    speed_over_time = pd.DataFrame(data)
    speed_over_time = speed_over_time.sort_values(by=['time'], ascending=True)
    speed_over_time.to_csv(os.path.join(directory, 'speed_over_time.csv'), index=False)

    return speed_over_time
    


def distance_from_centre(directory): 

    data = []

    files = os.listdir(directory)
    pseudo_files = [f for f in files if f.startswith('pseudo_population_') and f.endswith('.csv')]

    for pseudo_track in pseudo_files:
        file_path = os.path.join(directory, pseudo_track)
        df = pd.read_csv(file_path)
        df.sort_values(by='frame', ascending=True)

        centre_x = 0
        centre_y = 0

        for index, row in df.iterrows():
            x, y = row['x_body'], row['y_body']
            distance = np.sqrt((centre_x - x)**2 + (centre_y - y)**2)
            
            data.append({'file': row['filename'], 'track': row['track_id'], 'frame': row['frame'], 'distance_from_centre': distance})
        
    df_distance_over_time = pd.DataFrame(data)
    df_distance_over_time.to_csv(os.path.join(directory, 'distance_over_time.csv'), index=False)

    return df_distance_over_time


def trajectory(directory):

    dfs = []

    files = os.listdir(directory)
    pseudo_files = [f for f in files if f.startswith('pseudo_population_') and f.endswith('.csv')]

    for pseudo_track in pseudo_files:
        file_path = os.path.join(directory, pseudo_track)
        df = pd.read_csv(file_path)
        df.sort_values(by='frame', ascending=True)

        df['file'] = pseudo_track
        dfs.append(df)

    # Concatenate the dataframes 
    df = pd.concat(dfs, ignore_index=True)
    grouped_data = df.groupby(['file', 'track_id'])
    
    # definition to calculate angle 
    def angle_calculator(vector_A, vector_B):

        # convert to an array for mathmatical ease 
        A = np.array(vector_A, dtype=np.float64)
        B = np.array(vector_B, dtype=np.float64)
        
        # Ensure there are no NaN values in the vectors and check for zero-length vectors
        if not np.isnan(A).any() and not np.isnan(B).any():
            # calculate magnitude of the vector
            magnitude_A = np.linalg.norm(A)
            magnitude_B = np.linalg.norm(B)
            
            # ensure magnitude =! 0
            if magnitude_A != 0 and magnitude_B != 0:
                # Calculate the dot product
                dot_product = np.dot(A, B)
                
                # cosθ
                cos_theta = dot_product / (magnitude_A * magnitude_B)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure valid range for arccos
    
                # θ in radians
                theta_radians = np.arccos(cos_theta)
                # θ in degrees
                theta_degrees = np.degrees(theta_radians)
                return theta_degrees
        
        return np.nan
    
    angles = []
    data = []

    # really dont get why you have to iterate in such a way ????
    for (file, track_id), unique_track in grouped_data:
        unique_track = unique_track.sort_values(by='frame').reset_index(drop=True)

        for i in range(len(unique_track) - 1):

            head = unique_track.iloc[i][['x_head', "y_head"]].values
            body = unique_track.iloc[i][['x_body', 'y_body']].values
            tail = unique_track.iloc[i][['x_tail', 'y_tail']].values

            HB = head - body
            BT = tail - body 

            angle = angle_calculator(HB, BT)

            frame = unique_track.iloc[i]['frame']
            # filename = track_unique.iloc[i]['file']

            angles.append(angle)
            data.append({'time': frame, 'angle': angle, 'file': file})
    
    angle_over_time = pd.DataFrame(data)
    angle_over_time = angle_over_time.sort_values(by=['time'], ascending=True)
    angle_over_time.to_csv(os.path.join(directory, 'angle_over_time.csv'), index=False)



from itertools import combinations
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import os

def contact(directory, proximity_threshold=1):

    def process_track_pair(track_a, track_b, df, track_file):
        results = []

        track_a_data = df[df['track_id'] == track_a]
        track_b_data = df[df['track_id'] == track_b]

        common_frames = sorted(set(track_a_data['frame']).intersection(track_b_data['frame']))
        interaction_id_local = 0
        i = 0

        while i < len(common_frames):
            frame = common_frames[i]
            point_a = track_a_data[track_a_data['frame'] == frame][['x_body', 'y_body']].to_numpy(dtype=float)
            point_b = track_b_data[track_b_data['frame'] == frame][['x_body', 'y_body']].to_numpy(dtype=float)
            dist = np.linalg.norm(point_a - point_b)

            if dist < proximity_threshold:
                current_interaction = []
                while i < len(common_frames):
                    frame = common_frames[i]
                    point_a = track_a_data[track_a_data['frame'] == frame][['x_body', 'y_body']].to_numpy(dtype=float)
                    point_b = track_b_data[track_b_data['frame'] == frame][['x_body', 'y_body']].to_numpy(dtype=float)
                    dist = np.linalg.norm(point_a - point_b)

                    if dist < proximity_threshold:
                        current_interaction.append((frame, dist))
                        i += 1
                    else:
                        break

                interaction_id_local += 1
                for frame, dist in current_interaction:
                    results.append({
                        'file': track_file,
                        'interaction': interaction_id_local,
                        'frame': frame,
                        'Interaction Pair': (track_a, track_b),
                        'Distance': dist,
                    })
            else:
                i += 1
        return results

    all_data = []
    no_contacts = []

    files = os.listdir(directory)
    pseudo_files = [f for f in files if f.startswith('pseudo_population_') and f.endswith('.csv')]

    for pseudo_track in pseudo_files:
        print(pseudo_track)
        file_path = os.path.join(directory, pseudo_track)
        df = pd.read_csv(file_path)
        df = df.sort_values(by='frame')
        # df = df[df['frame'] < 600]

        track_ids = df['track_id'].unique()
        track_combinations = list(combinations(track_ids, 2))

        all_results = Parallel(n_jobs=-1)(
            delayed(process_track_pair)(track_a, track_b, df, pseudo_track)
            for track_a, track_b in track_combinations
        )

        flattened_results = [item for sublist in all_results for item in sublist]

        if not flattened_results:
            print(f"No contact results for {pseudo_track}")
            no_contacts.append(pseudo_track)
            continue

        results_df = pd.DataFrame(flattened_results)
        results_df.set_index('frame', inplace=True, drop=False)
        all_data.append(results_df)

    if not all_data:
        print("No contacts detected in any file.")
        return None

    interaction_data = pd.concat(all_data, ignore_index=True)
    # Assign global interaction IDs across files and pairs
    interaction_data['Interaction Number'] = (
        interaction_data
        .groupby(['file', 'interaction'])
        .ngroup() + 1  # make it start at 1
    )
    interaction_data.drop(columns=['interaction'], inplace=True)  # Drop the local ID if you don't need it



    durations = (
        interaction_data.groupby("Interaction Number")
        .agg(
            duration_seconds=("frame", "count"),
            file=("file", "first")
        )
    )

    contact_counts = durations.groupby("file").size().reset_index(name="contact_bouts")
    avg_durations = durations.groupby("file")["duration_seconds"].mean().reset_index(name="avg_duration_seconds")

    summary = pd.merge(contact_counts, avg_durations, on="file")

    if no_contacts:
        no_contact_df = pd.DataFrame({
            'file': no_contacts,
            'contact_bouts': 0,
            'avg_duration_seconds': np.nan
        })
        summary = pd.concat([summary, no_contact_df], ignore_index=True)

    summary = summary.sort_values("file")
    summary.to_csv(os.path.join(directory, 'contacts.csv'), index=False)

    return summary




# contact('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated')
# contact('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed')

# pseudo_population_euclidean_distance('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated')
# trajectory('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour/n10-pseudo-population-model')


# time_average_msd('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated', list(range(1, 101, 1)))
# time_average_msd('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed', list(range(1, 101, 1)))
# time_average_msd('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/socially-isolated', list(range(1, 101, 1)))
# time_average_msd('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/group-housed', list(range(1, 101, 1)))

pseudo_population_euclidean_distance('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/socially-isolated')
pseudo_population_euclidean_distance('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/group-housed')
pseudo_population_euclidean_distance('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated')
pseudo_population_euclidean_distance('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed')


distance_from_centre('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/socially-isolated')
distance_from_centre('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/group-housed')
distance_from_centre('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated')
distance_from_centre('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed')


trajectory('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/socially-isolated')
trajectory('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/group-housed')
trajectory('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated')
trajectory('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed')

