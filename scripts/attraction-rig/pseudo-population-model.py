
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

            df = df[df['frame'] < 601]

            for frame in df['frame'].unique():
                unique_frame = df[df['frame'] == frame]

                if unique_frame.empty or unique_frame[['x_body', 'y_body']].isnull().any().any():
                    print(f"Skipping frame {frame} in {pseudo_track} due to missing data.")
                    continue    

                body_coordinates = unique_frame[['x_body', 'y_body']].to_numpy()
                distance = cdist(body_coordinates, body_coordinates, 'euclidean')

                np.fill_diagonal(distance, np.nan)
                average_distance = np.nanmean(distance)

                data.append({'frame': frame, 'average_distance': average_distance, 'file': pseudo_track})

        df = pd.DataFrame(data)
        df = df.sort_values(by=['frame', 'file'], ascending=True)

        output_file = os.path.join(directory, 'euclidean_distances.csv')

        df.to_csv(output_file, index=False)
        return df

#pseudo_population_euclidean_distance('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated')

def time_average_msd(directory, taus):

    dfs = []

    files = os.listdir(directory)
    pseudo_files = [f for f in files if f.startswith('pseudo_population_') and f.endswith('.csv')]

    for pseudo_track in pseudo_files:
        file_path = os.path.join(directory, pseudo_track)
        df = pd.read_csv(file_path)
        df.sort_values(by='frame', ascending=True)
        # df['file'] = pseudo_track

        df = df[df['frame'] < 601]

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


#time_average_msd('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated', list(range(1, 101, 1)))


def ensemble_msd(directory):

    data = []

    files = os.listdir(directory)
    pseudo_files = [f for f in files if f.startswith('pseudo_population_') and f.endswith('.csv')]

    for pseudo_track in pseudo_files:
        file_path = os.path.join(directory, pseudo_track)
        df = pd.read_csv(file_path)
        df.sort_values(by='frame', ascending=True)

        df = df[df['frame'] < 601]

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


#ensemble_msd('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour/n2-pseudo-population-model')


def speed(directory):

    speed = []
    data = []
    
    files = os.listdir(directory)
    pseudo_files = [f for f in files if f.startswith('pseudo_population_') and f.endswith('.csv')]

    for pseudo_track in pseudo_files:
        file_path = os.path.join(directory, pseudo_track)
        df = pd.read_csv(file_path)
        df.sort_values(by='frame', ascending=True)

        df = df[df['frame'] < 601]

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
       
        speed_values = pd.DataFrame(speed)
        speed_values.to_csv(os.path.join(directory, 'speed_values.csv'), index=False)

        speed_over_time = pd.DataFrame(data)
        speed_over_time = speed_over_time.sort_values(by=['time'], ascending=True)
        speed_over_time.to_csv(os.path.join(directory, 'speed_over_time.csv'), index=False)

        return speed_values, speed_over_time
    

#speed('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated')


def distance_from_centre(directory): 

    data = []

    files = os.listdir(directory)
    pseudo_files = [f for f in files if f.startswith('pseudo_population_') and f.endswith('.csv')]

    for pseudo_track in pseudo_files:
        file_path = os.path.join(directory, pseudo_track)
        df = pd.read_csv(file_path)
        df.sort_values(by='frame', ascending=True)

        df = df[df['frame'] < 601]

        centre_x = 0
        centre_y = 0

        for index, row in df.iterrows():
            x, y = row['x_body'], row['y_body']
            distance = np.sqrt((centre_x - x)**2 + (centre_y - y)**2)
            
            data.append({'file': row['filename'], 'track': row['track_id'], 'frame': row['frame'], 'distance_from_centre': distance})
        
    df_distance_over_time = pd.DataFrame(data)
    df_distance_over_time.to_csv(os.path.join(directory, 'distance_over_time.csv'), index=False)

    return df_distance_over_time

distance_from_centre('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated')
distance_from_centre('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed')

def interaction_types(directory,threshold=1):
    data = []

    files = os.listdir(directory)
    pseudo_files = [f for f in files if f.startswith('pseudo_population_') and f.endswith('.csv')]

    for pseudo_track in pseudo_files:
        file_path = os.path.join(directory, pseudo_track)
        df = pd.read_csv(file_path)
        df.sort_values(by='frame', ascending=True)
        df = df[df['frame'] < 601]
        # track_ids = df['track_id'].unique()
        
        # Prepare to count interactions per file
        interaction_counts = {
            'head_head': 0,
            'tail_tail': 0,
            'body_body': 0,
            'head_tail': 0,
            'body_head': 0,
            'body_tail': 0,
            'file': pseudo_track,
        }
        
        for frame in df['frame'].unique():
            print(frame)
            frame_data = df[df['frame'] == frame]

            if len(frame_data) < 2:
                # Skip frames with fewer than two tracks
                continue
            
            # Create arrays for each body part including track ID for identification
            head_positions = frame_data[['track_id', 'x_head', 'y_head']].to_numpy()
            body_positions = frame_data[['track_id', 'x_body', 'y_body']].to_numpy()
            tail_positions = frame_data[['track_id', 'x_tail', 'y_tail']].to_numpy()
            
            # Calculate distances and find minimum interaction
            for interaction_type, (positions1, positions2) in {
                'head_head': (head_positions, head_positions),
                'tail_tail': (tail_positions, tail_positions),
                'body_body': (body_positions, body_positions),
                'head_tail': (head_positions, tail_positions),
                'body_head': (body_positions, head_positions),
                'body_tail': (body_positions, tail_positions),
            }.items():
                # Exclude same animal interaction by track_id
                mask = positions1[:, 0][:, None] != positions2[:, 0]
                distances = cdist(positions1[:, 1:], positions2[:, 1:])
                distances[~mask] = np.inf  # Invalidate same animal distances
                
                min_distance = np.min(distances)
                if min_distance < threshold:
                    interaction_counts[interaction_type] += 1
        
        data.append(interaction_counts)
    
    interaction_df = pd.DataFrame(data)
    melted_df = interaction_df.melt(id_vars='file', var_name='interaction_type', value_name='count').sort_values(by='file')
    melted_df.to_csv(os.path.join(directory, 'interaction_types.csv'), index=False)


#interaction_types('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated')


def trajectory(directory):

    dfs = []

    files = os.listdir(directory)
    pseudo_files = [f for f in files if f.startswith('pseudo_population_') and f.endswith('.csv')]

    for pseudo_track in pseudo_files:
        file_path = os.path.join(directory, pseudo_track)
        df = pd.read_csv(file_path)
        df.sort_values(by='frame', ascending=True)
        df = df[df['frame'] < 601]
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



#trajectory('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated')
# trajectory('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/10-minute-agarose-behaviour/n10-pseudo-population-model')

 