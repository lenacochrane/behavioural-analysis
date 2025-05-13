
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
from itertools import combinations
from joblib import Parallel, delayed



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



def contact(directory, proximity_threshold=1):

    def process_track_pair(track_a, track_b, df, track_file, proximity_threshold=1):
        results = []

        track_a_data = df[df['track_id'] == track_a]
        track_b_data = df[df['track_id'] == track_b]

        common_frames = sorted(set(track_a_data['frame']).intersection(track_b_data['frame']))

        if not common_frames:
            return results

        # Precompute node-node distances for all common frames
        parts = ['head', 'body', 'tail']
        distance_rows = []

        for frame in common_frames:
            row_a = track_a_data[track_a_data['frame'] == frame]
            row_b = track_b_data[track_b_data['frame'] == frame]

            if row_a.empty or row_b.empty:
                continue

            positions = {}
            for part in parts:
                positions[f'a_{part}'] = row_a[[f'x_{part}', f'y_{part}']].to_numpy().flatten()
                positions[f'b_{part}'] = row_b[[f'x_{part}', f'y_{part}']].to_numpy().flatten()

            distances = {
                'head_head': np.linalg.norm(positions['a_head'] - positions['b_head']),
                'body_body': np.linalg.norm(positions['a_body'] - positions['b_body']),
                'tail_tail': np.linalg.norm(positions['a_tail'] - positions['b_tail']),
                'head_tail': np.linalg.norm(positions['a_head'] - positions['b_tail']),
                'tail_head': np.linalg.norm(positions['a_tail'] - positions['b_head']),
                'body_head': np.linalg.norm(positions['a_body'] - positions['b_head']),
                'head_body': np.linalg.norm(positions['a_head'] - positions['b_body']),
                'body_tail': np.linalg.norm(positions['a_body'] - positions['b_tail']),
                'tail_body': np.linalg.norm(positions['a_tail'] - positions['b_body']),
            }

            for interaction_type, dist in distances.items():
                distance_rows.append({
                    'frame': frame,
                    'interaction_type': interaction_type,
                    'Distance': dist
                })

        if not distance_rows:
            return results

        # Convert to DataFrame
        dist_df = pd.DataFrame(distance_rows)

        # Get min distance & type per frame
        min_df = dist_df.groupby('frame').apply(
            lambda g: g.loc[g['Distance'].idxmin()]
        ).reset_index(drop=True)

        # Now iterate through min_df and build bouts
        interaction_id_local = 0
        i = 0
        frames = min_df['frame'].values

        while i < len(min_df):
            frame = frames[i]
            dist = min_df.loc[i, 'Distance']
            interaction_type = min_df.loc[i, 'interaction_type']

            if dist < proximity_threshold:
                current_bout = []

                while i < len(min_df):
                    frame = frames[i]
                    dist = min_df.loc[i, 'Distance']
                    interaction_type = min_df.loc[i, 'interaction_type']

                    if dist < proximity_threshold:
                        current_bout.append((frame, dist, interaction_type))
                        i += 1
                    else:
                        break
            else:
                i += 1
                continue

            # Check for frame continuity
            bout_frames = [f for f, _, _ in current_bout]
            if bout_frames[-1] - bout_frames[0] + 1 == len(bout_frames):
                interaction_id_local += 1
                for frame, dist, interaction_type in current_bout:
                    results.append({
                        'file': track_file,
                        'interaction': interaction_id_local,
                        'frame': frame,
                        'Interaction Pair': (track_a, track_b),
                        'Distance': dist,
                        'Interaction Type': interaction_type
                    })

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
            delayed(process_track_pair)(track_a, track_b, df, pseudo_track, proximity_threshold)
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

    for file in no_contacts:
        placeholder = pd.DataFrame([{
            'file': file,
            'interaction': np.nan,
            'frame': np.nan,
            'Interaction Pair': None,
            'Distance': np.nan,
            'Interaction Type': None,
            'Interaction Number': np.nan
        }])
        all_data.append(placeholder)

    interaction_data = pd.concat(all_data, ignore_index=True)

    interaction_data['Interaction Number'] = (
        interaction_data
        .groupby(['file','Interaction Pair', 'interaction'])
        .ngroup() + 1  # make it start at 1
    )
    interaction_data.drop(columns=['interaction'], inplace=True)  # Drop the local ID if you don't need it

    interaction_data = interaction_data.sort_values("file")
    output_filename = f"contacts_{proximity_threshold}mm.csv"
    interaction_data.to_csv(os.path.join(directory, output_filename), index=False)

    return interaction_data





def correlations(directory):

    dfs = []

    files = os.listdir(directory)
    pseudo_files = [f for f in files if f.startswith('pseudo_population_') and f.endswith('.csv')]

    for pseudo_track in pseudo_files:
        file_path = os.path.join(directory, pseudo_track)
        df = pd.read_csv(file_path)
        df = df.sort_values(by='frame', ascending=True)

        def speed(group, x, y):
            dx = group[x].diff()
            dy = group[y].diff()
            distance = np.sqrt(dx**2 + dy**2)
            dt = group['frame'].diff()
            speed = distance / dt.replace(0, np.nan) # Avoid division by zero
            return speed

        df['speed'] = df.groupby('track_id').apply(lambda group: speed(group, 'x_body', 'y_body')).reset_index(level=0, drop=True)
        df['acceleration'] = df.groupby('track_id')['speed'].diff() / df.groupby('track_id')['frame'].diff()
       

        def calculate_angle(df, v1_x, v1_y, v2_x, v2_y):
            dot_product = (df[v1_x] * df[v2_x]) + (df[v1_y] * df[v2_y])
            magnitude_v1 = np.hypot(df[v1_x], df[v1_y])  # Same as sqrt(x^2 + y^2
            magnitude_v2 = np.hypot(df[v2_x], df[v2_y])

            # Avoid division by zero
            cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure values are in valid range for arccos
            
            return np.degrees(np.arccos(cos_theta))  # Convert radians to degrees
        
        df['v1_x'] = df['x_head'] - df['x_body']
        df['v1_y'] = df['y_head'] - df['y_body']
        df['v2_x'] = df['x_tail'] - df['x_body']
        df['v2_y'] = df['y_tail'] - df['y_body']

        # Apply function correctly
        df['angle'] = calculate_angle(df, 'v1_x', 'v1_y', 'v2_x', 'v2_y')

        df['body-body'] = np.nan 

        for frame in df['frame'].unique():
            unique_frame =  df[df['frame'] == frame]
            if len(unique_frame) < 2:
                continue
            body_coordinates = unique_frame[['x_body', 'y_body']].to_numpy()
            distance = cdist(body_coordinates, body_coordinates, 'euclidean')
            np.fill_diagonal(distance, np.nan)

            # unique_frame['body-body'] = np.nanmin(distance, axis=1)
            df.loc[unique_frame.index, 'body-body'] = np.nanmin(distance, axis=1)

        dfs.append(df) 
               
    data = pd.concat(dfs, ignore_index=True)

    data.to_csv(os.path.join(directory, 'correlations.csv'), index=False)




#############################

#### FUNCTIONS TO CHOOSE ####

# contact('', proximity_threshold=)
# time_average_msd('', list(range(1, 101, 1)))
# correlations
# pseudo_population_euclidean_distance
# trajectory
# distance_from_centre

#############################



contact('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/socially-isolated', proximity_threshold=5)
contact('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/group-housed', proximity_threshold=5)
contact('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated', proximity_threshold=5)
contact('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed', proximity_threshold=5)








