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


class HoleAnalysis:

    def __init__(self, directory):

        self.directory = directory 
        self.coordinate_files = []
        self.track_files = [] # list of the files 
        self.hole_boundaries = []
        self.matching_pairs = []
        self.track_data = {}  # Initialize the track_data dictionary # actually has the data so we dont have to keep reloading 
        self.coordinates()
        self.tracks()
        self.hole_boundary()
        self.match_files()

    # METHOD COORDINATES: IDENTIFIES AND STORES THE HOLE COORDINATE FILES

    def coordinates(self):
        # 2024-05-20_16-08-22_td1_hole.csv
        self.coordinate_files = [f for f in os.listdir(self.directory) if f.endswith('hole.csv')]
        print(f"Coordinate files: {self.coordinate_files}")

    # METHOD TRACKS: IDENTIES AND STORES THE SLEAP TRACK FILES; TRACK DATA IS SUBSEQUENTLY READ  

    def tracks(self):
        # 2024-04-30_14-31-44_td5.000_2024-04-30_14-31-44_td5.analysis.csv
        self.track_files = [f for f in os.listdir(self.directory) if f.endswith('tracks.feather')]
        print(f"Track files: {self.track_files}")
        # load the data 
        for track_file in self.track_files: 
            track_path = os.path.join(self.directory, track_file)
            df = pd.read_feather(track_path)
            # Print to verify data right after loading
            # print(f"Loaded data from {track_file}:")
            # print(df[['frame', 'track_id', 'x_body', 'y_body']].head(10))
            pixels_to_mm = ['x_tail', 'y_tail', 'x_body', 'y_body', 'x_head', 'y_head']
            df[pixels_to_mm] = df[pixels_to_mm] * (90/1032) #conversion factor pixels to mm
            # df = df.round(2)  # Round values to 2 decimal places
            self.track_data[track_file] = df
        
            # self.track_data[track_file] = pd.read_feather(track_path)

   # METHOD SHORTEN: OPTIONAL METHOD TO SHORTEN THE TRACK FILES TO INCLUDE UP TO A CERTAIN FRAME  
    
    def shorten(self, frame=-1):

        for track_file in self.track_files:

            df = self.track_data[track_file]
            df = df[df['frame'] <= frame]
            self.track_data[track_file] = df # update the track data 

            # create path 
            shortened_path = os.path.join(self.directory, track_file.replace('.feather', f'_shortened_{frame}.feather'))
            # save 
            df.reset_index(drop=True, inplace=True)  # Feather requires a default integer index
            df.to_feather(shortened_path)  # Save the DataFrame without 'index=False'
            print(f"Shortened file saved: {shortened_path}")

    # METHOD POST_PROCESSING: 1) FILTERS TRACK'S AVERAGE INSTANCE SCORE < 0.9 2) 

    def post_processing(self):
        
        # FUNCTION INPUTS INDIVIDUAL TRACK DF DATA AND INCRIMENTALLY FILLS IN GAPS 
        def interpolate(track_df):
            # range of frames
            min_frame = track_df['frame'].min() 
            max_frame = track_df['frame'].max()
            # create Numpy Array of min-max 
            frame_range = np.arange(min_frame, max_frame + 1)
            # return difference between expected and actual frame numbers
            missing_frames = np.setdiff1d(frame_range, track_df['frame'].values)
  
            if len(missing_frames) == 0:
                return track_df
    
            track_name = track_df['track_id'].iloc[0]
            # create df for missing frames
            missing_df = pd.DataFrame({'frame': missing_frames, 'track_id': track_name})
            # join track data and missing tracks 
            df = pd.concat([track_df, missing_df]).sort_values(by='frame')

            # Interpolate for each coordinate pair
            coordinates = ['x_head', 'y_head', 'x_body', 'y_body', 'x_tail', 'y_tail']
            # add nan values for the missing data in the additional frames 
            for coord in coordinates:
                if coord not in df.columns:
                    df[coord] = np.nan 
            
            for coord in coordinates:
                # interpolate fills in missing values assuming a linear relation between known values
                df[coord] = df[coord].interpolate()
    
            # Forward-fill and backward-fill (dont think this is applicable here really- gaps at start and end of track)
            # for coord in coordinates:
            #     full_df[coord] = full_df[coord].ffill().bfill()
            return df
        

        for track_file in self.track_files:
            df = self.track_data[track_file]
            # unsure if in the feather file has the instance score -> get michael to include 
            # group by tracks, calculate mean per tracks, if True >= 0.9 include in df 
            df = df[df.groupby('track_id')['instance_score'].transform('mean') >= 0.9]

            # fill in track gaps 
            df = df.sort_values(by=['track_id', 'frame'])
            # applies the definition to each mini dataframe for tracks and then combines the results into a single dataframe
            df = df.groupby('track_id').apply(interpolate).reset_index(drop=True)

            # # Save the post-processed DataFrame to the original file path
            # file_path = os.path.join(self.directory, track_file)  # Combines directory and filename
            # df.to_feather(file_path)  # Save the DataFrame back to the file
            self.track_data[track_file] = df  # Update the in-memory version


    # METHOD HOLE_BOUNDARY: CREATES A POLYGON AROUND THE HOLE BOUNDARY WITH SCALAR OPTION
     # 1. CONVEX HULL: CONVEX SHAPE THAT ENCLOSES A SET OF POINTS (CONTINIOUS BOUNDARY)
     # 2. VERTICES: CORNER POINTS OF THE CONVEX SHAPE
     # 3. POLYGON: GEOMETRIC SHAPE FORMED BY CONNECTING THESE VERTICES

    def hole_boundary(self, scale_factor=1.0):  

        self.hole_boundaries = []

        for coordinates in self.coordinate_files:

            file_path = os.path.join(self.directory, coordinates)

            df = pd.read_csv(file_path, header=None, names=['x', 'y'])

            # Convert coordinates from pixels to millimeters using the same conversion factor
            conversion_factor = 90 / 1032
            df[['x', 'y']] = df[['x', 'y']] * conversion_factor

            points = df[['x', 'y']].values # values creates numpy array

            hull = ConvexHull(points)  

            # struggle with this understanding - ask callum 
            # this retrieves the points of the shapes 
            # defines the boundary points
            hull_points = points[hull.vertices]

            # create the polygon
            polygon = Polygon(hull_points)

            scaled_polygon = scale(polygon, xfact=scale_factor, yfact=scale_factor, origin='center') # polygon scaled uniform relative to center

            self.hole_boundaries.append(scaled_polygon)

            wkt_string = wkt_dumps(scaled_polygon) # Convert the scaled polygon to WKT format

            # Save the WKT string to a file with the same name as the original but with a .wkt extension
            # hole_boundary = coordinates.replace('.csv', '.wkt')
            hole_boundary = os.path.join(self.directory, coordinates.replace('.csv', '.wkt'))

            with open(hole_boundary, 'w') as f:
                f.write(wkt_string)
        
        print(f"Hole boundaries: {self.hole_boundaries}")
    
    
    # METHOD MATCH_FILES: MATCHES THE TRACK FILES WITH THEIR COORDINATE FILES (BY EXTENTION THE HOLE POLYGON)

    def match_files(self):

        for track_file in self.track_files:
            # split: splites the file name whenever there is an underscore
            # join the first three elements :3 with an underscore
            # 2024-05-20_16-08-22_td1_hole.csv -> 2024-05-20 16-08-22 td1 hole.csv -> 2024-05-20_16-08-22_td1
            track_prefix = '_'.join(track_file.split('_')[:3])
            track_prefix = track_prefix.replace('.tracks.feather', '')
            # track_prefix = track_prefix.rsplit('.', 1)[0]  # Remove the extension
            print(f"Track file: {track_file}, Prefix: {track_prefix}")

            # for hole_boundary in self.hole_boundaries:
            #     hole_prefix = '_'.join(hole_boundary.split('_')[:3])

            for i, coordinates_file in enumerate(self.coordinate_files):
                hole_prefix = '_'.join(coordinates_file.split('_')[:3])
                hole_prefix = hole_prefix.rsplit('.', 1)[0]  # Remove the extension
                print(f"Coordinate file: {coordinates_file}, Prefix: {hole_prefix}")

                if hole_prefix == track_prefix:
                    print(f"Match found: {track_file} with {coordinates_file}")
                    self.matching_pairs.append((track_file, self.hole_boundaries[i]))

                # if hole_prefix == track_prefix:
                #     self.matching_pairs.append((track_file, hole_boundary))

                # track_path = os.path.join(self.directory, track_file)
                # self.track_data[track_file] = pd.read_feather(track_path).round(2)
        
        print(f"Matching pairs: {self.matching_pairs}")
        print(f"Track data keys: {self.track_data.keys()}")


    # METHOD HOLE_CENTROID: REPLACE THE HOLE BOUNDARY WITH A HOLE CENTROID COORDINATE 

    def hole_centroid(self):

        updated_matching_pairs = [] # update matching pairs with funderals 

        for track_file, hole_boundary in self.matching_pairs:

            centroid = hole_boundary.centroid  # Calculate centroid of the polygon

            updated_matching_pairs.append((track_file, (centroid.x, centroid.y))) # centroid is a tuple
        
        self.matching_pairs = updated_matching_pairs
        print(f"Matching pairs with centroids: {self.matching_pairs}")
        return self.matching_pairs
    
    # METHOD DISTANCE_FROM_HOLE: CALCULATES DISTANCES FROM HOLE CENTROID 

    def distance_from_hole(self): 

        self.hole_centroid() # call the hole_centroid method 

        distances_from_hole = []
        data = []

        for track_file, centroid in self.matching_pairs: # track file is just the name of the file 

            df = self.track_data[track_file] # retreieve the data for that file name 
            for index, row in df.iterrows():
                x, y = row['x_body'], row['y_body']
                distance = np.sqrt((centroid[0] - x)**2 + (centroid[1] - y)**2)
                distances_from_hole.append(distance)
                data.append({'time': row.frame, 'distance_from_hole': distance, 'file': track_file})
        
        print("Distances from hole centroid:", distances_from_hole)

        if not distances_from_hole:
            print("No distances calculated, check data")
        else:

            df_distances = pd.DataFrame(distances_from_hole, columns=['Distance from hole'])
            df_distances.to_csv(os.path.join(self.directory, 'distance_from_hole_centroid.csv'), index=False)
            print(f"Distance from hole saved: {df_distances}")

            distance_hole_over_time = pd.DataFrame(data)
            distance_hole_over_time = distance_hole_over_time.sort_values(by=['time'], ascending=True)
            distance_hole_over_time.to_csv(os.path.join(self.directory, 'distance_hole_over_time.csv'), index=False)

            return df_distances
    
    # METHOD DISTANCE_FROM_CENTRE: CALCULATES DISTANCES FROM CENTRE COORDINATES 

    def distance_from_centre(self): 

        centre = (700, 700) 

        distances_from_centre = []

        for track_file in self.track_files:
            
            predictions = self.track_data[track_file]

            for index, row in predictions.iterrows():
                x, y = row['x_body'], row['y_body']
                distance = np.sqrt((centre[0] - x)**2 + (centre[1] - y)**2)
                distances_from_centre.append(distance)

        df_distances = pd.DataFrame(distances_from_centre, columns=['Distance from centre'])
        df_distances.to_csv(os.path.join(self.directory, 'distances_from_centre.csv'), index=False)
        print(f"Distance from centre saved: {df_distances}")

        return df_distances

    # METHOD EUCLIDEAN_DISTANCE: CALCULATES THE AVERAGE DISTANCE BETWEEN LARVAE ACCROSS FRAMES

    def euclidean_distance(self):

        data = []

        for track_file in self.track_files:
            track_data = self.track_data[track_file]

            # files should already have been joined and read above 

            # iterate over each unique frame in the file and calculate the average distance between the larvae  
            for frame in track_data['frame'].unique():

                unique_frame =  track_data[track_data['frame'] == frame]

                # cdist function requires two 2-dimensional array-like objects as inputs
                # create an array of the coordinates for that specific frame
                    
                body_coordinates = unique_frame[['x_body', 'y_body']].to_numpy()

                # The cdist function computes the distance between every pair of points in the two arrays passed to it.

                distance = cdist(body_coordinates, body_coordinates, 'euclidean')

                np.fill_diagonal(distance, np.nan)

                average_distance = np.nanmean(distance)

                data.append({'time': frame, 'average_distance': average_distance, 'file': track_file})

        df = pd.DataFrame(data)
        df = df.sort_values(by=['time', 'file'], ascending=True)

        df.to_csv(os.path.join(self.directory, 'euclidean_distances.csv'), index=False)
        print(f"Euclidean distance saved: {df}")
        return df
    

    # METHOD EUCLIDEAN DISTANCE VARIANCE: TO CALCULATE THE VARIANCE IN THE PLATEU OF THE EUCLIDEAN_DISTANCE DATA

    def euclidean_distance_variance(self, first_frame, last_frame):

        euclidean_df = self.euclidean_distance() # call the euclidean distance
        print('euclidean distance imported')

        euclidean_df = euclidean_df[(euclidean_df['time'] >= first_frame) & (euclidean_df['time'] <= last_frame)]

        distance_variance = euclidean_df.groupby('file')['average_distance'].var()
        # print(distance_variance)

        distance_variance_df = distance_variance.reset_index()
        distance_variance_df.columns = ['file', 'variance']

        # print(distance_variance_df.head())
        # print("DataFrame before saving:\n", distance_variance_df.info())

        distance_variance_df.to_csv(os.path.join(self.directory, 'average_distance_variance.csv'), index=False)
        return distance_variance

    
    # METHOD PROBABILITY_DENSITY: PROBABILITY DENSITY FOR INPUTTED 1D ARRAY
    
    @staticmethod
    def probability_density(df, ax=None, color=None, label=None, linestyle=None):

        data = df.iloc[:, 0].values #iloc indexed based selection - : all rows - 0 first column

        # Replace infinite values with NaN and drop them
        data = pd.Series(data).replace([np.inf, -np.inf], np.nan).dropna().values

        kde = gaussian_kde(data) #kernel estimate density function for the kde data 

        value_range = np.linspace(data.min(), data.max(), 100)
         # this generates a range of values over which the KDE will be evaluated.
        # np.linspace(start, stop, num)

        density = kde(value_range)
   
        # evaluates the KDE at each of the points in the range provided 

        if ax is None:
            ax = plt.gca()
        
        ax = sns.lineplot(x=value_range, y=density, ax=ax, color=color, label=label, linestyle=linestyle)

        return ax # ax = probability_density() to modify graph when called
    
    # METHOD SPEED: CALCULATES SPEED: 1) SPEED VALUES 2) SPEED OVER TIME 

    def speed(self):

        speed = []
        data = []

        for track_file in self.track_files:
            track_data = self.track_data[track_file]

            for track in track_data['track_id'].unique():
                track_unique = track_data[track_data['track_id'] == track]

                for i in range(len(track_unique) - 1):

                    row = track_unique.iloc[i]
                    next_row = track_unique.iloc[i+1]

                    distance = np.sqrt((row['x_body'] - next_row['x_body'])**2 + (row['y_body'] - next_row['y_body'])**2)

                    time1 = row['frame']
                    time2 = next_row['frame']

                    time = time2 - time1

                    speed_value = distance / time 

                    speed.append(speed_value)

                    data.append({'time': time2, 'speed': speed_value, 'file': track_file})
       
        speed_values = pd.DataFrame(speed)
        speed_values.to_csv(os.path.join(self.directory, 'speed_values.csv'), index=False)

        speed_over_time = pd.DataFrame(data)
        speed_over_time = speed_over_time.sort_values(by=['time'], ascending=True)
        speed_over_time.to_csv(os.path.join(self.directory, 'speed_over_time.csv'), index=False)

        return speed_values, speed_over_time

    # DEF ACCELERATION: 

    def acceleration(self):

        acceleration = []
        data = []

        for track_file in self.track_files:
            track_data = self.track_data[track_file]

            for track in track_data['track_id'].unique():
                track_unique = track_data[track_data['track_id'] == track]

                previous_speed = None
                previous_time = None

                for i in range(len(track_unique) - 1):

                    row = track_unique.iloc[i]
                    next_row = track_unique.iloc[i+1]

                    distance = np.sqrt((row['x_body'] - next_row['x_body'])**2 + (row['y_body'] - next_row['y_body'])**2)

                    time1 = row['frame']
                    time2 = next_row['frame']

                    time = time2 - time1

                    speed_value = distance / time 

                    if previous_speed is not None and previous_time is not None:
                        acceleration_value = (speed_value - previous_speed) / time 
                        acceleration.append(acceleration_value)
                        data.append({'time': time2, 'acceleration': acceleration_value, 'file': track_file})

                    previous_speed = speed_value
                    previous_time = time
        
        acceleration = pd.DataFrame(acceleration)
        acceleration.to_csv(os.path.join(self.directory, 'acceleration.csv'), index=False)

        acceleration_accross_time = pd.DataFrame(data)
        acceleration_accross_time = acceleration_accross_time.sort_values(by=['time'], ascending=True)
        acceleration_accross_time.to_csv(os.path.join(self.directory, 'acceleration_accross_time.csv'), index=False)

        return acceleration, acceleration_accross_time


    # METHOD ENSEMBLE_MSD: CALCULATES SQUARED DISTANCE FOR EVERY POSITION FROM FIRST TRACK APPEARANCE
     
    def ensemble_msd(self): 

        # frame distance and file 

        data = []
        
        for track_file in self.track_files:
            track_data = self.track_data[track_file]

            # calculate average x,y for first frame 
            # (needs to change such that it time 0 for each unique track compared back to)

            for track in track_data['track_id'].unique():
                track_unique = track_data[track_data['track_id'] == track].sort_values(by=['frame'])

                x0 = track_unique.iloc[0]['x_body']
                y0 = track_unique.iloc[0]['y_body']

                for i in range(len(track_unique)):

                    squared_distance = (track_unique.iloc[i]['x_body'] - x0)**2 + (track_unique.iloc[i]['y_body'] - y0)**2
                    # print(squared_distance)

                    frame = track_unique.iloc[i]['frame']

                    data.append({'time': frame, 'squared distance': squared_distance, 'file': track_file})
                
            # frame_0 = track_data[track_data['frame'] == 0]
            # x = frame_0['x_body'].mean()
            # y = frame_0['y_body'].mean()
            # for i, row in track_data.iterrows():
            #     squared_distance = (row['x_body'] - x)**2 + (row['y_body'] - y)**2
            #     frame = row['frame']
            #     data.append({'time': frame, 'squared distance': squared_distance, 'file': track_file})
        
        df = pd.DataFrame(data)
        df = df.sort_values(by=['time'], ascending=True)

        df.to_csv(os.path.join(self.directory, 'ensemble_msd.csv'), index=False)

        return df 

    # METHOD TIME_AVERAGE_MSD: 
      # taus given in list format e.g. list(range(1, 101, 1))

    def time_average_msd(self, taus):

        dfs = []

        # Iterate over track_data dictionary {'filename': dataframe}
        for filename, dataframe in self.track_data.items():
            # Add a new column to the dataframe with the filename
            dataframe['file'] = filename
            dfs.append(dataframe)

        # Concatenate the dataframes 
        df = pd.concat(dfs, ignore_index=True)

        df = df[["file", "track_id", "frame", "x_body", "y_body"]] # chose specific parts of the dataframe
 
        # one value per tau 
        def msd_per_tau(df, tau):

            squared_displacements = []

            grouped_data = df.groupby(['file', 'track_id'])

            # really dont get why you have to iterate in such a way ????
            for (file, track_id), unique_track in grouped_data:

                unique_track = unique_track.sort_values(by='frame').reset_index(drop=True)

                if len(unique_track) > tau:

                    initial_positions = unique_track[['x_body', 'y_body']].values[:-tau] # values up till tau as a NumPy array # positions from t to t-N-tau # represent starting points
                    tau_positions = unique_track[['x_body', 'y_body']].values[tau:] # values from tau onwards # t+tau to t-N # representing ending points 
                    disp = np.sum((tau_positions - initial_positions) ** 2, axis=1) # squared displacement for each pair
                    # # print(disp) 
                    # print(f"disp for tau={tau}: {disp}")
                    # print(type(disp))

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
        tau_msd_df.to_csv(os.path.join(self.directory, 'time_average_msd.csv'), index=False)
   
        return tau_msd_df #.dropna()
    
    # METHOD TRAJECTORY: CALCULATES TRAJECTORY ANGLES: 1) TRAJECTORY ANGLE VALUES 2) TRAJECTORY ANGLE OVER TIME 
      # ANGLE INBETWEEN 2 VECTORS: TAIL-BODY AND BODY-HEAD 

    def trajectory(self):

        dfs = []
        # Iterate over track_data dictionary {'filename': dataframe}
        for filename, dataframe in self.track_data.items():
            # Add a new column to the dataframe with the filename
            dataframe['file'] = filename
            dfs.append(dataframe)

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
        
        angle_values = pd.DataFrame(angles)
        angle_values.to_csv(os.path.join(self.directory, 'angle_values.csv'), index=False)

        angle_over_time = pd.DataFrame(data)
        angle_over_time = angle_over_time.sort_values(by=['time'], ascending=True)
        angle_over_time.to_csv(os.path.join(self.directory, 'angle_over_time.csv'), index=False)

        return angle_values, angle_over_time   

    # METHOD PROXIMITY: CALCULATES NUMBER OF 'PROXIMAL ENCOUNTERS' 1) AVERAGE  2) AVERAGE OVER TIME
      # CURRENTLY ACCOUNTS ONLY FOR BODY-BODY CONTACTS 

    def proximity(self, pixel=(50*(90/1032))):

        count = []
        data = []

        for track_file in self.track_files:
            track_data = self.track_data[track_file]
            track_data.to_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/plain-petri/test2.csv')

            # count for each frame 
            for frame in track_data['frame'].unique():
                unique_frame = track_data[track_data['frame'] == frame]

                body_coordinates = unique_frame[['x_body', 'y_body']].to_numpy() # cdist requires 2D NumPy array as input

                euclidean_distance = cdist(body_coordinates, body_coordinates, 'euclidean')

                proximity = (euclidean_distance < pixel) & (euclidean_distance > 0) # above 0 so track1 != track1 proximity 

                proximal_counters = np.sum(proximity)

                count.append(proximal_counters)
                data.append({'time': frame, 'proximity count': proximal_counters, 'file': track_file})

        
        proximal_counters = pd.DataFrame(count)
        proximal_counters.to_csv(os.path.join(self.directory, 'proximal_counters.csv'), index=False)

        proximity_over_time = pd.DataFrame(data)
        proximity_over_time = proximity_over_time.sort_values(by=['time'], ascending=True)
        proximity_over_time.to_csv(os.path.join(self.directory, 'proximity_over_time.csv'), index=False)

        return proximal_counters, proximity_over_time
    
    # DEF HOLE_COUNTER: COUNTS NUMBER OF LARVAE IN THE HOLE 

    # def hole_counter(self):

    #     count = []

    #     for track_file, hole_boundary in self.matching_pairs:
    #         df = self.track_data[track_file]

    #         # ASSUME NO LARVAE LEAVE - COUNT THE NUMBER OF LARVAE OUTSIDE HOLE BECAUSE INSIDE HOLE MIGHT BE HARD IF THEY ARE DIGGING A LOT 
    #         # TRACK ID IS IRRELEVANT SO CAN ITERATE OVER EVERY ROW 
    #         # REQUIRES FRAME 

    #         for frame, frame_data in df.groupby('frame'):

    #             # point is from the shapely library and is used to represent a point in 2d space 
    #             # contains to see if the point lies within the polygon 
    #             number_outside_hole = frame_data.apply(lambda row: not hole_boundary.contains(Point(row['x_body'], row['y_body'])), axis=1)

    #             count_outside = number_outside_hole.sum()
    #             count_inside = 10 - count_outside # assume 10 larvae

    #             count.append({'time': frame, 'count': count_inside, 'file': track_file})
        
    #     hole_count = pd.DataFrame(count)
    #     hole_count = hole_count.sort_values(by=['time'], ascending=True)
    #     hole_count.to_csv(os.path.join(self.directory, "hole_count.csv"), index=False)

    #     return hole_count
    
    
    # METHOD HOLE_COUNTER: COUNTS NUMBER OF LARVAE IN THE HOLE + IF LARVAE OUTSIDE HOLE IS IT MOVING (IDENTIFYING THOSE CLOSE AND BURROWING BENEATH SURFACE)

    # def hole_counter(self):

    #     count = []

    #     for track_file, hole_boundary in self.matching_pairs:
    #         df = self.track_data[track_file]

    #         # True/False is the point inside or outside the hole 
    #         df['outside_hole'] = df.apply(lambda row: not hole_boundary.contains(Point(row['x_body'], row['y_body'])), axis=1)
            
    #         # True/False within 10mm of the hole
    #         df['within_10mm'] = df.apply(lambda row: hole_boundary.exterior.distance(Point(row['x_body'], row['y_body'])) <= 10, axis=1)

    #         # calculate displacement between frames
    #         df['displacement'] = df.groupby('track_id').apply(lambda group: np.sqrt((group['x_body'].diff() ** 2) + (group['y_body'].diff() ** 2))).reset_index(drop=True)

    #         # cumulative distance over rolling window
    #         df['rolling_displacement'] = df.groupby('track_id')['displacement'].transform(lambda x: x.rolling(window=30, min_periods=1).sum()) 

    #         for frame, frame_data in df.groupby('frame'):

    #             outside_count = 0

    #             for i, row in frame_data.iterrows():

    #                 if row['outside_hole']:
    #                     # If it's within 10mm and not moving, skip it (ROLLING DISPLACEMENT BELOW)
    #                     if row['within_10mm'] and row['rolling_displacement'] < 3:
    #                         continue
    #                     else:
    #                         # If it's genuinely outside, increment the counter
    #                         outside_count += 1
                
    #             inside_count = 10 - outside_count 

    #             # count.append({'time': frame, 'count': inside_count, 'outside':outside_count , 'file': track_file})
    #             count.append({'time': frame, 'inside_count': inside_count, 'outside':outside_count , 'file': track_file})
        
    #     hole_count = pd.DataFrame(count)
    #     hole_count = hole_count.sort_values(by=['time'], ascending=True)
    #     hole_count.to_csv(os.path.join(self.directory, "hole_count.csv"), index=False)

    #     return hole_count


    # METHOD HOLE_COUNTER: COUNTS NUMBER OF LARVAE IN THE HOLE + IF LARVAE OUTSIDE HOLE IS IT MOVING (IDENTIFYING THOSE CLOSE AND BURROWING BENEATH SURFACE)
    # ALSO CHANGED IT SUCH THAT IT KEEPS THE ORIGINAL DATAFRAME SO I CAN DO A VIDEO CHECK 

    def hole_counter(self):

        count = []

        for track_file, hole_boundary in self.matching_pairs:
            df = self.track_data[track_file]

            # True/False is the point inside or outside the hole 
            df['outside_hole'] = df.apply(lambda row: not hole_boundary.contains(Point(row['x_body'], row['y_body'])), axis=1)
            
            # True/False within 10mm of the hole
            df['within_10mm'] = df.apply(lambda row: hole_boundary.exterior.distance(Point(row['x_body'], row['y_body'])) <= 10, axis=1)

            # calculate displacement between frames
            # chat gpt said resent index is necessary here due to the apply and groupby * ask michael or callum to explain this properly
            df['displacement'] = df.groupby('track_id').apply(lambda group: np.sqrt((group['x_body'].diff() ** 2) + (group['y_body'].diff() ** 2))).reset_index(drop=True)

            # cumulative distance over rolling window
            df['rolling_displacement'] = df.groupby('track_id')['displacement'].transform(lambda x: x.rolling(window=30, min_periods=1).sum()) 

            df['digging'] = df['within_10mm'] & (df['rolling_displacement'] < 3)

            df['moving_outside'] = df['outside_hole'] & ~df['digging']

            df.to_csv(os.path.join(self.directory, f"{track_file}_hole_data.csv"), index=False)

            for frame in df['frame'].unique():
                frame_df = df[df['frame'] == frame]

                outside_count = frame_df['moving_outside'].sum()
                inside_count = 10 - outside_count
                count.append({'time': frame, 'inside_count': inside_count, 'outside':outside_count , 'file': track_file})
        
        hole_count = pd.DataFrame(count)
        hole_count = hole_count.sort_values(by=['time'], ascending=True)
        hole_count.to_csv(os.path.join(self.directory, "hole_count.csv"), index=False)

        return hole_count
    
    # METHOD TIME_TO_ENTER: TIME TAKEN FOR EACH TRACK TO ENTER THE HOLE
      # ACCOUNT ONLY FOR THE TRACKS GENERATED IN THE FIRST 30 FRAMES (TRACKS GO MISSING ONCE IN HOLE AND REGENERATE NEW ONES WHICH WE DONT CARE ABOUT)

    def time_to_enter(self):

        times = []

        for track_file, hole_boundary in self.matching_pairs:
            df = self.track_data[track_file]
      
            for track in df['track_id'].unique():
                unique_track = df[df['track_id'] == track]
                unique_track = unique_track.sort_values(by=['frame'], ascending=True)

                # Check if the track appears for the first time after frame 30
                first_frame = unique_track['frame'].iloc[0]
                if first_frame > 30:
                    continue  # Skip tracks that appear after frame 30

                entered = False

                # frame at which it enters the hole 
                for row in unique_track.itertuples():
      
                    point = Point(row.x_body, row.y_body) # Create a Point object for each (x_body, y_body) pair
     
                    if hole_boundary.contains(point) or hole_boundary.touches(point):
                        print(row.frame)
                        times.append({'track': track, 'time': row.frame, 'file': track_file})
                        entered = True
                        break

                if not entered:
                    times.append({'track': track, 'time': np.nan, 'file': track_file})
        
        hole_entry_time = pd.DataFrame(times)
        hole_entry_time = hole_entry_time.sort_values(by=['file'], ascending=True)
        hole_entry_time.to_csv(os.path.join(self.directory, 'hole_entry_time.csv'), index=False)

        return hole_entry_time


    # DEF RETURNS: CALCULATES THE NUMBER OF LARVAE WHICH RETURN TO THE HOLE AND THE TIME TAKEN 

    def returns(self):

        data = []

        for track_file, hole_boundary in self.matching_pairs:
            df = self.track_data[track_file]

            df['point'] = df.apply(lambda row: Point(row.x_body, row.y_body), axis=1)

            for track in df['track_id'].unique():
                unique_track = df[df['track_id'] == track].sort_values(by=['frame'], ascending=True)

                # identify inside or touching 2d points - Boolean True if so 
                unique_track['potential point'] = unique_track['point'].apply(lambda row: hole_boundary.contains(row) or hole_boundary.touches(row))

                # shifts the rows up by one, such that if the following row were to contain True/False we would know 
                unique_track['following point'] = unique_track['potential point'].shift(-1)

                exit_frame = None

                # Identify rows which were within/touching the hole (potential point: True) and have now left the hole (following point: False)
                for i, row in unique_track.iterrows():
                    
                    if row['potential point'] and not row['following point']: # identify tracks which have left the hole boundary
                        exit_frame = row['frame']
                        continue 
                         
                    if exit_frame is not None:
                        if row['potential point']: # the track has reentered
                            return_frame = row['frame']
                            print(exit_frame, return_frame)
                            time_taken = return_frame - exit_frame
      
                            data.append({'track': track, 'return time': time_taken, 'exit frame': exit_frame, 'return frame': return_frame, 'file': track_file})
        
                            exit_frame = None
        
        returns = pd.DataFrame(data)
        print(returns.head())
        returns = returns.sort_values(by=['track'], ascending=True)
        returns.to_csv(os.path.join(self.directory, 'returns.csv'), index=False)

        return returns
    

    # METHOD HOLE_DEPARTURES: METHOD WHICH CALCULATES THE NUMBER OF LARVAE LEAVING THE HOLE

    def hole_departures(self):

        data = []

        for track_file, hole_boundary in self.matching_pairs:
            df = self.track_data[track_file]

            df['point'] = df.apply(lambda row: Point(row.x_body, row.y_body), axis=1)

            for track in df['track_id'].unique():
                unique_track = df[df['track_id'] == track].sort_values(by=['frame'], ascending=True)
                
                # IDENTIFY TRACKS WHICH ARE WITHIN/TOUCHING THE BOUNDARY
                unique_track['potential point'] = unique_track['point'].apply(lambda row: hole_boundary.contains(row) or hole_boundary.touches(row))

                # IS THE NEXT ROW INSIDE ALSO 
                unique_track['following point'] = unique_track['potential point'].shift(-1)

                # Identify rows which were within/touching the hole (potential point: True) and have now left the hole (following point: False)
                for i, row in unique_track.iterrows():

                    # IDENTIFY TRACKS WHICH HAVE NOW LEFT THE HOLE BOUNDARY!
                    
                    if row['potential point'] and not row['following point']: 
                        data.append({'track': track, 'exit frame': row['frame'], 'file': track_file})
                        continue 
        
        hole_departures = pd.DataFrame(data)
        hole_departures = hole_departures.sort_values(by=['track'], ascending=True)
        hole_departures.to_csv(os.path.join(self.directory, 'hole_departures.csv'), index=False)

        return hole_departures
                        


    # METHOD HOLE_ORIENTATION: CALCULATES LARVAE ORIENTATION FROM THE HOLE

    def hole_orientation(self):

        self.hole_centroid() # call the hole_centroid method 

        def angle_calculator(vector_A, vector_B):
            # convert to an array for mathmatical ease 
            A = np.array(vector_A)
            B = np.array(vector_B)
            # calculate the dot product
            dot_product = np.dot(A, B)
            # calculate the magnitude of vector (length / norm of vector)
            magnitude_A = np.linalg.norm(vector_A)
            magnitude_B = np.linalg.norm(vector_B)
            # cosθ
            cos_theta = dot_product / (magnitude_A * magnitude_B)
            # θ in radians
            theta_radians = np.arccos(cos_theta)
            # θ in degrees
            theta_degrees = np.degrees(theta_radians)
            return theta_degrees
        
        hole_orientations = []
        data = []

        for track_file, centroid in self.matching_pairs:
            df = self.track_data[track_file]
            
            for row in df.itertuples(): # tuple of each row 

                body = np.array([row.x_body, row.y_body])
                head = np.array([row.x_head, row.y_head])

                hole_body = np.array(centroid) - body 
                body_head = head - body

                angle = angle_calculator(hole_body, body_head)

                frame = row.frame

                hole_orientations.append(angle)

                data.append({'time': frame, 'hole orientation': angle, 'file': track_file})
        

        hole_orientations = pd.DataFrame(hole_orientations)
        hole_orientations.to_csv(os.path.join(self.directory, 'hole_orientations.csv'), index=False)

        hole_orientation_over_time = pd.DataFrame(data)
        hole_orientation_over_time = hole_orientation_over_time.sort_values(by=['time'], ascending=True)
        hole_orientation_over_time.to_csv(os.path.join(self.directory, 'hole_orientation_over_time.csv'), index=False)

        return hole_orientations, hole_orientation_over_time
    
    
    # METHOD NUMBER_DIGGING: THIS METHOD DETECTS HOW MANY LARVAE ARE DIGGING (IN ABSENCE OF MAN-MADE HOLE)

    def number_digging(self, total_larvae):

        dataframe_list = [] 

        for track_file in self.track_files:
            df = self.track_data[track_file]

            # mm_to_pixel = ['x_tail', 'y_tail', 'x_body', 'y_body', 'x_head', 'y_head']
            # df[mm_to_pixel] = df[mm_to_pixel] * (1032/90) #conversion factor mm to pixels 

            df = df.sort_values(by=['track_id', 'frame'])

            # Smooth the positions with a rolling window to reduce noise
            df['x_body'] = df['x_body'].rolling(window=5, min_periods=1).mean()
            df['y_body'] = df['y_body'].rolling(window=5, min_periods=1).mean()

            # Calculate the difference between consecutive rows for body coordinates
            df['dx'] = df.groupby('track_id')['x_body'].diff().fillna(0)
            df['dy'] = df.groupby('track_id')['y_body'].diff().fillna(0)

            # Calculate the Euclidean distance (hypotenuse) between consecutive points
            df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2)
            print(df[['dx', 'dy']].head(-1000))

            # Create a boolean mask where x,y movement is greater than 0.1 MM
            # CHANGE THIS DISTANCE - 
            df['is_moving'] = (df['dx'].abs() > 0.01) | (df['dy'].abs() > 0.01)
            # df['is_moving'] = True

            # Use a rolling window to check for sustained movement over the next 100 frames  
            df['future_movement'] = df.groupby('track_id')['is_moving'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())

            # Use a rolling window to check if the cumulative distance moved in the last 5 frames exceeds a threshold (e.g., 10 pixels)
            # ONLY CONSIDER IT MOVING IF IT HAS MOVED MORE THAN > PIXELS IN THE WINDOW OF A CERTAIN AMOUNT OF FRAMES  
            df['distance_rolled'] = df.groupby('track_id')['distance'].transform(lambda x: x.rolling(window=30, min_periods=1).sum())
            df['distance_check'] = df['distance_rolled'] > 2

            # # Print is_moving (boolean) and future_movement (numerical values), along with distance_rolled and distance_check
            # print(df[['frame', 'distance_rolled', 'distance_check', 'is_moving', 'future_movement']].head(-1500))
            
            # If future_movement is high enough, we can classify as "moving" 
            # unsure if the distance check should be an and 
            # distance check must be True 
            df['final_movement'] = (df['is_moving'] | (df['future_movement'] > 0.3)) & df['distance_check']
            # print(df['final_movement'].head(-2000))
            # df.to_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/plain-petri/testing-digging-behaviour/n10-food/df.csv')
          
            # Now count the moving frames per frame_idx
            moving_counts = df.groupby('frame')['final_movement'].sum().reset_index()

            # Rename the column for clarity
            moving_counts.columns = ['frame', 'moving_count']

            # Ensure we have a count for every frame
            full_frame_range = pd.DataFrame({'frame': range(int(df['frame'].min()), int(df['frame'].max()) + 1)})
            full_frame_counts = full_frame_range.merge(moving_counts, on='frame', how='left').fillna(0)

            # Convert counts to integers
            full_frame_counts['moving_count'] = full_frame_counts['moving_count'].astype(int)

            full_frame_counts['number digging'] = total_larvae - full_frame_counts['moving_count']

            full_frame_counts['file'] = track_file

            dataframe_list.append(full_frame_counts)
        
        number_digging = pd.concat(dataframe_list, ignore_index=True)
        number_digging = number_digging.sort_values(by=['frame'], ascending=True)
        number_digging.to_csv(os.path.join(self.directory, 'number_digging.csv'), index=False)

        return number_digging

    # THIS METHOD SUBTRACTS THE NUMBER MOVING FROM THE TOTAL LARVAE PRESENT - WANT TO INCLUDE ALL THE VIDEOS
    # HOWEVER THIS DOESNT NECESSARILY HELP WITH UNDERLYING PREDICTION ISSUES 
    # AND IF THEY DIG ON TOP OF ONE ANOTHER THIS IS AN ISSUE 

    # def number_digging(self):

    #     dataframe_list = [] 

    #     for track_file in self.track_files:
    #         df = self.track_data[track_file]

    #         # mm_to_pixel = ['x_tail', 'y_tail', 'x_body', 'y_body', 'x_head', 'y_head']
    #         # df[mm_to_pixel] = df[mm_to_pixel] * (1032/90) #conversion factor mm to pixels 

    #         df = df.sort_values(by=['track_id', 'frame'])

    #         # Smooth the positions with a rolling window to reduce noise
    #         df['x_body'] = df['x_body'].rolling(window=5, min_periods=1).mean()
    #         df['y_body'] = df['y_body'].rolling(window=5, min_periods=1).mean()

    #         # Calculate the difference between consecutive rows for body coordinates
    #         df['dx'] = df.groupby('track_id')['x_body'].diff().fillna(0)
    #         df['dy'] = df.groupby('track_id')['y_body'].diff().fillna(0)

    #         # Calculate the Euclidean distance (hypotenuse) between consecutive points
    #         df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2)
    #         print(df[['dx', 'dy']].head(-1000))

    #         # Create a boolean mask where x,y movement is greater than 0.1 pixel
    #         # CHANGE THIS DISTANCE - 
    #         df['is_moving'] = (df['dx'].abs() > 0.01) | (df['dy'].abs() > 0.01)
    #         # df['is_moving'] = True

    #         # Use a rolling window to check for sustained movement over the next 100 frames  
    #         df['future_movement'] = df.groupby('track_id')['is_moving'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())

    #         # Use a rolling window to check if the cumulative distance moved in the last 5 frames exceeds a threshold (e.g., 10 pixels)
    #         # ONLY CONSIDER IT MOVING IF IT HAS MOVED MORE THAN > PIXELS IN THE WINDOW OF A CERTAIN AMOUNT OF FRAMES  
    #         df['distance_rolled'] = df.groupby('track_id')['distance'].transform(lambda x: x.rolling(window=30, min_periods=1).sum())
    #         df['distance_check'] = df['distance_rolled'] > 2

    #         # # Print is_moving (boolean) and future_movement (numerical values), along with distance_rolled and distance_check
    #         # print(df[['frame', 'distance_rolled', 'distance_check', 'is_moving', 'future_movement']].head(-1500))
            
    #         # If future_movement is high enough, we can classify as "moving" 
    #         # unsure if the distance check should be an and 
    #         # distance check must be True 
    #         df['final_movement'] = (df['is_moving'] | (df['future_movement'] > 0.3)) & df['distance_check']
    #         # print(df['final_movement'].head(-2000))
    #         # df.to_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/plain-petri/testing-digging-behaviour/n10-food/df.csv')
          
    #         # Now count the moving frames per frame_idx
    #         moving_counts = df.groupby('frame')['final_movement'].sum().reset_index()

    #         # Rename the column for clarity
    #         moving_counts.columns = ['frame', 'moving_count']

    #         # Calculate the total larvae (unique track_ids) per frame
    #         total_larvae_per_frame = df.groupby('frame')['track_id'].nunique().reset_index()
    #         total_larvae_per_frame.columns = ['frame', 'total_larvae']

    #         # Merge the moving counts with the total larvae count per frame
    #         full_frame_counts = pd.merge(total_larvae_per_frame, moving_counts, on='frame', how='left').fillna(0)

    #         # Calculate the number of digging larvae
    #         full_frame_counts['number_digging'] = full_frame_counts['total_larvae'] - full_frame_counts['moving_count']

    #         # Add a column for the track file
    #         full_frame_counts['file'] = track_file

    #         dataframe_list.append(full_frame_counts)
        
    #     number_digging = pd.concat(dataframe_list, ignore_index=True)
    #     number_digging = number_digging.sort_values(by=['frame'], ascending=True)
    #     number_digging.to_csv(os.path.join(self.directory, 'number_digging.csv'), index=False)

    #     return number_digging








        # METHOD INITIAL_HOLE_FORMATION: TIME AT WHICH THE FIRST LARVAE BEGINS DIGGING
        # EXTRACTED FROM THE ABOVE !
    






















    # DIGGING IN ISOLATION/ HOW TO COMBINE THE MAN-MADE HOLE WITH THE NUMBER DIGGING OUTSIDE OF THIS
      # IDEK HOW TO DO THIS

    # METHOD CASTING: 

    # METHOD FOR TRACK OVERLAY IMAGES AND VIDEOS 










