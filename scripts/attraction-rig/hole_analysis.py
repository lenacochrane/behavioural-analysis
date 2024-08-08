import pandas as pd
import numpy as np
import os 
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
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
        self.track_files = [f for f in os.listdir(self.directory) if f.endswith('tracks.csv')]
        print(f"Track files: {self.track_files}")
        # load the data 
        for track_file in self.track_files: 
            track_path = os.path.join(self.directory, track_file)
            self.track_data[track_file] = pd.read_csv(track_path)

   # METHOD SHORTEN: OPTIONAL METHOD TO SHORTEN THE TRACK FILES TO INCLUDE UP TO A CERTAIN FRAME  
    
    def shorten(self, frame=-1):

        for track_file in self.track_files:

            df = self.track_data[track_file]

            df = df[df['frame'] <= frame]

            self.track_data[track_file] = df # update the track data 

            # create path 
            shortened_path = os.path.join(self.directory, track_file.replace('.csv', f'_shortened_{frame}.csv'))

            # save 
            df.to_csv(shortened_path, index=False)

            print(f"Shortened file saved: {shortened_path}")

    # METHOD HOLE_BOUNDARY: CREATES A POLYGON AROUND THE HOLE BOUNDARY WITH SCALAR OPTION
     # 1. CONVEX HULL: CONVEX SHAPE THAT ENCLOSES A SET OF POINTS (CONTINIOUS BOUNDARY)
     # 2. VERTICES: CORNER POINTS OF THE CONVEX SHAPE
     # 3. POLYGON: GEOMETRIC SHAPE FORMED BY CONNECTING THESE VERTICES

    def hole_boundary(self, scale_factor=1.0):  

        self.hole_boundaries = []

        for coordinates in self.coordinate_files:

            file_path = os.path.join(self.directory, coordinates)

            df = pd.read_csv(file_path, header=None, names=['x', 'y'])

            points = df[['x', 'y']].values # values creates numpy array

            hull = ConvexHull(points)

            # struggle with this understanding - ask callum 
            # this retrieves the points of the shapes 
            # defines the boundary points
            hull_points = points[hull.vertices]

            # create the polygon
            polygon = Polygon(hull_points)

            scaled_polygon = scale(polygon, xfact=scale_factor, yfact=scale_factor, origin='center')

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
            track_prefix = track_prefix.replace('.tracks.csv', '')
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

                track_path = os.path.join(self.directory, track_file)
                self.track_data[track_file] = pd.read_csv(track_path)
        
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

        for track_file, centroid in self.matching_pairs: # track file is just the name of the file 

            df = self.track_data[track_file] # retreieve the data for that file name 
            for index, row in df.iterrows():
                x, y = row['x_body'], row['y_body']
                distance = np.sqrt((centroid[0] - x)**2 + (centroid[1] - y)**2)
                distances_from_hole.append(distance)
        
        print("Distances from hole centroid:", distances_from_hole)

        if not distances_from_hole:
            print("No distances calculated, check data")
        else:

            df_distances = pd.DataFrame(distances_from_hole, columns=['Distance from hole'])
            df_distances.to_csv(os.path.join(self.directory, 'distance_from_hole_centroid.csv'), index=False)
            print(f"Distance from hole saved: {df_distances}")
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
    
    # METHOD PROBABILITY_DENSITY: PROBABILITY DENSITY FOR INPUTTED 1D ARRAY
    
    @staticmethod
    def probability_density(file, x=None, y=None):

        df = pd.read_csv(file)

        data = df.iloc[:, 0].values #iloc indexed based selection - : all rows - 0 first column

        # Replace infinite values with NaN and drop them
        data = pd.Series(data).replace([np.inf, -np.inf], np.nan).dropna().values

        kde = gaussian_kde(data) #kernel estimate density function for the kde data 

        value_range = np.linspace(data.min(), data.max(), 100)
         # this generates a range of values over which the KDE will be evaluated.
        # np.linspace(start, stop, num)

        density = kde(value_range)
        # evaluates the KDE at each of the points in the range provided 

        ax = sns.lineplot(x=value_range, y=density)

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

                    data.append({'time': time1, 'speed': speed_value, 'file': track_file})
       
        speed_values = pd.DataFrame(speed)
        speed_values.to_csv(os.path.join(self.directory, 'speed_values.csv'), index=False)

        speed_over_time = pd.DataFrame(data)
        speed_over_time = speed_over_time.sort_values(by=['time'], ascending=True)
        speed_over_time.to_csv(os.path.join(self.directory, 'speed_over_time.csv'), index=False)

        return speed_values, speed_over_time


    # METHOD ENSEMBLE_MSD: CALCULATES SQUARED DISTANCE FOR EVERY POSITION FROM TIME 0
     
    def ensemble_msd(self): 

        # frame distance and file 

        data = []
        
        for track_file in self.track_files:
            track_data = self.track_data[track_file]

            # calculate average x,y for first frame 
            # (needs to change such that it time 0 for each unique track compared back to)
            
            frame_0 = track_data[track_data['frame'] == 0]

            x = frame_0['x_body'].mean()
            y = frame_0['y_body'].mean()

            for i, row in track_data.iterrows():

                squared_distance = (row['x_body'] - x)**2 + (row['y_body'] - y)**2

                frame = row['frame']

                data.append({'time': frame, 'squared distance': squared_distance, 'file': track_file})
        
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


        


     # def hole_counter():
    
    # def returns(): #number returning to the hole 


    # METHOD TRAJECTORY / BODY TURNS / ANGLE ORIENTATION  

    # METHOD PROXIMITY

    # METHOD CASTING 

    # METHOD FOR TRACK OVERLAY IMAGES AND VIDEOS 










# i will have track csv and holes csv coordinates

# defintion which fills in the coordinates of the hole and removes it from the csv file idk like any larvae inside is not counted - 


# # things i would want to use in every analysis which is useful
# #   - iterate over every file in a directory 
#     - - THEN SCRIPT FOR ANALYSIS NOT IN HOLE OR IN HOLE - count number of tracks per frame but not those in the hole + CERTAIN RADIUS
# - RETURNS TO HOLE? - tracks which appear in certain radius and re enter the radis - like a counter 







