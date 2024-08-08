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

        return ax
    #     print("probability_density method called")
    #     print(f"Reading file: {file}")
        
    #     try:
    #         df = pd.read_csv(file)
    #         print("File read successfully")
    #         print(df.head())  # Print the first few rows to confirm contents
            
    #         data = df.iloc[:, 0].values
    #         data = pd.Series(data).replace([np.inf, -np.inf], np.nan).dropna().values

    #         kde = gaussian_kde(data)
    #         value_range = np.linspace(data.min(), data.max(), 100)
    #         density = kde(value_range)

    #         ax = sns.lineplot(x=value_range, y=density)
    #         plt.title('Probability Density Function')
    #         plt.xlabel('Values')
    #         plt.ylabel('Density')
    #         plt.show()

    #         return ax
    #     except Exception as e:
    #         print(f"An error occurred: {e}")

    
    # @staticmethod
    # def test_method():
    #     print("HoleAnalysis import successful!")



 



     # # def number_in_hole_counter():
    
    # # def returns(): #number returning to the hole 


    # METHOD ENSEMBLE_MSD: CALCULATES ENSEMBLE MSD 
     
    # def ensemble_msd(self):

    #     distance_per_track = []
    
    #     def msd_distance(x1, y1, x0, y0):
    #         msd = (x1 - x0)**2 + (y1- y0)**2
    #         return msd
        

    #     for track_file in self.track_files:
    

    # METHOD TIME_AVERAGE_MSD: 

    # def time_average_msd(self):




    # METHOD PROBABILITY_DENSITY: MUST TAKE A 1D ARRAY SO UNSURE HOW TO TACKLE THIS RN 

    # def probability_density(self):




    # METHOD SPEED: 




    # METHOD FOR TRACK OVERLAY IMAGES ETC? YES










# i will have track csv and holes csv coordinates

# defintion which fills in the coordinates of the hole and removes it from the csv file idk like any larvae inside is not counted - 


# # things i would want to use in every analysis which is useful
# #   - iterate over every file in a directory 
#     - - THEN SCRIPT FOR ANALYSIS NOT IN HOLE OR IN HOLE - count number of tracks per frame but not those in the hole + CERTAIN RADIUS
# - RETURNS TO HOLE? - tracks which appear in certain radius and re enter the radis - like a counter 







