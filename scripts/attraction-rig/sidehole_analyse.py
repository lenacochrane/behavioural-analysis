import pandas as pd
import numpy as np
import os 
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
from shapely.affinity import scale
from shapely.wkt import dumps as wkt_dumps
import seaborn as sns
import matplotlib.pyplot as plt


class Side_hole_analysis:

    def __init__(self, directory):

        self.directory = directory 
        self.coordinate_files = []
        self.track_files = []
        self.hole_boundaries = []
        self.matching_pairs = []
        self.track_data = {}  # Initialize the track_data dictionary # confuse by this
        self.coordinates()
        self.tracks()
        self.hole_boundary()
        self.match_files()

    def coordinates(self):
        # 2024-05-20_16-08-22_td1_hole.csv
        self.coordinate_files = [f for f in os.listdir(self.directory) if f.endswith('hole.csv')]
        print(f"Coordinate files: {self.coordinate_files}")

    def tracks(self):
        # 2024-04-30_14-31-44_td5.000_2024-04-30_14-31-44_td5.analysis.csv
        self.track_files = [f for f in os.listdir(self.directory) if f.endswith('tracks.csv')]
        print(f"Track files: {self.track_files}")


    # Creating a hole boundary will be achieved in three steps:
        #   Convex Hull: The smallest convex shape that encloses a set of points.
            # create continuous boundary around the points
        #   Vertices: The corner points of this convex shape.
        #   Polygon: A geometric shape formed by connecting these vertices in order.

# have to somohow read it ltr on 
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
    
    # METHOD MATCH_FILES:

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


    # METHOD HOLE_CENTROID: REPLACE THE HOLE BOUNDARY WITH A CENTROID COORDINATE 

    def hole_centroid(self):

        updated_matching_pairs = [] # update matching pairs with funderals 

        for track_file, hole_boundary in self.matching_pairs:

            centroid = hole_boundary.centroid  # Calculate centroid of the polygon

            updated_matching_pairs.append((track_file, (centroid.x, centroid.y))) # centroid is a tuple
        
        self.matching_pairs = updated_matching_pairs
        print(f"Matching pairs with centroids: {self.matching_pairs}")
        return self.matching_pairs
    
    # METHOD DISTANCE_FROM_HOLE: CALCULATES THE DISTANCES FROM THE HOLE CENTROID AND PLOTS A DISTRIBUTION

    def distance_from_hole(self): 

        self.hole_centroid() # call the hole_centroid method 

        distances_from_hole = []

        for track_file, centroid in self.matching_pairs:

            predictions = self.track_data[track_file]
            for index, row in predictions.iterrows():
                x, y = row['x_body'], row['y_body']
                distance = np.sqrt((centroid[0] - x)**2 + (centroid[1] - y)**2)
                distances_from_hole.append(distance)
        
        # plt.figure(figsize=(8, 6))
        # plt.hist(distances_from_hole, bins=70, edgecolor='black')
        # plt.xlabel('Distance from Hole', fontweight='bold')
        # plt.ylabel('Frequency', fontweight='bold')
        # plt.title('Distribution of Distances from Hole Centroid', fontweight='bold')
        # plt.show()
    

            # Debugging: Print the distances
        print("Distances from hole centroid:", distances_from_hole)

        if not distances_from_hole:
            print("No distances calculated, check data")
        else:
            plt.figure(figsize=(8, 6))
            plt.hist(distances_from_hole, bins=70, edgecolor='black')
            plt.xlabel('Distance from Hole', fontweight='bold')
            plt.ylabel('Frequency', fontweight='bold')
            plt.title('Distribution of Distances from Hole Centroid', fontweight='bold')
            plt.show()
    # # def number_in_hole_counter():
    
    # # def returns(): #number returning to the hole 



    def distance_from_centre(self): 

        centre = (700, 700) 

        distances_from_centre = []

        for track_file in self.track_files:
            
            predictions = self.track_data[track_file]

            for index, row in predictions.iterrows():
                x, y = row['x_body'], row['y_body']
                distance = np.sqrt((centre[0] - x)**2 + (centre[1] - y)**2)
                distances_from_centre.append(distance)
        

            plt.figure(figsize=(8, 6))
            plt.hist(distances_from_centre, bins=70, edgecolor='black')
            plt.xlabel('Distance from Centre', fontweight='bold')
            plt.ylabel('Frequency', fontweight='bold')
            plt.title('Distribution of Distances from Centre', fontweight='bold')
            plt.show()






# i will have track csv and holes csv coordinates

# defintion which fills in the coordinates of the hole and removes it from the csv file idk like any larvae inside is not counted - 


# # things i would want to use in every analysis which is useful
# #   - iterate over every file in a directory 
#     - - THEN SCRIPT FOR ANALYSIS NOT IN HOLE OR IN HOLE - count number of tracks per frame but not those in the hole + CERTAIN RADIUS
# - RETURNS TO HOLE? - tracks which appear in certain radius and re enter the radis - like a counter 







