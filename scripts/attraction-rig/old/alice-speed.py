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
from itertools import combinations
from joblib import Parallel, delayed
import re
from itertools import product
from collections import defaultdict, Counter
import itertools
from scipy.ndimage import label, find_objects
from scipy.spatial.distance import pdist


class HoleAnalysis:

    def __init__(self, directory):

        self.directory = directory 
        self.coordinate_files = []
        self.track_files = [] # list of the files 
        self.hole_boundaries = []
        self.matching_pairs = []
        self.track_data = {}  # Initialize the track_data dictionary # actually has the data so we dont have to keep reloading 
        
        self.perimeter()
        self.coordinates() # used by hole boundary 
        self.hole_boundary()
        self.tracks()
        self.match_files()
        self.conversion()

        self.use_shorten = True 
        self.shorten_duration = None

        # self.digging = None

    # METHOD COORDINATES: IDENTIFIES AND STORES THE HOLE COORDINATE FILES

    def coordinates(self):
        # 2024-05-20_16-08-22_td1_hole.csv
        self.coordinate_files = [f for f in os.listdir(self.directory) if f.endswith('hole.csv')]
        print(f"Coordinate files: {self.coordinate_files}")

    # METHOD TRACKS: IDENTIES AND STORES THE SLEAP TRACK FILES; TRACK DATA IS SUBSEQUENTLY READ  

    def tracks(self):
        # 2024-04-30_14-31-44_td5.000_2024-04-30_14-31-44_td5.analysis.csv
        self.track_files = [f for f in os.listdir(self.directory) if f.endswith('tracks.feather')]
    
        for track_file in self.track_files: 
            track_path = os.path.join(self.directory, track_file)
            df = pd.read_feather(track_path)
            # NEED DIAMATER CONVERSION FFS 
            # # cant access the perimeter right now here 
            # diameter = self.diameter()
            # print(diameter)

            # pixels_to_mm = ['x_tail', 'y_tail', 'x_body', 'y_body', 'x_head', 'y_head']
            # df[pixels_to_mm] = df[pixels_to_mm] * (90 / diameter)
            # print(df.head())
            self.track_data[track_file] = df
    
   # METHOD SHORTEN: OPTIONAL METHOD TO SHORTEN THE TRACK FILES TO INCLUDE UP TO A CERTAIN FRAME  
    
    def shorten(self, frame=-1):

        for track_file in self.track_files:

            df = self.track_data[track_file]
            df = df[df['frame'] <= frame]
            self.track_data[track_file] = df # update the track data 

            # # create path 
            # shortened_path = os.path.join(self.directory, track_file.replace('.feather', f'_shortened_{frame}.feather'))
            # # save 
            # df.reset_index(drop=True, inplace=True)  # Feather requires a default integer index
            # df.to_feather(shortened_path)  # Save the DataFrame without 'index=False'
            # print(f"Shortened file saved: {shortened_path}")
        self.use_shorten = True
        self.shorten_duration = frame  # e.g., 600

        
    ### METHOD DIGGING_MASK: FILTERS FOR NON-DIGGING LARVAE

    def digging_mask(self):

        for track_file in self.track_files:
            df = self.track_data[track_file]
            df = self.compute_digging(df)
            # df.to_csv(os.path.join(self.directory, 'digging.csv'), index=False) # get rid 
            self.track_data[track_file] = df[df['digging_status'] == False].copy()
    

    ### METHOD HOLE_MASK: FILTERS FOR NON-HOLE LARVAE

    def hole_mask(self):

        for track_file in self.track_files:

            df = self.track_data[track_file]
            mask = (~df['within_hole']) & (~df['digging_outside_hole'])  # exclude both
            df = df[mask].copy()  # update df with filtered version
            self.track_data[track_file] = df  # save it back
            # df.to_csv(os.path.join(self.directory, 'test.csv'), index=False)

        

    # METHOD POST_PROCESSING: 1) FILTERS TRACK'S AVERAGE INSTANCE SCORE < 0.9 

    def post_processing(self):
        
        for track_file in self.track_files:
            df = self.track_data[track_file]
            # group by tracks, calculate mean per tracks, if True >= 0.9 include in df 
            df = df[df.groupby('track_id')['instance_score'].transform('mean') >= 0.9]

            self.track_data[track_file] = df  # Update the in-memory version
    

    # METHOD PERIMETER: IDENTIFY XY CENTRE POINTS AND PERIMETER OF THE PETRI DISH

    def perimeter(self):
        
        # function to process the video 1) identify centre coordinates and the perimeter
        def process_video(video_path):
            video_name = os.path.splitext(os.path.basename(video_path))[0]

            # Check if the perimeter file already exists
            wkt_file_path = os.path.join(self.directory, f"{video_name}_perimeter.wkt")
            if os.path.exists(wkt_file_path):
                return

            def detect_largest_circle(frame):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_blurred = cv2.medianBlur(gray, 5)
        
                circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.0, minDist=100,
                                       param1=500, param2=50, minRadius=400, maxRadius=600)
                if circles is not None:
                    largest_circle = max(circles[0, :], key=lambda c: c[2])  # No rounding for accuracy
                    return largest_circle  # x, y, r (center coordinates and radius)
                return None

            def circle_to_polygon(x, y, radius, num_points=100):
                angles = np.linspace(0, 2 * np.pi, num_points)
                points = [(x + radius * np.cos(angle), y + radius * np.sin(angle)) for angle in angles]
                return Polygon(points)
            
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 10) # frame 10 
            ret, frame = cap.read()
            
            if ret:
                circle = detect_largest_circle(frame)
                if circle is not None:
                    x, y, r = circle
                    petri_dish_boundary = circle_to_polygon(x, y, r)

                    save_dir = self.directory
                    wkt_file_path = os.path.join(save_dir, f"{video_name}_perimeter.wkt")
                    with open(wkt_file_path, 'w') as f:
                        f.write(petri_dish_boundary.wkt)
                
                    # Draw the circle on the frame
                    cv2.circle(frame, (int(x), int(y)), int(r), (0, 255, 0), 2)

                    # Updated PNG-saving logic
                    frame_with_boundary_path = os.path.join(save_dir, f"{video_name}_perimeter.png")
                    cv2.imwrite(frame_with_boundary_path, frame)
            
                else:
                    print(f"No Perimeter detected for {video_name} .")
            else:
                print(f"Failed to extract the 10th frame from the video.")

            cap.release()
            return None
        
        # Iterate through video files in the directory
        video_files = [f for f in os.listdir(self.directory) if f.endswith('.mp4')]
        for file in video_files:
            video_path = os.path.join(self.directory, file)
            process_video(video_path)


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
            # conversion_factor = 90 / self.diameter
            # df[['x', 'y']] = df[['x', 'y']] * conversion_factor

            points = df[['x', 'y']].values # values creates numpy array

            hull = ConvexHull(points)  

            # struggle with this understanding - ask callum 
            # this retrieves the points of the shapes 
            # defines the boundary points
            hull_points = points[hull.vertices]

            # create the polygon
            polygon = Polygon(hull_points)

            # scaled_polygon = scale(polygon, xfact=scale_factor, yfact=scale_factor, origin='center') # polygon scaled uniform relative to center

            self.hole_boundaries.append(polygon) #change from scale to poly

            wkt_string = wkt_dumps(polygon) # Convert the scaled polygon to WKT format

            # Save the WKT string to a file with the same name as the original but with a .wkt extension
            # hole_boundary = coordinates.replace('.csv', '.wkt')
            hole_boundary = os.path.join(self.directory, coordinates.replace('.csv', '.wkt'))

            with open(hole_boundary, 'w') as f:
                f.write(wkt_string)
        
        print(f"Hole boundaries: {self.hole_boundaries}")
    
    
    # METHOD MATCH_FILES: MATCHES THE TRACK FILES WITH THEIR COORDINATE FILES (BY EXTENTION THE HOLE POLYGON)

    def match_files(self):
        # Initialize a list for all matching pairs
        self.matching_pairs = []

        # Gather all video and perimeter files
        video_files = [f for f in os.listdir(self.directory) if f.endswith('.mp4')]
        perimeter_files = [f for f in os.listdir(self.directory) if f.endswith('_perimeter.wkt')]

        # Iterate over all track files
        for track_file in self.track_files:
            # Extract the common prefix from the track file
            track_prefix = '_'.join(track_file.split('_')[:3]).replace('.tracks.feather', '')
            matched_data = {
                'track_file': track_file,
                'hole_boundary': None,
                'video_file': None,
                'perimeter_file': None}

            # Match with coordinate files (hole boundaries)
            for i, coordinates_file in enumerate(self.coordinate_files):
                hole_prefix = '_'.join(coordinates_file.split('_')[:3]).rsplit('.', 1)[0]
                if hole_prefix == track_prefix:
                    # print(f"Match found: {track_file} with {coordinates_file}")
                    matched_data['hole_boundary'] = self.hole_boundaries[i]  # Assign the associated hole boundary polygon

            # Match with video files
            for video_file in video_files:
                video_prefix = '_'.join(video_file.split('_')[:3]).rsplit('.', 1)[0]
                if video_prefix == track_prefix:
                    matched_data['video_file'] = video_file

            # Match with perimeter files
            for perimeter_file in perimeter_files:
                perimeter_prefix = '_'.join(perimeter_file.split('_')[:3]).rsplit('.', 1)[0]
                if perimeter_prefix == track_prefix:
                    matched_data['perimeter_file'] = perimeter_file
                    # print(f"Match found: {track_file} with {perimeter_file}")

                    # Read the perimeter file and parse it into a Polygon object
                    perimeter_path = os.path.join(self.directory, perimeter_file)
                    with open(perimeter_path, 'r') as f:
                        perimeter_wkt = f.read()

                    polygon = wkt.loads(perimeter_wkt)

                    matched_data['perimeter_polygon'] = polygon           
                    
            # Append the matched data to the matching_pairs list
            self.matching_pairs.append(matched_data)
    
    # METHOD CONVERSION:CONVERTS EACH FILE FROM PIXELS INTO MM

    def conversion(self):

        for match in self.matching_pairs:
            
            perimeter_polygon = match.get('perimeter_polygon')
            
            if perimeter_polygon:
                # Calculate the diameter of the perimeter 
                minx, miny, maxx, maxy = perimeter_polygon.bounds
                diameter = maxx - minx  # This assumes the perimeter is a circle and uses its width as the diameter.

                conversion_factor = 90 / diameter # 90mm 

                # IF PERIMETER DETECTED BADLY 
                threshold = 0.09 #
                if conversion_factor > threshold:
                    print(f"Conversion factor {conversion_factor:.3f} is above threshold for {match['track_file']}. Using default conversion factor:")
                    conversion_factor = 90 / 1032  # Use the old conversion factor
              

                # scaled_perimeter_polygon = scale(perimeter_polygon, xfact=conversion_factor, yfact=conversion_factor,  origin=(0, 0))
                perimeter_coordinates = np.array(perimeter_polygon.exterior.coords)
                perimeter_coordinates *= conversion_factor
                scaled_perimeter_polygon = Polygon(perimeter_coordinates)

                match['perimeter_polygon'] = scaled_perimeter_polygon  # Update the scaled polygon.

                # Apply conversion to hole boundaries.
                hole_boundary = match.get('hole_boundary')
                if hole_boundary:
                    # Scale the hole boundary using the conversion factor.

                    coordinates = np.array(hole_boundary.exterior.coords)

                    coordinates *= conversion_factor
                    scaled_polygon = Polygon(coordinates)
                    print(scaled_polygon)

                    match['hole_boundary'] = scaled_polygon

                    # scaled_hole_boundary = scale(hole_boundary, xfact=conversion_factor, yfact=conversion_factor, origin='center')
                    # match['hole_boundary'] = scaled_hole_boundary  # Update the scaled polygon.
                    # print(scaled_hole_boundary)

                track_file = match['track_file']
                track_data = self.track_data[track_file]

                pixel_columns = ['x_tail', 'y_tail', 'x_body', 'y_body', 'x_head', 'y_head']
                track_data[pixel_columns] = track_data[pixel_columns] * conversion_factor
                self.track_data[track_file] = track_data  # Update the track data.
                print(f"Conversion applied for {track_file} with conversion factor: {conversion_factor:.3f}")
            
            else:
                print(f"no perimeter detected for {match['track_file']}")
  
                conversion_factor = 90 / 1032 # the one i used to use 
                hole_boundary = match.get('hole_boundary')
                if hole_boundary:
                    coordinates = np.array(hole_boundary.exterior.coords)
                    coordinates *= conversion_factor
                    scaled_polygon = Polygon(coordinates)
                    print(scaled_polygon)
                    match['hole_boundary'] = scaled_polygon
                
                track_file = match['track_file']
                track_data = self.track_data[track_file]

                pixel_columns = ['x_tail', 'y_tail', 'x_body', 'y_body', 'x_head', 'y_head']
                track_data[pixel_columns] = track_data[pixel_columns] * conversion_factor
                self.track_data[track_file] = track_data  # Update the track data.
                print(f"Conversion applied for {track_file} with conversion factor: {conversion_factor:.3f}")


    # METHOD SPEED: CALCULATES SPEED: 1) SPEED VALUES 2) SPEED OVER TIME 

    def speed(self):

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

                    if time > 2:
                        continue

                    speed_value = distance / time 

                    data.append({'time': time2, 'speed': speed_value, 'file': track_file})
    
        speed_over_time = pd.DataFrame(data)
        speed_over_time = speed_over_time.sort_values(by=['time'], ascending=True)

        if self.shorten and self.shorten_duration is not None:
            suffix = f"_{self.shorten_duration}"
        else:
            suffix = ""

        filename = f"speed_over_time{suffix}.csv"

        speed_over_time.to_csv(os.path.join(self.directory, filename), index=False)

        return speed_over_time
    




analysis = HoleAnalysis("/path/to/your/datafolder")
analysis.speed()
