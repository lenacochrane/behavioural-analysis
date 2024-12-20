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
        # self.radius()

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

    # METHOD POST_PROCESSING: 1) FILTERS TRACK'S AVERAGE INSTANCE SCORE < 0.9 
    # # 2) FILLS INCREMENTALLY- REMOVED 

    def post_processing(self):
        
        # FUNCTION INPUTS INDIVIDUAL TRACK DF DATA AND INCRIMENTALLY FILLS IN GAPS 
        # def interpolate(track_df):
        #     # range of frames
        #     min_frame = track_df['frame'].min() 
        #     max_frame = track_df['frame'].max()
        #     # create Numpy Array of min-max 
        #     frame_range = np.arange(min_frame, max_frame + 1)
        #     # return difference between expected and actual frame numbers
        #     missing_frames = np.setdiff1d(frame_range, track_df['frame'].values)
  
        #     if len(missing_frames) == 0:
        #         return track_df
    
        #     track_name = track_df['track_id'].iloc[0]
        #     # create df for missing frames
        #     missing_df = pd.DataFrame({'frame': missing_frames, 'track_id': track_name})
        #     # join track data and missing tracks 
        #     df = pd.concat([track_df, missing_df]).sort_values(by='frame')

        #     # Interpolate for each coordinate pair
        #     coordinates = ['x_head', 'y_head', 'x_body', 'y_body', 'x_tail', 'y_tail']
        #     # add nan values for the missing data in the additional frames 
        #     for coord in coordinates:
        #         if coord not in df.columns:
        #             df[coord] = np.nan 
            
        #     for coord in coordinates:
        #         # interpolate fills in missing values assuming a linear relation between known values
        #         df[coord] = df[coord].interpolate()
    
        #     # Forward-fill and backward-fill (dont think this is applicable here really- gaps at start and end of track)
        #     # for coord in coordinates:
        #     #     full_df[coord] = full_df[coord].ffill().bfill()
        #     return df
        

        for track_file in self.track_files:
            df = self.track_data[track_file]

            # group by tracks, calculate mean per tracks, if True >= 0.9 include in df 
            df = df[df.groupby('track_id')['instance_score'].transform('mean') >= 0.9]

            # fill in track gaps 
            # df = df.sort_values(by=['track_id', 'frame'])
            # # applies the definition to each mini dataframe for tracks and then combines the results into a single dataframe
            # df = df.groupby('track_id').apply(interpolate).reset_index(drop=True)

            # # Save the post-processed DataFrame to the original file path
            # file_path = os.path.join(self.directory, track_file)  # Combines directory and filename
            # df.to_feather(file_path)  # Save the DataFrame back to the file
            self.track_data[track_file] = df  # Update the in-memory version
    

    # METHOD PERIMETER: IDENTIFY XY CENTRE POINTS AND PERIMETER OF THE PETRI DISH

    def perimeter(self):
        
        # function to process the video 1) identify centre coordinates and the perimeter
        def process_video(video_path):
            video_name = os.path.splitext(os.path.basename(video_path))[0]

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
                    print(f"Frame with boundary saved at {frame_with_boundary_path}.")
            
                else:
                    print("No circle detected.")
            else:
                print(f"Failed to extract the 10th frame from the video.")

            cap.release()
            return None
        
        # Iterate through video files in the directory
        video_files = [f for f in os.listdir(self.directory) if f.endswith('.mp4')]
        for file in video_files:
            video_path = os.path.join(self.directory, file)
            print(f"Processing video: {video_path}")
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
                'perimeter_file': None
            }

            # Match with coordinate files (hole boundaries)
            for i, coordinates_file in enumerate(self.coordinate_files):
                hole_prefix = '_'.join(coordinates_file.split('_')[:3]).rsplit('.', 1)[0]
                if hole_prefix == track_prefix:
                    print(f"Match found: {track_file} with {coordinates_file}")
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
                    print(f"Match found: {track_file} with {perimeter_file}")

                    # Read the perimeter file and parse it into a Polygon object
                    perimeter_path = os.path.join(self.directory, perimeter_file)
                    with open(perimeter_path, 'r') as f:
                        perimeter_wkt = f.read()


                    polygon = wkt.loads(perimeter_wkt)
                    scaling_factor = (90/1032)
                    scaled_polygon = scale(polygon, xfact=scaling_factor, yfact=scaling_factor, origin=(0, 0))
                    
                    matched_data['perimeter_polygon'] = scaled_polygon
                    

            # Append the matched data to the matching_pairs list
            self.matching_pairs.append(matched_data)

        print(f"All matching pairs: {self.matching_pairs}")


    # def match_files(self):

    #     for track_file in self.track_files:
   
    #         track_prefix = '_'.join(track_file.split('_')[:3])
    #         track_prefix = track_prefix.replace('.tracks.feather', '')

    #         for i, coordinates_file in enumerate(self.coordinate_files):
    #             hole_prefix = '_'.join(coordinates_file.split('_')[:3])
    #             hole_prefix = hole_prefix.rsplit('.', 1)[0]  # Remove the extension

    #             if hole_prefix == track_prefix:
    #                 print(f"Match found: {track_file} with {coordinates_file}")
    #                 self.matching_pairs.append((track_file, self.hole_boundaries[i]))

        
    #     print(f"Matching pairs: {self.matching_pairs}")
    #     print(f"Track data keys: {self.track_data.keys()}")


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

        # self.hole_centroid() # call the hole_centroid method 

        distances_from_hole = []
        data = []

        for match in self.matching_pairs:  # Access dictionaries instead of unpacking tuples
            track_file = match['track_file']
            hole_boundary = match['hole_boundary']
            
            if hole_boundary is None:
                print(f"No hole boundary for track file: {track_file}")
                continue

            df = self.track_data[track_file]

            centroid = hole_boundary.centroid

            for index, row in df.iterrows():
                x, y = row['x_body'], row['y_body']
                distance = np.sqrt((centroid.x - x)**2 + (centroid.y - y)**2)
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

        distances_from_centre = []
        data = []

        for match in self.matching_pairs:
            track_file = match['track_file']
            perimeter = match.get('perimeter_polygon')
            
            if perimeter is None:
                print(f"No perimeter polygon available for track file: {track_file}")
                continue

            centre_x, centre_y = perimeter.centroid.x, perimeter.centroid.y

            predictions = self.track_data[track_file]

            for index, row in predictions.iterrows():
                x, y = row['x_body'], row['y_body']
                distance = np.sqrt((centre_x - x)**2 + (centre_y - y)**2)
                distances_from_centre.append(distance)
                data.append({'file': track_file, 'frame': row['frame'], 'distance_from_centre': distance})

        df_distances = pd.DataFrame(distances_from_centre, columns=['Distance from centre'])
        df_distances.to_csv(os.path.join(self.directory, 'distances_from_centre.csv'), index=False)
        print(f"Distance from centre saved: {df_distances}")

        df_distance_over_time = pd.DataFrame(data)
        df_distance_over_time.to_csv(os.path.join(self.directory, 'distance_over_time.csv'), index=False)
        print(f'Distance over time saved: {df_distance_over_time}')

        return df_distances, df_distance_over_time

    # def distance_from_centre(self): 

    #     factor = 700 * (90/1032)

    #     centre = (factor, factor) 

    #     distances_from_centre = []
    #     data = []

    #     for track_file in self.track_files:
            
    #         predictions = self.track_data[track_file]

    #         for index, row in predictions.iterrows():
    #             x, y = row['x_body'], row['y_body']
    #             distance = np.sqrt((centre[0] - x)**2 + (centre[1] - y)**2)
    #             distances_from_centre.append(distance)
    #             data.append({'file': track_file, 'frame': row['frame'], 'distance_from_centre': distance})

    #     df_distances = pd.DataFrame(distances_from_centre, columns=['Distance from centre'])
    #     df_distances.to_csv(os.path.join(self.directory, 'distances_from_centre.csv'), index=False)
    #     print(f"Distance from centre saved: {df_distances}")

    #     df_distance_over_time = pd.DataFrame(data)
    #     df_distance_over_time.to_csv(os.path.join(self.directory, 'distance_over_time.csv'), index=False)

    #     return df_distances

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
    
    # METHOD ENSEMBLE_MSD: CALCULATES SQUARED DISTANCE FOR EVERY POSITION FROM THE CENTROID COORDINATES
    
    def ensemble_msd(self):

        data = []

        for match in self.matching_pairs:
            track_file = match['track_file']
            perimeter = match.get('perimeter_polygon')

            # Ensure the perimeter polygon is available
            if perimeter is None:
                print(f"No perimeter polygon available for track file: {track_file}")
                continue

            # Calculate the centroid of the perimeter polygon
            centre_x, centre_y = perimeter.centroid.x, perimeter.centroid.y

            track_data = self.track_data[track_file]

            for track_id in track_data['track_id'].unique():
                track_unique = track_data[track_data['track_id'] == track_id].sort_values(by=['frame']).reset_index(drop=True)

                for _, row in track_unique.iterrows():
                    squared_distance = (row['x_body'] - centre_x) ** 2 + (row['y_body'] - centre_y) ** 2
                    data.append({
                    'time': row['frame'], 
                    'squared_distance': squared_distance, 
                    'file': track_file
                })
                    
        # Create a DataFrame from the MSD data
        df = pd.DataFrame(data)
        df = df.sort_values(by=['time'], ascending=True)

        # Save the DataFrame as a CSV file
        output_path = os.path.join(self.directory, 'ensemble_msd.csv')
        df.to_csv(output_path, index=False)
        print(f"Ensemble MSD saved to {output_path}")
        return df 





    # # METHOD ENSEMBLE_MSD: CALCULATES SQUARED DISTANCE FOR EVERY POSITION FROM FIRST TRACK APPEARANCE
     
    # def ensemble_msd(self): 

    #     # frame distance and file 

    #     data = []

    #     for match in self.matching_pairs:

    #         track_file = match['track_file']
    #         perimeter = match.get('perimeter_polygon')
            
    #         if perimeter is None:
    #             print(f"No perimeter polygon available for track file: {track_file}")
    #             continue

    #         centre_x, centre_y = perimeter.centroid.x, perimeter.centroid.y

        
    #     for track_file in self.track_files:
    #         track_data = self.track_data[track_file]

    #         # calculate average x,y for first frame 
    #         # (needs to change such that it time 0 for each unique track compared back to)

    #         for track in track_data['track_id'].unique():
    #             track_unique = track_data[track_data['track_id'] == track].sort_values(by=['frame'])

    #             x0 = track_unique.iloc[0]['x_body']
    #             y0 = track_unique.iloc[0]['y_body']

    #             for i in range(len(track_unique)):

    #                 squared_distance = (track_unique.iloc[i]['x_body'] - x0)**2 + (track_unique.iloc[i]['y_body'] - y0)**2
    #                 # print(squared_distance)

    #                 frame = track_unique.iloc[i]['frame']

    #                 data.append({'time': frame, 'squared distance': squared_distance, 'file': track_file})
        
    #     df = pd.DataFrame(data)
    #     df = df.sort_values(by=['time'], ascending=True)

    #     df.to_csv(os.path.join(self.directory, 'ensemble_msd.csv'), index=False)

    #     return df 



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
    
    # METHOD HOLE_COUNTER: COUNTS NUMBER OF LARVAE IN THE HOLE + IF LARVAE OUTSIDE HOLE IS IT MOVING (IDENTIFYING THOSE CLOSE AND BURROWING BENEATH SURFACE)
    # ALSO CHANGED IT SUCH THAT IT KEEPS THE ORIGINAL DATAFRAME SO I CAN DO A VIDEO CHECK 

    def hole_counter(self):

        count = []

        for match in self.matching_pairs:
            track_file = match['track_file']
            hole_boundary = match['hole_boundary']
            
            if hole_boundary is None:
                print(f"No hole boundary for track file: {track_file}")
                continue

            df = self.track_data[track_file]

            # True/False is the point inside or outside the hole 
            # Apply a buffer to expand the hole boundary slightly # feel like this is buffering the original boundary by 1.5 - unsure why the scale didint work 
            buffered_boundary = hole_boundary.buffer(1.5)  

            # Update the condition to use the buffered boundary
            df['outside_hole'] = df.apply(lambda row: not (buffered_boundary.contains(Point(row['x_body'], row['y_body'])) or 
                                               buffered_boundary.touches(Point(row['x_body'], row['y_body']))), axis=1)

            
            # True/False within 10mm of the hole
            df['within_10mm'] = df.apply(lambda row: buffered_boundary.exterior.distance(Point(row['x_body'], row['y_body'])) <= 10, axis=1)

            # calculate displacement between frames
            # chat gpt said resent index is necessary here due to the apply and groupby * ask michael or callum to explain this properly
            df['displacement'] = df.groupby('track_id').apply(lambda group: np.sqrt((group['x_body'].diff() ** 2) + (group['y_body'].diff() ** 2))).reset_index(drop=True)

            # cumalative displacement
            df['cumulative_displacement'] = df.groupby('track_id')['displacement'].cumsum()

            # cumalative displacement rate - rolling window 20 frames
            # first 19 frames - fillna0 
            df['cumulative_displacement_rate'] = df.groupby('track_id')['cumulative_displacement'].apply(lambda x: x.diff(20) / 20).fillna(0)


            # # cumulative distance over rolling window
            # df['rolling_displacement'] = df.groupby('track_id')['displacement'].transform(lambda x: x.rolling(window=30, min_periods=1).sum()) 

            # df['digging'] = df['within_10mm'] & (df['rolling_displacement'] < 3)

            # standard deviation of x y movement 
            df['x_std'] = df.groupby('track_id')['x_body'].transform(lambda x: x.rolling(window=20, min_periods=1).std())
            df['y_std'] = df.groupby('track_id')['y_body'].transform(lambda x: x.rolling(window=20, min_periods=1).std())
            df['overall_std'] = np.sqrt(df['x_std']**2 + df['y_std']**2)


            df['digging'] = df['within_10mm'] & ((df['cumulative_displacement_rate'] < 0.5) | (df['overall_std'] < 3))

            df['moving_outside'] = df['outside_hole'] & ~df['digging']

            df.to_csv(os.path.join(self.directory, f"{track_file}_hole_data.csv"), index=False)
            print(f"Saving to file: {os.path.join(self.directory, f'{track_file}_hole_data.csv')}")


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

        for match in self.matching_pairs:
            track_file = match['track_file']
            hole_boundary = match['hole_boundary']
            
            if hole_boundary is None:
                print(f"No hole boundary for track file: {track_file}")
                continue

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

        for match in self.matching_pairs:
            track_file = match['track_file']
            hole_boundary = match['hole_boundary']

            if hole_boundary is None:
                print(f"No hole boundary for track file: {track_file}")
                continue

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

        for match in self.matching_pairs:
            track_file = match['track_file']
            hole_boundary = match['hole_boundary']

            if hole_boundary is None:
                print(f"No hole boundary for track file: {track_file}")
                continue

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
    # MODIFYING 2024-10-28

    def number_digging(self, total_larvae):

        dataframe_list = [] 

        for track_file in self.track_files:
            df = self.track_data[track_file]

            df = df.sort_values(by=['track_id', 'frame'])

            # DISTANCE MOVED 

            # Smooth the positions with a rolling window to reduce noise
            df['x'] = df['x_body'].rolling(window=5, min_periods=1).mean()
            df['y'] = df['y_body'].rolling(window=5, min_periods=1).mean()

            # Calculate the difference between consecutive rows for body coordinates
            df['dx'] = df.groupby('track_id')['x'].diff().fillna(0)
            df['dy'] = df.groupby('track_id')['y'].diff().fillna(0)

            # Calculate the Euclidean distance 
            df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2)

            # Create a boolean mask where x,y movement is greater than 0.1 MM 
            df['is_moving'] = df['distance'] > 0.1

            # CUMALTIVE DISTANCE 

            df['cumulative_displacement'] = df.groupby('track_id')['distance'].cumsum()

            df['cumulative_displacement_rate'] = df.groupby('track_id')['cumulative_displacement'].apply(lambda x: x.diff(5) / 5).fillna(0)
            
            # STANDARD DEVIATION OF BODY X, Y COORDINATES 

            df['x_std'] = df.groupby('track_id')['x'].transform(lambda x: x.rolling(window=5, min_periods=1).std())
            df['y_std'] = df.groupby('track_id')['y'].transform(lambda x: x.rolling(window=5, min_periods=1).std())
            df['overall_std'] = np.sqrt(df['x_std']**2 + df['y_std']**2)

            # FINAL MOVEMENT - THEY ARE BOTH QUITE GOOD TBF 

            # df['final_movement'] = (df['is_moving']) | ((df['overall_std'] > 0.09) & (df['cumulative_displacement_rate'] > 0.1))
            df['final_movement'] = (df['cumulative_displacement_rate'] > 0.05) | ((df['overall_std'] > 0.09) & (df['is_moving']))

            # SMOOTH ROLLING WINDOW FOR FINAL MOVEMENT 

            # Apply a rolling window with majority voting to smooth out the 'final_movement' column
            window_size = 20 # Adjust the window size as needed
            df['smoothed_final_movement'] = (df['final_movement']
                                             .rolling(window=window_size, center=True) # centre rolling window
                                             .apply(lambda x: x.sum() >= (window_size / 2)) # Majority 
                                             .fillna(0) # start and end fill with 0 = False
                                             .astype(bool)) # all returned True/False


            # df.to_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/test-number-digging/withrollingwindow.csv')
          
            # Now count the moving frames per frame_idx
            moving_counts = df.groupby('frame')['smoothed_final_movement'].sum().reset_index()

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







    # def radius(self):
    













        # METHOD INITIAL_HOLE_FORMATION: TIME AT WHICH THE FIRST LARVAE BEGINS DIGGING
        # EXTRACTED FROM THE ABOVE !
    






















    # DIGGING IN ISOLATION/ HOW TO COMBINE THE MAN-MADE HOLE WITH THE NUMBER DIGGING OUTSIDE OF THIS
      # IDEK HOW TO DO THIS

    # METHOD CASTING: 

    # METHOD FOR TRACK OVERLAY IMAGES AND VIDEOS 










