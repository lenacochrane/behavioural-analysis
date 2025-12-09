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


class MultiHoleAnalysis:

    def __init__(self, directory):

        self.directory = directory 
        self.coordinate_files = {}
        self.track_files = [] # list of the files 
        self.hole_boundaries = {}
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
        # self.coordinate_files = [f for f in os.listdir(self.directory) if f.endswith('hole.csv')]
        # print(f"Coordinate files: {self.coordinate_files}")

        self.coordinate_files = {}

        all_hole_files = [f for f in os.listdir(self.directory) if f.endswith(".csv") and "_hole" in f]

        for file in all_hole_files:
            # Split into prefix + suffix after "_hole"
            filename, hole = file.split("_hole", 1)  # right might be "3.csv" or ".csv"

            prefix = filename  # e.g. 2025-03-04_15-45-11_td7
            suffix = hole.replace(".csv", "").strip()  # e.g. "3" or ""

            if suffix == "":
                hole_label = "hole1"
            else:
                hole_label = f"hole{suffix}"

            # Insert into grouped dict
            if prefix not in self.coordinate_files:
                self.coordinate_files[prefix] = {}

            self.coordinate_files[prefix][hole_label] = file

        print("Coordinate files grouped by prefix:")
        for prefix, holes in self.coordinate_files.items():
            print(f"  {prefix}: {sorted(holes.keys())}")


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
    
    
    def merged_dataframes(self):
        
        dfs = []
        for track_file in self.track_files:
            df = self.track_data[track_file]
            df['file'] = track_file
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)
        output = os.path.join(self.directory, 'merged.track.feather')
        df.to_feather(output)


    # METHOD HOLE_BOUNDARY: CREATES A POLYGON AROUND THE HOLE BOUNDARY WITH SCALAR OPTION
     # 1. CONVEX HULL: CONVEX SHAPE THAT ENCLOSES A SET OF POINTS (CONTINIOUS BOUNDARY)
     # 2. VERTICES: CORNER POINTS OF THE CONVEX SHAPE
     # 3. POLYGON: GEOMETRIC SHAPE FORMED BY CONNECTING THESE VERTICES

    def hole_boundary(self, scale_factor=1.0):  

        self.hole_boundaries = {}

        for prefix, holes in self.coordinate_files.items():
            self.hole_boundaries[prefix] = {}

            for hole_number, csv_name in holes.items():
                file_path = os.path.join(self.directory, csv_name)

                df = pd.read_csv(file_path, header=None, names=["x", "y"])

                points = df[['x', 'y']].values # values creates numpy array

                hull = ConvexHull(points)  
                hull_points = points[hull.vertices]
                polygon = Polygon(hull_points)

                self.hole_boundaries[prefix][hole_number] = polygon

                # Save WKT alongside the CSV, keeping hole label in the filename
                wkt_string = wkt_dumps(polygon)
                wkt_path = os.path.join(self.directory, csv_name.replace(".csv", ".wkt"))
                with open(wkt_path, "w") as f:
                    f.write(wkt_string)
        
        print("Hole boundaries grouped by prefix:")
        for prefix, holes in self.hole_boundaries.items():
            print(f"  {prefix}: {sorted(holes.keys())}")



    
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
                "hole_boundaries": self.hole_boundaries.get(track_prefix, {}),
                'video_file': None,
                'perimeter_file': None,
                "perimeter_polygon": None,}


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


        print("Matched files:")
        for m in self.matching_pairs:
            print(
                f"  {m['track_file']} -> holes: {sorted(m['hole_boundaries'].keys())}, "
                f"video: {m['video_file']}, perimeter: {m['perimeter_file']}"
            )
    


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

                # --- scale ALL hole polygons (multi-hole aware) ---
                hole_boundaries = match.get("hole_boundaries", {})
                if hole_boundaries:
                    scaled_holes = {}
                    for hole_label, hole_poly in hole_boundaries.items():
                        coords = np.array(hole_poly.exterior.coords)
                        coords *= conversion_factor
                        scaled_holes[hole_label] = Polygon(coords)
                    match["hole_boundaries"] = scaled_holes


                track_file = match['track_file']
                track_data = self.track_data[track_file]

                pixel_columns = ['x_tail', 'y_tail', 'x_body', 'y_body', 'x_head', 'y_head']
                track_data[pixel_columns] = track_data[pixel_columns] * conversion_factor
                self.track_data[track_file] = track_data  # Update the track data.
                print(f"Conversion applied for {track_file} with conversion factor: {conversion_factor:.3f}")
            
            else:
                print(f"no perimeter detected for {match['track_file']}")
  
                conversion_factor = 90 / 1032 # the one i used to use 
                hole_boundaries = match.get("hole_boundaries", {})
                if hole_boundaries:
                    scaled_holes = {}
                    for hole_label, hole_poly in hole_boundaries.items():
                        coords = np.array(hole_poly.exterior.coords)
                        coords *= conversion_factor
                        scaled_holes[hole_label] = Polygon(coords)
                    match["hole_boundaries"] = scaled_holes
                
                track_file = match['track_file']
                track_data = self.track_data[track_file]

                pixel_columns = ['x_tail', 'y_tail', 'x_body', 'y_body', 'x_head', 'y_head']
                track_data[pixel_columns] = track_data[pixel_columns] * conversion_factor
                self.track_data[track_file] = track_data  # Update the track data.
                print(f"Conversion applied for {track_file} with conversion factor: {conversion_factor:.3f}")



    # METHOD COMPUTE_DIGGING: THIS METHOD DETECTS IF LARVAE ARE DIGGING (IN ABSENCE OF MAN-MADE HOLE)

    def compute_digging(self, df):
        df = df.sort_values(['track_id', 'frame']).reset_index(drop=True)

        # Smooth x and y
        df['x'] = (
            df.groupby('track_id', group_keys=False)['x_body']
            .apply(lambda x: x.rolling(window=5, min_periods=1).mean())
        )
        df['y'] = (
            df.groupby('track_id', group_keys=False)['y_body']
            .apply(lambda y: y.rolling(window=5, min_periods=1).mean())
        )

        # Differences
        df['dx'] = df.groupby('track_id')['x'].diff().fillna(0)
        df['dy'] = df.groupby('track_id')['y'].diff().fillna(0)

        # Distance and moving status
        df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2)
        df['is_moving'] = df['distance'] > 0.1

        # Cumulative and std
        df['cumulative_displacement'] = df.groupby('track_id')['distance'].cumsum()
        df['cumulative_displacement_rate'] = df.groupby('track_id')['cumulative_displacement'].apply(lambda x: x.diff(10) / 10).fillna(0)

        df['x_std'] = df.groupby('track_id')['x'].transform(lambda x: x.rolling(window=10, min_periods=1).std())
        df['y_std'] = df.groupby('track_id')['y'].transform(lambda x: x.rolling(window=10, min_periods=1).std())
        df['overall_std'] = np.sqrt(df['x_std']**2 + df['y_std']**2)

        df['movement_score'] = df['cumulative_displacement_rate'] * df['overall_std']

        df['final_movement'] = (df['cumulative_displacement_rate'] > 0.1) | (df['movement_score'] > 0.25)

        ## smoothed final movement
        window_size = 50
        df['digging_status'] = (
            df.groupby('track_id')['final_movement']
            .transform(lambda x: (~x).rolling(window=window_size, center=False).apply(lambda r: r.sum() >= (window_size * 0.8)).fillna(0).astype(bool))
        )

        ### backfilling TRUE for larvae that actually end up digging 

        df['prev'] = (
                df.groupby('track_id')['digging_status']
                .shift(1)
                .fillna(False)
            )
        df['false_true'] = df['digging_status'] & ~df['prev'] # digging status = True ; prev frame digging status = False


        df['future_digging'] = (
        df.groupby('track_id')['digging_status']
        .rolling(window=50, min_periods=50)
        .sum()
        .shift(-49)
        .reset_index(level=0, drop=True)
    )
        df['long_digging'] = df['false_true'] & (df['future_digging'] >= 50)

        # 1) Initialize backfill column
        df['backfill'] = False

        # 2) Loop per track
        for track_id, group in df.groupby('track_id'):
            idx   = group.index
            starts = idx[group.loc[idx, 'long_digging']]
            for s in starts:
                pre = max(idx.min(), s - 30)
                df.loc[pre:s-1, 'backfill'] = True  # back-fill up to the frame *before* 

        df['digging_status'] = df['digging_status'] | df['backfill']

        df.drop(columns=['backfill', 'long_digging', 'false_true', 'future_digging'], inplace=True)

        # df.to_csv(os.path.join(self.directory, 'test.csv'), index=False)

        return df


    











############################################  ---- HOLES ----  ############################################        

    # METHOD COMPUTE_HOLE: DETECTS WHETHER LARVAE IS WITHIN HOLE 

    def compute_hole(self):

        for match in self.matching_pairs:
            track_file = match['track_file']
            hole_boundaries = match.get('hole_boundaries', {})  # <-- NOW A DICT

            if not hole_boundaries:
                print(f"No hole boundary for track file: {track_file}")
                continue

            df = self.track_data[track_file]
            df = df.sort_values(['track_id', 'frame']).reset_index(drop=True)

            
            buffered = {label: poly.buffer(1.5) for label, poly in hole_boundaries.items()}
           
            def in_any_hole(row):
                p = Point(row['x_body'], row['y_body'])
                return any(b.contains(p) or b.touches(p) for b in buffered.values())

            df['in_hole'] = df.apply(in_any_hole, axis=1)

            # OPTIONAL: store which hole (keeps hole1/hole2 identity)
            def which_hole(row):
                p = Point(row['x_body'], row['y_body'])
                for label, b in buffered.items():
                    if b.contains(p) or b.touches(p):
                        return label
                return None

            df['which_hole'] = df.apply(which_hole, axis=1)

            # --- 2) digging near hole (ANY hole) ---
            def within_10mm_any(row):
                p = Point(row['x_body'], row['y_body'])
                return any(b.exterior.distance(p) <= 10 for b in buffered.values())

            df['within_10mm'] = df.apply(within_10mm_any, axis=1)
            

            df['displacement'] = (
                    np.hypot(
                        df.groupby('track_id')['x_body'].diff(),
                        df.groupby('track_id')['y_body'].diff()
                    ).fillna(np.nan))

            df['cumulative_displacement'] = df.groupby('track_id')['displacement'].cumsum()

            df['cumulative_displacement_rate'] = df.groupby('track_id',  group_keys=False)['cumulative_displacement'].apply(lambda x: x.diff(10) / 10).fillna(0)

            df['frame_diff'] = df.groupby('track_id')['frame'].diff() #SPEED FOR LATER CALCULATIONS (same as displacement but just to be sure)
            df['speed'] = df['displacement'] / df['frame_diff']
            df['speed'] = df['speed'].fillna(0)

            df['x_std'] = df.groupby('track_id')['x_body'].transform(lambda x: x.rolling(window=10, min_periods=1).std())
            df['y_std'] = df.groupby('track_id')['y_body'].transform(lambda x: x.rolling(window=10, min_periods=1).std())
            df['overall_std'] = np.sqrt(df['x_std']**2 + df['y_std']**2)

            # df['digging_near_hole'] = df['within_10mm'] & ((df['cumulative_displacement_rate'] < 0.4) | (df['overall_std'] < 0.8))

            df['digging_score'] = df['cumulative_displacement_rate'] * df['overall_std']
            df['digging_near_hole'] = (df['within_10mm']) & (df['digging_score'] < 0.25)


            # 3. combine
            df['hole'] = df['in_hole'] | df['digging_near_hole']

            # 3. threshold for hole rolling window
            df['within_hole'] = (
                df.groupby('track_id')['hole']
                .transform(lambda x: x.rolling(window=30, min_periods=1).sum() >= 30)
                .astype(bool))
            
            # 4. backfils true if larvae IS within hole 
            def expand_on_transitions(series, buffer=30): # transition from False to True
                result = series.copy()
                mask = series.astype(bool).values
                transitions = (mask[1:] & ~mask[:-1])  # detect False → True

                for i in np.where(transitions)[0] + 1:  # shift by 1 because diff is one step ahead
                    start = max(0, i - buffer + 1)
                    result.iloc[start:i + 1] = True

                return result

            df['within_hole'] = df.groupby('track_id')['within_hole'].transform(expand_on_transitions)

            def fill_short_false_gaps(series, max_gap):
                arr = series.values.astype(bool)
                inverse = ~arr  # find False runs
                # Label contiguous False regions
                labeled, num_features = label(inverse)

                # Flip short False runs to True
                for i in range(1, num_features + 1):
                    indices = np.where(labeled == i)[0]
                    if len(indices) <= max_gap:
                        arr[indices] = True

                return pd.Series(arr, index=series.index)

            # Apply it per track
            df['within_hole'] = df.groupby('track_id')['within_hole'].transform(
                lambda s: fill_short_false_gaps(s, max_gap=10)) # below 10 frames for returns filled out
            
            # Apply same logic to larvae outside 10mm zone
            df['digging_outside_hole'] = (
                (df['within_10mm'] == False) &
                (df['digging_score'] < 0.2) &
                (df['frame'] > 80)) # at the start they are slow
            

            def enforce_minimum_duration(series, min_duration=20):
                arr = series.values.astype(bool)
                output = np.zeros_like(arr, dtype=bool)

                start = 0
                while start < len(arr):
                    if arr[start]:
                        end = start
                        while end < len(arr) and arr[end]:
                            end += 1
                        if (end - start) >= min_duration:
                            output[start:end] = True
                        start = end
                    else:
                        start += 1
                return pd.Series(output, index=series.index)
            

            # Apply filtering: only keep runs of >= 20 frames
            df['digging_outside_hole'] = df.groupby('track_id')['digging_outside_hole'].transform(
                lambda s: enforce_minimum_duration(s, min_duration=20))


            self.track_data[track_file] = df
          


    # METHOD HOLE_COUNTER: COUNTS NUMBER OF LARVAE IN HOLE 

    def hole_counter(self):

        data = []

        for match in self.matching_pairs:
            track_file = match['track_file']
            
            df = self.track_data[track_file]
            hole_labels = list(match['hole_boundaries'].keys())

            for frame in df['frame'].unique():
                frame_df = df[df['frame'] == frame]

                per_hole = frame_df['which_hole'].value_counts(dropna=True)

                for hole_label in hole_labels:
                    inside = int(per_hole.get(hole_label, 0))
                    data.append({
                        'file': track_file,
                        'time': frame,
                        'hole': hole_label,
                        'inside_count': inside
                    })

        hole_count = pd.DataFrame(data).sort_values(by=['time'], ascending=True)
        hole_count.to_csv(os.path.join(self.directory, "counts.csv"), index=False)
        return hole_count
    
    
    # METHOD TIME_TO_ENTER: CALCULATES TIME TO ENTER HOLE 
    def time_to_enter(self):

        data = []

        for match in self.matching_pairs:
            track_file = match['track_file']
        
            df = self.track_data[track_file]
      
            for track in df['track_id'].unique():
                unique_track = df[df['track_id'] == track]
                unique_track = unique_track.sort_values(by=['frame'], ascending=True)

                entered = False
                passes = 0
                in_vicinity = False  # track whether we're currently inside the 10mm zone

                for row in unique_track.itertuples():
                    if row.within_hole:
                        data.append({
                            'file': track_file,
                            'track': track,
                            'time': row.frame,
                            'passes_before_entry': passes,
                            'entered_hole': row.which_hole
                        })
                        entered = True
                        break

                    elif row.within_10mm and not row.within_hole:
                        if not in_vicinity:
                            passes += 1
                            in_vicinity = True  # mark entry into the vicinity
                    else:
                        in_vicinity = False  # exited the vicinity
                    
                if not entered:
                        data.append({
                            'file': track_file,
                            'track': track,
                            'time': 3600,
                            'passes_before_entry': passes,
                            'entered_hole': None
                        })

        hole_entry_time = pd.DataFrame(data)
        hole_entry_time = hole_entry_time.sort_values(by=['file'], ascending=True)
        hole_entry_time.to_csv(os.path.join(self.directory, 'entry_time.csv'), index=False)

        return hole_entry_time



######



    # DEF RETURNS: CALCULATES THE NUMBER OF LARVAE WHICH RETURN TO THE HOLE AND THE TIME TAKEN 
    def returns(self):

        data = []

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file]

            for track in df['track_id'].unique():
                track_df = df[df['track_id'] == track].sort_values(by='frame').reset_index(drop=True)
                states = track_df['within_hole'].astype(bool).values
                frames = track_df['frame'].values

                i = 1
                while i < len(states):
                    # Detect True → False transition (exit)
                    if states[i - 1] and not states[i]:
                        exit_frame = frames[i]

                        # Now search for the next False → True (re-entry)
                        j = i + 1
                        while j < len(states):
                            if not states[j - 1] and states[j]:
                                return_frame = frames[j]
                                return_time = return_frame - exit_frame
    
                                return_distance = track_df.loc[
                                        (track_df['frame'] > exit_frame) & (track_df['frame'] <= return_frame),
                                        'displacement'].sum() #displacement was calculated in the compute_hole 

                                data.append({
                                    'file': track_file,
                                    'track': track,
                                    'exit frame': exit_frame,
                                    'return frame': return_frame,
                                    'return_time': return_time,
                                    'distance_covered': return_distance

                                })

                                i = j  # move outer loop forward after return
                                break
                            j += 1
                    i += 1

        df_returns = pd.DataFrame(data)
        df_returns = df_returns.sort_values(by=['file', 'track', 'exit frame'])
        df_returns.to_csv(os.path.join(self.directory, 'returns.csv'), index=False)
        return df_returns
    

    # METHOD HOLE_DEPARTURES: CALCULATES THE NUMBER OF LARVAE LEAVING THE HOLE

    def hole_departures(self):

        data = []

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file]

            for track in df['track_id'].unique():
                track_df = df[df['track_id'] == track].sort_values(by='frame').reset_index(drop=True)
                states = track_df['within_hole'].astype(bool).values
                frames = track_df['frame'].values

                count = 0

                i = 1
                while i < len(states):
                    # Detect True → False transition (exit)
                    if states[i - 1] and not states[i]:
                        count += 1
                    i += 1

                data.append({'file': track_file, 'track': track, 'departures': count})

        hole_departures = pd.DataFrame(data)
        hole_departures = hole_departures.sort_values(by=['track'], ascending=True)
        hole_departures.to_csv(os.path.join(self.directory, 'hole_departures.csv'), index=False)

        return hole_departures
    
    # METHOD HOLE_ENTRY-DEPARTURE_LATENCY:

    def hole_entry_departure_latency(self):

        data = []

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file].sort_values(by='frame')

            # Get list of unique frames
            all_frames = df['frame'].unique()
            all_frames.sort()

            # Track previous inside state
            prev_inside_ids = set()

            for i, frame in enumerate(all_frames):
                current_df = df[df['frame'] == frame]
                current_inside_ids = set(current_df[current_df['within_hole']]['track_id'])

                # Detect new entries: tracks that were not inside before but are now
                new_entries = current_inside_ids - prev_inside_ids

                for entrant in new_entries:
                    # Count how many are in hole including this new one
                    count_at_entry = len(current_inside_ids)

                    # Now search forward for first time a larva leaves
                    latency = None
                    departure_happened = False

                    for j in range(i + 1, len(all_frames)):
                        next_frame = all_frames[j]
                        next_df = df[df['frame'] == next_frame]
                        next_inside_ids = set(next_df[next_df['within_hole']]['track_id'])

                        if len(next_inside_ids) < len(current_inside_ids):
                            latency = next_frame - frame
                            departure_happened = True
                            break

                    data.append({
                        'file': track_file,
                        'entry_frame': frame,
                        'number_in_hole_at_entry': count_at_entry,
                        'latency_to_next_departure': latency,
                        'departure_happened': departure_happened
                    })

                # Update previous state
                prev_inside_ids = current_inside_ids

        # Convert to DataFrame
        result = pd.DataFrame(data)
        result = result.sort_values(by=['file', 'entry_frame'])
        result.to_csv(os.path.join(self.directory, 'hole_entry_departure_latency.csv'), index=False)



    # METHOD SPEED_HOLE():

    def speed_hole(self):

        summary = []

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file]

            for track_id, group in df.groupby('track_id'):
                group = group.sort_values('frame')
                # Get first entry frame
                hole_frames = group[group['within_hole'] == True]['frame']
                if hole_frames.empty:
                    continue  # skip if never entered

                first_entry = hole_frames.iloc[0]

                # # Speeds before entering the hole
                # before_mask = (group['frame'] < first_entry)
                # speeds_before = group.loc[before_mask, 'speed']

                # # Speeds after entering (but not in hole)
                # after_mask = (group['frame'] > first_entry) & (~group['within_hole'])
                # speeds_after = group.loc[after_mask, 'speed']

                before_mask = (
                    (group['frame'] < first_entry) &
                    (~group['digging_outside_hole'])
                )
                speeds_before = group.loc[before_mask, 'speed']

                # Speeds after entering (exclude digging outside hole)
                after_mask = (
                    (group['frame'] > first_entry) &
                    (~group['within_hole']) &
                    (~group['digging_outside_hole'])
                )
                speeds_after = group.loc[after_mask, 'speed']

                # Compare individual larva behavior before/after

                summary.append({
                    'file': track_file,
                    'track_id': track_id,
                    'first_entry_frame': first_entry,
                    'mean_speed_before': speeds_before.mean(),
                    'mean_speed_after': speeds_after.mean(),
                    'n_frames_before': len(speeds_before),
                    'n_frames_after': len(speeds_after)
                })
        
        speed_comparison = pd.DataFrame(summary)
        speed_comparison.to_csv(os.path.join(self.directory, 'hole_speed.csv'), index=False)



    # METHOD DISTANCE_FROM_HOLE: CALCULATES DISTANCES FROM HOLE CENTROID  

    def distance_from_hole(self): 

        data = []

        for match in self.matching_pairs:  
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
                data.append({'time': row.frame, 'distance_from_hole': distance, 'speed': row.speed, 'file': track_file})
    
        if not data:
            print("No distances calculated, check data")
        else:

            distance_hole_over_time = pd.DataFrame(data)
            distance_hole_over_time = distance_hole_over_time.sort_values(by=['time'], ascending=True)
            distance_hole_over_time.to_csv(os.path.join(self.directory, 'hole_distance.csv'), index=False)


   






    # METHOD POTENTIAL_ENTRIES: NUMBER OF TIMES LARVAE ARE WITHIN VISCINITY OF HOLE BUT CHOSE NOT TO ENTER 
 
    def hole_entry_probability(self):
        data = []
        min_near_frames = 15

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file].sort_values(['track_id', 'frame'])

            for track_id, track_df in df.groupby('track_id'):
                near = track_df['within_10mm'].values.astype(bool)
                hole = track_df['within_hole'].values.astype(bool)
                frames = track_df['frame'].values

                i = 0
                while i <= len(near) - min_near_frames -1:
                    if all(near[i:i + min_near_frames]):

                        decision_frame = frames[i + min_near_frames]

                        # Look ahead to see if they enter the hole
                        entered = any(hole[i + min_near_frames : i + min_near_frames + 30])

                        others_inside = df[
                            (df['frame'] == decision_frame) &     # Only consider rows for the current decision frame
                            (df['track_id'] != track_id) &        # Exclude the current larva (we want others only)
                            (df['within_hole'])                   # Only count those larvae currently inside the hole
                        ].shape[0]                                # Get the number of such rows = number of others in hole

                        data.append({
                            'file': track_file,
                            'track': track_id,
                            'decision_frame': decision_frame,
                            'entry': entered,
                            'number_inside_hole': others_inside
                        })
                        
                        if entered:
                            # Skip until they leave both the hole AND the 10mm vicinity
                            j = i + min_near_frames + 30
                            while j < len(hole) and (hole[j] or near[j]):
                                j += 1
                            i = j
                        else:
                            # They didn't enter — skip ahead until they leave the 10mm vicinity
                            j = i + min_near_frames
                            while j < len(near) and near[j]:
                                j += 1
                            i = j
                    
                    else:
                        i += 1

        pd.DataFrame(data).to_csv(os.path.join(self.directory, 'hole_probability.csv'), index=False)

    
    # METHOD HOLE_STATUS: STATUS OF LARVAE REGARDING HOLE: NIAVE OR 

    def hole_status(self):

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file].sort_values(['track_id', 'frame']).copy()

            entry_frame_map = (
                df[df['within_hole']]
                .groupby('track_id')['frame']
                .min()
                .to_dict()
            )

            df['entry_frame'] = df['track_id'].map(entry_frame_map)
            

            df['hole_status'] = np.where(
                df['entry_frame'].isna(), 'naive',               # never entered
                np.where(df['frame'] < df['entry_frame'], 'naive', 'exposed')
            )

            df = df.drop(columns=['entry_frame'])
    
            self.track_data[track_file] = df
    

    # HOLE_STATUS_INTERACTION: TYPE OF INTERACTION OCCURING BETWEEN LARVAE 

    def hole_status_interactions(self, threshold=1): # modified interaction_type_bout method

        max_gap = 4

        def unify_interaction_type(part1, part2):
            return '_'.join(sorted([part1, part2]))
        
        def get_closest_part_pair(coords, id1, id2):
            min_dist = float('inf')
            closest = None
            for p1 in coords:
                for p2 in coords:
                    coord1 = coords[p1].get(id1)
                    coord2 = coords[p2].get(id2)
                    if coord1 is None or coord2 is None:
                        continue
                    dist = np.linalg.norm(coord1 - coord2)
                    if dist < min_dist:
                        min_dist = dist
                        closest = unify_interaction_type(p1, p2)
            return closest

        body_parts = ['head', 'body', 'tail']
        interaction_pairs = list(itertools.product(body_parts, body_parts))
        unified_types = sorted(set(unify_interaction_type(p1, p2) for p1, p2 in interaction_pairs))

        bouts = []

        for track_file in self.track_files:
            df = self.track_data[track_file].copy()
            df.sort_values(by='frame', inplace=True)

            active_bouts = {}
            bout_counter = 0

            for frame in df['frame'].unique():
                frame_data = df[df['frame'] == frame]
                track_ids = frame_data['track_id'].unique()

                # Coordinates lookup
                coords = {
                    part: {
                        row['track_id']: np.array([row[f'x_{part}'], row[f'y_{part}']])
                        for _, row in frame_data.iterrows()
                    }
                    for part in body_parts
                }

                interacting_pairs = {}

                for id1, id2 in itertools.combinations(track_ids, 2):
                    interactions = []

                    for part1, part2 in interaction_pairs:
                        coord1 = coords[part1].get(id1)
                        coord2 = coords[part2].get(id2)
                        if coord1 is None or coord2 is None:
                            continue
                        dist = np.linalg.norm(coord1 - coord2)
                        if dist < threshold:
                            interactions.append(unify_interaction_type(part1, part2))

                    if interactions:
                        # interacting_pairs[(id1, id2)] = interactions
                        pair_key = tuple(sorted((id1, id2)))
                        interacting_pairs[pair_key] = interactions

                current_pairs = set(interacting_pairs)

                # Process ended or gap-extending bouts
                for pair in list(active_bouts):
                    if pair not in current_pairs:
                        active_bouts[pair]['gap_count'] += 1
                        if active_bouts[pair]['gap_count'] <= max_gap:
                            id1, id2 = pair
                            closest_type = None
                            min_dist = float('inf')
                            for part1, part2 in interaction_pairs:
                                coord1 = coords[part1].get(id1)
                                coord2 = coords[part2].get(id2)
                                if coord1 is None or coord2 is None:
                                    continue
                                dist = np.linalg.norm(coord1 - coord2)
                                if dist < min_dist:
                                    min_dist = dist
                                    closest_type = unify_interaction_type(part1, part2)
                            if closest_type:
                                active_bouts[pair]['interactions'].append(closest_type)
                                active_bouts[pair]['end_frame'] = frame
                        else:
                            # End bout
                            bout = active_bouts.pop(pair)
                            start, end = bout['start_frame'], bout['end_frame']
                            duration = end - start + 1
                            interactions = bout['interactions']
                            interaction_counts = Counter(interactions)
                            initial_type = interactions[0]
                            predominant_type = interaction_counts.most_common(1)[0][0]
                            status1 = df[(df['track_id'] == pair[0]) & (df['frame'] == start)]['hole_status'].values[0]
                            status2 = df[(df['track_id'] == pair[1]) & (df['frame'] == start)]['hole_status'].values[0]
                            hole_status_pair = '-'.join(sorted([status1, status2]))
                            bout_data = {
                                'file': track_file,
                                'bout_id': bout['bout_id'],
                                'track_1': pair[0],
                                'track_2': pair[1],
                                'start_frame': start,
                                'end_frame': end,
                                'interaction_duration': duration,
                                'initial_type': initial_type,
                                'predominant_type': predominant_type,
                                'hole_status_pair': hole_status_pair,
                            }
                            for t in unified_types:
                                bout_data[t] = interaction_counts.get(t, 0)
                            bouts.append(bout_data)

                # Update or start new bouts
                for pair, interactions in interacting_pairs.items():
                    if pair in active_bouts:
                        active_bouts[pair]['end_frame'] = frame
                        active_bouts[pair]['interactions'].extend(interactions)
                        active_bouts[pair]['gap_count'] = 0
                    else:
                        active_bouts[pair] = {
                            'bout_id': bout_counter,
                            'start_frame': frame,
                            'end_frame': frame,
                            'interactions': interactions.copy(),
                            'gap_count': 0
                        }
                        bout_counter += 1

            # Finalize remaining bouts
            for pair, bout in active_bouts.items():
                start, end = bout['start_frame'], bout['end_frame']
                duration = end - start + 1
                interactions = bout['interactions']
                interaction_counts = Counter(interactions)
                initial_type = interactions[0]
                predominant_type = interaction_counts.most_common(1)[0][0]
                status1 = df[(df['track_id'] == pair[0]) & (df['frame'] == start)]['hole_status'].values[0]
                status2 = df[(df['track_id'] == pair[1]) & (df['frame'] == start)]['hole_status'].values[0]
                hole_status_pair = '-'.join(sorted([status1, status2]))
                bout_data = {
                    'file': track_file,
                    'bout_id': bout['bout_id'],
                    'track_1': pair[0],
                    'track_2': pair[1],
                    'start_frame': start,
                    'end_frame': end,
                    'interaction_duration': duration,
                    'initial_type': initial_type,
                    'predominant_type': predominant_type,
                    'hole_status_pair': hole_status_pair,
                }
                for t in unified_types:
                    bout_data[t] = interaction_counts.get(t, 0)
                bouts.append(bout_data)

        bout_df = pd.DataFrame(bouts).sort_values(by=['file', 'bout_id'])
        bout_df.to_csv(os.path.join(self.directory, 'interaction_status_type.csv'), index=False)
        return bout_df
    



    def interactions_return(self, threshold=1):

        max_gap = 4

        def unify_interaction_type(p1, p2):
            return '_'.join(sorted([p1, p2]))
        
        def get_closest_part_pair(coords, id1, id2):
            min_dist = float('inf')
            closest = None
            for p1 in coords:
                for p2 in coords:
                    coord1 = coords[p1].get(id1)
                    coord2 = coords[p2].get(id2)
                    if coord1 is None or coord2 is None:
                        continue
                    dist = np.linalg.norm(coord1 - coord2)
                    if dist < min_dist:
                        min_dist = dist
                        closest = unify_interaction_type(p1, p2)
            return closest

        body_parts = ['head', 'body', 'tail']
        interaction_pairs = list(itertools.product(body_parts, body_parts))
        unified_types = sorted(set(unify_interaction_type(p1, p2) for p1, p2 in interaction_pairs))

        bouts = []

        for track_file in self.track_files:
            df = self.track_data[track_file].copy()
            df.sort_values(by='frame', inplace=True)

            # Get exposed larvae (they have an entry_frame)
            entry_frames = df[df['within_hole']].groupby('track_id')['frame'].min()
            exposed_ids = entry_frames.index.tolist()

            for larva_id in exposed_ids: #filter for larvae which entered the hole
                sub = df[df['track_id'] == larva_id].copy()
                sub = sub.sort_values(by='frame')

                # Find all exit (True→False) and return (False→True) transitions
                sub['prev'] = sub['within_hole'].shift()
                exits = sub[(sub['within_hole'] == False) & (sub['prev'] == True)]
                returns = sub[(sub['within_hole'] == True) & (sub['prev'] == False)]

                exit_frames = exits['frame'].values
                return_frames = returns['frame'].values

                # Handle matching exits to returns
                for i, exit_frame in enumerate(exit_frames):
                    later_returns = return_frames[return_frames > exit_frame]
                    if len(later_returns) > 0:
                        reentry_frame = later_returns[0]
                        returned = True
                        return_duration = reentry_frame - exit_frame + 1
                    else:
                        reentry_frame = sub['frame'].max()
                        returned = False
                        return_duration = False
                        
                    # Get all frames between exit and reentry
                    window = df[
                        (df['frame'] >= exit_frame) & 
                        (df['frame'] <= reentry_frame)
                    ]

                    # Get interaction bouts in this window involving the exiting larva
                    active_bouts = {}
                    bout_counter = 0
                    bouts_this_exit = []

                    for frame in window['frame'].unique():
                        frame_data = window[window['frame'] == frame]
                        track_ids = frame_data['track_id'].unique()

                        coords = {
                            part: {
                                row['track_id']: np.array([row[f'x_{part}'], row[f'y_{part}']])
                                for _, row in frame_data.iterrows()
                            }
                            for part in body_parts
                        }

                        interacting_pairs = {}
                        for id1, id2 in itertools.combinations(track_ids, 2):
                            if larva_id not in (id1, id2):
                                continue  # Only care about interactions involving this larva

                            partner_id = id2 if id1 == larva_id else id1  # Get the other larva
                            partner_row = frame_data[frame_data['track_id'] == partner_id]

                            if partner_row.empty or partner_row['within_hole'].values[0]:
                                continue  # Skip if partner is in hole

                            interactions = []
                            for part1, part2 in interaction_pairs:
                                coord1 = coords[part1].get(id1)
                                coord2 = coords[part2].get(id2)
                                if coord1 is None or coord2 is None:
                                    continue
                                if np.linalg.norm(coord1 - coord2) < threshold:
                                    interactions.append(unify_interaction_type(part1, part2))

                            if interactions:
                                # interacting_pairs[(id1, id2)] = interactions
                                pair_key = tuple(sorted((id1, id2)))
                                interacting_pairs[pair_key] = interactions


                        current_pairs = set(interacting_pairs)
                        

                        # Process ended or extended gaps
                        for pair in list(active_bouts):
                            if pair not in current_pairs:
                                active_bouts[pair]['gap_count'] += 1
                                if active_bouts[pair]['gap_count'] <= max_gap:
                                    id1, id2 = pair
                                    fallback = get_closest_part_pair(coords, id1, id2)
                                    if fallback:
                                        active_bouts[pair]['interactions'].append(fallback)
                                    active_bouts[pair]['end_frame'] = frame
                                else:
                                    # End bout
                                    bout = active_bouts.pop(pair)
                                    start, end = bout['start_frame'], bout['end_frame']
                                    interactions = bout['interactions']
                                    interaction_counts = Counter(interactions)
                                    initial_type = interactions[0]
                                    predominant_type = interaction_counts.most_common(1)[0][0]
                                    partner = [x for x in pair if x != larva_id][0]
                                    partner_status = df[(df['track_id'] == partner) & (df['frame'] == start)]['hole_status'].values[0]

                                    bout_data = {
                                        'file': track_file,
                                        'exiting_larva': larva_id,
                                        'exit_index': i,
                                        'returned_to_hole': returned,
                                        'start_frame': start,
                                        'end_frame': end,
                                        'return_time': return_duration,
                                        'interacted': True,
                                        'partner': partner,
                                        'partner_status': partner_status,
                                        'duration': end - start + 1,
                                        'initial_type': initial_type,
                                        'predominant_type': predominant_type,
                                    }
                                    for t in unified_types:
                                        bout_data[t] = interaction_counts.get(t, 0)
                                    bouts_this_exit.append(bout_data)

                        # Extend or start bouts
                        for pair, interactions in interacting_pairs.items():
                            if pair in active_bouts:
                                active_bouts[pair]['end_frame'] = frame
                                active_bouts[pair]['interactions'].extend(interactions)
                                active_bouts[pair]['gap_count'] = 0
                            else:
                                active_bouts[pair] = {
                                    'bout_id': bout_counter,
                                    'start_frame': frame,
                                    'end_frame': frame,
                                    'interactions': interactions.copy(),
                                    'gap_count': 0
                                }
                                bout_counter += 1

                    # Close remaining bouts
                    for pair, bout in active_bouts.items():
                        start, end = bout['start_frame'], bout['end_frame']
                        interactions = bout['interactions']
                        interaction_counts = Counter(interactions)
                        initial_type = interactions[0]
                        predominant_type = interaction_counts.most_common(1)[0][0]
                        partner = [x for x in pair if x != larva_id][0]
                        partner_status = df[(df['track_id'] == partner) & (df['frame'] == start)]['hole_status'].values[0]

                        bout_data = {
                            'file': track_file,
                            'exiting_larva': larva_id,
                            'exit_index': i,
                            'returned_to_hole': returned,
                            'start_frame': start,
                            'end_frame': end,
                            'return_time': return_duration,
                            'interacted': True,
                            'partner': partner,
                            'partner_status': partner_status,
    
                            'duration': end - start + 1,
                            'initial_type': initial_type,
                            'predominant_type': predominant_type,
                        }
                        for t in unified_types:
                            bout_data[t] = interaction_counts.get(t, 0)
                        bouts_this_exit.append(bout_data)

                    if not bouts_this_exit:
                        fallback = {
                            'file': track_file,
                            'exiting_larva': larva_id,
                            'exit_index': i,
                            'returned_to_hole': returned,
                            'start_frame': exit_frame,
                            'end_frame': reentry_frame,
                            'return_time': return_duration,
                            'interacted': False,
                            'partner': None,
                            'partner_status': None,
                            'duration': None,
                            'initial_type': None,
                            'predominant_type': None,
                        }
                        for t in unified_types:
                            fallback[t] = None
                        bouts_this_exit.append(fallback)

                    bouts.extend(bouts_this_exit)

        bout_df = pd.DataFrame(bouts).sort_values(by=['file', 'exiting_larva', 'exit_index', 'start_frame'])
        bout_df.to_csv(os.path.join(self.directory, 'interactions_return.csv'), index=False)
        return bout_df



                        
                            







if __name__ == "__main__":
    directory = "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/holes/N10-4-HOLE/SOCIAL-ISOLATION"
    mha = MultiHoleAnalysis(directory)
    mha.compute_hole()
    mha.hole_counter()
    mha.time_to_enter()

    directory = "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/holes/N10-4-HOLE/GROUP-HOUSED"
    mha = MultiHoleAnalysis(directory)
    mha.compute_hole()
    mha.hole_counter()
    mha.time_to_enter()












