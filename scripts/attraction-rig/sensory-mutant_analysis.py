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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
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


class SensoryMutantAnalysis:

    def __init__(self, directory):

        self.directory = directory 
        self.track_files = [] # list of the files 
        self.matching_pairs = []
        self.track_data = {}  # Initialize the track_data dictionary # actually has the data so we dont have to keep reloading 
        
        self.perimeter()
        self.tracks()
        self.match_files()

        self.use_shorten = True 
        self.shorten_duration = None



    # METHOD TRACKS: IDENTIES AND STORES THE SLEAP TRACK FILES; TRACK DATA IS SUBSEQUENTLY READ  

    def tracks(self):
        # 2024-04-30_14-31-44_td5.000_2024-04-30_14-31-44_td5.analysis.csv
        self.track_files = [f for f in os.listdir(self.directory) if f.endswith('tracks.feather')]
    
        for track_file in self.track_files: 
            track_path = os.path.join(self.directory, track_file)
            df = pd.read_feather(track_path)
            self.track_data[track_file] = df
    
   # METHOD SHORTEN: OPTIONAL METHOD TO SHORTEN THE TRACK FILES TO INCLUDE UP TO A CERTAIN FRAME  
    
    def shorten(self, frame=-1):

        for track_file in self.track_files:

            df = self.track_data[track_file]
            df = df[df['frame'] <= frame]
            self.track_data[track_file] = df # update the track data 

        self.use_shorten = True
        self.shorten_duration = frame  # e.g., 600

        
    ### METHOD DIGGING_MASK: FILTERS FOR NON-DIGGING LARVAE

    def digging_mask(self):

        for track_file in self.track_files:
            df = self.track_data[track_file]
            df = self.compute_digging(df)
            df.to_csv(os.path.join(self.directory, 'digging.csv'), index=False) # get rid 
            self.track_data[track_file] = df[df['digging_status'] == False].copy()
    
    

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
                'video_file': None,
                'perimeter_file': None}


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
                match['centre'] = np.array([
                    scaled_perimeter_polygon.centroid.x,
                    scaled_perimeter_polygon.centroid.y
                ])


                track_file = match['track_file']
                track_data = self.track_data[track_file]

                pixel_columns = ['x_tail', 'y_tail', 'x_body', 'y_body', 'x_head', 'y_head']
                track_data[pixel_columns] = track_data[pixel_columns] * conversion_factor
                self.track_data[track_file] = track_data  # Update the track data.
                print(f"Conversion applied for {track_file} with conversion factor: {conversion_factor:.3f}")
            
            else:
                print(f"no perimeter detected for {match['track_file']}")
  
                conversion_factor = 90 / 1032 # the one i used to use 
                match['centre'] = np.array([700, 700]) * conversion_factor
                
                track_file = match['track_file']
                track_data = self.track_data[track_file]

                pixel_columns = ['x_tail', 'y_tail', 'x_body', 'y_body', 'x_head', 'y_head']
                track_data[pixel_columns] = track_data[pixel_columns] * conversion_factor
                self.track_data[track_file] = track_data  # Update the track data.
                print(f"Conversion applied for {track_file} with conversion factor: {conversion_factor:.3f}")
    

    def compute_digging(self, df):
        df = df.sort_values(['track_id', 'frame']).reset_index(drop=True)

        def close_short_gaps(mask, max_gap=20):
            mask = np.asarray(mask, dtype=bool).copy()
            i = 0
            while i < len(mask):
                if mask[i]:
                    i += 1
                    continue

                start = i
                while i < len(mask) and not mask[i]:
                    i += 1

                if start > 0 and i < len(mask) and (i - start) <= max_gap:
                    mask[start:i] = True

            return mask

        # Smooth body position before calculating local confinement.
        df['x'] = (
            df.groupby('track_id')['x_body']
            .transform(lambda x: x.rolling(window=5, min_periods=1, center=True).median())
        )
        df['y'] = (
            df.groupby('track_id')['y_body']
            .transform(lambda y: y.rolling(window=5, min_periods=1, center=True).median())
        )

        # Differences and frame-to-frame speed.
        df['dx'] = df.groupby('track_id')['x'].diff().fillna(0)
        df['dy'] = df.groupby('track_id')['y'].diff().fillna(0)
        df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2)
        df['is_moving'] = df['distance'] > 0.1

        grouped = df.groupby('track_id')
        for window in (20, 30, 60):
            df[f'path_{window}'] = (
                grouped['distance']
                .transform(lambda x: x.rolling(window=window, min_periods=1).sum())
            )
            df[f'displacement_{window}'] = np.sqrt(
                (df['x'] - grouped['x'].shift(window))**2
                + (df['y'] - grouped['y'].shift(window))**2
            )
            x_std = grouped['x'].transform(lambda x: x.rolling(window=window, min_periods=1).std())
            y_std = grouped['y'].transform(lambda y: y.rolling(window=window, min_periods=1).std())
            df[f'position_std_{window}'] = np.sqrt(x_std**2 + y_std**2)

        df['pose_length'] = (
            np.sqrt((df['x_head'] - df['x_body'])**2 + (df['y_head'] - df['y_body'])**2)
            + np.sqrt((df['x_body'] - df['x_tail'])**2 + (df['y_body'] - df['y_tail'])**2)
        )
        df['pose_length_smooth'] = (
            grouped['pose_length']
            .transform(lambda x: x.rolling(window=20, min_periods=1, center=True).median())
        )

        df['confined_movement'] = (
            (
                (df['path_20'] <= 24)
                & (df['displacement_20'] <= 20)
                & (df['position_std_20'] <= 7)
            )
            | (
                (df['path_30'] <= 35)
                & (df['displacement_30'] <= 26)
                & (df['position_std_30'] <= 9)
            )
            | (
                (df['path_60'] <= 55)
                & (df['displacement_60'] <= 30)
                & (df['position_std_60'] <= 12)
            )
        ).fillna(False)
        df['compact_posture'] = (df['pose_length_smooth'] <= 38).fillna(False)
        df['digging_status'] = False

        min_run = 280 # 300 for agarose # trying 280 for food - plates
        max_gap = 20
        backfill = 20
        min_after_compact = 120

        for track_id, group in df.groupby('track_id'):
            idx = group.index.to_numpy()
            confined = close_short_gaps(group['confined_movement'].to_numpy(), max_gap=max_gap)
            compact = group['compact_posture'].to_numpy()
            track_digging = np.zeros(len(group), dtype=bool)

            i = 0
            while i < len(group):
                if not confined[i]:
                    i += 1
                    continue

                start = i
                while i < len(group) and confined[i]:
                    i += 1
                end = i

                if (end - start) < min_run:
                    continue

                compact_idx = np.where(compact[start:end])[0]
                if len(compact_idx) == 0:
                    continue

                onset = start + compact_idx[0]
                if (end - onset) < min_after_compact:
                    continue

                onset = max(0, onset - backfill)
                track_digging[onset:end] = True

            df.loc[idx, 'digging_status'] = track_digging

        # df.to_csv(os.path.join(self.directory, 'test.csv'), index=False)

        return df

    

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

                    data.append({'time': time2, 'speed': speed_value, 'file': track_file, 'track': track})
    
        speed_over_time = pd.DataFrame(data)
        speed_over_time = speed_over_time.sort_values(by=['file', 'track', 'time'], ascending=True)

        filename = f"speed_over_time.csv"

        speed_over_time.to_csv(os.path.join(self.directory, filename), index=False)

        return speed_over_time


    def distance_travelled(self, max_frame_gap=2):
        """Calculate cumulative body-centre path length for every larva.

        Consecutive body positions are joined only when they are no more than
        ``max_frame_gap`` frames apart. This prevents a long tracking gap from
        being counted as a single movement. The result contains one row per
        tracked frame, allowing the frame at which a distance was reached to
        be identified. After ``conversion()`` has been called, the returned
        distances are in millimetres.
        """

        data = []

        for track_file in self.track_files:
            track_data = self.track_data[track_file]

            for track in track_data['track_id'].dropna().unique():
                positions = (
                    track_data.loc[
                        track_data['track_id'] == track,
                        ['frame', 'x_body', 'y_body']
                    ]
                    .dropna(subset=['frame'])
                    .sort_values('frame')
                    .copy()
                )

                frame_gap = positions['frame'].diff()
                step_distance = np.hypot(
                    positions['x_body'].diff(),
                    positions['y_body'].diff()
                )
                valid_step = (frame_gap > 0) & (frame_gap <= max_frame_gap)
                positions['distance_travelled'] = (
                    step_distance.where(valid_step, 0).fillna(0).cumsum()
                )
                positions['file'] = track_file
                positions['track'] = track

                data.append(positions[[
                    'file',
                    'track',
                    'frame',
                    'distance_travelled'
                ]])

        if data:
            distance_df = pd.concat(data, ignore_index=True)
        else:
            distance_df = pd.DataFrame(columns=[
                'file', 'track', 'frame', 'distance_travelled'
            ])

        distance_df = distance_df.sort_values(
            by=['file', 'track', 'frame'],
            ascending=True
        ).reset_index(drop=True)

        if self.use_shorten and self.shorten_duration is not None:
            suffix = f"_{self.shorten_duration}"
        else:
            suffix = ""

        filename = f"distance_travelled{suffix}.csv"
        distance_df.to_csv(os.path.join(self.directory, filename), index=False)

        return distance_df



    def size(self):

        data = []

        for match in self.matching_pairs:
            track_file = match['track_file']
            track_data = self.track_data[track_file].copy()

            centre = match.get('centre')
            if centre is None:
                perimeter_polygon = match.get('perimeter_polygon')
                if perimeter_polygon:
                    centre = np.array([perimeter_polygon.centroid.x, perimeter_polygon.centroid.y])
                else:
                    centre = np.array([700, 700])

            track_data['size'] = (
                np.sqrt(
                    (track_data['x_head'] - track_data['x_body'])**2
                    + (track_data['y_head'] - track_data['y_body'])**2
                )
                + np.sqrt(
                    (track_data['x_body'] - track_data['x_tail'])**2
                    + (track_data['y_body'] - track_data['y_tail'])**2
                )
            )

            track_data['distance_from_centre'] = np.sqrt(
                (track_data['x_body'] - centre[0])**2
                + (track_data['y_body'] - centre[1])**2
            )

            data.append(
                track_data[[
                    'frame',
                    'track_id',
                    'size',
                    'distance_from_centre'
                ]].assign(file=track_file)
            )

        size_df = pd.concat(data, ignore_index=True)
        size_df = size_df.rename(columns={'track_id': 'track'})
        size_df = size_df[[
            'file',
            'frame',
            'track',
            'size',
            'distance_from_centre'
        ]]
        size_df = size_df.sort_values(by=['file', 'track', 'frame'], ascending=True)


        filename = f"size.csv"
        size_df.to_csv(os.path.join(self.directory, filename), index=False)

        return size_df



    def nearest_neighbour(self):

        dfs = []

        parts = ['head', 'body', 'tail']

        def unify_interaction_type(p1, p2):
            return '-'.join(sorted([p1, p2]))

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file]

            df = df.sort_values(by='frame', ascending=True)
            df['filename'] = track_file

            # --------------------------------------------------
            # SPEED + ACCELERATION
            # --------------------------------------------------
            def speed(group, x, y):
                dx = group[x].diff()
                dy = group[y].diff()
                dist = np.sqrt(dx**2 + dy**2)
                dt = group['frame'].diff()
                return dist / dt.replace(0, np.nan)

            df['speed'] = (
                df.groupby('track_id')
                .apply(lambda g: speed(g, 'x_body', 'y_body'))
                .reset_index(level=0, drop=True)
            )

            df['acceleration'] = (
                df.groupby('track_id')['speed'].diff()
                / df.groupby('track_id')['frame'].diff()
            )

            # --------------------------------------------------
            # BODY ANGLE (UNCHANGED)
            # --------------------------------------------------
            df['v1_x'] = df['x_head'] - df['x_body']
            df['v1_y'] = df['y_head'] - df['y_body']
            df['v2_x'] = df['x_tail'] - df['x_body']
            df['v2_y'] = df['y_tail'] - df['y_body']

            def calculate_angle(df, v1_x, v1_y, v2_x, v2_y):
                dot = df[v1_x] * df[v2_x] + df[v1_y] * df[v2_y]
                mag1 = np.hypot(df[v1_x], df[v1_y])
                mag2 = np.hypot(df[v2_x], df[v2_y])
                cos = np.clip(dot / (mag1 * mag2), -1, 1)
                return np.degrees(np.arccos(cos))

            df['angle'] = calculate_angle(df, 'v1_x', 'v1_y', 'v2_x', 'v2_y')

            # --------------------------------------------------
            # OUTPUT COLUMNS
            # --------------------------------------------------
            df['body-body'] = np.nan

            df['other_id'] = np.nan
            df['closest_node_interaction'] = np.nan
            df['closest_node_distance'] = np.nan
            df['approach_angle'] = np.nan

            df['head_other_id'] = np.nan
            df['closest_node_to_head'] = np.nan
            df['head_distance'] = np.nan

            # --------------------------------------------------
            # PER-FRAME COMPUTATION
            # --------------------------------------------------
            for frame, frame_df in df.groupby('frame'):
                if frame_df['track_id'].nunique() < 2:
                    continue

                # ==========================
                # BODY–BODY NEAREST
                # ==========================
                valid_body = frame_df[['x_body', 'y_body']].notna().all(axis=1)
                if valid_body.sum() < 2:
                    continue

                body_frame_df = frame_df[valid_body]
                body_coords = body_frame_df[['x_body', 'y_body']].to_numpy(float)
                D_body = cdist(body_coords, body_coords)
                np.fill_diagonal(D_body, np.nan)

                df.loc[
                    body_frame_df.index,
                    'body-body'
                ] = np.nanmin(D_body, axis=1)

                # ==========================
                # NODE–NODE NEAREST
                # ==========================
                node_rows = []
                for idx, row in frame_df.iterrows():
                    for part in parts:
                        node_rows.append({
                            'index': idx,
                            'track_id': row['track_id'],
                            'part': part,
                            'x': row[f'x_{part}'],
                            'y': row[f'y_{part}'],
                        })

                nodes = pd.DataFrame(node_rows)
                # coords = nodes[['x', 'y']].to_numpy(float) ##
                # D = cdist(coords, coords) ##

                # group node table by focal larva row (df index)
                for focal_idx, focal_nodes in nodes.groupby('index'):
                    focal_track = focal_nodes['track_id'].iloc[0]

                    other_nodes = nodes[nodes['track_id'] != focal_track]
                    if other_nodes.empty:
                        continue

                    A = focal_nodes[['x', 'y']].to_numpy(float)      # 3x2 (head/body/tail)
                    B = other_nodes[['x', 'y']].to_numpy(float)      # (3*(n-1))x2

                    D = cdist(A, B)

                    if np.isnan(D).all():
                        continue

                    a, b = np.unravel_index(np.nanargmin(D), D.shape)

                    focal_part = focal_nodes.iloc[a]['part']
                    nearest = other_nodes.iloc[b]

                    interaction = unify_interaction_type(focal_part, nearest['part'])

                    df.at[focal_idx, 'other_id'] = nearest['track_id']
                    df.at[focal_idx, 'closest_node_interaction'] = interaction
                    df.at[focal_idx, 'closest_node_distance'] = D[a, b]

                    # NEW: closest other node to the focal HEAD
                    focal_head = focal_nodes[focal_nodes['part'] == 'head'][['x', 'y']].to_numpy(float)
                    # if focal_head.size == 2: #one row with two values e.g. xy dont want nans 
                    if focal_head.shape[0] != 0:
        
                        Dh = cdist(focal_head, B)  # 1 x (3*(n-1))
                        if not np.isnan(Dh).all():
                            b_h = int(np.nanargmin(Dh))
                            nearest_h = other_nodes.iloc[b_h]
                            df.at[focal_idx, 'head_other_id'] = nearest_h['track_id']
                            df.at[focal_idx, 'closest_node_to_head'] = nearest_h['part']
                            df.at[focal_idx, 'head_distance'] = float(Dh[0, b_h])


                    # approach angle: body->head vs head->(nearest node)
                    v_body_head = np.array([
                        df.at[focal_idx, 'x_head'] - df.at[focal_idx, 'x_body'],
                        df.at[focal_idx, 'y_head'] - df.at[focal_idx, 'y_body']
                    ])

                    v_head_other = np.array([
                        nearest['x'] - df.at[focal_idx, 'x_head'],
                        nearest['y'] - df.at[focal_idx, 'y_head']
                    ])

                    if np.linalg.norm(v_body_head) > 0 and np.linalg.norm(v_head_other) > 0:
                        cos = np.dot(v_body_head, v_head_other) / (
                            np.linalg.norm(v_body_head) * np.linalg.norm(v_head_other)
                        )
                        df.at[focal_idx, 'approach_angle'] = np.degrees(
                            np.arccos(np.clip(cos, -1, 1))
                        )


            dfs.append(df)

        data = pd.concat(dfs, ignore_index=True)

        suffix = f"_{self.shorten_duration}" if self.shorten and self.shorten_duration else ""
        filename = f"nearest_neighbour{suffix}.csv"
        data.to_csv(os.path.join(self.directory, filename), index=False)




    def interaction_type_bout(self):

        threshold = 1.0           # must hit this to START a bout
        continue_threshold = 1.5  # once started, can CONTINUE while min_dist < this

        def unify_interaction_type(part1, part2):
            return '_'.join(sorted([part1, part2]))

        body_parts = ['head', 'body', 'tail']
        interaction_pairs = list(itertools.product(body_parts, body_parts))

        unified_types = [
            'head_head', 'tail_tail', 'body_body',
            'body_head', 'body_tail', 'head_tail'
        ]

        bouts = []

        for track_file in self.track_files:
            df = self.track_data[track_file].copy()
            df.sort_values(by='frame', inplace=True)

            active_bouts = {}  # key: (id1, id2) -> bout dict
            bout_counter = 0

            for frame in df['frame'].unique():
                frame_data = df[df['frame'] == frame]
                track_ids = frame_data['track_id'].unique()

                # Build coordinate lookups for each part
                coords = {
                    part: {
                        row['track_id']: np.array([row[f'x_{part}'], row[f'y_{part}']])
                        for _, row in frame_data.iterrows()
                    }
                    for part in body_parts
                }

                # pairs with any <1mm contacts this frame (used to START bouts + log real interactions)
                interacting_pairs = {}  # pair_key -> list of interaction types (<1mm)

                # pairs with min distance <1.5mm this frame (used to CONTINUE bouts)
                close_pairs = {}        # pair_key -> closest_type (min-distance type)

                for id1, id2 in itertools.combinations(track_ids, 2):

                    interactions = []
                    min_dist = float('inf')
                    closest_type = None

                    for part1, part2 in interaction_pairs:
                        coord1 = coords[part1].get(id1)
                        coord2 = coords[part2].get(id2)
                        if coord1 is None or coord2 is None:
                            continue

                        dist = np.linalg.norm(coord1 - coord2)

                        # track minimum distance + its type
                        if dist < min_dist:
                            min_dist = dist
                            closest_type = unify_interaction_type(part1, part2)

                        # record all true contact types (<1mm)
                        if dist < threshold:
                            interactions.append(unify_interaction_type(part1, part2))

                    pair_key = tuple(sorted((id1, id2)))

                    # continuation condition: within 1.5mm
                    if closest_type is not None and min_dist < continue_threshold:
                        close_pairs[pair_key] = closest_type

                    # start/true-contact condition: any <1mm
                    if interactions:
                        interacting_pairs[pair_key] = interactions

                current_close = set(close_pairs.keys())

                # 1) END bouts that are no longer within 1.5mm
                for pair in list(active_bouts.keys()):
                    if pair not in current_close:
                        bout = active_bouts.pop(pair)
                        interactions_all = bout['interactions']
                        if interactions_all:
                            type_counts = Counter(interactions_all)
                            bout_data = {
                                'file': track_file,
                                'bout_id': bout['bout_id'],
                                'track_1': pair[0],
                                'track_2': pair[1],
                                'start_frame': bout['start_frame'],
                                'end_frame': bout['end_frame'],
                                'duration': bout['end_frame'] - bout['start_frame'] + 1,
                                'initial_type': interactions_all[0],
                                'predominant_type': Counter(interactions_all).most_common(1)[0][0],
                            }
                            for t in unified_types:
                                bout_data[t] = type_counts.get(t, 0)
                            bouts.append(bout_data)

                # 2) UPDATE existing bouts that are still within 1.5mm
                for pair in list(active_bouts.keys()):
                    # (pair must be in close_pairs here)
                    active_bouts[pair]['end_frame'] = frame

                    if pair in interacting_pairs:
                        # real interactions (<1mm)
                        active_bouts[pair]['interactions'].extend(interacting_pairs[pair])
                    else:
                        # between 1.0 and 1.5mm: filler closest type
                        active_bouts[pair]['interactions'].append(close_pairs[pair])


                # 3) START new bouts ONLY if they hit <1mm this frame
                for pair, interactions in interacting_pairs.items():
                    if pair in active_bouts:
                        continue
                    active_bouts[pair] = {
                        'bout_id': bout_counter,
                        'start_frame': frame,
                        'end_frame': frame,
                        'interactions': interactions.copy(),
                    }
                    bout_counter += 1

            # Finalize remaining bouts at end of file
            for pair, bout in active_bouts.items():
                interactions_all = bout['interactions']
                if interactions_all:
                    type_counts = Counter(interactions_all)
                    bout_data = {
                        'file': track_file,
                        'bout_id': bout['bout_id'],
                        'track_1': pair[0],
                        'track_2': pair[1],
                        'start_frame': bout['start_frame'],
                        'end_frame': bout['end_frame'],
                        'duration': bout['end_frame'] - bout['start_frame'] + 1,
                        'initial_type': interactions_all[0],
                        'predominant_type': Counter(interactions_all).most_common(1)[0][0],
                    }
                    for t in unified_types:
                        bout_data[t] = type_counts.get(t, 0)
                    bouts.append(bout_data)

        bout_df = pd.DataFrame(bouts).sort_values(by=['file', 'bout_id'])
        bout_df.to_csv(os.path.join(self.directory, "interaction_type_bout.csv"), index=False)
        return bout_df

    
    def interaction_types_closest(self, threshold=1):

        """
        Frame-level closest-contact detection (no bouts).
        For each larval pair per frame:
        - compute all 9 node-node distances
        - keep only the minimum distance + its node-node type
        - only log frames where min distance < threshold
        Output: one row per (file, frame, pair) contact frame
        """

        data = []
        no_contacts = []

        parts = ['head', 'body', 'tail']
        interaction_pairs = list(itertools.product(parts, parts))

        def unify_interaction_type(part1, part2):
            return '_'.join(sorted([part1, part2]))

        def process_track_pair(track_a, track_b, df, track_file):
            results = []
            track_a_data = df[df['track_id'] == track_a]
            track_b_data = df[df['track_id'] == track_b]

            common_frames = sorted(set(track_a_data['frame']).intersection(track_b_data['frame']))
            if not common_frames:
                return results

            for frame in common_frames:
                row_a = track_a_data[track_a_data['frame'] == frame]
                row_b = track_b_data[track_b_data['frame'] == frame]
                if row_a.empty or row_b.empty:
                    continue

                # build coords
                coords_a = {p: row_a[[f'x_{p}', f'y_{p}']].to_numpy().flatten() for p in parts}
                coords_b = {p: row_b[[f'x_{p}', f'y_{p}']].to_numpy().flatten() for p in parts}

                # compute all 9 distances, keep minimum
                min_dist = float('inf')
            #   min_type = None
                min_part_a = None
                min_part_b = None
                for part1, part2 in interaction_pairs:
                    dist = np.linalg.norm(coords_a[part1] - coords_b[part2])
                    if dist < min_dist:
                        min_dist = dist
                        min_part_a = part1
                        min_part_b = part2
                        # min_type = unify_interaction_type(part1, part2)

                if min_dist < threshold:
                    results.append({
                        'file': track_file,
                        'frame': frame,
                        'Interaction Pair': tuple(sorted((track_a, track_b))),
                        'track_0': track_a,
                        'track_1': track_b,
                        'track_0_node': min_part_a,
                        'track_1_node': min_part_b,
                        'Distance': min_dist,
                        'Closest Interaction Type': unify_interaction_type(min_part_a, min_part_b)
                    })

            return results

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file].sort_values(by='frame')

            track_ids = sorted(df['track_id'].unique()) # 0 always first
            track_combinations = list(combinations(track_ids, 2))

            all_results = Parallel(n_jobs=-1)(
                delayed(process_track_pair)(track_a, track_b, df, track_file)
                for track_a, track_b in track_combinations
            )

            flattened_results = [item for sublist in all_results for item in sublist]
            if not flattened_results:
                print(f"No closest-contact frames for {track_file}")
                no_contacts.append(track_file)
                continue

            data.append(pd.DataFrame(flattened_results))

        # placeholders for files with none
        for file in no_contacts:
            data.append(pd.DataFrame([{
                'file': file,
                'frame': np.nan,
                'Interaction Pair': None,
                'Distance': np.nan,
                'Closest Interaction Type': None
            }]))

        closest_df = pd.concat(data, ignore_index=True)

        if self.shorten and self.shorten_duration is not None:
            suffix = f"_{self.shorten_duration}"
        else:
            suffix = ""

        filename = f"closest_contacts_{threshold}mm{suffix}.csv"
        closest_df.to_csv(os.path.join(self.directory, filename), index=False)

        return closest_df

    

    def euclidean_distance(self):

        data = []

        for track_file in self.track_files:
            track_data = self.track_data[track_file]


            for frame in track_data['frame'].unique():

                unique_frame =  track_data[track_data['frame'] == frame]

                # cdist function requires two 2-dimensional array-like objects as inputs
                # create an array of the coordinates for that specific frame
                    
                unique_frame = unique_frame.dropna(subset=['x_body', 'y_body'])
                if unique_frame['track_id'].nunique() < 2:
                    average_distance = np.nan
                    data.append({'time': frame, 'average_distance': average_distance, 'file': track_file})
                    continue

                body_coordinates = unique_frame[['x_body', 'y_body']].to_numpy()

                # The cdist function computes the distance between every pair of points in the two arrays passed to it.

                distance = cdist(body_coordinates, body_coordinates, 'euclidean')

                np.fill_diagonal(distance, np.nan)

                average_distance = np.nanmean(distance)

                data.append({'time': frame, 'average_distance': average_distance, 'file': track_file})

        df = pd.DataFrame(data)
        df = df.sort_values(by=['time', 'file'], ascending=True)

        if self.shorten and self.shorten_duration is not None:
            suffix = f"_{self.shorten_duration}"
        else:
            suffix = ""

        filename = f"euclidean_distances{suffix}.csv"
        df.to_csv(os.path.join(self.directory, filename), index=False)

        return df
    

    def distance_from_centre(self): 

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

                data.append({'file': track_file, 'frame': row['frame'], 'track': row['track_id'], 'distance_from_centre': distance})

        df_distance_over_time = pd.DataFrame(data)

        filename = "distance_from_centre.csv"
    
        df_distance_over_time.to_csv(os.path.join(self.directory, filename), index=False)
        print(f'Distance over time saved: {df_distance_over_time}')

        return df_distance_over_time
    




if __name__ == "__main__":

    directories = [
      
        "/Volumes/lab-windingm-1/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/sensory/9047",
        "/Volumes/lab-windingm-1/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/sensory/wiii8",
        "/Volumes/lab-windingm-1/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/sensory/anosmic",
        "/Volumes/lab-windingm-1/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/sensory/33300",
        "/Volumes/lab-windingm-1/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/sensory/23129",
    ]


    for directory in directories:

        print(f"\nProcessing: {directory}")

        analysis = SensoryMutantAnalysis(directory)


        analysis.digging_mask()
        analysis.conversion()


        # analysis.size()
        # analysis.interaction_types_closest() 
        # analysis.euclidean_distance()
        # analysis.speed()
        # analysis.distance_travelled()
        # analysis.nearest_neighbour()
        # analysis.interaction_type_bout() 
        # analysis.distance_travelled()
        analysis.distance_from_centre()

     
  

    
