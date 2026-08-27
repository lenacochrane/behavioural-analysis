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


class FedStarvedAnalysis:

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
            # df.to_csv(os.path.join(self.directory, 'digging.csv'), index=False) # get rid 
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


                track_file = match['track_file']
                track_data = self.track_data[track_file]

                pixel_columns = ['x_tail', 'y_tail', 'x_body', 'y_body', 'x_head', 'y_head']
                track_data[pixel_columns] = track_data[pixel_columns] * conversion_factor
                self.track_data[track_file] = track_data  # Update the track data.
                print(f"Conversion applied for {track_file} with conversion factor: {conversion_factor:.3f}")
            
            else:
                print(f"no perimeter detected for {match['track_file']}")
  
                conversion_factor = 90 / 1032 # the one i used to use 
                
                track_file = match['track_file']
                track_data = self.track_data[track_file]

                pixel_columns = ['x_tail', 'y_tail', 'x_body', 'y_body', 'x_head', 'y_head']
                track_data[pixel_columns] = track_data[pixel_columns] * conversion_factor
                self.track_data[track_file] = track_data  # Update the track data.
                print(f"Conversion applied for {track_file} with conversion factor: {conversion_factor:.3f}")

    
    # METHOD FILTERING_FILES: KEEPS FILES WHERE THE TWO LARVAE COME CLOSE ENOUGH

    def filtering_files(self, head_node_threshold=5, node_contact_threshold=1):

        results = []
        included_track_files = []
        included_matching_pairs = []

        def below_threshold(values, threshold, inclusive=False):
            values = np.asarray(values, dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                return False
            if inclusive:
                return bool(np.min(values) <= threshold)
            return bool(np.min(values) < threshold)

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file].copy()

            head_to_any_node_within_5mm = False
            any_node_node_contact_within_1mm = False

            if not df.empty:
                df = df.sort_values(['frame', 'track_id'])
                row_counts = df.groupby('frame')['track_id'].transform('size')
                track_counts = df.groupby('frame')['track_id'].transform('nunique')
                df_two = df[(row_counts == 2) & (track_counts == 2)]

                if not df_two.empty:
                    first_larvae = df_two.groupby('frame').nth(0)
                    second_larvae = df_two.groupby('frame').nth(1)

                    nodes_1 = np.stack([
                        first_larvae[['x_head', 'y_head']].to_numpy(dtype=float),
                        first_larvae[['x_body', 'y_body']].to_numpy(dtype=float),
                        first_larvae[['x_tail', 'y_tail']].to_numpy(dtype=float),
                    ], axis=1)

                    nodes_2 = np.stack([
                        second_larvae[['x_head', 'y_head']].to_numpy(dtype=float),
                        second_larvae[['x_body', 'y_body']].to_numpy(dtype=float),
                        second_larvae[['x_tail', 'y_tail']].to_numpy(dtype=float),
                    ], axis=1)

                    head_1 = nodes_1[:, 0, :]
                    head_2 = nodes_2[:, 0, :]
                    head_1_to_nodes_2 = np.linalg.norm(nodes_2 - head_1[:, None, :], axis=2)
                    head_2_to_nodes_1 = np.linalg.norm(nodes_1 - head_2[:, None, :], axis=2)

                    head_to_any_node_within_5mm = below_threshold(
                        np.concatenate([
                            head_1_to_nodes_2.ravel(),
                            head_2_to_nodes_1.ravel(),
                        ]),
                        head_node_threshold,
                        inclusive=True
                    )

                    node_node_distances = np.linalg.norm(
                        nodes_1[:, :, None, :] - nodes_2[:, None, :, :],
                        axis=3
                    )
                    any_node_node_contact_within_1mm = below_threshold(
                        node_node_distances.ravel(),
                        node_contact_threshold
                    )

            passed_filter = head_to_any_node_within_5mm or any_node_node_contact_within_1mm

            results.append({
                'file_name': track_file,
                'passed_filter': 'Y' if passed_filter else 'N',
                'head_to_any_node_within_5mm': 'Y' if head_to_any_node_within_5mm else 'N',
                'any_node_node_contact_within_1mm': 'Y' if any_node_node_contact_within_1mm else 'N',
            })

            if passed_filter:
                included_track_files.append(track_file)
                included_matching_pairs.append(match)

        included_files = pd.DataFrame(results)
        included_files.to_csv(os.path.join(self.directory, "included_files.csv"), index=False)

        self.track_files = included_track_files
        self.matching_pairs = included_matching_pairs
        self.track_data = {
            track_file: self.track_data[track_file]
            for track_file in included_track_files
        }

        print(f"Saved included files filter to: {os.path.join(self.directory, 'included_files.csv')}")
        print(f"Included {len(included_track_files)} of {len(results)} files after filtering.")

        return included_files




    def file_summary(self, cutoff_frame=3599):
        
        results = []

        for track_file in self.track_files:

            df = self.track_data[track_file].copy()

            video_name = track_file.replace(".tracks.feather", "")

            for track_id, track_df in df.groupby("track_id"):

                frames_present = sorted(track_df["frame"].unique())

                total_frames_present = len(frames_present)

                present_at_end = total_frames_present >= cutoff_frame

                left = total_frames_present < cutoff_frame


                came_back = False

                if left:
                    frame_diffs = np.diff(frames_present)
                    came_back = any(frame_diffs > 1)

                results.append({
                    "video_name": video_name,
                    "track": track_id,
                    "total_number_frames": total_frames_present,
                    "present_at_end": present_at_end,
                    "left": left,
                    "came_back": came_back
                })

        results_df = pd.DataFrame(results)

        save_path = os.path.join(self.directory, "file_summary.csv")
        results_df.to_csv(save_path, index=False)

        print(f"Saved file summary to: {save_path}")

        return results_df
    

    

    def larvae_present_over_time(self):

        data = []

        for track_file in self.track_files:
            df = self.track_data[track_file].copy()

            summary = (
                df.groupby("frame")["track_id"]
                .nunique()
                .reset_index(name="n_larvae_present")
            )

            summary["file"] = track_file
            data.append(summary)

        result = pd.concat(data, ignore_index=True)


        filename = f"larvae_present_over_time.csv"
        result.to_csv(os.path.join(self.directory, filename), index=False)

        return result
    

    # METHOD DISTANCE_FROM_CENTRE: CALCULATES DISTANCES FROM CENTRE COORDINATES 

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

        if self.shorten and self.shorten_duration is not None:
            suffix = f"_{self.shorten_duration}"
        else:
            suffix = ""

        filename = f"distance_from_centre{suffix}.csv"
    
        df_distance_over_time.to_csv(os.path.join(self.directory, filename), index=False)
        print(f'Distance over time saved: {df_distance_over_time}')

        return df_distance_over_time

    # METHOD EUCLIDEAN_DISTANCE: CALCULATES THE AVERAGE DISTANCE BETWEEN LARVAE ACCROSS FRAMES

    def euclidean_distance(self):

        data = []

        for track_file in self.track_files:
            track_data = self.track_data[track_file]


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

        if self.shorten and self.shorten_duration is not None:
            suffix = f"_{self.shorten_duration}"
        else:
            suffix = ""

        filename = f"euclidean_distances{suffix}.csv"
        df.to_csv(os.path.join(self.directory, filename), index=False)

        return df

    



    
    
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

                    data.append({'time': time2, 'speed': speed_value, 'file': track_file, 'track': track})
    
        speed_over_time = pd.DataFrame(data)
        speed_over_time = speed_over_time.sort_values(by=['file', 'track', 'time'], ascending=True)

        if self.shorten and self.shorten_duration is not None:
            suffix = f"_{self.shorten_duration}"
        else:
            suffix = ""

        filename = f"speed_over_time{suffix}.csv"

        speed_over_time.to_csv(os.path.join(self.directory, filename), index=False)

        return speed_over_time
    


    # METHOD ACCELERATION: 

    def acceleration(self):

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
                    if time > 2:
                        continue

                    speed_value = distance / time 

                    if previous_speed is not None and previous_time is not None:
                        acceleration_value = (speed_value - previous_speed) / time 
                        data.append({'time': time2, 'acceleration': acceleration_value, 'file': track_file})

                    previous_speed = speed_value
                    previous_time = time
    

        acceleration_accross_time = pd.DataFrame(data)
        acceleration_accross_time = acceleration_accross_time.sort_values(by=['time'], ascending=True)

        if self.shorten and self.shorten_duration is not None:
            suffix = f"_{self.shorten_duration}"
        else:
            suffix = ""

        filename = f"acceleration_accross_time{suffix}.csv"

        acceleration_accross_time.to_csv(os.path.join(self.directory, filename), index=False)
        return acceleration_accross_time
        
    
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

        if self.shorten and self.shorten_duration is not None:
            suffix = f"_{self.shorten_duration}"
        else:
            suffix = ""

        filename = f"ensemble_msd{suffix}.csv"

        # Save the DataFrame as a CSV file
        output_path = os.path.join(self.directory, filename)
        df.to_csv(output_path, index=False)
        print(f"Ensemble MSD saved to {output_path}")
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

        if self.shorten and self.shorten_duration is not None:
            suffix = f"_{self.shorten_duration}"
        else:
            suffix = ""

        filename = f"time_average_msd{suffix}.csv"

        tau_msd_df.to_csv(os.path.join(self.directory, filename), index=False)
   
        return tau_msd_df #.dropna()
    
    # METHOD ANGLE: CALCULATES TRAJECTORY ANGLES: 1) TRAJECTORY ANGLE VALUES 2) TRAJECTORY ANGLE OVER TIME 
      # ANGLE INBETWEEN 2 VECTORS: TAIL-BODY AND BODY-HEAD 

    def angle(self):

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
        

        angle_over_time = pd.DataFrame(data)
        angle_over_time = angle_over_time.sort_values(by=['time'], ascending=True)

        if self.shorten and self.shorten_duration is not None:
            suffix = f"_{self.shorten_duration}"
        else:
            suffix = ""

        filename = f"angle_over_time{suffix}.csv"

        angle_over_time.to_csv(os.path.join(self.directory, filename), index=False)

        return angle_over_time  


    def trajectory(self, proximity_threshold=1, window=15):

        output_dir = os.path.join(self.directory, "trajectories")
        os.makedirs(output_dir, exist_ok=True)

        parts = ['head', 'body', 'tail']
        interaction_pairs = list(itertools.product(parts, parts))
        colours = {
            0: "#1f77b4",  # nice blue for track 0
            1: "#7b2cbf",  # nice purple for track 1
        }

        def compute_min_distance(row_a, row_b):
            coords_a = {p: np.array([row_a[f'x_{p}'], row_a[f'y_{p}']], dtype=float) for p in parts}
            coords_b = {p: np.array([row_b[f'x_{p}'], row_b[f'y_{p}']], dtype=float) for p in parts}

            min_dist = np.inf
            min_pair = None

            for p1, p2 in interaction_pairs:
                dist = np.linalg.norm(coords_a[p1] - coords_b[p2])
                if dist < min_dist:
                    min_dist = dist
                    min_pair = (p1, p2)

            return min_dist, min_pair

        def first_no_contact_frame(df_a, df_b, common_frames, interaction_frame):
            for f in common_frames:
                if f <= interaction_frame:
                    continue

                row_a = df_a[df_a['frame'] == f]
                row_b = df_b[df_b['frame'] == f]
                if row_a.empty or row_b.empty:
                    continue

                min_dist, _ = compute_min_distance(row_a.iloc[0], row_b.iloc[0])
                if min_dist > proximity_threshold:
                    return f

            return None

        def rotate_points(points, angle):
            rotation = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            return points @ rotation.T

        def temporal_colours(colour, n):
            base = np.array(plt.matplotlib.colors.to_rgb(colour))
            white = np.array([1, 1, 1], dtype=float)
            shades = []

            for i in range(n):
                if n == 1:
                    fade = 0
                else:
                    fade = i / (n - 1)

                shade = (base * (1 - fade * 0.7)) + (white * fade * 0.7)
                shades.append(shade)

            return shades

        def plot_temporal_head_path(ax, head, colour, linewidth=1.4, dot_size=9, label=None):
            head = np.asarray(head, dtype=float)
            valid = np.isfinite(head).all(axis=1)
            head = head[valid]

            if len(head) == 0:
                return

            dot_colours = temporal_colours(colour, len(head))

            if len(head) > 1:
                points = head.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                segment_colours = temporal_colours(colour, len(segments))
                line = LineCollection(
                    segments,
                    colors=segment_colours,
                    linewidths=linewidth,
                    label=label
                )
                ax.add_collection(line)

            ax.scatter(
                head[:, 0],
                head[:, 1],
                color=dot_colours,
                s=dot_size,
                alpha=0.9,
                edgecolors='none'
            )

        def transform_trajectory(rows_by_track, track_ids):
            start_rows = [
                rows_by_track[track_id].iloc[0]
                for track_id in track_ids
                if not rows_by_track[track_id].empty
            ]

            if len(start_rows) != 2:
                return None

            start_heads = np.array([
                [row['x_head'], row['y_head']]
                for row in start_rows
            ], dtype=float)
            origin = np.nanmean(start_heads, axis=0)

            early_vectors = []
            for track_id in track_ids:
                rows = rows_by_track[track_id].sort_values('frame').head(4)
                head_xy = rows[['x_head', 'y_head']].to_numpy(dtype=float)
                if len(head_xy) > 1:
                    steps = head_xy[1:] - head_xy[:-1]
                    early_vectors.extend(steps[np.isfinite(steps).all(axis=1)])

            if early_vectors:
                mean_vector = np.nanmean(np.array(early_vectors), axis=0)
            else:
                mean_vector = np.array([0, 0], dtype=float)

            if np.linalg.norm(mean_vector) > 0:
                angle = (np.pi / 2) - np.arctan2(mean_vector[1], mean_vector[0])
            else:
                angle = 0

            transformed = {}

            for track_id in track_ids:
                rows = rows_by_track[track_id].sort_values('frame').copy()

                body_xy = rows[['x_body', 'y_body']].to_numpy(dtype=float) - origin
                head_xy = rows[['x_head', 'y_head']].to_numpy(dtype=float) - origin

                body_xy = rotate_points(body_xy, angle)
                head_xy = rotate_points(head_xy, angle)

                transformed[track_id] = {
                    'frames': rows['frame'].to_numpy(),
                    'body': body_xy,
                    'head': head_xy,
                }

            track_0 = 0 if 0 in transformed else track_ids[0]
            track_0_head = transformed[track_0]['head']
            track_0_head = track_0_head[np.isfinite(track_0_head).all(axis=1)]

            if len(track_0_head) > 0 and track_0_head[0, 0] > 0:
                for track_id in transformed:
                    transformed[track_id]['body'][:, 0] *= -1
                    transformed[track_id]['head'][:, 0] *= -1

            all_points = []
            for track_id in track_ids:
                body_xy = transformed[track_id]['body']
                head_xy = transformed[track_id]['head']

                all_points.extend(body_xy[np.isfinite(body_xy).all(axis=1)])
                all_points.extend(head_xy[np.isfinite(head_xy).all(axis=1)])

            return transformed, all_points

        trajectories = []

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file].sort_values('frame')

            track_ids = sorted(df['track_id'].unique())
            if len(track_ids) != 2:
                continue

            df_a = df[df['track_id'] == track_ids[0]]
            df_b = df[df['track_id'] == track_ids[1]]

            common_frames = sorted(set(df_a['frame']).intersection(df_b['frame']))
            if not common_frames:
                continue

            interaction_number = 0
            next_allowed_frame = -np.inf

            for frame in common_frames:
                if frame < next_allowed_frame:
                    continue

                row_a = df_a[df_a['frame'] == frame]
                row_b = df_b[df_b['frame'] == frame]
                if row_a.empty or row_b.empty:
                    continue

                min_dist, min_pair = compute_min_distance(row_a.iloc[0], row_b.iloc[0])

                if min_dist < proximity_threshold and set(min_pair) == {'head'}:
                    no_contact_frame = first_no_contact_frame(
                        df_a,
                        df_b,
                        common_frames,
                        frame
                    )

                    if no_contact_frame is None:
                        continue

                    interaction_number += 1
                    next_allowed_frame = no_contact_frame + window + 1
                    plot_frames = range(no_contact_frame, no_contact_frame + window + 1)

                    rows_by_track = {}
                    for track_id in track_ids:
                        rows = df[
                            (df['track_id'] == track_id) &
                            (df['frame'].isin(plot_frames))
                        ].copy()
                        rows_by_track[track_id] = rows

                    transformed = transform_trajectory(rows_by_track, track_ids)
                    if transformed is None:
                        continue

                    trajectory, all_points = transformed

                    trajectories.append({
                        'file': track_file,
                        'interaction_number': interaction_number,
                        'interaction_frame': frame,
                        'no_contact_frame': no_contact_frame,
                        'track_ids': track_ids,
                        'trajectory': trajectory,
                        'all_points': all_points,
                    })

        if not trajectories:
            print(f"No head-head trajectories found in {self.directory}")
            return pd.DataFrame()

        all_points = []
        for item in trajectories:
            all_points.extend(item['all_points'])

        all_points = np.array(all_points, dtype=float)
        all_points = all_points[np.isfinite(all_points).all(axis=1)]

        if len(all_points) == 0:
            axis_limit = 1
        else:
            axis_limit = np.nanmax(np.abs(all_points))
            axis_limit = max(axis_limit * 1.15, 1)

        def plot_one_axis(ax, item, show_labels=False):
            for track_id in item['track_ids']:
                colour = colours.get(track_id, "#333333")
                trajectory = item['trajectory'][track_id]
                head = trajectory['head']

                plot_temporal_head_path(ax, head, colour)

            ax.axhline(0, color="#dddddd", linewidth=0.8)
            ax.axvline(0, color="#dddddd", linewidth=0.8)
            ax.set_xlim(-axis_limit, axis_limit)
            ax.set_ylim(-axis_limit, axis_limit)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, color="#eeeeee", linewidth=0.6)
            if show_labels:
                ax.set_xlabel("x")
                ax.set_ylabel("y")
            else:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            ax.set_title(
                f"{item['file']}\n"
                f"int {item['interaction_number']} | "
                f"c {item['interaction_frame']} | "
                f"nc {item['no_contact_frame']}",
                fontsize=5
            )

        def add_empty_page(pdf, title):
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                title,
                ha="center",
                va="center",
                fontsize=9
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        def write_grid_pdf(pdf_path, items, empty_title, plots_per_page=25):
            with PdfPages(pdf_path) as pdf:
                if not items:
                    add_empty_page(pdf, empty_title)
                    return

                for start in range(0, len(items), plots_per_page):
                    page_items = items[start:start + plots_per_page]
                    fig, axes = plt.subplots(5, 5, figsize=(11, 11))
                    axes = axes.flatten()

                    for ax, item in zip(axes, page_items):
                        plot_one_axis(ax, item)

                    for ax in axes[len(page_items):]:
                        ax.axis("off")

                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

        def mean_track_xy(items, track_id, node):
            max_len = window + 1
            xs = []
            ys = []

            for item in items:
                if track_id not in item['trajectory']:
                    continue

                xy = item['trajectory'][track_id][node]
                x = np.full(max_len, np.nan)
                y = np.full(max_len, np.nan)
                n = min(len(xy), max_len)

                if n > 0:
                    x[:n] = xy[:n, 0]
                    y[:n] = xy[:n, 1]

                xs.append(x)
                ys.append(y)

            if not xs:
                return None

            return np.column_stack((
                np.nanmean(np.array(xs), axis=0),
                np.nanmean(np.array(ys), axis=0)
            ))

        def mean_sd_track_xy(items, track_id, node):
            max_len = window + 1
            xs = []
            ys = []

            for item in items:
                if track_id not in item['trajectory']:
                    continue

                xy = item['trajectory'][track_id][node]
                x = np.full(max_len, np.nan)
                y = np.full(max_len, np.nan)
                n = min(len(xy), max_len)

                if n > 0:
                    x[:n] = xy[:n, 0]
                    y[:n] = xy[:n, 1]

                xs.append(x)
                ys.append(y)

            if not xs:
                return None

            xs = np.array(xs, dtype=float)
            ys = np.array(ys, dtype=float)
            return {
                'mean': np.column_stack((
                    np.nanmean(xs, axis=0),
                    np.nanmean(ys, axis=0)
                )),
                'sd_x': np.nanstd(xs, axis=0),
                'sd_y': np.nanstd(ys, axis=0),
            }

        def write_mean_pdf(pdf_path, items, title, node):
            with PdfPages(pdf_path) as pdf:
                if not items:
                    add_empty_page(pdf, f"No {node} trajectories for {title}")
                    return

                fig, ax = plt.subplots(figsize=(4.8, 4.8))
                track_ids = sorted({
                    track_id
                    for item in items
                    for track_id in item['track_ids']
                })
                legend_handles = []
                mean_points = []

                for track_id in track_ids:
                    colour = colours.get(track_id, "#333333")
                    node_mean = mean_track_xy(items, track_id, node)

                    if node_mean is None:
                        continue

                    mean_points.extend(node_mean[np.isfinite(node_mean).all(axis=1)])

                    plot_temporal_head_path(
                        ax,
                        node_mean,
                        colour,
                        linewidth=2.5,
                        dot_size=22,
                        label=f"track {track_id} {node} mean"
                    )
                    legend_handles.append(Line2D(
                        [0],
                        [0],
                        color=colour,
                        marker='o',
                        linewidth=2.5,
                        markersize=5,
                        label=f"track {track_id} head mean"
                    ))

                ax.axhline(0, color="#dddddd", linewidth=0.8)
                ax.axvline(0, color="#dddddd", linewidth=0.8)
                mean_points = np.array(mean_points, dtype=float)
                if len(mean_points) == 0:
                    mean_axis_limit = 1
                else:
                    mean_axis_limit = np.nanmax(np.abs(mean_points))
                    mean_axis_limit = max(mean_axis_limit * 1.35, 0.25)

                ax.set_xlim(-mean_axis_limit, mean_axis_limit)
                ax.set_ylim(-mean_axis_limit, mean_axis_limit)
                ax.set_aspect('equal', adjustable='box')
                ax.grid(True, color="#eeeeee", linewidth=0.6)
                ax.set_xlabel("x, rotated and centred")
                ax.set_ylabel("y, rotated and centred")
                ax.set_title(f"{title} mean trajectories\nn = {len(items)}", fontsize=9)
                ax.legend(handles=legend_handles, fontsize=7, loc="upper right", frameon=False)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        def write_mean_with_trace_pdf(pdf_path, items, title):
            with PdfPages(pdf_path) as pdf:
                if not items:
                    add_empty_page(pdf, f"No trajectories for {title}")
                    return

                fig, ax = plt.subplots(figsize=(4.8, 4.8))
                track_ids = sorted({
                    track_id
                    for item in items
                    for track_id in item['track_ids']
                })
                legend_handles = []
                plotted_points = []

                for track_id in track_ids:
                    colour = colours.get(track_id, "#333333")

                    for item in items:
                        if track_id not in item['trajectory']:
                            continue

                        head = np.asarray(item['trajectory'][track_id]['head'], dtype=float)
                        valid = np.isfinite(head).all(axis=1)
                        head = head[valid]

                        if len(head) == 0:
                            continue

                        plotted_points.extend(head)
                        ax.plot(
                            head[:, 0],
                            head[:, 1],
                            color=colour,
                            linewidth=0.45,
                            alpha=0.12,
                            zorder=1
                        )

                    head_mean = mean_track_xy(items, track_id, 'head')

                    if head_mean is None:
                        continue

                    plotted_points.extend(head_mean[np.isfinite(head_mean).all(axis=1)])

                    plot_temporal_head_path(
                        ax,
                        head_mean,
                        colour,
                        linewidth=2.5,
                        dot_size=22,
                        label=f"track {track_id} head mean"
                    )
                    legend_handles.append(Line2D(
                        [0],
                        [0],
                        color=colour,
                        marker='o',
                        linewidth=2.5,
                        markersize=5,
                        label=f"track {track_id} head mean"
                    ))

                ax.axhline(0, color="#dddddd", linewidth=0.8)
                ax.axvline(0, color="#dddddd", linewidth=0.8)
                plotted_points = np.array(plotted_points, dtype=float)
                if len(plotted_points) == 0:
                    trace_axis_limit = 1
                else:
                    trace_axis_limit = np.nanmax(np.abs(plotted_points))
                    trace_axis_limit = max(trace_axis_limit * 1.35, 0.25)

                ax.set_xlim(-trace_axis_limit, trace_axis_limit)
                ax.set_ylim(-trace_axis_limit, trace_axis_limit)
                ax.set_aspect('equal', adjustable='box')
                ax.grid(True, color="#eeeeee", linewidth=0.6)
                ax.set_xlabel("x, rotated and centred")
                ax.set_ylabel("y, rotated and centred")
                ax.set_title(f"{title} mean trajectories with traces\nn = {len(items)}", fontsize=9)
                ax.legend(handles=legend_handles, fontsize=7, loc="upper right", frameon=False)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        def write_mean_sd_pdf(pdf_path, items, title):
            with PdfPages(pdf_path) as pdf:
                if not items:
                    add_empty_page(pdf, f"No trajectories for {title}")
                    return

                fig, ax = plt.subplots(figsize=(4.8, 4.8))
                track_ids = sorted({
                    track_id
                    for item in items
                    for track_id in item['track_ids']
                })
                legend_handles = []
                plotted_points = []

                for track_id in track_ids:
                    colour = colours.get(track_id, "#333333")
                    stats = mean_sd_track_xy(items, track_id, 'head')

                    if stats is None:
                        continue

                    head_mean = stats['mean']
                    valid = np.isfinite(head_mean).all(axis=1)
                    plotted_points.extend(head_mean[valid])

                    sd_points = np.column_stack((
                        head_mean[:, 0] + stats['sd_x'],
                        head_mean[:, 0] - stats['sd_x'],
                        head_mean[:, 1] + stats['sd_y'],
                        head_mean[:, 1] - stats['sd_y'],
                    ))
                    plotted_points.extend(sd_points[np.isfinite(sd_points).all(axis=1)][:, [0, 2]])
                    plotted_points.extend(sd_points[np.isfinite(sd_points).all(axis=1)][:, [1, 3]])

                    every = max(1, window // 12)
                    error_idx = np.where(valid)[0][::every]
                    ax.errorbar(
                        head_mean[error_idx, 0],
                        head_mean[error_idx, 1],
                        xerr=stats['sd_x'][error_idx],
                        yerr=stats['sd_y'][error_idx],
                        fmt='none',
                        ecolor=colour,
                        elinewidth=0.7,
                        alpha=0.22,
                        capsize=0,
                        zorder=1
                    )

                    plot_temporal_head_path(
                        ax,
                        head_mean,
                        colour,
                        linewidth=2.5,
                        dot_size=22,
                        label=f"track {track_id} head mean"
                    )
                    legend_handles.append(Line2D(
                        [0],
                        [0],
                        color=colour,
                        marker='o',
                        linewidth=2.5,
                        markersize=5,
                        label=f"track {track_id} head mean +/- SD"
                    ))

                ax.axhline(0, color="#dddddd", linewidth=0.8)
                ax.axvline(0, color="#dddddd", linewidth=0.8)
                plotted_points = np.array(plotted_points, dtype=float)
                if len(plotted_points) == 0:
                    axis_limit = 1
                else:
                    axis_limit = np.nanmax(np.abs(plotted_points))
                    axis_limit = max(axis_limit * 1.15, 0.25)

                ax.set_xlim(-axis_limit, axis_limit)
                ax.set_ylim(-axis_limit, axis_limit)
                ax.set_aspect('equal', adjustable='box')
                ax.grid(True, color="#eeeeee", linewidth=0.6)
                ax.set_xlabel("x, rotated and centred")
                ax.set_ylabel("y, rotated and centred")
                ax.set_title(f"{title} mean trajectories +/- SD\nn = {len(items)}", fontsize=9)
                ax.legend(handles=legend_handles, fontsize=7, loc="upper right", frameon=False)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        window_prefix = f"window{window}_"
        first_pdf_name = f"{window_prefix}first.pdf"
        other_pdf_name = f"{window_prefix}other.pdf"

        first_pdf = os.path.join(output_dir, first_pdf_name)
        other_pdf = os.path.join(output_dir, other_pdf_name)
        first_mean_pdf = os.path.join(output_dir, f"{window_prefix}first_mean.pdf")
        other_mean_pdf = os.path.join(output_dir, f"{window_prefix}other_mean.pdf")
        first_mean_withtrace_pdf = os.path.join(output_dir, f"{window_prefix}first_mean_withtrace.pdf")
        other_mean_withtrace_pdf = os.path.join(output_dir, f"{window_prefix}other_mean_withtrace.pdf")
        all_mean_pdf = os.path.join(output_dir, f"{window_prefix}all_mean.pdf")
        first_mean_sd_pdf = os.path.join(output_dir, f"{window_prefix}first_mean_sd.pdf")
        all_mean_sd_pdf = os.path.join(output_dir, f"{window_prefix}all_mean_sd.pdf")

        first_contact_cutoff_frame = 1200

        def trajectory_group(item):
            if (
                item['interaction_number'] == 1
                and item['interaction_frame'] < first_contact_cutoff_frame
            ):
                return 'first'
            if item['interaction_number'] > 1:
                return 'other'
            return 'excluded_first_after_cutoff'

        first_trajectories = [item for item in trajectories if trajectory_group(item) == 'first']
        other_trajectories = [item for item in trajectories if trajectory_group(item) == 'other']

        write_grid_pdf(first_pdf, first_trajectories, "No first head-head trajectories found")
        write_grid_pdf(other_pdf, other_trajectories, "No later head-head trajectories found")
        write_mean_pdf(first_mean_pdf, first_trajectories, "first")
        write_mean_pdf(other_mean_pdf, other_trajectories, "other")
        write_mean_with_trace_pdf(first_mean_withtrace_pdf, first_trajectories, "first")
        write_mean_with_trace_pdf(other_mean_withtrace_pdf, other_trajectories, "other")
        write_mean_pdf(all_mean_pdf, trajectories, "all")
        write_mean_sd_pdf(first_mean_sd_pdf, first_trajectories, "first")
        write_mean_sd_pdf(all_mean_sd_pdf, trajectories, "all")

        summary = pd.DataFrame([
            {
                'file': item['file'],
                'interaction_number': item['interaction_number'],
                'interaction_frame': item['interaction_frame'],
                'no_contact_frame': item['no_contact_frame'],
                'trajectory_group': trajectory_group(item),
                'pdf': (
                    first_pdf_name if trajectory_group(item) == 'first'
                    else other_pdf_name if trajectory_group(item) == 'other'
                    else ''
                ),
            }
            for item in trajectories
        ])

        summary.to_csv(
            os.path.join(output_dir, f"{window_prefix}trajectory_summary.csv"),
            index=False
        )
        return summary

    def trajectory_figures(self, proximity_threshold=1, window=10):

        output_dir = "/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/fed-starved/trajectories/poster"
        os.makedirs(output_dir, exist_ok=True)

        path_parts = os.path.normpath(self.directory).split(os.sep)
        condition_label = "-".join(path_parts[-2:])
        condition = os.path.basename(os.path.normpath(self.directory))

        parts = ['head', 'body', 'tail']
        interaction_pairs = list(itertools.product(parts, parts))
        colours = {
            0: "#1f77b4",
            1: "#7b2cbf",
        }

        if condition == "fed-fed":
            colours = {
                0: "#0B5D2A", #2F5597
                1: "#0B5D2A", #2F5597
            }

        if condition == "starved-starved":
            colours = {
                0: "#edb700", #9E2F35
                1: "#edb700", #9E2F35
            }

        if condition == "fed-starved":
            colours = {
                0: "#edb700", #F3D00E 
                1: "#0B5D2A", #0B5D2A
            }

        def compute_min_distance(row_a, row_b):
            coords_a = {p: np.array([row_a[f'x_{p}'], row_a[f'y_{p}']], dtype=float) for p in parts}
            coords_b = {p: np.array([row_b[f'x_{p}'], row_b[f'y_{p}']], dtype=float) for p in parts}

            min_dist = np.inf
            min_pair = None

            for p1, p2 in interaction_pairs:
                dist = np.linalg.norm(coords_a[p1] - coords_b[p2])
                if dist < min_dist:
                    min_dist = dist
                    min_pair = (p1, p2)

            return min_dist, min_pair

        def first_no_contact_frame(df_a, df_b, common_frames, interaction_frame):
            for f in common_frames:
                if f <= interaction_frame:
                    continue

                row_a = df_a[df_a['frame'] == f]
                row_b = df_b[df_b['frame'] == f]
                if row_a.empty or row_b.empty:
                    continue

                min_dist, _ = compute_min_distance(row_a.iloc[0], row_b.iloc[0])
                if min_dist > proximity_threshold:
                    return f

            return None

        def rotate_points(points, angle):
            rotation = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            return points @ rotation.T

        def first_valid_point(points):
            points = np.asarray(points, dtype=float).copy()
            valid = np.isfinite(points).all(axis=1)

            if valid.any():
                return points[np.where(valid)[0][0]]

            return None

        def register_to_origin(points, origin):
            points = np.asarray(points, dtype=float).copy()

            if origin is not None:
                points -= origin

            return points

        def temporal_colours(colour, n):
            base = np.array(plt.matplotlib.colors.to_rgb(colour))
            white = np.array([1, 1, 1], dtype=float)
            shades = []

            for i in range(n):
                fade = 0 if n == 1 else i / (n - 1)
                shade = (base * (1 - fade * 0.55)) + (white * fade * 0.55)
                shades.append(shade)

            return shades

        def plot_clean_axes(ax):
            ax.grid(False)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)

        def plot_temporal_path(ax, xy, colour, linewidth=2.5, dot_size=40, alpha=0.9, zorder=2):
            xy = np.asarray(xy, dtype=float)
            valid = np.isfinite(xy).all(axis=1)
            xy = xy[valid]

            if len(xy) == 0:
                return

            dot_colours = temporal_colours(colour, len(xy))

            if len(xy) > 1:
                points = xy.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                line = LineCollection(
                    segments,
                    colors=temporal_colours(colour, len(segments)),
                    linewidths=linewidth,
                    alpha=alpha,
                    zorder=zorder
                )
                ax.add_collection(line)

            ax.scatter(
                xy[:, 0],
                xy[:, 1],
                color=dot_colours,
                s=dot_size,
                alpha=alpha,
                edgecolors='none',
                zorder=zorder + 1
            )

        def transform_trajectory(rows_by_track, track_ids):
            start_rows = [
                rows_by_track[track_id].iloc[0]
                for track_id in track_ids
                if not rows_by_track[track_id].empty
            ]

            if len(start_rows) != 2:
                return None

            start_heads = np.array([
                [row['x_head'], row['y_head']]
                for row in start_rows
            ], dtype=float)
            origin = np.nanmean(start_heads, axis=0)

            early_vectors = []
            for track_id in track_ids:
                rows = rows_by_track[track_id].sort_values('frame').head(4)
                head_xy = rows[['x_head', 'y_head']].to_numpy(dtype=float)
                if len(head_xy) > 1:
                    steps = head_xy[1:] - head_xy[:-1]
                    early_vectors.extend(steps[np.isfinite(steps).all(axis=1)])

            if early_vectors:
                mean_vector = np.nanmean(np.array(early_vectors), axis=0)
            else:
                mean_vector = np.array([0, 0], dtype=float)

            if np.linalg.norm(mean_vector) > 0:
                angle = (np.pi / 2) - np.arctan2(mean_vector[1], mean_vector[0])
            else:
                angle = 0

            transformed = {}

            for track_id in track_ids:
                rows = rows_by_track[track_id].sort_values('frame').copy()

                head_xy = rows[['x_head', 'y_head']].to_numpy(dtype=float) - origin

                head_xy = rotate_points(head_xy, angle)

                transformed[track_id] = {
                    'head': head_xy,
                }

            track_0 = 0 if 0 in transformed else track_ids[0]
            track_0_head = transformed[track_0]['head']
            track_0_head = track_0_head[np.isfinite(track_0_head).all(axis=1)]

            if len(track_0_head) > 0 and track_0_head[0, 0] > 0:
                for track_id in transformed:
                    transformed[track_id]['head'][:, 0] *= -1

            head_by_track = {}

            for track_id in track_ids:
                head_origin = first_valid_point(transformed[track_id]['head'])
                head_by_track[track_id] = register_to_origin(transformed[track_id]['head'], head_origin)

            return head_by_track

        first_trajectories = []

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file].sort_values('frame')

            track_ids = sorted(df['track_id'].unique())
            if len(track_ids) != 2:
                continue

            df_a = df[df['track_id'] == track_ids[0]]
            df_b = df[df['track_id'] == track_ids[1]]

            common_frames = sorted(set(df_a['frame']).intersection(df_b['frame']))
            if not common_frames:
                continue

            interaction_number = 0
            next_allowed_frame = -np.inf

            for frame in common_frames:
                if frame < next_allowed_frame:
                    continue

                row_a = df_a[df_a['frame'] == frame]
                row_b = df_b[df_b['frame'] == frame]
                if row_a.empty or row_b.empty:
                    continue

                min_dist, min_pair = compute_min_distance(row_a.iloc[0], row_b.iloc[0])

                if min_dist < proximity_threshold and set(min_pair) == {'head'}:
                    no_contact_frame = first_no_contact_frame(
                        df_a,
                        df_b,
                        common_frames,
                        frame
                    )

                    if no_contact_frame is None:
                        continue

                    interaction_number += 1
                    next_allowed_frame = no_contact_frame + window + 1
                    plot_frames = range(no_contact_frame, no_contact_frame + window + 1)

                    rows_by_track = {}
                    for track_id in track_ids:
                        rows = df[
                            (df['track_id'] == track_id) &
                            (df['frame'].isin(plot_frames))
                        ].copy()
                        rows_by_track[track_id] = rows

                    trajectory = transform_trajectory(rows_by_track, track_ids)
                    if trajectory is None:
                        continue

                    if interaction_number == 1 and frame < 1200:
                        first_trajectories.append({
                            'track_ids': track_ids,
                            'head': trajectory,
                        })

        def add_empty_page(pdf, title):
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.axis("off")
            ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=9)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        def mean_track_head(items, track_id):
            max_len = window + 1
            xs = []
            ys = []

            for item in items:
                if track_id not in item['head']:
                    continue

                xy = item['head'][track_id]
                x = np.full(max_len, np.nan)
                y = np.full(max_len, np.nan)
                n = min(len(xy), max_len)

                if n > 0:
                    x[:n] = xy[:n, 0]
                    y[:n] = xy[:n, 1]

                xs.append(x)
                ys.append(y)

            if not xs:
                return None

            mean_xy = np.column_stack((
                np.nanmean(np.array(xs), axis=0),
                np.nanmean(np.array(ys), axis=0)
            ))
            return register_to_origin(mean_xy, first_valid_point(mean_xy))

        pdf_path = os.path.join(output_dir, f"{condition_label}.pdf")

        with PdfPages(pdf_path) as pdf:
            if not first_trajectories:
                add_empty_page(pdf, "No first head trajectories found")
            else:
                fig, ax = plt.subplots(figsize=(4.8, 4.8))
                track_ids = sorted({
                    track_id
                    for item in first_trajectories
                    for track_id in item['track_ids']
                })
                legend_handles = []

                for track_id in track_ids:
                    colour = colours.get(track_id, "#333333")

                    for item in first_trajectories:
                        if track_id not in item['head']:
                            continue

                        xy = np.asarray(item['head'][track_id], dtype=float)
                        xy = xy[np.isfinite(xy).all(axis=1)]
                        if len(xy) == 0:
                            continue

                        ax.plot(
                            xy[:, 0],
                            xy[:, 1],
                            color=colour,
                            linewidth=1,
                            alpha=0.3,
                            zorder=1
                        )

                    mean_xy = mean_track_head(first_trajectories, track_id)
                    if mean_xy is None:
                        continue

                    plot_temporal_path(ax, mean_xy, colour)
                    legend_handles.append(Line2D(
                        [0],
                        [0],
                        color=colour,
                        marker='o',
                        linewidth=2.5,
                        markersize=5,
                        label=f"track {track_id} head mean"
                    ))

                ax.set_xlim(-15, 15)
                ax.set_ylim(-15, 15)
                ax.set_aspect('equal', adjustable='box')
                plot_clean_axes(ax)
                ax.set_xlabel("x, rotated and registered")
                ax.set_ylabel("y, rotated and registered")
                ax.set_title(f"first head mean trajectories with traces\nn = {len(first_trajectories)}", fontsize=9)
                ax.legend(handles=legend_handles, fontsize=7, loc="upper right", frameon=False)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        return None


    def trajectories_before_figures(self, proximity_threshold=1.5, window=60):

        output_dir = "/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/fed-starved/trajectories"
        os.makedirs(output_dir, exist_ok=True)

        path_parts = os.path.normpath(self.directory).split(os.sep)
        condition_label = "-".join(path_parts[-2:])
        condition = os.path.basename(os.path.normpath(self.directory))

        parts = ['head', 'body', 'tail']
        interaction_pairs = list(itertools.product(parts, parts))
        colours = {
            0: "#1f77b4",
            1: "#7b2cbf",
        }

        if condition == "fed-fed":
            colours = {
                0: "#2F5597",
                1: "#2F5597",
            }

        if condition == "starved-starved":
            colours = {
                0: "#9E2F35",
                1: "#9E2F35",
            }

        if condition == "fed-starved":
            colours = {
                0: "#F3D00E", #FDDA0D
                1: "#0B5D2A",
            }

        def compute_min_distance(row_a, row_b):
            coords_a = {p: np.array([row_a[f'x_{p}'], row_a[f'y_{p}']], dtype=float) for p in parts}
            coords_b = {p: np.array([row_b[f'x_{p}'], row_b[f'y_{p}']], dtype=float) for p in parts}

            min_dist = np.inf
            min_pair = None

            for p1, p2 in interaction_pairs:
                dist = np.linalg.norm(coords_a[p1] - coords_b[p2])
                if dist < min_dist:
                    min_dist = dist
                    min_pair = (p1, p2)

            return min_dist, min_pair

        def first_no_contact_frame(df_a, df_b, common_frames, interaction_frame):
            for f in common_frames:
                if f <= interaction_frame:
                    continue

                row_a = df_a[df_a['frame'] == f]
                row_b = df_b[df_b['frame'] == f]
                if row_a.empty or row_b.empty:
                    continue

                min_dist, _ = compute_min_distance(row_a.iloc[0], row_b.iloc[0])
                if min_dist > proximity_threshold:
                    return f

            return None

        def rotate_points(points, angle):
            rotation = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            return points @ rotation.T

        def first_valid_point(points):
            points = np.asarray(points, dtype=float).copy()
            valid = np.isfinite(points).all(axis=1)

            if valid.any():
                return points[np.where(valid)[0][0]]

            return None

        def last_valid_point(points):
            points = np.asarray(points, dtype=float).copy()
            valid = np.isfinite(points).all(axis=1)

            if valid.any():
                return points[np.where(valid)[0][-1]]

            return None

        def register_to_origin(points, origin):
            points = np.asarray(points, dtype=float).copy()

            if origin is not None:
                points -= origin

            return points

        def temporal_colours(colour, n):
            base = np.array(plt.matplotlib.colors.to_rgb(colour))
            white = np.array([1, 1, 1], dtype=float)
            shades = []

            for i in range(n):
                fade = 0 if n == 1 else i / (n - 1)
                shade = (base * (1 - fade * 0.7)) + (white * fade * 0.7)
                shades.append(shade)

            return shades

        def plot_clean_axes(ax):
            ax.grid(False)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)

        def plot_temporal_path(ax, xy, colour, linewidth=2.5, dot_size=10, alpha=0.9, zorder=2):
            xy = np.asarray(xy, dtype=float)
            valid = np.isfinite(xy).all(axis=1)
            xy = xy[valid]

            if len(xy) == 0:
                return

            dot_colours = temporal_colours(colour, len(xy))

            if len(xy) > 1:
                points = xy.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                line = LineCollection(
                    segments,
                    colors=temporal_colours(colour, len(segments)),
                    linewidths=linewidth,
                    alpha=alpha,
                    zorder=zorder
                )
                ax.add_collection(line)

            ax.scatter(
                xy[:, 0],
                xy[:, 1],
                color=dot_colours,
                s=dot_size,
                alpha=alpha,
                edgecolors='none',
                zorder=zorder + 1
            )

        def transform_trajectory(rows_by_track, track_ids):
            start_rows = [
                rows_by_track[track_id].iloc[0]
                for track_id in track_ids
                if not rows_by_track[track_id].empty
            ]

            if len(start_rows) != 2:
                return None

            start_heads = np.array([
                [row['x_head'], row['y_head']]
                for row in start_rows
            ], dtype=float)
            origin = np.nanmean(start_heads, axis=0)

            early_vectors = []
            for track_id in track_ids:
                rows = rows_by_track[track_id].sort_values('frame').head(4)
                head_xy = rows[['x_head', 'y_head']].to_numpy(dtype=float)
                if len(head_xy) > 1:
                    steps = head_xy[1:] - head_xy[:-1]
                    early_vectors.extend(steps[np.isfinite(steps).all(axis=1)])

            if early_vectors:
                mean_vector = np.nanmean(np.array(early_vectors), axis=0)
            else:
                mean_vector = np.array([0, 0], dtype=float)

            if np.linalg.norm(mean_vector) > 0:
                angle = (np.pi / 2) - np.arctan2(mean_vector[1], mean_vector[0])
            else:
                angle = 0

            transformed = {}

            for track_id in track_ids:
                rows = rows_by_track[track_id].sort_values('frame').copy()
                head_xy = rows[['x_head', 'y_head']].to_numpy(dtype=float) - origin
                head_xy = rotate_points(head_xy, angle)
                transformed[track_id] = {
                    'head': head_xy,
                }

            track_0 = 0 if 0 in transformed else track_ids[0]
            track_0_head = transformed[track_0]['head']
            track_0_head = track_0_head[np.isfinite(track_0_head).all(axis=1)]

            if len(track_0_head) > 0 and track_0_head[0, 0] > 0:
                for track_id in transformed:
                    transformed[track_id]['head'][:, 0] *= -1

            head_by_track = {}

            for track_id in track_ids:
                head_origin = last_valid_point(transformed[track_id]['head'])
                head_by_track[track_id] = register_to_origin(transformed[track_id]['head'], head_origin)

            return head_by_track

        first_trajectories = []

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file].sort_values('frame')

            track_ids = sorted(df['track_id'].unique())
            if len(track_ids) != 2:
                continue

            df_a = df[df['track_id'] == track_ids[0]]
            df_b = df[df['track_id'] == track_ids[1]]

            common_frames = sorted(set(df_a['frame']).intersection(df_b['frame']))
            if not common_frames:
                continue

            interaction_number = 0
            next_allowed_frame = -np.inf

            for frame in common_frames:
                if frame < next_allowed_frame:
                    continue

                row_a = df_a[df_a['frame'] == frame]
                row_b = df_b[df_b['frame'] == frame]
                if row_a.empty or row_b.empty:
                    continue

                min_dist, min_pair = compute_min_distance(row_a.iloc[0], row_b.iloc[0])

                if min_dist < proximity_threshold and set(min_pair) == {'head'}:
                    no_contact_frame = first_no_contact_frame(
                        df_a,
                        df_b,
                        common_frames,
                        frame
                    )

                    if no_contact_frame is None:
                        continue

                    interaction_number += 1
                    next_allowed_frame = no_contact_frame + 1
                    plot_frames = range(frame - window, frame + 1)

                    rows_by_track = {}
                    for track_id in track_ids:
                        rows = df[
                            (df['track_id'] == track_id) &
                            (df['frame'].isin(plot_frames))
                        ].copy()
                        rows_by_track[track_id] = rows

                    trajectory = transform_trajectory(rows_by_track, track_ids)
                    if trajectory is None:
                        continue

                    if interaction_number == 1 and frame < 1200:
                        first_trajectories.append({
                            'track_ids': track_ids,
                            'head': trajectory,
                        })

        def add_empty_page(pdf, title):
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.axis("off")
            ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=9)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        def mean_track_head(items, track_id):
            max_len = window + 1
            xs = []
            ys = []

            for item in items:
                if track_id not in item['head']:
                    continue

                xy = item['head'][track_id]
                x = np.full(max_len, np.nan)
                y = np.full(max_len, np.nan)
                n = min(len(xy), max_len)

                if n > 0:
                    x[:n] = xy[:n, 0]
                    y[:n] = xy[:n, 1]

                xs.append(x)
                ys.append(y)

            if not xs:
                return None

            mean_xy = np.column_stack((
                np.nanmean(np.array(xs), axis=0),
                np.nanmean(np.array(ys), axis=0)
            ))
            return register_to_origin(mean_xy, last_valid_point(mean_xy))

        pdf_path = os.path.join(output_dir, f"before_{condition_label}.pdf")

        with PdfPages(pdf_path) as pdf:
            if not first_trajectories:
                add_empty_page(pdf, "No first head trajectories before contact found")
            else:
                fig, ax = plt.subplots(figsize=(4.8, 4.8))
                track_ids = sorted({
                    track_id
                    for item in first_trajectories
                    for track_id in item['track_ids']
                })
                legend_handles = []

                for track_id in track_ids:
                    colour = colours.get(track_id, "#333333")

                    for item in first_trajectories:
                        if track_id not in item['head']:
                            continue

                        xy = np.asarray(item['head'][track_id], dtype=float)
                        xy = xy[np.isfinite(xy).all(axis=1)]
                        if len(xy) == 0:
                            continue

                        ax.plot(
                            xy[:, 0],
                            xy[:, 1],
                            color=colour,
                            linewidth=1,
                            alpha=0.5,
                            zorder=1
                        )

                    mean_xy = mean_track_head(first_trajectories, track_id)
                    if mean_xy is None:
                        continue

                    plot_temporal_path(ax, mean_xy, colour)
                    legend_handles.append(Line2D(
                        [0],
                        [0],
                        color=colour,
                        marker='o',
                        linewidth=2.5,
                        markersize=5,
                        label=f"track {track_id} head mean"
                    ))

                # ax.set_xlim(-60, 60) # gh
                # ax.set_ylim(-60, 60) # gh
                ax.set_xlim(-80, 80) # si
                ax.set_ylim(-80, 80) # si
                ax.set_aspect('equal', adjustable='box')
                plot_clean_axes(ax)
                ax.set_xlabel("x, rotated and registered")
                ax.set_ylabel("y, rotated and registered")
                ax.set_title(f"first head mean trajectories before contact\nn = {len(first_trajectories)}", fontsize=9)
                ax.legend(handles=legend_handles, fontsize=7, loc="upper right", frameon=False)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        return None


    def trajectory_2(
        self,
        proximity_threshold=1,
        window=15,
        output_dir=None
    ):

        if output_dir is None:
            output_dir = os.path.join(self.directory, "trajectories")
        os.makedirs(output_dir, exist_ok=True)

        parts = ['head', 'body', 'tail']
        interaction_pairs = list(itertools.product(parts, parts))
        colours = {
            0: "#1f77b4",
            1: "#7b2cbf",
        }

        def compute_min_distance(row_a, row_b):
            coords_a = {p: np.array([row_a[f'x_{p}'], row_a[f'y_{p}']], dtype=float) for p in parts}
            coords_b = {p: np.array([row_b[f'x_{p}'], row_b[f'y_{p}']], dtype=float) for p in parts}

            min_dist = np.inf
            min_pair = None

            for p1, p2 in interaction_pairs:
                dist = np.linalg.norm(coords_a[p1] - coords_b[p2])
                if dist < min_dist:
                    min_dist = dist
                    min_pair = (p1, p2)

            return min_dist, min_pair

        def first_no_contact_frame(df_a, df_b, common_frames, interaction_frame):
            for f in common_frames:
                if f <= interaction_frame:
                    continue

                row_a = df_a[df_a['frame'] == f]
                row_b = df_b[df_b['frame'] == f]
                if row_a.empty or row_b.empty:
                    continue

                min_dist, _ = compute_min_distance(row_a.iloc[0], row_b.iloc[0])
                if min_dist > proximity_threshold:
                    return f

            return None

        def rotate_points(points, angle):
            rotation = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            return points @ rotation.T

        def first_valid_point(points):
            points = np.asarray(points, dtype=float).copy()
            valid = np.isfinite(points).all(axis=1)

            if valid.any():
                return points[np.where(valid)[0][0]]

            return None

        def register_to_origin(points, origin):
            points = np.asarray(points, dtype=float).copy()

            if origin is not None:
                points -= origin

            return points

        def temporal_colours(colour, n):
            base = np.array(plt.matplotlib.colors.to_rgb(colour))
            white = np.array([1, 1, 1], dtype=float)
            shades = []

            for i in range(n):
                fade = 0 if n == 1 else i / (n - 1)
                shade = (base * (1 - fade * 0.7)) + (white * fade * 0.7)
                shades.append(shade)

            return shades

        def plot_clean_axes(ax):
            ax.grid(False)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)

        def plot_temporal_head_path(ax, head, colour, linewidth=1.4, dot_size=9, label=None, alpha=0.9, zorder=2):
            head = np.asarray(head, dtype=float)
            valid = np.isfinite(head).all(axis=1)
            head = head[valid]

            if len(head) == 0:
                return

            dot_colours = temporal_colours(colour, len(head))

            if len(head) > 1:
                points = head.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                segment_colours = temporal_colours(colour, len(segments))
                line = LineCollection(
                    segments,
                    colors=segment_colours,
                    linewidths=linewidth,
                    alpha=alpha,
                    label=label,
                    zorder=zorder
                )
                ax.add_collection(line)

            ax.scatter(
                head[:, 0],
                head[:, 1],
                color=dot_colours,
                s=dot_size,
                alpha=alpha,
                edgecolors='none',
                zorder=zorder + 1
            )

        def transform_trajectory(rows_by_track, track_ids):
            start_rows = [
                rows_by_track[track_id].iloc[0]
                for track_id in track_ids
                if not rows_by_track[track_id].empty
            ]

            if len(start_rows) != 2:
                return None

            start_heads = np.array([
                [row['x_head'], row['y_head']]
                for row in start_rows
            ], dtype=float)
            origin = np.nanmean(start_heads, axis=0)

            early_vectors = []
            for track_id in track_ids:
                rows = rows_by_track[track_id].sort_values('frame').head(4)
                head_xy = rows[['x_head', 'y_head']].to_numpy(dtype=float)
                if len(head_xy) > 1:
                    steps = head_xy[1:] - head_xy[:-1]
                    early_vectors.extend(steps[np.isfinite(steps).all(axis=1)])

            if early_vectors:
                mean_vector = np.nanmean(np.array(early_vectors), axis=0)
            else:
                mean_vector = np.array([0, 0], dtype=float)

            if np.linalg.norm(mean_vector) > 0:
                angle = (np.pi / 2) - np.arctan2(mean_vector[1], mean_vector[0])
            else:
                angle = 0

            transformed = {}

            for track_id in track_ids:
                rows = rows_by_track[track_id].sort_values('frame').copy()

                body_xy = rows[['x_body', 'y_body']].to_numpy(dtype=float) - origin
                head_xy = rows[['x_head', 'y_head']].to_numpy(dtype=float) - origin

                body_xy = rotate_points(body_xy, angle)
                head_xy = rotate_points(head_xy, angle)

                transformed[track_id] = {
                    'frames': rows['frame'].to_numpy(),
                    'body': body_xy,
                    'head': head_xy,
                }

            track_0 = 0 if 0 in transformed else track_ids[0]
            track_0_head = transformed[track_0]['head']
            track_0_head = track_0_head[np.isfinite(track_0_head).all(axis=1)]

            if len(track_0_head) > 0 and track_0_head[0, 0] > 0:
                for track_id in transformed:
                    transformed[track_id]['body'][:, 0] *= -1
                    transformed[track_id]['head'][:, 0] *= -1

            all_points = []
            for track_id in track_ids:
                head_origin = first_valid_point(transformed[track_id]['head'])
                transformed[track_id]['body'] = register_to_origin(transformed[track_id]['body'], head_origin)
                transformed[track_id]['head'] = register_to_origin(transformed[track_id]['head'], head_origin)

                body_xy = transformed[track_id]['body']
                head_xy = transformed[track_id]['head']

                all_points.extend(body_xy[np.isfinite(body_xy).all(axis=1)])
                all_points.extend(head_xy[np.isfinite(head_xy).all(axis=1)])

            return transformed, all_points

        trajectories = []

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file].sort_values('frame')

            track_ids = sorted(df['track_id'].unique())
            if len(track_ids) != 2:
                continue

            df_a = df[df['track_id'] == track_ids[0]]
            df_b = df[df['track_id'] == track_ids[1]]

            common_frames = sorted(set(df_a['frame']).intersection(df_b['frame']))
            if not common_frames:
                continue

            interaction_number = 0
            next_allowed_frame = -np.inf

            for frame in common_frames:
                if frame < next_allowed_frame:
                    continue

                row_a = df_a[df_a['frame'] == frame]
                row_b = df_b[df_b['frame'] == frame]
                if row_a.empty or row_b.empty:
                    continue

                min_dist, min_pair = compute_min_distance(row_a.iloc[0], row_b.iloc[0])

                if min_dist < proximity_threshold and set(min_pair) == {'head'}:
                    no_contact_frame = first_no_contact_frame(
                        df_a,
                        df_b,
                        common_frames,
                        frame
                    )

                    if no_contact_frame is None:
                        continue

                    interaction_number += 1
                    next_allowed_frame = no_contact_frame + window + 1
                    plot_frames = range(no_contact_frame, no_contact_frame + window + 1)

                    rows_by_track = {}
                    for track_id in track_ids:
                        rows = df[
                            (df['track_id'] == track_id) &
                            (df['frame'].isin(plot_frames))
                        ].copy()
                        rows_by_track[track_id] = rows

                    transformed = transform_trajectory(rows_by_track, track_ids)
                    if transformed is None:
                        continue

                    trajectory, all_points = transformed

                    trajectories.append({
                        'file': track_file,
                        'interaction_number': interaction_number,
                        'interaction_frame': frame,
                        'no_contact_frame': no_contact_frame,
                        'track_ids': track_ids,
                        'trajectory': trajectory,
                        'all_points': all_points,
                    })

        if not trajectories:
            print(f"No head-head trajectories found in {self.directory}")
            return pd.DataFrame()

        def add_empty_page(pdf, title):
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                title,
                ha="center",
                va="center",
                fontsize=9
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        def mean_track_xy(items, track_id, node):
            max_len = window + 1
            xs = []
            ys = []

            for item in items:
                if track_id not in item['trajectory']:
                    continue

                xy = item['trajectory'][track_id][node]
                x = np.full(max_len, np.nan)
                y = np.full(max_len, np.nan)
                n = min(len(xy), max_len)

                if n > 0:
                    x[:n] = xy[:n, 0]
                    y[:n] = xy[:n, 1]

                xs.append(x)
                ys.append(y)

            if not xs:
                return None

            mean_xy = np.column_stack((
                np.nanmean(np.array(xs), axis=0),
                np.nanmean(np.array(ys), axis=0)
            ))
            return register_to_origin(mean_xy, first_valid_point(mean_xy))

        def set_axis_from_points(ax, plotted_points):
            plotted_points = np.array(plotted_points, dtype=float)
            plotted_points = plotted_points[np.isfinite(plotted_points).all(axis=1)]

            if len(plotted_points) == 0:
                axis_limit = 1
            else:
                axis_limit = np.nanmax(np.abs(plotted_points))
                axis_limit = max(axis_limit * 1.35, 0.25)

            ax.set_xlim(-axis_limit, axis_limit)
            ax.set_ylim(-axis_limit, axis_limit)
            ax.set_aspect('equal', adjustable='box')
            plot_clean_axes(ax)

        def write_mean_pdf(pdf_path, items, title, node):
            with PdfPages(pdf_path) as pdf:
                if not items:
                    add_empty_page(pdf, f"No {node} trajectories for {title}")
                    return

                fig, ax = plt.subplots(figsize=(4.8, 4.8))
                track_ids = sorted({
                    track_id
                    for item in items
                    for track_id in item['track_ids']
                })
                legend_handles = []
                mean_points = []

                for track_id in track_ids:
                    colour = colours.get(track_id, "#333333")
                    node_mean = mean_track_xy(items, track_id, node)

                    if node_mean is None:
                        continue

                    mean_points.extend(node_mean[np.isfinite(node_mean).all(axis=1)])

                    plot_temporal_head_path(
                        ax,
                        node_mean,
                        colour,
                        linewidth=2.5,
                        dot_size=22,
                        label=f"track {track_id} {node} mean"
                    )
                    legend_handles.append(Line2D(
                        [0],
                        [0],
                        color=colour,
                        marker='o',
                        linewidth=2.5,
                        markersize=5,
                        label=f"track {track_id} {node} mean"
                    ))

                set_axis_from_points(ax, mean_points)
                ax.set_xlabel("x, rotated and registered")
                ax.set_ylabel("y, rotated and registered")
                ax.set_title(f"{title} {node} mean trajectories\nn = {len(items)}", fontsize=9)
                ax.legend(handles=legend_handles, fontsize=7, loc="upper right", frameon=False)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        def write_mean_with_trace_pdf(pdf_path, items, title, node):
            with PdfPages(pdf_path) as pdf:
                if not items:
                    add_empty_page(pdf, f"No {node} trajectories for {title}")
                    return

                fig, ax = plt.subplots(figsize=(4.8, 4.8))
                track_ids = sorted({
                    track_id
                    for item in items
                    for track_id in item['track_ids']
                })
                legend_handles = []
                plotted_points = []

                for track_id in track_ids:
                    colour = colours.get(track_id, "#333333")

                    for item in items:
                        if track_id not in item['trajectory']:
                            continue

                        xy = np.asarray(item['trajectory'][track_id][node], dtype=float)
                        valid = np.isfinite(xy).all(axis=1)
                        xy = xy[valid]

                        if len(xy) == 0:
                            continue

                        plotted_points.extend(xy)
                        ax.plot(
                            xy[:, 0],
                            xy[:, 1],
                            color=colour,
                            linewidth=0.45,
                            alpha=0.12,
                            zorder=1
                        )

                    node_mean = mean_track_xy(items, track_id, node)

                    if node_mean is None:
                        continue

                    plotted_points.extend(node_mean[np.isfinite(node_mean).all(axis=1)])

                    plot_temporal_head_path(
                        ax,
                        node_mean,
                        colour,
                        linewidth=2.5,
                        dot_size=22,
                        label=f"track {track_id} {node} mean"
                    )
                    legend_handles.append(Line2D(
                        [0],
                        [0],
                        color=colour,
                        marker='o',
                        linewidth=2.5,
                        markersize=5,
                        label=f"track {track_id} {node} mean"
                    ))

                set_axis_from_points(ax, plotted_points)
                ax.set_xlabel("x, rotated and registered")
                ax.set_ylabel("y, rotated and registered")
                ax.set_title(f"{title} {node} mean trajectories with traces\nn = {len(items)}", fontsize=9)
                ax.legend(handles=legend_handles, fontsize=7, loc="upper right", frameon=False)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        first_contact_cutoff_frame = 1200

        def trajectory_group(item):
            if (
                item['interaction_number'] == 1
                and item['interaction_frame'] < first_contact_cutoff_frame
            ):
                return 'first'
            if item['interaction_number'] > 1:
                return 'other'
            return 'excluded_first_after_cutoff'

        first_trajectories = [item for item in trajectories if trajectory_group(item) == 'first']
        other_trajectories = [item for item in trajectories if trajectory_group(item) == 'other']

        window_prefix = f"window{window}_"
        for node in ['head', 'body']:
            first_mean_withtrace_pdf = os.path.join(output_dir, f"{window_prefix}first_mean_withtrace_{node}.pdf")
            first_mean_pdf = os.path.join(output_dir, f"{window_prefix}first_mean_{node}.pdf")
            other_mean_withtrace_pdf = os.path.join(output_dir, f"{window_prefix}other_mean_withtrace_{node}.pdf")
            other_mean_pdf = os.path.join(output_dir, f"{window_prefix}other_mean_{node}.pdf")

            write_mean_with_trace_pdf(first_mean_withtrace_pdf, first_trajectories, "first", node)
            write_mean_pdf(first_mean_pdf, first_trajectories, "first", node)
            write_mean_with_trace_pdf(other_mean_withtrace_pdf, other_trajectories, "other", node)
            write_mean_pdf(other_mean_pdf, other_trajectories, "other", node)

        summary = pd.DataFrame([
            {
                'file': item['file'],
                'interaction_number': item['interaction_number'],
                'interaction_frame': item['interaction_frame'],
                'no_contact_frame': item['no_contact_frame'],
                'trajectory_group': trajectory_group(item),
            }
            for item in trajectories
        ])

        summary.to_csv(
            os.path.join(output_dir, f"{window_prefix}trajectory_2_summary.csv"),
            index=False
        )
        return summary



    def trajectory_before(self, proximity_threshold=1, window=15):

        output_dir = os.path.join(self.directory, "trajectory_before")
        os.makedirs(output_dir, exist_ok=True)

        parts = ['head', 'body', 'tail']
        interaction_pairs = list(itertools.product(parts, parts))
        colours = {
            0: "#1f77b4",  # nice blue for track 0
            1: "#7b2cbf",  # nice purple for track 1
        }

        def compute_min_distance(row_a, row_b):
            coords_a = {p: np.array([row_a[f'x_{p}'], row_a[f'y_{p}']], dtype=float) for p in parts}
            coords_b = {p: np.array([row_b[f'x_{p}'], row_b[f'y_{p}']], dtype=float) for p in parts}

            min_dist = np.inf
            min_pair = None

            for p1, p2 in interaction_pairs:
                dist = np.linalg.norm(coords_a[p1] - coords_b[p2])
                if dist < min_dist:
                    min_dist = dist
                    min_pair = (p1, p2)

            return min_dist, min_pair

        def first_no_contact_frame(df_a, df_b, common_frames, interaction_frame):
            for f in common_frames:
                if f <= interaction_frame:
                    continue

                row_a = df_a[df_a['frame'] == f]
                row_b = df_b[df_b['frame'] == f]
                if row_a.empty or row_b.empty:
                    continue

                min_dist, _ = compute_min_distance(row_a.iloc[0], row_b.iloc[0])
                if min_dist > proximity_threshold:
                    return f

            return None

        def rotate_points(points, angle):
            rotation = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            return points @ rotation.T

        def temporal_colours(colour, n):
            base = np.array(plt.matplotlib.colors.to_rgb(colour))
            white = np.array([1, 1, 1], dtype=float)
            shades = []

            for i in range(n):
                if n == 1:
                    fade = 0
                else:
                    fade = i / (n - 1)

                shade = (base * (1 - fade * 0.7)) + (white * fade * 0.7)
                shades.append(shade)

            return shades

        def plot_temporal_head_path(ax, head, colour, linewidth=1.4, dot_size=9, label=None):
            head = np.asarray(head, dtype=float)
            valid = np.isfinite(head).all(axis=1)
            head = head[valid]

            if len(head) == 0:
                return

            dot_colours = temporal_colours(colour, len(head))

            if len(head) > 1:
                points = head.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                segment_colours = temporal_colours(colour, len(segments))
                line = LineCollection(
                    segments,
                    colors=segment_colours,
                    linewidths=linewidth,
                    label=label
                )
                ax.add_collection(line)

            ax.scatter(
                head[:, 0],
                head[:, 1],
                color=dot_colours,
                s=dot_size,
                alpha=0.9,
                edgecolors='none'
            )

        def transform_trajectory(rows_by_track, track_ids):
            start_rows = [
                rows_by_track[track_id].iloc[0]
                for track_id in track_ids
                if not rows_by_track[track_id].empty
            ]

            if len(start_rows) != 2:
                return None

            start_heads = np.array([
                [row['x_head'], row['y_head']]
                for row in start_rows
            ], dtype=float)
            origin = np.nanmean(start_heads, axis=0)

            early_vectors = []
            for track_id in track_ids:
                rows = rows_by_track[track_id].sort_values('frame').head(4)
                head_xy = rows[['x_head', 'y_head']].to_numpy(dtype=float)
                if len(head_xy) > 1:
                    steps = head_xy[1:] - head_xy[:-1]
                    early_vectors.extend(steps[np.isfinite(steps).all(axis=1)])

            if early_vectors:
                mean_vector = np.nanmean(np.array(early_vectors), axis=0)
            else:
                mean_vector = np.array([0, 0], dtype=float)

            if np.linalg.norm(mean_vector) > 0:
                angle = (np.pi / 2) - np.arctan2(mean_vector[1], mean_vector[0])
            else:
                angle = 0

            transformed = {}

            for track_id in track_ids:
                rows = rows_by_track[track_id].sort_values('frame').copy()

                body_xy = rows[['x_body', 'y_body']].to_numpy(dtype=float) - origin
                head_xy = rows[['x_head', 'y_head']].to_numpy(dtype=float) - origin

                body_xy = rotate_points(body_xy, angle)
                head_xy = rotate_points(head_xy, angle)

                transformed[track_id] = {
                    'frames': rows['frame'].to_numpy(),
                    'body': body_xy,
                    'head': head_xy,
                }

            track_0 = 0 if 0 in transformed else track_ids[0]
            track_0_head = transformed[track_0]['head']
            track_0_head = track_0_head[np.isfinite(track_0_head).all(axis=1)]

            if len(track_0_head) > 0 and track_0_head[0, 0] > 0:
                for track_id in transformed:
                    transformed[track_id]['body'][:, 0] *= -1
                    transformed[track_id]['head'][:, 0] *= -1

            all_points = []
            for track_id in track_ids:
                body_xy = transformed[track_id]['body']
                head_xy = transformed[track_id]['head']

                all_points.extend(body_xy[np.isfinite(body_xy).all(axis=1)])
                all_points.extend(head_xy[np.isfinite(head_xy).all(axis=1)])

            return transformed, all_points

        trajectories = []

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file].sort_values('frame')

            track_ids = sorted(df['track_id'].unique())
            if len(track_ids) != 2:
                continue

            df_a = df[df['track_id'] == track_ids[0]]
            df_b = df[df['track_id'] == track_ids[1]]

            common_frames = sorted(set(df_a['frame']).intersection(df_b['frame']))
            if not common_frames:
                continue

            interaction_number = 0
            next_allowed_frame = -np.inf

            for frame in common_frames:
                if frame < next_allowed_frame:
                    continue

                row_a = df_a[df_a['frame'] == frame]
                row_b = df_b[df_b['frame'] == frame]
                if row_a.empty or row_b.empty:
                    continue

                min_dist, min_pair = compute_min_distance(row_a.iloc[0], row_b.iloc[0])

                if min_dist < proximity_threshold and set(min_pair) == {'head'}:
                    no_contact_frame = first_no_contact_frame(
                        df_a,
                        df_b,
                        common_frames,
                        frame
                    )

                    if no_contact_frame is None:
                        continue

                    interaction_number += 1
                    next_allowed_frame = no_contact_frame + 1
                    plot_frames = range(frame - window, frame + 1)

                    rows_by_track = {}
                    for track_id in track_ids:
                        rows = df[
                            (df['track_id'] == track_id) &
                            (df['frame'].isin(plot_frames))
                        ].copy()
                        rows_by_track[track_id] = rows

                    transformed = transform_trajectory(rows_by_track, track_ids)
                    if transformed is None:
                        continue

                    trajectory, all_points = transformed

                    trajectories.append({
                        'file': track_file,
                        'interaction_number': interaction_number,
                        'interaction_frame': frame,
                        'no_contact_frame': no_contact_frame,
                        'track_ids': track_ids,
                        'trajectory': trajectory,
                        'all_points': all_points,
                    })

        if not trajectories:
            print(f"No head-head trajectories before contact found in {self.directory}")
            return pd.DataFrame()

        all_points = []
        for item in trajectories:
            all_points.extend(item['all_points'])

        all_points = np.array(all_points, dtype=float)
        all_points = all_points[np.isfinite(all_points).all(axis=1)]

        if len(all_points) == 0:
            axis_limit = 1
        else:
            axis_limit = np.nanmax(np.abs(all_points))
            axis_limit = max(axis_limit * 1.15, 1)

        def plot_one_axis(ax, item, show_labels=False):
            for track_id in item['track_ids']:
                colour = colours.get(track_id, "#333333")
                trajectory = item['trajectory'][track_id]
                head = trajectory['head']

                plot_temporal_head_path(ax, head, colour)

            ax.axhline(0, color="#dddddd", linewidth=0.8)
            ax.axvline(0, color="#dddddd", linewidth=0.8)
            ax.set_xlim(-axis_limit, axis_limit)
            ax.set_ylim(-axis_limit, axis_limit)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, color="#eeeeee", linewidth=0.6)
            if show_labels:
                ax.set_xlabel("x")
                ax.set_ylabel("y")
            else:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            ax.set_title(
                f"{item['file']}\n"
                f"int {item['interaction_number']} | "
                f"c {item['interaction_frame']} | "
                f"nc {item['no_contact_frame']}",
                fontsize=5
            )

        def add_empty_page(pdf, title):
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                title,
                ha="center",
                va="center",
                fontsize=9
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        def write_grid_pdf(pdf_path, items, empty_title, plots_per_page=25):
            with PdfPages(pdf_path) as pdf:
                if not items:
                    add_empty_page(pdf, empty_title)
                    return

                for start in range(0, len(items), plots_per_page):
                    page_items = items[start:start + plots_per_page]
                    fig, axes = plt.subplots(5, 5, figsize=(11, 11))
                    axes = axes.flatten()

                    for ax, item in zip(axes, page_items):
                        plot_one_axis(ax, item)

                    for ax in axes[len(page_items):]:
                        ax.axis("off")

                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

        def mean_track_xy(items, track_id, node):
            max_len = window + 1
            xs = []
            ys = []

            for item in items:
                if track_id not in item['trajectory']:
                    continue

                xy = item['trajectory'][track_id][node]
                x = np.full(max_len, np.nan)
                y = np.full(max_len, np.nan)
                n = min(len(xy), max_len)

                if n > 0:
                    x[:n] = xy[:n, 0]
                    y[:n] = xy[:n, 1]

                xs.append(x)
                ys.append(y)

            if not xs:
                return None

            return np.column_stack((
                np.nanmean(np.array(xs), axis=0),
                np.nanmean(np.array(ys), axis=0)
            ))

        def write_mean_pdf(pdf_path, items, title):
            with PdfPages(pdf_path) as pdf:
                if not items:
                    add_empty_page(pdf, f"No trajectories for {title}")
                    return

                fig, ax = plt.subplots(figsize=(4.8, 4.8))
                track_ids = sorted({
                    track_id
                    for item in items
                    for track_id in item['track_ids']
                })
                legend_handles = []
                mean_points = []

                for track_id in track_ids:
                    colour = colours.get(track_id, "#333333")
                    head_mean = mean_track_xy(items, track_id, 'head')

                    if head_mean is None:
                        continue

                    mean_points.extend(head_mean[np.isfinite(head_mean).all(axis=1)])

                    plot_temporal_head_path(
                        ax,
                        head_mean,
                        colour,
                        linewidth=2.5,
                        dot_size=22,
                        label=f"track {track_id} head mean"
                    )
                    legend_handles.append(Line2D(
                        [0],
                        [0],
                        color=colour,
                        marker='o',
                        linewidth=2.5,
                        markersize=5,
                        label=f"track {track_id} head mean"
                    ))

                ax.axhline(0, color="#dddddd", linewidth=0.8)
                ax.axvline(0, color="#dddddd", linewidth=0.8)
                mean_points = np.array(mean_points, dtype=float)
                if len(mean_points) == 0:
                    mean_axis_limit = 1
                else:
                    mean_axis_limit = np.nanmax(np.abs(mean_points))
                    mean_axis_limit = max(mean_axis_limit * 1.35, 0.25)

                ax.set_xlim(-mean_axis_limit, mean_axis_limit)
                ax.set_ylim(-mean_axis_limit, mean_axis_limit)
                ax.set_aspect('equal', adjustable='box')
                ax.grid(True, color="#eeeeee", linewidth=0.6)
                ax.set_xlabel("x, rotated and centred")
                ax.set_ylabel("y, rotated and centred")
                ax.set_title(f"{title} mean trajectories\nn = {len(items)}", fontsize=9)
                ax.legend(handles=legend_handles, fontsize=7, loc="upper right", frameon=False)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        window_prefix = f"window{window}_"
        first_pdf_name = f"{window_prefix}first.pdf"
        other_pdf_name = f"{window_prefix}other.pdf"

        first_pdf = os.path.join(output_dir, first_pdf_name)
        other_pdf = os.path.join(output_dir, other_pdf_name)
        first_mean_pdf = os.path.join(output_dir, f"{window_prefix}first_mean.pdf")
        other_mean_pdf = os.path.join(output_dir, f"{window_prefix}other_mean.pdf")
        all_mean_pdf = os.path.join(output_dir, f"{window_prefix}all_mean.pdf")

        first_contact_cutoff_frame = 1200

        def trajectory_group(item):
            if (
                item['interaction_number'] == 1
                and item['interaction_frame'] < first_contact_cutoff_frame
            ):
                return 'first'
            if item['interaction_number'] > 1:
                return 'other'
            return 'excluded_first_after_cutoff'

        first_trajectories = [item for item in trajectories if trajectory_group(item) == 'first']
        other_trajectories = [item for item in trajectories if trajectory_group(item) == 'other']

        write_grid_pdf(first_pdf, first_trajectories, "No first head-head trajectories before contact found")
        write_grid_pdf(other_pdf, other_trajectories, "No later head-head trajectories before contact found")
        write_mean_pdf(first_mean_pdf, first_trajectories, "first")
        write_mean_pdf(other_mean_pdf, other_trajectories, "other")
        write_mean_pdf(all_mean_pdf, trajectories, "all")

        summary = pd.DataFrame([
            {
                'file': item['file'],
                'interaction_number': item['interaction_number'],
                'interaction_frame': item['interaction_frame'],
                'no_contact_frame': item['no_contact_frame'],
                'trajectory_group': trajectory_group(item),
                'pdf': (
                    first_pdf_name if trajectory_group(item) == 'first'
                    else other_pdf_name if trajectory_group(item) == 'other'
                    else ''
                ),
            }
            for item in trajectories
        ])

        summary.to_csv(
            os.path.join(output_dir, f"{window_prefix}trajectory_summary.csv"),
            index=False
        )
        return summary

    
    # METHOD MOVEMENT_DIRECTION: CALCULATES THE DIRECTION OF MOVEMENT BASED ON BODY NODES OVER TIME 
    def movement_direction(self):

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
        
        data = []

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file]
            df = df.sort_values(['track_id', 'frame'])

            for track_id, group in df.groupby('track_id'):
                group = group.sort_values(by='frame')

                body_positions = group[['x_body', 'y_body']].to_numpy(dtype=float)

                vectors = body_positions[1:] - body_positions[:-1] # foo[:-1] (slice) give me everything up to, but not including, the last item
                # makes two lists which are then subtracted to give vector between consecutive frames

                angles = [angle_calculator(vectors[i], vectors[i+1]) for i in range(len(vectors)-1)]

                # angle_frames = group['frame'].to_numpy()[2:]
                angle_frames = group['frame'].to_numpy()[1:-1]


                data.append(pd.DataFrame({
                    'file': track_file,
                    'track_id': track_id,
                    'frame': angle_frames,
                    'movement_angle': angles
                }))
        
        angle_df = pd.concat(data, ignore_index=True)
        angle_df.to_csv(os.path.join(self.directory, "movement_direction.csv"), index=False)
        return angle_df
    


    # METHOD COMPUTE_DIGGING: THIS METHOD DETECTS IF LARVAE ARE DIGGING (IN ABSENCE OF MAN-MADE HOLE)

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


    # METHOD TOTAL_DIGGING: THIS METHOD DETECTS HOW MANY LARVAE ARE DIGGING 

    def total_digging(self, total_larvae=None, cleaned=False):

        digging_behaviour = []

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file]
            df = self.compute_digging(df)  # apply dynamic method

            digging_behaviour.append({
                'file': track_file,
                'digging_behaviour': 'Y' if df['digging_status'].any() else 'N'
            })

        result = pd.DataFrame(digging_behaviour)
        filename = "digging_behaviour.csv"

        result.to_csv(os.path.join(self.directory, filename), index=False)
        return result
    

    ### METHOD DIGGING_BEHAVIOUR:

    def digging_behaviour(self):

        single_larvae = []
        two_larvae = []

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file]
            df = self.compute_digging(df)

            for frame, group in df.groupby('frame'):
                digging = group[group['digging_status']]
                not_digging = group[~group['digging_status']]

                # Exactly 1 digger: distance to all others
                if len(digging) == 1 and not not_digging.empty:

                    digger_coords = digging[['x_body', 'y_body']].values
                    others_coords = not_digging[['x_body', 'y_body']].values
                    distances = cdist(digger_coords, others_coords)[0]

                    for target_id, dist in zip(not_digging['track_id'], distances):
                        single_larvae.append({
                            'frame': frame,
                            'file': track_file,
                            'digger_id': digging['track_id'].values[0],
                            'target_id': target_id,
                            'distance': dist
                        })
                # Exactly 2 diggers: mutual distance
                elif len(digging) == 2:
                    coords = digging[['x_body', 'y_body']].values
                    dist = np.linalg.norm(coords[0] - coords[1])
                    ids = digging['track_id'].values
                    two_larvae.append({
                        'frame': frame,
                        'file': track_file,
                        'digger_id_1': ids[0],
                        'digger_id_2': ids[1],
                        'distance': dist
                    })

        df_single = pd.DataFrame(single_larvae)    
        df_two = pd.DataFrame(two_larvae)

        df_single.to_csv(os.path.join(self.directory, 'digging_distances_single.csv'), index=False)
        df_two.to_csv(os.path.join(self.directory, 'digging_distances_pair.csv'), index=False)

    


    ### METHOD INTERACTION_TYPES_CLOSEST: COUNTS CLOSEST! PROXIMAL CONTACTS BETWEEN LARVAE (1MM THRESHOLD) 
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


    # METHOD FOOD_PLATES_INTERACTIONS: INTERACTION ANALYSIS WITH DIGGING CONTEXT KEPT

    def food_plates_interactions(self, threshold=1.0, continue_threshold=1.5):

        """
        Food plates need the digging rows kept, not masked out.
        This method:
        - detects digging before mm conversion
        - marks whether each larva has already dug by each frame
        - detects frame-level contacts
        - detects bouts using the same start/continue logic as interaction_type_bout
        """

        body_parts = ['head', 'body', 'tail']
        interaction_pairs = list(itertools.product(body_parts, body_parts))

        unified_types = [
            'head_head', 'tail_tail', 'body_body',
            'body_head', 'body_tail', 'head_tail'
        ]

        contact_columns = [
            'file', 'frame', 'Interaction Pair',
            'track_1', 'track_2', 'track_1_node', 'track_2_node',
            'Distance', 'Closest Interaction Type',
            'track_1_digging_status', 'track_2_digging_status',
            'track_1_has_dug', 'track_2_has_dug',
            'active_digging_count', 'active_digging_context',
            'dug_count', 'dug_context'
        ]

        bout_columns = [
            'file', 'bout_id', 'track_1', 'track_2',
            'start_frame', 'end_frame', 'duration',
            'initial_type', 'predominant_type',
            'start_track_1_digging_status', 'start_track_2_digging_status',
            'start_track_1_has_dug', 'start_track_2_has_dug',
            'start_active_digging_context', 'start_dug_context',
            'end_track_1_digging_status', 'end_track_2_digging_status',
            'end_track_1_has_dug', 'end_track_2_has_dug',
            'end_active_digging_context', 'end_dug_context',
            'track_1_digging_frames', 'track_2_digging_frames',
            'either_digging_frames', 'both_digging_frames',
            'track_1_dug_during_bout', 'track_2_dug_during_bout',
            'dug_count_during_bout', 'dug_context_during_bout',
            'any_digging_during_bout', 'both_digging_same_frame_during_bout',
        ] + unified_types

        def unify_interaction_type(part1, part2):
            return '_'.join(sorted([part1, part2]))

        def digging_context(count):
            if count == 0:
                return 'neither_dug'
            if count == 1:
                return 'one_dug'
            return 'both_dug'

        def get_conversion_factor(match):
            perimeter_polygon = match.get('perimeter_polygon')

            if perimeter_polygon:
                minx, miny, maxx, maxy = perimeter_polygon.bounds
                diameter = maxx - minx
                conversion_factor = 90 / diameter

                threshold = 0.09
                if conversion_factor > threshold:
                    conversion_factor = 90 / 1032
            else:
                conversion_factor = 90 / 1032

            return conversion_factor

        def prepare_food_plate_df(df, conversion_factor):
            df = self.compute_digging(df.copy())
            df = df.sort_values(['track_id', 'frame']).reset_index(drop=True)
            df['has_dug'] = df.groupby('track_id')['digging_status'].cummax()

            pixel_columns = ['x_tail', 'y_tail', 'x_body', 'y_body', 'x_head', 'y_head']
            df[pixel_columns] = df[pixel_columns] * conversion_factor

            return df

        def pair_digging_summary(frame_data, pair):
            row_1 = frame_data[frame_data['track_id'] == pair[0]]
            row_2 = frame_data[frame_data['track_id'] == pair[1]]

            if row_1.empty or row_2.empty:
                return None

            row_1 = row_1.iloc[0]
            row_2 = row_2.iloc[0]

            track_1_digging = bool(row_1['digging_status'])
            track_2_digging = bool(row_2['digging_status'])
            track_1_has_dug = bool(row_1['has_dug'])
            track_2_has_dug = bool(row_2['has_dug'])

            active_count = int(track_1_digging) + int(track_2_digging)
            dug_count = int(track_1_has_dug) + int(track_2_has_dug)

            return {
                'track_1_digging_status': track_1_digging,
                'track_2_digging_status': track_2_digging,
                'track_1_has_dug': track_1_has_dug,
                'track_2_has_dug': track_2_has_dug,
                'active_digging_count': active_count,
                'active_digging_context': digging_context(active_count),
                'dug_count': dug_count,
                'dug_context': digging_context(dug_count)
            }

        def add_digging_to_bout(bout, summary):
            bout['end_digging'] = summary.copy()
            bout['track_1_digging_frames'] += int(summary['track_1_digging_status'])
            bout['track_2_digging_frames'] += int(summary['track_2_digging_status'])
            bout['either_digging_frames'] += int(summary['active_digging_count'] > 0)
            bout['both_digging_frames'] += int(summary['active_digging_count'] == 2)

        def finalize_bout(track_file, pair, bout, bouts):
            interactions_all = bout['interactions']
            if not interactions_all:
                return

            type_counts = Counter(interactions_all)
            start = bout['start_digging']
            end = bout['end_digging']
            track_1_dug_during_bout = bout['track_1_digging_frames'] > 0
            track_2_dug_during_bout = bout['track_2_digging_frames'] > 0
            dug_count_during_bout = (
                int(track_1_dug_during_bout)
                + int(track_2_dug_during_bout)
            )

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
                'start_track_1_digging_status': start['track_1_digging_status'],
                'start_track_2_digging_status': start['track_2_digging_status'],
                'start_track_1_has_dug': start['track_1_has_dug'],
                'start_track_2_has_dug': start['track_2_has_dug'],
                'start_active_digging_context': start['active_digging_context'],
                'start_dug_context': start['dug_context'],
                'end_track_1_digging_status': end['track_1_digging_status'],
                'end_track_2_digging_status': end['track_2_digging_status'],
                'end_track_1_has_dug': end['track_1_has_dug'],
                'end_track_2_has_dug': end['track_2_has_dug'],
                'end_active_digging_context': end['active_digging_context'],
                'end_dug_context': end['dug_context'],
                'track_1_digging_frames': bout['track_1_digging_frames'],
                'track_2_digging_frames': bout['track_2_digging_frames'],
                'either_digging_frames': bout['either_digging_frames'],
                'both_digging_frames': bout['both_digging_frames'],
                'track_1_dug_during_bout': track_1_dug_during_bout,
                'track_2_dug_during_bout': track_2_dug_during_bout,
                'dug_count_during_bout': dug_count_during_bout,
                'dug_context_during_bout': digging_context(dug_count_during_bout),
                'any_digging_during_bout': bout['either_digging_frames'] > 0,
                'both_digging_same_frame_during_bout': bout['both_digging_frames'] > 0,
            }

            for t in unified_types:
                bout_data[t] = type_counts.get(t, 0)

            bouts.append(bout_data)

        contact_rows = []
        bouts = []

        for match in self.matching_pairs:
            track_file = match['track_file']
            conversion_factor = get_conversion_factor(match)
            df = prepare_food_plate_df(self.track_data[track_file], conversion_factor)
            df.sort_values(by='frame', inplace=True)

            active_bouts = {}
            bout_counter = 0
            file_contact_count = 0

            for frame in df['frame'].unique():
                frame_data = df[df['frame'] == frame]
                track_ids = sorted(frame_data['track_id'].unique())

                coords = {
                    part: {
                        row['track_id']: np.array([row[f'x_{part}'], row[f'y_{part}']])
                        for _, row in frame_data.iterrows()
                    }
                    for part in body_parts
                }

                interacting_pairs = {}
                close_pairs = {}

                for id1, id2 in itertools.combinations(track_ids, 2):

                    interactions = []
                    min_dist = float('inf')
                    closest_type = None
                    closest_part_1 = None
                    closest_part_2 = None

                    for part1, part2 in interaction_pairs:
                        coord1 = coords[part1].get(id1)
                        coord2 = coords[part2].get(id2)
                        if coord1 is None or coord2 is None:
                            continue

                        dist = np.linalg.norm(coord1 - coord2)

                        if dist < min_dist:
                            min_dist = dist
                            closest_type = unify_interaction_type(part1, part2)
                            closest_part_1 = part1
                            closest_part_2 = part2

                        if dist < threshold:
                            interactions.append(unify_interaction_type(part1, part2))

                    pair_key = tuple(sorted((id1, id2)))

                    summary = pair_digging_summary(frame_data, pair_key)
                    # Active digging itself should not be counted as interaction.
                    # The has_dug columns are still kept for later non-digging contacts.
                    if summary is None or summary['active_digging_count'] > 0:
                        continue

                    if closest_type is not None and min_dist < continue_threshold:
                        close_pairs[pair_key] = closest_type

                    if interactions:
                        interacting_pairs[pair_key] = interactions
                        summary = pair_digging_summary(frame_data, pair_key)

                        if summary is not None:
                            contact_rows.append({
                                'file': track_file,
                                'frame': frame,
                                'Interaction Pair': pair_key,
                                'track_1': pair_key[0],
                                'track_2': pair_key[1],
                                'track_1_node': closest_part_1,
                                'track_2_node': closest_part_2,
                                'Distance': min_dist,
                                'Closest Interaction Type': closest_type,
                                **summary
                            })
                            file_contact_count += 1

                current_close = set(close_pairs.keys())

                for pair in list(active_bouts.keys()):
                    if pair not in current_close:
                        bout = active_bouts.pop(pair)
                        finalize_bout(track_file, pair, bout, bouts)

                for pair in list(active_bouts.keys()):
                    summary = pair_digging_summary(frame_data, pair)
                    if summary is None:
                        continue

                    active_bouts[pair]['end_frame'] = frame
                    add_digging_to_bout(active_bouts[pair], summary)

                    if pair in interacting_pairs:
                        active_bouts[pair]['interactions'].extend(interacting_pairs[pair])
                    else:
                        active_bouts[pair]['interactions'].append(close_pairs[pair])

                for pair, interactions in interacting_pairs.items():
                    if pair in active_bouts:
                        continue

                    summary = pair_digging_summary(frame_data, pair)
                    if summary is None:
                        continue

                    active_bouts[pair] = {
                        'bout_id': bout_counter,
                        'start_frame': frame,
                        'end_frame': frame,
                        'interactions': interactions.copy(),
                        'start_digging': summary.copy(),
                        'end_digging': summary.copy(),
                        'track_1_digging_frames': 0,
                        'track_2_digging_frames': 0,
                        'either_digging_frames': 0,
                        'both_digging_frames': 0,
                    }
                    add_digging_to_bout(active_bouts[pair], summary)
                    bout_counter += 1

            for pair, bout in active_bouts.items():
                finalize_bout(track_file, pair, bout, bouts)

            if file_contact_count == 0:
                contact_rows.append({
                    'file': track_file,
                    'frame': np.nan,
                    'Interaction Pair': None,
                    'Distance': np.nan,
                    'Closest Interaction Type': None,
                })
                print(f"No food-plate contact frames for {track_file}")

        contact_df = pd.DataFrame(contact_rows)
        for column in contact_columns:
            if column not in contact_df.columns:
                contact_df[column] = np.nan
        contact_df = contact_df[contact_columns]

        bout_df = pd.DataFrame(bouts)
        for column in bout_columns:
            if column not in bout_df.columns:
                bout_df[column] = np.nan
        bout_df = bout_df[bout_columns]

        if not bout_df.empty:
            bout_df = bout_df.sort_values(by=['file', 'bout_id'])

        if self.shorten and self.shorten_duration is not None:
            suffix = f"_{self.shorten_duration}"
        else:
            suffix = ""

        contact_filename = f"food_plate_interaction_frames.csv"
        bout_filename = f"food_plate_interaction_bouts.csv"

        contact_df.to_csv(os.path.join(self.directory, contact_filename), index=False)
        bout_df.to_csv(os.path.join(self.directory, bout_filename), index=False)

        return contact_df, bout_df


    # METHOD INTERACTION_BOUT_DYNAMICS: 

    def interaction_bout_dynamics(self): ### method above must be run already 

        bout_df = pd.read_csv(os.path.join(self.directory, "interaction_type_bout.csv"))

        # Melt into long form: one row per larva per bout
        larva_rows = []
        for _, row in bout_df.iterrows():
            for role in ['track_1', 'track_2']:
                larva_id = row[role]
                partner_id = row['track_2'] if role == 'track_1' else row['track_1'] # want both prospectives 
                larva_rows.append({
                    'file': row['file'],
                    'larva_id': larva_id,
                    'partner_id': partner_id,
                    'start_frame': row['start_frame'],
                    'end_frame': row['end_frame'],
                    'duration': row['duration'],
                    'initial_type': row['initial_type'],
                    'predominant_type': row['predominant_type']
                    # 'original_bout_id': row['bout_id']
                })

        df = pd.DataFrame(larva_rows)
        df.sort_values(by=['file', 'larva_id', 'start_frame'], inplace=True)

        df['bout_number'] = df.groupby(['file', 'larva_id']).cumcount() + 1 #bout id per larva 
        df['time_since_last_bout'] = df.groupby(['file', 'larva_id'])['start_frame'].diff().fillna(pd.NA) # time since last bout

        # Previous partner
        df['prev_partner'] = df.groupby(['file', 'larva_id'])['partner_id'].shift()
        df['same_partner'] = df['partner_id'] == df['prev_partner']
        # df['same_partner'] = df['same_partner'].map({True: 'yes', False: 'no', pd.NA: pd.NA})

        # Save bout-level breakdown (optional, comment out if not wanted)
        out_file = os.path.join(self.directory, "inter_bout_dynamics.csv")
        df.to_csv(out_file, index=False)



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
                body_coords = frame_df[['x_body', 'y_body']].to_numpy(float)
                D_body = cdist(body_coords, body_coords)
                np.fill_diagonal(D_body, np.nan)

                df.loc[
                    frame_df.index,
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




    ## FOR RASTA PLOTTING
    def head_head_interaction_type_over_time(self, proximity_threshold=1):

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
                min_type = None
                all_dists = []  # (dist, part1, part2, unified_type)
                for part1, part2 in interaction_pairs:
                    dist = np.linalg.norm(coords_a[part1] - coords_b[part2])
                    unified = unify_interaction_type(part1, part2)
                    all_dists.append((dist, part1, part2, unified))

                    if dist < min_dist:
                        min_dist = dist
                        min_type = unified
                
                touching = min_dist < proximity_threshold


                # ---- choose "relevant" interaction type with HH / HT priority ----
                relevant_type = None
                if touching:
                    best_special = None  # (dist, unified_type)

                    for dist, p1, p2, unified in all_dists:
                        if unified not in ('head_head', 'head_tail'):
                            continue
                        if dist < proximity_threshold:
                            if best_special is None or dist < best_special[0]:
                                best_special = (dist, unified)

                    if best_special is not None:
                        # prefer head_head / head_tail if present under threshold
                        relevant_type = best_special[1]
                    else:
                        # fallback: just use the global minimum type
                        relevant_type = min_type


                results.append({
                'file': track_file,
                'frame': frame,
                'Interaction Pair': tuple(sorted((track_a, track_b))),
                'touching': touching,
                'Distance': min_dist if touching else np.nan,
                'Closest Interaction Type': min_type if touching else None,
                'Relevant Interaction Type': relevant_type if touching else None,})

            return results


        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file].sort_values(by='frame')

            track_ids = df['track_id'].unique()
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
                'touching': False,
                'Distance': np.nan,
                'Closest Interaction Type': None,
                'Relevant Interaction Type': None,
            }]))


        closest_df = pd.concat(data, ignore_index=True)

        if self.shorten and self.shorten_duration is not None:
            suffix = f"_{self.shorten_duration}"
        else:
            suffix = ""

        filename = f"closest_contacts_{proximity_threshold}mm{suffix}_overtime.csv"
        closest_df.to_csv(os.path.join(self.directory, filename), index=False)

        return closest_df
    




    def head_approach_angle(self):

        def angle_calculator(vector_A, vector_B):
            # Same helper as in movement_direction
            A = np.array(vector_A, dtype=np.float64)
            B = np.array(vector_B, dtype=np.float64)

            if not np.isnan(A).any() and not np.isnan(B).any():
                mag_A = np.linalg.norm(A)
                mag_B = np.linalg.norm(B)

                if mag_A != 0 and mag_B != 0:
                    dot_product = np.dot(A, B)
                    cos_theta = dot_product / (mag_A * mag_B)
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    theta_radians = np.arccos(cos_theta)
                    theta_degrees = np.degrees(theta_radians)
                    return theta_degrees

            return np.nan
        

        body_parts = ['head', 'body', 'tail']
        dfs = []
        
        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file]
            df['file'] = track_file     # ← create column first
            df = df.sort_values(by=['file', 'frame'])

            # BODY-HEAD VECTOR
            df['body_head_dx'] = df['x_head'] - df['x_body']
            df['body_head_dy'] = df['y_head'] - df['y_body']
            
            # INITIALISE COLUMNS
            df['body_body_distance'] = np.nan
            df['other_id'] = np.nan
            df['closest_other_node'] = np.nan
            df['head_other_dx'] = np.nan
            df['head_other_dy'] = np.nan
            df['approach_angle'] = np.nan
            df['closest_other_node_distance'] = np.nan


            for frame, frame_group in df.groupby('frame'):
                if frame_group['track_id'].nunique() != 2:
                    continue

                rows = frame_group.sort_values('track_id')
                row1 = rows.iloc[0]
                row2 = rows.iloc[1]

                idx1 = row1.name
                idx2 = row2.name

                body1 = np.array([row1['x_body'], row1['y_body']], float)
                body2 = np.array([row2['x_body'], row2['y_body']], float)
                body_body_dist = np.linalg.norm(body2 - body1)

                df.at[idx1, 'body_body_distance'] = body_body_dist
                df.at[idx2, 'body_body_distance'] = body_body_dist

                # build node arrays for each larva: [head, body, tail]
                nodes1 = np.array([
                    [row1['x_head'], row1['y_head']],
                    [row1['x_body'], row1['y_body']],
                    [row1['x_tail'], row1['y_tail']],
                ], dtype=float)

                nodes2 = np.array([
                    [row2['x_head'], row2['y_head']],
                    [row2['x_body'], row2['y_body']],
                    [row2['x_tail'], row2['y_tail']],
                ], dtype=float)

                 # head positions
                head1 = nodes1[0]  # larva 1 head
                head2 = nodes2[0]  # larva 2 head

                # --- focal = larva 1, other = larva 2 ---
                diffs_1 = nodes2 - head1           # head1 -> each node of larva 2
                dists_1 = np.linalg.norm(diffs_1, axis=1)

                if not np.isnan(dists_1).all():
                    idx_min_1 = np.nanargmin(dists_1)  # 0=head, 1=body, 2=tail
                    df.at[idx1, 'other_id'] = row2['track_id']
                    df.at[idx1, 'closest_other_node'] = body_parts[idx_min_1]

                    closest_vec_1 = diffs_1[idx_min_1]          # head1 -> closest node on larva 2
                    v_body_head_1 = np.array(
                        [row1['body_head_dx'], row1['body_head_dy']], dtype=float
                    )
                    dist1 = np.linalg.norm(closest_vec_1)
                    df.at[idx1, 'closest_other_node_distance'] = dist1

                    if not (np.isnan(v_body_head_1).any() or np.linalg.norm(v_body_head_1) == 0):
                        angle_1 = angle_calculator(v_body_head_1, closest_vec_1)
                        df.at[idx1, 'head_other_dx'] = closest_vec_1[0]
                        df.at[idx1, 'head_other_dy'] = closest_vec_1[1]
                        df.at[idx1, 'approach_angle'] = angle_1


                # --- focal = larva 2, other = larva 1 ---
                diffs_2 = nodes1 - head2           # head2 -> each node of larva 1
                dists_2 = np.linalg.norm(diffs_2, axis=1)

                if not np.isnan(dists_2).all():
                    idx_min_2 = np.nanargmin(dists_2)
                    df.at[idx2, 'other_id'] = row1['track_id']
                    df.at[idx2, 'closest_other_node'] = body_parts[idx_min_2]

                    closest_vec_2 = diffs_2[idx_min_2]          # head2 -> closest node on larva 1
                    v_body_head_2 = np.array(
                        [row2['body_head_dx'], row2['body_head_dy']], dtype=float
                    )
                    dist2 = np.linalg.norm(closest_vec_2)
                    df.at[idx2, 'closest_other_node_distance'] = dist2

                    if not (np.isnan(v_body_head_2).any() or np.linalg.norm(v_body_head_2) == 0):
                        angle_2 = angle_calculator(v_body_head_2, closest_vec_2)
                        df.at[idx2, 'head_other_dx'] = closest_vec_2[0]
                        df.at[idx2, 'head_other_dy'] = closest_vec_2[1]
                        df.at[idx2, 'approach_angle'] = angle_2
            
            dfs.append(df)
        
        result_df = pd.concat(dfs, ignore_index=True)
        result_df.to_csv(os.path.join(self.directory, "head_head_approach_angles.csv"), index=False)

    



    def head_head_first_contact(self, proximity_threshold=1, window=10):

        data = []

        parts = ['head', 'body', 'tail']
        interaction_pairs = list(itertools.product(parts, parts))

        def heading_angle(body, head, tail):
            """
            Heading defined by tail->body and body->head vectors.
            Returns angle in degrees, 180 = forward.
            """
            v1 = np.array(body) - np.array(tail)   # tail -> body
            v2 = np.array(head) - np.array(body)   # body -> head

            if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
                return np.nan

            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_theta = np.clip(cos_theta, -1, 1)
            return np.degrees(np.arccos(cos_theta))
        
        def compute_speed_body(row_prev, row_curr):
            dx = row_curr['x_body'] - row_prev['x_body']
            dy = row_curr['y_body'] - row_prev['y_body']
            return np.sqrt(dx*dx + dy*dy)

        def compute_speed_tail(row_prev, row_curr):
            dx = row_curr['x_tail'] - row_prev['x_tail']
            dy = row_curr['y_tail'] - row_prev['y_tail']
            return np.sqrt(dx*dx + dy*dy)

        def compute_speed_head(row_prev, row_curr):
            dx = row_curr['x_head'] - row_prev['x_head']
            dy = row_curr['y_head'] - row_prev['y_head']
            return np.sqrt(dx*dx + dy*dy)

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file].sort_values('frame')

            track_ids = sorted(df['track_id'].unique())
            if len(track_ids) != 2:
                continue

            track_a, track_b = track_ids

            df_a = df[df['track_id'] == track_a]
            df_b = df[df['track_id'] == track_b]

            common_frames = sorted(set(df_a['frame']).intersection(df_b['frame']))
            if not common_frames:
                continue

            hh_frame = None

            # ---- find FIRST head–head contact ----
            for frame in common_frames:
                row_a = df_a[df_a['frame'] == frame]
                row_b = df_b[df_b['frame'] == frame]

                if row_a.empty or row_b.empty:
                    continue

                coords_a = {p: row_a[[f'x_{p}', f'y_{p}']].to_numpy().flatten() for p in parts}
                coords_b = {p: row_b[[f'x_{p}', f'y_{p}']].to_numpy().flatten() for p in parts}

                min_dist = np.inf
                min_pair = None

                for p1, p2 in interaction_pairs:
                    dist = np.linalg.norm(coords_a[p1] - coords_b[p2])
                    if dist < min_dist:
                        min_dist = dist
                        min_pair = (p1, p2)

                if min_dist < proximity_threshold and set(min_pair) == {'head'}:
                    hh_frame = frame
                    break

            if hh_frame is None:
                continue  # no head–head contact in this file

            start = hh_frame
            end = hh_frame + window
            interaction_min_dist = np.inf


            # ---- extract kinematics after first contact ----
            for frame in range(start, end + 1):
                frame_rows = df[df['frame'] == frame]
                if frame_rows['track_id'].nunique() != 2:
                    continue

                # ---- compute min node-node distance for THIS frame ----
                rows = frame_rows.sort_values('track_id')
                row_a = rows.iloc[0]
                row_b = rows.iloc[1]

                coords_a = {p: np.array([row_a[f'x_{p}'], row_a[f'y_{p}']]) for p in parts}
                coords_b = {p: np.array([row_b[f'x_{p}'], row_b[f'y_{p}']]) for p in parts}

                frame_min_dist = np.inf
                for p1, p2 in interaction_pairs:
                    dist = np.linalg.norm(coords_a[p1] - coords_b[p2])
                    if dist < frame_min_dist:
                        frame_min_dist = dist

               


                for _, row in frame_rows.iterrows():
                    body = (row['x_body'], row['y_body'])
                    head = (row['x_head'], row['y_head'])
                    tail = (row['x_tail'], row['y_tail'])

                    angle = heading_angle(body, head, tail)

                    prev = df[
                        (df['track_id'] == row['track_id']) &
                        (df['frame'] == frame - 1)
                    ]

                    if not prev.empty:
                        speed_body = compute_speed_body(prev.iloc[0], row)
                        speed_head = compute_speed_head(prev.iloc[0], row)
                        speed_tail = compute_speed_tail(prev.iloc[0], row)
                    else:
                        speed_body = np.nan
                        speed_head = np.nan
                        speed_tail = np.nan


                    data.append({
                        'file': track_file,
                        'frame': frame,
                        'rel_frame': frame - hh_frame,
                        'track_id': row['track_id'],
                        'speed_body': speed_body,
                        'speed_head': speed_head,
                        'speed_tail': speed_tail,
                        'heading_angle': angle,
                        'min_distance': frame_min_dist 
                    })

        result_df = pd.DataFrame(data)
        result_df.to_csv(
            os.path.join(self.directory, 'head_head_first_contact_kinematics.csv'),
            index=False
        )

        return result_df







    def head_head_contacts_kinematics_over_time(self, proximity_threshold=1, window=10):

        data = []

        parts = ['head', 'body', 'tail']
        interaction_pairs = list(itertools.product(parts, parts))

        def heading_angle(body, head, tail):
            v1 = np.array(body) - np.array(tail)
            v2 = np.array(head) - np.array(body)

            if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
                return np.nan

            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_theta = np.clip(cos_theta, -1, 1)
            return np.degrees(np.arccos(cos_theta))

        def compute_speed_body(row_prev, row_curr):
            dx = row_curr['x_body'] - row_prev['x_body']
            dy = row_curr['y_body'] - row_prev['y_body']
            return np.sqrt(dx*dx + dy*dy)

        def compute_speed_tail(row_prev, row_curr):
            dx = row_curr['x_tail'] - row_prev['x_tail']
            dy = row_curr['y_tail'] - row_prev['y_tail']
            return np.sqrt(dx*dx + dy*dy)

        def compute_speed_head(row_prev, row_curr):
            dx = row_curr['x_head'] - row_prev['x_head']
            dy = row_curr['y_head'] - row_prev['y_head']
            return np.sqrt(dx*dx + dy*dy)

        def compute_heading_angle(row):
            body = (row['x_body'], row['y_body'])
            head = (row['x_head'], row['y_head'])
            tail = (row['x_tail'], row['y_tail'])
            return heading_angle(body, head, tail)

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file].sort_values('frame')

            track_ids = sorted(df['track_id'].unique())
            if len(track_ids) != 2:
                continue

            df_a = df[df['track_id'] == track_ids[0]]
            df_b = df[df['track_id'] == track_ids[1]]

            common_frames = sorted(set(df_a['frame']).intersection(df_b['frame']))
            if not common_frames:
                continue

            interaction_number = 0
            next_allowed_frame = -np.inf

            # ---- scan frames sequentially ----
            for frame in common_frames:

                if frame < next_allowed_frame:
                    continue

                row_a = df_a[df_a['frame'] == frame]
                row_b = df_b[df_b['frame'] == frame]
                if row_a.empty or row_b.empty:
                    continue

                coords_a = {p: row_a[[f'x_{p}', f'y_{p}']].to_numpy().flatten() for p in parts}
                coords_b = {p: row_b[[f'x_{p}', f'y_{p}']].to_numpy().flatten() for p in parts}

                min_dist = np.inf
                min_pair = None
                for p1, p2 in interaction_pairs:
                    dist = np.linalg.norm(coords_a[p1] - coords_b[p2])
                    if dist < min_dist:
                        min_dist = dist
                        min_pair = (p1, p2)

                # ---- detect new head–head interaction ----
                if min_dist < proximity_threshold and set(min_pair) == {'head'}:
                    interaction_number += 1
                    interaction_frame = frame
                    next_allowed_frame = frame + window + 1


                    # pull initial body xy DIRECTLY
                    row_a0 = df_a[df_a["frame"] == interaction_frame].iloc[0]
                    row_b0 = df_b[df_b["frame"] == interaction_frame].iloc[0]

                    x0_a, y0_a = row_a0["x_body"], row_a0["y_body"]
                    x0_b, y0_b = row_b0["x_body"], row_b0["y_body"]

                    start_pos = {
                        track_ids[0]: (x0_a, y0_a),
                        track_ids[1]: (x0_b, y0_b),
                    }
     
                    # ---- extract kinematics window before and after contact ----
                    for f in range(interaction_frame - window, interaction_frame + window + 1):
                        frame_rows = df[df['frame'] == f]
                        if frame_rows['track_id'].nunique() != 2:
                            continue

                        rows = frame_rows.sort_values('track_id')
                        row_a = rows.iloc[0]
                        row_b = rows.iloc[1]

                        coords_a = {p: np.array([row_a[f'x_{p}'], row_a[f'y_{p}']]) for p in parts}
                        coords_b = {p: np.array([row_b[f'x_{p}'], row_b[f'y_{p}']]) for p in parts}


                        frame_min_dist = np.inf
                        for p1, p2 in interaction_pairs:
                            dist = np.linalg.norm(coords_a[p1] - coords_b[p2])
                            if dist < frame_min_dist:
                                frame_min_dist = dist

                        for _, row in frame_rows.iterrows():
                            angle = compute_heading_angle(row)

                            prev = df[
                                (df['track_id'] == row['track_id']) &
                                (df['frame'] == f - 1)
                            ]

                            speed_body = (
                                compute_speed_body(prev.iloc[0], row)
                                if not prev.empty else np.nan
                            )

                            speed_head = (
                                compute_speed_head(prev.iloc[0], row)
                                if not prev.empty else np.nan
                            )

                            speed_tail = (
                                compute_speed_tail(prev.iloc[0], row)
                                if not prev.empty else np.nan
                            )

                            previous_angle = (
                                compute_heading_angle(prev.iloc[0])
                                if not prev.empty else np.nan
                            )

                            heading_angle_change = (
                                abs(angle - previous_angle)
                                if not np.isnan(angle) and not np.isnan(previous_angle)
                                else np.nan
                            )

                            x0, y0 = start_pos[row['track_id']]
                            dist_from_start = np.sqrt(
                                (row['x_body'] - x0)**2 +
                                (row['y_body'] - y0)**2
                            )

                            data.append({
                                'file': track_file,
                                'interaction_number': interaction_number,
                                'frame': f,
                                'rel_frame': f - interaction_frame,
                                'rel_time_seconds': f - interaction_frame,
                                'track_id': row['track_id'],
                                'speed_body': speed_body,
                                'speed_head': speed_head,
                                'speed_tail': speed_tail,
                                'heading_angle': angle,
                                'heading_angle_change': heading_angle_change,
                                'min_distance': frame_min_dist,
                                'dist_from_start': dist_from_start,
                                # coordinates for trajectory plotting
                                'x_head': row['x_head'],
                                'y_head': row['y_head'],
                                'x_body': row['x_body'],
                                'y_body': row['y_body'],
                                'x_tail': row['x_tail'],
                                'y_tail': row['y_tail'],
                            })

        result_df = pd.DataFrame(data)
        result_df.to_csv(
            os.path.join(self.directory, 'head_head_contacts_kinematics_over_time.csv'),
            index=False
        )

        return result_df




    def head_head_contacts_kinematics_over_time_nocontacts(self, proximity_threshold=1, window=10):

        data = []

        parts = ['head', 'body', 'tail']
        interaction_pairs = list(itertools.product(parts, parts))

        def heading_angle(body, head, tail):
            v1 = np.array(body) - np.array(tail)
            v2 = np.array(head) - np.array(body)

            if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
                return np.nan

            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_theta = np.clip(cos_theta, -1, 1)
            return np.degrees(np.arccos(cos_theta))

        def compute_speed_body(row_prev, row_curr):
            dx = row_curr['x_body'] - row_prev['x_body']
            dy = row_curr['y_body'] - row_prev['y_body']
            return np.sqrt(dx*dx + dy*dy)

        def compute_speed_tail(row_prev, row_curr):
            dx = row_curr['x_tail'] - row_prev['x_tail']
            dy = row_curr['y_tail'] - row_prev['y_tail']
            return np.sqrt(dx*dx + dy*dy)

        def compute_speed_head(row_prev, row_curr):
            dx = row_curr['x_head'] - row_prev['x_head']
            dy = row_curr['y_head'] - row_prev['y_head']
            return np.sqrt(dx*dx + dy*dy)

        def compute_heading_angle(row):
            body = (row['x_body'], row['y_body'])
            head = (row['x_head'], row['y_head'])
            tail = (row['x_tail'], row['y_tail'])
            return heading_angle(body, head, tail)

        def compute_approach_angle(row_focal, row_other):
            body_head_vec = np.array([
                row_focal['x_head'] - row_focal['x_body'],
                row_focal['y_head'] - row_focal['y_body']
            ], dtype=float)

            focal_head = np.array([row_focal['x_head'], row_focal['y_head']], dtype=float)
            other_nodes = np.array([
                [row_other['x_head'], row_other['y_head']],
                [row_other['x_body'], row_other['y_body']],
                [row_other['x_tail'], row_other['y_tail']]
            ], dtype=float)

            head_to_other_nodes = other_nodes - focal_head
            distances = np.linalg.norm(head_to_other_nodes, axis=1)

            if np.isnan(distances).all():
                return np.nan

            closest_vec = head_to_other_nodes[np.nanargmin(distances)]

            if np.isnan(body_head_vec).any() or np.isnan(closest_vec).any():
                return np.nan

            body_head_mag = np.linalg.norm(body_head_vec)
            closest_mag = np.linalg.norm(closest_vec)

            if body_head_mag == 0 or closest_mag == 0:
                return np.nan

            cos_theta = np.dot(body_head_vec, closest_vec) / (body_head_mag * closest_mag)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            return np.degrees(np.arccos(cos_theta))

        def compute_approach_angle_head(row_focal, row_other):
            body_head_vec = np.array([
                row_focal['x_head'] - row_focal['x_body'],
                row_focal['y_head'] - row_focal['y_body']
            ], dtype=float)

            head_to_other_head_vec = np.array([
                row_other['x_head'] - row_focal['x_head'],
                row_other['y_head'] - row_focal['y_head']
            ], dtype=float)

            if np.isnan(body_head_vec).any() or np.isnan(head_to_other_head_vec).any():
                return np.nan

            body_head_mag = np.linalg.norm(body_head_vec)
            head_to_other_head_mag = np.linalg.norm(head_to_other_head_vec)

            if body_head_mag == 0 or head_to_other_head_mag == 0:
                return np.nan

            cos_theta = np.dot(body_head_vec, head_to_other_head_vec) / (
                body_head_mag * head_to_other_head_mag
            )
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            return np.degrees(np.arccos(cos_theta))

        def compute_min_distance(row_a, row_b):
            coords_a = {p: np.array([row_a[f'x_{p}'], row_a[f'y_{p}']]) for p in parts}
            coords_b = {p: np.array([row_b[f'x_{p}'], row_b[f'y_{p}']]) for p in parts}

            min_dist = np.inf
            min_pair = None

            for p1, p2 in interaction_pairs:
                dist = np.linalg.norm(coords_a[p1] - coords_b[p2])
                if dist < min_dist:
                    min_dist = dist
                    min_pair = (p1, p2)

            return min_dist, min_pair

        def first_no_contact_frame(df_a, df_b, common_frames, interaction_frame):
            for f in common_frames:
                if f <= interaction_frame:
                    continue

                row_a = df_a[df_a['frame'] == f]
                row_b = df_b[df_b['frame'] == f]
                if row_a.empty or row_b.empty:
                    continue

                frame_min_dist, _ = compute_min_distance(row_a.iloc[0], row_b.iloc[0])

                # after-window starts only once every node-node distance is > threshold
                if frame_min_dist > proximity_threshold:
                    return f

            return None

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file].sort_values('frame')

            track_ids = sorted(df['track_id'].unique())
            if len(track_ids) != 2:
                continue

            df_a = df[df['track_id'] == track_ids[0]]
            df_b = df[df['track_id'] == track_ids[1]]

            common_frames = sorted(set(df_a['frame']).intersection(df_b['frame']))
            if not common_frames:
                continue

            interaction_number = 0
            next_allowed_frame = -np.inf

            # ---- scan frames sequentially ----
            for frame in common_frames:

                if frame < next_allowed_frame:
                    continue

                row_a = df_a[df_a['frame'] == frame]
                row_b = df_b[df_b['frame'] == frame]
                if row_a.empty or row_b.empty:
                    continue

                min_dist, min_pair = compute_min_distance(row_a.iloc[0], row_b.iloc[0])

                # ---- detect new head-head interaction ----
                if min_dist < proximity_threshold and set(min_pair) == {'head'}:
                    interaction_frame = frame
                    no_contact_frame = first_no_contact_frame(
                        df_a,
                        df_b,
                        common_frames,
                        interaction_frame
                    )

                    if no_contact_frame is None:
                        continue

                    interaction_number += 1
                    next_allowed_frame = no_contact_frame + window + 1

                    # pull initial head xy from the first no-contact frame
                    row_a0 = df_a[df_a["frame"] == no_contact_frame].iloc[0]
                    row_b0 = df_b[df_b["frame"] == no_contact_frame].iloc[0]

                    x0_a, y0_a = row_a0["x_head"], row_a0["y_head"]
                    x0_b, y0_b = row_b0["x_head"], row_b0["y_head"]

                    start_pos = {
                        track_ids[0]: (x0_a, y0_a),
                        track_ids[1]: (x0_b, y0_b),
                    }

                    before_frames = range(interaction_frame - window, interaction_frame)
                    after_frames = range(no_contact_frame, no_contact_frame + window + 1)
                    frames_to_extract = list(before_frames) + list(after_frames)

                    # ---- extract before contact and after the first no-contact frame ----
                    for f in frames_to_extract:
                        frame_rows = df[df['frame'] == f]
                        if frame_rows['track_id'].nunique() != 2:
                            continue

                        rows = frame_rows.sort_values('track_id')
                        row_a = rows.iloc[0]
                        row_b = rows.iloc[1]

                        frame_min_dist, _ = compute_min_distance(row_a, row_b)
                        head_distance = np.sqrt(
                            (row_a['x_head'] - row_b['x_head'])**2 +
                            (row_a['y_head'] - row_b['y_head'])**2
                        )

                        if f < interaction_frame:
                            rel_frame = f - interaction_frame
                        else:
                            rel_frame = f - no_contact_frame

                        for _, row in frame_rows.iterrows():
                            angle = compute_heading_angle(row)
                            other_row = row_b if row['track_id'] == row_a['track_id'] else row_a
                            approach_angle = compute_approach_angle(row, other_row)
                            approach_angle_head = compute_approach_angle_head(row, other_row)

                            prev = df[
                                (df['track_id'] == row['track_id']) &
                                (df['frame'] == f - 1)
                            ]

                            speed_body = (
                                compute_speed_body(prev.iloc[0], row)
                                if not prev.empty else np.nan
                            )

                            speed_head = (
                                compute_speed_head(prev.iloc[0], row)
                                if not prev.empty else np.nan
                            )

                            speed_tail = (
                                compute_speed_tail(prev.iloc[0], row)
                                if not prev.empty else np.nan
                            )

                            previous_angle = (
                                compute_heading_angle(prev.iloc[0])
                                if not prev.empty else np.nan
                            )

                            heading_angle_change = (
                                abs(angle - previous_angle)
                                if not np.isnan(angle) and not np.isnan(previous_angle)
                                else np.nan
                            )

                            x0, y0 = start_pos[row['track_id']]
                            dist_from_start = np.sqrt(
                                (row['x_head'] - x0)**2 +
                                (row['y_head'] - y0)**2
                            )

                            data.append({
                                'file': track_file,
                                'interaction_number': interaction_number,
                                'interaction_frame': interaction_frame,
                                'no_contact_frame': no_contact_frame,
                                'frame': f,
                                'rel_frame': rel_frame,
                                'rel_time_seconds': rel_frame,
                                'track_id': row['track_id'],
                                'speed_body': speed_body,
                                'speed_head': speed_head,
                                'speed_tail': speed_tail,
                                'heading_angle': angle,
                                'heading_angle_change': heading_angle_change,
                                'approach_angle': approach_angle,
                                'approach_angle_head': approach_angle_head,
                                'min_distance': frame_min_dist,
                                'head_distance': head_distance,
                                'dist_from_start': dist_from_start,
                                # coordinates for trajectory plotting
                                'x_head': row['x_head'],
                                'y_head': row['y_head'],
                                'x_body': row['x_body'],
                                'y_body': row['y_body'],
                                'x_tail': row['x_tail'],
                                'y_tail': row['y_tail'],
                            })

        result_df = pd.DataFrame(data)

        if not result_df.empty:
            group_cols = ['file', 'interaction_number', 'track_id']
            result_df = result_df.sort_values(group_cols + ['rel_frame']).copy()

            result_df['directional_alignment_cosine'] = np.nan
            result_df['track0_projection_onto_track1'] = np.nan

            for _, event in result_df.groupby(['file', 'interaction_number']):
                track_ids = sorted(event['track_id'].dropna().unique())
                if len(track_ids) != 2:
                    continue

                track0, track1 = track_ids
                track0_steps = (
                    event[event['track_id'] == track0]
                    .sort_values('rel_frame')
                    .set_index('rel_frame')[['x_body', 'y_body']]
                    .diff()
                )
                track1_steps = (
                    event[event['track_id'] == track1]
                    .sort_values('rel_frame')
                    .set_index('rel_frame')[['x_body', 'y_body']]
                    .diff()
                )

                common_rel_frames = track0_steps.index.intersection(track1_steps.index)
                for rel_frame in common_rel_frames:
                    track0_vec = track0_steps.loc[rel_frame].to_numpy(dtype=float)
                    track1_vec = track1_steps.loc[rel_frame].to_numpy(dtype=float)

                    if not (
                        np.isfinite(track0_vec).all() and
                        np.isfinite(track1_vec).all()
                    ):
                        continue

                    track0_norm = np.linalg.norm(track0_vec)
                    track1_norm = np.linalg.norm(track1_vec)
                    if track0_norm == 0 or track1_norm == 0:
                        continue

                    directional_alignment = (
                        np.dot(track0_vec, track1_vec) /
                        (track0_norm * track1_norm)
                    )
                    track1_unit_vec = track1_vec / track1_norm
                    track0_projection = np.dot(track0_vec, track1_unit_vec)

                    row_mask = (
                        (result_df['file'] == event['file'].iloc[0]) &
                        (result_df['interaction_number'] == event['interaction_number'].iloc[0]) &
                        (result_df['rel_frame'] == rel_frame)
                    )
                    result_df.loc[row_mask, 'directional_alignment_cosine'] = directional_alignment
                    result_df.loc[row_mask, 'track0_projection_onto_track1'] = track0_projection

            dx_head = result_df.groupby(group_cols)['x_head'].diff()
            dy_head = result_df.groupby(group_cols)['y_head'].diff()
            result_df['step_distance_head'] = np.sqrt(dx_head**2 + dy_head**2)
            result_df['step_distance_head'] = result_df['step_distance_head'].fillna(0)
            result_df['cum_distance_start'] = result_df.groupby(group_cols)['step_distance_head'].cumsum()

            result_df['cum_distance_0'] = np.nan
            result_df['distance_from_0'] = np.nan
            for _, idx in result_df.groupby(group_cols).groups.items():
                group = result_df.loc[idx].sort_values('rel_frame')

                if not (group['rel_frame'] == 0).any():
                    continue

                zero_idx = group.index[group['rel_frame'] == 0][0]
                zero_x = result_df.at[zero_idx, 'x_head']
                zero_y = result_df.at[zero_idx, 'y_head']

                result_df.at[zero_idx, 'cum_distance_0'] = 0
                result_df.loc[group.index, 'distance_from_0'] = np.sqrt(
                    (group['x_head'] - zero_x)**2 +
                    (group['y_head'] - zero_y)**2
                )

                after = group[group['rel_frame'] >= 0]
                after_steps = after['step_distance_head'].copy()
                after_steps.iloc[0] = 0
                result_df.loc[after.index, 'cum_distance_0'] = after_steps.cumsum()

                before = group[group['rel_frame'] <= 0].sort_values('rel_frame', ascending=False)
                backward_steps = np.sqrt(
                    before['x_head'].diff()**2 + before['y_head'].diff()**2
                ).fillna(0)
                result_df.loc[before.index, 'cum_distance_0'] = backward_steps.cumsum()

            result_df = result_df.drop(columns=['step_distance_head'])

        result_df.to_csv(
            os.path.join(self.directory, f'head_head_contacts_kinematics_over_time_nocontacts_ref-window{window}.csv'),
            index=False
        )

        return result_df






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
    





    def prob_contact(
        self,
        contact_threshold=1.0,
        max_distance=10,
        distance_bin_size=1,
        angle_bin_size=30
    ):

        parts = ['head', 'body', 'tail']

        raw_columns = [
            'file',
            'episode_id',
            'episode_start_frame',
            'episode_end_frame',
            'frame',
            'focal_id',
            'stim_id',
            'nearest_node',
            'head_distance',
            'distance_bin',
            'approach_angle',
            'angle_bin',
            'contact_during_episode',
            'contact_within_window'
        ]

        summary_columns = [
            'file',
            'distance_bin',
            'angle_bin',
            'nearest_node',
            'n_observations',
            'n_contacts',
            'prob_contact'
        ]

        def head_to_other_distance(row_focal, row_stim):

            hx = row_focal['x_head']
            hy = row_focal['y_head']

            if pd.isna(hx) or pd.isna(hy):
                return np.nan, None

            min_dist = np.inf
            nearest_node = None

            for part in parts:
                x = row_stim[f'x_{part}']
                y = row_stim[f'y_{part}']

                if pd.isna(x) or pd.isna(y):
                    continue

                dist = np.hypot(hx - x, hy - y)

                if dist < min_dist:
                    min_dist = dist
                    nearest_node = part

            if nearest_node is None:
                return np.nan, None

            return float(min_dist), nearest_node

        def approach_angle(row_focal, row_stim, nearest_node):

            hx = row_focal['x_head']
            hy = row_focal['y_head']
            bx = row_focal['x_body']
            by = row_focal['y_body']

            heading = np.array([hx - bx, hy - by], dtype=float)

            if np.isnan(heading).any() or np.linalg.norm(heading) == 0:
                return np.nan

            tx = row_stim[f'x_{nearest_node}']
            ty = row_stim[f'y_{nearest_node}']

            if pd.isna(tx) or pd.isna(ty):
                return np.nan

            target = np.array([tx - hx, ty - hy], dtype=float)

            if np.isnan(target).any() or np.linalg.norm(target) == 0:
                return np.nan

            cosang = np.dot(heading, target) / (
                np.linalg.norm(heading) * np.linalg.norm(target)
            )
            cosang = np.clip(cosang, -1, 1)

            return float(np.degrees(np.arccos(cosang)))

        def min_node_distance(row_a, row_b):

            min_dist = np.inf

            for part_a in parts:
                coord_a = np.array([
                    row_a[f'x_{part_a}'],
                    row_a[f'y_{part_a}']
                ], dtype=float)

                if np.isnan(coord_a).any():
                    continue

                for part_b in parts:
                    coord_b = np.array([
                        row_b[f'x_{part_b}'],
                        row_b[f'y_{part_b}']
                    ], dtype=float)

                    if np.isnan(coord_b).any():
                        continue

                    dist = np.linalg.norm(coord_a - coord_b)

                    if dist < min_dist:
                        min_dist = dist

            return min_dist

        distance_edges = np.arange(
            0,
            max_distance + distance_bin_size,
            distance_bin_size
        )
        angle_edges = np.arange(
            0,
            180 + angle_bin_size,
            angle_bin_size
        )

        distance_labels = [
            f'{distance_edges[i]:g}-{distance_edges[i + 1]:g}'
            for i in range(len(distance_edges) - 1)
        ]
        angle_labels = [
            f'{angle_edges[i]:g}-{angle_edges[i + 1]:g}'
            for i in range(len(angle_edges) - 1)
        ]

        def label_bin(value, edges, labels):

            if pd.isna(value):
                return np.nan

            for i in range(len(labels)):
                if i == 0 and edges[i] <= value <= edges[i + 1]:
                    return labels[i]
                if edges[i] < value <= edges[i + 1]:
                    return labels[i]

            return np.nan

        data = []

        for match in self.matching_pairs:

            track_file = match['track_file']
            df = self.track_data[track_file].copy()
            df = df.sort_values(['frame', 'track_id'])

            frame_groups = {
                frame: frame_df
                for frame, frame_df in df.groupby('frame')
            }

            frames = sorted(frame_groups.keys())
            all_track_ids = sorted(df['track_id'].dropna().unique())
            episode_counter = 0

            for id1, id2 in combinations(all_track_ids, 2):

                i = 0

                while i < len(frames):

                    frame = frames[i]
                    frame_df = frame_groups[frame]
                    row1 = frame_df[frame_df['track_id'] == id1]
                    row2 = frame_df[frame_df['track_id'] == id2]

                    if row1.empty or row2.empty:
                        i += 1
                        continue

                    row1 = row1.iloc[0]
                    row2 = row2.iloc[0]
                    pair_distance = min_node_distance(row1, row2)

                    if (
                        not np.isfinite(pair_distance)
                        or pair_distance > max_distance
                    ):
                        i += 1
                        continue

                    episode_counter += 1
                    episode_id = f'{track_file}_episode_{episode_counter}'
                    episode_start_frame = frame
                    episode_end_frame = frame
                    episode_rows = []
                    seen_distance_bins = {
                        id1: set(),
                        id2: set()
                    }
                    contacted = False

                    while i < len(frames):

                        frame = frames[i]
                        frame_df = frame_groups[frame]
                        row1 = frame_df[frame_df['track_id'] == id1]
                        row2 = frame_df[frame_df['track_id'] == id2]

                        if row1.empty or row2.empty:
                            break

                        row1 = row1.iloc[0]
                        row2 = row2.iloc[0]
                        pair_distance = min_node_distance(row1, row2)

                        if (
                            not np.isfinite(pair_distance)
                            or pair_distance > max_distance
                        ):
                            break

                        episode_end_frame = frame

                        if pair_distance < contact_threshold:
                            contacted = True
                            i += 1
                            break

                        for focal_id, stim_id, focal, stim in [
                            (id1, id2, row1, row2),
                            (id2, id1, row2, row1)
                        ]:
                            head_distance, nearest_node = head_to_other_distance(
                                focal,
                                stim
                            )

                            if (
                                nearest_node is None
                                or pd.isna(head_distance)
                                or head_distance <= 0
                                or head_distance > max_distance
                            ):
                                continue

                            distance_bin = label_bin(
                                head_distance,
                                distance_edges,
                                distance_labels
                            )

                            if pd.isna(distance_bin):
                                continue

                            if distance_bin in seen_distance_bins[focal_id]:
                                continue

                            angle = approach_angle(focal, stim, nearest_node)

                            if pd.isna(angle):
                                continue

                            angle_bin = label_bin(
                                angle,
                                angle_edges,
                                angle_labels
                            )

                            if pd.isna(angle_bin):
                                continue

                            seen_distance_bins[focal_id].add(distance_bin)
                            episode_rows.append({
                                'file': track_file,
                                'episode_id': episode_id,
                                'episode_start_frame': episode_start_frame,
                                'episode_end_frame': episode_end_frame,
                                'frame': frame,
                                'focal_id': focal_id,
                                'stim_id': stim_id,
                                'nearest_node': nearest_node,
                                'head_distance': head_distance,
                                'distance_bin': distance_bin,
                                'approach_angle': angle,
                                'angle_bin': angle_bin,
                            })

                        i += 1

                    for row in episode_rows:
                        row['episode_end_frame'] = episode_end_frame
                        row['contact_during_episode'] = contacted
                        row['contact_within_window'] = contacted
                        data.append(row)

                    if contacted:
                        while i < len(frames):
                            frame_df = frame_groups[frames[i]]
                            row1 = frame_df[frame_df['track_id'] == id1]
                            row2 = frame_df[frame_df['track_id'] == id2]

                            if row1.empty or row2.empty:
                                break

                            pair_distance = min_node_distance(
                                row1.iloc[0],
                                row2.iloc[0]
                            )

                            if (
                                not np.isfinite(pair_distance)
                                or pair_distance > max_distance
                            ):
                                break

                            i += 1

                    elif i < len(frames):
                        i += 1

        raw_df = pd.DataFrame(data, columns=raw_columns)

        if raw_df.empty:
            summary_df = pd.DataFrame(columns=summary_columns)
        else:
            summary_df = (
                raw_df
                .dropna(subset=['distance_bin', 'angle_bin'])
                .groupby(
                    ['file', 'distance_bin', 'angle_bin', 'nearest_node'],
                    observed=True
                )
                .agg(
                    n_observations=('contact_during_episode', 'size'),
                    n_contacts=('contact_during_episode', 'sum')
                )
                .reset_index()
            )
            summary_df['prob_contact'] = (
                summary_df['n_contacts']
                / summary_df['n_observations']
            )
            summary_df = summary_df[summary_columns]

        raw_df.to_csv(
            os.path.join(self.directory, 'prob_contact_raw.csv'),
            index=False
        )
        summary_df.to_csv(
            os.path.join(self.directory, 'prob_contact_summary.csv'),
            index=False
        )

        return raw_df, summary_df





    def pairwise_approach_probability(self, threshold=10):

        parts = ['head', 'body', 'tail']

        data = []

        def min_node_distance(row_a, row_b):

            min_dist = np.inf

            for part_a in parts:
                coord_a = np.array([
                    row_a[f'x_{part_a}'],
                    row_a[f'y_{part_a}']
                ])

                for part_b in parts:
                    coord_b = np.array([
                        row_b[f'x_{part_b}'],
                        row_b[f'y_{part_b}']
                    ])

                    dist = np.linalg.norm(coord_a - coord_b)

                    if dist < min_dist:
                        min_dist = dist

            return min_dist

        for match in self.matching_pairs:

            track_file = match['track_file']
            df = self.track_data[track_file].copy()

            frame_groups = {
                frame: frame_df
                for frame, frame_df in df.groupby('frame')
            }

            frames = sorted(frame_groups.keys())

            for i in range(len(frames) - 1):

                frame = frames[i]
                next_frame = frames[i + 1]

                current_df = frame_groups[frame]
                next_df = frame_groups[next_frame]

                current_tracks = set(current_df['track_id'])
                next_tracks = set(next_df['track_id'])

                common_tracks = current_tracks.intersection(next_tracks)

                if len(common_tracks) < 2:
                    continue

                for id1, id2 in combinations(sorted(common_tracks), 2):

                    row1_now = current_df[current_df['track_id'] == id1]
                    row2_now = current_df[current_df['track_id'] == id2]

                    row1_next = next_df[next_df['track_id'] == id1]
                    row2_next = next_df[next_df['track_id'] == id2]

                    if (
                        row1_now.empty or row2_now.empty
                        or row1_next.empty or row2_next.empty
                    ):
                        continue

                    row1_now = row1_now.iloc[0]
                    row2_now = row2_now.iloc[0]

                    row1_next = row1_next.iloc[0]
                    row2_next = row2_next.iloc[0]

                    distance_now = min_node_distance(row1_now, row2_now)

                    if distance_now <= 0 or distance_now > threshold:
                        continue

                    distance_next = min_node_distance(row1_next, row2_next)

                    delta_distance = distance_next - distance_now

                    data.append({
                        'file': track_file,
                        'frame': frame,
                        'track_1': id1,
                        'track_2': id2,
                        'distance': distance_now,
                        'next_distance': distance_next,
                        'delta_distance': delta_distance,
                        'approach': delta_distance < 0
                    })

        result = pd.DataFrame(data)

        result.to_csv(
            os.path.join(
                self.directory,
                f'pairwise_approach_probability.csv'
            ),
            index=False
        )

        return result
    


    def individual_approach_probability(self, threshold=10):

        parts = ['head', 'body', 'tail']

        data = []

        def head_to_other_distance(row_focal, row_stim):

            hx = row_focal['x_head']
            hy = row_focal['y_head']

            min_dist = np.inf
            nearest_node = None

            for part in parts:
                x = row_stim[f'x_{part}']
                y = row_stim[f'y_{part}']

                if pd.isna(x) or pd.isna(y):
                    continue

                dist = np.hypot(hx - x, hy - y)

                if dist < min_dist:
                    min_dist = dist
                    nearest_node = part

            return min_dist, nearest_node

        def approach_angle(row_focal, row_stim, nearest_node):

            hx = row_focal['x_head']
            hy = row_focal['y_head']
            bx = row_focal['x_body']
            by = row_focal['y_body']

            v_heading = np.array([hx - bx, hy - by], dtype=float)

            if np.linalg.norm(v_heading) == 0:
                return np.nan

            tx = row_stim[f'x_{nearest_node}']
            ty = row_stim[f'y_{nearest_node}']

            if pd.isna(tx) or pd.isna(ty):
                return np.nan

            v_target = np.array([tx - hx, ty - hy], dtype=float)

            if np.linalg.norm(v_target) == 0:
                return np.nan

            cosang = np.dot(v_heading, v_target) / (
                np.linalg.norm(v_heading) * np.linalg.norm(v_target)
            )

            cosang = np.clip(cosang, -1, 1)

            return float(np.degrees(np.arccos(cosang)))

        def body_speed(row_now, row_next):

            return float(np.hypot(
                row_next['x_body'] - row_now['x_body'],
                row_next['y_body'] - row_now['y_body']
            ))

        for match in self.matching_pairs:

            track_file = match['track_file']
            df = self.track_data[track_file].copy()

            frame_groups = {
                frame: frame_df
                for frame, frame_df in df.groupby('frame')
            }

            frames = sorted(frame_groups.keys())

            for i in range(len(frames) - 1):

                frame = frames[i]
                next_frame = frames[i + 1]

                current_df = frame_groups[frame]
                next_df = frame_groups[next_frame]

                current_tracks = set(current_df['track_id'])
                next_tracks = set(next_df['track_id'])

                common_tracks = current_tracks.intersection(next_tracks)

                if len(common_tracks) < 2:
                    continue

                for focal_id in sorted(common_tracks):
                    for stim_id in sorted(common_tracks):

                        if focal_id == stim_id:
                            continue

                        focal_now = current_df[current_df['track_id'] == focal_id]
                        stim_now = current_df[current_df['track_id'] == stim_id]

                        focal_next = next_df[next_df['track_id'] == focal_id]
                        stim_next = next_df[next_df['track_id'] == stim_id]

                        if (
                            focal_now.empty or stim_now.empty
                            or focal_next.empty or stim_next.empty
                        ):
                            continue

                        focal_now = focal_now.iloc[0]
                        stim_now = stim_now.iloc[0]

                        focal_next = focal_next.iloc[0]
                        stim_next = stim_next.iloc[0]

                        distance_now, nearest_node = head_to_other_distance(
                            focal_now,
                            stim_now
                        )

                        if nearest_node is None:
                            continue

                        if distance_now <= 0 or distance_now > threshold:
                            continue

                        distance_next, next_nearest_node = head_to_other_distance(
                            focal_next,
                            stim_next
                        )

                        delta_distance = distance_next - distance_now

                        angle = approach_angle(
                            focal_now,
                            stim_now,
                            nearest_node
                        )

                        focal_speed = body_speed(focal_now, focal_next)
                        stim_speed = body_speed(stim_now, stim_next)

                        data.append({
                            'file': track_file,
                            'frame': frame,
                            'focal_id': focal_id,
                            'stim_id': stim_id,
                            'nearest_node': nearest_node,
                            'next_nearest_node': next_nearest_node,
                            'distance': distance_now,
                            'next_distance': distance_next,
                            'delta_distance': delta_distance,
                            'approach': delta_distance < 0,
                            'approach_angle': angle,
                            'focal_speed': focal_speed,
                            'stim_speed': stim_speed
                        })

        result = pd.DataFrame(data)

        result.to_csv(
            os.path.join(
                self.directory,
                f'individual_approach_probability.csv'
            ),
            index=False
        )

        return result
    

    



if __name__ == "__main__":

    directories = [
        "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/agarose-plates/group-housed/fed-fed",
        "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/agarose-plates/group-housed/fed-starved",
        "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/agarose-plates/group-housed/starved-starved",
        # "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/agarose-plates/socially-isolated/fed-fed",
        # "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/agarose-plates/socially-isolated/fed-starved",
        # "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/agarose-plates/socially-isolated/starved-starved",


        # "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/food-plates/group-housed/fed-fed",
        # "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/food-plates/group-housed/fed-starved",
        # "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/food-plates/group-housed/starved-starved",
        # "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/food-plates/socially-isolated/fed-fed",
        # "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/food-plates/socially-isolated/fed-starved",
        # "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/food-plates/socially-isolated/starved-starved",

    ]


    for directory in directories:

        print(f"\nProcessing: {directory}")

        analysis = FedStarvedAnalysis(directory)


        # analysis.shorten(frame=1200) #filter for 20 minutes 

        # analysis.total_digging()
        # analysis.larvae_present_over_time() # run without digging mask!!!
        # analysis.file_summary()  # run without digging mask!!! 



        ## must do for agarose plates
        analysis.digging_mask()
        analysis.conversion()
        analysis.filtering_files()


        # analysis.interaction_types_closest() 
        # analysis.head_head_first_contact() # do i rly need this given i can do first and other below - think get rid off - get rid 
        # analysis.head_head_contacts_kinematics_over_time() 
        # analysis.head_head_contacts_kinematics_over_time_nocontacts(proximity_threshold=1.5, window=10) # better than above i believe 
        # analysis.head_head_contacts_kinematics_over_time_nocontacts(proximity_threshold=1.5, window=60) # better than above i believe 
        # analysis.head_approach_angle()
        # analysis.nearest_neighbour()
        # analysis.interaction_type_bout() 
        # analysis.pairwise_approach_probability()
        # analysis.individual_approach_probability()  
        # analysis.prob_contact()
        # analysis.trajectory(window=20)
        # analysis.trajectory(proximity_threshold=1.5, window=10)
        # analysis.trajectory(proximity_threshold=1.5, window=30)
        # analysis.trajectory(proximity_threshold=1.5, window=60)
        # analysis.trajectory_before(proximity_threshold=1.5, window=10)
        # analysis.trajectory_before(proximity_threshold=1.5, window=30)
        # analysis.trajectory_before(proximity_threshold=1.5, window=60)


        analysis.trajectory_figures(proximity_threshold=1.5, window=10)
        # analysis.trajectories_before_figures(proximity_threshold=1.5, window=60)




        # analysis.head_head_interaction_type_over_time() #is this for rasta?
        # analysis.euclidean_distance()
        # analysis.angle()
        # analysis.speed()
        # analysis.acceleration()
        # analysis.movement_direction()





        # Food plates: keep digging larvae and annotate interactions by digging context.
        # analysis.food_plates_interactions()
     
  
