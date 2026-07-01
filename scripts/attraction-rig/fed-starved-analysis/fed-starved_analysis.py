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


class FedStarvedAnalysis:

    def __init__(self, directory):

        self.directory = directory 
        self.track_files = [] # list of the files 
        self.matching_pairs = []
        self.track_data = {}  # Initialize the track_data dictionary # actually has the data so we dont have to keep reloading 
        
        self.perimeter()
        self.tracks()
        self.match_files()
        self.conversion()

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

    def filtering_files(self, head_head_threshold=5, head_contact_threshold=1):

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

            head_head_within_5mm = False
            head_contact_within_1mm = False

            if not df.empty:
                df = df.sort_values(['frame', 'track_id'])
                row_counts = df.groupby('frame')['track_id'].transform('size')
                track_counts = df.groupby('frame')['track_id'].transform('nunique')
                df_two = df[(row_counts == 2) & (track_counts == 2)]

                if not df_two.empty:
                    first_larvae = df_two.groupby('frame').nth(0)
                    second_larvae = df_two.groupby('frame').nth(1)

                    head_1 = first_larvae[['x_head', 'y_head']].to_numpy(dtype=float)
                    head_2 = second_larvae[['x_head', 'y_head']].to_numpy(dtype=float)

                    head_head_distances = np.linalg.norm(head_1 - head_2, axis=1)
                    head_head_within_5mm = below_threshold(
                        head_head_distances,
                        head_head_threshold,
                        inclusive=True
                    )

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

                    head_1_to_nodes_2 = np.linalg.norm(nodes_2 - head_1[:, None, :], axis=2)
                    head_2_to_nodes_1 = np.linalg.norm(nodes_1 - head_2[:, None, :], axis=2)
                    head_contact_within_1mm = below_threshold(
                        np.concatenate([
                            head_1_to_nodes_2.ravel(),
                            head_2_to_nodes_1.ravel(),
                        ]),
                        head_contact_threshold
                    )

            passed_filter = head_head_within_5mm or head_contact_within_1mm

            results.append({
                'file_name': track_file,
                'passed_filter': 'Y' if passed_filter else 'N',
                'head_head_within_5mm': 'Y' if head_head_within_5mm else 'N',
                'head_contact_within_1mm': 'Y' if head_contact_within_1mm else 'N',
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
    





    def interacted_before_leaving(self):

        bout_df = self.interaction_type_bout()
        results = []

        for track_file in self.track_files:

            df = self.track_data[track_file]

            counts = (
                df.groupby("frame")["track_id"]
                .nunique()
                .sort_index()
            )

            leave_mask = (counts.shift(1) == 2) & (counts < 2)

            if leave_mask.any():
                leave_frame = leave_mask[leave_mask].index[0]
                left = True
            else:
                leave_frame = np.nan
                left = False

            file_bouts = bout_df[bout_df["file"] == track_file]

            if left:
                bouts_before_leave = file_bouts[file_bouts["start_frame"] < leave_frame]
            else:
                bouts_before_leave = file_bouts.iloc[0:0]

            results.append({
                "file": track_file,
                "left": left,
                "leave_frame": leave_frame,
                "interacted_before_leaving": len(bouts_before_leave) > 0,
                "n_bouts_before_leaving": len(bouts_before_leave),
                "total_interaction_duration_before_leaving": (
                    bouts_before_leave["duration"].sum()
                    if len(bouts_before_leave) > 0 else 0
                )

            })

        summary = pd.DataFrame(results)
        summary.to_csv(os.path.join(self.directory, "interacted_before_leaving.csv"), index=False)

        return summary








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
        

        angle_over_time = pd.DataFrame(data)
        angle_over_time = angle_over_time.sort_values(by=['time'], ascending=True)

        if self.shorten and self.shorten_duration is not None:
            suffix = f"_{self.shorten_duration}"
        else:
            suffix = ""

        filename = f"angle_over_time{suffix}.csv"

        angle_over_time.to_csv(os.path.join(self.directory, filename), index=False)

        return angle_over_time  

    
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

        min_run = 300
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

        data = []

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file]
            df = self.compute_digging(df)  # apply dynamic method

            if cleaned:
                df['count'] = df.groupby('frame')['track_id'].transform('nunique')
            else:
                df['count'] = total_larvae

            summary = df.groupby('frame').agg(
                number_digging=('digging_status', 'sum'),
                count=('count', 'first')  # same for all rows in group
            ).reset_index()

            summary['moving'] = summary['count'] - summary['number_digging']

            summary['normalised_digging'] = (summary['number_digging'] / summary['count']) * 100
            summary['file'] = track_file
            data.append(summary)


        result = pd.concat(data, ignore_index=True)
        result = result.sort_values(by='frame', ascending=True)

        if self.shorten and self.shorten_duration is not None:
            suffix = f"_{self.shorten_duration}"
        else:
            suffix = ""

        filename = f"number_digging{suffix}.csv"

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



    ### METHOD CONTACTS: IDENTIFY INTERACTION FREQUENCY AND DURATION

    def contacts(self, proximity_threshold=1): 

        data = []
        no_contacts = []

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

            # Convert to DataFrame - per interaction
            dist_df = pd.DataFrame(distance_rows)

            # Get min distance & node-node type per frame
            # frame | interaction-type | distance
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


        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file]
            df = df.sort_values(by='frame', ascending=True)
            df['filename'] = track_file

            track_ids = df['track_id'].unique()
            track_combinations = list(combinations(track_ids, 2))

            all_results = Parallel(n_jobs=-1)(
                delayed(process_track_pair)(track_a, track_b, df, track_file, proximity_threshold)
                for track_a, track_b in track_combinations
            )

            flattened_results = [item for sublist in all_results for item in sublist]
            if not flattened_results:
                print(f"No contact results for {track_file}")
                no_contacts.append(track_file)
                continue

            results_df = pd.DataFrame(flattened_results)
            results_df.set_index('frame', inplace=True, drop=False)
            data.append(results_df)

        ### for files with no interactions- create placeholders 
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
            data.append(placeholder)

        interaction_data = pd.concat(data, ignore_index=True)

        # Assign global interaction IDs across files and pairs
        interaction_data['Interaction Number'] = (
            interaction_data
            .groupby(['file','Interaction Pair','interaction'])
            .ngroup() + 1  # make it start at 1
        )
        interaction_data.drop(columns=['interaction'], inplace=True)  # Drop the local ID if you don't need it

        if self.shorten and self.shorten_duration is not None:
            suffix = f"_{self.shorten_duration}"
        else:
            suffix = ""

        filename = f"contacts_{proximity_threshold}mm{suffix}.csv"

        interaction_data.to_csv(os.path.join(self.directory, filename), index=False)




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







    def head_head_contacts_kinematics_over_time(self, proximity_threshold=1, window=15):

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
     
                    # ---- extract kinematics window ----
                    for f in range(interaction_frame, interaction_frame + window + 1):
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
                            body = (row['x_body'], row['y_body'])
                            head = (row['x_head'], row['y_head'])
                            tail = (row['x_tail'], row['y_tail'])

                            angle = heading_angle(body, head, tail)

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
                                'track_id': row['track_id'],
                                'speed_body': speed_body,
                                'speed_head': speed_head,
                                'speed_tail': speed_tail,
                                'heading_angle': angle,
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
                                'duration': bout['end_frame'] - bout['start_frame'],
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
        "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/agarose-plates/socially-isolated/fed-fed",
        "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/agarose-plates/socially-isolated/fed-starved",
        "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/agarose-plates/socially-isolated/starved-starved",
    ]




    for directory in directories:

        print(f"\nProcessing: {directory}")

        analysis = FedStarvedAnalysis(directory)


        # analysis.larvae_present_over_time() # run without digging mask!!!
        # analysis.file_summary()  # run without digging mask!!! 
         


        # analysis.digging_mask() 


        # analysis.interaction_types_closest()
        # analysis.head_head_first_contact()
        # analysis.head_head_contacts_kinematics_over_time()
        # analysis.head_approach_angle()
        # analysis.nearest_neighbour()
        # analysis.interaction_type_bout() 
        # analysis.interacted_before_leaving()
        # analysis.pairwise_approach_probability()
        # analysis.individual_approach_probability()  






        # analysis.speed()
        # analysis.head_head_interaction_type_over_time()
        # analysis.euclidean_distance()
        # analysis.trajectory()
        # analysis.speed()
        # analysis.acceleration()
        # analysis.contacts() # old school not even sure i use it in the end in the other script either
        # analysis.movement_direction()
     
  


