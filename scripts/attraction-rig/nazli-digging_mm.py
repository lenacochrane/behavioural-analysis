import cv2
from shapely.geometry import Polygon, Point
import pandas as pd
import numpy as np
import os
from shapely import wkt


def mm_conversion(directory): #### CONVERT PIXELS TO MM 

    ## == Identify Perimeter of the Petri Dish 
    
    def process_video(video_path):  
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # Check if the perimeter file already exists
        wkt_file_path = os.path.join(directory, f"{video_name}_perimeter.wkt")
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

                save_dir = directory
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
    video_files = [f for f in os.listdir(directory) if f.endswith('.mp4')]
    for file in video_files:
        video_path = os.path.join(directory, file)
        process_video(video_path)
    

    ## == Match Files

    video_files = [f for f in os.listdir(directory) if f.endswith('.mp4')]
    perimeter_files = [f for f in os.listdir(directory) if f.endswith('_perimeter.wkt')]
    track_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    matched_data = []

    for track_file in track_files:
        prefix = track_file.replace('.analysis', '').replace('.csv', '') ## need to remove analysis.csv
        
        match = {
            'track_file': track_file,
            'video_file': None,
            'perimeter_file': None,
            'perimeter_polygon': None,
        }

        for v in video_files:
            if '_'.join(v.split('_')[:3]).rsplit('.', 1)[0] == prefix:
                match['video_file'] = v

        for p in perimeter_files:
            if '_'.join(p.split('_')[:3]).rsplit('.', 1)[0] == prefix:
                match['perimeter_file'] = p
                perimeter_path = os.path.join(directory, p)
                with open(perimeter_path, 'r') as f:
                    perimeter_wkt = f.read()

                    try:
                        polygon = wkt.loads(perimeter_wkt)
                        match['perimeter_polygon'] = polygon
                        print(f"✅ Loaded perimeter for {track_file} → {p}")
                    except Exception as e:
                        print(f"❌ Failed to load WKT for {p}: {e}")

                # polygon = wkt.loads(perimeter_wkt)
                # match['perimeter_polygon'] = polygon

        matched_data.append(match)


    ## == Convert Pixels to MM

    for match in matched_data:
        track_file = match['track_file']
        track_path = os.path.join(directory, track_file)
        df = pd.read_csv(track_path)

        perimeter_polygon = match['perimeter_polygon']
        if perimeter_polygon:
            minx, _, maxx, _ = perimeter_polygon.bounds
            diameter = maxx - minx
            conversion_factor = 90 / diameter

            if conversion_factor > 0.09:
                print(f"⚠️ Conversion factor {conversion_factor:.3f} too high for {track_file}, reverting to default.")
                conversion_factor = 90 / 1032
        else:
            print(f"⚠️ No perimeter found for {track_file}, using default conversion.")
            conversion_factor = 90 / 1032

        # Apply conversion
        pixel_columns = ['tail.x', 'tail.y', 'body.x', 'body.y', 'head.x', 'head.y']
        df[pixel_columns] = df[pixel_columns] * conversion_factor
        path =  track_path = os.path.join(directory, f"{track_file}_edited.csv")

        df.to_csv(track_path, index=False)
        print(f"✅ Converted {track_file} with factor {conversion_factor:.3f}")


def digging(df): #### REMOVE DIGGING LARVAE

    df = df.sort_values(['track', 'frame_idx']).reset_index(drop=True)

    # Smooth x and y
    df['x'] = (
        df.groupby('track', group_keys=False)['body.x']
        .apply(lambda x: x.rolling(window=5, min_periods=1).mean())
    )
    df['y'] = (
        df.groupby('track', group_keys=False)['body.y']
        .apply(lambda y: y.rolling(window=5, min_periods=1).mean())
    )

    # Differences
    df['dx'] = df.groupby('track')['x'].diff().fillna(0)
    df['dy'] = df.groupby('track')['y'].diff().fillna(0)

    # Distance and moving status
    df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2)
    df['is_moving'] = df['distance'] > 0.2

    # Cumulative and std
    df['cumulative_displacement'] = df.groupby('track')['distance'].cumsum()
    df['cumulative_displacement_rate'] = df.groupby('track')['cumulative_displacement'].apply(lambda x: x.diff(10) / 10).fillna(0)

    df['x_std'] = df.groupby('track')['x'].transform(lambda x: x.rolling(window=10, min_periods=1).std())
    df['y_std'] = df.groupby('track')['y'].transform(lambda x: x.rolling(window=10, min_periods=1).std())
    df['overall_std'] = np.sqrt(df['x_std']**2 + df['y_std']**2)

    df['final_movement'] = (df['cumulative_displacement_rate'] > 0.1) | ((df['overall_std'] > 1) & (df['is_moving']))
    
    ## smoothed final movement
    window_size = 20
    df['digging_status'] = (
        df.groupby('track')['final_movement']
        .transform(lambda x: (~x).rolling(window=window_size, center=False).apply(lambda r: r.sum() >= (window_size / 2)).fillna(0).astype(bool))
    )

    ### backfilling TRUE for larvae that actually end up digging 
    df['prev'] = (
            df.groupby('track')['digging_status']
            .shift(1)
            .fillna(False)
        )
    df['false_true'] = df['digging_status'] & ~df['prev'] # digging status = True ; prev frame digging status = False


    df['future_digging'] = (
    df.groupby('track')['digging_status']
    .rolling(window=15, min_periods=15)
    .sum()
    .shift(-14)
    .reset_index(level=0, drop=True)
)
    df['long_digging'] = df['false_true'] & (df['future_digging'] >= 15)

    # 1) Initialize backfill column
    df['backfill'] = False

    # 2) Loop per track
    for track_id, group in df.groupby('track'):
        idx   = group.index
        starts = idx[group.loc[idx, 'long_digging']]
        for s in starts:
            pre = max(idx.min(), s - 30)
            df.loc[pre:s-1, 'backfill'] = True  # back-fill up to the frame *before* 

    df['digging_status'] = df['digging_status'] | df['backfill']

    df.drop(columns=['backfill', 'long_digging', 'false_true', 'future_digging'], inplace=True)

    # df = df[df['digging_status'] == False].reset_index(drop=True)

    return df





directory = '/Users/cochral/Desktop/SLAEP/TRain/test'


mm_conversion(directory)

# APRENTLY DIGGING NOT SAVING

for file in os.listdir(directory):
    if file.endswith('_edited.csv'):
        path = os.path.join(directory, file)
        df = pd.read_csv(path)
        df = digging(df)
        df.to_csv(path, index=False)
        print(f' ✅ Removed Digging {file}')





















        
        




            


        
