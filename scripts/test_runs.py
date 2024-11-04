import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pyarrow.feather as feather



# df = pd.read_excel('/Volumes/lab-windingm/home/users/cochral/plug_camera/staging/kir2.1/Kir2.1.xlsx')

# # Exclude rows where the 'day' column contains a hyphen between days
# df_filtered = df[~df['day'].str.contains('-')]

# # Apply the rest of your operations on the filtered DataFrame
# df_filtered['gal4 line'] = df_filtered['gal4'].apply(lambda x: 'empty' if x == 'empty' else 'line')

# # Plot the data
# sns.barplot(data=df_filtered, x='age', y='pupa number', hue='gal4 line', palette='viridis', edgecolor='black', linewidth=2)

# plt.xlabel('Age of Kir2.1 Females')
# plt.ylabel('Pupae Number')
# plt.title('Pupae Count',fontweight='bold')
# plt.show()

import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pyarrow.feather as feather
import cv2
import numpy as np
from shapely.geometry import Point, Polygon

# Step 1: Detect the largest circle (Petri dish) in the 10th frame
def detect_largest_circle(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.medianBlur(gray, 5)

    # Use Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.0, minDist=100,
                               param1=300, param2=50, minRadius=400, maxRadius=600)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        largest_circle = max(circles, key=lambda c: c[2])  # Get the largest circle (based on radius)
        return largest_circle  # x, y, r (center coordinates and radius)
    
    return None

# Step 2: Convert circle to Shapely polygon (approximate the perimeter of the circle)
def circle_to_polygon(x, y, radius, num_points=100):
    angles = np.linspace(0, 2 * np.pi, num_points)
    points = [(x + radius * np.cos(angle), y + radius * np.sin(angle)) for angle in angles]
    return Polygon(points)  # Return Shapely Polygon

# Step 3: Extract the 10th frame from the video
video_path = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/test-hole-counter/2024-08-27_13-50-20_td3.mp4'
cap = cv2.VideoCapture(video_path)

# Go to the 10th frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
ret, frame = cap.read()

if ret:
    # Step 4: Detect the circle in the 10th frame
    circle = detect_largest_circle(frame)
    
    if circle is not None:
        x, y, r = circle  # x, y: center of the circle, r: radius
        
        # Step 5: Convert the circle to a Shapely polygon
        petri_dish_boundary = circle_to_polygon(x, y, r)
        save_path = os.path.join('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/test-hole-counter/', 'petri_dish_boundary.wkt')
        
        # Step 6: Save the polygon to a WKT file for later use
        with open(save_path, 'w') as f:
            f.write(petri_dish_boundary.wkt)  # Write as WKT format

        print("Petri dish boundary saved as WKT.")
        
        # Step 7: Draw the circle on the frame (green with thickness 2)
        cv2.circle(frame, (x, y), r, (0, 255, 0), 2)  # Draw circle in green (BGR format)

        # Optionally, save the frame with the drawn boundary
        frame_with_boundary_path = os.path.join('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/test-hole-counter/', 'frame_with_boundary.png')
        cv2.imwrite(frame_with_boundary_path, frame)
        print(f"Frame with boundary saved at {frame_with_boundary_path}")
        
    else:
        print("No circle detected.")


cap.release()
