import os
import cv2
import numpy as np
from shapely.geometry import Polygon

def process_video(video_path):

    
    # Function to detect the largest circle in a frame
    def detect_largest_circle(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.medianBlur(gray, 5)

        # Use Hough Circle Transform to detect circles
        circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.0, minDist=100,
                                   param1=500, param2=50, minRadius=400, maxRadius=600)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            largest_circle = max(circles, key=lambda c: c[2])  # Get the largest circle (based on radius)
            return largest_circle  # x, y, r (center coordinates and radius)
        
        return None

    # Function to convert circle to Shapely polygon
    def circle_to_polygon(x, y, radius, num_points=100):
        angles = np.linspace(0, 2 * np.pi, num_points)
        points = [(x + radius * np.cos(angle), y + radius * np.sin(angle)) for angle in angles]
        return Polygon(points)  # Return Shapely Polygon

    # Read the video
    cap = cv2.VideoCapture(video_path)

    # Go to the 10th frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
    ret, frame = cap.read()

    if ret:
        # Detect the largest circle in the frame
        circle = detect_largest_circle(frame)
        
        if circle is not None:
            x, y, r = circle  # x, y: center of the circle, r: radius
            
            # Convert the circle to a Shapely polygon
            petri_dish_boundary = circle_to_polygon(x, y, r)
            
            # Save the polygon to a WKT file for later use
            save_dir = os.path.dirname(video_path)
            save_path_wkt = os.path.join(save_dir, 'petri_dish_boundary.wkt')
            with open(save_path_wkt, 'w') as f:
                f.write(petri_dish_boundary.wkt)  # Write as WKT format
            print(f"Petri dish boundary saved as WKT at {save_path_wkt}.")
            
            # Draw the circle on the frame (green with thickness 2)
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)  # Draw circle in green (BGR format)

            # Save the frame with the drawn boundary
            frame_with_boundary_path = os.path.join(save_dir, 'frame_with_boundary1.png')
            cv2.imwrite(frame_with_boundary_path, frame)
            print(f"Frame with boundary saved at {frame_with_boundary_path}.")
            
        else:
            print("No circle detected.")
    
    else:
        print(f"Failed to extract the 10th frame from the video.")

    cap.release()

# Example usage
video_path = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/test-leaving-perimeter/2024-07-30_10-58-18_td4.mp4'
process_video(video_path)
