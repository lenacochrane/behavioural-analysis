import tkinter as tk # Python's standard GUI library
# filedialog: Provides dialogs to open/save files or directories
# messagebox: Provides simple message boxes to show info, warnings, or errors
from tkinter import filedialog, messagebox
import cv2 # Library for computer vision tasks like image and video processing
import pandas as pd
import os # Provides functions for interacting with the operating system
import csv

class SideHoleGui:
    def __init__(self, root): # self initialise # root window is main window of gui
        # assigns root to the instance variable self.root
        # allows other methods within the class to access the root window
        self.root = root   
        self.root.title("Hole Analyser") # title of root window 
        # initialise two instance variables:
        self.directory = "" # empty string to hold the user directory
        self.video_files = [] # empty list to hold the video files
        self.hole_coordinates = [] # empty list to hold the coordinates of hole 
        self.drawing = False  # Initialize the drawing flag
        self.current_video_index = 0 # Initialize the current video index

        # create a label in the root window 
        self.label = tk.Label(root, text="Select Video Directory")
        # pack makes the label visible 
        self.label.pack(pady=10) # adds space around the label (seperate other widgets)

        self.select_button = tk.Button(root, text="Select Directory", command=self.select_directory) # creates button widget
        self.select_button.pack(pady=10)

        self.process_button = tk.Button(root, text="Process Videos", command=self.process_videos)
        self.process_button.pack(pady=10)

# +---------------------------------------+
# | Hole Analyser                         |   <- Window Title
# |                                       |
# |       [ Select Video Directory ]      |   <- Label
# |                                       |
# |       [ Select Directory ]            |   <- Button
# |                                       |
# |       [ Process Videos ]              |   <- Button
# |                                       |
# +---------------------------------------+

    # METHOD TO SELECT DIRECTORY
    # this will be called when the select directory button is clicked
    def select_directory(self):
        # navigate and select a directory 
        self.directory = filedialog.askdirectory()
        self.video_files = [f for f in os.listdir(self.directory) if f.endswith('.mp4')]
        self.video_files.sort()  # Ensure files are processed in sorted order

    # METHOD TO PROCESS MP4 VIDEOS  
    def process_videos(self):
        if self.current_video_index < len(self.video_files):
            video_file = self.video_files[self.current_video_index]
            video_path = os.path.join(self.directory, video_file)  # Ensure full path is used
            print(f"Processing video: {video_path}")
            self.process_video(video_path)
        else:
            messagebox.showinfo("Processing Complete", "All videos have been processed.")


    # METHOD TO PROCESS INDIVIDUAL FILE
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path) # opens video

        if not cap.isOpened():
            messagebox.showerror("Error", f"Cannot open video: {video_path}")
            self.current_video_index += 1  # Move to the next video
            self.process_videos()
            return

        # currently takes the first frame from the video 
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", f"Cannot read video: {video_path}")
            self.current_video_index += 1  # Move to the next video
            self.process_videos()
            return

        self.temp_frame = frame.copy()  # Copy the frame for drawing purposes

        # Allow user to draw the hole perimeter
        cv2.imshow("Draw Hole Perimeter", frame) # Displays first frame in a window titled "Draw Hole Perimeter"

        cv2.setMouseCallback("Draw Hole Perimeter", self.draw_hole_perimeter, frame)
        cv2.waitKey(0) # wait until user finished and key pressed 

        # Save hole coordinates
        hole_file = video_path.replace('.mp4', '_hole.csv')
        with open(hole_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.hole_coordinates)  # coordinate into different rows

        # resets to an empty list
        self.hole_coordinates = []

    def draw_hole_perimeter(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.hole_coordinates.append((x, y))

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                prev_point = self.hole_coordinates[-1]
                self.hole_coordinates.append((x, y))
                cv2.line(self.temp_frame, prev_point, (x, y), (0, 0, 255), 2)
                cv2.imshow("Draw Hole Perimeter", self.temp_frame)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            # Ensure the last segment of the perimeter is drawn
            if len(self.hole_coordinates) > 1:
                prev_point = self.hole_coordinates[-1]
                cv2.line(self.temp_frame, prev_point, self.hole_coordinates[0], (0, 0, 255), 2)
                cv2.imshow("Draw Hole Perimeter", self.temp_frame)

        elif event == cv2.EVENT_RBUTTONDOWN:
            cv2.destroyAllWindows()
            self.current_video_index += 1  # Move to the next video
            self.process_videos()  # Continue processing the next video


