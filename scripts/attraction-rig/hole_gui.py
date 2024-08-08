import tkinter as tk # Python's standard GUI library
# filedialog: Provides dialogs to open/save files or directories
# messagebox: Provides simple message boxes to show info, warnings, or errors
from tkinter import filedialog, messagebox
import cv2 # Library for computer vision tasks like image and video processing
import pandas as pd
import os # Provides functions for interacting with the operating system
import csv

class HoleGui:
    def __init__(self, root):
        self.root = root
        self.root.title("Hole Analyser")
        self.directory = ""
        self.video_files = []
        self.hole_coordinates = []
        self.drawing = False
        self.current_video_index = 0

        self.label = tk.Label(root, text="Select Video Directory")
        self.label.pack(pady=10)

        self.select_button = tk.Button(root, text="Select Directory", command=self.select_directory)
        self.select_button.pack(pady=10)

        self.process_button = tk.Button(root, text="Process Videos", command=self.process_videos)
        self.process_button.pack(pady=10)

    def select_directory(self):
        self.directory = filedialog.askdirectory()
        self.video_files = [f for f in os.listdir(self.directory) if f.endswith('.mp4')]
        self.video_files.sort()
        print(f"Selected directory: {self.directory}")
        print(f"Video files found: {self.video_files}")

    def process_videos(self):
        if self.current_video_index < len(self.video_files):
            video_file = self.video_files[self.current_video_index]
            video_path = os.path.join(self.directory, video_file)
            print(f"Processing video: {video_path}")
            self.process_video(video_path)
        else:
            print("All videos processed.")
            messagebox.showinfo("Processing Complete", "All videos have been processed.")
            self.root.destroy()  # Force close the Tkinter window

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Cannot open video: {video_path}")
            messagebox.showerror("Error", f"Cannot open video: {video_path}")
            self.current_video_index += 1
            self.process_videos()
            return

        ret, frame = cap.read()
        if not ret:
            print(f"Error: Cannot read video: {video_path}")
            messagebox.showerror("Error", f"Cannot read video: {video_path}")
            self.current_video_index += 1
            self.process_videos()
            return

        self.temp_frame = frame.copy()
        self.hole_coordinates = []  # Reset coordinates for each video

        cv2.imshow("Draw Hole Perimeter", self.temp_frame)
        cv2.setMouseCallback("Draw Hole Perimeter", self.draw_hole_perimeter, self.temp_frame)
        cv2.waitKey(0)

    def draw_hole_perimeter(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.hole_coordinates.append((x, y))
            print(f"Point added: {(x, y)}")

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                prev_point = self.hole_coordinates[-1]
                self.hole_coordinates.append((x, y))
                cv2.line(self.temp_frame, prev_point, (x, y), (0, 0, 255), 2)
                cv2.imshow("Draw Hole Perimeter", self.temp_frame)
                print(f"Drawing line from {prev_point} to {(x, y)}")

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if len(self.hole_coordinates) > 1:
                prev_point = self.hole_coordinates[-1]
                cv2.line(self.temp_frame, prev_point, self.hole_coordinates[0], (0, 0, 255), 2)
                cv2.imshow("Draw Hole Perimeter", self.temp_frame)
                print(f"Closed the hole perimeter by drawing line from {prev_point} to {self.hole_coordinates[0]}")

        elif event == cv2.EVENT_RBUTTONDOWN:
            print("Right button clicked, saving coordinates and moving to next video.")
            self.save_coordinates()
            cv2.destroyAllWindows()
            self.current_video_index += 1
            self.process_videos()

    def save_coordinates(self):
        if self.hole_coordinates:
            video_file = self.video_files[self.current_video_index]
            video_path = os.path.join(self.directory, video_file)
            hole_file = os.path.join(self.directory, os.path.basename(video_path).replace('.mp4', '_hole.csv'))
            print(f"Saving coordinates to {hole_file} with {len(self.hole_coordinates)} points")
            try:
                with open(hole_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(self.hole_coordinates)
                print(f"Coordinates saved to {hole_file}")
            except Exception as e:
                print(f"Error saving coordinates to {hole_file}: {e}")



